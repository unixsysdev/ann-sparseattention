"""
The contribution claim: a learned shared projection makes Q and K live in
the same distribution by construction, so vanilla ANN works without any
attention-aware index trick.

This script runs the cleanest test of that claim. For each trained layer,
on the same eval batches:

  * teacher topK     — softmax(QK^T) head-aggregated, take top-K
  * raw-QK topK      — exact top-K cosine over the model's native Q,K post-RoPE
                       (head-mean aggregated to a single vector per token).
                       This is the upper bound on what *any* ANN method
                       (FAISS HNSW, RetrievalAttention's RoarGraph, etc.)
                       could achieve over native Q/K.
  * learned topK     — exact top-K cosine over the trained Q_s, K_s.
                       Upper bound on what FAISS over the learned space
                       could achieve.

We then measure mass@K (sum of teacher attention probability captured by
the retrieved set) of raw-QK vs learned, against the per-head teacher.
If learned >> raw, the projection is doing real work and validates the
"trained shared retrieval space" thesis.

Usage:
    python compare_retrieval.py --ckpt search_step_2000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from data import build_eval_data  # noqa: E402
from model import (  # noqa: E402
    FrozenForwardCapture,
    SearchProjectionModule,
)


def _causal_topk(q: torch.Tensor, k: torch.Tensor, K: int) -> torch.Tensor:
    """q, k: [B, L, d]. Returns top-K indices [B, L, K], causal-respecting."""
    B, L, _ = q.shape
    sim = torch.bmm(q, k.transpose(1, 2))
    mask = torch.ones(L, L, device=sim.device, dtype=torch.bool).tril()
    sim = sim.masked_fill(~mask, -1e9)
    return sim.topk(min(K, L), dim=-1).indices


def _mass_against_per_head_teacher(
    teacher_full: torch.Tensor,    # [B, H, L, L] — full per-head softmax
    retrieved: torch.Tensor,        # [B, L, K]
    K: int,
) -> torch.Tensor:
    """
    For each (b, h, q) with q >= K, return the fraction of teacher attention
    mass that the retrieval captured. The K floor is to avoid the early
    positions where the causal window is smaller than K and the comparison
    is degenerate. Returns: scalar mean.
    """
    B, H, L, _ = teacher_full.shape
    device = teacher_full.device
    grid = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    grid.scatter_(-1, retrieved, True)
    grid = grid.unsqueeze(1).expand(B, H, L, L)
    mass = (teacher_full * grid.to(teacher_full.dtype)).sum(-1)  # [B, H, L]
    pos = torch.arange(L, device=device).view(1, 1, L)
    keep = (pos >= K).expand(B, H, L)
    return mass.masked_select(keep).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-batches", type=int, default=12)
    parser.add_argument("--K", default="16,32,64,128,256")
    args = parser.parse_args()

    K_values = tuple(int(x) for x in args.K.split(","))
    K_max = max(K_values)

    cfg = Config()
    print(f"Loading {cfg.base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    layers_to_train = [
        i for i in cfg.full_attention_layer_indices
        if i not in cfg.reserved_full_attention_indices
    ]

    search = SearchProjectionModule(
        d_model=base.config.hidden_size,
        d_search=cfg.d_search,
        layer_indices=layers_to_train,
        use_mlp=cfg.use_mlp_proj,
    ).to(base.device).to(torch.bfloat16)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    search.load_state_dict(ckpt["search_module"])
    search.eval()
    print(f"Loaded ckpt step {ckpt['step']} for layers {layers_to_train}")

    capture = FrozenForwardCapture(base, layers_to_train, qk_reconstruction=True)
    eval_data = list(build_eval_data(tokenizer, cfg, num_batches=args.num_batches))

    # Accumulators: per (K, method, layer) -> list of mass@K values across batches.
    methods = ("raw_qk", "learned")
    acc = {
        K: {m: {l: [] for l in layers_to_train} for m in methods}
        for K in K_values
    }

    for bi, batch in enumerate(eval_data):
        input_ids = batch["input_ids"].to(base.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(base.device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(base.device)

        h_dict, w_dict = capture.run(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        with torch.no_grad():
            q_s_dict, k_s_dict = search(h_dict)
        captured_qk = capture._captured_qk  # {layer: (Q, K) post-RoPE [B, H, L, d_head]}

        for layer in layers_to_train:
            teacher_full = w_dict[layer].float()  # [B, H, L, L]

            # --- Raw Q/K (head-mean aggregation to single vector per token) ---
            q_raw, k_raw = captured_qk[layer]  # [B, H_q, L, d_head], [B, H_kv, L, d_head]
            q_agg = q_raw.float().mean(dim=1)
            k_agg = k_raw.float().mean(dim=1)
            q_agg = F.normalize(q_agg, dim=-1)
            k_agg = F.normalize(k_agg, dim=-1)
            raw_topk_full = _causal_topk(q_agg, k_agg, K_max)  # [B, L, K_max]

            # --- Learned Q_s / K_s (already a single shared vector per token) ---
            q_s = F.normalize(q_s_dict[layer].float(), dim=-1)
            k_s = F.normalize(k_s_dict[layer].float(), dim=-1)
            learned_topk_full = _causal_topk(q_s, k_s, K_max)

            for K in K_values:
                raw_topk_K = raw_topk_full[..., :K]
                learned_topk_K = learned_topk_full[..., :K]
                acc[K]["raw_qk"][layer].append(
                    _mass_against_per_head_teacher(teacher_full, raw_topk_K, K)
                )
                acc[K]["learned"][layer].append(
                    _mass_against_per_head_teacher(teacher_full, learned_topk_K, K)
                )

        if bi % 3 == 0:
            print(f"  batch {bi+1}/{len(eval_data)} done")

    # ---- aggregate ----
    summary = {"model": cfg.base_model_name, "ckpt": args.ckpt, "by_K": {}}
    print(f"\n{'='*72}")
    print(f"mass@K — fraction of teacher attention captured by retrieval set")
    print(f"  raw_qk : exact top-K over head-mean-aggregated post-RoPE Q,K")
    print(f"  learned: exact top-K over trained search projections (d=64)")
    print(f"{'='*72}\n")
    print(f"{'K':>4}  {'method':<10} " + " ".join(f"L{l:02d}" for l in layers_to_train) + "   avg")
    for K in K_values:
        summary["by_K"][K] = {}
        for method in methods:
            per_layer = {
                l: sum(acc[K][method][l]) / max(1, len(acc[K][method][l]))
                for l in layers_to_train
            }
            avg = sum(per_layer.values()) / len(per_layer)
            summary["by_K"][K][method] = {
                "per_layer": {str(l): per_layer[l] for l in layers_to_train},
                "avg": avg,
            }
            print(
                f"{K:>4}  {method:<10} "
                + " ".join(f"{per_layer[l]:.3f}" for l in layers_to_train)
                + f"   {avg:.3f}"
            )
        print()

    # Headline ratio at K=128.
    if 128 in K_values:
        learned_avg_128 = summary["by_K"][128]["learned"]["avg"]
        raw_avg_128 = summary["by_K"][128]["raw_qk"]["avg"]
        ratio = learned_avg_128 / max(raw_avg_128, 1e-9)
        summary["learned_over_raw_K128"] = ratio
        print(f"Learned vs raw mass@K=128: {learned_avg_128:.3f} / {raw_avg_128:.3f} = {ratio:.2f}×")

    out_path = os.path.splitext(args.ckpt)[0] + ".compare_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
