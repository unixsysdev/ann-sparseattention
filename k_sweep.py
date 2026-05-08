"""
K-retrieve sweep on a trained search-projection checkpoint.

Loads the search module from a checkpoint, runs eval at K in
{16, 32, 64, 128, 256, 512}, reports PPL gap and recall@K per layer.
This produces the speed/quality Pareto curve.

Usage:
    python k_sweep.py --ckpt /tmp/checkpoints/search_step_2000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from data import build_eval_data  # noqa: E402
from eval import (  # noqa: E402
    _per_position_mass_at_k,
    _per_position_recall,
    compute_perplexity,
)
import inference  # noqa: E402
from inference import install_ann_attention, uninstall_ann_attention  # noqa: E402
from model import (  # noqa: E402
    FrozenForwardCapture,
    SearchProjectionModule,
    aggregate_heads,
)


def config_from_checkpoint(ckpt: dict) -> Config:
    cfg = Config()
    ckpt_cfg = ckpt.get("config", {})
    for key, value in ckpt_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def k_sweep(
    ckpt_path: str,
    K_values=(16, 32, 64, 128, 256, 512),
    num_batches: int = 16,
    use_faiss: bool = None,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    if use_faiss is None:
        use_faiss = cfg.use_faiss_hnsw_at_eval

    print(f"Loading base model {cfg.base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    layers_to_train = [
        i for i in cfg.full_attention_layer_indices
        if i not in cfg.reserved_full_attention_indices
    ]

    search = SearchProjectionModule(
        d_model=base_model.config.hidden_size,
        d_search=cfg.d_search,
        layer_indices=layers_to_train,
        use_mlp=cfg.use_mlp_proj,
    ).to(base_model.device).to(torch.bfloat16)

    search.load_state_dict(ckpt["search_module"])
    search.eval()
    print(f"Loaded ckpt step {ckpt['step']}")

    capture = FrozenForwardCapture(
        base_model, layers_to_train, qk_reconstruction=cfg.qk_reconstruction
    )

    eval_data = list(build_eval_data(tokenizer, cfg, num_batches=num_batches))

    # Precompute teacher attention + search outputs once per batch.
    # Cache attention_mask + position_ids alongside the activations so every
    # downstream call (recall, mass@K, full PPL, ANN PPL) sees the same
    # mask/positions and PPL is comparable across K.
    print("Pre-running teacher captures...")
    cached: list = []
    for batch in eval_data:
        input_ids = batch["input_ids"].to(base_model.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(base_model.device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(base_model.device)
        h_dict, w_dict = capture.run(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        with torch.no_grad():
            q_dict, k_dict = search(h_dict)
        cached.append(
            (input_ids, attention_mask, position_ids, h_dict, w_dict, q_dict, k_dict)
        )

    # Reference full-attention PPL (computed once, not per K).
    print("Computing full-attention PPL...")
    full_ppls = []
    for input_ids, attn_m, pos_ids, *_ in cached:
        full_ppls.append(compute_perplexity(base_model, input_ids, attn_m, pos_ids))
    ppl_full = sum(full_ppls) / len(full_ppls)
    print(f"  ppl_full = {ppl_full:.4f}")

    results = {"ppl_full": ppl_full, "by_K": {}}

    for K in K_values:
        print(f"\n=== K = {K} ===")
        # Recall@K + mass@K (using cached captures)
        per_layer_recall = {idx: [] for idx in layers_to_train}
        per_layer_mass = {idx: [] for idx in layers_to_train}
        for input_ids, attn_m, pos_ids, h_dict, w_dict, q_dict, k_dict in cached:
            for idx in layers_to_train:
                teacher_full = w_dict[idx]   # [B, H, L, L]
                teacher = aggregate_heads(
                    teacher_full, mode=cfg.teacher_head_aggregation
                )
                rec = _per_position_recall(teacher, q_dict[idx], k_dict[idx], K)
                B, L = rec.shape
                pos = torch.arange(L, device=rec.device).unsqueeze(0).expand(B, L)
                mask = pos >= K
                per_layer_recall[idx].extend(rec.masked_select(mask).tolist())

                _, mass = _per_position_mass_at_k(
                    teacher_full, q_dict[idx], k_dict[idx], K
                )
                per_layer_mass[idx].extend(mass.masked_select(mask).tolist())

        recall_per_layer = {
            idx: sum(per_layer_recall[idx]) / max(1, len(per_layer_recall[idx]))
            for idx in layers_to_train
        }
        recall_avg = sum(recall_per_layer.values()) / len(recall_per_layer)
        mass_per_layer = {
            idx: sum(per_layer_mass[idx]) / max(1, len(per_layer_mass[idx]))
            for idx in layers_to_train
        }
        mass_avg = sum(mass_per_layer.values()) / len(mass_per_layer)

        # PPL gap with ANN substitution at this K
        wrappers = install_ann_attention(
            base_model,
            search,
            layers_to_train,
            K_retrieve=K,
            use_faiss=use_faiss,
            use_hnsw=use_faiss,
            hnsw_M=cfg.faiss_hnsw_M,
            hnsw_ef_construction=cfg.faiss_hnsw_ef_construction,
            hnsw_ef_search=cfg.faiss_hnsw_ef_search,
        )
        try:
            inference.FAISS_STATS.clear()
            ann_ppls = [
                compute_perplexity(base_model, ids, attn_m, pos_ids)
                for ids, attn_m, pos_ids, *_ in cached
            ]
        finally:
            uninstall_ann_attention(wrappers)
        ppl_ann = sum(ann_ppls) / len(ann_ppls)
        ppl_gap = (ppl_ann - ppl_full) / ppl_full

        # Aggregate FAISS retrieval-quality stats over all calls (one per
        # trained layer per forward batch).
        if inference.FAISS_STATS:
            n = len(inference.FAISS_STATS)
            faiss_diag = {
                "self_pad_rate": sum(s["self_pad_rate"] for s in inference.FAISS_STATS) / n,
                "causal_fill_rate": sum(s["causal_fill_rate"] for s in inference.FAISS_STATS) / n,
                "self_attn_rate": sum(s["self_attn_rate"] for s in inference.FAISS_STATS) / n,
            }
        else:
            faiss_diag = {}

        results["by_K"][K] = {
            "recall_avg": recall_avg,
            "recall_per_layer": recall_per_layer,
            "mass_avg": mass_avg,
            "mass_per_layer": mass_per_layer,
            "ppl_ann": ppl_ann,
            "ppl_gap_relative": ppl_gap,
            "faiss_diag": faiss_diag,
        }
        print(
            f"  mass_avg   = {mass_avg:.4f}   "
            f"recall_avg = {recall_avg:.4f}   "
            f"ppl_ann = {ppl_ann:.4f}   ppl_gap = {ppl_gap:+.3%}"
        )
        if faiss_diag:
            print(
                f"  [faiss] self_pad={faiss_diag['self_pad_rate']:.3f}  "
                f"causal_fill={faiss_diag['causal_fill_rate']:.3f}  "
                f"self_attn={faiss_diag['self_attn_rate']:.3f}"
            )

    out_path = os.path.splitext(ckpt_path)[0] + ".k_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument(
        "--K", default="16,32,64,128,256,512",
        help="Comma-separated list of K values"
    )
    parser.add_argument(
        "--use-faiss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FAISS/HNSW retrieval. Defaults to the checkpoint config.",
    )
    args = parser.parse_args()
    K_values = tuple(int(x) for x in args.K.split(","))
    k_sweep(
        args.ckpt,
        K_values=K_values,
        num_batches=args.num_batches,
        use_faiss=args.use_faiss,
    )


if __name__ == "__main__":
    main()
