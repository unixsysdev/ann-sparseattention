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
    _per_position_recall,
    compute_perplexity,
)
from inference import install_ann_attention, uninstall_ann_attention  # noqa: E402
from model import (  # noqa: E402
    FrozenForwardCapture,
    SearchProjectionModule,
    aggregate_heads,
)


def k_sweep(ckpt_path: str, K_values=(16, 32, 64, 128, 256, 512), num_batches: int = 16):
    cfg = Config()

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

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    search.load_state_dict(ckpt["search_module"])
    search.eval()
    print(f"Loaded ckpt step {ckpt['step']}")

    capture = FrozenForwardCapture(
        base_model, layers_to_train, qk_reconstruction=cfg.qk_reconstruction
    )

    eval_data = list(build_eval_data(tokenizer, cfg, num_batches=num_batches))

    # Precompute teacher attention + search outputs once per batch.
    print("Pre-running teacher captures...")
    cached: list = []
    for batch in eval_data:
        input_ids = batch["input_ids"].to(base_model.device)
        h_dict, w_dict = capture.run(input_ids)
        with torch.no_grad():
            q_dict, k_dict = search(h_dict)
        cached.append((input_ids, h_dict, w_dict, q_dict, k_dict))

    # Reference full-attention PPL (computed once, not per K).
    print("Computing full-attention PPL...")
    full_ppls = []
    for input_ids, *_ in cached:
        full_ppls.append(compute_perplexity(base_model, input_ids))
    ppl_full = sum(full_ppls) / len(full_ppls)
    print(f"  ppl_full = {ppl_full:.4f}")

    results = {"ppl_full": ppl_full, "by_K": {}}

    for K in K_values:
        print(f"\n=== K = {K} ===")
        # Recall@K (using cached captures)
        per_layer = {idx: [] for idx in layers_to_train}
        for input_ids, h_dict, w_dict, q_dict, k_dict in cached:
            for idx in layers_to_train:
                teacher = aggregate_heads(
                    w_dict[idx], mode=cfg.teacher_head_aggregation
                )
                rec = _per_position_recall(teacher, q_dict[idx], k_dict[idx], K)
                B, L = rec.shape
                pos = torch.arange(L, device=rec.device).unsqueeze(0).expand(B, L)
                mask = pos >= K
                per_layer[idx].extend(rec.masked_select(mask).tolist())

        recall_per_layer = {
            idx: sum(per_layer[idx]) / max(1, len(per_layer[idx]))
            for idx in layers_to_train
        }
        recall_avg = sum(recall_per_layer.values()) / len(recall_per_layer)

        # PPL gap with ANN substitution at this K
        wrappers = install_ann_attention(
            base_model,
            search,
            layers_to_train,
            K_retrieve=K,
            use_faiss=cfg.use_faiss_hnsw_at_eval,
            use_hnsw=cfg.use_faiss_hnsw_at_eval,
            hnsw_M=cfg.faiss_hnsw_M,
            hnsw_ef_construction=cfg.faiss_hnsw_ef_construction,
            hnsw_ef_search=cfg.faiss_hnsw_ef_search,
        )
        try:
            ann_ppls = [compute_perplexity(base_model, ids) for ids, *_ in cached]
        finally:
            uninstall_ann_attention(wrappers)
        ppl_ann = sum(ann_ppls) / len(ann_ppls)
        ppl_gap = (ppl_ann - ppl_full) / ppl_full

        results["by_K"][K] = {
            "recall_avg": recall_avg,
            "recall_per_layer": recall_per_layer,
            "ppl_ann": ppl_ann,
            "ppl_gap_relative": ppl_gap,
        }
        print(
            f"  recall_avg = {recall_avg:.4f}   "
            f"ppl_ann = {ppl_ann:.4f}   ppl_gap = {ppl_gap:+.3%}"
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
    args = parser.parse_args()
    K_values = tuple(int(x) for x in args.K.split(","))
    k_sweep(args.ckpt, K_values=K_values, num_batches=args.num_batches)


if __name__ == "__main__":
    main()
