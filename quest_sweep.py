"""
Quest-style page baseline on the same eval pipeline as k_sweep.py.

This is a correctness/prototype baseline, not an optimized Quest runtime. It
uses native post-RoPE Q/K page min/max metadata to select pages, filters tokens
through the same causal/block-causal mask as the rest of the repo, and runs the
existing sparse-attention gather path for PPL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_eval_data, model_attention_mask  # noqa: E402
from eval import _query_has_k_valid_keys, compute_perplexity  # noqa: E402
from inference import (  # noqa: E402
    _normalize_allowed_mask,
    _quest_page_search,
    install_quest_attention,
    uninstall_ann_attention,
)
from k_sweep import config_from_checkpoint  # noqa: E402
from model import FrozenForwardCapture  # noqa: E402


def _repeat_kv_to_q_heads(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    if q.shape[1] == k.shape[1]:
        return k
    repeat = q.shape[1] // k.shape[1]
    return k.repeat_interleave(repeat, dim=1)


def _quest_mass_recall(
    teacher_full: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    K: int,
    page_size: int,
    model_mask: torch.Tensor,
    allowed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-query [B,L] mass and recall averaged over heads."""
    B, H, L, _ = teacher_full.shape
    k_rep = _repeat_kv_to_q_heads(q, k)
    retrieved, retrieved_valid = _quest_page_search(
        q,
        k_rep,
        K,
        page_size=page_size,
        key_mask=model_mask,
        return_valid_mask=True,
    )

    retrieved_safe = retrieved.masked_fill(~retrieved_valid, 0)
    search_grid = torch.zeros(B, H, L, L, dtype=torch.bool, device=q.device)
    search_grid.scatter_(-1, retrieved_safe, retrieved_valid)

    mass = (teacher_full * search_grid.to(teacher_full.dtype)).sum(-1).mean(dim=1)

    allowed = allowed_mask.bool()
    teacher_masked = teacher_full.masked_fill(~allowed.unsqueeze(1), -1e9)
    teacher_top = teacher_masked.topk(min(K, L), dim=-1).indices
    teacher_grid = torch.zeros(B, H, L, L, dtype=torch.bool, device=q.device)
    teacher_grid.scatter_(-1, teacher_top, True)
    inter = (teacher_grid & search_grid).sum(-1)
    denom = torch.minimum(
        torch.full((B, L), min(K, L), device=q.device, dtype=torch.long),
        allowed.sum(dim=-1),
    ).clamp(min=1)
    recall = (inter.float() / denom.unsqueeze(1).float()).mean(dim=1)
    return mass, recall


def quest_sweep(
    ckpt_path: str,
    K_values=(128, 256, 512),
    num_batches: int = 16,
    skip_batches: int = 0,
    page_size: int = 16,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)

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

    layers = [
        i for i in cfg.full_attention_layer_indices
        if i not in cfg.reserved_full_attention_indices
    ]
    capture = FrozenForwardCapture(base_model, layers, qk_reconstruction=True)

    eval_data_all = list(
        build_eval_data(tokenizer, cfg, num_batches=num_batches + skip_batches)
    )
    eval_data = eval_data_all[skip_batches:]
    print("Pre-running teacher captures...")
    cached = []
    for batch in eval_data:
        input_ids = batch["input_ids"].to(base_model.device)
        token_mask = batch.get("attention_mask")
        if token_mask is not None:
            token_mask = token_mask.to(base_model.device)
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            segment_ids = segment_ids.to(base_model.device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(base_model.device)
        model_mask = model_attention_mask(
            token_mask,
            segment_ids,
            block_causal_mask=getattr(cfg, "block_causal_mask", False),
            dtype=base_model.dtype,
        )
        allowed_mask = _normalize_allowed_mask(model_mask, input_ids.shape[1])
        if allowed_mask is None:
            L = input_ids.shape[1]
            allowed_mask = torch.ones(L, L, device=base_model.device, dtype=torch.bool).tril()
            allowed_mask = allowed_mask.unsqueeze(0).expand(input_ids.shape[0], L, L)
        h_dict, w_dict = capture.run(
            input_ids, attention_mask=model_mask, position_ids=position_ids
        )
        qk_dict = {idx: capture._captured_qk[idx] for idx in layers}
        cached.append((input_ids, token_mask, model_mask, allowed_mask, position_ids, w_dict, qk_dict))

    print("Computing full-attention PPL...")
    ppl_full = sum(
        compute_perplexity(base_model, ids, model_m, pos_ids, target_mask=token_m)
        for ids, token_m, model_m, _allowed_m, pos_ids, *_ in cached
    ) / len(cached)
    print(f"  ppl_full = {ppl_full:.4f}")

    results = {
        "ppl_full": ppl_full,
        "page_size": page_size,
        "by_K": {},
    }
    for K in K_values:
        print(f"\n=== Quest K = {K} page_size={page_size} ===")
        per_layer_mass = {idx: [] for idx in layers}
        per_layer_recall = {idx: [] for idx in layers}
        for _ids, _token_m, model_m, allowed_m, _pos_ids, w_dict, qk_dict in cached:
            for idx in layers:
                q, k = qk_dict[idx]
                mass, recall = _quest_mass_recall(
                    w_dict[idx],
                    q,
                    k,
                    K,
                    page_size,
                    model_m,
                    allowed_m,
                )
                B, L = mass.shape
                keep = _query_has_k_valid_keys(
                    L, K, mass.device, B, attention_allowed_mask=allowed_m
                )
                if keep.any():
                    per_layer_mass[idx].extend(mass.masked_select(keep).tolist())
                    per_layer_recall[idx].extend(recall.masked_select(keep).tolist())

        mass_per_layer = {
            idx: (
                sum(per_layer_mass[idx]) / len(per_layer_mass[idx])
                if per_layer_mass[idx] else float("nan")
            )
            for idx in layers
        }
        recall_per_layer = {
            idx: (
                sum(per_layer_recall[idx]) / len(per_layer_recall[idx])
                if per_layer_recall[idx] else float("nan")
            )
            for idx in layers
        }
        finite_mass = [v for v in mass_per_layer.values() if not math.isnan(v)]
        finite_recall = [v for v in recall_per_layer.values() if not math.isnan(v)]
        mass_avg = sum(finite_mass) / max(1, len(finite_mass))
        recall_avg = sum(finite_recall) / max(1, len(finite_recall))

        wrappers = install_quest_attention(
            base_model, layers, K_retrieve=K, page_size=page_size
        )
        try:
            quest_ppls = [
                compute_perplexity(base_model, ids, model_m, pos_ids, target_mask=token_m)
                for ids, token_m, model_m, _allowed_m, pos_ids, *_ in cached
            ]
        finally:
            uninstall_ann_attention(wrappers)
        ppl_quest = sum(quest_ppls) / len(quest_ppls)
        ppl_gap = (ppl_quest - ppl_full) / ppl_full

        results["by_K"][K] = {
            "mass_avg": mass_avg,
            "mass_per_layer": mass_per_layer,
            "recall_avg": recall_avg,
            "recall_per_layer": recall_per_layer,
            "ppl_quest": ppl_quest,
            "ppl_gap_relative": ppl_gap,
        }
        print(
            f"  mass_avg = {mass_avg:.4f}   recall_avg = {recall_avg:.4f}   "
            f"ppl_quest = {ppl_quest:.4f}   ppl_gap = {ppl_gap:+.3%}"
        )

    skip_tag = f"_skip{skip_batches}" if skip_batches else ""
    out_path = os.path.splitext(ckpt_path)[0] + f".quest_page{page_size}{skip_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--K", default="128,256,512")
    parser.add_argument("--page-size", type=int, default=16)
    args = parser.parse_args()
    quest_sweep(
        args.ckpt,
        K_values=tuple(int(x) for x in args.K.split(",")),
        num_batches=args.num_batches,
        skip_batches=args.skip_batches,
        page_size=args.page_size,
    )


if __name__ == "__main__":
    main()
