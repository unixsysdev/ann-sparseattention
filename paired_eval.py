"""
Paired PPL/NLL comparison for full attention, learned search, and Quest.

Runs all methods on the exact same eval batches and bootstraps per-batch loss
deltas. This removes most dataset-slice variance from learned-vs-Quest claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_eval_data, model_attention_mask  # noqa: E402
from eval import compute_nll  # noqa: E402
from inference import (  # noqa: E402
    install_ann_attention,
    install_quest_attention,
    uninstall_ann_attention,
)
from k_sweep import config_from_checkpoint  # noqa: E402
from model import SearchProjectionModule  # noqa: E402


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _bootstrap_ci(
    xs: list[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    if not xs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": _mean(xs), "lo": lo, "hi": hi}


def _method_nlls(
    base_model,
    cached_batches,
    method: str,
    layers: list[int],
    K: int,
    search=None,
    use_faiss: bool = False,
    page_size: int = 16,
    cfg=None,
) -> list[float]:
    wrappers = []
    if method == "learned":
        wrappers = install_ann_attention(
            base_model,
            search,
            layers,
            K_retrieve=K,
            use_faiss=use_faiss,
            use_hnsw=use_faiss,
            hnsw_M=getattr(cfg, "faiss_hnsw_M", 32),
            hnsw_ef_construction=getattr(cfg, "faiss_hnsw_ef_construction", 40),
            hnsw_ef_search=getattr(cfg, "faiss_hnsw_ef_search", 64),
        )
    elif method == "quest":
        wrappers = install_quest_attention(
            base_model, layers, K_retrieve=K, page_size=page_size
        )
    elif method != "full":
        raise ValueError(f"unknown method: {method}")

    try:
        return [
            compute_nll(base_model, ids, model_m, pos_ids, target_mask=token_m)
            for ids, token_m, model_m, pos_ids in cached_batches
        ]
    finally:
        if wrappers:
            uninstall_ann_attention(wrappers)


def paired_eval(
    ckpt_path: str,
    K: int = 128,
    num_batches: int = 32,
    skip_batches: int = 0,
    page_size: int = 16,
    use_faiss: bool = False,
    n_boot: int = 10000,
    seed: int = 0,
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
    search = SearchProjectionModule(
        d_model=base_model.config.hidden_size,
        d_search=cfg.d_search,
        layer_indices=layers,
        use_mlp=cfg.use_mlp_proj,
    ).to(base_model.device).to(torch.bfloat16)
    search.load_state_dict(ckpt["search_module"])
    search.eval()

    eval_data_all = list(
        build_eval_data(tokenizer, cfg, num_batches=num_batches + skip_batches)
    )
    eval_data = eval_data_all[skip_batches:]
    if len(eval_data) < num_batches:
        print(
            f"[warn] requested {num_batches} batches after skip={skip_batches}, "
            f"got {len(eval_data)}"
        )

    print("Caching eval batches...")
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
        cached.append((input_ids, token_mask, model_mask, position_ids))

    print("Running full attention...")
    full = _method_nlls(base_model, cached, "full", layers, K)
    print("Running learned search...")
    learned = _method_nlls(
        base_model,
        cached,
        "learned",
        layers,
        K,
        search=search,
        use_faiss=use_faiss,
        cfg=cfg,
    )
    print("Running Quest-style page baseline...")
    quest = _method_nlls(
        base_model,
        cached,
        "quest",
        layers,
        K,
        page_size=page_size,
    )

    learned_delta = [a - b for a, b in zip(learned, full)]
    quest_delta = [a - b for a, b in zip(quest, full)]
    diff = [a - b for a, b in zip(learned_delta, quest_delta)]

    result = {
        "ckpt": ckpt_path,
        "step": ckpt.get("step"),
        "K": K,
        "page_size": page_size,
        "num_batches": len(cached),
        "skip_batches": skip_batches,
        "use_faiss": use_faiss,
        "nll": {
            "full_mean": _mean(full),
            "learned_mean": _mean(learned),
            "quest_mean": _mean(quest),
        },
        "ppl": {
            "full": math.exp(_mean(full)),
            "learned": math.exp(_mean(learned)),
            "quest": math.exp(_mean(quest)),
        },
        "relative_ppl_gap": {
            "learned_vs_full": math.exp(_mean(learned) - _mean(full)) - 1.0,
            "quest_vs_full": math.exp(_mean(quest) - _mean(full)) - 1.0,
            "learned_vs_quest": math.exp(_mean(diff)) - 1.0,
        },
        "paired_nll_delta": {
            "learned_minus_full": _bootstrap_ci(learned_delta, n_boot, seed),
            "quest_minus_full": _bootstrap_ci(quest_delta, n_boot, seed + 1),
            "learned_minus_quest": _bootstrap_ci(diff, n_boot, seed + 2),
        },
        "per_batch": [
            {
                "batch": i,
                "full_nll": f,
                "learned_nll": l,
                "quest_nll": q,
                "learned_minus_full": l - f,
                "quest_minus_full": q - f,
                "learned_minus_quest": l - q,
            }
            for i, (f, l, q) in enumerate(zip(full, learned, quest))
        ],
    }

    suffix = "faiss" if use_faiss else "exact"
    skip_tag = f"_skip{skip_batches}" if skip_batches else ""
    out_path = (
        os.path.splitext(ckpt_path)[0]
        + f".paired_K{K}_{suffix}_quest_page{page_size}{skip_tag}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    ci = result["paired_nll_delta"]["learned_minus_quest"]
    print(
        f"full ppl={result['ppl']['full']:.4f} "
        f"learned ppl={result['ppl']['learned']:.4f} "
        f"quest ppl={result['ppl']['quest']:.4f}"
    )
    print(
        "learned-quest NLL delta: "
        f"{ci['mean']:+.6f} [{ci['lo']:+.6f}, {ci['hi']:+.6f}]"
    )
    print(
        "relative PPL gaps: "
        f"learned/full={result['relative_ppl_gap']['learned_vs_full']:+.3%} "
        f"quest/full={result['relative_ppl_gap']['quest_vs_full']:+.3%} "
        f"learned/quest={result['relative_ppl_gap']['learned_vs_quest']:+.3%}"
    )
    print(f"Wrote {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--use-faiss", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paired_eval(
        args.ckpt,
        K=args.K,
        num_batches=args.num_batches,
        skip_batches=args.skip_batches,
        page_size=args.page_size,
        use_faiss=args.use_faiss,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
