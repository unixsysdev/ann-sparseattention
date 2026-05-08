"""
Decode-like dynamic-index proxy.

The current ANN attention wrapper is prefill-only (`use_cache=False`), so this
script does not run generation-time cache updates. Instead it asks the narrower
question the cache experiment would depend on:

  For suffix queries that behave like decoded tokens, how much teacher attention
  mass can learned retrieval capture if the search index is dynamic (all prior
  suffix keys are visible) versus static (only prefill keys plus a recent local
  decode window are visible)?

This is a retrieval-capability proxy, not a wall-clock or task-accuracy result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from data import build_eval_data, model_attention_mask  # noqa: E402
from model import FrozenForwardCapture, SearchProjectionModule  # noqa: E402


def config_from_checkpoint(ckpt: dict) -> Config:
    cfg = Config()
    for key, value in ckpt.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _suffix_static_mask(
    dynamic_allowed: torch.Tensor,
    prefill_len: int,
    local_window: int,
) -> torch.Tensor:
    """
    dynamic_allowed: [B, L, L] bool causal/segment mask.

    Static proxy:
      - prefill queries see the same keys as dynamic
      - suffix queries see prefill keys plus the recent local suffix window
      - older suffix keys are hidden, simulating a frozen prefill index
    """
    B, L, _ = dynamic_allowed.shape
    device = dynamic_allowed.device
    q = torch.arange(L, device=device).view(1, L, 1)
    k = torch.arange(L, device=device).view(1, 1, L)
    in_prefill_query = q < prefill_len
    key_in_prefill = k < prefill_len
    key_in_recent_suffix = (k >= prefill_len) & (k >= q - local_window + 1)
    static_visible = in_prefill_query | key_in_prefill | key_in_recent_suffix
    return dynamic_allowed & static_visible.expand(B, L, L)


def _mass_at_k(
    teacher_full: torch.Tensor,
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    K: int,
    allowed: torch.Tensor,
    query_keep: torch.Tensor,
) -> float:
    B, H, L, _ = teacher_full.shape
    q_n = F.normalize(q_search, dim=-1)
    k_n = F.normalize(k_search, dim=-1)
    sim = torch.bmm(q_n, k_n.transpose(1, 2)).masked_fill(~allowed, -1e9)
    top = sim.topk(min(K, L), dim=-1).indices

    grid = torch.zeros(B, L, L, dtype=torch.bool, device=teacher_full.device)
    grid.scatter_(-1, top, True)
    mass = (teacher_full * grid.unsqueeze(1).to(teacher_full.dtype)).sum(-1)
    keep = query_keep.unsqueeze(1).expand(B, H, L)
    if keep.sum() == 0:
        return float("nan")
    return float(mass.masked_select(keep).mean().item())


def _available_teacher_mass(
    teacher_full: torch.Tensor,
    allowed: torch.Tensor,
    query_keep: torch.Tensor,
) -> float:
    mass = (teacher_full * allowed.unsqueeze(1).to(teacher_full.dtype)).sum(-1)
    keep = query_keep.unsqueeze(1).expand_as(mass)
    if keep.sum() == 0:
        return float("nan")
    return float(mass.masked_select(keep).mean().item())


def run_proxy(
    ckpt_path: str,
    num_batches: int,
    K: int,
    prefill_len: int,
    local_window: int,
    out_path: str | None,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)

    print(f"Loading base model {cfg.base_model_name} ...")
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

    layers = [
        i
        for i in cfg.full_attention_layer_indices
        if i not in cfg.reserved_full_attention_indices
    ]
    search = SearchProjectionModule(
        d_model=base.config.hidden_size,
        d_search=cfg.d_search,
        layer_indices=layers,
        use_mlp=cfg.use_mlp_proj,
    ).to(base.device).to(torch.bfloat16)
    search.load_state_dict(ckpt["search_module"])
    search.eval()
    print(f"Loaded step {ckpt['step']} for layers {layers}")

    capture = FrozenForwardCapture(base, layers, qk_reconstruction=cfg.qk_reconstruction)
    eval_data = list(build_eval_data(tokenizer, cfg, num_batches=num_batches))

    per_layer = {
        idx: {
            "dynamic_mass": [],
            "static_mass": [],
            "static_teacher_available": [],
        }
        for idx in layers
    }

    for bi, batch in enumerate(eval_data):
        input_ids = batch["input_ids"].to(base.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(base.device)
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            segment_ids = segment_ids.to(base.device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(base.device)

        model_mask = model_attention_mask(
            attention_mask,
            segment_ids,
            block_causal_mask=getattr(cfg, "block_causal_mask", False),
            dtype=base.dtype,
        )
        if model_mask is not None and model_mask.dim() == 4:
            dynamic_allowed = model_mask[:, 0] >= 0
        else:
            L = input_ids.shape[1]
            dynamic_allowed = torch.ones(L, L, device=base.device, dtype=torch.bool).tril()
            dynamic_allowed = dynamic_allowed.unsqueeze(0).expand(input_ids.shape[0], L, L)
            if attention_mask is not None:
                dynamic_allowed = dynamic_allowed & attention_mask[:, None, :].bool()

        static_allowed = _suffix_static_mask(dynamic_allowed, prefill_len, local_window)
        B, L = input_ids.shape
        q_pos = torch.arange(L, device=base.device).view(1, L).expand(B, L)
        query_keep = q_pos >= prefill_len
        query_keep = query_keep & (dynamic_allowed.sum(dim=-1) >= min(K, L))
        query_keep = query_keep & (static_allowed.sum(dim=-1) > 0)

        h_dict, w_dict = capture.run(
            input_ids,
            attention_mask=model_mask,
            position_ids=position_ids,
        )
        with torch.no_grad():
            q_dict, k_dict = search(h_dict)

        print(
            f"[batch {bi}] suffix queries kept: "
            f"{int(query_keep.sum().item())}/{query_keep.numel()}"
        )
        for idx in layers:
            teacher = w_dict[idx]
            dynamic_mass = _mass_at_k(
                teacher, q_dict[idx], k_dict[idx], K, dynamic_allowed, query_keep
            )
            static_mass = _mass_at_k(
                teacher, q_dict[idx], k_dict[idx], K, static_allowed, query_keep
            )
            static_avail = _available_teacher_mass(teacher, static_allowed, query_keep)
            per_layer[idx]["dynamic_mass"].append(dynamic_mass)
            per_layer[idx]["static_mass"].append(static_mass)
            per_layer[idx]["static_teacher_available"].append(static_avail)

    summary = {
        "checkpoint": ckpt_path,
        "step": ckpt["step"],
        "K": K,
        "prefill_len": prefill_len,
        "local_window": local_window,
        "num_batches": len(eval_data),
        "layers": {},
    }
    for idx, vals in per_layer.items():
        dyn = torch.tensor(vals["dynamic_mass"], dtype=torch.float32)
        sta = torch.tensor(vals["static_mass"], dtype=torch.float32)
        avail = torch.tensor(vals["static_teacher_available"], dtype=torch.float32)
        summary["layers"][str(idx)] = {
            "dynamic_mass": float(torch.nanmean(dyn).item()),
            "static_mass": float(torch.nanmean(sta).item()),
            "static_teacher_available": float(torch.nanmean(avail).item()),
            "dynamic_minus_static": float(torch.nanmean(dyn - sta).item()),
        }

    dyn_avg = torch.tensor(
        [v["dynamic_mass"] for v in summary["layers"].values()], dtype=torch.float32
    )
    sta_avg = torch.tensor(
        [v["static_mass"] for v in summary["layers"].values()], dtype=torch.float32
    )
    avail_avg = torch.tensor(
        [v["static_teacher_available"] for v in summary["layers"].values()],
        dtype=torch.float32,
    )
    summary["aggregate"] = {
        "dynamic_mass": float(dyn_avg.mean().item()),
        "static_mass": float(sta_avg.mean().item()),
        "static_teacher_available": float(avail_avg.mean().item()),
        "dynamic_minus_static": float((dyn_avg - sta_avg).mean().item()),
    }

    print(json.dumps(summary, indent=2))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--prefill-len", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_proxy(
        args.ckpt,
        args.num_batches,
        args.K,
        args.prefill_len,
        args.local_window,
        args.out,
    )


if __name__ == "__main__":
    main()
