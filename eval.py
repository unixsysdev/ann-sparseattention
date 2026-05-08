"""
Evaluation: recall@K, end-to-end perplexity (full vs. ANN-substituted),
MoE router match rate.

The dashboard panels in project.md map onto the metrics produced here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from data import build_eval_data, model_attention_mask
from inference import (
    install_ann_attention,
    uninstall_ann_attention,
)
from model import aggregate_heads


def _causal_mask(L: int, device) -> torch.Tensor:
    return torch.ones(L, L, device=device, dtype=torch.bool).tril()


def _query_has_k_valid_keys(
    L: int,
    K: int,
    device,
    B: int,
    attention_allowed_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Return [B, L] queries with enough valid keys for a K-way comparison."""
    K_eff = min(K, L)
    if attention_allowed_mask is not None:
        return attention_allowed_mask.bool().sum(dim=-1) >= K_eff
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    return (pos + 1) >= K_eff


def _per_position_mass_at_k(
    teacher_full: torch.Tensor,   # [B, H, L, L]  per-head teacher distribution
    q_search: torch.Tensor,        # [B, L, d_search]
    k_search: torch.Tensor,
    K: int,
    attention_allowed_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-attention mass captured by the search top-K.

    Returns
      per_head_mass: [B, H, L]   — sum of teacher prob at retrieved positions
      avg_mass:      [B, L]      — averaged over heads

    This is the metric that actually correlates with PPL preservation — set
    recall@K is binary "did we hit the right keys?" but mass@K weights each
    retrieved key by how much probability the teacher actually puts on it.
    """
    B, H, L, _ = teacher_full.shape
    device = q_search.device
    allowed = attention_allowed_mask.bool() if attention_allowed_mask is not None else _causal_mask(L, device)

    q_n = F.normalize(q_search, dim=-1)
    k_n = F.normalize(k_search, dim=-1)
    sim = torch.bmm(q_n, k_n.transpose(1, 2)).masked_fill(~allowed, -1e9)
    K_eff = min(K, L)
    search_top = sim.topk(K_eff, dim=-1).indices                  # [B, L, K]

    # Build a [B, L, L] bool mask of retrieved positions, then broadcast to heads.
    grid = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    grid.scatter_(-1, search_top, True)                            # [B, L, L]
    grid = grid.unsqueeze(1).expand(B, H, L, L)

    # For each (b, h, q), sum teacher probabilities over the retrieved keys.
    per_head_mass = (teacher_full * grid.to(teacher_full.dtype)).sum(-1)  # [B, H, L]
    avg_mass = per_head_mass.mean(dim=1)                                  # [B, L]
    return per_head_mass, avg_mass


def _per_position_recall(
    teacher: torch.Tensor,  # [B, L, L] head-aggregated
    q_search: torch.Tensor,  # [B, L, d_search]
    k_search: torch.Tensor,
    K: int,
    attention_allowed_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Vectorized recall@K per query position.

    For each (b, q): recall_q = |teacher_topK ∩ search_topK| / min(K, q+1).
    Returns [B, L] (positions where the denominator is 0 are NaN; caller must
    mask).
    """
    B, L, _ = q_search.shape
    device = q_search.device

    allowed = attention_allowed_mask.bool() if attention_allowed_mask is not None else _causal_mask(L, device)
    teacher_masked = teacher.masked_fill(~allowed, -1e9)

    K_eff = min(K, L)
    teacher_top = teacher_masked.topk(K_eff, dim=-1).indices  # [B, L, K]

    q_n = F.normalize(q_search, dim=-1)
    k_n = F.normalize(k_search, dim=-1)
    sim = torch.bmm(q_n, k_n.transpose(1, 2)).masked_fill(~allowed, -1e9)
    search_top = sim.topk(K_eff, dim=-1).indices  # [B, L, K]

    # Vectorized intersection size: scatter both into a [B, L, L] bool grid and AND.
    teacher_grid = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    search_grid = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    teacher_grid.scatter_(-1, teacher_top, True)
    search_grid.scatter_(-1, search_top, True)
    inter = (teacher_grid & search_grid).sum(-1)  # [B, L]

    # Denominator: min(K, number of valid keys for the query).
    denom = torch.minimum(
        torch.full((B, L), K_eff, device=device, dtype=torch.long),
        allowed.sum(dim=-1),
    ).clamp(min=1)
    return inter.float() / denom.float()


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    target_mask: torch.Tensor = None,
) -> float:
    """
    Standard NLL averaged over tokens, exp() at the end.

    Loss is averaged only over positions where attention_mask == 1 in the
    *target* (i.e. shifted) range. This way padded positions don't get
    counted in PPL even when the model still produces logits there.
    """
    with torch.no_grad():
        kwargs = dict(input_ids=input_ids, use_cache=False)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = model(**kwargs)
        logits = out.logits  # [B, L, V]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1)

    if target_mask is None and attention_mask is not None and attention_mask.dim() == 2:
        target_mask = attention_mask
    if target_mask is not None:
        # Only count tokens whose target position was real (mask==1).
        shifted_mask = target_mask[..., 1:].contiguous().view(-1).bool()
        flat_logits = flat_logits[shifted_mask]
        flat_labels = flat_labels[shifted_mask]

    loss = F.cross_entropy(flat_logits, flat_labels, reduction="mean")
    return float(torch.exp(loss).item())


def compute_nll(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    target_mask: torch.Tensor = None,
) -> float:
    """Return mean next-token NLL on the same token set as compute_perplexity."""
    with torch.no_grad():
        kwargs = dict(input_ids=input_ids, use_cache=False)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = model(**kwargs)
        logits = out.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1)

    if target_mask is None and attention_mask is not None and attention_mask.dim() == 2:
        target_mask = attention_mask
    if target_mask is not None:
        shifted_mask = target_mask[..., 1:].contiguous().view(-1).bool()
        flat_logits = flat_logits[shifted_mask]
        flat_labels = flat_labels[shifted_mask]

    return float(F.cross_entropy(flat_logits, flat_labels, reduction="mean").item())


def compute_perplexity_full_attention(
    base_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    target_mask: torch.Tensor = None,
) -> float:
    return compute_perplexity(
        base_model, input_ids, attention_mask, position_ids, target_mask
    )


def compute_perplexity_ann_substituted(
    base_model,
    search_module,
    input_ids: torch.Tensor,
    config,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    target_mask: torch.Tensor = None,
    return_router_stability: bool = False,
) -> Tuple[float, float]:
    """
    Run forward with ANN substitution on the trained layers; return
    (perplexity, router_match_rate). router_match_rate is 1.0 for dense
    models (no router); for MoE it compares top-1 expert choices between
    full-attention and ANN-substituted forwards.
    """
    layers_to_train = [
        i for i in config.full_attention_layer_indices
        if i not in config.reserved_full_attention_indices
    ]

    fwd_kwargs = dict(input_ids=input_ids, use_cache=False)
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        fwd_kwargs["position_ids"] = position_ids
    if return_router_stability:
        fwd_kwargs["output_router_logits"] = True

    with torch.no_grad():
        full_out = base_model(**fwd_kwargs)
    full_logits = full_out.logits[..., :-1, :].contiguous().view(-1, full_out.logits.size(-1)).float()
    full_labels = input_ids[..., 1:].contiguous().view(-1)
    if target_mask is None and attention_mask is not None and attention_mask.dim() == 2:
        target_mask = attention_mask
    if target_mask is not None:
        shifted_mask = target_mask[..., 1:].contiguous().view(-1).bool()
        full_logits = full_logits[shifted_mask]
        full_labels = full_labels[shifted_mask]
    ppl_full = float(torch.exp(F.cross_entropy(full_logits, full_labels, reduction="mean")))

    wrappers = install_ann_attention(
        base_model,
        search_module,
        layers_to_train,
        K_retrieve=config.K_retrieve_eval,
        use_faiss=getattr(config, "use_faiss_hnsw_at_eval", False),
        use_hnsw=getattr(config, "use_faiss_hnsw_at_eval", False),
        hnsw_M=getattr(config, "faiss_hnsw_M", 32),
        hnsw_ef_construction=getattr(config, "faiss_hnsw_ef_construction", 40),
        hnsw_ef_search=getattr(config, "faiss_hnsw_ef_search", 64),
    )
    try:
        with torch.no_grad():
            ann_out = base_model(**fwd_kwargs)
    finally:
        uninstall_ann_attention(wrappers)

    ann_logits = ann_out.logits[..., :-1, :].contiguous().view(-1, ann_out.logits.size(-1)).float()
    ann_labels = input_ids[..., 1:].contiguous().view(-1)
    if target_mask is not None:
        shifted_mask = target_mask[..., 1:].contiguous().view(-1).bool()
        ann_logits = ann_logits[shifted_mask]
        ann_labels = ann_labels[shifted_mask]
    ppl_ann = float(torch.exp(F.cross_entropy(ann_logits, ann_labels, reduction="mean")))

    # NOTE: we return ppl_ann separately (not gap), and pair it with the
    # already-computed ppl_full at the call site.
    if not return_router_stability:
        return ppl_ann, 1.0

    # Router match rate across MoE layers (top-1 expert per token).
    # Dense models have no router; return 1.0 (trivially stable).
    full_rl = getattr(full_out, "router_logits", None)
    ann_rl = getattr(ann_out, "router_logits", None)
    if full_rl is None or ann_rl is None:
        return ppl_ann, 1.0
    matches = []
    for f, a in zip(full_rl, ann_rl):
        if f is None or a is None:
            continue
        m = (f.argmax(-1) == a.argmax(-1)).float().mean().item()
        matches.append(m)
    router_match = sum(matches) / max(1, len(matches))
    # Also stash ppl_full for the caller.
    compute_perplexity_ann_substituted._last_ppl_full = ppl_full
    return ppl_ann, router_match


def evaluate(base_model, search_module, capture, config, tokenizer) -> Dict:
    """
    Returns the W&B-loggable metrics dict described in project.md.
    """
    search_module.eval()
    eval_data = build_eval_data(tokenizer, config)

    layers_to_train = [
        i for i in config.full_attention_layer_indices
        if i not in config.reserved_full_attention_indices
    ]

    K_eval = config.K_retrieve_eval
    K_curve = sorted(set(config.K_retrieve_search) | {K_eval})

    metrics: Dict = {
        "eval/recall_at_K_per_layer": {},   # at K_eval
        "eval/recall_at_K_avg": 0.0,         # at K_eval
        "eval/recall_curve": {},             # K -> avg recall across layers
        "eval/recall_curve_per_layer": {},   # layer_idx -> {K -> recall}
        "eval/mass_at_K_per_layer": {},     # at K_eval
        "eval/mass_at_K_avg": 0.0,           # at K_eval — primary retrieval metric
        "eval/mass_curve": {},               # K -> avg mass across layers
        "eval/ppl_full": 0.0,
        "eval/ppl_ann": 0.0,
        "eval/ppl_gap_relative": 0.0,
        "eval/qk_alignment_per_layer": {},
    }

    # Per-layer, per-K accumulators.
    recall_acc: Dict[int, Dict[int, List[float]]] = {
        idx: {K: [] for K in K_curve} for idx in layers_to_train
    }
    mass_acc: Dict[int, Dict[int, List[float]]] = {
        idx: {K: [] for K in K_curve} for idx in layers_to_train
    }
    full_ppls, ann_ppls, router_matches = [], [], []

    with torch.no_grad():
        for batch in eval_data:
            input_ids = batch["input_ids"].to(base_model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(base_model.device)
            segment_ids = batch.get("segment_ids")
            if segment_ids is not None:
                segment_ids = segment_ids.to(base_model.device)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(base_model.device)
            model_mask = model_attention_mask(
                attention_mask,
                segment_ids,
                block_causal_mask=getattr(config, "block_causal_mask", False),
                dtype=base_model.dtype,
            )
            allowed_mask = None
            if model_mask is not None and model_mask.dim() == 4:
                allowed_mask = (model_mask[:, 0] >= 0)

            # --- recall@K via captured teacher attention ---
            hidden_states_dict, attn_weights_dict = capture.run(
                input_ids, attention_mask=model_mask, position_ids=position_ids
            )
            q_dict, k_dict = search_module(hidden_states_dict)

            for layer_idx in layers_to_train:
                teacher_full = attn_weights_dict[layer_idx]   # [B, H, L, L]
                teacher = aggregate_heads(
                    teacher_full, mode=config.teacher_head_aggregation
                )
                for K in K_curve:
                    rec = _per_position_recall(
                        teacher,
                        q_dict[layer_idx],
                        k_dict[layer_idx],
                        K,
                        attention_allowed_mask=allowed_mask,
                    )  # [B, L]
                    B, L = rec.shape
                    mask = _query_has_k_valid_keys(
                        L, K, rec.device, B, attention_allowed_mask=allowed_mask
                    )
                    vals = rec.masked_select(mask)
                    recall_acc[layer_idx][K].extend(vals.tolist())

                    # mass@K: teacher attention probability captured by
                    # the search top-K. Better than recall when softmax
                    # is sharp.
                    _, mass = _per_position_mass_at_k(
                        teacher_full,
                        q_dict[layer_idx],
                        k_dict[layer_idx],
                        K,
                        attention_allowed_mask=allowed_mask,
                    )  # [B, L]
                    mass_vals = mass.masked_select(mask)
                    mass_acc[layer_idx][K].extend(mass_vals.tolist())

            # --- end-to-end ppl + router stability ---
            ppl_full = compute_perplexity_full_attention(
                base_model, input_ids, model_mask, position_ids, target_mask=attention_mask
            )
            ppl_ann, router_match = compute_perplexity_ann_substituted(
                base_model,
                search_module,
                input_ids,
                config,
                attention_mask=model_mask,
                position_ids=position_ids,
                target_mask=attention_mask,
                return_router_stability=config.verify_routing_stability,
            )
            full_ppls.append(ppl_full)
            ann_ppls.append(ppl_ann)
            router_matches.append(router_match)

    # Recall@K_eval per layer + average.
    for layer_idx in layers_to_train:
        vals = recall_acc[layer_idx][K_eval]
        metrics["eval/recall_at_K_per_layer"][layer_idx] = (
            sum(vals) / max(1, len(vals))
        )
        metrics["eval/recall_curve_per_layer"][layer_idx] = {
            K: (sum(recall_acc[layer_idx][K]) / max(1, len(recall_acc[layer_idx][K])))
            for K in K_curve
        }
    metrics["eval/recall_at_K_avg"] = (
        sum(metrics["eval/recall_at_K_per_layer"].values())
        / max(1, len(metrics["eval/recall_at_K_per_layer"]))
    )
    metrics["eval/recall_curve"] = {
        K: (
            sum(metrics["eval/recall_curve_per_layer"][li][K] for li in layers_to_train)
            / max(1, len(layers_to_train))
        )
        for K in K_curve
    }

    # mass@K aggregation
    for layer_idx in layers_to_train:
        vals = mass_acc[layer_idx][K_eval]
        metrics["eval/mass_at_K_per_layer"][layer_idx] = (
            sum(vals) / max(1, len(vals))
        )
    metrics["eval/mass_at_K_avg"] = (
        sum(metrics["eval/mass_at_K_per_layer"].values())
        / max(1, len(metrics["eval/mass_at_K_per_layer"]))
    )
    metrics["eval/mass_curve"] = {
        K: sum(
            sum(mass_acc[li][K]) / max(1, len(mass_acc[li][K]))
            for li in layers_to_train
        ) / max(1, len(layers_to_train))
        for K in K_curve
    }

    metrics["eval/ppl_full"] = sum(full_ppls) / max(1, len(full_ppls))
    metrics["eval/ppl_ann"] = sum(ann_ppls) / max(1, len(ann_ppls))
    metrics["eval/ppl_gap_relative"] = (
        (metrics["eval/ppl_ann"] - metrics["eval/ppl_full"])
        / max(1e-9, metrics["eval/ppl_full"])
    )

    # Only emit the router-match metric when MoE stability is meaningful.
    # On dense models the value is trivially 1.0 and clutters the dashboard.
    if config.verify_routing_stability:
        metrics["eval/router_match_rate"] = (
            sum(router_matches) / max(1, len(router_matches))
        )

    search_module.train()
    return metrics
