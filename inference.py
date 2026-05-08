"""
ANN-substituted attention.

Two retrieval paths:
  * `_exact_topk_search`  — builds the dense [B, L, L] similarity matrix and
    takes top-K. Quadratic in L; used for analysis (recall, mass@K, PPL gap).
  * `_faiss_topk_search`  — per-batch CPU FAISS HNSW index. Correct, but
    a research-quality prototype: it does GPU→CPU transfers, builds an
    index per forward, and filters causal hits with a Python loop. Not a
    deployable runtime. A production runtime would use a GPU-resident topk
    kernel (Triton / CUTLASS) or a paged GPU index that's incrementally
    updated alongside the KV cache.

Both paths share the same wrapper that monkey-patches a target layer's
self-attention forward:
  1. Compute Q, K, V + Qwen3 q_norm/k_norm + RoPE as the original does.
  2. Get (q_search, k_search) from the trained SearchProjection.
  3. Retrieve top-K key indices (causal-respecting).
  4. Run standard attention restricted to the retrieved K keys.

The helpers in this module set `use_cache=False`, so the substitution path
is prefill-only. Adding decode-mode requires either incremental
index updates per generated token, or a different wrapper that consumes the
KV cache directly. Out of scope for the pilot/headline reported here.
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _exact_topk_search(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    K: int,
    causal: bool = True,
    key_mask: torch.Tensor = None,
    return_valid_mask: bool = False,
) -> torch.Tensor:
    """
    q_search, k_search: [B, L, d_search].
    key_mask: optional [B, L] (1 = real token, 0 = pad) — pad keys are
    excluded from retrieval candidates.
    Returns indices [B, L, K] of top-K keys by cosine similarity of search
    vectors, restricted to causal (key index <= query index).
    """
    B, L, _ = q_search.shape
    raw_key_mask = key_mask
    key_mask = _normalize_key_mask(raw_key_mask, L)
    allowed_mask = _normalize_allowed_mask(raw_key_mask, L)
    q_n = F.normalize(q_search, dim=-1)
    k_n = F.normalize(k_search, dim=-1)
    sim = torch.bmm(q_n, k_n.transpose(1, 2))  # [B, L, L]
    valid = None
    if allowed_mask is not None:
        valid = allowed_mask
        sim = sim.masked_fill(~allowed_mask, -1e9)
    elif causal:
        mask = torch.ones(L, L, device=sim.device, dtype=torch.bool).tril()
        valid = mask.unsqueeze(0).expand(B, L, L)
        sim = sim.masked_fill(~mask, -1e9)
    if key_mask is not None and allowed_mask is None:
        # Block pad keys for every query.
        key_valid = key_mask.unsqueeze(1).bool()
        valid = valid & key_valid if valid is not None else key_valid.expand(B, L, L)
        sim = sim.masked_fill(~key_valid, -1e9)
    K_eff = min(K, L)
    top = sim.topk(K_eff, dim=-1).indices  # [B, L, K_eff]
    top_valid = None
    if valid is not None:
        top_valid = valid.gather(-1, top)
        fallback_key_mask = key_mask
        if fallback_key_mask is None:
            fallback_key_mask = torch.ones(B, L, dtype=torch.bool, device=sim.device)
        fallback = _fallback_key_indices(fallback_key_mask, L, allowed_mask).unsqueeze(-1)
        top = torch.where(top_valid, top, fallback)
    if return_valid_mask:
        if top_valid is None:
            top_valid = torch.ones_like(top, dtype=torch.bool)
        return top, top_valid
    return top


# Diagnostics for the FAISS path. Every call appends a dict to FAISS_STATS;
# callers reset and aggregate as they wish:
#   self_pad_rate: fraction of (b, q, k) slots filled with the query position
#     itself because FAISS over-fetch + causal filter left < K real causal
#     hits (high for early queries q < K).
#   causal_fill_rate: fraction of slots filled with a strictly-prior position
#     (retrieved < q) — the actual useful retrieval signal.
#   self_attn_rate: fraction at retrieved == q legitimately returned by FAISS.
# If self_pad_rate is non-trivial, K-sweep numbers are partially driven by
# self-padding rather than learned retrieval.
FAISS_STATS: list = []


def _normalize_key_mask(key_mask: torch.Tensor, L: int) -> Optional[torch.Tensor]:
    """
    Return a [B, L] boolean key-valid mask from either the original tokenizer
    attention_mask ([B, L], 1=real) or the expanded HF causal mask
    ([B, 1, L, L], 0/finite=allowed, -inf/min=masked).
    """
    if key_mask is None:
        return None
    if key_mask.dim() == 2:
        return key_mask[:, :L].bool()
    if key_mask.dim() == 4:
        # HF passes the already-expanded additive causal mask down to attention.
        # A key is real if any query row is allowed to attend to that key.
        km = key_mask[..., :L, :L]
        if km.dtype == torch.bool:
            return (~km).any(dim=-2).squeeze(1)
        return (km >= 0).any(dim=-2).squeeze(1)
    raise ValueError(f"Unsupported key_mask shape: {tuple(key_mask.shape)}")


def _normalize_allowed_mask(key_mask: torch.Tensor, L: int) -> Optional[torch.Tensor]:
    """Return [B, L, L] query-key allowed mask when a 4D attention mask is supplied."""
    if key_mask is None or key_mask.dim() != 4:
        return None
    km = key_mask[..., :L, :L]
    if km.dtype == torch.bool:
        return ~km.squeeze(1)
    return (km >= 0).squeeze(1)


def _fallback_key_indices(
    key_mask: torch.Tensor,
    L: int,
    allowed_mask: torch.Tensor = None,
) -> torch.Tensor:
    """For each query position, choose the latest real causal key as filler."""
    B = key_mask.shape[0]
    device = key_mask.device
    key_pos = torch.arange(L, device=device).view(1, 1, L)
    if allowed_mask is not None:
        valid = allowed_mask
    else:
        q_pos = torch.arange(L, device=device).view(1, L, 1)
        valid = key_mask[:, None, :] & (key_pos <= q_pos)
    scored = torch.where(
        valid,
        key_pos.expand(B, L, L),
        torch.full((B, L, L), -1, device=device, dtype=key_pos.dtype),
    )
    fallback = scored.max(dim=-1).values
    return fallback.clamp(min=0)


def _faiss_topk_search(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    K: int,
    causal: bool = True,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 64,
    key_mask: torch.Tensor = None,
    return_valid_mask: bool = False,
) -> torch.Tensor:
    """
    FAISS-backed approximate top-K.

    use_hnsw=True (default for the headline result):
      Builds an HNSW index per batch with default-ish params. Demonstrates
      that the alignment training has produced search vectors that work with
      an off-the-shelf ANN index — the OOD-fix demonstration.

    use_hnsw=False:
      Exact inner product (IndexFlatIP) — used for reference comparisons.

    Falls back to `_exact_topk_search` if faiss is not installed.
    """
    try:
        import faiss
    except ImportError:
        return _exact_topk_search(
            q_search,
            k_search,
            K,
            causal=causal,
            return_valid_mask=return_valid_mask,
        )

    B, L, d = q_search.shape
    raw_key_mask = key_mask
    key_mask = _normalize_key_mask(raw_key_mask, L)
    allowed_mask = _normalize_allowed_mask(raw_key_mask, L)
    K_eff = min(K, L)

    if allowed_mask is not None:
        return _faiss_topk_search_allowed_segments(
            q_search,
            k_search,
            K_eff,
            allowed_mask,
            key_mask,
            use_hnsw=use_hnsw,
            hnsw_M=hnsw_M,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            return_valid_mask=return_valid_mask,
        )

    out = torch.empty(B, L, K_eff, dtype=torch.long, device=q_search.device)
    out_valid = torch.empty(B, L, K_eff, dtype=torch.bool, device=q_search.device)

    # Diagnostic counters: how many slots got self-padded vs. filled with a
    # strictly-prior causal neighbor.
    n_self_pad = 0          # padded with q (FAISS returned fewer than K causal hits)
    n_strict_prior = 0      # retrieved index < q
    n_at_self = 0           # retrieved index == q (legitimate self-attention)
    n_total = 0

    # Per-batch pad mask in CPU bool form for cheap filtering inside the loop.
    if key_mask is not None:
        pad_b = (~key_mask.bool()).cpu()  # True at pad
        fallback_keys = _fallback_key_indices(key_mask, L, allowed_mask)
    else:
        pad_b = None
        fallback_keys = None

    for b in range(B):
        kb = k_search[b].detach().float().cpu().numpy()
        qb = q_search[b].detach().float().cpu().numpy()
        # Cosine == inner product on L2-normalized vectors.
        kb_n = kb / (1e-9 + (kb ** 2).sum(-1, keepdims=True) ** 0.5)
        qb_n = qb / (1e-9 + (qb ** 2).sum(-1, keepdims=True) ** 0.5)

        if use_hnsw:
            index = faiss.IndexHNSWFlat(d, hnsw_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = hnsw_ef_construction
            index.hnsw.efSearch = hnsw_ef_search
        else:
            index = faiss.IndexFlatIP(d)
        index.add(kb_n)

        # Over-fetch then filter causal violations.
        over = min(L, max(K_eff * 4, K_eff + 16))
        _, ids = index.search(qb_n, over)
        ids_t = torch.from_numpy(ids).to(q_search.device)  # [L, over]
        q_pos = torch.arange(L, device=q_search.device).unsqueeze(-1)
        valid = ids_t <= q_pos
        if allowed_mask is not None:
            allowed_b = allowed_mask[b].to(q_search.device)
            row_pos = torch.arange(L, device=q_search.device).unsqueeze(-1)
            valid = allowed_b[row_pos, ids_t.clamp(min=0)]
        if pad_b is not None:
            # Drop retrieved positions that point at pad keys.
            pad_b_dev = pad_b[b].to(q_search.device)         # [L]
            is_pad_key = pad_b_dev[ids_t.clamp(min=0)]       # [L, over]
            valid = valid & ~is_pad_key
        ids_t = ids_t.masked_fill(~valid, -1)
        for q in range(L):
            row = ids_t[q]
            row = row[row >= 0][: K_eff]
            n_real = int(row.numel())
            if n_real < K_eff:
                fallback = int(fallback_keys[b, q].item()) if fallback_keys is not None else int(q)
                pad = torch.full(
                    (K_eff - n_real,),
                    fallback,
                    device=q_search.device,
                    dtype=torch.long,
                )
                row = torch.cat([row, pad])
                n_self_pad += K_eff - n_real
            real = row[:n_real]
            n_strict_prior += int((real < q).sum().item())
            n_at_self += int((real == q).sum().item())
            n_total += K_eff
            out[b, q, : K_eff] = row[: K_eff]
            out_valid[b, q, : K_eff] = torch.arange(
                K_eff, device=q_search.device
            ) < n_real

    FAISS_STATS.append(
        {
            "self_pad_rate": n_self_pad / max(1, n_total),
            "causal_fill_rate": n_strict_prior / max(1, n_total),
            "self_attn_rate": n_at_self / max(1, n_total),
            "B": B, "L": L, "K": K_eff,
        }
    )
    if return_valid_mask:
        return out, out_valid
    return out


def _faiss_topk_search_allowed_segments(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    K_eff: int,
    allowed_mask: torch.Tensor,
    key_mask: torch.Tensor,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 64,
    return_valid_mask: bool = False,
):
    """FAISS search with per-segment indexes derived from a [B,L,L] mask."""
    import faiss

    B, L, d = q_search.shape
    out = torch.empty(B, L, K_eff, dtype=torch.long, device=q_search.device)
    out_valid = torch.empty(B, L, K_eff, dtype=torch.bool, device=q_search.device)
    fallback_key_mask = key_mask
    if fallback_key_mask is None:
        fallback_key_mask = torch.ones(B, L, dtype=torch.bool, device=q_search.device)
    fallback_keys = _fallback_key_indices(fallback_key_mask, L, allowed_mask)

    n_self_pad = 0
    n_strict_prior = 0
    n_at_self = 0
    n_total = 0

    for b in range(B):
        starts = torch.full((L,), -1, dtype=torch.long, device=q_search.device)
        for q in range(L):
            valid = allowed_mask[b, q].nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                starts[q] = valid[0]

        for start in starts.unique().tolist():
            if start < 0:
                continue
            q_rows = (starts == start).nonzero(as_tuple=False).flatten()
            if q_rows.numel() == 0:
                continue
            end = int(q_rows.max().item()) + 1
            seg_len = end - int(start)
            if seg_len <= 0:
                continue

            kb = k_search[b, start:end].detach().float().cpu().numpy()
            qb = q_search[b, q_rows].detach().float().cpu().numpy()
            kb_n = kb / (1e-9 + (kb ** 2).sum(-1, keepdims=True) ** 0.5)
            qb_n = qb / (1e-9 + (qb ** 2).sum(-1, keepdims=True) ** 0.5)

            if use_hnsw:
                index = faiss.IndexHNSWFlat(d, hnsw_M, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = hnsw_ef_construction
                index.hnsw.efSearch = hnsw_ef_search
            else:
                index = faiss.IndexFlatIP(d)
            index.add(kb_n)

            over = min(seg_len, max(K_eff * 4, K_eff + 16))
            _, ids = index.search(qb_n, over)
            ids_t = torch.from_numpy(ids).to(q_search.device) + int(start)

            for i, q_t in enumerate(q_rows):
                q = int(q_t.item())
                row = ids_t[i]
                valid = allowed_mask[b, q, row.clamp(min=0, max=L - 1)]
                row = row.masked_fill(~valid, -1)
                row = row[row >= 0][:K_eff]
                n_real = int(row.numel())
                if n_real < K_eff:
                    fallback = int(fallback_keys[b, q].item())
                    pad = torch.full(
                        (K_eff - n_real,),
                        fallback,
                        device=q_search.device,
                        dtype=torch.long,
                    )
                    row = torch.cat([row, pad])
                    n_self_pad += K_eff - n_real
                real = row[:n_real]
                n_strict_prior += int((real < q).sum().item())
                n_at_self += int((real == q).sum().item())
                n_total += K_eff
                out[b, q] = row[:K_eff]
                out_valid[b, q] = torch.arange(K_eff, device=q_search.device) < n_real

        missing = starts < 0
        if missing.any():
            q_rows = missing.nonzero(as_tuple=False).flatten()
            fill = fallback_keys[b, q_rows].unsqueeze(-1).expand(-1, K_eff)
            out[b, q_rows] = fill
            out_valid[b, q_rows] = False
            n_self_pad += int(q_rows.numel()) * K_eff
            n_total += int(q_rows.numel()) * K_eff

    FAISS_STATS.append(
        {
            "self_pad_rate": n_self_pad / max(1, n_total),
            "causal_fill_rate": n_strict_prior / max(1, n_total),
            "self_attn_rate": n_at_self / max(1, n_total),
            "B": B, "L": L, "K": K_eff,
        }
    )
    if return_valid_mask:
        return out, out_valid
    return out


def _gather_kv(
    k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k, v: [B, H_kv, L, d_head]
    indices: [B, L, K] or [B, H, L, K] (key positions, in [0, L))
    Returns:
      k_gathered: [B, H_kv, L, K, d_head]
      v_gathered: [B, H_kv, L, K, d_head]
    """
    B, H_kv, L, d_head = k.shape
    K = indices.shape[-1]
    # Expand to [B, H_kv, L, K, d_head] index. A [B,H,L,K] index supports
    # head-specific selectors such as Quest pages; [B,L,K] broadcasts to heads.
    if indices.dim() == 3:
        idx = indices.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, L, K, d_head)
    elif indices.dim() == 4:
        idx = indices.unsqueeze(-1).expand(B, H_kv, L, K, d_head)
    else:
        raise ValueError(f"Unsupported retrieval index shape: {tuple(indices.shape)}")
    k_exp = k.unsqueeze(2).expand(B, H_kv, L, L, d_head)  # [B, H_kv, L_q, L_k, d]
    v_exp = v.unsqueeze(2).expand(B, H_kv, L, L, d_head)
    k_gathered = k_exp.gather(3, idx)
    v_gathered = v_exp.gather(3, idx)
    return k_gathered, v_gathered


def _ann_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    retrieved: torch.Tensor,
    retrieved_valid: torch.Tensor = None,
) -> torch.Tensor:
    """
    q: [B, H_q,  L, d_head]
    k: [B, H_kv, L, d_head]
    v: [B, H_kv, L, d_head]
    retrieved: [B, L, K]   key indices, causal-respecting
    Returns: [B, H_q, L, d_head]
    """
    B, H_q, L, d_head = q.shape
    H_kv = k.shape[1]
    if H_q != H_kv:
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    # k_gathered: [B, H, L, K, d_head]
    k_g, v_g = _gather_kv(k, v, retrieved)
    # scores: einsum over d_head; q [B,H,L,d_head] vs k_g [B,H,L,K,d_head]
    scores = torch.einsum("bhld,bhlkd->bhlk", q, k_g) / math.sqrt(d_head)
    if retrieved_valid is not None:
        if retrieved_valid.dim() == 3:
            retrieved_valid = retrieved_valid.unsqueeze(1)
        scores = scores.masked_fill(
            ~retrieved_valid,
            torch.finfo(scores.dtype).min,
        )
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    # out: [B, H, L, d_head]
    out = torch.einsum("bhlk,bhlkd->bhld", weights, v_g)
    return out


def _quest_page_search(
    q: torch.Tensor,
    k: torch.Tensor,
    K: int,
    page_size: int = 16,
    key_mask: torch.Tensor = None,
    return_valid_mask: bool = False,
) -> torch.Tensor:
    """
    Quest-style page retrieval over native post-RoPE Q/K.

    q, k: [B, H, L, d_head] after repeating KV heads to query-head count.
    Returns token indices [B, H, L, K_eff] and optionally a validity mask.
    """
    B, H, L, d = q.shape
    K_eff = min(K, L)
    pages_to_take = max(1, math.ceil(K_eff / page_size))
    padded_L = math.ceil(L / page_size) * page_size
    P = padded_L // page_size
    device = q.device

    allowed = _normalize_allowed_mask(key_mask, L)
    if allowed is None:
        causal = torch.ones(L, L, device=device, dtype=torch.bool).tril()
        allowed = causal.unsqueeze(0).expand(B, L, L)

    pad_len = padded_L - L
    if pad_len:
        k_pad = F.pad(k, (0, 0, 0, pad_len))
    else:
        k_pad = k
    k_pages = k_pad.view(B, H, P, page_size, d)
    k_min = k_pages.min(dim=3).values
    k_max = k_pages.max(dim=3).values

    token_pad = torch.arange(padded_L, device=device).view(P, page_size)
    page_token_valid = token_pad < L
    page_allowed = torch.zeros(B, L, P, dtype=torch.bool, device=device)
    for p in range(P):
        tok = token_pad[p][page_token_valid[p]]
        if tok.numel() > 0:
            page_allowed[:, :, p] = allowed[:, :, tok].any(dim=-1)

    retrieved = torch.empty(B, H, L, K_eff, dtype=torch.long, device=device)
    retrieved_valid = torch.empty(B, H, L, K_eff, dtype=torch.bool, device=device)
    offsets = torch.arange(page_size, device=device)
    fallback_key_mask = _normalize_key_mask(key_mask, L)
    if fallback_key_mask is None:
        fallback_key_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    fallback = _fallback_key_indices(fallback_key_mask, L, allowed)

    for b in range(B):
        page_allowed_b = page_allowed[b]
        for h in range(H):
            q_bh = q[b, h].float()  # [L, d]
            choice = torch.where(
                q_bh.unsqueeze(1) >= 0,
                k_max[b, h].float().unsqueeze(0),
                k_min[b, h].float().unsqueeze(0),
            )
            scores = (q_bh.unsqueeze(1) * choice).sum(dim=-1)
            scores = scores.masked_fill(~page_allowed_b, -1e9)
            page_top = scores.topk(min(pages_to_take, P), dim=-1).indices
            tok = (page_top.unsqueeze(-1) * page_size + offsets).flatten(1)
            tok = tok[:, :K_eff]
            tok_valid = tok < L
            tok_clamped = tok.clamp(max=L - 1)
            row = torch.arange(L, device=device).unsqueeze(-1)
            tok_valid = tok_valid & allowed[b, row, tok_clamped]
            fill = fallback[b].unsqueeze(-1).expand(L, K_eff)
            retrieved[b, h] = torch.where(tok_valid, tok_clamped, fill)
            retrieved_valid[b, h] = tok_valid

    if return_valid_mask:
        return retrieved, retrieved_valid
    return retrieved


# =============================================================================
# Wrapper that monkey-patches a target attention's forward
# =============================================================================


class ANNAttentionWrapper:
    """
    Wraps a single self-attention module. When forward is called, runs the
    standard Q/K/V projections + RoPE, then substitutes ANN-restricted
    attention for the dense softmax. The surrounding layer (output projection,
    residual, MLP) is unchanged.
    """

    def __init__(
        self,
        attention_module,
        search_projection,
        K_retrieve: int,
        use_faiss: bool = False,
        use_hnsw: bool = True,
        hnsw_M: int = 32,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 64,
    ):
        self.attention_module = attention_module
        self.search_projection = search_projection
        self.K_retrieve = K_retrieve
        self.use_faiss = use_faiss
        self.use_hnsw = use_hnsw
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.original_forward = attention_module.forward

    def install(self):
        attn = self.attention_module
        wrapper = self

        def patched_forward(self, hidden_states, *args, **kwargs):
            B, L, _ = hidden_states.shape
            num_heads = self.config.num_attention_heads
            num_kv = getattr(self.config, "num_key_value_heads", num_heads)
            head_dim = getattr(
                self.config, "head_dim", self.config.hidden_size // num_heads
            )

            q = self.q_proj(hidden_states).view(B, L, num_heads, head_dim)
            k = self.k_proj(hidden_states).view(B, L, num_kv, head_dim)
            v = self.v_proj(hidden_states).view(B, L, num_kv, head_dim)

            # Qwen3 applies q_norm/k_norm on head_dim before RoPE.
            if hasattr(self, "q_norm"):
                q = self.q_norm(q)
            if hasattr(self, "k_norm"):
                k = self.k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            pos_emb = kwargs.get("position_embeddings", None)
            if pos_emb is not None:
                cos, sin = pos_emb
                q, k = _apply_rotary(q, k, cos, sin)

            # Pull the model's attention_mask from kwargs so retrieval can
            # exclude pad key positions. Without this the ANN top-K may
            # include pad keys, giving the model garbage to attend to.
            key_mask = kwargs.get("attention_mask", None)

            with torch.no_grad():
                q_search, k_search = wrapper.search_projection(hidden_states)
                if wrapper.use_faiss:
                    retrieved, retrieved_valid = _faiss_topk_search(
                        q_search,
                        k_search,
                        wrapper.K_retrieve,
                        use_hnsw=wrapper.use_hnsw,
                        hnsw_M=wrapper.hnsw_M,
                        hnsw_ef_construction=wrapper.hnsw_ef_construction,
                        hnsw_ef_search=wrapper.hnsw_ef_search,
                        key_mask=key_mask,
                        return_valid_mask=True,
                    )
                else:
                    retrieved, retrieved_valid = _exact_topk_search(
                        q_search,
                        k_search,
                        wrapper.K_retrieve,
                        key_mask=key_mask,
                        return_valid_mask=True,
                    )

            attn_out = _ann_attention(
                q, k, v, retrieved, retrieved_valid
            )  # [B, H, L, d_head]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
            attn_out = self.o_proj(attn_out)
            return attn_out, None

        attn.forward = types.MethodType(patched_forward, attn)

    def uninstall(self):
        self.attention_module.forward = self.original_forward


class QuestAttentionWrapper:
    """Quest-style min/max page selector over native Q/K for baseline eval."""

    def __init__(
        self,
        attention_module,
        K_retrieve: int,
        page_size: int = 16,
    ):
        self.attention_module = attention_module
        self.K_retrieve = K_retrieve
        self.page_size = page_size
        self.original_forward = attention_module.forward

    def install(self):
        attn = self.attention_module
        wrapper = self

        def patched_forward(self, hidden_states, *args, **kwargs):
            B, L, _ = hidden_states.shape
            num_heads = self.config.num_attention_heads
            num_kv = getattr(self.config, "num_key_value_heads", num_heads)
            head_dim = getattr(
                self.config, "head_dim", self.config.hidden_size // num_heads
            )

            q = self.q_proj(hidden_states).view(B, L, num_heads, head_dim)
            k = self.k_proj(hidden_states).view(B, L, num_kv, head_dim)
            v = self.v_proj(hidden_states).view(B, L, num_kv, head_dim)

            if hasattr(self, "q_norm"):
                q = self.q_norm(q)
            if hasattr(self, "k_norm"):
                k = self.k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            pos_emb = kwargs.get("position_embeddings", None)
            if pos_emb is not None:
                cos, sin = pos_emb
                q, k = _apply_rotary(q, k, cos, sin)

            if num_heads != num_kv:
                repeat = num_heads // num_kv
                k_for_search = k.repeat_interleave(repeat, dim=1)
            else:
                k_for_search = k

            key_mask = kwargs.get("attention_mask", None)
            retrieved, retrieved_valid = _quest_page_search(
                q,
                k_for_search,
                wrapper.K_retrieve,
                page_size=wrapper.page_size,
                key_mask=key_mask,
                return_valid_mask=True,
            )

            attn_out = _ann_attention(q, k, v, retrieved, retrieved_valid)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
            attn_out = self.o_proj(attn_out)
            return attn_out, None

        attn.forward = types.MethodType(patched_forward, attn)

    def uninstall(self):
        self.attention_module.forward = self.original_forward


def install_ann_attention(
    base_model,
    search_module,
    layer_indices: List[int],
    K_retrieve: int,
    use_faiss: bool = False,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 64,
) -> List[ANNAttentionWrapper]:
    """
    Install ANN-substituted attention on every layer in `layer_indices`.
    Returns the list of wrappers so callers can uninstall later.
    """
    wrappers = []
    for idx in layer_indices:
        attn_module = base_model.model.layers[idx].self_attn
        proj = search_module.projections[str(idx)]
        w = ANNAttentionWrapper(
            attn_module,
            proj,
            K_retrieve,
            use_faiss=use_faiss,
            use_hnsw=use_hnsw,
            hnsw_M=hnsw_M,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
        w.install()
        wrappers.append(w)
    return wrappers


def install_quest_attention(
    base_model,
    layer_indices: List[int],
    K_retrieve: int,
    page_size: int = 16,
) -> List[QuestAttentionWrapper]:
    wrappers = []
    for idx in layer_indices:
        attn_module = base_model.model.layers[idx].self_attn
        w = QuestAttentionWrapper(attn_module, K_retrieve, page_size=page_size)
        w.install()
        wrappers.append(w)
    return wrappers


def uninstall_ann_attention(wrappers: List[ANNAttentionWrapper]):
    for w in wrappers:
        w.uninstall()


def run_with_ann_substitution(
    base_model,
    search_module,
    input_ids: torch.Tensor,
    layer_indices: List[int],
    K_retrieve: int,
    output_router_logits: bool = False,
    use_faiss: bool = False,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 64,
):
    """
    Run a forward pass with ANN-substituted attention on the given layers.
    Restores the original attention forwards on exit.
    """
    wrappers = install_ann_attention(
        base_model,
        search_module,
        layer_indices,
        K_retrieve,
        use_faiss=use_faiss,
        use_hnsw=use_hnsw,
        hnsw_M=hnsw_M,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    try:
        with torch.no_grad():
            kwargs = dict(input_ids=input_ids, use_cache=False)
            if output_router_logits:
                kwargs["output_router_logits"] = True
            return base_model(**kwargs)
    finally:
        uninstall_ann_attention(wrappers)
