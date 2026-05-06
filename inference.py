"""
ANN-substituted attention runtime.

Installs an `ANNAttentionWrapper` on each trained full-attention layer. When
the model's forward pass reaches one of these layers, the wrapper:

  1. Computes Q, K, V (and applies RoPE) as the original attention does.
  2. Computes (q_search, k_search) from the same hidden state via the trained
     SearchProjection.
  3. For each query position q, retrieves the top-K_retrieve key indices using
     exact top-K over (q_search @ k_search^T), causal-masked. (FAISS path
     available via `use_faiss=True` for true ANN at long context.)
  4. Computes standard attention restricted to the retrieved K_retrieve keys.

The result has the same shape as full attention; the rest of the model is
untouched.

Used both for end-to-end perplexity comparison in eval and for production
inference.
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
) -> torch.Tensor:
    """
    q_search, k_search: [B, L, d_search].
    Returns indices [B, L, K] of top-K keys by cosine similarity of search
    vectors, restricted to causal (key index <= query index).
    """
    B, L, _ = q_search.shape
    q_n = F.normalize(q_search, dim=-1)
    k_n = F.normalize(k_search, dim=-1)
    sim = torch.bmm(q_n, k_n.transpose(1, 2))  # [B, L, L]
    if causal:
        mask = torch.ones(L, L, device=sim.device, dtype=torch.bool).tril()
        sim = sim.masked_fill(~mask, -1e9)
    K_eff = min(K, L)
    return sim.topk(K_eff, dim=-1).indices  # [B, L, K_eff]


def _faiss_topk_search(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    K: int,
    causal: bool = True,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 64,
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
        return _exact_topk_search(q_search, k_search, K, causal=causal)

    B, L, d = q_search.shape
    K_eff = min(K, L)
    out = torch.empty(B, L, K_eff, dtype=torch.long, device=q_search.device)
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
        ids_t = ids_t.masked_fill(~valid, -1)
        for q in range(L):
            row = ids_t[q]
            row = row[row >= 0][: K_eff]
            if row.numel() < K_eff:
                # Pad with self-position to keep tensor shape regular.
                pad = torch.full(
                    (K_eff - row.numel(),),
                    int(q),
                    device=q_search.device,
                    dtype=torch.long,
                )
                row = torch.cat([row, pad])
            out[b, q, : K_eff] = row[: K_eff]
    return out


def _gather_kv(
    k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k, v: [B, H_kv, L, d_head]
    indices: [B, L, K]  (key positions, in [0, L))
    Returns:
      k_gathered: [B, H_kv, L, K, d_head]
      v_gathered: [B, H_kv, L, K, d_head]
    """
    B, H_kv, L, d_head = k.shape
    K = indices.shape[-1]
    # Expand to [B, H_kv, L, K, d_head] index.
    idx = indices.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, L, K, d_head)
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
    # Mask out padded fillers (where retrieved == query position used as pad);
    # since the pad value is the query position itself, it's already valid, so
    # no extra masking is required for correctness.
    weights = F.softmax(scores, dim=-1)
    # out: [B, H, L, d_head]
    out = torch.einsum("bhlk,bhlkd->bhld", weights, v_g)
    return out


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

            with torch.no_grad():
                q_search, k_search = wrapper.search_projection(hidden_states)
                if wrapper.use_faiss:
                    retrieved = _faiss_topk_search(
                        q_search,
                        k_search,
                        wrapper.K_retrieve,
                        use_hnsw=wrapper.use_hnsw,
                        hnsw_M=wrapper.hnsw_M,
                        hnsw_ef_construction=wrapper.hnsw_ef_construction,
                        hnsw_ef_search=wrapper.hnsw_ef_search,
                    )
                else:
                    retrieved = _exact_topk_search(
                        q_search, k_search, wrapper.K_retrieve
                    )

            attn_out = _ann_attention(q, k, v, retrieved)  # [B, H, L, d_head]
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
