"""
Search projection module + frozen forward capture + loss functions.

The frozen forward capture supports two modes:
  - output_attentions=True (Option 1 from the perf doc): simple, but disables
    FlashAttention on the trained layers.
  - QK reconstruction (Option 3, recommended): monkey-patches the target
    attention modules to capture (Q, K) post-RoPE; teacher attention weights
    are reconstructed as softmax(QK^T / sqrt(d_head)) outside the FA path.
    This keeps FlashAttention enabled end-to-end.

The 5-line verification test (perf doc) lives in tests/test_qk_reconstruction.py.
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SearchProjection
# =============================================================================


class SearchProjection(nn.Module):
    """
    Per-layer search projection. Linear by default; MLP optional.
    Computes q_search and k_search from hidden states.
    """

    def __init__(
        self,
        d_model: int,
        d_search: int,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_search = d_search

        if use_mlp:
            d_hidden = 2 * d_search
            self.W_Qs = nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_search, bias=False),
            )
            self.W_Ks = nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_search, bias=False),
            )
        else:
            self.W_Qs = nn.Linear(d_model, d_search, bias=False)
            self.W_Ks = nn.Linear(d_model, d_search, bias=False)

        # Small variance so search starts close to noise but with nonzero
        # gradient signal.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: [B, L, d_model] -> q, k each [B, L, d_search]."""
        return self.W_Qs(hidden_states), self.W_Ks(hidden_states)


class SearchProjectionModule(nn.Module):
    """
    Container holding all per-layer search projections.
    Indexed by absolute layer index in the base model (not 0..N_proj-1).
    """

    def __init__(
        self,
        d_model: int,
        d_search: int,
        layer_indices: List[int],
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.projections = nn.ModuleDict(
            {
                str(idx): SearchProjection(d_model, d_search, use_mlp, dropout)
                for idx in self.layer_indices
            }
        )

    def forward(
        self, hidden_states_dict: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        hidden_states_dict: {layer_idx: hidden_state_tensor [B, L, d_model]}
        Returns: (q_search_dict, k_search_dict), keyed by layer_idx.
        """
        q_dict: Dict[int, torch.Tensor] = {}
        k_dict: Dict[int, torch.Tensor] = {}
        for idx in self.layer_indices:
            h = hidden_states_dict[idx]
            q, k = self.projections[str(idx)](h)
            q_dict[idx] = q
            k_dict[idx] = k
        return q_dict, k_dict


# =============================================================================
# Frozen forward capture
# =============================================================================


class FrozenForwardCapture:
    """
    Runs a frozen forward pass on the base model and captures, for each target
    layer:
      - hidden state going INTO that layer's self-attention (post input-LN)
      - teacher attention weights [B, H, L, L]

    Two paths:

      qk_reconstruction=False (Option 1)
        Calls the model with output_attentions=True. HF falls back to eager
        attention on those layers but the rest of the model keeps FA. We read
        the attention weights tuple from the model output.

      qk_reconstruction=True  (Option 3, recommended by perf doc)
        Monkey-patches each target self_attn.forward to additionally capture
        (Q, K) post-RoPE into a side buffer. The forward pass runs with
        FlashAttention end-to-end. Teacher weights are reconstructed as
        softmax(QK^T / sqrt(d_head)) outside the FA path (and only for the
        ~8 trained layers, an O(L^2) op done once per forward).

    MoE-specific: also captures router decisions if `capture_router=True`.
    """

    def __init__(
        self,
        base_model,
        target_layer_indices: List[int],
        qk_reconstruction: bool = True,
        capture_router: bool = False,
    ):
        self.base_model = base_model
        self.target_indices = list(target_layer_indices)
        self.qk_reconstruction = qk_reconstruction
        self.capture_router = capture_router

        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.attn_weights: Dict[int, torch.Tensor] = {}
        self.router_top1: Dict[int, torch.Tensor] = {}
        self._captured_qk: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self._hooks: List = []
        self._patched_attn: List[Tuple[object, callable]] = []  # (module, original_forward)

    # ---- hidden state hook (input to self_attn after input layer norm) ----

    def _install_hidden_state_hooks(self):
        for idx in self.target_indices:
            layer = self.base_model.model.layers[idx]
            attn_module = layer.self_attn

            def pre_hook(module, args, kwargs, _layer_idx=idx):
                # Qwen3MoeAttention.forward signature is (hidden_states, ...).
                # `hidden_states` is positional or in kwargs. It is the
                # post-input-layernorm tensor we want.
                if "hidden_states" in kwargs:
                    h = kwargs["hidden_states"]
                else:
                    h = args[0]
                self.hidden_states[_layer_idx] = h.detach()

            self._hooks.append(
                attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

    # ---- Option 1: read attention weights from forward output ----

    def _install_attn_weight_hooks(self):
        for idx in self.target_indices:
            attn_module = self.base_model.model.layers[idx].self_attn

            def hook(module, args, kwargs, output, _layer_idx=idx):
                # HF attention modules return (attn_output, attn_weights, ...)
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    self.attn_weights[_layer_idx] = output[1].detach()

            self._hooks.append(
                attn_module.register_forward_hook(hook, with_kwargs=True)
            )

    # ---- Option 3: monkey-patch each target self_attn to capture (Q, K) ----

    def _install_qk_capture_patches(self):
        """
        Wrap target attention modules so that during their forward we capture
        the post-RoPE Q and K. We don't replace the attention math — we just
        sniff Q and K via tensor hooks on q_proj/k_proj and recover the
        post-RoPE versions by replicating the model's RoPE call inline.

        Implementation: register forward hooks on q_proj/k_proj, then on the
        attention module itself, recompute RoPE from position_embeddings stored
        in the attention module call.
        """
        for idx in self.target_indices:
            attn_module = self.base_model.model.layers[idx].self_attn
            original_forward = attn_module.forward

            def make_patched_forward(_attn, _idx, _orig):
                def patched_forward(self, hidden_states, *args, **kwargs):
                    # Save inputs needed to recompute Q, K post-RoPE.
                    pos_emb = kwargs.get("position_embeddings", None)

                    # Run the original forward; afterwards, replicate the model's
                    # exact Q/K pipeline: proj -> view -> q_norm/k_norm (Qwen3
                    # specifics; RMSNorm on head_dim) -> transpose -> RoPE.
                    out = _orig(hidden_states, *args, **kwargs)

                    with torch.no_grad():
                        B, L, _ = hidden_states.shape
                        num_heads = self.config.num_attention_heads
                        num_kv_heads = getattr(
                            self.config, "num_key_value_heads", num_heads
                        )
                        head_dim = getattr(self.config, "head_dim", None)
                        if head_dim is None:
                            head_dim = self.config.hidden_size // num_heads

                        q = self.q_proj(hidden_states).view(B, L, num_heads, head_dim)
                        k = self.k_proj(hidden_states).view(B, L, num_kv_heads, head_dim)

                        # Qwen3 applies q_norm/k_norm on head_dim BEFORE RoPE.
                        if hasattr(self, "q_norm"):
                            q = self.q_norm(q)
                        if hasattr(self, "k_norm"):
                            k = self.k_norm(k)

                        q = q.transpose(1, 2)  # [B, H, L, d_head]
                        k = k.transpose(1, 2)

                        if pos_emb is not None:
                            cos, sin = pos_emb
                            q, k = _apply_rotary(q, k, cos, sin)

                    self._capture_buf[_idx] = (q.detach(), k.detach())
                    return out

                return patched_forward

            # Bind a per-module reference so the closure can stash captures.
            attn_module._capture_buf = self._captured_qk
            attn_module.forward = types.MethodType(
                make_patched_forward(attn_module, idx, original_forward), attn_module
            )
            self._patched_attn.append((attn_module, original_forward))

    def _remove_qk_capture_patches(self):
        for module, original_forward in self._patched_attn:
            module.forward = original_forward
            if hasattr(module, "_capture_buf"):
                delattr(module, "_capture_buf")
        self._patched_attn = []

    # ---- main entry ----

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Run the frozen forward, return:
          hidden_states_dict: {layer_idx: [B, L, d_model]}
          attn_weights_dict:  {layer_idx: [B, H, L, L]}
        """
        self.hidden_states.clear()
        self.attn_weights.clear()
        self.router_top1.clear()
        self._captured_qk.clear()

        self._install_hidden_state_hooks()
        if self.qk_reconstruction:
            self._install_qk_capture_patches()
        else:
            self._install_attn_weight_hooks()

        try:
            with torch.no_grad():
                kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                if position_ids is not None:
                    kwargs["position_ids"] = position_ids
                if not self.qk_reconstruction:
                    kwargs["output_attentions"] = True
                if self.capture_router:
                    kwargs["output_router_logits"] = True

                outputs = self.base_model(**kwargs)

                if self.capture_router and getattr(outputs, "router_logits", None):
                    # router_logits is a tuple of per-MoE-layer logits.
                    # We can't always map MoE layer index -> absolute layer
                    # index without inspecting the model, so we record the
                    # full tuple and let the caller handle it.
                    self.router_top1["__all__"] = tuple(
                        rl.detach().argmax(-1) if rl is not None else None
                        for rl in outputs.router_logits
                    )
        finally:
            self._remove_hooks()
            if self.qk_reconstruction:
                self._remove_qk_capture_patches()

        # Reconstruct attention weights for Option 3.
        if self.qk_reconstruction:
            for idx, (q, k) in self._captured_qk.items():
                self.attn_weights[idx] = _reconstruct_attn_weights(q, k)

        return self.hidden_states, self.attn_weights


# =============================================================================
# RoPE + reconstruction helpers
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding. Matches transformers' Qwen3 RoPE.
    cos/sin shapes are typically [B, L, head_dim] (or [1, L, head_dim]).
    """
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [B, 1, L, head_dim]
        sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _reconstruct_attn_weights(
    q: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """
    q: [B, H_q, L, d_head], k: [B, H_kv, L, d_head] (GQA: H_kv may divide H_q).
    Returns full attention weights [B, H_q, L, L] = softmax(QK^T / sqrt(d_head),
    causal-masked).
    """
    B, H_q, L, d_head = q.shape
    H_kv = k.shape[1]
    if H_q != H_kv:
        # Repeat KV heads to match query heads (GQA).
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)

    # fp32 for numerical stability of softmax
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) / math.sqrt(d_head)
    causal = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    return F.softmax(scores, dim=-1).to(q.dtype)


# =============================================================================
# Loss functions
# =============================================================================


def aggregate_heads(attn_weights: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """attn_weights: [B, H, L, L] -> [B, L, L]."""
    if mode == "max":
        return attn_weights.max(dim=1).values
    if mode == "mean":
        return attn_weights.mean(dim=1)
    raise ValueError(f"Unknown aggregation mode: {mode}")


def contrastive_loss_layer(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    teacher_attn: torch.Tensor,
    K_pos: int = 16,
    tau: float = 0.07,
    fp32: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    InfoNCE with teacher-derived positives.

    q_search, k_search: [B, L, d_search]
    teacher_attn: [B, L, L] aggregated across heads
    """
    if fp32:
        q_search = q_search.float()
        k_search = k_search.float()
        teacher_attn = teacher_attn.float()

    B, L, _ = q_search.shape
    device = q_search.device

    q_norm = F.normalize(q_search, dim=-1)
    k_norm = F.normalize(k_search, dim=-1)

    sim_search = torch.bmm(q_norm, k_norm.transpose(1, 2)) / tau  # [B, L, L]

    causal = torch.ones(L, L, device=device, dtype=torch.bool).tril()
    sim_masked = sim_search.masked_fill(~causal, -1e9)
    teacher_masked = teacher_attn.masked_fill(~causal, -1e9)

    K_eff = min(K_pos, L)
    pos_indices = teacher_masked.topk(K_eff, dim=-1).indices  # [B, L, K_eff]

    # A position is a valid positive if it lies in the causal window for q.
    q_positions = torch.arange(L, device=device).view(1, L, 1)
    valid_pos_mask = pos_indices <= q_positions  # [B, L, K_eff]

    pos_scores = sim_masked.gather(-1, pos_indices)  # [B, L, K_eff]
    pos_scores = pos_scores.masked_fill(~valid_pos_mask, -1e9)

    log_num = torch.logsumexp(pos_scores, dim=-1)   # [B, L]
    log_denom = torch.logsumexp(sim_masked, dim=-1)  # [B, L]

    # Skip queries that have no valid context (position 0 has only itself; we
    # don't get useful contrastive signal there).
    query_valid = torch.arange(L, device=device).unsqueeze(0).expand(B, L) > 0

    loss_per_token = -(log_num - log_denom)
    loss = loss_per_token.masked_select(query_valid).mean()

    with torch.no_grad():
        q_mean = q_norm.mean(dim=(0, 1))
        k_mean = k_norm.mean(dim=(0, 1))
        qk_mean_cos = F.cosine_similarity(
            q_mean.unsqueeze(0), k_mean.unsqueeze(0)
        ).item()

    diagnostics = {"qk_mean_cosine": qk_mean_cos}
    return loss, diagnostics


def distillation_loss_layer(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    teacher_attn: torch.Tensor,
    tau: float = 1.0,
    fp32: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    KL(teacher || student) over attention distributions.
    """
    if fp32:
        q_search = q_search.float()
        k_search = k_search.float()
        teacher_attn = teacher_attn.float()

    B, L, d_s = q_search.shape
    device = q_search.device

    sim = torch.bmm(q_search, k_search.transpose(1, 2)) / math.sqrt(d_s)
    causal = torch.ones(L, L, device=device, dtype=torch.bool).tril()
    sim_masked = sim.masked_fill(~causal, -1e9)
    teacher_masked_zero = teacher_attn.masked_fill(~causal, 0.0)

    teacher_dist = teacher_masked_zero / (
        teacher_masked_zero.sum(-1, keepdim=True) + 1e-9
    )
    student_log_dist = F.log_softmax(sim_masked / tau, dim=-1)

    eps = 1e-9
    teacher_log = torch.log(teacher_dist + eps)
    kl_per_token = (teacher_dist * (teacher_log - student_log_dist)).sum(-1)

    query_valid = torch.arange(L, device=device).unsqueeze(0).expand(B, L) > 0
    loss = kl_per_token.masked_select(query_valid).mean()

    return loss, {}


def total_loss(
    q_search_dict: Dict[int, torch.Tensor],
    k_search_dict: Dict[int, torch.Tensor],
    teacher_attn_dict: Dict[int, torch.Tensor],
    config,
) -> Tuple[torch.Tensor, Dict]:
    """Sum losses across layers, return total + per-layer diagnostics."""
    layer_losses = {"contrastive": [], "distillation": [], "diag": {}}

    for layer_idx in q_search_dict:
        teacher = aggregate_heads(
            teacher_attn_dict[layer_idx], mode=config.teacher_head_aggregation
        )

        L_cont, diag_cont = contrastive_loss_layer(
            q_search_dict[layer_idx],
            k_search_dict[layer_idx],
            teacher,
            K_pos=config.K_pos,
            tau=config.tau_contrastive,
            fp32=config.fp32_loss_math,
        )
        L_distill, _ = distillation_loss_layer(
            q_search_dict[layer_idx],
            k_search_dict[layer_idx],
            teacher,
            tau=config.tau_distillation,
            fp32=config.fp32_loss_math,
        )

        layer_losses["contrastive"].append(L_cont)
        layer_losses["distillation"].append(L_distill)
        layer_losses["diag"][layer_idx] = diag_cont

    L_cont_total = torch.stack(layer_losses["contrastive"]).mean()
    L_distill_total = torch.stack(layer_losses["distillation"]).mean()

    L_total = (
        config.alpha_contrastive * L_cont_total
        + config.beta_distillation * L_distill_total
    )

    return L_total, {
        "loss/total": L_total.item(),
        "loss/contrastive": L_cont_total.item(),
        "loss/distillation": L_distill_total.item(),
        "diag/per_layer": layer_losses["diag"],
    }
