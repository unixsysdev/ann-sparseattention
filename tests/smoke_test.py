"""
Pre-pilot smoke test. Run this before kicking off the multi-hour training to
catch the silent-failure modes that would waste compute:

  1. Hooks fire on every targeted layer (no missing entries).
  2. Captured attention is non-trivial (not uniform — would indicate FA is
     silently being used despite the qk_reconstruction path, or the wrong
     tensor was captured).
  3. Search projections produce gradients on a single training step.
  4. Loss is finite (no NaN / inf).

Also runs ~50 training steps so we see a loss curve before committing.

Usage:
    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import (  # noqa: E402
    FrozenForwardCapture,
    SearchProjectionModule,
    aggregate_heads,
    total_loss,
)


def main():
    cfg = Config()
    # Tighten things for a quick smoke test.
    cfg.seq_len = 1024
    cfg.batch_size = 1
    cfg.total_steps = 50
    cfg.warmup_steps = 5

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {cfg.base_model_name} ...")
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

    print(
        f"Model loaded: {type(base_model).__name__}  "
        f"layers={base_model.config.num_hidden_layers}  "
        f"hidden={base_model.config.hidden_size}"
    )

    layers_to_train = [
        i for i in cfg.full_attention_layer_indices
        if i not in cfg.reserved_full_attention_indices
    ]
    print(f"Training layers: {layers_to_train}")

    search = SearchProjectionModule(
        d_model=base_model.config.hidden_size,
        d_search=cfg.d_search,
        layer_indices=layers_to_train,
        use_mlp=cfg.use_mlp_proj,
    ).to(base_model.device).to(torch.bfloat16)

    capture = FrozenForwardCapture(
        base_model, layers_to_train, qk_reconstruction=cfg.qk_reconstruction
    )

    sample = tokenizer(
        "The quick brown fox jumps over the lazy dog. " * 64,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.seq_len,
    )
    input_ids = sample["input_ids"].to(base_model.device)

    # ---- 1. Hook capture sanity ----
    h_dict, w_dict = capture.run(input_ids)
    missing = set(layers_to_train) - set(h_dict.keys())
    assert not missing, f"Missing hidden states for layers: {missing}"
    missing_w = set(layers_to_train) - set(w_dict.keys())
    assert not missing_w, f"Missing attn weights for layers: {missing_w}"
    print(f"  hooks fired on all {len(layers_to_train)} layers")

    # ---- 2. Attention weights non-trivial ----
    for idx, w in w_dict.items():
        # If FA silently kicked in we'd often see weights == None or NaNs.
        assert torch.isfinite(w).all(), f"Layer {idx}: non-finite attn weights"
        # Uniform attention → max prob ~ 1/L. Real attention should have
        # peaks. With L=1024, 1/L ≈ 1e-3; demand max > 0.01 to be safe.
        max_attn = w.max().item()
        assert max_attn > 0.01, (
            f"Layer {idx} attention is suspiciously uniform: max={max_attn:.4f}. "
            f"This usually means FA was silently used and weights weren't "
            f"materialized. Verify qk_reconstruction path."
        )
    print("  attn weights are non-trivial")

    # ---- 3. Forward + backward through search projections ----
    optimizer = torch.optim.AdamW(search.parameters(), lr=cfg.learning_rate)
    losses = []
    for step in range(cfg.total_steps):
        h_dict, w_dict = capture.run(input_ids)
        q_dict, k_dict = search(h_dict)
        loss, log = total_loss(q_dict, k_dict, w_dict, cfg)
        assert torch.isfinite(loss).item(), f"Step {step}: non-finite loss"
        loss.backward()
        # Verify gradients exist on every search projection.
        if step == 0:
            for name, p in search.named_parameters():
                assert p.grad is not None, f"No gradient on {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient on {name}"
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 10 == 0:
            print(
                f"  step {step:3d}  loss={loss.item():.4f}  "
                f"contrastive={log['loss/contrastive']:.4f}  "
                f"distill={log['loss/distillation']:.4f}"
            )

    # ---- 4. Loss should be descending (rough sanity) ----
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    print(f"  early-mean={early:.4f}  late-mean={late:.4f}  Δ={late - early:+.4f}")
    if late >= early:
        print(
            "  WARN: loss not descending in 50 steps. Common causes: "
            "lr too low, projections frozen, teacher attention is uniform."
        )
    else:
        print("  loss is descending")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
