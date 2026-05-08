"""
Training loop for the search projections.

Applies the perf optimizations from read_and_take_notice_before_code.md:
  - Liger kernels (apply_liger_kernel_to_qwen3_moe) on the frozen base model
  - FA-3 with FA-2 fallback (attn_implementation)
  - QK-reconstruction capture path (default; preserves FA on the trained
    layers, see model.FrozenForwardCapture)
  - torch.compile on the search projections (mode="max-autotune")
  - Optional torch.compile on the base model (off by default; flaky for MoE)
  - Disable gradient checkpointing on the *base* model (not training it)
  - fp32 loss math
  - pin_memory + prefetch in the dataloader (in data.py)

Also includes the MoE defensive coding called out in the spec:
  - explicit router-param freezing verification
  - periodic param-drift check on the base model
  - router top-1 match rate at eval (in eval.py)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from data import build_long_context_dataloader
from eval import evaluate
from model import (
    FrozenForwardCapture,
    SearchProjectionModule,
    total_loss,
)


# =============================================================================
# Helpers
# =============================================================================


def get_layers_to_train(config: Config) -> List[int]:
    """Full-attention layers minus the reserved ones (first and last)."""
    return [
        idx
        for idx in config.full_attention_layer_indices
        if idx not in config.reserved_full_attention_indices
    ]


def setup_wandb(config: Config):
    wandb.init(
        project=config.wandb_project,
        entity=getattr(config, "wandb_entity", None),
        name=config.wandb_run_name,
        config=vars(config),
    )
    wandb.define_metric("eval/recall_at_K_avg", summary="max")
    wandb.define_metric("eval/ppl_gap_relative", summary="min")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("diag/qk_alignment_avg", summary="max")
    if config.verify_routing_stability:
        wandb.define_metric("eval/router_match_rate", summary="max")


def freeze_base_model(base_model, freeze_router: bool = True):
    """
    Freeze all base-model parameters. Explicitly verify router params are
    frozen — the spec calls this out as MoE-specific defensive coding.
    """
    for _, param in base_model.named_parameters():
        param.requires_grad = False

    if freeze_router:
        router_param_names = [
            n
            for n, _ in base_model.named_parameters()
            if any(key in n.lower() for key in ["router", "gate", "expert_proj"])
        ]
        params = dict(base_model.named_parameters())
        for name in router_param_names:
            assert not params[name].requires_grad, f"Router param {name} not frozen!"
        print(
            f"[MoE safety] Verified {len(router_param_names)} router params are frozen."
        )


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def maybe_apply_liger(base_model, config: Config):
    """
    Apply Liger fused kernels. Dispatch order:

      1. Architecture-matched entrypoint (`qwen3` / `qwen3_moe` if MoE).
      2. LLaMA entrypoint — Qwen3 architecture is close enough that
         apply_liger_kernel_to_llama works in practice (RMSNorm, SwiGLU,
         RoPE — all the same shapes). Recommended fallback.
      3. Manual module-level patching of RMSNorm + SwiGLU as a last resort.
    """
    if not config.use_liger_kernels:
        return

    arch = type(base_model).__name__.lower()
    is_moe = "moe" in arch

    try:
        import liger_kernel.transformers as lk
    except ImportError:
        print("[perf] liger_kernel not installed — skipping fused kernels.")
        return

    common_kwargs = dict(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=True,
        swiglu=True,
        model=base_model,
    )

    candidates = (
        ["apply_liger_kernel_to_qwen3_moe", "apply_liger_kernel_to_qwen3",
         "apply_liger_kernel_to_llama"]
        if is_moe
        else ["apply_liger_kernel_to_qwen3", "apply_liger_kernel_to_llama",
              "apply_liger_kernel_to_qwen3_moe"]
    )
    for name in candidates:
        fn = getattr(lk, name, None)
        if fn is None:
            continue
        try:
            fn(**common_kwargs)
            print(f"[perf] Liger kernels applied via {name}.")
            return
        except Exception as e:
            print(f"[perf] {name} failed: {e}")

    # Manual fallback: monkey-patch RMSNorm and SwiGLU MLP modules in place.
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        n_rms = 0
        for module in base_model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                # Copy over weight + eps, swap the module's forward.
                module.forward = LigerRMSNorm.forward.__get__(module, type(module))
                n_rms += 1
        if n_rms > 0:
            print(f"[perf] Manually patched {n_rms} RMSNorm modules with Liger.")
    except Exception as e:
        print(f"[perf] Manual Liger fallback failed: {e}")
        print("[perf] Continuing without Liger kernels.")


def load_base_model(config: Config):
    # Force SDPA into the fast path. If a shape doesn't have a flash or
    # mem-efficient kernel we want to know (loud error) rather than silently
    # falling back to the slow math path.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    if not config.flash_attention:
        candidates = ["eager"]
    elif config.use_flash_attention_3:
        candidates = ["flash_attention_3", "flash_attention_2", "sdpa", "eager"]
    else:
        candidates = ["flash_attention_2", "sdpa", "eager"]

    base_model = None
    for impl in candidates:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                dtype=torch.bfloat16 if config.bf16 else torch.float32,
                device_map="auto",
                attn_implementation=impl,
            )
            print(f"[perf] attention implementation: {impl}")
            break
        except (ValueError, ImportError, RuntimeError, AssertionError) as e:
            print(f"[perf] {impl} unavailable ({type(e).__name__}: {e}); trying next.")
    if base_model is None:
        raise RuntimeError("No attention implementation worked.")

    base_model.eval()
    # We don't backprop through the base model, so checkpointing only hurts speed.
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()

    maybe_apply_liger(base_model, config)
    freeze_base_model(base_model, freeze_router=config.freeze_router)

    return base_model, tokenizer


def maybe_compile(search_module, base_model, config: Config):
    """Compile the search projections (always) and base model (opt-in)."""
    if config.compile_search_module:
        try:
            search_module = torch.compile(search_module, mode="max-autotune")
            print("[perf] search_module compiled (max-autotune).")
        except Exception as e:
            print(f"[perf] search_module compile failed ({e}) — using uncompiled.")

    if config.compile_base_model:
        try:
            compiled = torch.compile(
                base_model, mode="reduce-overhead", dynamic=True, fullgraph=False
            )
            test_input = torch.randint(
                0, 1000, (1, 1024), device=base_model.device
            )
            with torch.no_grad():
                out_uncompiled = base_model(test_input).logits
                out_compiled = compiled(test_input).logits
            if torch.allclose(out_uncompiled, out_compiled, atol=1e-3):
                base_model = compiled
                print("[perf] base model compiled successfully.")
            else:
                print("[perf] compiled base model diverges; using uncompiled.")
        except Exception as e:
            print(f"[perf] base model compile failed ({e}); using uncompiled.")

    return search_module, base_model


def save_checkpoint(search_module, optimizer, scheduler, step: int, config: Config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"search_step_{step}.pt")
    sd = (
        search_module._orig_mod.state_dict()
        if hasattr(search_module, "_orig_mod")
        else search_module.state_dict()
    )
    torch.save(
        {
            "step": step,
            "search_module": sd,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": vars(config),
        },
        path,
    )
    print(f"[ckpt] step {step} -> {path}")

    # Rotate: keep only the last N + the highest-step checkpoint.
    keep = getattr(config, "keep_last_n_checkpoints", 5)
    ckpts = sorted(
        [f for f in os.listdir(config.checkpoint_dir) if f.startswith("search_step_") and f.endswith(".pt")],
        key=lambda n: int(n.removeprefix("search_step_").removesuffix(".pt")),
    )
    for old in ckpts[:-keep]:
        try:
            os.remove(os.path.join(config.checkpoint_dir, old))
        except OSError:
            pass


def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("search_step_") and f.endswith(".pt")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda n: int(n.removeprefix("search_step_").removesuffix(".pt")))
    return os.path.join(checkpoint_dir, ckpts[-1])


def maybe_resume(search_module, optimizer, scheduler, config: Config) -> int:
    """If auto_resume and a checkpoint exists, load it and return its step."""
    if not getattr(config, "auto_resume", False):
        return 0
    path = find_latest_checkpoint(config.checkpoint_dir)
    if path is None:
        return 0
    print(f"[ckpt] resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    target = (
        search_module._orig_mod
        if hasattr(search_module, "_orig_mod")
        else search_module
    )
    target.load_state_dict(ckpt["search_module"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])


def snapshot_base_params(base_model) -> Dict[str, torch.Tensor]:
    """
    Take a snapshot of all base parameters for the periodic drift check.
    We snapshot a small fixed subset on CPU to keep memory in check.
    """
    snap = {}
    for i, (name, param) in enumerate(base_model.named_parameters()):
        if i % 64 != 0:
            continue
        snap[name] = param.detach().cpu().clone()
    return snap


def assert_no_param_drift(base_model, snapshot: Dict[str, torch.Tensor]):
    for name, original in snapshot.items():
        param = dict(base_model.named_parameters())[name]
        if not torch.allclose(param.detach().cpu(), original, atol=1e-6):
            raise RuntimeError(f"Base model param {name} drifted during training!")


# =============================================================================
# Diagnostic printer (matches the dashboard rules in project.md)
# =============================================================================


def print_actionable_diagnostic(eval_results: Dict, step: int, config: Config):
    """
    PPL gap is the primary signal — it tells us whether the model's output
    is preserved under ANN substitution. Recall is a secondary shape metric
    (high recall = both set and ranking right; low recall + low PPL = the
    search found the keys that actually carry weight and disagrees only on
    the near-zero tail, which is fine).
    """
    recall = eval_results.get("eval/recall_at_K_avg", 0.0)
    ppl_gap = eval_results.get("eval/ppl_gap_relative", float("inf"))
    print(f"\n[step {step}] === ACTIONABLE DIAGNOSTIC ===")
    print(f"  Recall@K_eval: {recall:.3f}")
    print(f"  PPL gap (relative): {ppl_gap:.3%}")

    if ppl_gap > 0.05:
        if recall < 0.5:
            print("  >> SIGNAL: Both quality (PPL) and set recall are poor.")
            print("  >> ACTION: Increase d_search (try 128), train longer, or "
                  "switch to MLP projection.")
        else:
            print("  >> SIGNAL: Set recall OK but ranking wrong — softmax weight "
                  "is misallocated within the retrieved set.")
            print(
                f"  >> ACTION: Increase beta_distillation "
                f"(currently {config.beta_distillation}). "
                f"Try: {config.beta_distillation * 2}"
            )
    else:
        # ppl_gap <= 5% — technique is working
        if recall < 0.5:
            print("  >> WORKING: Low set recall but quality preserved — the search "
                  "finds the keys that actually carry softmax weight; "
                  "disagreement is on the near-zero tail (expected).")
        else:
            print("  >> WORKING: High recall and quality preserved. Both set and "
                  "ranking aligned with the teacher.")
    print()


# =============================================================================
# Training loop
# =============================================================================


def train(config: Config):
    setup_wandb(config)
    layers_to_train = get_layers_to_train(config)
    print(f"Training search projections for layers: {layers_to_train}")
    print(f"Reserved as full attention: {config.reserved_full_attention_indices}")

    base_model, tokenizer = load_base_model(config)

    # Sanity: every layer-to-train must actually exist in the model and have
    # the expected attribute. Catch a misconfigured `full_attention_layer_indices`
    # before wasting compute.
    for idx in layers_to_train:
        layer = base_model.model.layers[idx]
        assert hasattr(layer, "self_attn"), (
            f"Layer {idx} has no self_attn — likely a DeltaNet layer; remove it "
            f"from full_attention_layer_indices."
        )

    d_model = base_model.config.hidden_size
    if d_model != config.d_model:
        print(
            f"[config] config.d_model={config.d_model} but model has hidden_size="
            f"{d_model}; using model's value."
        )

    search_module = SearchProjectionModule(
        d_model=d_model,
        d_search=config.d_search,
        layer_indices=layers_to_train,
        use_mlp=config.use_mlp_proj,
        dropout=config.proj_dropout,
    ).to(base_model.device).to(torch.bfloat16 if config.bf16 else torch.float32)

    n_train_params = sum(p.numel() for p in search_module.parameters())
    print(
        f"Trainable parameters: {n_train_params:,} ({n_train_params / 1e6:.2f}M)"
    )
    wandb.run.summary["trainable_params_M"] = n_train_params / 1e6

    # Log resolved model facts as they actually loaded — not the config defaults.
    # This makes the W&B run self-describing if anything mismatched expectations.
    cfg_obj = base_model.config
    wandb.config.update(
        {
            "resolved/model_class": type(base_model).__name__,
            "resolved/num_hidden_layers": cfg_obj.num_hidden_layers,
            "resolved/hidden_size": cfg_obj.hidden_size,
            "resolved/num_attention_heads": cfg_obj.num_attention_heads,
            "resolved/num_key_value_heads": getattr(
                cfg_obj, "num_key_value_heads", cfg_obj.num_attention_heads
            ),
            "resolved/head_dim": getattr(
                cfg_obj, "head_dim", cfg_obj.hidden_size // cfg_obj.num_attention_heads
            ),
            "resolved/max_position_embeddings": cfg_obj.max_position_embeddings,
            "resolved/torch_dtype": str(base_model.dtype),
            "resolved/attn_implementation": getattr(
                cfg_obj, "_attn_implementation", "unknown"
            ),
            "resolved/layers_to_train": layers_to_train,
            "resolved/has_layer_types": hasattr(cfg_obj, "layer_types"),
        },
        allow_val_change=True,
    )

    capture = FrozenForwardCapture(
        base_model,
        layers_to_train,
        qk_reconstruction=config.qk_reconstruction,
    )

    optimizer = AdamW(
        search_module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_schedule(s, config.warmup_steps, config.total_steps),
    )

    search_module, base_model = maybe_compile(search_module, base_model, config)

    train_loader = build_long_context_dataloader(
        tokenizer,
        config.train_dataset,
        config.seq_len,
        config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pack=config.sequence_packing,
        dataset_config=getattr(config, "train_dataset_config", None),
    )

    base_param_snapshot = snapshot_base_params(base_model)

    step = maybe_resume(search_module, optimizer, scheduler, config)
    if step > 0:
        print(f"[ckpt] resumed at step {step}/{config.total_steps}")
    grad_accum_count = 0
    optimizer.zero_grad()

    while step < config.total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(base_model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(base_model.device)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(base_model.device)

            try:
                hidden_states_dict, attn_weights_dict = capture.run(
                    input_ids, attention_mask, position_ids=position_ids
                )
            except Exception as e:
                print(f"[step {step}] Forward capture failed: {e}")
                wandb.log({"errors/forward_capture": 1}, step=step)
                continue

            missing = set(layers_to_train) - set(hidden_states_dict.keys())
            if missing:
                print(f"[step {step}] Missing hidden states for layers: {missing}")
                wandb.log({"errors/missing_layers": len(missing)}, step=step)
                continue
            missing_attn = set(layers_to_train) - set(attn_weights_dict.keys())
            if missing_attn:
                print(f"[step {step}] Missing attn weights for layers: {missing_attn}")
                wandb.log({"errors/missing_attn": len(missing_attn)}, step=step)
                continue

            q_dict, k_dict = search_module(hidden_states_dict)
            loss, log_dict = total_loss(
                q_dict, k_dict, attn_weights_dict, config,
                attention_mask=attention_mask,
            )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            grad_accum_count += 1

            if grad_accum_count >= config.gradient_accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    search_module.parameters(), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_accum_count = 0
                step += 1

                if step % config.log_every == 0:
                    qk_align_avg = sum(
                        d["qk_mean_cosine"]
                        for d in log_dict["diag/per_layer"].values()
                    ) / len(log_dict["diag/per_layer"])

                    wandb.log(
                        {
                            **{k: v for k, v in log_dict.items() if k.startswith("loss/")},
                            "diag/qk_alignment_avg": qk_align_avg,
                            "train/grad_norm": float(grad_norm),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/step": step,
                        },
                        step=step,
                    )

                if step % config.eval_every == 0:
                    # Periodic safety check: base model must not have drifted.
                    assert_no_param_drift(base_model, base_param_snapshot)

                    eval_results = evaluate(
                        base_model, search_module, capture, config, tokenizer
                    )
                    wandb.log(eval_results, step=step)
                    print_actionable_diagnostic(eval_results, step, config)

                if step % config.save_every == 0:
                    save_checkpoint(search_module, optimizer, scheduler, step, config)

                if step >= config.total_steps:
                    save_checkpoint(search_module, optimizer, scheduler, step, config)
                    break


if __name__ == "__main__":
    import argparse

    from config import (
        make_headline_config,
        make_headline_d128_config,
        make_pilot_d64_clean_config,
        make_pilot_d64_packed_config,
        make_pilot_d128_config,
        make_pilot_d128_packed_config,
        make_pilot_d256_config,
        make_pilot_d256_packed_config,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="pilot",
        choices=[
            "pilot",
            "pilot_d64_clean",
            "pilot_d64_packed",
            "pilot_d128",
            "pilot_d128_packed",
            "pilot_d256",
            "pilot_d256_packed",
            "headline_d64",
            "headline_d128",
        ],
        help="Which preset config to use.",
    )
    args = parser.parse_args()

    if args.config == "pilot":
        cfg = Config()
    elif args.config == "pilot_d64_clean":
        cfg = make_pilot_d64_clean_config()
    elif args.config == "pilot_d64_packed":
        cfg = make_pilot_d64_packed_config()
    elif args.config == "pilot_d128":
        cfg = make_pilot_d128_config()
    elif args.config == "pilot_d128_packed":
        cfg = make_pilot_d128_packed_config()
    elif args.config == "pilot_d256":
        cfg = make_pilot_d256_config()
    elif args.config == "pilot_d256_packed":
        cfg = make_pilot_d256_packed_config()
    elif args.config == "headline_d64":
        cfg = make_headline_config()
    elif args.config == "headline_d128":
        cfg = make_headline_d128_config()

    print(f"[config] preset={args.config} run={cfg.wandb_run_name} "
          f"steps={cfg.total_steps} layers={len(get_layers_to_train(cfg))} "
          f"d_search={cfg.d_search} ckpt_dir={cfg.checkpoint_dir}")
    train(cfg)
