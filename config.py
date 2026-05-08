"""
Configuration for ANN-attention distillation training.

Default: 1-day pilot run on Qwen3-4B-Instruct-2507 — the dense 36-layer
Instruct model (Qwen3ForCausalLM, hidden_size=2560, 32 Q heads / 8 KV heads
GQA, head_dim=128, full RoPE, 262K native context).

Note on naming: "Qwen3.5-4B" is a *hybrid* (Gated-DeltaNet + full attention,
every 4th layer full) multimodal model; not what we want for the "pure full
attention, no hybrid" pilot. Use `Qwen/Qwen3-4B-Instruct-2507`.

Headline run: see make_headline_config() at the bottom of this file.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Model
    base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Qwen3-4B-Instruct-2507: 36 layers, ALL standard self-attention (no
    # hybrid). Pilot trains projections on every 4th layer, 6 layers total.
    # `reserved_full_attention_indices` is a leftover of the hybrid-MoE plan;
    # for a dense model just leave it empty.
    full_attention_layer_indices: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24]
    )
    reserved_full_attention_indices: List[int] = field(default_factory=list)

    # Search projection
    d_model: int = 2560  # Qwen3-4B hidden_size; auto-detected at load time
    d_search: int = 64
    use_mlp_proj: bool = False  # linear only for the 1-day pilot
    proj_dropout: float = 0.0

    # Training (pilot defaults)
    seq_len: int = 4096
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    total_steps: int = 1000        # original pilot plateau'd at ~step 1000;
                                   # 2K saw recall 50.7% -> 50.9%, basically tied
    warmup_steps: int = 100
    eval_every: int = 250          # 4 eval points (250/500/750/1000) — enough
                                   # resolution to see convergence shape across
                                   # d_search values
    save_every: int = 200
    keep_last_n_checkpoints: int = 5

    # Loss weights (α=β=1 for the pilot)
    alpha_contrastive: float = 1.0
    beta_distillation: float = 1.0

    # Loss-specific
    K_pos: int = 16
    tau_contrastive: float = 0.07
    tau_distillation: float = 1.0
    teacher_head_aggregation: str = "max"

    # Inference / eval
    K_retrieve_eval: int = 128  # primary K reported in PPL gap
    K_retrieve_search: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512]  # recall curve at eval
    )
    use_faiss_hnsw_at_eval: bool = True  # the "off-the-shelf FAISS works" demo
    faiss_hnsw_M: int = 32              # default FAISS HNSW params
    faiss_hnsw_ef_construction: int = 40
    faiss_hnsw_ef_search: int = 64

    # MoE-specific safety (off for dense Qwen3 4B)
    freeze_router: bool = False
    verify_routing_stability: bool = False

    # Data
    train_dataset: str = "Salesforce/wikitext"
    train_dataset_config: Optional[str] = "wikitext-103-raw-v1"
    eval_long_context: str = "longbench"

    # Logging
    wandb_project: str = "ann-sparse"
    wandb_entity: Optional[str] = "dalletest123"
    wandb_run_name: Optional[str] = None
    log_every: int = 50

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True  # see train.py — disabled on the *base* model
    flash_attention: bool = True
    use_flash_attention_3: bool = True  # FA-3 on H100/H200/B200; falls back to FA-2
    use_liger_kernels: bool = True
    compile_search_module: bool = True
    compile_base_model: bool = False  # opt-in; OK for dense too but not core to pilot
    qk_reconstruction: bool = True
    fp32_loss_math: bool = True
    sequence_packing: bool = False  # off by default; packed examples need a
                                    # block-causal mask which transformers'
                                    # default forward doesn't build for us.
                                    # Off = correct; on = faster but leaks
                                    # attention across packed boundaries.
    block_causal_mask: bool = False  # when packing, isolate packed documents
                                     # with segment-level causal attention.
    num_workers: int = 4
    prefetch_factor: int = 4

    # Eval data
    eval_num_batches: int = 16

    # Checkpointing
    checkpoint_dir: str = "/tmp/checkpoints"
    auto_resume: bool = True       # load latest ckpt in checkpoint_dir if present


def make_headline_config() -> Config:
    """
    Headline run after the pilot succeeds. Trains every attention layer
    except the first (layer 0, sees raw token embeddings before context is
    mixed) and the last (layer 35, directly produces output logits — errors
    are unrecoverable). 34 of 36 layers = 11.1M trainable params.

    Why "all but 2" instead of a curated subset: the deployment-relevant
    claim is "we made attention sub-linear on a real model," which requires
    showing the technique works on essentially every layer, not just the
    easy ones a curated subset might be hiding behind.

    Step-time budget at 8K context, batch 8: ~1.5-2.5s/step. If it's slower
    than that, options: drop K_pos 16->8, sample contrastive negatives, or
    raise batch size if memory allows.
    """
    cfg = Config()
    cfg.seq_len = 8192
    cfg.total_steps = 6000
    cfg.warmup_steps = 200
    cfg.eval_every = 1000
    cfg.save_every = 500
    cfg.keep_last_n_checkpoints = 8
    cfg.full_attention_layer_indices = list(range(36))
    cfg.reserved_full_attention_indices = [0, 35]
    cfg.eval_num_batches = 32
    cfg.wandb_run_name = "headline-34layers-d64"
    cfg.checkpoint_dir = "/tmp/checkpoints_headline"
    return cfg


def make_pilot_d64_clean_config() -> Config:
    """
    Pilot rerun with packing off and the post-packing-fix data path.
    Identical to the original pilot otherwise: 6 layers, d=64, 2K steps.
    Used as the baseline for the d_search ablation.
    """
    cfg = Config()
    cfg.d_search = 64
    cfg.wandb_run_name = "pilot-d64-clean"
    cfg.checkpoint_dir = "/tmp/checkpoints_d64"
    return cfg


def make_pilot_d64_packed_config() -> Config:
    """
    Packed d=64 ablation. This reproduces the original high-density pilot
    regime for fast d_search comparison. Known caveat: packed examples do not
    get a true block-causal segment mask in the default HF forward path.
    """
    cfg = Config()
    cfg.d_search = 64
    cfg.sequence_packing = True
    cfg.wandb_run_name = "pilot-d64-packed"
    cfg.checkpoint_dir = "/tmp/checkpoints_packed_d64"
    return cfg


def make_pilot_d128_config() -> Config:
    """d_search=128 capacity ablation. Same training budget as the pilot."""
    cfg = Config()
    cfg.d_search = 128
    cfg.wandb_run_name = "pilot-d128"
    cfg.checkpoint_dir = "/tmp/checkpoints_d128"
    return cfg


def make_pilot_d128_packed_config() -> Config:
    """Packed d_search=128 capacity ablation."""
    cfg = make_pilot_d64_packed_config()
    cfg.d_search = 128
    cfg.wandb_run_name = "pilot-d128-packed"
    cfg.checkpoint_dir = "/tmp/checkpoints_packed_d128"
    return cfg


def make_pilot_d128_block_config() -> Config:
    """Packed d=128 with segment-level block-causal masking (clean path)."""
    cfg = make_pilot_d128_packed_config()
    cfg.block_causal_mask = True
    cfg.wandb_run_name = "pilot-d128-block-causal"
    cfg.checkpoint_dir = "/tmp/checkpoints_block_d128"
    return cfg


def make_pilot_d256_config() -> Config:
    """d_search=256 capacity ablation. Same training budget as the pilot."""
    cfg = Config()
    cfg.d_search = 256
    cfg.wandb_run_name = "pilot-d256"
    cfg.checkpoint_dir = "/tmp/checkpoints_d256"
    return cfg


def make_pilot_d256_packed_config() -> Config:
    """Packed d_search=256 capacity ablation."""
    cfg = make_pilot_d64_packed_config()
    cfg.d_search = 256
    cfg.wandb_run_name = "pilot-d256-packed"
    cfg.checkpoint_dir = "/tmp/checkpoints_packed_d256"
    return cfg


def make_headline_d128_config() -> Config:
    """
    Capacity ablation: same as headline but d_search=128.
    Tests whether the pilot's PPL-gap plateau is set by training (no), data
    (no), or projection capacity (the question). If d=128 closes the gap
    further, capacity was the bottleneck; if not, the technique has bottomed
    out for this architecture.
    """
    cfg = make_headline_config()
    cfg.d_search = 128
    cfg.wandb_run_name = "headline-d128-34layers"
    cfg.checkpoint_dir = "/tmp/checkpoints_headline_d128"
    return cfg


def make_all36_d128_block_config() -> Config:
    """
    All-attention-layer clean run: d_search=128 on every layer, with packed
    block-causal masking. This is the strongest "ANN all-out" pilot.

    Batch size is reduced because QK reconstruction stores teacher attention
    for every trained layer. At 4K context, batch 8 is plausible for 6 layers
    but too aggressive for all 36 layers.
    """
    cfg = make_pilot_d128_block_config()
    cfg.full_attention_layer_indices = list(range(36))
    cfg.reserved_full_attention_indices = []
    cfg.batch_size = 2
    cfg.gradient_accumulation_steps = 4  # keep effective batch near the 6-layer pilot
    cfg.total_steps = 1000
    cfg.eval_every = 250
    cfg.save_every = 100
    cfg.keep_last_n_checkpoints = 10
    cfg.eval_num_batches = 8
    cfg.log_every = 25
    cfg.wandb_run_name = "all36-d128-block-causal"
    cfg.checkpoint_dir = "/tmp/checkpoints_all36_d128_block"
    return cfg
