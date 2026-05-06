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
    total_steps: int = 2000
    warmup_steps: int = 100
    eval_every: int = 500
    save_every: int = 200          # ~10 ckpts in the pilot; resume on crash
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
    sequence_packing: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4

    # Eval data
    eval_num_batches: int = 16

    # Checkpointing
    checkpoint_dir: str = "/tmp/checkpoints"
    auto_resume: bool = True       # load latest ckpt in checkpoint_dir if present


def make_headline_config() -> Config:
    """
    Headline run after the pilot succeeds. Bumps seq_len to 8K, 8K total
    steps. 9 well-distributed layers (every 4th, layers 2..34) — fewer
    layers but trained more thoroughly is a stronger signal than 17 layers
    superficially trained.
    """
    cfg = Config()
    cfg.seq_len = 8192
    cfg.total_steps = 8000
    cfg.warmup_steps = 400
    cfg.eval_every = 1000
    cfg.save_every = 2000
    cfg.full_attention_layer_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34]
    cfg.eval_num_batches = 32
    cfg.wandb_run_name = "headline-8k-9layers"
    return cfg
