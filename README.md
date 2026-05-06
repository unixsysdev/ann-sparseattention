# ann-sparseattention

Train tiny per-layer "search projections" on a frozen LLM that replicate the
attention's top-K preferences in a low-dimensional space, so we can swap dense
quadratic attention for an off-the-shelf ANN index (FAISS HNSW) at inference
and lose almost no model quality.

Pilot result on `Qwen/Qwen3-4B-Instruct-2507`, 2K training steps on WikiText-103,
6 trained layers, 2M trainable parameters:

- **PPL gap (full vs ANN-substituted): ~1.2%** at step 500, holding through the run
- **Recall@K=128 climbing** from random → ~50% by step 500

Checkpoints + headline results are mirrored at
[https://huggingface.co/dalletest123/ann-sparseattention](https://huggingface.co/dalletest123/ann-sparseattention).

## How it works

For each full-attention layer `i` we train two linear projections
`W_Qs^i, W_Ks^i ∈ R^{d_model × d_search}` (d_search=64), so that for any
hidden state `h`,

```
q_search = W_Qs^i h        k_search = W_Ks^i h
softmax(q_search · k_search^T)  ranks the same keys as
softmax(QK^T / √d_head)         (the teacher's attention)
```

Two losses, summed across layers:

- **InfoNCE** with teacher-derived positives (top-`K_pos` keys from the
  teacher's attention serve as positives for each query).
- **KL(teacher ‖ student)** on the full attention distribution.

At inference, we monkey-patch each trained layer's attention forward to:

1. Compute `q_search`, `k_search` from the same hidden state.
2. Build a per-batch FAISS HNSW index over `k_search` (default params).
3. Retrieve top-`K_retrieve` positions (causal-respecting) per query.
4. Run standard attention restricted to those `K_retrieve` keys.

The base model's parameters are never touched. Only ~2M parameters trained
total per run.

## Repo layout

```
config.py        Run config (pilot defaults; make_headline_config() for follow-up)
model.py         SearchProjection, FrozenForwardCapture (with QK reconstruction
                 trick: capture (Q, K) post-RoPE while the forward stays in FA),
                 contrastive + KL distillation losses
data.py          Long-context packed dataloader (sequence packing,
                 pin_memory, prefetch)
inference.py     ANN-substituted attention forward (FAISS HNSW or exact top-K)
eval.py          Recall@K curve, full-vs-ANN PPL, MoE router stability
train.py         Training loop, Liger setup, FA-3→FA-2→SDPA→eager fallback,
                 base-model freeze + drift check, auto-resume from latest ckpt
tests/           QK reconstruction verification + 50-step smoke test
```

## Quick start

```bash
pip install -r requirements.txt
export WANDB_API_KEY=<key>      # only — never check it in
export HF_TOKEN=<token>         # for faster Hub downloads

# Pre-launch checks
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3-4B-Instruct-2507'))"
python tests/test_qk_reconstruction.py
python tests/smoke_test.py

# Pilot
python train.py
```

## Configuration

The default `Config` is the 1-day pilot:

| Knob | Pilot | Headline |
|---|---|---|
| `seq_len` | 4096 | 8192 |
| `batch_size` | 8 | 8 |
| `total_steps` | 2000 | 8000 |
| `full_attention_layer_indices` | `[4,8,12,16,20,24]` (6 layers) | `[2,6,10,14,18,22,26,30,34]` (9 layers) |
| `d_search` | 64 | 64 |
| `K_retrieve_eval` | 128 | 128 |

Switch with `from config import Config, make_headline_config; cfg = make_headline_config()`.

## Performance choices

- `attn_implementation` resolves at load time as
  `flash_attention_3 → flash_attention_2 → sdpa → eager`. On B200 with no
  flash-attn package installed, SDPA wins — its built-in flash backend is
  ~80-90% of FA-2's throughput with zero build dependency.
- Liger kernels applied via `apply_liger_kernel_to_qwen3` (RMSNorm, SwiGLU,
  RoPE fused — typically 30-50% faster forward).
- The QK-reconstruction trick keeps SDPA/FA fast on the trained layers:
  we monkey-patch them to capture `(Q, K)` post-RoPE, then reconstruct
  `softmax(QK^T/√d)` ourselves *after* the forward returns. The forward
  never sets `output_attentions=True` (which would force eager).
- `torch.compile(search_module, mode="max-autotune")` on the search
  projections; base model uncompiled (works but flaky for novel architectures).
- bf16 throughout; loss math cast to fp32 for numerical stability of softmax.

## Verifying the QK reconstruction

The post-RoPE Q/K capture must match what the model's eager attention computes
or distillation supervision is wrong. The test asserts top-32 agreement
> 99% per layer:

```bash
python tests/test_qk_reconstruction.py --model Qwen/Qwen3-4B-Instruct-2507
# layer 0: PASS  max|Δ|=2.54e-02  top-32 agree=0.9963
# layer 1: PASS  max|Δ|=5.27e-02  top-32 agree=0.9941
# ...
# QK reconstruction verified.
```

The bf16 max-abs differences (~0.05) are just numerical noise; the
*ranking* of attention positions matches.

## Reproducing the pilot

```bash
git clone git@github.com:unixsysdev/ann-sparseattention.git
cd ann-sparseattention
pip install -r requirements.txt
python train.py
```

A single H100/H200/B200 + 8GB GPU RAM for the 4B model + ~10GB for activations
at 4K context, batch 8.

## License

MIT.
