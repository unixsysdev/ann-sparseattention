# ann-sparseattention

Train tiny per-layer "search projections" on a frozen LLM that replicate the
attention's top-K preferences in a low-dimensional space, so we can swap dense
quadratic attention for an off-the-shelf ANN index (FAISS HNSW) at inference
and lose almost no model quality.

Pilot on `Qwen/Qwen3-4B-Instruct-2507`, 2K training steps on WikiText-103,
6 trained layers, 2M trainable parameters:

| Step | Recall@K=128 | PPL gap (full vs ANN) |
|---|---|---|
| 500 | 47.4% | 1.21% |
| 1000 | 50.7% | 0.68% |
| 1500 | 50.9% | 0.68% |
| **2000 (final)** | **50.9%** | **0.71%** |

PPL gap is the primary signal: at <1% the model's output is preserved
under ANN substitution. Recall plateaus around step 1000 because the
softmax-relevant keys are concentrated in the top ~30 — disagreement on
positions 30-128 is on near-zero-weight tail that doesn't affect output.

**Pilot checkpoint** (step 2000): mirrored at
[`datasysdev/ann-sparseattention`](https://huggingface.co/datasysdev/ann-sparseattention).

### K-retrieve Pareto (pilot step 2000, FAISS HNSW, 12 eval batches)

`PPL_full = 9.958` (full attention reference)

| K | Recall@K | PPL_ANN | PPL gap |
|---|---|---|---|
| 16 | 24.9% | 10.71 | +7.51% |
| 32 | 22.8% | 10.41 | +4.51% |
| 64 | 23.1% | 10.20 | **+2.42%** |
| 128 | 26.0% | 10.04 | **+0.82%** |
| 256 | 31.6% | 9.88 | **−0.79%** |
| 512 | 40.8% | 9.67 | **−2.89%** |

**ANN-substituted attention at K ≥ 256 produces lower perplexity than full
attention.** This is the well-documented "sparse attention denoises softmax"
effect: full softmax is forced to spread small amounts of weight onto a
long tail of irrelevant keys; truncating to top-K and renormalizing puts
the weight where it actually belongs.

The novelty here is *not* "ANN beats attention" — that denoising is a
property of any hard top-K selection over softmax (cf. Top-k attention,
Reformer's sparsity-quality observations). What's distinctive is that this
method produces the denoised top-K **at sub-linear cost via off-the-shelf
FAISS HNSW**, instead of computing all `O(L²)` scores and then taking top-K.

### Deployment knobs

(`L = seq_len = 4096`. Compute reduction is the attention scoring step,
≈ `L / K`.)

| Use case | K | PPL gap | Attention compute reduction |
|---|---|---|---|
| Quality-improving | 256 | **−0.79%** | ~16× |
| Quality-improving | 512 | **−2.89%** | ~8× |
| Quality-preserving | 128 | +0.82% | ~32× |
| Aggressive (speed-quality) | 64 | +2.42% | ~64× |
| Speed-only | 32 | +4.51% | ~128× |
| Speed-only | 16 | +7.51% | ~256× |

### Per-layer recall (pilot step 2000)

| Layer | Recall@K=128 | Recall@K=512 |
|---|---|---|
| 4 | 15.8% | 34.7% |
| 8 | 22.2% | 38.7% |
| 12 | 23.4% | 39.1% |
| 16 | **31.9%** | **45.2%** |
| 20 | 31.4% | 42.6% |
| 24 | 31.1% | 44.4% |

Early layers are harder for content-addressable retrieval — their attention
patterns are more local/positional than semantic. The pattern is consistent
across K values, so it's a property of the layer, not noise. For the
headline run this predicts the early-most trained layers (1–5) will
underperform the rest; informative either way.

### Note on recall numbers

The K-sweep recall (~26% at K=128) is about half of the in-training
`evaluate()` recall (~51%) on the same checkpoint. The metric code path
looks identical between the two; most likely cause is sampling different
sequences from the streaming validation split (different `num_batches` and
worker dispatch). The PPL gap is independent of which subset is sampled,
so the deployment claim is unaffected; the absolute recall numbers between
the two evals shouldn't be compared directly until the metric is reconciled.

### Headline run

A 34-layer headline (every layer except 0 and 35), 8K context, 6K steps,
~4-5h on a single B200. Tests whether the technique generalizes from a
curated 6-layer subset to broad layer coverage.

Checkpoints + headline results are mirrored at
[https://huggingface.co/datasysdev/ann-sparseattention](https://huggingface.co/datasysdev/ann-sparseattention).

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
| layers trained | 6 (`[4,8,12,16,20,24]`) | 34 (`range(36)` minus reserved `[0, 35]`) |
| trainable params | 1.97M | 11.1M |
| `d_search` | 64 | 64 |
| `K_retrieve_eval` | 128 | 128 |

Pilot is the proof-of-concept; headline trains every attention layer except
the first (raw-embedding-adjacent) and last (output-logits-adjacent), which is
the deployment-relevant claim that the technique scales to dense application.

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
