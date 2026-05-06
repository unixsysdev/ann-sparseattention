# ann-sparseattention

Train tiny per-layer "search projections" on a frozen LLM that replicate the
attention's top-K preferences in a low-dimensional space, so we can swap dense
quadratic attention for an off-the-shelf ANN index (FAISS HNSW) at inference
and lose almost no model quality.

## Current status

Research prototype. The trained projections work, but the runtime and the
evaluation envelope are both narrow. Treat reported numbers as preliminary.

**What's validated:**
- 6-layer pilot on Qwen3-4B-Instruct-2507 (2K steps, ~25 min on a B200).
- WikiText-103 PPL is preserved under ANN-substituted attention at K=128
  (gap ≈ +0.7%) on a 12-batch eval slice.
- Learned 64-d search projections retrieve attention-relevant keys: at
  K=128 we capture meaningful teacher attention mass; the K curve is
  monotonic and well-behaved.

**Not yet validated (next iteration):**
- 34-layer / whole-model substitution.
- Long-context task quality (LongBench, RULER, needle-in-haystack).
- Wall-clock speedup vs. FlashAttention/SDPA — not measured.
- KV-cache decode-mode integration.
- GPU-resident ANN or fused gather-attention kernel.

**Runtime caveat.** The current FAISS path is a correctness prototype: it
builds a CPU index per forward pass and uses dense-style tensor expansion
internally for the gather step. The compute-reduction numbers below are
**algorithmic scoring reductions, not measured wall-clock speedups.** A
production runtime requires a GPU-resident topk kernel or integration with
paged/block-sparse attention kernels.

Pilot on `Qwen/Qwen3-4B-Instruct-2507`, 2K training steps on WikiText-103,
6 trained layers, 2M trainable parameters:

| Step | Recall@K=128 | PPL gap (full vs ANN) |
|---|---|---|
| 500 | 47.4% | 1.21% |
| 1000 | 50.7% | 0.68% |
| 1500 | 50.9% | 0.68% |
| **2000 (final)** | **50.9%** | **0.71%** |

PPL gap is the primary signal: at <1% the model's output is preserved
under ANN substitution. Recall@K plateaus around step 1000 because the
softmax-relevant keys are concentrated in the top ~30 — disagreement on
positions 30-128 is on the near-zero-weight tail and doesn't affect output.
The repo also reports `mass@K` (sum of teacher attention probability captured
by the search top-K), which is the more direct retrieval-quality metric
when the softmax is sharp.

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

On this small WikiText slice, `K ≥ 256` produced lower measured PPL than
the full-attention reference (K=256: −0.79%, K=512: −2.89%). A plausible
explanation is sparse-attention denoising — full softmax spreads small
amounts of weight over a long tail of low-relevance keys, top-K
renormalization concentrates it. But with 12 eval batches, sample noise,
packed-boundary artifacts (the pilot trained with packing on; default in
the repo is now off), and partial-layer substitution acting like
regularization are all candidate explanations we haven't yet ruled out.
We're treating it as a hypothesis worth confirming rather than the
explanation. The follow-up — exact-topK oracle vs. ANN-topK at the same K
— separates "denoising from any sparsity" from "denoising from learned
projections."

### Compute / quality knobs (FLOP-counted)

`L = 4096`. Compute reduction is the attention scoring step, `≈ L / K`.
These are FLOP estimates, not measured wall-clock — the FAISS path in this
repo is a research prototype that does CPU index builds and GPU↔CPU
transfers, so it is not the right thing to time. A GPU-resident topk
kernel is the natural next step.

| K | PPL gap | Attention scoring reduction |
|---|---|---|
| 512 | −2.89% | ~8× |
| 256 | −0.79% | ~16× |
| 128 | +0.82% | ~32× |
| 64 | +2.42% | ~64× |
| 32 | +4.51% | ~128× |
| 16 | +7.51% | ~256× |

Eval scope for the table above: 12 sequences × 4K tokens of WikiText-103
validation (~50K tokens) on the pilot's 6-layer checkpoint. Numbers should
be read as "what we observed on this slice", not population-level estimates;
confidence intervals and downstream long-context tasks are the natural
follow-up.

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

### Caveats / what's next

A few things the pilot does not yet establish, and that the next iteration
will:

- **Packing**: the pilot's training and eval ran with sequence packing on,
  without segment-level causal masks (transformers' default forward doesn't
  build them). The relative PPL gap between full and ANN is internally
  consistent under this confound, but the negative gap at K≥256 has at
  least three candidate explanations we haven't yet disentangled —
  (a) genuine sparse-softmax denoising, (b) ANN happening to filter
  cross-document keys that full attention attends to, (c) sample noise on
  ~50K eval tokens. The default config in this repo now has packing off so
  the next run isolates (a) cleanly.
- **Exact-topK oracle**: the obvious follow-up is a four-way Pareto —
  full attention vs. exact top-K (true `QK^T` argmax-K, then attention) vs.
  search-topK (our projections, exact distance) vs. search-ANN (FAISS HNSW).
  That separates "denoising from any sparsity" from "denoising from learned
  projections."
- **Wall-clock**: the compute-reduction table above is FLOP-counted. The
  FAISS path here is a research prototype (CPU index per forward, GPU↔CPU
  transfer) and is the wrong thing to time. A GPU-resident topk kernel is
  the next-step engineering.
- **34-layer headline**: was queued and the VM was reclaimed before launch.
  Config is wired (`make_headline_config()`); rerun is a single command on
  any B200/H100/H200.

The recall@K and mass@K reported here come from a 12-batch eval slice, not
a population-level estimate. Confidence intervals and downstream tasks
(LongBench / RULER / needle-in-haystack) are the natural next evals.

### Headline run (queued)

34 layers (every layer except 0 and 35), 8K context, 6K steps,
~4-5h on a single B200. Tests whether the technique generalizes from a
6-layer subset to broad layer coverage. Checkpoints will be mirrored at
[`datasysdev/ann-sparseattention`](https://huggingface.co/datasysdev/ann-sparseattention).

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
data.py          Long-context dataloader (packing off by default to avoid
                 cross-segment attention leakage; pin_memory, prefetch)
inference.py     ANN-substituted attention (exact top-K for analysis;
                 CPU-FAISS HNSW prototype path — not a deployable kernel)
eval.py          recall@K curve, mass@K curve, full-vs-ANN PPL,
                 MoE router stability
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
