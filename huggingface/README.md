---
license: mit
library_name: transformers
tags:
  - sparse-attention
  - approximate-nearest-neighbors
  - faiss
  - qwen3
  - long-context
base_model: Qwen/Qwen3-4B-Instruct-2507
---

# ann-sparseattention

This repository mirrors checkpoints and result artifacts for
[`unixsysdev/ann-sparseattention`](https://github.com/unixsysdev/ann-sparseattention).

The method trains tiny per-layer query/key "search projections" on a frozen
LLM so that attention-relevant keys are nearest neighbors in a low-dimensional
search space. At inference, candidate selection can be done with standard ANN
machinery such as FAISS HNSW, then ordinary attention is computed over the
retrieved native KV vectors.

This is a research prototype, not a production sparse-attention runtime.

## Current status

Validated clean pilot:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset: WikiText-103
- Context: 4096 tokens
- Clean masking: packed block-causal segment isolation
- Recommended clean checkpoint:
  `checkpoints_block_d128/search_step_1000.pt`
- Trained layers: `[4, 8, 12, 16, 20, 24]`
- `d_search=128`
- Trainable parameters: 3.93M
- K=128: +0.07% PPL gap vs full attention
- K=256: +0.01% PPL gap vs full attention

Broad-layer experiments:

- `checkpoints_all36_d128_block/protected/search_step_500_keep.pt` is the best
  all-36 checkpoint observed so far.
- All-36 step 500: recall@K=0.816, PPL gap +3.23%.
- All-36 step 750 regressed to +3.96% despite stable recall.
- Per-layer mass@K identified L00/L01/L02 as the weak early layers.
- A follow-up all32 run reserves full attention on `[0, 1, 2, 35]` and trains
  layers `3..34`; checkpoints will be mirrored here as they become useful.

## Important results

### Clean six-layer block-causal result

| K | Recall@K | mass@K | sparse PPL | PPL gap |
|---:|---:|---:|---:|---:|
| 128 | 0.744 | 0.787 | 30.47 | +0.07% |
| 256 | 0.879 | 0.953 | 30.45 | +0.01% |
| 512 | n/a | n/a | 30.45 | +0.01% |

The large negative PPL gaps from earlier packed-with-leakage experiments are
not used as clean claims. With block-causal masking, the robust claim is
full-attention parity on the six-layer pilot.

### Quest-style page baseline

| Method | K | Recall@K | mass@K | PPL | PPL gap |
|---|---:|---:|---:|---:|---:|
| learned search exact | 128 | 0.744 | 0.787 | 30.47 | +0.07% |
| Quest-style page | 128 | 0.669 | 0.727 | 30.41 | -0.11% |
| learned search exact | 256 | 0.879 | 0.953 | 30.45 | +0.01% |
| Quest-style page | 256 | 0.838 | 0.909 | 30.45 | +0.03% |

Paired 32-batch NLL evaluation:

| K | full PPL | learned PPL | Quest PPL | learned - Quest NLL delta, 95% CI | Read |
|---:|---:|---:|---:|---:|---|
| 128 | 28.03 | 28.07 | 28.01 | +0.00205 `[+0.00160, +0.00251]` | Quest slightly better |
| 256 | 28.03 | 28.04 | 28.04 | -0.00005 `[-0.00029, +0.00018]` | statistical tie |

The honest claim is retrieval-fidelity and ANN-compatibility, not a PPL win
over Quest.

### FAISS/HNSW compatibility

The corrected clean FAISS path builds per-segment HNSW indexes when a
block-causal mask is present.

| Method | K | PPL | PPL gap |
|---|---:|---:|---:|
| learned exact | 128 | 30.47 | +0.07% |
| learned FAISS/HNSW | 128 | 30.47 | +0.09% |
| learned exact | 256 | 30.45 | +0.01% |
| learned FAISS/HNSW | 256 | 30.46 | +0.04% |

This validates that the learned search vectors are compatible with
off-the-shelf ANN. It is not a wall-clock result: the prototype uses CPU FAISS
and per-forward index construction.

### All-36 result so far

| Step | Recall@K eval | PPL gap |
|---:|---:|---:|
| 250 | 0.805 | +6.27% |
| 500 | 0.816 | +3.23% |
| 750 | 0.817 | +3.96% |

All-36 is feasible but not parity under current hyperparameters. Step 500 is
kept because it is the best observed PPL checkpoint.

Per-layer step-500 mass@K at K=128:

| Layer | raw-QK | learned | delta |
|---:|---:|---:|---:|
| L00 | 0.922 | 0.780 | -0.142 |
| L01 | 0.918 | 0.851 | -0.067 |
| L02 | 0.939 | 0.899 | -0.040 |
| L03 | 0.939 | 0.924 | -0.015 |
| L04 | 0.944 | 0.933 | -0.011 |
| L05 | 0.964 | 0.947 | -0.017 |
| L06 | 0.956 | 0.936 | -0.020 |
| L07 | 0.982 | 0.982 | +0.000 |
| L08 | 0.971 | 0.970 | -0.001 |
| L09 | 0.959 | 0.976 | +0.017 |
| L20 | 0.959 | 0.975 | +0.016 |
| L21 | 0.966 | 0.979 | +0.014 |
| L34 | 0.976 | 0.960 | -0.016 |
| L35 | 0.980 | 0.967 | -0.013 |
| avg | 0.966 | 0.960 | -0.006 |

The next run reserves `[0, 1, 2, 35]` and trains layers `3..34`.

First diagnostic from the active all32 run:

| Step | Recall@K eval | PPL gap | Read |
|---:|---:|---:|---|
| 250 | 0.812 | +2.28% | already better than all36 best training eval |

This is not a final result; the run is continuing toward step 1000.

## Positioning against related methods

The paper frames this method as closest in asymptotic shape to Reformer and
closest in practical baseline behavior to Quest.

| Method | Selection mechanism | Query-aware | Trained | Asymptotic | Exact softmax |
|---|---|---|---|---|---|
| Full attention | all keys | n/a | n/a | O(N²) | yes |
| Reformer | LSH hashing | yes | no | O(N log N) | over bucket |
| Performer | random features | n/a | no | O(N) | no |
| BigBird | window + random + global | mostly no | no | O(N) | over pattern |
| Longformer | sliding window + global | mostly no | no | O(N) | over pattern |
| NSA-style methods | block compression/selection | partial | partial | O(N²) proxy | yes |
| Quest | min/max page heuristic | yes | no | O(N) | over pages |
| This work | trained low-dim retrieval | yes | yes | O(N log N) | over retrieved set |

This is a design-positioning table, not a claim of completed production
superiority. The clean result proves the approach for the six-layer pilot; the
active all32 reserved-layer run tests whether broad near-whole-model
substitution can preserve that quality.

This method targets a different deployment scenario than native
sliding-window/state-space/hybrid architectures such as Mistral-style sliding
window, Mamba, or Qwen3.6 Gated DeltaNet hybrids. Those models are trained from
scratch with their sparse or hybrid mechanism in place. This work is post-hoc:
train a base model with full attention for maximum expressivity, then add
lightweight retrieval projections afterward to make inference sub-linear without
changing base weights.

## Checkpoints

Important checkpoint paths in this HF repo:

- `checkpoints_block_d128/search_step_1000.pt`: clean six-layer d128 parity checkpoint.
- `checkpoints_all36_d128_block/protected/search_step_500_keep.pt`: best observed all-36 checkpoint so far.
- `checkpoints_all36_d128_block/search_step_800.pt`: latest all-36 checkpoint before stopping for analysis.
- `checkpoints_all32_d128_block_reserve_0_1_2_35/`: active follow-up, uploaded as useful checkpoints are saved.

These checkpoints contain the trained search projection module and optimizer
state. They do not contain or modify the base Qwen model weights.

## Limitations

- No production wall-clock speedup has been measured.
- No GPU-resident ANN or fused sparse attention kernel yet.
- No autoregressive KV-cache integration yet.
- Dynamic indexing is currently supported only by a retrieval-mass proxy.
- Main clean results are single-model and mostly single-seed.
- All-36 broad substitution is not full-attention parity yet.

Use the GitHub repository for runnable code, scripts, and the LaTeX paper draft.
