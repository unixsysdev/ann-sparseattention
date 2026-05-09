# All32 Reserved-Edge Final Summary (May 9, 2026)

Configuration: `all32_d128_block`, Qwen3-4B-Instruct-2507, clean packed block-causal masking, d_search=128, layers `3..34` substituted, layers `[0, 1, 2, 35]` kept as full attention.

## Training trajectory

| Step | Recall@K eval | PPL gap |
|---:|---:|---:|
| 250 | 0.812 | +2.283% |
| 500 | 0.823 | +1.753% |
| 750 | 0.825 | +1.943% |
| 1000 | 0.825 | +1.746% |

Final checkpoint: `/tmp/checkpoints_all32_d128_block_reserve_0_1_2_35/search_step_1000.pt`.

## Retrieval comparison at step 1000

Post-hoc `compare_retrieval` on the substituted layers:

| K | raw-QK mass | learned mass | Read |
|---:|---:|---:|---|
| 128 | 0.969 | 0.971 | learned matches/slightly exceeds raw-QK |
| 256 | 0.994 | 0.993 | tied |

## Exact K-sweep at step 1000

2-batch clean block-causal slice, `PPL_full = 20.5349`.

| K | mass@K | Recall@K | sparse PPL | PPL gap |
|---:|---:|---:|---:|---:|
| 16 | 0.546 | 0.518 | 24.86 | +21.064% |
| 32 | 0.627 | 0.572 | 21.85 | +6.422% |
| 64 | 0.722 | 0.652 | 20.94 | +1.974% |
| 128 | 0.807 | 0.746 | 20.66 | +0.590% |
| 256 | 0.902 | 0.876 | 20.52 | -0.062% |

K=512 is omitted from the headline table because the current metric path returns zero mass/recall when K exceeds valid same-segment causal keys for most queries. The sparse-attention PPL line ran, but the metric should be fixed and rerun before using K=512 publicly.

## Coverage picture

| Configuration | Layers substituted | Coverage | PPL gap | Read |
|---|---:|---:|---:|---|
| Clean six-layer pilot | 6/36 | 17% | +0.07% at K=128 | quality-preserving pilot |
| all32 reserved-edge | 32/36 | 89% | +1.746% train eval; +0.590% exact sweep | near-parity broad substitution |
| all36 | 36/36 | 100% | +3.23% best observed | full substitution costs quality |

Next headline experiments: standard short-context capability evals (HellaSwag, ARC-Easy, ARC-Challenge), needle-in-haystack at 8K/16K/32K, K=512 metric fix, and a 12/18/20-layer coverage Pareto sweep.
