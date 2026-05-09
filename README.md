# ANN Sparse Attention Runtime Snapshot

This branch is intentionally runtime-only. It contains the llama.cpp runtime integration, build artifacts, prompts, logs, benchmark summaries, and selected checkpoints for the learned ANN sparse-attention experiments. It is **not** intended to be merged into `main`.

Detailed docs:

- [Runtime status and benchmark details](runtime/README.md)
- [Engineering status](runtime/ANN_LLAMA_CPP_STATUS.md)
- [Model manifest and GGUF notes](runtime/models/MODELS.md)
- [Password recall HNSW raw summary](runtime/results/llama_recall_password_hnsw/password_recall_hnsw_summary.md)
- [Sample outputs](runtime/results/llama_samples/sample_outputs.md)
- [Exact 1k smoke matrix](runtime/results/llama_ann/summary.md)
- [HNSW smoke matrix](runtime/results/llama_ann_hnsw/hnsw_summary.md)
- [16.8k long-context exact summary](runtime/results/llama_longctx/longctx_exact_16k_summary.md)
- [33.6k long-context exact summary](runtime/results/llama_longctx/longctx_exact_32k_summary.md)

## What Is Implemented

The patched llama.cpp runtime supports Qwen3 ANN projection tensors embedded in GGUF and adds an `S` search-key cache beside the native K/V cache. Prefill remains full attention. During decode, sparse layers compute learned search queries, retrieve candidate positions, gather native K/V, and run exact softmax over the retrieved subset.

Two decode paths exist:

- **Exact learned top-K**: correctness oracle and currently the faster sparse path.
- **HNSW approximate ANN**: enabled with `LLAMA_ANN_SEARCH=hnsw`; this is real approximate candidate selection through vendored `hnswlib`, but currently rebuilds indices per sparse layer per decode token. It validates behavior, not final production latency.

Tested on:

- CPU
- ROCm on AMD Strix Halo / Radeon 8060S Graphics

## Model Variants Tested

- `base_full`: original Qwen3-4B-Instruct-2507 F16 GGUF, full attention.
- `ann_6layer`: sparse substitution on the 6-layer pilot set.
- `ann_all32`: 32-of-36 sparse substitution, reserving edge layers.
- `ann_all36`: all attention layers sparse-substituted.

The merged ANN runtime GGUFs are hosted on Hugging Face:

- [`Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf)
- [`Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf)
- [`Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf)

These are extremely preliminary runtime artifacts. They are useful for
reproducing the current llama.cpp integration tests, not for production
deployment or publication-strength performance claims. See
[runtime/models/MODELS.md](runtime/models/MODELS.md) for the model manifest.

## Key Speed Results

### 1k Context Smoke Test, Exact Path

At 1k context, attention is not yet the bottleneck, so sparse attention does not show a meaningful speed win.

| backend | model | prompt tok/s | decode tok/s | KV/S cache |
|---|---:|---:|---:|---:|
| CPU | base | 205.75 | 10.37 | 144 MiB |
| CPU | ANN 6-layer | 183.38 | 10.75 | 153 MiB |
| CPU | ANN all32 | 184.92 | 10.10 | 153 MiB |
| CPU | ANN all36 | 185.28 | 9.84 | 153 MiB |
| ROCm | base | 1838.42 | 23.35 | 144 MiB |
| ROCm | ANN 6-layer | 1820.46 | 22.99 | 153 MiB |
| ROCm | ANN all32 | 1788.14 | 22.00 | 153 MiB |
| ROCm | ANN all36 | 1792.84 | 21.83 | 153 MiB |

### 33.6k Context, Exact Path

At 33.6k context, the exact all32 sparse path starts to show the expected long-context speed direction.

| model | prompt tok/s | decode tok/s | eval ms/token | total ms | prefix quality note |
|---|---:|---:|---:|---:|---|
| base | 506.89 | 16.01 | 62.47 | 67263.52 | coherent |
| ANN 6-layer exact | 504.62 | 16.19 | 61.78 | 67551.54 | coherent |
| ANN all32 exact | 502.74 | 18.55 | 53.91 | 67678.54 | faster, but weaker repeated-prompt output |
| ANN all36 exact | 501.92 | 18.91 | 52.87 | 67772.32 | faster, visibly degraded output |

Interpretation:

- ANN all32 exact decode is about `+16%` faster than base at 33.6k tokens.
- ANN all36 is slightly faster but lower quality.
- Prefill remains full attention, so total wall time is still dominated by prompt evaluation.
- The synthetic repeated prompt is useful for speed stress, not final semantic quality.

## HNSW Approximate ANN Results

HNSW currently runs as a correctness bridge. It performs real ANN candidate selection but rebuilds the HNSW index inside decode, so it is slow, especially for 32/36 sparse layers.

| backend | model | prompt tok/s | decode tok/s | eval ms/token | output prefix |
|---|---:|---:|---:|---:|---|
| CPU | ANN 6-layer HNSW | 206.90 | 5.82 | 171.75 | `The provided text consists` |
| CPU | ANN all32 HNSW | 198.52 | 2.33 | 428.78 | `The cache is filled` |
| CPU | ANN all36 HNSW | 200.72 | 2.13 | 470.39 | `The given text is` |
| ROCm | ANN 6-layer HNSW | 1800.32 | 9.25 | 108.09 | `The provided text consists` |
| ROCm | ANN all32 HNSW | 1857.50 | 2.66 | 376.22 | `The cache is filled` |
| ROCm | ANN all36 HNSW | 1834.51 | 2.40 | 417.16 | `The sentences listed above` |

The next runtime step is persistent per-layer HNSW indices with incremental insertion, plus GPU-resident candidate selection to remove CPU graph splits.

## Password Recall Benchmark, HNSW ANN Path

This is the small recall test requested after the runtime integration. It places an exact password near the beginning of a 1k, 2k, or 4k-token prompt, adds irrelevant filler, and asks the model to output only the password. Base uses full attention. ANN variants use `LLAMA_ANN_SEARCH=hnsw`, so this is approximate ANN retrieval, not exact top-K.

| backend | model | mode | target ctx | actual tokens | pass | prompt tok/s | decode tok/s | KV/S MiB | answer |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ROCm | base_full | full | 1024 | 1018 | yes | 2004.5 | 23.79 | 1152.0 | `VIOLET-7319-RIVER` |
| ROCm | ann_6layer_hnsw | hnsw | 1024 | 1018 | yes | 2006.84 | 8.08 | 1224.0 | `VIOLET-7319-RIVER` |
| ROCm | ann_all32_hnsw | hnsw | 1024 | 1018 | yes | 1991.96 | 2.02 | 1224.0 | contains `VIOLET-7319-RIVER` |
| ROCm | ann_all36_hnsw | hnsw | 1024 | 1018 | NO | 1973.2 | 1.83 | 1224.0 | corrupted password |
| ROCm | base_full | full | 2048 | 2026 | yes | 1889.44 | 23.71 | 1152.0 | `ORBIT-4826-LANTERN` |
| ROCm | ann_6layer_hnsw | hnsw | 2048 | 2026 | yes | 1857.67 | 4.57 | 1224.0 | `ORBIT-4826-LANTERN` |
| ROCm | ann_all32_hnsw | hnsw | 2048 | 2026 | NO | 1860.68 | 1.02 | 1224.0 | corrupted password |
| ROCm | ann_all36_hnsw | hnsw | 2048 | 2026 | NO | 1865.89 | 0.92 | 1224.0 | failed |
| ROCm | base_full | full | 4096 | 4090 | yes | 1691.76 | 23.07 | 1152.0 | `CIPHER-9051-MARBLE` |
| ROCm | ann_6layer_hnsw | hnsw | 4096 | 4090 | yes | 1667.51 | 2.33 | 1224.0 | `CIPHER-9051-MARBLE` |
| ROCm | ann_all32_hnsw | hnsw | 4096 | 4090 | NO | 1616.89 | 0.49 | 1224.0 | corrupted password |
| ROCm | ann_all36_hnsw | hnsw | 4096 | 4090 | NO | 1616.87 | 0.44 | 1224.0 | failed |
| CPU | base_full | full | 1024 | 1018 | yes | 210.95 | 7.47 | 1152.0 | `VIOLET-7319-RIVER` |
| CPU | ann_6layer_hnsw | hnsw | 1024 | 1018 | yes | 199.53 | 4.75 | 1224.0 | `VIOLET-7319-RIVER` |
| CPU | ann_all32_hnsw | hnsw | 1024 | 1018 | yes | 192.17 | 1.72 | 1224.0 | `VIOLET-7319-RIVER` |
| CPU | ann_all36_hnsw | hnsw | 1024 | 1018 | NO | 195.81 | 1.61 | 1224.0 | corrupted password |
| CPU | base_full | full | 2048 | 2026 | yes | 177.2 | 9.03 | 1152.0 | `ORBIT-4826-LANTERN` |
| CPU | ann_6layer_hnsw | hnsw | 2048 | 2026 | yes | 180.25 | 3.24 | 1224.0 | `ORBIT-4826-LANTERN` |
| CPU | ann_all32_hnsw | hnsw | 2048 | 2026 | NO | 174.96 | 0.93 | 1224.0 | corrupted password |
| CPU | ann_all36_hnsw | hnsw | 2048 | 2026 | NO | 183.85 | 0.84 | 1224.0 | failed |
| CPU | base_full | full | 4096 | 4090 | yes | 160.73 | 6.67 | 1152.0 | `CIPHER-9051-MARBLE` |
| CPU | ann_6layer_hnsw | hnsw | 4096 | 4090 | yes | 155.3 | 1.82 | 1224.0 | `CIPHER-9051-MARBLE` |
| CPU | ann_all32_hnsw | hnsw | 4096 | 4090 | NO | 158.75 | 0.47 | 1224.0 | corrupted password |
| CPU | ann_all36_hnsw | hnsw | 4096 | 4090 | NO | 155.49 | 0.42 | 1224.0 | failed |

Interpretation:

- `ann_6layer_hnsw` passes all 1k/2k/4k exact-password recall tests on CPU and ROCm.
- `ann_all32_hnsw` passes 1k but fails exact recall at 2k and 4k.
- `ann_all36_hnsw` fails exact recall across the test.
- This supports the earlier quality conclusion: aggressive all-layer substitution is not currently safe for exact-recall tasks with the HNSW bridge.

## Exact Sample Outputs

Short prompt outputs are coherent across base, exact ANN variants, and 6-layer HNSW. The full text is in [runtime/results/llama_samples/sample_outputs.md](runtime/results/llama_samples/sample_outputs.md).

Example all32 exact output:

```text
Sparse attention reduces computational cost by selectively connecting query vectors to a subset of key-value pairs, rather than all. This enables faster inference on large models. Practical tradeoff: reduced context modeling due to limited attention spans, potentially degrading long-range dependency capture. Failure mode: catastrophic forgetting of distant context when sparse connections are too restrictive, leading to hallucinations or factual errors.
```

## Current Assessment

What is real now:

- llama.cpp loads learned ANN projection tensors from GGUF.
- CPU and ROCm builds run base, 6-layer, all32, and all36 variants.
- Exact learned top-K decode works and shows a long-context speed signal.
- HNSW approximate candidate selection works on CPU and ROCm.
- 6-layer HNSW passes exact password recall up to 4k tokens.

What is not solved yet:

- HNSW is not production-optimized; it rebuilds indices during decode.
- all32/all36 HNSW fail exact password recall beyond 1k in this strict benchmark.
- Prefill remains full attention.
- The base Qwen GGUF is not mirrored here; use the upstream/base GGUF locally and the ANN GGUFs linked above for these runtime tests.

## Next Engineering Steps

1. Persistent per-layer HNSW indices instead of per-token rebuilds.
2. Incremental insertion of decoded search keys.
3. GPU-resident candidate selection for ROCm.
4. Fused sparse gather-attention kernel.
5. Real long-context quality benchmarks beyond synthetic filler prompts.
