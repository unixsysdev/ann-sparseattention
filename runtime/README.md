# ANN Sparse Attention llama.cpp Runtime Branch

This branch is a runtime snapshot for the learned ANN sparse-attention llama.cpp integration. It is not intended to be merged into `main` as research code. It holds the patched runtime, build outputs, prompts, logs, and reproducibility artifacts for the Strix Halo experiments.

Date: 2026-05-09
Hardware tested: AMD Strix Halo / Radeon 8060S Graphics through ROCm, plus CPU.
Runtime branch: `feat/llama-ann-runtime`.

## What Is In This Branch

- `runtime/llama.cpp-ann/`: vendored llama.cpp source with the ANN sparse-attention patch.
- `runtime/builds/llama-cpu/`: CPU build output.
- `runtime/builds/llama-hip/`: ROCm build output.
- `runtime/checkpoints/`: selected PyTorch checkpoints used to create ANN GGUFs.
- `runtime/scripts/merge_ann_checkpoint_to_gguf.py`: checkpoint-to-GGUF merger.
- `runtime/prompts/`: smoke, sample, and long-context prompts.
- `runtime/results/`: raw logs and summary tables from all runtime tests.
- `runtime/ANN_LLAMA_CPP_STATUS.md`: detailed engineering status.
- `runtime/models/MODELS.md`: model manifest and GGUF handling notes.

Merged ANN runtime GGUFs are hosted on Hugging Face:

- [`Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf)
- [`Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf)
- [`Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf`](https://huggingface.co/datasysdev/ann-sparseattention/blob/main/gguf/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf)

These files are extremely preliminary runtime artifacts for reproducing the
current tests. They should not be treated as production models or final
benchmark artifacts.

## Implemented Runtime Paths

### Exact learned sparse attention

The exact path is the correctness oracle. During prefill, the model uses full attention and populates the native K/V cache plus an `S` search-key cache. During decode (`n_tokens == 1`), sparse layers compute a learned query projection, score against cached learned search keys, take top-K candidates, gather native K/V, and run exact softmax attention over the selected subset.

### HNSW approximate ANN path

The HNSW path is enabled with:

```bash
LLAMA_ANN_SEARCH=hnsw
```

It uses vendored `hnswlib` to select candidates from the learned `S` cache, then runs exact softmax over gathered native K/V. This is real approximate ANN candidate selection, but it is a correctness bridge: it rebuilds the HNSW index inside decode instead of maintaining persistent per-layer dynamic indices. It is not the final optimized runtime.

## Short-Context Smoke Matrix, Exact Path

Prompt: `runtime/prompts/long_prompt.txt`  
Context: `-c 1024`  
Generation: `-n 16`  
ROCm flag: `-ngl 99`

| backend | model | S cache MiB | prompt tok/s | decode tok/s | eval ms | output prefix |
|---|---:|---:|---:|---:|---:|---|
| CPU | base | 0.0 | 205.75 | 10.37 | 1447.10 | `The provided text consists of repeated calibration sentences` |
| CPU | ANN 6-layer | 9.0 | 183.38 | 10.75 | 1395.86 | `The provided text consists of repeated calibration sentences` |
| CPU | ANN all32 | 9.0 | 184.92 | 10.10 | 1485.60 | `The provided text is a series of repeated, generic calibratio` |
| CPU | ANN all36 | 9.0 | 185.28 | 9.84 | 1525.03 | `The provided text consists of a series of repetitive calibrat` |
| ROCm | base | 0.0 | 1838.42 | 23.35 | 642.39 | `The provided text consists of repeated calibration sentences` |
| ROCm | ANN 6-layer | 9.0 | 1820.46 | 22.99 | 652.32 | `The provided text consists of repeated calibration sentences` |
| ROCm | ANN all32 | 9.0 | 1788.14 | 22.00 | 681.92 | `The provided text is a series of repetitive calibration sente` |
| ROCm | ANN all36 | 9.0 | 1792.84 | 21.83 | 687.17 | `The provided text consists of repetitive calibration sentence` |

Interpretation: at 1k context, there is no meaningful speed win. Attention is not yet the bottleneck, and the sparse path adds graph/gather overhead.

## Short-Context Memory Usage

From llama.cpp memory breakdown logs, context `-c 1024`:

| backend | model | model MiB | KV/S context MiB | compute MiB | total self MiB | notes |
|---|---:|---:|---:|---:|---:|---|
| CPU | base | 7672 | 144 | 306 | 8123 | no search cache |
| CPU | ANN 6-layer | 7680 | 153 | 306 | 8139 | includes `S` cache |
| CPU | ANN all32 | 7712 | 153 | 306 | 8172 | includes `S` cache |
| CPU | ANN all36 | 7717 | 153 | 306 | 8177 | includes `S` cache |
| ROCm | base | 7672 | 144 | 301 | 8118 | GPU self allocation |
| ROCm | ANN 6-layer | 7680 | 153 | 301 | 8134 | GPU self allocation |
| ROCm | ANN all32 | 7712 | 153 | 301 | 8167 | GPU self allocation |
| ROCm | ANN all36 | 7717 | 153 | 301 | 8172 | GPU self allocation |

Raw logs: `runtime/results/llama_ann/`.

## HNSW Smoke Matrix

Prompt: `runtime/prompts/long_prompt.txt`  
Context: `-c 1024`  
Generation: `-n 4`

| backend | model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---:|---|
| CPU | ANN 6-layer | 206.90 | 5.82 | 171.75 | 4364.30 | `The provided text consists` |
| CPU | ANN all32 | 198.52 | 2.33 | 428.78 | 5297.88 | `The cache is filled` |
| CPU | ANN all36 | 200.72 | 2.13 | 470.39 | 5378.84 | `The given text is` |
| ROCm | ANN 6-layer | 1800.32 | 9.25 | 108.09 | 768.59 | `The provided text consists` |
| ROCm | ANN all32 | 1857.50 | 2.66 | 376.22 | 1559.60 | `The cache is filled` |
| ROCm | ANN all36 | 1834.51 | 2.40 | 417.16 | 1687.48 | `The sentences listed above` |

Interpretation: HNSW candidate selection works on CPU and ROCm, but this bridge is slower than exact because it rebuilds the index for each sparse layer and decode token. The next runtime task is persistent per-layer HNSW indices with incremental insertion.

## Long-Context Decode Speed, Exact Path

These tests use synthetic repeated archive prompts. They are useful for throughput measurement and stress-testing long-context decode, but they are not a semantic quality benchmark.

### 16.8k-token prompt

Prompt: `runtime/prompts/long_16k_prompt.txt`  
Token count: 16,811  
Context: `-c 32768`  
Generation: `-n 32`

| model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---|
| base | 845.21 | 19.52 | 51.24 | 21505.49 | `Each bullet point should be concise and directly relevant to the research. - Sparse Attention...` |
| ANN 6-layer exact | 834.53 | 18.94 | 52.79 | 21807.23 | `Each bullet point must be exactly 100 words. Do not include any markdown...` |
| ANN all32 exact | 828.69 | 19.94 | 50.16 | 21864.02 | `Each bullet point should be no more than 20 words...` |
| ANN all36 exact | 828.55 | 19.81 | 50.47 | 21877.34 | `Each bullet point is a . 0 .` |

### 33.6k-token prompt

Prompt: `runtime/prompts/long_32k_prompt.txt`  
Token count: 33,601  
Context: `-c 65536`  
Generation: `-n 16`

| model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---|
| base | 506.89 | 16.01 | 62.47 | 67263.52 | `The summary should be concise and avoid technical jargon. - Sparse attention` |
| ANN 6-layer exact | 504.62 | 16.19 | 61.78 | 67551.54 | `The summary should be in English and in a formal tone. - The archive systematically` |
| ANN all32 exact | 502.74 | 18.55 | 53.91 | 67678.54 | `The record of the record of a record of the archive...` |
| ANN all36 exact | 501.92 | 18.91 | 52.87 | 67772.32 | `The research archive contains a 111 final` |

Interpretation:

- At 33.6k context, ANN all32 exact decode is faster than base: `18.55` vs `16.01` tok/s, about `+16%`.
- ANN all36 is also faster but quality is visibly worse, matching the research-side finding that all36 has larger PPL degradation.
- Prefill is still full attention for every variant, so total wall time is dominated by prompt evaluation.
- The long-context repeated prompt exposes quality differences: all32/all36 prefixes are weaker than base/6-layer, especially all36. This needs real long-context task evaluation before deployment claims.

## Long-Context Memory Usage

ROCm runs from `runtime/results/llama_longctx/`:

| prompt | model | model MiB | KV/S context MiB | compute MiB | total self MiB | host MiB |
|---|---:|---:|---:|---:|---:|---:|
| 16.8k | base | 7672 | 4608 | 301 | 12582 | 815 |
| 16.8k | ANN 6-layer | 7680 | 4896 | 301 | 12877 | 815 |
| 16.8k | ANN all32 | 7712 | 4896 | 301 | 12910 | 815 |
| 16.8k | ANN all36 | 7717 | 4896 | 301 | 12915 | 815 |
| 33.6k | base | 7672 | 9216 | 301 | 17190 | 879 |
| 33.6k | ANN 6-layer | 7680 | 9792 | 301 | 17773 | 879 |
| 33.6k | ANN all32 | 7712 | 9792 | 301 | 17806 | 879 |
| 33.6k | ANN all36 | 7717 | 9792 | 301 | 17811 | 879 |

The ANN variants add the learned search-key cache (`S`) on top of the normal K/V cache. At 33.6k tokens, this adds about 576 MiB of context memory versus base.

## Exact Sample Outputs

The following section is copied from `runtime/results/llama_samples/sample_outputs.md` so the branch README contains the words, not just throughput numbers.

# Sample Outputs

Prompt:
```
Write a concise technical explanation of sparse attention for a systems engineer. Include one practical tradeoff and one failure mode. Keep it under 120 words.
```

## base

```text
  

Sparse attention reduces computational cost by selectively computing attention weights between a subset of query-key pairs, rather than all pairs. This improves scalability in large models.  

*Tradeoff*: Reduced context modeling may lead to missed long-range dependencies, especially in sequential tasks.  

*Failure mode*: If sparsity is too aggressive, it can cause catastrophic forgetting of critical relationships, such as in time-series forecasting where temporal patterns are essential.  

Sparse attention enables efficient inference but requires careful tuning to


```

- common_perf_print: prompt eval time =     104.49 ms /    32 tokens (    3.27 ms per token,   306.24 tokens per second)
- common_perf_print:        eval time =    3921.01 ms /    95 runs   (   41.27 ms per token,    24.23 tokens per second)
- common_perf_print:       total time =    4038.82 ms /   127 tokens

## ann_6layer_exact

```text
  

Sparse attention reduces computational cost by selectively connecting attention heads to a subset of tokens, rather than all tokens. This improves scalability in large models. Practical tradeoff: reduced context modeling due to limited token interactions, potentially degrading long-range dependency capture. Failure mode: catastrophic forgetting of key relationships when sparse connections are poorly designed—e.g., missing critical token pairs in a sequence. This can lead to hallucinated outputs or incorrect inferences. Sparse attention is effective only when sparsity


```

- common_perf_print: prompt eval time =     108.84 ms /    32 tokens (    3.40 ms per token,   294.02 tokens per second)
- common_perf_print:        eval time =    3993.56 ms /    95 runs   (   42.04 ms per token,    23.79 tokens per second)
- common_perf_print:       total time =    4122.99 ms /   127 tokens

## ann_all32_exact

```text
  

Sparse attention reduces computational cost by selectively connecting query vectors to a subset of key-value pairs, rather than all. This enables faster inference on large models. Practical tradeoff: reduced context modeling due to limited attention spans, potentially degrading long-range dependency capture. Failure mode: catastrophic forgetting of distant context when sparse connections are too restrictive, leading to hallucinations or factual errors. Sparse attention is effective in resource-constrained environments but requires careful tuning of sparsity patterns and connection thresholds to


```

- common_perf_print: prompt eval time =     109.16 ms /    32 tokens (    3.41 ms per token,   293.16 tokens per second)
- common_perf_print:        eval time =    4143.08 ms /    95 runs   (   43.61 ms per token,    22.93 tokens per second)
- common_perf_print:       total time =    4268.47 ms /   127 tokens

## ann_all36_exact

```text
  

Sparse attention reduces computational cost by selectively connecting query vectors to a subset of key-value pairs, rather than all. This enables faster inference on large models. Practical tradeoff: reduced context modeling due to limited attention spans, potentially degrading long-range dependency capture. Failure mode: catastrophic forgetting of distant context when sparse connections are too restrictive, leading to hallucinations or factual errors. Sparse attention is effective in resource-constrained environments but requires careful tuning of sparsity patterns and thresholds to balance


```

- common_perf_print: prompt eval time =     108.77 ms /    32 tokens (    3.40 ms per token,   294.20 tokens per second)
- common_perf_print:        eval time =    4163.16 ms /    95 runs   (   43.82 ms per token,    22.82 tokens per second)
- common_perf_print:       total time =    4292.24 ms /   127 tokens

## ann_6layer_hnsw

```text
  

Sparse attention reduces computational cost by selectively applying attention mechanisms only to a subset of input tokens, rather than all pairs. This improves efficiency in large models by focusing computation on relevant token interactions. Tradeoff: reduced model accuracy due to missed interactions. Failure mode: if too sparse, the model may fail to capture critical dependencies, leading to hallucination or misclassification. In practice, sparse attention balances coverage and efficiency—too sparse risks losing essential context, while too dense negates performance


```

- common_perf_print: prompt eval time =     104.94 ms /    32 tokens (    3.28 ms per token,   304.94 tokens per second)
- common_perf_print:        eval time =    4206.13 ms /    95 runs   (   44.28 ms per token,    22.59 tokens per second)
- common_perf_print:       total time =    4324.81 ms /   127 tokens



## Password Recall Benchmark, HNSW ANN Path

This benchmark puts an exact password near the beginning of a 1k, 2k, or 4k-token prompt, fills the middle with irrelevant text, and asks the model to output only the password. Base uses normal full attention. ANN variants use `LLAMA_ANN_SEARCH=hnsw`, so this tests the approximate ANN retrieval path rather than exact top-K.

The rerun uses `-n 32` so exact-string failures are not just answer-budget truncation. Full raw outputs and memory lines are in `runtime/results/llama_recall_password_hnsw/password_recall_hnsw_summary.md`.

| backend | model | mode | target ctx | actual tokens | pass | prompt tok/s | decode tok/s | KV/S MiB | answer |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| rocm | base_full | full | 1024 | 1018 | yes | 2004.5 | 23.79 | 1152.0 | `Do not add extra text.  VIOLET-7319-RIVER [end of text]` |
| rocm | ann_6layer_hnsw | hnsw | 1024 | 1018 | yes | 2006.84 | 8.08 | 1224.0 | `Do not add extra text.  VIOLET-7319-RIVER [end of text]` |
| rocm | ann_all32_hnsw | hnsw | 1024 | 1018 | yes | 1991.96 | 2.02 | 1224.0 | `Do not add extra text.  VIOLET-7319-RENT VIOLET-7319-RIVER VIOLET` |
| rocm | ann_all36_hnsw | hnsw | 1024 | 1018 | NO | 1973.2 | 1.83 | 1224.0 | `Do not explain.  VIOLET-731173-71 RIVER  VIOLET-731173.` |
| rocm | base_full | full | 2048 | 2026 | yes | 1889.44 | 23.71 | 1152.0 | `Do not add extra text.  ORBIT-4826-LANTERN [end of text]` |
| rocm | ann_6layer_hnsw | hnsw | 2048 | 2026 | yes | 1857.67 | 4.57 | 1224.0 | `Do not add extra text.  ORBIT-4826-LANTERN [end of text]` |
| rocm | ann_all32_hnsw | hnsw | 2048 | 2026 | NO | 1860.68 | 1.02 | 1224.0 | `Do not add extra text.  ORR-4826, ORBIT-4826, ORR- note, OR, this is the` |
| rocm | ann_all36_hnsw | hnsw | 2048 | 2026 | NO | 1865.89 | 0.92 | 1224.0 | `Do not respond with only thing that the following is. memset system is irrelevant.ergency of the following:ity of the following: 1 The is irrelevant.` |
| rocm | base_full | full | 4096 | 4090 | yes | 1691.76 | 23.07 | 1152.0 | `Do not add any extra text.  CIPHER-9051-MARBLE [end of text]` |
| rocm | ann_6layer_hnsw | hnsw | 4096 | 4090 | yes | 1667.51 | 2.33 | 1224.0 | `Do not add extra text.  CIPHER-9051-MARBLE [end of text]` |
| rocm | ann_all32_hnsw | hnsw | 4096 | 4090 | NO | 1616.89 | 0.49 | 1224.0 | `Do not add extra words.  Answer the secret password is: CIPHER-9051- 1234- new line.  1` |
| rocm | ann_all36_hnsw | hnsw | 4096 | 4090 | NO | 1616.87 | 0.44 | 1224.0 | `Do not than the fills the  Continue to continuing. 11 paragraphs the first line 111 11 11 1 1` |
| cpu | base_full | full | 1024 | 1018 | yes | 210.95 | 7.47 | 1152.0 | `Do not add extra text.  VIOLET-7319-RIVER [end of text]` |
| cpu | ann_6layer_hnsw | hnsw | 1024 | 1018 | yes | 199.53 | 4.75 | 1224.0 | `Do not add extra text.  VIOLET-7319-RIVER [end of text]` |
| cpu | ann_all32_hnsw | hnsw | 1024 | 1018 | yes | 192.17 | 1.72 | 1224.0 | `Do not add extra text.  VIOLET-7319-RIVER  The correct password is: VIOLET-7319-RIVER` |
| cpu | ann_all36_hnsw | hnsw | 1024 | 1018 | NO | 195.81 | 1.61 | 1224.0 | `Do not infer anything else.  VIOLET-7319-RIVR  VIOLET-73119-RIVER  VIO` |
| cpu | base_full | full | 2048 | 2026 | yes | 177.2 | 9.03 | 1152.0 | `Do not add extra text.  ORBIT-4826-LANTERN [end of text]` |
| cpu | ann_6layer_hnsw | hnsw | 2048 | 2026 | yes | 180.25 | 3.24 | 1224.0 | `Do not add extra text.  ORBIT-4826-LANTERN [end of text]` |
| cpu | ann_all32_hnsw | hnsw | 2048 | 2026 | NO | 174.96 | 0.93 | 1224.0 | `Do not add extra text.  ORBIT-4826-EXACTLY-1234-1234-1234` |
| cpu | ann_all36_hnsw | hnsw | 2048 | 2026 | NO | 183.85 | 0.84 | 1224.0 | `Do not respond to the password.  ORBIT-  The secret is the secret password is. 2  OR: 4...  not the` |
| cpu | base_full | full | 4096 | 4090 | yes | 160.73 | 6.67 | 1152.0 | `Do not add any extra text.  CIPHER-9051-MARBLE [end of text]` |
| cpu | ann_6layer_hnsw | hnsw | 4096 | 4090 | yes | 155.3 | 1.82 | 1224.0 | `Do not add extra text.  CIPHER-9051-MARBLE [end of text]` |
| cpu | ann_all32_hnsw | hnsw | 4096 | 4090 | NO | 158.75 | 0.47 | 1224.0 | `Do not include any further.  The secret password is: CIPHER-1. The password is not to be ignored. The question. The prompt is. The` |
| cpu | ann_all36_hnsw | hnsw | 4096 | 4090 | NO | 155.49 | 0.42 | 1224.0 | `Do not forget the following  aergency  fill in filling the password. fill  ...ergency  The secret is the C memory test.  1...` |

Interpretation:

- `ann_6layer_hnsw` recalled all three passwords at 1k, 2k, and 4k on both ROCm and CPU.
- `ann_all32_hnsw` recalled the 1k password on both ROCm and CPU, but failed exact recall at 2k and 4k.
- `ann_all36_hnsw` failed exact recall across this test, consistent with the broader quality degradation seen for full-layer substitution.
- Base full attention recalled all three passwords after increasing generation budget to `-n 32`.
- HNSW decode throughput is currently poor for all32/all36 because the bridge rebuilds one HNSW index per sparse layer per decode token. These numbers are correctness/quality diagnostics, not production latency.

## Current Engineering Assessment

What is real now:

- Patched llama.cpp loads Qwen3 ANN projection tensors from GGUF.
- CPU and ROCm builds run base, 6-layer, all32, and all36 variants.
- Exact learned top-K decode works and is the correctness oracle.
- HNSW approximate candidate selection works in both CPU and ROCm builds.
- 33.6k-token exact all32 decode shows a speedup over base (`+16%`) on this hardware.
- Short prompt generations are coherent across tested variants.

What is not solved yet:

- HNSW is not production-optimized; it rebuilds indices during decode.
- Long-context sample quality on repeated synthetic prompts is weaker for all32/all36, especially all36.
- Prefill remains full attention, so end-to-end wall time is still prefill dominated for long prompts.
- The original base Qwen GGUF is not mirrored in this repo; use an upstream/base local GGUF with the ANN GGUFs linked above.

## Next Runtime Work

1. Persistent per-layer HNSW indices instead of rebuilding per decode token.
2. Incremental insertion of decoded search keys into those indices.
3. Explicit CLI/runtime flags for ANN search mode and HNSW parameters.
4. GPU-resident candidate selection for ROCm to remove CPU graph splits.
5. Real long-context quality benchmarks: needle-in-haystack, LongBench/RULER subset, and non-repeated document prompts.
