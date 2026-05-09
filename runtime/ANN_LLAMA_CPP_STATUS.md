# llama.cpp ANN Sparse Attention Runtime Status

Date: 2026-05-09

## What is implemented

- Cloned upstream `llama.cpp` at `runtime/llama.cpp-ann`.
- Added Qwen3 support for optional trained search projection tensors using the existing GGUF indexer tensor names:
  - `blk.{layer}.indexer.proj.weight` for `W_Qs`
  - `blk.{layer}.indexer.attn_k.weight` for `W_Ks`
- Added an `S` search-key cache beside the native K/V cache.
- Added Qwen3 decode-time learned sparse attention for `n_tokens == 1`:
  - prefill remains full attention and populates K/V/S caches
  - decode computes `q_search`, scores against cached `k_search`, takes exact top-K with `ggml_top_k`, gathers native K/V with `ggml_get_rows`, then runs exact softmax attention over the gathered subset
- Added an HNSW candidate-selection mode using vendored `hnswlib`, selected with `LLAMA_ANN_SEARCH=hnsw`:
  - HNSW builds over the learned `S` search-key cache
  - the retrieved candidates are still passed through native K/V exact softmax attention
  - on ROCm, HNSW candidate selection runs as a CPU custom op while the model remains GPU-offloaded
- Built and smoke-tested both CPU and ROCm (`gfx1151`, Strix Halo / Radeon 8060S).

The exact path is the correctness oracle. The HNSW path is real approximate ANN candidate selection, but the current bridge rebuilds the HNSW index inside the decode graph instead of maintaining persistent per-layer dynamic indices. It validates CPU/ROCm integration and output behavior, not final speed.

## Builds

- CPU: `runtime/builds/llama-cpu/bin/llama-completion`
- ROCm: `runtime/builds/llama-hip/bin/llama-completion`

ROCm was built with:

```bash
PATH=/opt/rocm/bin:$PATH HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
cmake -S runtime/llama.cpp-ann -B runtime/builds/llama-hip -G Ninja \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
```

## Models

Base:

- `runtime/models/Qwen3-4B-Instruct-2507-F16.gguf`

Merged ANN GGUFs:

- `runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf`
- `runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf`
- `runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf`

There is also one bad first merge kept for traceability:

- `runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-bad-shape.gguf`

Do not use the bad-shape file; the projection tensor dimensions were written transposed.

## Converter

The checkpoint-to-GGUF merge script is:

```bash
runtime/scripts/merge_ann_checkpoint_to_gguf.py
```

Example:

```bash
python3 runtime/scripts/merge_ann_checkpoint_to_gguf.py \
  --base-gguf runtime/models/Qwen3-4B-Instruct-2507-F16.gguf \
  --checkpoint runtime/checkpoints/checkpoints_all32_d128_block_reserve_0_1_2_35/search_step_1000.pt \
  --output runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf \
  --top-k 128
```

## Smoke Matrix

Prompt: `runtime/prompts/long_prompt.txt`

Command pattern:

```bash
runtime/builds/llama-hip/bin/llama-completion \
  -m runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf \
  -f runtime/prompts/long_prompt.txt \
  -n 16 -c 1024 -t 8 -fa on -ngl 99 \
  --no-warmup --no-display-prompt --no-conversation -s 123 --temp 0
```

Results are logged under `runtime/results/llama_ann/`.

| backend | model | S cache MiB | prompt tok/s | decode tok/s | eval ms |
|---|---:|---:|---:|---:|---:|
| CPU | base | 0.0 | 205.75 | 10.37 | 1447.10 |
| CPU | ANN 6-layer | 9.0 | 183.38 | 10.75 | 1395.86 |
| CPU | ANN all32 | 9.0 | 184.92 | 10.10 | 1485.60 |
| CPU | ANN all36 | 9.0 | 185.28 | 9.84 | 1525.03 |
| ROCm | base | 0.0 | 1838.42 | 23.35 | 642.39 |
| ROCm | ANN 6-layer | 9.0 | 1820.46 | 22.99 | 652.32 |
| ROCm | ANN all32 | 9.0 | 1788.14 | 22.00 | 681.92 |
| ROCm | ANN all36 | 9.0 | 1792.84 | 21.83 | 687.17 |

Interpretation:

- CPU and ROCm both load and run the merged ANN GGUFs.
- The patched runtime allocates an `S` search cache for ANN variants and no `S` cache for base.
- Decode graph node counts increase with substituted-layer coverage:
  - base: `1267`
  - 6-layer: `1279 (bs=512), 1357 (bs=1)`
  - all32: `1331 (bs=512), 1747 (bs=1)`
  - all36: `1339 (bs=512), 1806 (bs=1)`
- Exact top-K sparse decode is slightly slower than base on this small test. That is expected because the current path validates the learned sparse attention mechanics but does not yet use HNSW/FAISS-style approximate ANN.

## HNSW Smoke Matrix

Run with:

```bash
LLAMA_ANN_SEARCH=hnsw runtime/builds/llama-hip/bin/llama-completion \
  -m runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf \
  -f runtime/prompts/long_prompt.txt \
  -n 4 -c 1024 -t 8 -fa on -ngl 99 \
  --no-warmup --no-display-prompt --no-conversation -s 123 --temp 0
```

Results are logged under `runtime/results/llama_ann_hnsw/`.

| backend | model | prompt tok/s | decode tok/s | eval ms/token | total ms |
|---|---:|---:|---:|---:|---:|
| CPU | ANN 6-layer | 206.90 | 5.82 | 171.75 | 4364.30 |
| CPU | ANN all32 | 198.52 | 2.33 | 428.78 | 5297.88 |
| CPU | ANN all36 | 200.72 | 2.13 | 470.39 | 5378.84 |
| ROCm | ANN 6-layer | 1800.32 | 9.25 | 108.09 | 768.59 |
| ROCm | ANN all32 | 1857.50 | 2.66 | 376.22 | 1559.60 |
| ROCm | ANN all36 | 1834.51 | 2.40 | 417.16 | 1687.48 |

Interpretation:

- HNSW candidate selection runs successfully on both CPU and ROCm builds.
- ROCm offloads the model while scheduling the HNSW custom op on CPU, producing many graph splits in decode.
- This bridge is slower than exact top-K because it rebuilds the HNSW index for each sparse layer and decode token. A production implementation needs persistent per-layer HNSW indices, incremental insertion, and ideally a GPU-resident candidate-selection backend.

## Sample Output Check

Readable generations are logged under `runtime/results/llama_samples/`.

Prompt:

```text
Write a concise technical explanation of sparse attention for a systems engineer. Include one practical tradeoff and one failure mode. Keep it under 120 words.
```

All tested variants produced coherent English on this short prompt:

- base
- ANN 6-layer exact
- ANN all32 exact
- ANN all36 exact
- ANN 6-layer HNSW

The sample file to inspect directly is:

```bash
runtime/results/llama_samples/sample_outputs.md
```

## Long-Context Decode Tests

These tests use synthetic repeated archive prompts. They are useful for decode-throughput measurement but not a strong semantic quality benchmark.

### 16.8k-token prompt

Prompt: `runtime/prompts/long_16k_prompt.txt`

| model | prompt tok/s | decode tok/s | eval ms/token |
|---|---:|---:|---:|
| base | 845.21 | 19.52 | 51.24 |
| ANN 6-layer exact | 834.53 | 18.94 | 52.79 |
| ANN all32 exact | 828.69 | 19.94 | 50.16 |
| ANN all36 exact | 828.55 | 19.81 | 50.47 |

### 33.6k-token prompt

Prompt: `runtime/prompts/long_32k_prompt.txt`

| model | prompt tok/s | decode tok/s | eval ms/token |
|---|---:|---:|---:|
| base | 506.89 | 16.01 | 62.47 |
| ANN 6-layer exact | 504.62 | 16.19 | 61.78 |
| ANN all32 exact | 502.74 | 18.55 | 53.91 |
| ANN all36 exact | 501.92 | 18.91 | 52.87 |

Interpretation:

- At 1k context, ANN substitution does not show speedup because attention is not yet the bottleneck.
- At 33.6k context, the all32 exact path is faster than base on decode (`18.55` vs `16.01` tok/s, about `+16%`).
- Prefill remains full attention for all variants, so total wall time is still dominated by prompt evaluation.
- The 33.6k synthetic repeated prompt produced visibly weaker prefixes for the all32/all36 variants than for base/6-layer. This is consistent with the training results: 32-layer is near-parity but not identical, and 36-layer is lower quality. Proper quality testing needs non-repetitive long-context tasks, not repeated filler text.

## Next Engineering Step

Move from the HNSW correctness bridge to a production runtime:

1. Keep the exact path as the correctness oracle.
2. Replace per-token HNSW rebuilds with persistent per-layer dynamic HNSW indices.
3. Add incremental insertion of decoded search keys into those indices.
4. Expose explicit runtime flags for `ann-search exact|hnsw`, `ann-k`, `ann-m`, and `ann-ef-search` instead of the current `LLAMA_ANN_SEARCH` environment variable.
5. Add a GPU-resident candidate-selection backend for ROCm to remove CPU graph splits.
