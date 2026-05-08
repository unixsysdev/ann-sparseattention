# Candidate-Scoring Operation Count

This is an analytic operation-count proxy, not a wall-clock benchmark.
It counts the per-query work to identify candidate keys before running
the sparse attention softmax and value multiply over the selected keys.

## Assumptions

- Native head dimension: `d_head = 128`.
- Learned search dimension: `d_search = 128`.
- Quest page size: `page_size = 16`.
- HNSW parameters: `M = 32`, `ef_search = 64`.

Per-query scoring formulas:

- Full attention: `N * d_head = N * 128`.
- Quest: `(N / page_size) * 2 * d_head = N * 16`.
- Learned HNSW: `M * ef_search * log2(N) * d_search = 262,144 * log2(N)`.

Under these constants, the Quest/HNSW operation-count crossover is approximately `297,937` tokens.
Smaller HNSW settings move the crossover earlier; higher-recall settings move it later.

## Table

| Context | Full ops/query | Quest ops/query | Learned HNSW ops/query | Quest / learned |
|---:|---:|---:|---:|---:|
| 4K | 512,000 | 64,000 | 3,136,759 | 0.02x |
| 8K | 1,024,000 | 128,000 | 3,398,903 | 0.04x |
| 16K | 2,048,000 | 256,000 | 3,661,047 | 0.07x |
| 32K | 4,096,000 | 512,000 | 3,923,191 | 0.13x |
| 64K | 8,192,000 | 1,024,000 | 4,185,335 | 0.24x |
| 128K | 16,384,000 | 2,048,000 | 4,447,479 | 0.46x |
| 256K | 32,768,000 | 4,096,000 | 4,709,623 | 0.87x |
| 512K | 65,536,000 | 8,192,000 | 4,971,767 | 1.65x |
| 1M | 128,000,000 | 16,000,000 | 5,224,942 | 3.06x |
| 2M | 256,000,000 | 32,000,000 | 5,487,086 | 5.83x |
| 4M | 512,000,000 | 64,000,000 | 5,749,230 | 11.13x |

## Interpretation

Quest is cheaper than this high-recall HNSW proxy below the few-hundred-thousand-token regime.
At 1M context, Quest costs about 16M scalar ops/query while learned HNSW costs about 5.2M,
a roughly 3x operation-count advantage for learned projections.

This does not establish production wall-clock speedup. That still requires GPU-resident ANN
retrieval and decode/KV-cache integration. Memory bandwidth may further favor learned ANN at
very long context, but that is not included in this FLOP-only proxy.
