# llama.cpp Learned Sparse Attention HNSW Smoke Matrix

Prompt: /home/marcel/SparseAttention/runtime/prompts/long_prompt.txt

| backend | model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---:|---|
| CPU | ann_6layer | 206.90 | 5.82 | 171.75 | 4364.30 | ` The provided text consists  ` |
| CPU | ann_all32 | 198.52 | 2.33 | 428.78 | 5297.88 | ` The cache is filled  ` |
| CPU | ann_all36 | 200.72 | 2.13 | 470.39 | 5378.84 | ` The given text is  ` |
| ROCm | ann_6layer | 1800.32 | 9.25 | 108.09 | 768.59 | ` The provided text consists  ` |
| ROCm | ann_all32 | 1857.50 | 2.66 | 376.22 | 1559.60 | ` The cache is filled  ` |
| ROCm | ann_all36 | 1834.51 | 2.40 | 417.16 | 1687.48 | ` The sentences listed above  ` |
