# Long Context Exact Decode Test

Prompt: /home/marcel/SparseAttention/runtime/prompts/long_16k_prompt.txt
Token count: 16811

| model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---|
| base | 845.21 | 19.52 | 51.24 | 21505.49 | ` Each bullet point should be concise and directly relevant to the research.  - **Sparse Attention**: The archive explore` |
| ann_6layer_exact | 834.53 | 18.94 | 52.79 | 21807.23 | ` Each bullet point must be exactly 100 words. Do not include any markdown, do not include any headers or titles, do not ` |
| ann_all32_exact | 828.69 | 19.94 | 50.16 | 21864.02 | ` Each bullet point should be no more than 20 words.  - The pattern of sparse attention is used to generate a summary.  W` |
| ann_all36_exact | 828.55 | 19.81 | 50.47 | 21877.34 | ` Each bullet point is a           .          0 .    ` |
