# Long Context Exact Decode Test

Prompt: /home/marcel/SparseAttention/runtime/prompts/long_32k_prompt.txt
Token count: 33601

| model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |
|---|---:|---:|---:|---:|---|
| base | 506.89 | 16.01 | 62.47 | 67263.52 | ` The summary should be concise and avoid technical jargon.  - **Sparse attention**  ` |
| ann_6layer_exact | 504.62 | 16.19 | 61.78 | 67551.54 | ` The summary should be in English and in a formal tone.  - The archive systematically  ` |
| ann_all32_exact | 502.74 | 18.55 | 53.91 | 67678.54 | ` The record of the record of a record of the archive, the archive of the  ` |
| ann_all36_exact | 501.92 | 18.91 | 52.87 | 67772.32 | ` The research archive contains a 111  final       ` |
