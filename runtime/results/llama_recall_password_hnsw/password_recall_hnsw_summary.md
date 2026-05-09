# Password Recall Benchmark: Full Attention vs HNSW ANN

Task: place an exact secret password near the beginning of the prompt, add irrelevant filler, then ask the model to output only the password.

Base uses normal full attention. ANN variants use `LLAMA_ANN_SEARCH=hnsw`, i.e. approximate ANN retrieval through the current HNSW bridge.

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

## Exact Outputs and Memory Lines

### rocm / base_full / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RIVER [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93322 + (9126 =  7672 +    1152 +     301) +       24527 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_6layer_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RIVER [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93299 + (9205 =  7680 +    1224 +     301) +       24470 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all32_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RENT
VIOLET-7319-RIVER
VIOLET
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93826 + (9238 =  7712 +    1224 +     301) +       23911 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all36_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not explain.

VIOLET-731173-71 RIVER

VIOLET-731173.
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93638 + (9243 =  7717 +    1224 +     301) +       24093 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / base_full / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORBIT-4826-LANTERN [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93810 + (9126 =  7672 +    1152 +     301) +       24039 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_6layer_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORBIT-4826-LANTERN [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93788 + (9205 =  7680 +    1224 +     301) +       23981 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all32_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORR-4826, ORBIT-4826, ORR- note, OR, this is the
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93940 + (9238 =  7712 +    1224 +     301) +       23797 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all36_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not respond with only thing that the following is. memset system is irrelevant.ergency of the following:ity of the following: 1 The is irrelevant.
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93762 + (9243 =  7717 +    1224 +     301) +       23969 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / base_full / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not add any extra text.

CIPHER-9051-MARBLE [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93850 + (9126 =  7672 +    1152 +     301) +       23999 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_6layer_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not add extra text.

CIPHER-9051-MARBLE [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93824 + (9205 =  7680 +    1224 +     301) +       23945 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all32_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not add extra words.

Answer the secret password is: CIPHER-9051- 1234- new line.  1
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93369 + (9238 =  7712 +    1224 +     301) +       24367 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### rocm / ann_all36_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not than the fills the  Continue to continuing. 11 paragraphs the first line 111 11 11 1 1
```

Memory lines:
```text
common_memory_breakdown_print: |   - Host                   |                     767 =   741 +       0 +      26                |
common_memory_breakdown_print: | memory breakdown [MiB]     |  total    free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - ROCm0 (8060S Graphics) | 126976 = 93483 + (9243 =  7717 +    1224 +     301) +       24249 |
common_memory_breakdown_print: |   - Host                   |                    767 =   741 +       0 +      26                |
```

### cpu / base_full / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RIVER [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
```

### cpu / ann_6layer_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RIVER [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
```

### cpu / ann_all32_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not add extra text.

VIOLET-7319-RIVER

The correct password is: VIOLET-7319-RIVER
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
```

### cpu / ann_all36_hnsw / target 1024

Expected: `VIOLET-7319-RIVER`

```text
Do not infer anything else.

VIOLET-7319-RIVR

VIOLET-73119-RIVER

VIO
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
```

### cpu / base_full / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORBIT-4826-LANTERN [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
```

### cpu / ann_6layer_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORBIT-4826-LANTERN [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
```

### cpu / ann_all32_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not add extra text.

ORBIT-4826-EXACTLY-1234-1234-1234
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
```

### cpu / ann_all36_hnsw / target 2048

Expected: `ORBIT-4826-LANTERN`

```text
Do not respond to the password.

ORBIT-
 The secret is the secret password is. 2

OR: 4...
 not the
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
```

### cpu / base_full / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not add any extra text.

CIPHER-9051-MARBLE [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9136 =  7672 +    1152 +     311                |
```

### cpu / ann_6layer_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not add extra text.

CIPHER-9051-MARBLE [end of text]
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9215 =  7680 +    1224 +     311                |
```

### cpu / ann_all32_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not include any further.

The secret password is: CIPHER-1. The password is not to be ignored. The question. The prompt is. The
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9248 =  7712 +    1224 +     311                |
```

### cpu / ann_all36_hnsw / target 4096

Expected: `CIPHER-9051-MARBLE`

```text
Do not forget the following
 aergency

fill in filling the password. fill

...ergency

The secret is the C memory test.  1...
```

Memory lines:
```text
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Host               |                 9253 =  7717 +    1224 +     311                |
```
