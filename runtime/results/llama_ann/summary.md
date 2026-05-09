# llama.cpp ANN Runtime Smoke Matrix

| backend | model | S cache MiB | prompt tok/s | decode tok/s | eval ms | output prefix |
|---|---:|---:|---:|---:|---:|---|
| cpu | ann_6layer | 9.0 | 183.38 | 10.75 | 1395.86 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of repeated calibration sentences ` |
| cpu | ann_all32 | 9.0 | 184.92 | 10.1 | 1485.6 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text is a series of repeated, generic calibratio` |
| cpu | ann_all36 | 9.0 | 185.28 | 9.84 | 1525.03 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of a series of repetitive calibrat` |
| cpu | base | 0.0 | 205.75 | 10.37 | 1447.1 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of repeated calibration sentences ` |
| gpu | ann_6layer | 9.0 | 1820.46 | 22.99 | 652.32 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of repeated calibration sentences ` |
| gpu | ann_all32 | 9.0 | 1788.14 | 22.0 | 681.92 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text is a series of repetitive calibration sente` |
| gpu | ann_all36 | 9.0 | 1792.84 | 21.83 | 687.17 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of repetitive calibration sentence` |
| gpu | base | 0.0 | 1838.42 | 23.35 | 642.39 | `n_ctx = 1024, n_batch = 2048, n_predict = 16, n_keep = 0   The provided text consists of repeated calibration sentences ` |
