# Batch Latency Model -- Fitted Results

## Model

- **LLM**: Qwen3-14B (Qwen3ForCausalLM), 40 layers, hidden_size 5120, bfloat16
- **GPU**: NVIDIA A800-SXM4-80GB
- **Driver**: 550.144.03
- **vLLM**: v0.19.0+precompiled (commit 2a69949b), editable install
- **Attention backend**: FlashAttention v2
- **Engine config**: enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_model_len=8192, max_num_seqs=64

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p | 2.1780390205e-03 | Prefill: quadratic term coefficient for l_q^2 |
| b_p | 0.0000000000e+00 | Prefill: cross-term coefficient for l_q x l_kv |
| c_p | 1.4042626471e+02 | Prefill: linear term coefficient for l_q |
| a_d | 7.9332422620e-02 | Decode: linear term coefficient for l_kv |
| b_d | 2.0226741536e+02 | Decode: constant term per decode request |
| alpha | 9.9000000000e-01 | Decode: frequency scaling exponent (0 < alpha < 1) |
| t_c | 2.1146895649e+01 | Batch-level constant overhead (ms) |

**Note on b_p**: This parameter is zero because prefix caching was disabled (`enable_prefix_caching=False`), meaning `l_kv = 0` for all prefill requests. The `l_q * l_kv` cross-term has no variation in the training data. This is expected -- the term only becomes relevant when prefix caching is enabled.

**Note on alpha**: The fitted value of alpha = 0.99 is close to 1.0, indicating that on this GPU (A800-SXM4-80GB) with fixed memory clock at 1593 MHz, decode latency scales nearly inversely with graphics clock frequency, similar to prefill. This suggests that at the tested frequency range (510-1335 MHz), the memory bandwidth bottleneck is not strongly decoupled from the graphics clock.

## GPU Frequency Information

- **GPU model**: NVIDIA A800-SXM4-80GB
- **Memory clock (fixed)**: 1593 MHz (only supported value; memory clock locking not supported on this GPU)
- **Graphics clock range profiled**: 510 - 1335 MHz (up to 94.7% of max 1410 MHz)
- **Number of distinct graphics frequencies**: 56 target frequencies (61 unique observed values due to minor GPU clock jitter)
- **Graphics clock frequencies profiled (MHz)**: 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335
- **Fitted alpha value**: 0.99 (interpretation: decode latency scales as 1/f^0.99, nearly 1/f)

## Evaluation Metrics

### Overall (balanced subsample for fitting, 80/20 split)

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | 13.35% | 13.15% |
| MAE    | 5.7480 ms | 5.6751 ms |
| RMSE   | 8.6401 ms | 8.6100 ms |
| R^2    | 0.986792 | 0.986808 |

### Per-Type MAPE (evaluated on full 101K dataset)

| Batch Type | Count | MAPE | MAE | Median wall_time |
|------------|-------|------|-----|-----------------|
| Prefill-only | 3,639 | 8.19% | 7.77 ms | 83.8 ms |
| Mixed | 2,608 | 11.14% | 16.57 ms | 124.9 ms |
| Decode-only | 94,914 | 14.01% | 4.85 ms | 28.8 ms |
| **All** | **101,161** | **13.73%** | **5.26 ms** | **29.8 ms** |

**Fitting approach**: Non-linear optimization (scipy L-BFGS-B) with MAPE loss and non-negative parameter constraints, initialized from grid search over alpha (step 0.005) with NNLS regression.

**Note on decode MAPE**: Decode-only MAPE (14.01%) is inflated because most decode batches have wall_time ~25-30 ms, where the constant overhead t_c (~21 ms) dominates. An absolute error of 4.85 ms on a 29 ms batch yields ~17% relative error. The actual decode time beyond overhead is only ~4-8 ms, and l_kv has minimal impact on single-decode latency on this hardware (l_kv from 82 to 1290 changes wall_time by only ~0.8 ms at 900 MHz).

## Data Summary

- **Total batches profiled**: 101,161
- **Balanced training set**: 34,446 (after outlier removal + decode subsampling at 5x)
- **Train set size**: 27,556
- **Test set size**: 6,890
- **GPU graphics frequency range**: 510 - 1335 MHz (61 unique observed values)
- **Fixed memory clock**: 1593 MHz
- **Pure prefill batches**: 3,639
- **Pure decode batches**: 94,914
- **Mixed batches**: 2,608
- **Prefill l_q range**: [32, 4096]
- **Decode l_kv range**: [32, 2846]
- **Batch size range**: [1, 64]
- **Wall time range**: 19.67 - 4335.71 ms

## Usage

To predict batch execution time at GPU frequency f (MHz):

```
T_pd(f, B) = (1/f) * (a_p * sum_lq_sq + b_p * sum_lq_lkv + c_p * sum_lq)
           + (1/f^alpha) * (a_d * sum_lkv_decode + b_d * num_decode)
           + t_c
```

With fitted numeric values:

```
T_pd(f, B) = (1/f) * (2.178e-03 * sum(l_q^2) + 0 * sum(l_q*l_kv) + 140.43 * sum(l_q))
           + (1/f^0.99) * (0.0793 * sum(l_kv_decode) + 202.27 * num_decode)
           + 21.15
```

Where:
- `f` is in MHz
- `sum(l_q^2)` = sum of l_q^2 over all prefill requests in the batch
- `sum(l_q*l_kv)` = sum of l_q * l_kv over all prefill requests (zero when prefix caching is off)
- `sum(l_q)` = sum of l_q over all prefill requests
- `sum(l_kv_decode)` = sum of l_kv over all decode requests
- `num_decode` = number of decode requests in the batch
- Result `T_pd` is in milliseconds
