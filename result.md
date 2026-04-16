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
| a_p | 1.6575068145e-03 | Prefill: quadratic term coefficient for l_q^2 |
| b_p | 1.0373923942e-12 | Prefill: cross-term coefficient for l_q x l_kv |
| c_p | 1.4215956444e+02 | Prefill: linear term coefficient for l_q |
| a_d | 2.5172785109e-01 | Decode: linear term coefficient for l_kv |
| b_d | 3.5959726078e+02 | Decode: constant term per decode request |
| alpha | 9.9500000000e-01 | Decode: frequency scaling exponent (0 < alpha < 1) |
| t_c | 2.0273290124e+01 | Batch-level constant overhead (ms) |

**Note on b_p**: This parameter is effectively zero because prefix caching was disabled (`enable_prefix_caching=False`), meaning `l_kv = 0` for all prefill requests. The `l_q * l_kv` cross-term has no variation in the training data, so `b_p` is unconstrained. This is expected behavior -- the term only becomes relevant when prefix caching is enabled.

**Note on alpha**: The fitted value of alpha = 0.995 is close to 1.0, indicating that on this GPU (A800-SXM4-80GB) with fixed memory clock at 1593 MHz, decode latency scales nearly inversely with graphics clock frequency, similar to prefill. This suggests that at the tested frequency range (510-1335 MHz), the memory bandwidth bottleneck is not strongly decoupled from the graphics clock.

## GPU Frequency Information

- **GPU model**: NVIDIA A800-SXM4-80GB
- **Memory clock (fixed)**: 1593 MHz (only supported value; memory clock locking not supported on this GPU)
- **Graphics clock range profiled**: 510 - 1335 MHz (up to 94.7% of max 1410 MHz)
- **Number of distinct graphics frequencies**: 56
- **Graphics clock frequencies profiled (MHz)**: 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335
- **Fitted alpha value**: 0.995 (interpretation: decode latency scales as 1/f^0.995, nearly 1/f)

## Evaluation Metrics

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | 12.81% | 12.74% |
| MAE    | 5.2110 ms | 5.1938 ms |
| RMSE   | 7.0834 ms | 7.0478 ms |
| R^2    | 0.997716 | 0.997859 |

**Fitting approach**: Grid search over alpha (step 0.005 in [0.01, 1.0)) + Linear Regression (best of two approaches evaluated; non-linear L-BFGS-B refinement produced marginally worse test MAPE of 12.80%).

## Data Summary

- **Total batches profiled**: 40,345
- **After outlier removal (0.5%-99.5%)**: 39,941
- **Train set size**: 31,952
- **Test set size**: 7,989
- **GPU graphics frequency range**: 510 - 1335 MHz
- **Fixed memory clock**: 1593 MHz
- **Number of distinct graphics frequencies**: 56
- **Pure prefill batches**: 9,271
- **Pure decode batches**: 27,944
- **Mixed batches**: 3,130
- **Prefill l_q range**: [32, 4096]
- **Decode l_kv range**: [32, 2126]
- **Batch size range**: [1, 48]
- **Wall time range**: 19.87 - 4328.34 ms

## Usage

To predict batch execution time at GPU frequency f (MHz):

```
T_pd(f, B) = (1/f) * (a_p * sum_lq_sq + b_p * sum_lq_lkv + c_p * sum_lq)
           + (1/f^alpha) * (a_d * sum_lkv_decode + b_d * num_decode)
           + t_c
```

With fitted numeric values:

```
T_pd(f, B) = (1/f) * (1.6575e-03 * sum(l_q^2) + 1.037e-12 * sum(l_q*l_kv) + 142.16 * sum(l_q))
           + (1/f^0.995) * (0.2517 * sum(l_kv_decode) + 359.60 * num_decode)
           + 20.27
```

Where:
- `f` is in MHz
- `sum(l_q^2)` = sum of l_q^2 over all prefill requests in the batch
- `sum(l_q*l_kv)` = sum of l_q * l_kv over all prefill requests
- `sum(l_q)` = sum of l_q over all prefill requests
- `sum(l_kv_decode)` = sum of l_kv over all decode requests
- `num_decode` = number of decode requests in the batch
- Result `T_pd` is in milliseconds
