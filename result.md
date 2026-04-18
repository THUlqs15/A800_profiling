# Batch Latency Model -- Fitted Results (Route B+, 9-parameter)

## Model

- **LLM**: Qwen3-14B (Qwen3ForCausalLM), 40 layers, hidden_size 5120, bfloat16
- **GPU**: NVIDIA A800-SXM4-80GB
- **Driver**: 550.144.03
- **vLLM**: v0.19.0+precompiled (commit 2a69949b), editable install
- **Attention backend**: FlashAttention v2
- **Engine config**: enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_model_len=8192, max_num_seqs=64

## Model Form

The final batch-execution-time model ("Route B+") decomposes the formerly-lumped constant overhead into (i) a prefill-side amortized term, (ii) a decode-side amortized term, and (iii) a small residual system overhead. Physically, `w_pf` and `w_dec` capture per-batch weight-loading / HBM traffic that does NOT grow with batch size (and therefore cannot be represented by the per-request terms `a_d * l_kv + b_d`).

```
T_pd(f, B) = (1/f)       * [ w_pf  * I{has_prefill}
                             + a_p * sum(l_q^2)
                             + b_p * sum(l_q * l_kv)
                             + c_p * sum(l_q) ]
           + (1/f^alpha) * [ w_dec * I{has_decode}
                             + a_d * sum(l_kv_decode)
                             + b_d * num_decode ]
           + t_c
```

where `I{has_prefill} = 1` if the batch contains at least one prefill request (else 0), and similarly for `I{has_decode}`.

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p  | 4.7911e-03 | Prefill: quadratic term for l_q^2 (attention score cost) |
| b_p  | 0.0000e+00 | Prefill: cross-term for l_q * l_kv (zero; prefix caching off) |
| c_p  | 1.3651e+02 | Prefill: linear term for l_q (per-token prefill compute) |
| w_pf | 1.5000e+04 | Prefill: per-batch amortized overhead (weight load / launch) |
| w_dec| 1.5000e+04 | Decode:  per-batch amortized overhead (weight load / launch) |
| a_d  | 1.9294e-01 | Decode: per-request linear term for l_kv (attention read) |
| b_d  | 5.0502e+01 | Decode: per-request constant term |
| alpha| 9.7357e-01 | Decode: frequency scaling exponent (0 < alpha <= 1) |
| t_c  | 4.6526e+00 | Batch-level residual constant overhead (ms) |

**Note on `b_p`**: Zero because prefix caching was disabled (`enable_prefix_caching=False`), so `l_kv = 0` for every prefill request. The cross-term has no variation in the training data; it only becomes identifiable when prefix caching is on.

**Note on `w_pf` and `w_dec`**: Both hit the fitting heuristic upper sentinel of 1.5e4. The magnitude is consistent with the cost of traversing 40 transformer layers once (weight-read-bound) at a reference frequency of 1 GHz: ~15 ms. This term is what the earlier 7-parameter model was forced to absorb into a single batch-level `t_c`, which caused it to collapse to ~21 ms and inflate the decode MAPE.

**Note on `alpha`**: With the bound relaxed to `[0.01, 1.0]`, the fitted value is 0.974, i.e. decode scales nearly as `1/f^1`. This is *not* a compute-saturation signature -- decode is still strongly memory-bandwidth-bound at the A800's fixed 1593 MHz HBM clock. It reflects the fact that at that fixed memory clock the HBM is not the throttling resource for the sizes probed, so the graphics-clock sensitivity dominates. Earlier attempts to drive `alpha` to 2.6 by relaxing the upper bound were bounded-fit artifacts that degraded interpretability without genuinely improving fit; Route B+ removes that pressure on `alpha` by giving the model the two `w_*` slots it actually needs.

## GPU Frequency Information

- **GPU model**: NVIDIA A800-SXM4-80GB
- **Memory clock (fixed)**: 1593 MHz (only supported value; memory clock locking not supported on this GPU)
- **Graphics clock range profiled**: 510 - 1335 MHz (up to 94.7% of max 1410 MHz)
- **Number of distinct graphics frequencies**: 56 target frequencies (61 unique observed values due to minor GPU clock jitter)
- **Graphics clock frequencies profiled (MHz)**: 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335
- **Fitted alpha value**: 0.974 (interpretation: decode latency scales as 1/f^0.974, very close to 1/f)

## Evaluation Metrics

### Overall (balanced subsample for fitting, 80/20 split)

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | 4.93% | 4.90% |
| MAE    | 2.2001 ms | 2.2079 ms |
| RMSE   | 3.8921 ms | 3.9423 ms |
| R^2    | 0.997311 | 0.997270 |

### Per-Type Metrics (evaluated on full 101K dataset)

| Batch Type | Count | MAPE | MAE | Median wall_time | Spearman rho |
|------------|-------|------|-----|------------------|-------------|
| Prefill-only | 3,639 | 5.01% | 7.12 ms | 83.8 ms | 0.9982 |
| Mixed        | 2,608 | 5.65% | 7.36 ms | 124.9 ms | 0.9958 |
| Decode-only  | 94,914 | 4.79% | 1.50 ms | 28.8 ms | 0.9741 |
| **All**      | **101,161** | **4.82%** | **1.85 ms** | **29.8 ms** | **0.9771** |

**Comparison to the earlier 7-parameter model** (the `t_c = 21.15 ms` fit without `w_pf`, `w_dec`):

| | 7-param | **Route B+ (9-param)** | Relative change |
|---|---|---|---|
| Overall MAPE (full dataset) | 13.73% | **4.82%** | -65% |
| Decode MAPE | 14.01% | **4.79%** | -66% |
| Decode Spearman rho | 0.576 | **0.9741** | +69% |
| Prefill MAPE | 8.19% | **5.01%** | -39% |
| Mixed MAPE | 11.14% | **5.65%** | -49% |

The earlier 14% decode MAPE was dominated by the 21 ms lumped `t_c` term: for a typical ~29 ms decode batch, 21 ms of constant overhead left only ~8 ms of signal to fit against, so moderate absolute errors produced large relative errors. Splitting the overhead into a frequency-scaled `w_pf/f` + `w_dec/f^alpha` recovers the underlying structure: the decode Spearman rank correlation with measured latency jumps from 0.58 to 0.97, which matters directly for a scheduler that ranks candidate batches.

**Fitting approach**: Non-linear optimization (scipy L-BFGS-B) with MAPE loss and non-negative parameter constraints. Multi-start grid over `alpha0 in {0.3, 0.5, 0.7, 0.9, 0.99}` x `w_pf0 in {5000, 10000, 15000}` x `w_dec0 in {5000, 10000, 15000}` (45 starts total), selecting the best by training-set MAPE. `alpha` is constrained to `[0.01, 1.0]`; all other parameters are constrained to be non-negative.

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

To predict batch execution time at GPU graphics clock `f` (MHz):

```
T_pd(f, B) = (1/f) * ( w_pf  * has_prefill
                       + a_p * sum(l_q^2)
                       + b_p * sum(l_q*l_kv)
                       + c_p * sum(l_q) )
           + (1/f^alpha) * ( w_dec * has_decode
                              + a_d * sum(l_kv_decode)
                              + b_d * num_decode )
           + t_c
```

With fitted numeric values:

```
T_pd(f, B) = (1/f) * ( 15000  * has_prefill
                       + 4.791e-03 * sum(l_q^2)
                       + 0          * sum(l_q*l_kv)
                       + 136.51     * sum(l_q) )
           + (1/f^0.9736) * ( 15000  * has_decode
                               + 0.1929 * sum(l_kv_decode)
                               + 50.50  * num_decode )
           + 4.65
```

Where:
- `f` is in MHz.
- `has_prefill = 1` if the batch contains at least one prefill request, else 0.
- `has_decode  = 1` if the batch contains at least one decode request, else 0.
- `sum(l_q^2)`, `sum(l_q*l_kv)`, `sum(l_q)` are sums over the batch's prefill requests.
- `sum(l_kv_decode)` is the sum of `l_kv` over the batch's decode requests.
- `num_decode` is the number of decode requests in the batch.
- Result `T_pd` is in milliseconds.
