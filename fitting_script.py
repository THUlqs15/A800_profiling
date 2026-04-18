#!/usr/bin/env python3
"""
Fitting script for batch latency model -- Route B+ (9-parameter).

Reads profiling data (JSONL), computes aggregate features per batch,
and fits the 9-parameter model with *per-batch* prefill/decode overhead
terms (to capture the amortized weight-loading / kernel-launch cost that
is paid once per batch, not once per request):

  T_pd(f, B) = (1/f) * [w_pf * I{has_prefill}
                        + a_p * sum(l_q^2)
                        + b_p * sum(l_q * l_kv)
                        + c_p * sum(l_q)]
             + (1/f^alpha) * [w_dec * I{has_decode}
                              + a_d * sum(l_kv_decode)
                              + b_d * num_decode]
             + t_c

Where I{has_prefill} = 1 if the batch contains >=1 prefill request (else 0),
and I{has_decode}  = 1 if the batch contains >=1 decode request  (else 0).

This is the "Route B+" form: decomposes the formerly-lumped constant overhead
(t_c ~= 21 ms in the earlier 7-param fit) into (i) a prefill-side amortized
term w_pf/f, (ii) a decode-side amortized term w_dec/f^alpha, and (iii) a
small residual system overhead t_c. Physically, w_pf and w_dec absorb the
per-batch weight-loading / HBM traffic that does NOT grow with batch size.

Fitting strategy:
  - MAPE loss, non-negative constraints on all parameters, alpha in [0.01, 1.0]
  - scipy L-BFGS-B with multi-start (45 starts: 5 alpha x 3 w_pf x 3 w_dec)
  - 80/20 split on a balanced subsample (decode-only capped at 5x prefill+mixed)

Usage:
    python fitting_script.py --input profiling_data.jsonl --output-dir .
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- Data loading & feature extraction ----------

def load_profiling_data(filepath):
    """Load JSONL profiling data."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_features(records):
    """Compute aggregate features for each batch record.

    Returns
    -------
    features : dict of numpy arrays
        f             : GPU graphics clock (MHz)
        sum_lq_sq     : sum of l_q^2 over prefill requests
        sum_lq_lkv    : sum of l_q * l_kv over prefill requests
        sum_lq        : sum of l_q over prefill requests
        sum_lkv_decode: sum of l_kv over decode requests
        num_decode    : number of decode requests in the batch
        has_prefill   : 1.0 if batch has >=1 prefill request, else 0.0
        has_decode    : 1.0 if batch has >=1 decode request, else 0.0
    y : numpy array of wall_time_ms
    btype : numpy array of str ("prefill" / "decode" / "mixed")
    """
    n = len(records)
    f = np.zeros(n)
    sum_lq_sq = np.zeros(n)
    sum_lq_lkv = np.zeros(n)
    sum_lq = np.zeros(n)
    sum_lkv_decode = np.zeros(n)
    num_decode = np.zeros(n)
    has_prefill = np.zeros(n)
    has_decode = np.zeros(n)
    y = np.zeros(n)
    btype = []

    for i, rec in enumerate(records):
        f[i] = rec["gpu_freq_mhz"]
        y[i] = rec["wall_time_ms"]
        types = set()
        for req in rec["requests"]:
            l_q = req["l_q"]
            l_kv = req["l_kv"]
            types.add(req["type"])
            if req["type"] == "prefill":
                sum_lq_sq[i] += l_q ** 2
                sum_lq_lkv[i] += l_q * l_kv
                sum_lq[i] += l_q
            else:
                sum_lkv_decode[i] += l_kv
                num_decode[i] += 1
        has_prefill[i] = 1.0 if "prefill" in types else 0.0
        has_decode[i] = 1.0 if "decode" in types else 0.0
        if types == {"prefill"}:
            btype.append("prefill")
        elif types == {"decode"}:
            btype.append("decode")
        else:
            btype.append("mixed")

    features = {
        "f": f,
        "sum_lq_sq": sum_lq_sq,
        "sum_lq_lkv": sum_lq_lkv,
        "sum_lq": sum_lq,
        "sum_lkv_decode": sum_lkv_decode,
        "num_decode": num_decode,
        "has_prefill": has_prefill,
        "has_decode": has_decode,
    }
    return features, y, np.array(btype)


# ---------- Route B+ model ----------

def predict_routeBplus(params, features):
    """Route B+ prediction. `params` is a 9-tuple
    (a_p, b_p, c_p, w_pf, w_dec, a_d, b_d, alpha, t_c)."""
    a_p, b_p, c_p, w_pf, w_dec, a_d, b_d, alpha, t_c = params
    f = features["f"]
    alpha_c = np.clip(alpha, 0.01, 1.0)
    f_alpha = np.power(f, alpha_c)
    prefill_term = (w_pf * features["has_prefill"]
                    + a_p * features["sum_lq_sq"]
                    + b_p * features["sum_lq_lkv"]
                    + c_p * features["sum_lq"])
    decode_term = (w_dec * features["has_decode"]
                   + a_d * features["sum_lkv_decode"]
                   + b_d * features["num_decode"])
    return prefill_term / f + decode_term / f_alpha + t_c


def compute_mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def fit_routeBplus(features, y):
    """Multi-start L-BFGS-B fit (MAPE loss) for the 9-parameter Route B+ model."""

    def loss(p):
        y_pred = predict_routeBplus(p, features)
        mask = y > 0
        return np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask]))

    bounds = [
        (0.0, None),    # a_p
        (0.0, None),    # b_p
        (0.0, None),    # c_p
        (0.0, None),    # w_pf  (per-batch prefill-side overhead)
        (0.0, None),    # w_dec (per-batch decode-side overhead)
        (0.0, None),    # a_d
        (0.0, None),    # b_d
        (0.01, 1.0),    # alpha  (0 < alpha <= 1 for memory-bound decode)
        (0.0, None),    # t_c    (residual)
    ]

    best, best_loss = None, float("inf")
    for a0 in (0.3, 0.5, 0.7, 0.9, 0.99):
        for wpf0 in (5000.0, 10000.0, 15000.0):
            for wdec0 in (5000.0, 10000.0, 15000.0):
                x0 = [0.002, 0.0, 140.0, wpf0, wdec0, 0.05, 50.0, a0, 5.0]
                res = minimize(loss, x0, method="L-BFGS-B",
                               bounds=bounds,
                               options={"maxiter": 15000, "ftol": 1e-12})
                if res.fun < best_loss:
                    best_loss, best = res.fun, res.x
    return best


def evaluate(params, features, y, label=""):
    y_pred = predict_routeBplus(params, features)
    mape = compute_mape(y, y_pred)
    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))
    if label:
        print(f"\n{label}:")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  MAE  : {mae:.4f} ms")
        print(f"  RMSE : {rmse:.4f} ms")
        print(f"  R^2  : {r2:.6f}")
    return {"mape": mape, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Route B+ batch latency model (9 parameters)")
    parser.add_argument("--input", type=str,
                        default="/home/ubuntu/lqs/profiling_data.jsonl")
    parser.add_argument("--output-dir", type=str,
                        default="/home/ubuntu/lqs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load & basic filter
    print(f"Loading data from {args.input}...")
    records = load_profiling_data(args.input)
    print(f"Loaded {len(records)} records")
    records = [r for r in records
               if r["num_requests"] > 0 and r["gpu_freq_mhz"] > 0]
    print(f"After basic filter: {len(records)} records")

    features_all, y_all, btype_all = compute_features(records)
    unique_freqs = np.unique(features_all["f"])
    n_prefill = int((btype_all == "prefill").sum())
    n_decode = int((btype_all == "decode").sum())
    n_mixed = int((btype_all == "mixed").sum())

    # Data summary
    all_lq = []
    all_lkv = []
    batch_sizes = []
    for rec in records:
        batch_sizes.append(rec["num_requests"])
        for req in rec["requests"]:
            if req["type"] == "prefill":
                all_lq.append(req["l_q"])
            else:
                all_lkv.append(req["l_kv"])
    print("\nData summary:")
    print(f"  Unique frequencies : {len(unique_freqs)}")
    print(f"  Freq range (MHz)   : {unique_freqs.min():.0f} - {unique_freqs.max():.0f}")
    print(f"  Wall time (ms)     : {y_all.min():.2f} - {y_all.max():.2f}")
    print(f"  Prefill batches    : {n_prefill}")
    print(f"  Decode  batches    : {n_decode}")
    print(f"  Mixed   batches    : {n_mixed}")
    if all_lq:
        print(f"  Prefill l_q range  : [{min(all_lq)}, {max(all_lq)}]")
    if all_lkv:
        print(f"  Decode  l_kv range : [{min(all_lkv)}, {max(all_lkv)}]")
    print(f"  Batch size range   : [{min(batch_sizes)}, {max(batch_sizes)}]")

    # Outlier removal 0.5% - 99.5%
    p99 = np.percentile(y_all, 99.5)
    p01 = np.percentile(y_all, 0.5)
    keep = (y_all <= p99) & (y_all >= p01)

    # Balance decode-only to <= 5x (prefill + mixed)
    rng = np.random.RandomState(args.seed)
    idx_p = np.where(keep & (btype_all == "prefill"))[0]
    idx_d = np.where(keep & (btype_all == "decode"))[0]
    idx_m = np.where(keep & (btype_all == "mixed"))[0]
    max_decode = 5 * (len(idx_p) + len(idx_m))
    if len(idx_d) > max_decode:
        idx_d = rng.choice(idx_d, size=max_decode, replace=False)
        print(f"\nBalancing: subsampled decode-only to {len(idx_d)}")
    balanced_idx = np.concatenate([idx_p, idx_d, idx_m])
    rng.shuffle(balanced_idx)

    # 80/20 train/test split
    split = int(0.8 * len(balanced_idx))
    train_idx = balanced_idx[:split]
    test_idx = balanced_idx[split:]
    print(f"\nBalanced: {len(balanced_idx)}  (train={len(train_idx)}, test={len(test_idx)})")

    features_train = {k: v[train_idx] for k, v in features_all.items()}
    features_test = {k: v[test_idx] for k, v in features_all.items()}
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # Fit
    print("\n" + "=" * 60)
    print("Route B+ (9-param) multi-start L-BFGS-B fit")
    print("=" * 60)
    p = fit_routeBplus(features_train, y_train)

    params = {
        "a_p":   float(p[0]),
        "b_p":   float(p[1]),
        "c_p":   float(p[2]),
        "w_pf":  float(p[3]),
        "w_dec": float(p[4]),
        "a_d":   float(p[5]),
        "b_d":   float(p[6]),
        "alpha": float(p[7]),
        "t_c":   float(p[8]),
    }
    print("\nFitted parameters:")
    for k, v in params.items():
        print(f"  {k:6s} = {v:.6e}")

    train_m = evaluate(p, features_train, y_train, "Train (balanced subsample)")
    test_m = evaluate(p, features_test, y_test, "Test  (balanced subsample)")

    # Per-type metrics on the FULL (unfiltered) dataset
    print("\n" + "=" * 60)
    print("Per-type metrics on FULL 101K dataset")
    print("=" * 60)
    y_pred_all = predict_routeBplus(p, features_all)
    per_type = {}
    for bt in ("prefill", "mixed", "decode", "ALL"):
        if bt == "ALL":
            mask = np.ones(len(y_all), dtype=bool)
        else:
            mask = (btype_all == bt)
        yt, yp = y_all[mask], y_pred_all[mask]
        mape = compute_mape(yt, yp)
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))
        med = float(np.median(yt))
        rho, _ = spearmanr(yt, yp)
        per_type[bt] = {
            "count": int(mask.sum()),
            "mape": mape, "mae": mae, "rmse": rmse, "r2": r2,
            "median_wt": med, "spearman_rho": float(rho),
        }
        print(f"  {bt:8s} n={int(mask.sum()):6d}  MAPE={mape:5.2f}%  "
              f"MAE={mae:6.3f}ms  median_wt={med:6.1f}ms  rho={rho:.4f}")

    # Save fitted_params.json
    output_json = os.path.join(args.output_dir, "fitted_params.json")
    out = {
        "model_form": (
            "T_pd(f, B) = (1/f)*(w_pf*I{has_prefill} + a_p*sum(l_q^2) "
            "+ b_p*sum(l_q*l_kv) + c_p*sum(l_q)) + "
            "(1/f^alpha)*(w_dec*I{has_decode} + a_d*sum(l_kv_decode) "
            "+ b_d*num_decode) + t_c"
        ),
        "params": params,
        "train_metrics": {k: v for k, v in train_m.items() if k != "y_pred"},
        "test_metrics": {k: v for k, v in test_m.items() if k != "y_pred"},
        "per_type_full_dataset": per_type,
        "approach": "Route B+ (9-param, per-batch prefill/decode overhead "
                    "+ per-request terms + shared t_c)",
        "data_summary": {
            "total_records": len(records),
            "balanced_filtered": int(len(balanced_idx)),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "unique_frequencies": int(len(unique_freqs)),
            "freq_range": [float(unique_freqs.min()), float(unique_freqs.max())],
            "pure_prefill_batches": n_prefill,
            "pure_decode_batches": n_decode,
            "mixed_batches": n_mixed,
            "prefill_lq_range": [int(min(all_lq)), int(max(all_lq))] if all_lq else None,
            "decode_lkv_range": [int(min(all_lkv)), int(max(all_lkv))] if all_lkv else None,
            "batch_size_range": [int(min(batch_sizes)), int(max(batch_sizes))],
        },
    }
    with open(output_json, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nFitted parameters saved to {output_json}")


if __name__ == "__main__":
    main()
