#!/usr/bin/env python3
"""
Fitting script for batch latency model.

Reads profiling data (JSONL), computes aggregate features per batch,
and fits the 7-parameter model:

T_pd(f, B) = (1/f) * [a_p * sum_lq_sq + b_p * sum_lq_lkv + c_p * sum_lq]
           + (1/f^alpha) * [a_d * sum_lkv_decode + b_d * num_decode]
           + t_c

Uses two approaches:
  1. Grid search over alpha + linear regression
  2. Non-linear optimization (scipy) for refinement

Usage:
    python fitting_script.py --input profiling_data.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_profiling_data(filepath):
    """Load JSONL profiling data and compute aggregate features per batch."""
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

    Returns:
        features: dict with numpy arrays for each feature
        y: measured wall_time_ms
    """
    n = len(records)
    f = np.zeros(n)          # GPU freq in MHz
    sum_lq_sq = np.zeros(n)  # sum of l_q^2 for prefill
    sum_lq_lkv = np.zeros(n) # sum of l_q * l_kv for prefill
    sum_lq = np.zeros(n)     # sum of l_q for prefill
    sum_lkv_decode = np.zeros(n)  # sum of l_kv for decode
    num_decode = np.zeros(n)      # count of decode requests
    y = np.zeros(n)          # measured time in ms

    for i, rec in enumerate(records):
        f[i] = rec["gpu_freq_mhz"]
        y[i] = rec["wall_time_ms"]
        for req in rec["requests"]:
            l_q = req["l_q"]
            l_kv = req["l_kv"]
            if req["type"] == "prefill":
                sum_lq_sq[i] += l_q ** 2
                sum_lq_lkv[i] += l_q * l_kv
                sum_lq[i] += l_q
            else:  # decode
                sum_lkv_decode[i] += l_kv
                num_decode[i] += 1

    return {
        "f": f,
        "sum_lq_sq": sum_lq_sq,
        "sum_lq_lkv": sum_lq_lkv,
        "sum_lq": sum_lq,
        "sum_lkv_decode": sum_lkv_decode,
        "num_decode": num_decode,
    }, y


def build_design_matrix(features, alpha):
    """Build the linear design matrix for a given alpha value.

    Columns: [sum_lq_sq/f, sum_lq_lkv/f, sum_lq/f,
              sum_lkv_decode/f^alpha, num_decode/f^alpha, 1]
    """
    f = features["f"]
    f_alpha = np.power(f, alpha)

    X = np.column_stack([
        features["sum_lq_sq"] / f,       # a_p term
        features["sum_lq_lkv"] / f,      # b_p term
        features["sum_lq"] / f,          # c_p term
        features["sum_lkv_decode"] / f_alpha,  # a_d term
        features["num_decode"] / f_alpha,      # b_d term
        np.ones(len(f)),                       # t_c term
    ])
    return X


def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def approach1_grid_search(features_train, y_train, features_test, y_test):
    """Grid search over alpha + non-negative linear regression."""
    from sklearn.linear_model import LinearRegression
    best_mape = float("inf")
    best_alpha = None
    best_coefs = None

    alphas = np.arange(0.01, 1.0, 0.005)

    for alpha in alphas:
        X_train = build_design_matrix(features_train, alpha)
        X_test = build_design_matrix(features_test, alpha)

        # Use non-negative least squares for physically meaningful params
        from scipy.optimize import nnls
        coefs, _ = nnls(X_train, y_train)

        y_pred_test = X_test @ coefs
        mape = compute_mape(y_test, y_pred_test)

        if mape < best_mape:
            best_mape = mape
            best_alpha = alpha
            best_coefs = coefs

    params = {
        "a_p": best_coefs[0],
        "b_p": best_coefs[1],
        "c_p": best_coefs[2],
        "a_d": best_coefs[3],
        "b_d": best_coefs[4],
        "t_c": best_coefs[5],
        "alpha": best_alpha,
    }

    return params, best_coefs, best_mape


def approach2_nonlinear(features, y, initial_params):
    """Refine parameters using scipy non-linear optimization."""
    f = features["f"]

    def predict(params):
        a_p, b_p, c_p, a_d, b_d, alpha, t_c = params
        alpha = np.clip(alpha, 0.01, 0.99)
        f_alpha = np.power(f, alpha)
        return (
            (1 / f) * (a_p * features["sum_lq_sq"]
                       + b_p * features["sum_lq_lkv"]
                       + c_p * features["sum_lq"])
            + (1 / f_alpha) * (a_d * features["sum_lkv_decode"]
                               + b_d * features["num_decode"])
            + t_c
        )

    def loss(params):
        y_pred = predict(params)
        # Use MAPE as loss for better percentage-based optimization
        mask = y > 0
        return np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask]))

    x0 = [
        initial_params["a_p"],
        initial_params["b_p"],
        initial_params["c_p"],
        initial_params["a_d"],
        initial_params["b_d"],
        initial_params["alpha"],
        initial_params["t_c"],
    ]

    bounds = [
        (0, None),      # a_p >= 0
        (0, None),      # b_p >= 0
        (0, None),      # c_p >= 0
        (0, None),      # a_d >= 0
        (0, None),      # b_d >= 0
        (0.01, 0.99),   # alpha
        (0, None),      # t_c >= 0
    ]

    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 10000, "ftol": 1e-12})

    params = {
        "a_p": result.x[0],
        "b_p": result.x[1],
        "c_p": result.x[2],
        "a_d": result.x[3],
        "b_d": result.x[4],
        "alpha": result.x[5],
        "t_c": result.x[6],
    }
    return params, result


def evaluate_model(params, features, y, label=""):
    """Evaluate model on given data, return metrics dict."""
    f = features["f"]
    alpha = params["alpha"]
    f_alpha = np.power(f, alpha)

    y_pred = (
        (1 / f) * (params["a_p"] * features["sum_lq_sq"]
                    + params["b_p"] * features["sum_lq_lkv"]
                    + params["c_p"] * features["sum_lq"])
        + (1 / f_alpha) * (params["a_d"] * features["sum_lkv_decode"]
                           + params["b_d"] * features["num_decode"])
        + params["t_c"]
    )

    mape = compute_mape(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    if label:
        print(f"\n{label}:")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  MAE:   {mae:.4f} ms")
    print(f"  RMSE:  {rmse:.4f} ms")
    print(f"  R²:    {r2:.6f}")

    return {"mape": mape, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def main():
    parser = argparse.ArgumentParser(description="Fit batch latency model")
    parser.add_argument("--input", type=str,
                        default="/home/ubuntu/lqs/profiling_data.jsonl",
                        help="Input JSONL profiling data file")
    parser.add_argument("--output-dir", type=str,
                        default="/home/ubuntu/lqs",
                        help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print(f"Loading data from {args.input}...")
    records = load_profiling_data(args.input)
    print(f"Loaded {len(records)} batch records")

    # Filter out batches with zero requests (warmup edge cases)
    records = [r for r in records if r["num_requests"] > 0 and r["gpu_freq_mhz"] > 0]
    print(f"After filtering: {len(records)} records")

    # Compute features
    features, y = compute_features(records)

    # Print data summary
    unique_freqs = np.unique(features["f"])
    print(f"\nData Summary:")
    print(f"  Unique GPU frequencies: {len(unique_freqs)}")
    print(f"  Frequency range: {unique_freqs.min():.0f} - {unique_freqs.max():.0f} MHz")
    print(f"  Wall time range: {y.min():.2f} - {y.max():.2f} ms")

    # Count batch types
    n_prefill_only = 0
    n_decode_only = 0
    n_mixed = 0
    for rec in records:
        types = set(r["type"] for r in rec["requests"])
        if types == {"prefill"}:
            n_prefill_only += 1
        elif types == {"decode"}:
            n_decode_only += 1
        else:
            n_mixed += 1
    print(f"  Pure prefill batches: {n_prefill_only}")
    print(f"  Pure decode batches:  {n_decode_only}")
    print(f"  Mixed batches:        {n_mixed}")

    # Collect l_q and l_kv ranges
    all_lq_prefill = []
    all_lkv_decode = []
    batch_sizes = []
    for rec in records:
        batch_sizes.append(rec["num_requests"])
        for req in rec["requests"]:
            if req["type"] == "prefill":
                all_lq_prefill.append(req["l_q"])
            else:
                all_lkv_decode.append(req["l_kv"])
    if all_lq_prefill:
        print(f"  Prefill l_q range: [{min(all_lq_prefill)}, {max(all_lq_prefill)}]")
    if all_lkv_decode:
        print(f"  Decode l_kv range: [{min(all_lkv_decode)}, {max(all_lkv_decode)}]")
    print(f"  Batch size range: [{min(batch_sizes)}, {max(batch_sizes)}]")

    # Filter outliers (e.g., very first batch at each frequency may be warm-up)
    # Remove batches with wall_time > 99th percentile (likely startup artifacts)
    p99 = np.percentile(y, 99.5)
    p01 = np.percentile(y, 0.5)
    mask = (y <= p99) & (y >= p01)
    records_filtered = [r for r, m in zip(records, mask) if m]

    # Balance dataset: subsample decode-only to avoid drowning prefill/mixed
    rng = np.random.RandomState(args.seed)
    prefill_recs = [r for r in records_filtered
                    if set(req["type"] for req in r["requests"]) == {"prefill"}]
    decode_recs = [r for r in records_filtered
                   if set(req["type"] for req in r["requests"]) == {"decode"}]
    mixed_recs = [r for r in records_filtered
                  if len(set(req["type"] for req in r["requests"])) > 1]

    # Cap decode-only at 5x the size of (prefill + mixed) combined
    max_decode = 5 * (len(prefill_recs) + len(mixed_recs))
    if len(decode_recs) > max_decode:
        idx = rng.choice(len(decode_recs), size=max_decode, replace=False)
        decode_recs = [decode_recs[i] for i in idx]
        print(f"\nBalancing: subsampled decode-only from {n_decode_only} to {len(decode_recs)}")

    records_filtered = prefill_recs + decode_recs + mixed_recs
    rng.shuffle(records_filtered)

    features_filtered, y_filtered = compute_features(records_filtered)
    print(f"\nAfter outlier removal (0.5%-99.5%): {len(records_filtered)} records")

    # Split into train/test (80/20)
    n = len(y_filtered)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Split features
    features_train = {k: v[train_idx] for k, v in features_filtered.items()}
    features_test = {k: v[test_idx] for k, v in features_filtered.items()}
    y_train = y_filtered[train_idx]
    y_test = y_filtered[test_idx]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Approach 1: Grid search over alpha
    print("\n" + "=" * 60)
    print("Approach 1: Grid search over alpha + Linear Regression")
    print("=" * 60)

    params1, _, mape1 = approach1_grid_search(
        features_train, y_train, features_test, y_test)

    print(f"\nBest alpha: {params1['alpha']:.4f}")
    print(f"Parameters:")
    for k, v in params1.items():
        print(f"  {k}: {v:.10e}")

    train_metrics1 = evaluate_model(params1, features_train, y_train, "Train metrics")
    test_metrics1 = evaluate_model(params1, features_test, y_test, "Test metrics")

    # Approach 2: Non-linear refinement
    print("\n" + "=" * 60)
    print("Approach 2: Non-linear optimization (scipy L-BFGS-B)")
    print("=" * 60)

    # Fit on training data
    params2, opt_result = approach2_nonlinear(features_train, y_train, params1)

    print(f"\nOptimizer converged: {opt_result.success}")
    print(f"Parameters:")
    for k, v in params2.items():
        print(f"  {k}: {v:.10e}")

    train_metrics2 = evaluate_model(params2, features_train, y_train, "Train metrics")
    test_metrics2 = evaluate_model(params2, features_test, y_test, "Test metrics")

    # Choose the better approach based on test MAPE
    if test_metrics2["mape"] < test_metrics1["mape"]:
        best_params = params2
        best_train = train_metrics2
        best_test = test_metrics2
        best_approach = "Approach 2 (Non-linear optimization)"
    else:
        best_params = params1
        best_train = train_metrics1
        best_test = test_metrics1
        best_approach = "Approach 1 (Grid search + Linear Regression)"

    print(f"\n{'=' * 60}")
    print(f"Best approach: {best_approach}")
    print(f"Test MAPE: {best_test['mape']:.2f}%")
    print(f"{'=' * 60}")

    # Save fitted parameters to JSON
    output_json = os.path.join(args.output_dir, "fitted_params.json")
    with open(output_json, "w") as f:
        json.dump({
            "params": best_params,
            "train_metrics": {k: v for k, v in best_train.items() if k != "y_pred"},
            "test_metrics": {k: v for k, v in best_test.items() if k != "y_pred"},
            "approach": best_approach,
            "data_summary": {
                "total_records": len(records),
                "filtered_records": len(records_filtered),
                "train_size": len(y_train),
                "test_size": len(y_test),
                "unique_frequencies": len(unique_freqs),
                "freq_range": [float(unique_freqs.min()), float(unique_freqs.max())],
                "pure_prefill_batches": n_prefill_only,
                "pure_decode_batches": n_decode_only,
                "mixed_batches": n_mixed,
                "prefill_lq_range": [min(all_lq_prefill), max(all_lq_prefill)] if all_lq_prefill else None,
                "decode_lkv_range": [min(all_lkv_decode), max(all_lkv_decode)] if all_lkv_decode else None,
                "batch_size_range": [min(batch_sizes), max(batch_sizes)],
            },
        }, f, indent=2)
    print(f"\nFitted parameters saved to {output_json}")


if __name__ == "__main__":
    main()
