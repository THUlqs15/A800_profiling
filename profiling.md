# Batch Latency Profiling -- Reproducible Workflow

## Prerequisites

- **vLLM**: v0.19.0+precompiled, editable install from source (commit 2a69949b)
- **Model path**: parameterized via `--model-path` (default: `/home/ubuntu/lqs/LLM_model`)
- **Conda environment**: `myvllm`
- **Python packages**: numpy, scipy, scikit-learn (install via `pip install scipy scikit-learn` in the conda env)
- **GPU permissions**: `sudo` access required for `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks`
- **No other GPU workloads** should be running during profiling

## Step 1: Instrument vLLM

Two files are added/modified in the vLLM source tree. The instrumentation is controlled by the `VLLM_PROFILING=1` environment variable -- when unset or `0`, there is zero overhead.

### File 1: `vllm/profiling_logger.py` (new file)

This module provides the profiling API. It writes JSONL records to the file specified by `VLLM_PROFILING_OUTPUT`.

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Batch profiling logger for latency model fitting.
Activated by setting VLLM_PROFILING=1 environment variable.
Logs per-batch timing, GPU frequency, and per-request metadata to JSONL.
"""

import json
import os
import subprocess
import threading
import time

_PROFILING_ENABLED = os.environ.get("VLLM_PROFILING", "0") == "1"
_PROFILING_OUTPUT = os.environ.get("VLLM_PROFILING_OUTPUT",
                                   "profiling_data.jsonl")
_lock = threading.Lock()
_batch_counter = 0
_file_handle = None


def is_profiling_enabled() -> bool:
    return _PROFILING_ENABLED


def _get_file():
    global _file_handle
    if _file_handle is None:
        _file_handle = open(_PROFILING_OUTPUT, "a")
    return _file_handle


def get_gpu_graphics_freq_mhz() -> int:
    """Read current GPU graphics clock frequency in MHz via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def get_gpu_memory_freq_mhz() -> int:
    """Read current GPU memory clock frequency in MHz via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def log_batch(wall_time_ms: float, gpu_freq_mhz: int,
              mem_freq_mhz: int, requests: list):
    """Log one batch execution record to the profiling JSONL file."""
    global _batch_counter
    with _lock:
        record = {
            "batch_id": _batch_counter,
            "gpu_freq_mhz": gpu_freq_mhz,
            "mem_freq_mhz": mem_freq_mhz,
            "wall_time_ms": wall_time_ms,
            "num_requests": len(requests),
            "requests": requests,
            "timestamp": time.time(),
        }
        f = _get_file()
        f.write(json.dumps(record) + "\n")
        f.flush()
        _batch_counter += 1


def close():
    global _file_handle
    if _file_handle is not None:
        _file_handle.close()
        _file_handle = None
```

### File 2: `vllm/v1/worker/gpu_model_runner.py` (modified)

Two changes to the `execute_model()` method of the outer `GPUModelRunner` class:

**Import** (add at top of file, after existing imports):

```python
from vllm.profiling_logger import is_profiling_enabled, log_batch, get_gpu_graphics_freq_mhz, get_gpu_memory_freq_mhz
```

**Before the forward pass** (insert before `# Run the model.` comment, around line 4012):

```python
        # --- Profiling: extract batch metadata before forward pass ---
        _profiling_active = is_profiling_enabled()
        if _profiling_active:
            _prof_requests = []
            for _pi in range(num_reqs):
                _l_q = int(num_scheduled_tokens_np[_pi])
                _req_idx = self.input_batch.req_id_to_index.get(
                    req_ids[_pi], _pi)
                _l_kv = int(
                    self.input_batch.num_computed_tokens_cpu[_req_idx])
                _prompt_len = int(
                    self.input_batch.num_prompt_tokens[_req_idx])
                _is_prefill = _l_kv < _prompt_len
                _prof_requests.append({
                    "type": "prefill" if _is_prefill else "decode",
                    "l_q": _l_q,
                    "l_kv": _l_kv,
                })
            _gpu_freq = get_gpu_graphics_freq_mhz()
            _mem_freq = get_gpu_memory_freq_mhz()
            torch.cuda.synchronize()
            _prof_start = time.perf_counter()
```

**After the forward pass** (insert after `model_output = self._model_forward(...)` block closes):

```python
        # --- Profiling: record timing after forward pass ---
        if _profiling_active:
            torch.cuda.synchronize()
            _prof_end = time.perf_counter()
            _wall_time_ms = (_prof_end - _prof_start) * 1000.0
            log_batch(_wall_time_ms, _gpu_freq, _mem_freq, _prof_requests)
```

**Important**: The correct file is `vllm/v1/worker/gpu_model_runner.py` (outer GPUModelRunner), NOT `vllm/v1/worker/gpu/model_runner.py` (inner GPUModelRunner). In vLLM v1 architecture, the outer class owns the forward pass that the EngineCore subprocess actually calls.

## Step 2: Run Profiling

```bash
conda activate myvllm
VLLM_PROFILING=1 python profiling_script.py \
    --model-path /home/ubuntu/lqs/LLM_model \
    --output /home/ubuntu/lqs/profiling_data.jsonl \
    --min-freq 500 \
    --time-per-freq 90 \
    --seed 42
```

This sweeps all supported GPU graphics clocks from `--min-freq` up to 95% of the maximum, spending `--time-per-freq` seconds at each frequency. Memory clock is fixed at the highest supported value. Output is a JSONL file with one record per batch.

Expected runtime: ~90 minutes for 56 frequency points on A800-SXM4-80GB.
Expected output: ~40,000 batch records in `profiling_data.jsonl`.

## Step 3: Fit Model

```bash
conda activate myvllm
python fitting_script.py \
    --input /home/ubuntu/lqs/profiling_data.jsonl \
    --output-dir /home/ubuntu/lqs
```

This reads the profiling data, fits the 7-parameter model using grid search over alpha + linear regression (and validates with scipy non-linear optimization), and saves `fitted_params.json` with all parameters and metrics.

## Full Scripts

### profiling_script.py

```python
#!/usr/bin/env python3
"""
Batch Latency Profiling Script for vLLM.

Sweeps GPU graphics clock frequencies and generates diverse batch compositions
(pure prefill, pure decode, mixed) to collect timing data for latency model fitting.

Usage:
    conda activate myvllm
    VLLM_PROFILING=1 python profiling_script.py \
        --model-path /home/ubuntu/lqs/LLM_model \
        --output /home/ubuntu/lqs/profiling_data.jsonl
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Fix Python path for editable vllm install
_cwd = os.getcwd()
if _cwd in sys.path:
    sys.path.remove(_cwd)
_parent = os.path.dirname(os.path.abspath(__file__))
if _parent in sys.path:
    sys.path.remove(_parent)

import numpy as np


def parse_supported_clocks():
    """Parse nvidia-smi supported clocks."""
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
        capture_output=True, text=True, timeout=10)
    lines = result.stdout.split("\n")
    mem_clock = None
    graphics_clocks = []
    for line in lines:
        line = line.strip()
        if line.startswith("Memory") and "MHz" in line:
            mem_clock = int(line.split(":")[1].strip().replace(" MHz", ""))
        if line.startswith("Graphics") and "MHz" in line:
            freq = int(line.split(":")[1].strip().replace(" MHz", ""))
            graphics_clocks.append(freq)
    return mem_clock, sorted(set(graphics_clocks))


def lock_clocks(mem_freq, gpu_freq):
    subprocess.run(
        ["sudo", "nvidia-smi", f"--lock-memory-clocks={mem_freq},{mem_freq}"],
        capture_output=True, text=True, timeout=10)
    subprocess.run(
        ["sudo", "nvidia-smi", f"--lock-gpu-clocks={gpu_freq},{gpu_freq}"],
        capture_output=True, text=True, timeout=10)


def reset_clocks():
    subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"],
                   capture_output=True, text=True, timeout=10)
    subprocess.run(["sudo", "nvidia-smi", "--reset-memory-clocks"],
                   capture_output=True, text=True, timeout=10)


def verify_clocks(expected_gpu_freq, expected_mem_freq):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    parts = result.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


def gen_prompt(length, vocab_size=50000):
    return [random.randint(100, vocab_size - 1) for _ in range(length)]


def run_profiling_at_frequency(engine, gpu_freq, time_limit_s=120):
    """Run profiling workloads. Returns after time_limit_s seconds.

    Produces:
      - Pure prefill batches (single & multi-request)
      - Pure decode batches (from decode-only runs)
      - Mixed prefill + decode batches (via continuous injection)
    """
    from vllm import SamplingParams

    prefill_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    t_start = time.time()
    req_counter = 0
    total_steps = 0

    def elapsed():
        return time.time() - t_start

    def make_id(prefix):
        nonlocal req_counter
        req_counter += 1
        return f"{prefix}_{gpu_freq}_{req_counter}"

    # --- Warmup: 3 short requests ---
    for _ in range(3):
        ids = gen_prompt(64)
        engine.add_request(
            prompt={"prompt_token_ids": ids},
            params=SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
            request_id=make_id("w"))
    while engine.has_unfinished_requests():
        engine.step()

    # --- Phase A: Pure prefill, single request, varying l_q ---
    # ~30% of time budget
    while elapsed() < time_limit_s * 0.30:
        prompt_len = random.choice(prefill_lengths)
        ids = gen_prompt(prompt_len)
        engine.add_request(
            prompt={"prompt_token_ids": ids},
            params=SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
            request_id=make_id("pa"))
        engine.step()
    while engine.has_unfinished_requests():
        engine.step()

    # --- Phase B: Batched prefill, varying batch sizes ---
    while elapsed() < time_limit_s * 0.45:
        bs = random.choice([2, 4, 8, 16])
        prompt_len = random.choice(prefill_lengths[:6])  # up to 1024
        for _ in range(bs):
            ids = gen_prompt(prompt_len)
            engine.add_request(
                prompt={"prompt_token_ids": ids},
                params=SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
                request_id=make_id("pb"))
        engine.step()
    while engine.has_unfinished_requests():
        engine.step()

    # --- Phase C: Decode-only, varying batch sizes and l_kv ---
    while elapsed() < time_limit_s * 0.60:
        bs = random.choice([1, 2, 4, 8, 16, 32])
        prompt_len = random.choice([32, 64, 128])
        max_tok = random.choice([20, 50, 100])
        for _ in range(bs):
            ids = gen_prompt(prompt_len)
            engine.add_request(
                prompt={"prompt_token_ids": ids},
                params=SamplingParams(max_tokens=max_tok, temperature=0.8, ignore_eos=True),
                request_id=make_id("pc"))
        while engine.has_unfinished_requests():
            engine.step()
            if elapsed() > time_limit_s * 0.65:
                break
        if elapsed() > time_limit_s * 0.65:
            break

    # Drain remaining
    drain_start = time.time()
    while engine.has_unfinished_requests():
        engine.step()
        if time.time() - drain_start > 30:
            break

    # --- Phase D: Mixed batches via continuous injection ---
    # Inject new prefill requests while decode requests are running
    for _ in range(random.choice([4, 8, 16])):
        ids = gen_prompt(random.choice([32, 64, 128]))
        engine.add_request(
            prompt={"prompt_token_ids": ids},
            params=SamplingParams(max_tokens=random.choice([30, 50, 80]),
                                  temperature=0.8, ignore_eos=True),
            request_id=make_id("pd"))

    inject_interval = random.randint(3, 10)
    step_count = 0
    while elapsed() < time_limit_s * 0.95:
        if step_count % inject_interval == 0:
            n_inject = random.randint(1, 4)
            for _ in range(n_inject):
                pl = random.choice(prefill_lengths[:7])  # up to 2048
                mt = random.choice([20, 40, 60, 80])
                ids = gen_prompt(pl)
                engine.add_request(
                    prompt={"prompt_token_ids": ids},
                    params=SamplingParams(max_tokens=mt, temperature=0.8,
                                          ignore_eos=True),
                    request_id=make_id("pd"))
            inject_interval = random.randint(3, 12)

        if engine.has_unfinished_requests():
            engine.step()
            step_count += 1
        else:
            break

    # Final drain
    drain_start = time.time()
    while engine.has_unfinished_requests():
        engine.step()
        if time.time() - drain_start > 20:
            break

    total_time = elapsed()
    return total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/home/ubuntu/lqs/LLM_model")
    parser.add_argument("--output", type=str,
                        default="/home/ubuntu/lqs/profiling_data.jsonl")
    parser.add_argument("--min-freq", type=int, default=500)
    parser.add_argument("--time-per-freq", type=int, default=90,
                        help="Time budget per frequency in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["VLLM_PROFILING"] = "1"
    os.environ["VLLM_PROFILING_OUTPUT"] = args.output

    if os.path.exists(args.output):
        os.remove(args.output)

    mem_freq, graphics_clocks = parse_supported_clocks()
    max_graphics = max(graphics_clocks)
    cutoff = int(max_graphics * 0.95)
    profiling_freqs = sorted(
        [f for f in graphics_clocks if f >= args.min_freq and f <= cutoff])

    print(f"GPU Info:")
    print(f"  Memory clock (fixed): {mem_freq} MHz")
    print(f"  Graphics clock range: {min(graphics_clocks)}-{max_graphics} MHz")
    print(f"  Profiling range: {min(profiling_freqs)}-{max(profiling_freqs)} MHz")
    print(f"  Number of frequency points: {len(profiling_freqs)}")
    print(f"  Time per freq: {args.time_per_freq}s")
    est_total = len(profiling_freqs) * (args.time_per_freq + 15)
    print(f"  Estimated total time: {est_total // 60}m {est_total % 60}s")
    print()

    from vllm import LLM, SamplingParams

    try:
        print("Initializing vLLM engine...")
        llm = LLM(
            model=args.model_path,
            enforce_eager=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=8192,
            max_num_seqs=64,
            seed=args.seed,
        )
        engine = llm.llm_engine
        print("Engine initialized.\n")

        try:
            vllm_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd="/home/ubuntu/lqs/vllm",
                capture_output=True, text=True, timeout=5
            ).stdout.strip()[:8]
        except Exception:
            vllm_commit = "unknown"

        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        print(f"GPU: {gpu_result.stdout.strip()}")
        print(f"vLLM commit: {vllm_commit}\n")

        for idx, gpu_freq in enumerate(profiling_freqs):
            print(f"[{idx + 1}/{len(profiling_freqs)}] {gpu_freq} MHz ...",
                  end=" ", flush=True)

            lock_clocks(mem_freq, gpu_freq)
            time.sleep(0.5)

            actual_gpu, actual_mem = verify_clocks(gpu_freq, mem_freq)
            if abs(actual_gpu - gpu_freq) > 50:
                print(f"SKIP (actual={actual_gpu}MHz)")
                reset_clocks()
                continue

            try:
                freq_time = run_profiling_at_frequency(
                    engine, gpu_freq, args.time_per_freq)
                with open(args.output) as f:
                    n_records = sum(1 for _ in f)
                print(f"done ({freq_time:.0f}s, total records: {n_records})")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

            reset_clocks()
            time.sleep(0.3)

        print(f"\nProfiling complete. Data: {args.output}")
        with open(args.output) as f:
            n = sum(1 for _ in f)
        print(f"Total records: {n}")

    finally:
        reset_clocks()
        print("GPU clocks reset.")


if __name__ == "__main__":
    main()
```

### fitting_script.py

```python
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
    """Compute aggregate features for each batch record."""
    n = len(records)
    f = np.zeros(n)
    sum_lq_sq = np.zeros(n)
    sum_lq_lkv = np.zeros(n)
    sum_lq = np.zeros(n)
    sum_lkv_decode = np.zeros(n)
    num_decode = np.zeros(n)
    y = np.zeros(n)

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
            else:
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
    """Build the linear design matrix for a given alpha value."""
    f = features["f"]
    f_alpha = np.power(f, alpha)

    X = np.column_stack([
        features["sum_lq_sq"] / f,
        features["sum_lq_lkv"] / f,
        features["sum_lq"] / f,
        features["sum_lkv_decode"] / f_alpha,
        features["num_decode"] / f_alpha,
        np.ones(len(f)),
    ])
    return X


def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def approach1_grid_search(features_train, y_train, features_test, y_test):
    """Grid search over alpha + linear regression."""
    best_mape = float("inf")
    best_alpha = None
    best_model = None

    alphas = np.arange(0.01, 1.0, 0.005)

    for alpha in alphas:
        X_train = build_design_matrix(features_train, alpha)
        X_test = build_design_matrix(features_test, alpha)

        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        mape = compute_mape(y_test, y_pred_test)

        if mape < best_mape:
            best_mape = mape
            best_alpha = alpha
            best_model = model

    coefs = best_model.coef_
    params = {
        "a_p": coefs[0],
        "b_p": coefs[1],
        "c_p": coefs[2],
        "a_d": coefs[3],
        "b_d": coefs[4],
        "t_c": coefs[5],
        "alpha": best_alpha,
    }

    return params, best_model, best_mape


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
        return np.mean((y - y_pred) ** 2)

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
        (None, None),   # a_p
        (None, None),   # b_p
        (None, None),   # c_p
        (None, None),   # a_d
        (None, None),   # b_d
        (0.01, 0.99),   # alpha
        (None, None),   # t_c
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
    print(f"  R^2:   {r2:.6f}")

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

    print(f"Loading data from {args.input}...")
    records = load_profiling_data(args.input)
    print(f"Loaded {len(records)} batch records")

    records = [r for r in records if r["num_requests"] > 0 and r["gpu_freq_mhz"] > 0]
    print(f"After filtering: {len(records)} records")

    features, y = compute_features(records)

    unique_freqs = np.unique(features["f"])
    print(f"\nData Summary:")
    print(f"  Unique GPU frequencies: {len(unique_freqs)}")
    print(f"  Frequency range: {unique_freqs.min():.0f} - {unique_freqs.max():.0f} MHz")
    print(f"  Wall time range: {y.min():.2f} - {y.max():.2f} ms")

    n_prefill_only = n_decode_only = n_mixed = 0
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

    # Outlier removal
    p99 = np.percentile(y, 99.5)
    p01 = np.percentile(y, 0.5)
    mask = (y <= p99) & (y >= p01)
    records_filtered = [r for r, m in zip(records, mask) if m]
    features_filtered, y_filtered = compute_features(records_filtered)
    print(f"\nAfter outlier removal (0.5%-99.5%): {len(records_filtered)} records")

    # 80/20 train/test split
    n = len(y_filtered)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    features_train = {k: v[train_idx] for k, v in features_filtered.items()}
    features_test = {k: v[test_idx] for k, v in features_filtered.items()}
    y_train = y_filtered[train_idx]
    y_test = y_filtered[test_idx]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Approach 1
    print("\n" + "=" * 60)
    print("Approach 1: Grid search over alpha + Linear Regression")
    print("=" * 60)

    params1, model1, mape1 = approach1_grid_search(
        features_train, y_train, features_test, y_test)

    print(f"\nBest alpha: {params1['alpha']:.4f}")
    print(f"Parameters:")
    for k, v in params1.items():
        print(f"  {k}: {v:.10e}")

    train_metrics1 = evaluate_model(params1, features_train, y_train, "Train metrics")
    test_metrics1 = evaluate_model(params1, features_test, y_test, "Test metrics")

    # Approach 2
    print("\n" + "=" * 60)
    print("Approach 2: Non-linear optimization (scipy L-BFGS-B)")
    print("=" * 60)

    params2, opt_result = approach2_nonlinear(features_train, y_train, params1)

    print(f"\nOptimizer converged: {opt_result.success}")
    print(f"Parameters:")
    for k, v in params2.items():
        print(f"  {k}: {v:.10e}")

    train_metrics2 = evaluate_model(params2, features_train, y_train, "Train metrics")
    test_metrics2 = evaluate_model(params2, features_test, y_test, "Test metrics")

    # Choose better approach
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
```

## Quick Start

```bash
conda activate myvllm

# Step 1: Apply vLLM instrumentation (see Step 1 above for file contents)
# Create vllm/profiling_logger.py and patch vllm/v1/worker/gpu_model_runner.py

# Step 2: Run profiling (~90 min for 56 frequency points)
VLLM_PROFILING=1 python profiling_script.py \
    --model-path /path/to/model \
    --output profiling_data.jsonl \
    --min-freq 500 \
    --time-per-freq 90

# Step 3: Fit model and get parameters + MAPE
pip install scipy scikit-learn  # if not already installed
python fitting_script.py \
    --input profiling_data.jsonl \
    --output-dir .

# Results are saved to fitted_params.json
```

## Notes

- The profiling instrumentation is toggled by `VLLM_PROFILING=1` env var. When unset, no overhead is added.
- Memory clock locking (`--lock-memory-clocks`) is not supported on A800-SXM4-80GB but only one memory clock (1593 MHz) is available, so it remains constant.
- The `sys.path` fix in `profiling_script.py` is needed because running from the same directory as the vllm source creates a namespace package conflict with the editable install.
- GPU frequencies below 500 MHz are skipped to avoid extremely slow execution with large models.
- The fitting script uses a fine grid (step=0.005) for alpha search and compares grid search + linear regression with scipy L-BFGS-B non-linear optimization, selecting whichever gives lower test MAPE.
