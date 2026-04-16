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
    --time-per-freq 120 \
    --seed 42
```

This sweeps all supported GPU graphics clocks from `--min-freq` up to 95% of the maximum, spending `--time-per-freq` seconds at each frequency. Memory clock is fixed at the highest supported value. Output is a JSONL file with one record per batch.

- Expected runtime: ~2 hours for 56 frequency points on A800-SXM4-80GB
- Expected output: ~100,000 batch records in `profiling_data.jsonl`
- Decode max_tokens: 50-1000 (long decode to accumulate large l_kv values up to ~2800)

## Step 3: Fit Model

```bash
conda activate myvllm
pip install scipy scikit-learn  # if not already installed
python fitting_script.py \
    --input /home/ubuntu/lqs/profiling_data.jsonl \
    --output-dir /home/ubuntu/lqs
```

This reads the profiling data, balances the dataset (subsamples decode-only batches to avoid overwhelming prefill/mixed), fits the 7-parameter model using:
1. Grid search over alpha with NNLS (non-negative least squares) regression
2. Non-linear optimization (scipy L-BFGS-B) with MAPE loss and non-negative parameter constraints

Saves `fitted_params.json` with all parameters and metrics.

## Full Scripts

### profiling_script.py

See `profiling_script.py` in this repository.

Key design choices:
- **4 profiling phases** per frequency: (A) pure prefill single request 30%, (B) batched prefill 15%, (C) decode-only with long sequences 20%, (D) mixed via continuous injection 30%
- **Long decode sequences**: max_tokens up to 1000 with `ignore_eos=True` to accumulate large l_kv values
- **sys.path fix**: removes CWD from Python path to avoid namespace conflict with editable vllm install
- **Clock verification**: reads back actual GPU frequency after locking, skips if >50 MHz deviation

### fitting_script.py

See `fitting_script.py` in this repository.

Key design choices:
- **Data balancing**: decode-only batches capped at 5x (prefill + mixed) to prevent imbalance
- **NNLS regression**: non-negative least squares ensures physically meaningful parameters
- **MAPE loss**: approach 2 uses MAPE (not MSE) as optimization objective for better percentage-based fitting
- **Non-negative bounds**: all 7 parameters constrained >= 0 in non-linear optimization
- **Outlier removal**: 0.5%-99.5% percentile filtering

## Quick Start

```bash
conda activate myvllm

# Step 1: Apply vLLM instrumentation (see Step 1 above for file contents)
# Create vllm/profiling_logger.py and patch vllm/v1/worker/gpu_model_runner.py

# Step 2: Run profiling (~2 hours for 56 frequency points)
VLLM_PROFILING=1 python profiling_script.py \
    --model-path /path/to/model \
    --output profiling_data.jsonl \
    --min-freq 500 \
    --time-per-freq 120

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
- The fitting script uses a fine grid (step=0.005) for alpha search and compares NNLS grid search with scipy L-BFGS-B non-linear optimization (MAPE loss), selecting whichever gives lower test MAPE.
- Decode-only batches dominate the dataset (~94%) due to long decode sequences. The fitting script balances this by subsampling decode-only to 5x the combined count of prefill + mixed batches.
