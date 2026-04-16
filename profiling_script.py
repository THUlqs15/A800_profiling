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
    # ~40% of time budget
    phase_a_end = t_start + time_limit_s * 0.30
    steps_a = 0
    while elapsed() < time_limit_s * 0.30:
        prompt_len = random.choice(prefill_lengths)
        ids = gen_prompt(prompt_len)
        engine.add_request(
            prompt={"prompt_token_ids": ids},
            params=SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
            request_id=make_id("pa"))
        engine.step()
        steps_a += 1
    while engine.has_unfinished_requests():
        engine.step()
        steps_a += 1

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
        steps_a += 1
    while engine.has_unfinished_requests():
        engine.step()
        steps_a += 1

    # --- Phase C: Decode-only, varying batch sizes and l_kv ---
    while elapsed() < time_limit_s * 0.65:
        bs = random.choice([1, 2, 4, 8, 16])
        prompt_len = random.choice([32, 64, 128])
        max_tok = random.choice([50, 100, 200, 500, 800, 1000])
        for _ in range(bs):
            ids = gen_prompt(prompt_len)
            engine.add_request(
                prompt={"prompt_token_ids": ids},
                params=SamplingParams(max_tokens=max_tok, temperature=0.8, ignore_eos=True),
                request_id=make_id("pc"))
        while engine.has_unfinished_requests():
            engine.step()
            if elapsed() > time_limit_s * 0.70:
                break
        # If time exceeded, break
        if elapsed() > time_limit_s * 0.70:
            break

    # Drain remaining
    drain_start = time.time()
    while engine.has_unfinished_requests():
        engine.step()
        if time.time() - drain_start > 60:
            break

    # --- Phase D: Mixed batches via continuous injection ---
    # Inject new prefill requests while decode requests are running
    mix_start = time.time()
    # Start background decoding with longer sequences
    for _ in range(random.choice([4, 8, 16])):
        ids = gen_prompt(random.choice([32, 64, 128]))
        engine.add_request(
            prompt={"prompt_token_ids": ids},
            params=SamplingParams(max_tokens=random.choice([100, 200, 400, 600]),
                                  temperature=0.8, ignore_eos=True),
            request_id=make_id("pd"))

    inject_interval = random.randint(3, 10)
    step_count = 0
    while elapsed() < time_limit_s * 0.95:
        if step_count % inject_interval == 0:
            # Inject 1-4 new prefill+decode requests
            n_inject = random.randint(1, 4)
            for _ in range(n_inject):
                pl = random.choice(prefill_lengths[:7])  # up to 2048
                mt = random.choice([50, 100, 200, 400, 600, 800])
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

    # Final drain (with timeout)
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
    parser.add_argument("--time-per-freq", type=int, default=120,
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
    est_total = len(profiling_freqs) * (args.time_per_freq + 15)  # +15 overhead
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
                # Count records for this freq
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
