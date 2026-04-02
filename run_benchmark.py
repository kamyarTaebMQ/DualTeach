"""
Benchmark orchestrator: student vs teacher, fully live, apples-to-apples.

Starts each model as a FastAPI service, runs the load test, then shuts it
down before starting the next one so GPU memory is fully freed between runs.

Usage:
    python run_benchmark.py \
        --student_checkpoint checkpoints/student/epoch_5 \
        --teacher_model_id  google/gemma-3-12b-it \
        --concurrency 1 5 10 20 50 \
        --num_requests 50 \
        --max_new_tokens 64
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────────
STUDENT_PORT = 8010
TEACHER_PORT = 8011
HEALTH_TIMEOUT_S = 180      # max seconds to wait for service startup
MAX_NEW_TOKENS   = 64       # fixed for both models


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_problems(n: int = 100) -> list[str]:
    print(f"Loading {n} GSM8K test problems ...")
    ds = load_dataset("openai/gsm8k", "main")
    problems = [s["question"] for s in ds["test"]][:n]
    print(f"  {len(problems)} problems ready.")
    return problems


async def wait_for_health(url: str, timeout: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll /health until the service is ready or timeout expires."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url, timeout=3.0)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def fetch_info(base_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{base_url}/info", timeout=5.0)
            return r.json()
        except Exception:
            return {}


async def single_request(
    client: httpx.AsyncClient,
    url: str,
    problem: str,
    req_id: int,
) -> dict:
    payload = {"problem": problem, "max_new_tokens": MAX_NEW_TOKENS}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=120.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_id": req_id,
            "client_latency_ms": round(elapsed_ms, 2),
            "success": r.status_code == 200,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_id": req_id,
            "client_latency_ms": round(elapsed_ms, 2),
            "success": False,
            "error": str(exc),
        }


async def run_load_test(
    solve_url: str,
    problems: list[str],
    concurrency: int,
    num_requests: int,
) -> dict:
    print(f"\n  concurrency={concurrency}, requests={num_requests} ...")
    sem = asyncio.Semaphore(concurrency)

    async def bounded(client, problem, req_id):
        async with sem:
            return await single_request(client, solve_url, problem, req_id)

    async with httpx.AsyncClient() as client:
        # warmup
        await single_request(client, solve_url, problems[0], -1)

        t_start  = time.perf_counter()
        pool     = [problems[i % len(problems)] for i in range(num_requests)]
        tasks    = [bounded(client, p, i) for i, p in enumerate(pool)]
        results  = await asyncio.gather(*tasks)
        total_s  = time.perf_counter() - t_start

    successes = [r for r in results if r["success"]]
    latencies = [r["client_latency_ms"] for r in successes]
    n_ok      = len(successes)

    if latencies:
        p50 = round(float(np.percentile(latencies, 50)), 1)
        p95 = round(float(np.percentile(latencies, 95)), 1)
        p99 = round(float(np.percentile(latencies, 99)), 1)
        mean = round(float(np.mean(latencies)), 1)
    else:
        p50 = p95 = p99 = mean = None

    rps = round(n_ok / total_s, 2) if total_s > 0 else 0.0
    sr  = round(n_ok / num_requests, 4)

    print(f"    success={sr*100:.0f}%  RPS={rps}  p95={p95} ms  mean={mean} ms")
    return {
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successful": n_ok,
        "failed": num_requests - n_ok,
        "success_rate": sr,
        "total_time_s": round(total_s, 2),
        "throughput_rps": rps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
    }


async def benchmark_service(
    base_url: str,
    solve_url: str,
    problems: list[str],
    concurrency_levels: list[int],
    num_requests: int,
) -> tuple[list[dict], dict]:
    """Run load tests at all concurrency levels. Returns (stats_list, info)."""
    info = await fetch_info(base_url)
    stats = []
    for c in concurrency_levels:
        s = await run_load_test(solve_url, problems, c, num_requests)
        stats.append(s)
    return stats, info


# ══════════════════════════════════════════════════════════════════════════════
#  Service lifecycle
# ══════════════════════════════════════════════════════════════════════════════

def start_service(
    source: str,
    model_name: str,
    port: int,
    is_hub_model: bool,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "serve.py",
        "--model_name", model_name,
        "--port", str(port),
    ]
    if is_hub_model:
        cmd += ["--model_id", source]
    else:
        cmd += ["--checkpoint", source]

    print(f"\n[start] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


def stop_service(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[stop] Service stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════════

def print_latex_table(student_stats, student_info, teacher_stats, teacher_info):
    student_name = student_info.get("model", "Student (1.1B)")
    teacher_name = teacher_info.get("model", "Teacher (12B)")
    student_vram = student_info.get("vram_gb", "?")
    teacher_vram = teacher_info.get("vram_gb", "?")

    def fmt_row(stats: dict) -> str:
        c    = stats["concurrency"]
        p95  = stats["p95_ms"]
        rps  = stats["throughput_rps"]
        sr   = stats["success_rate"]
        if p95 is None or sr == 0:
            return f"      & {c:>3} & --- & --- & OOM \\\\"
        status = r"\checkmark"
        return f"      & {c:>3} & {p95:>8.1f} & {rps:>5.2f} & {status} \\\\"

    print("\n" + "="*70)
    print("LaTeX TABLE (copy-paste ready)")
    print("="*70)
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Web service benchmark: DualTeach student vs.\ teacher")
    print(f"    under concurrent load. Student VRAM: {student_vram}\\,GB,")
    print(f"    Teacher VRAM: {teacher_vram}\\,GB.")
    print(r"    max\_new\_tokens\,=\,64 for both.}")
    print(r"  \label{tab:service}")
    print(r"  \setlength{\tabcolsep}{4pt}")
    print(r"  \begin{tabular}{llccc}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{Users} & \textbf{p95 (ms)} & \textbf{RPS} & \textbf{Status} \\")
    print(r"    \midrule")
    print(f"    \\multirow{{{len(student_stats)}}}{{*}}{{\\shortstack[l]{{{student_name}\\\\({student_vram}\\,GB)}}}}")
    for s in student_stats:
        print(fmt_row(s))
    print(r"    \midrule")
    print(f"    \\multirow{{{len(teacher_stats)}}}{{*}}{{\\shortstack[l]{{{teacher_name}\\\\({teacher_vram}\\,GB)}}}}")
    for s in teacher_stats:
        print(fmt_row(s))
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("="*70)


def print_plain_table(student_stats, student_info, teacher_stats, teacher_info):
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Student VRAM: {student_info.get('vram_gb','?')} GB")
    print(f"Teacher VRAM: {teacher_info.get('vram_gb','?')} GB")
    print(f"max_new_tokens = {MAX_NEW_TOKENS}")
    print()
    header = f"{'Model':<22} {'Users':>6} {'p95 ms':>9} {'RPS':>7} {'Success':>8} {'Status'}"
    print(header)
    print("-" * len(header))

    def row(name, s):
        c   = s["concurrency"]
        p95 = s["p95_ms"]
        rps = s["throughput_rps"]
        sr  = s["success_rate"]
        if p95 is None or sr == 0:
            return f"{name:<22} {c:>6} {'---':>9} {'---':>7} {'0%':>8}   OOM"
        return f"{name:<22} {c:>6} {p95:>9.1f} {rps:>7.2f} {sr*100:>7.0f}%   OK"

    sn = student_info.get("model", "Student")
    tn = teacher_info.get("model", "Teacher")
    for s in student_stats:
        print(row(sn, s))
    print()
    for s in teacher_stats:
        print(row(tn, s))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_checkpoint", default="checkpoints/student/epoch_5")
    parser.add_argument("--teacher_model_id",   default="google/gemma-3-12b-it")
    parser.add_argument("--student_name",        default="DualTeach-Student-1.1B")
    parser.add_argument("--teacher_name",        default="Teacher-12B-IT")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--problems",     type=int, default=100)
    args = parser.parse_args()

    problems = load_problems(args.problems)

    all_results = {}

    # ── Student ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STUDENT: {args.student_name}")
    print(f"  checkpoint: {args.student_checkpoint}")
    print(f"{'='*60}")

    proc_s = start_service(args.student_checkpoint, args.student_name, STUDENT_PORT, is_hub_model=False)
    student_stats = student_info = None
    try:
        base = f"http://localhost:{STUDENT_PORT}"
        ready = await wait_for_health(f"{base}/health")
        if not ready:
            print("[ERROR] Student service did not start in time.")
            stop_service(proc_s)
            sys.exit(1)
        print("[ready] Student service is up.")
        student_stats, student_info = await benchmark_service(
            base_url=base,
            solve_url=f"{base}/solve",
            problems=problems,
            concurrency_levels=args.concurrency,
            num_requests=args.num_requests,
        )
        all_results["student"] = {"info": student_info, "stats": student_stats}
    finally:
        stop_service(proc_s)

    # Give GPU time to free memory
    print("Waiting 10s for GPU memory to release ...")
    await asyncio.sleep(10)

    # ── Teacher ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TEACHER: {args.teacher_name}")
    print(f"  model_id: {args.teacher_model_id}")
    print(f"{'='*60}")

    proc_t = start_service(args.teacher_model_id, args.teacher_name, TEACHER_PORT, is_hub_model=True)
    teacher_stats = teacher_info = None
    try:
        base = f"http://localhost:{TEACHER_PORT}"
        ready = await wait_for_health(f"{base}/health")
        if not ready:
            print("[ERROR] Teacher service did not start in time.")
            stop_service(proc_t)
            sys.exit(1)
        print("[ready] Teacher service is up.")
        teacher_stats, teacher_info = await benchmark_service(
            base_url=base,
            solve_url=f"{base}/solve",
            problems=problems,
            concurrency_levels=args.concurrency,
            num_requests=args.num_requests,
        )
        all_results["teacher"] = {"info": teacher_info, "stats": teacher_stats}
    finally:
        stop_service(proc_t)

    # ── Save results ─────────────────────────────────────────────────────────
    out = Path("results/benchmark_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved → {out}")

    # ── Print tables ─────────────────────────────────────────────────────────
    print_plain_table(student_stats, student_info, teacher_stats, teacher_info)
    print_latex_table(student_stats, student_info, teacher_stats, teacher_info)


if __name__ == "__main__":
    asyncio.run(main())
