#!/usr/bin/env python3
"""Benchmark inference performance for vLLM vs SGLang.

This script measures throughput, latency, and memory usage for both inference
engines. Run it in separate environments for accurate comparison:

    # vLLM environment
    source .venv-vllm/bin/activate
    python scripts/benchmark_inference.py --engine vllm

    # SGLang environment
    source .venv-sglang/bin/activate
    python scripts/benchmark_inference.py --engine sglang

For RL-specific benchmarks that test prefix caching:
    python scripts/benchmark_inference.py --engine sglang --test-prefix-caching
"""

# IMPORTANT: Import unsloth BEFORE any other ML libraries to prevent early CUDA initialization.
# This must happen before importing transformers, torch, vllm, or the art package.
# See: https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
import os
os.environ["IMPORT_UNSLOTH"] = "1"  # Tell art package to import unsloth early

try:
    import unsloth  # noqa: F401
except ImportError:
    pass  # unsloth not installed, continue without it

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

# Sample prompts simulating agent trajectories with shared prefixes
SYSTEM_PROMPT = """You are a helpful AI assistant participating in a reinforcement learning training loop. You help users with various tasks including coding, analysis, and general questions. Be concise and accurate in your responses."""

# Prompts with shared prefix (tests RadixAttention benefit)
SHARED_PREFIX = """Here is the context for this task:

The user is working on a Python project that involves data processing. They have the following code structure:

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        self.data = pd.read_csv(filepath)
        return self.data
    
    def process(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded")
        # Processing logic here
        return self.data
```

Based on this context, please help with the following:

"""

VARIED_SUFFIXES = [
    "What is the time complexity of the load_data method?",
    "How can we add error handling to the load_data method?",
    "Write a unit test for the process method.",
    "Add type hints to improve the code quality.",
    "Implement a save_data method that writes to CSV.",
    "Add logging to track data processing steps.",
    "How would you parallelize the process method?",
    "Add input validation to the constructor.",
]

# Completely different prompts (no shared prefix)
INDEPENDENT_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
    "What's the difference between TCP and UDP?",
    "Define 'machine learning' briefly.",
    "What year did World War II end?",
    "Name three programming languages.",
]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    engine: str
    model: str
    test_type: str
    num_requests: int
    total_tokens_generated: int
    total_time_seconds: float
    throughput_tokens_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    ttft_avg_ms: float
    ttft_p99_ms: float
    memory_used_gb: float
    errors: int = 0


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    latency_ms: float
    ttft_ms: float
    tokens_generated: int
    error: bool = False


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_vllm_benchmark(
    model: str,
    num_requests: int,
    max_tokens: int,
    concurrency: int,
    test_prefix_caching: bool,
) -> BenchmarkResult:
    """Run benchmark using vLLM."""
    try:
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.sampling_params import SamplingParams
    except ImportError:
        print("vLLM not installed. Install with: pip install openpipe-art[backend]")
        sys.exit(1)
    
    print(f"Starting vLLM engine for {model}...")
    
    engine_args = AsyncEngineArgs(
        model=model,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    # Note: In vLLM 0.13.0 (V1 engine), from_engine_args is NOT async
    engine = AsyncLLM.from_engine_args(engine_args)
    
    # Warmup
    print("Warming up...")
    params = SamplingParams(max_tokens=10, temperature=0.0)
    async for _ in engine.generate("Hello", params, request_id="warmup"):
        pass
    
    # Build prompts
    if test_prefix_caching:
        prompts = [
            SHARED_PREFIX + VARIED_SUFFIXES[i % len(VARIED_SUFFIXES)]
            for i in range(num_requests)
        ]
        test_type = "prefix_caching"
    else:
        prompts = [
            INDEPENDENT_PROMPTS[i % len(INDEPENDENT_PROMPTS)]
            for i in range(num_requests)
        ]
        test_type = "independent"
    
    async def process_request(prompt: str, request_id: str) -> RequestMetrics:
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        start_time = time.perf_counter()
        ttft = None
        tokens = 0
        
        try:
            async for output in engine.generate(prompt, params, request_id=request_id):
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000
                tokens = len(output.outputs[0].token_ids)
            
            latency = (time.perf_counter() - start_time) * 1000
            return RequestMetrics(
                latency_ms=latency,
                ttft_ms=ttft or latency,
                tokens_generated=tokens,
            )
        except Exception as e:
            print(f"Error: {e}")
            return RequestMetrics(latency_ms=0, ttft_ms=0, tokens_generated=0, error=True)
    
    print(f"Running {num_requests} requests ({test_type}) with concurrency {concurrency}...")
    start_time = time.perf_counter()
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(prompt: str, idx: int) -> RequestMetrics:
        async with semaphore:
            return await process_request(prompt, f"req_{idx}")
    
    tasks = [bounded_request(p, i) for i, p in enumerate(prompts)]
    metrics = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    valid_metrics = [m for m in metrics if not m.error]
    latencies = [m.latency_ms for m in valid_metrics]
    ttfts = [m.ttft_ms for m in valid_metrics]
    total_tokens = sum(m.tokens_generated for m in valid_metrics)
    
    # Get memory usage
    try:
        import torch
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        memory_gb = 0.0
    
    return BenchmarkResult(
        engine="vllm",
        model=model,
        test_type=test_type,
        num_requests=num_requests,
        total_tokens_generated=total_tokens,
        total_time_seconds=total_time,
        throughput_tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        ttft_avg_ms=statistics.mean(ttfts) if ttfts else 0,
        ttft_p99_ms=percentile(ttfts, 99),
        memory_used_gb=memory_gb,
        errors=len([m for m in metrics if m.error]),
    )


def run_sglang_benchmark_sync(
    model: str,
    num_requests: int,
    max_tokens: int,
    concurrency: int,
    test_prefix_caching: bool,
) -> BenchmarkResult:
    """Run benchmark using SGLang HTTP server.
    
    SGLang's Engine class has event loop issues, so we use the HTTP server
    approach instead: start server as subprocess, query via OpenAI-compatible API.
    """
    import subprocess
    import signal
    import requests
    from openai import OpenAI
    
    port = 30000
    host = "127.0.0.1"
    
    print(f"Starting SGLang server for {model}...")
    
    # Start SGLang server as subprocess
    server_process = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", host,
            "--port", str(port),
            "--mem-fraction-static", "0.9",
            "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    
    # Wait for server to be ready
    print("Waiting for server to start...")
    server_ready = False
    for _ in range(120):  # 2 minute timeout
        try:
            resp = requests.get(f"http://{host}:{port}/v1/models", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(1)
    
    if not server_ready:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        raise RuntimeError("SGLang server failed to start")
    
    print("Server ready!")
    
    # Create OpenAI client
    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
    
    # Warmup
    print("Warming up...")
    client.completions.create(model=model, prompt="Hello", max_tokens=10)
    
    # Build prompts
    if test_prefix_caching:
        prompts = [
            SHARED_PREFIX + VARIED_SUFFIXES[i % len(VARIED_SUFFIXES)]
            for i in range(num_requests)
        ]
        test_type = "prefix_caching"
    else:
        prompts = [
            INDEPENDENT_PROMPTS[i % len(INDEPENDENT_PROMPTS)]
            for i in range(num_requests)
        ]
        test_type = "independent"
    
    def process_request_sync(prompt: str) -> RequestMetrics:
        start_time = time.perf_counter()
        ttft = None
        tokens = 0
        
        try:
            # Use streaming to measure TTFT
            stream = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0,
                stream=True,
            )
            
            for chunk in stream:
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000
                if chunk.choices and chunk.choices[0].text:
                    tokens += 1  # Approximate: 1 chunk ≈ 1 token
            
            latency = (time.perf_counter() - start_time) * 1000
            return RequestMetrics(
                latency_ms=latency,
                ttft_ms=ttft or latency,
                tokens_generated=tokens,
            )
        except Exception as e:
            print(f"Error: {e}")
            return RequestMetrics(latency_ms=0, ttft_ms=0, tokens_generated=0, error=True)
    
    print(f"Running {num_requests} requests ({test_type}) with concurrency {concurrency}...")
    start_time = time.perf_counter()
    
    # Run requests with thread pool for concurrency
    import concurrent.futures
    metrics = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        metrics = list(executor.map(process_request_sync, prompts))
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    valid_metrics = [m for m in metrics if not m.error]
    latencies = [m.latency_ms for m in valid_metrics]
    ttfts = [m.ttft_ms for m in valid_metrics]
    total_tokens = sum(m.tokens_generated for m in valid_metrics)
    
    # Get memory usage (approximate from server)
    try:
        import torch
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        memory_gb = 0.0
    
    # Cleanup - kill server
    print("Shutting down server...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait(timeout=10)
    except Exception:
        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
    
    return BenchmarkResult(
        engine="sglang",
        model=model,
        test_type=test_type,
        num_requests=num_requests,
        total_tokens_generated=total_tokens,
        total_time_seconds=total_time,
        throughput_tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        ttft_avg_ms=statistics.mean(ttfts) if ttfts else 0,
        ttft_p99_ms=percentile(ttfts, 99),
        memory_used_gb=memory_gb,
        errors=len([m for m in metrics if m.error]),
    )


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"Benchmark Results: {result.engine.upper()} ({result.test_type})")
    print(f"{'='*70}")
    print(f"Model: {result.model}")
    print(f"Requests: {result.num_requests} (Errors: {result.errors})")
    print(f"{'-'*70}")
    print(f"{'Metric':<30} {'Value':>20}")
    print(f"{'-'*70}")
    print(f"{'Total tokens':<30} {result.total_tokens_generated:>20,}")
    print(f"{'Total time (s)':<30} {result.total_time_seconds:>20.2f}")
    print(f"{'Throughput (tok/s)':<30} {result.throughput_tokens_per_second:>20,.1f}")
    print(f"{'-'*70}")
    print(f"{'Avg latency (ms)':<30} {result.avg_latency_ms:>20.1f}")
    print(f"{'P50 latency (ms)':<30} {result.p50_latency_ms:>20.1f}")
    print(f"{'P95 latency (ms)':<30} {result.p95_latency_ms:>20.1f}")
    print(f"{'P99 latency (ms)':<30} {result.p99_latency_ms:>20.1f}")
    print(f"{'-'*70}")
    print(f"{'Avg TTFT (ms)':<30} {result.ttft_avg_ms:>20.1f}")
    print(f"{'P99 TTFT (ms)':<30} {result.ttft_p99_ms:>20.1f}")
    print(f"{'-'*70}")
    print(f"{'Memory used (GB)':<30} {result.memory_used_gb:>20.2f}")
    print(f"{'='*70}\n")


def compare_results(results: list[BenchmarkResult]) -> None:
    """Compare results from multiple runs."""
    if len(results) < 2:
        return
    
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    # Group by test type
    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_type.setdefault(r.test_type, []).append(r)
    
    for test_type, type_results in by_type.items():
        if len(type_results) < 2:
            continue
        
        print(f"\n{test_type.upper()} TEST:")
        print(f"{'-'*80}")
        
        base = type_results[0]
        
        def pct_change(new: float, old: float) -> str:
            if old == 0:
                return "N/A"
            change = ((new - old) / old) * 100
            sign = "+" if change > 0 else ""
            return f"{sign}{change:.1f}%"
        
        header = f"{'Metric':<25}"
        for r in type_results:
            header += f" {r.engine:>15}"
        if len(type_results) == 2:
            header += f" {'Change':>12}"
        print(header)
        print("-" * 80)
        
        metrics = [
            ("Throughput (tok/s)", "throughput_tokens_per_second", True),
            ("Avg Latency (ms)", "avg_latency_ms", False),
            ("P99 Latency (ms)", "p99_latency_ms", False),
            ("Avg TTFT (ms)", "ttft_avg_ms", False),
            ("Memory (GB)", "memory_used_gb", False),
        ]
        
        for name, attr, higher_better in metrics:
            row = f"{name:<25}"
            values = [getattr(r, attr) for r in type_results]
            for v in values:
                row += f" {v:>15.1f}"
            if len(type_results) == 2:
                change = pct_change(values[1], values[0])
                # Add indicator for better/worse
                if higher_better:
                    indicator = "↑" if values[1] > values[0] else "↓"
                else:
                    indicator = "↓" if values[1] < values[0] else "↑"
                row += f" {change:>10} {indicator}"
            print(row)
    
    print(f"{'='*80}\n")


async def main_vllm(args) -> list[BenchmarkResult]:
    """Run vLLM benchmarks (async)."""
    results = []
    
    result = await run_vllm_benchmark(
        args.model,
        args.num_requests,
        args.max_tokens,
        args.concurrency,
        args.test_prefix_caching,
    )
    results.append(result)
    print_results(result)
    
    # If testing prefix caching, also run without for comparison
    if args.test_prefix_caching:
        print("\nRunning comparison without prefix caching...")
        result2 = await run_vllm_benchmark(
            args.model,
            args.num_requests,
            args.max_tokens,
            args.concurrency,
            False,
        )
        results.append(result2)
        print_results(result2)
        compare_results(results)
    
    return results


def main_sglang(args) -> list[BenchmarkResult]:
    """Run SGLang benchmarks (sync - SGLang uses run_until_complete internally)."""
    results = []
    
    result = run_sglang_benchmark_sync(
        args.model,
        args.num_requests,
        args.max_tokens,
        args.concurrency,
        args.test_prefix_caching,
    )
    results.append(result)
    print_results(result)
    
    # If testing prefix caching, also run without for comparison
    if args.test_prefix_caching:
        print("\nRunning comparison without prefix caching...")
        result2 = run_sglang_benchmark_sync(
            args.model,
            args.num_requests,
            args.max_tokens,
            args.concurrency,
            False,
        )
        results.append(result2)
        print_results(result2)
        compare_results(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM vs SGLang inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with vLLM
  python benchmark_inference.py --engine vllm --num-requests 50

  # Test SGLang prefix caching benefit
  python benchmark_inference.py --engine sglang --test-prefix-caching

  # Full comparison (run in respective environments)
  python benchmark_inference.py --engine vllm --output results_vllm.json
  python benchmark_inference.py --engine sglang --output results_sglang.json
        """
    )
    parser.add_argument(
        "--engine",
        choices=["vllm", "sglang"],
        required=True,
        help="Which engine to benchmark",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to benchmark (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests (default: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent requests (default: 8)",
    )
    parser.add_argument(
        "--test-prefix-caching",
        action="store_true",
        help="Test with shared prefix prompts (shows RadixAttention benefit)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    # Run benchmark - vLLM uses asyncio, SGLang is sync
    if args.engine == "vllm":
        results = asyncio.run(main_vllm(args))
    else:
        # SGLang must run outside asyncio (it uses run_until_complete internally)
        results = main_sglang(args)
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
