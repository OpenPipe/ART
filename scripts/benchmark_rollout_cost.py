#!/usr/bin/env python3
"""
RL Rollout Cost Comparison: SGLang vs vLLM

Measures the prefix caching benefit of SGLang's RadixAttention for RL rollouts.
All rollouts share a long prefix (article/context), which is the typical pattern
in agentic RL training.

This benchmark focuses ONLY on rollout/inference costs - no training.

Usage:
    python scripts/benchmark_rollout_cost.py --backend sglang --output results_sglang.json
    python scripts/benchmark_rollout_cost.py --backend vllm --output results_vllm.json
    python scripts/benchmark_rollout_cost.py --compare results_sglang.json results_vllm.json
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import aiohttp

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "just-the-facts"))

# GPU hourly costs (USD)
GPU_COSTS = {
    "H100": 3.50,
    "A100_80GB": 2.50,
    "A100_40GB": 1.80,
    "A10G": 1.00,
    "L4": 0.70,
    "default": 2.00,
}

SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"


@dataclass
class RolloutResult:
    """Benchmark results for rollout-only comparison."""
    backend: str
    model: str
    gpu_type: str
    num_batches: int
    rollouts_per_batch: int
    
    # Timing
    total_time_seconds: float
    avg_batch_time_seconds: float
    batch_times: list[float] = field(default_factory=list)
    
    # Throughput
    total_rollouts: int = 0
    rollouts_per_second: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    
    # Cost
    gpu_hours: float = 0.0
    estimated_cost_usd: float = 0.0
    cost_per_1k_rollouts_usd: float = 0.0


def get_gpu_info() -> tuple[str, float]:
    """Get GPU type and hourly cost."""
    gpu_type = "default"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            for key in GPU_COSTS:
                if key.lower().replace("_", "") in name.replace("-", "").replace(" ", ""):
                    gpu_type = key
                    break
    except Exception:
        pass
    return gpu_type, GPU_COSTS.get(gpu_type, GPU_COSTS["default"])


async def wait_for_server(host: str, port: int, timeout: float = 180.0) -> None:
    """Wait for server to be ready."""
    start_time = time.time()
    print("Waiting for server to start", end="", flush=True)
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        print(" ready!")
                        return
        except Exception:
            pass
        print(".", end="", flush=True)
        await asyncio.sleep(2)
    raise TimeoutError(f"\nServer did not start within {timeout} seconds. Check server logs.")


def start_vllm_server(model_name: str) -> subprocess.Popen:
    """Start vLLM server as subprocess with high capacity settings."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--gpu-memory-utilization", "0.88",  # Safe GPU memory allocation
        "--max-num-seqs", "128",  # High but safe concurrent sequences
        "--enable-prefix-caching",
    ]
    print(f"Starting vLLM server with high capacity settings")
    print(f"  --max-num-seqs 128")
    print(f"  --gpu-memory-utilization 0.88")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create process group for clean shutdown
    )


def start_sglang_server(model_name: str) -> subprocess.Popen:
    """Start SGLang server as subprocess with high capacity settings."""
    # Try to find SGLang server venv
    sglang_python = sys.executable
    if os.path.exists(".venv-sglang-server/bin/python"):
        sglang_python = os.path.abspath(".venv-sglang-server/bin/python")
        print(f"Using SGLang server venv: {sglang_python}")
    
    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--mem-fraction-static", "0.88",  # Safe GPU memory allocation
        "--max-running-requests", "128",  # High but safe concurrent requests
        "--max-total-tokens", "49152",  # High token capacity
    ]
    print(f"Starting SGLang server with high capacity settings")
    print(f"  --max-running-requests 128")
    print(f"  --max-total-tokens 49152")
    print(f"  --mem-fraction-static 0.88")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create process group for clean shutdown
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Stop server subprocess."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


async def run_benchmark(
    backend_type: str,
    model_name: str,
    num_batches: int,
    rollouts_per_batch: int,
    max_concurrent: int = 16,
) -> RolloutResult:
    """Run rollout-only benchmark (NO training, pure inference)."""
    
    from openai import AsyncOpenAI
    
    # Import just-the-facts scenario/scraping
    from just_the_facts.scenarios import train_scenarios
    from just_the_facts.utils import scrape_article
    
    gpu_type, gpu_cost = get_gpu_info()
    
    print(f"\n{'='*60}")
    print(f"Rollout Cost Benchmark: {backend_type.upper()} (INFERENCE ONLY)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hr)")
    print(f"Batches: {num_batches}")
    print(f"Rollouts/batch: {rollouts_per_batch}")
    print(f"Total rollouts: {num_batches * rollouts_per_batch}")
    print(f"{'='*60}\n")
    
    # Kill any existing servers
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    await asyncio.sleep(2)
    
    # Start server (inference only - no training!)
    print(f"Starting {backend_type} server...")
    if backend_type == "sglang":
        server_proc = start_sglang_server(model_name)
    else:
        server_proc = start_vllm_server(model_name)
    
    try:
        await wait_for_server(SERVER_HOST, SERVER_PORT)
        print("Server ready!\n")
        
        # Create OpenAI client pointing to local server
        client = AsyncOpenAI(
            api_key="dummy",
            base_url=f"http://{SERVER_HOST}:{SERVER_PORT}/v1",
        )
        
        # Warm up
        print("Warming up...")
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        print("Warm-up complete.\n")
        
        batch_times: list[float] = []
        total_rollouts = 0
        total_tokens = 0
        
        scenarios = train_scenarios[:num_batches]
        total_start = time.perf_counter()
        
        for batch_idx, scenario in enumerate(scenarios):
            # Check if server is still alive
            if server_proc.poll() is not None:
                raise RuntimeError(f"Server process died with code {server_proc.returncode}")
            
            print(f"Batch {batch_idx + 1}/{num_batches}: {scenario.article_url[:50]}...")
            
            # Scrape article (shared prefix for all rollouts in batch) with timeout
            try:
                article_text = await asyncio.wait_for(
                    scrape_article(scenario.article_url),
                    timeout=30.0  # 30 second timeout for scraping
                )
                
                # Limit article length to prevent extremely long inputs
                # Very long articles cause timeouts during generation
                max_article_chars = 8000  # ~2000 tokens
                if len(article_text) > max_article_chars:
                    print(f"  📏 Article too long ({len(article_text)} chars), truncating to {max_article_chars}")
                    article_text = article_text[:max_article_chars] + "..."
                    
            except asyncio.TimeoutError:
                print(f"  ⚠️  Article scraping timed out, skipping batch")
                continue
            except Exception as e:
                print(f"  ⚠️  Article scraping failed: {e}, skipping batch")
                continue
            
            system_msg = "You are an unbiased summarizer of news articles. Summarize the key facts in 300 words or less."
            user_msg = f"Article:\n\n{article_text}"
            
            batch_start = time.perf_counter()
            
            # Run rollouts fully concurrent - server has high capacity
            async def single_rollout(idx):
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                            max_tokens=500,
                        ),
                        timeout=90.0  # 90 second timeout per rollout (longer articles need more time)
                    )
                    print(".", end="", flush=True)  # Success indicator
                    return resp
                except asyncio.TimeoutError:
                    print("T", end="", flush=True)  # Timeout
                    return None
                except Exception as e:
                    print("E", end="", flush=True)  # Error
                    return None
            
            # Run all rollouts fully parallel (server configured for high capacity)
            print(f"  Running {rollouts_per_batch} rollouts: ", end="", flush=True)
            responses = await asyncio.gather(*[
                single_rollout(i) for i in range(rollouts_per_batch)
            ], return_exceptions=True)
            print(" done", flush=True)
            
            # Filter out None/failed responses
            successful_responses = [r for r in responses if r is not None and not isinstance(r, Exception)]
            failed_count = len(responses) - len(successful_responses)
            if failed_count > 0:
                print(f"  ⚠️  {failed_count}/{rollouts_per_batch} rollouts failed")
            
            if not successful_responses:
                print(f"  ❌ All rollouts failed, skipping batch")
                continue
            
            responses = successful_responses
            
            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
            
            # Count tokens
            batch_tokens = sum(
                len(r.choices[0].message.content or "") // 4 
                for r in responses
            )
            total_rollouts += len(responses)
            total_tokens += batch_tokens
            
            print(f"  {len(responses)} rollouts in {batch_time:.2f}s ({len(responses)/batch_time:.1f}/s)")
        
        total_time = time.perf_counter() - total_start
        
    finally:
        print("\nShutting down server...")
        stop_server(server_proc)
    
    # Calculate metrics
    gpu_hours = total_time / 3600
    estimated_cost = gpu_hours * gpu_cost
    cost_per_1k = (estimated_cost / total_rollouts) * 1000 if total_rollouts > 0 else 0
    
    return RolloutResult(
        backend=backend_type,
        model=model_name,
        gpu_type=gpu_type,
        num_batches=num_batches,
        rollouts_per_batch=rollouts_per_batch,
        total_time_seconds=total_time,
        avg_batch_time_seconds=sum(batch_times) / len(batch_times) if batch_times else 0,
        batch_times=batch_times,
        total_rollouts=total_rollouts,
        rollouts_per_second=total_rollouts / total_time if total_time > 0 else 0,
        tokens_generated=total_tokens,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        gpu_hours=gpu_hours,
        estimated_cost_usd=estimated_cost,
        cost_per_1k_rollouts_usd=cost_per_1k,
    )


def print_results(r: RolloutResult) -> None:
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {r.backend.upper()}")
    print(f"{'='*60}")
    print(f"Model: {r.model}")
    print(f"GPU: {r.gpu_type}")
    
    print(f"\n⏱️  TIMING:")
    print(f"  Total: {r.total_time_seconds:.1f}s")
    print(f"  Avg batch: {r.avg_batch_time_seconds:.2f}s")
    
    print(f"\n🚀 THROUGHPUT:")
    print(f"  Rollouts: {r.total_rollouts}")
    print(f"  Rollouts/sec: {r.rollouts_per_second:.2f}")
    print(f"  Tokens/sec: {r.tokens_per_second:.0f}")
    
    print(f"\n💰 COST:")
    print(f"  GPU hours: {r.gpu_hours:.4f}")
    print(f"  Estimated cost: ${r.estimated_cost_usd:.4f}")
    print(f"  Cost/1K rollouts: ${r.cost_per_1k_rollouts_usd:.4f}")
    
    print(f"{'='*60}\n")


def compare_results(sglang_file: str, vllm_file: str) -> None:
    """Compare SGLang vs vLLM results and output the savings percentage."""
    with open(sglang_file) as f:
        sg = json.load(f)
    with open(vllm_file) as f:
        vl = json.load(f)
    
    print(f"\n{'='*70}")
    print("SGLang vs vLLM: RL Rollout Cost Comparison")
    print(f"{'='*70}")
    print(f"Model: {sg['model']}")
    print(f"Batches: {sg['num_batches']}, Rollouts/batch: {sg['rollouts_per_batch']}")
    print(f"Total rollouts: {sg['total_rollouts']}")
    
    print(f"\n{'Metric':<30} {'vLLM':>15} {'SGLang':>15} {'Savings':>12}")
    print("-" * 70)
    
    # Time comparison
    time_savings = (vl['total_time_seconds'] - sg['total_time_seconds']) / vl['total_time_seconds'] * 100
    print(f"{'Total time (s)':<30} {vl['total_time_seconds']:>15.1f} {sg['total_time_seconds']:>15.1f} {time_savings:>11.1f}%")
    
    # Throughput comparison
    throughput_gain = (sg['rollouts_per_second'] - vl['rollouts_per_second']) / vl['rollouts_per_second'] * 100
    print(f"{'Rollouts/sec':<30} {vl['rollouts_per_second']:>15.2f} {sg['rollouts_per_second']:>15.2f} {throughput_gain:>+11.1f}%")
    
    # Cost comparison
    cost_savings = (vl['cost_per_1k_rollouts_usd'] - sg['cost_per_1k_rollouts_usd']) / vl['cost_per_1k_rollouts_usd'] * 100
    print(f"{'Cost/1K rollouts ($)':<30} {vl['cost_per_1k_rollouts_usd']:>15.4f} {sg['cost_per_1k_rollouts_usd']:>15.4f} {cost_savings:>11.1f}%")
    
    # The headline number
    print(f"\n{'='*70}")
    print(f"📊 HEADLINE: SGLang saves {cost_savings:.0f}% on RL rollout costs")
    print(f"   (due to RadixAttention prefix caching for shared agent contexts)")
    print(f"{'='*70}")
    
    # Projected savings at scale
    print(f"\n📈 PROJECTED SAVINGS:")
    for name, rollouts in [("10K rollouts", 10_000), ("100K rollouts", 100_000), ("1M rollouts", 1_000_000)]:
        vl_cost = vl['cost_per_1k_rollouts_usd'] * (rollouts / 1000)
        sg_cost = sg['cost_per_1k_rollouts_usd'] * (rollouts / 1000)
        savings = vl_cost - sg_cost
        print(f"  {name}: Save ${savings:.2f} ({cost_savings:.0f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="RL Rollout Cost Benchmark (SGLang vs vLLM)")
    parser.add_argument("--backend", choices=["sglang", "vllm"], help="Backend to benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches (each uses different article)")
    parser.add_argument("--rollouts-per-batch", type=int, default=32, help="Rollouts per batch (share same prefix)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("SGLANG", "VLLM"), help="Compare two result files")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    if not args.backend:
        parser.error("--backend required unless using --compare")
    
    result = asyncio.run(run_benchmark(
        args.backend,
        args.model,
        args.num_batches,
        args.rollouts_per_batch,
    ))
    
    print_results(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
