"""
Benchmark inference performance for Qwen2.5-7B-Instruct using ART.

This script sends 5 concurrent requests with approximately 1000 input tokens
and requests approximately 1000 output tokens (max_tokens=1000), repeating
for 10 iterations. It measures per-request latencies and summarizes statistics.
"""

import asyncio
import statistics
import time

from dotenv import load_dotenv

import art
from art.local import LocalBackend

load_dotenv()


async def timed_request(client, model_name, prompt, max_tokens, temperature):
    """Execute a single model request and measure elapsed time and token usage."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start
    print(elapsed)
    prompt_tokens = None
    completion_tokens = None
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
    return {
        "response": response,
        "elapsed": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def main():
    # Define prompt (approx 1000 input tokens) and model
    prompt = (
        "Hello world. " * 500
    ).strip() + "Please repeat the entire prompt back to me verbatim"
    # Output tokens to request
    max_tokens = 1000
    temperature = 1.0
    model = art.TrainableModel(
        name="benchmark-qwen2.5-14b-instruct",
        project="benchmark-vllm",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )
    backend = LocalBackend()
    await model.register(backend)

    # Prepare for inference
    client = model.openai_client()
    iterations = 1
    concurrency = 800
    # Track iteration-level durations
    durations = []
    # Track per-request timings and token usage
    per_request_durations = []
    per_request_prompt_tokens = []
    per_request_completion_tokens = []

    for i in range(1, iterations + 1):
        print(
            f"Iteration {i}/{iterations}: sending {concurrency} concurrent requests..."
        )
        iteration_start = time.perf_counter()
        # launch concurrent requests and time each individually
        tasks = [
            timed_request(client, model.name, prompt, max_tokens, temperature)
            for _ in range(concurrency)
        ]
        # Wait for all responses
        results = await asyncio.gather(*tasks)
        # Record iteration duration
        iteration_elapsed = time.perf_counter() - iteration_start
        durations.append(iteration_elapsed)
        print(f"  Iteration time: {iteration_elapsed:.2f} seconds")
        # Record per-request stats
        for res in results:
            per_request_durations.append(res["elapsed"])
            if res["prompt_tokens"] is not None:
                per_request_prompt_tokens.append(res["prompt_tokens"])
            if res["completion_tokens"] is not None:
                per_request_completion_tokens.append(res["completion_tokens"])

    # Compute statistics
    total_time = sum(durations)
    min_time = min(durations)
    max_time = max(durations)
    avg_time = statistics.mean(durations)
    std_time = statistics.stdev(durations) if len(durations) > 1 else 0.0
    total_requests = iterations * concurrency
    avg_per_request = total_time / total_requests
    # Compute per-request statistics
    pr_min = min(per_request_durations) if per_request_durations else 0.0
    pr_max = max(per_request_durations) if per_request_durations else 0.0
    pr_avg = statistics.mean(per_request_durations) if per_request_durations else 0.0
    pr_std = (
        statistics.stdev(per_request_durations)
        if len(per_request_durations) > 1
        else 0.0
    )
    avg_prompt_tokens = (
        statistics.mean(per_request_prompt_tokens)
        if per_request_prompt_tokens
        else None
    )
    avg_completion_tokens = (
        statistics.mean(per_request_completion_tokens)
        if per_request_completion_tokens
        else None
    )

    # Report results
    print("\nInference benchmark results:")
    print(f"  Iterations:    {iterations}")
    print(f"  Concurrency:   {concurrency}")
    print(f"  Total time:    {total_time:.2f} s")
    print(f"  Min iteration: {min_time:.2f} s")
    print(f"  Max iteration: {max_time:.2f} s")
    print(f"  Avg iteration: {avg_time:.2f} s")
    print(f"  Std dev iter:  {std_time:.2f} s")
    print(f"  Avg per req:   {avg_per_request:.2f} s/request")
    # Per-request latency statistics
    print(f"  Min request time: {pr_min:.2f} s")
    print(f"  Max request time: {pr_max:.2f} s")
    print(f"  Avg request time: {pr_avg:.2f} s")
    print(f"  Std dev request time: {pr_std:.2f} s")
    if avg_prompt_tokens is not None:
        print(f"  Avg prompt tokens: {avg_prompt_tokens:.2f}")
    if avg_completion_tokens is not None:
        print(f"  Avg completion tokens: {avg_completion_tokens:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
