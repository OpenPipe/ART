#!/usr/bin/env python3
"""
End-to-end benchmark: SGLang + Megatron vs vLLM + Megatron.

Both backends use IDENTICAL rollout code: a single streaming call with
stream_options.include_usage=true for accurate token counting + TTFT.

SGLang path uses verl-style architecture:
  - Server starts ONCE and NEVER restarts
  - sleep(kv_cache+weights) / wake_up(kv_cache) for memory management
  - Merged LoRA weights saved to disk, reloaded via /update_weights

vLLM path uses existing ART MegatronBackend (sleep/wake).

Each step: rollout (timed) → Megatron train → next rollout with updated weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ===================================================================
# Worker — isolated subprocess per backend
# ===================================================================

def run_worker(backend: str, cfg: dict, results_path: str) -> None:
    import asyncio
    import aiohttp
    import torch

    from benchmarks.sglang_vs_vllm.metrics_collector import (
        BenchmarkRun, RequestMetrics, StepMetrics,
        get_gpu_memory_usage_nvidia_smi,
    )
    from benchmarks.sglang_vs_vllm.config import generate_benchmark_prompts

    logger.info(f"[{backend}] Worker PID={os.getpid()} GPUs={torch.cuda.device_count()}")

    # Extract config values
    model_id = cfg["model"]
    dataset = cfg["dataset"]
    num_steps = cfg["num_steps"]
    num_rollouts = cfg["num_rollouts"]
    concurrency = cfg["concurrency"]
    max_output_tokens = cfg["max_output_tokens"]
    max_seq_length = cfg["max_seq_length"]
    tp = cfg["tp"]
    gpu_mem = cfg["gpu_mem"]
    vllm_port = cfg["vllm_port"]
    sglang_port = cfg["sglang_port"]
    sglang_python = cfg["sglang_python"]
    seed = cfg["seed"]
    lr = cfg["learning_rate"]
    output_dir = cfg["output_dir"]

    # ---- helpers ---------------------------------------------------

    async def stream_rollout(
        base_url: str, model_name: str,
        prompts: list, max_tok: int, conc: int,
        api_key: str | None = None,
    ) -> list[RequestMetrics]:
        """Streaming rollout — IDENTICAL for both SGLang and vLLM.

        Uses stream_options.include_usage=true to get accurate server-side
        token counts in the final SSE chunk, while also measuring TTFT
        from the first content chunk. One rollout, both metrics, fair
        comparison.
        """
        sem = asyncio.Semaphore(conc)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async def _one(idx, msgs):
            async with sem:
                t0 = time.perf_counter()
                ttft = comp_tok = 0
                err = None
                first = False
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.post(
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json={"model": model_name, "messages": msgs,
                                  "max_tokens": max_tok, "temperature": 1.0,
                                  "stream": True,
                                  "stream_options": {"include_usage": True}},
                            timeout=aiohttp.ClientTimeout(total=300),
                        ) as r:
                            if r.status != 200:
                                err = f"HTTP {r.status}: {(await r.text())[:200]}"
                            else:
                                async for raw in r.content:
                                    line = raw.decode().strip()
                                    if not line.startswith("data: "):
                                        continue
                                    d = line[6:]
                                    if d == "[DONE]":
                                        break
                                    try:
                                        c = json.loads(d)
                                        # TTFT: first chunk with content
                                        if not first and c.get("choices"):
                                            if c["choices"][0].get("delta", {}).get("content"):
                                                ttft = time.perf_counter() - t0
                                                first = True
                                        # Token count: usage chunk (final, per OpenAI spec)
                                        # Overwrites any previous chunk-based count
                                        if c.get("usage"):
                                            comp_tok = c["usage"].get("completion_tokens", 0)
                                        elif c.get("choices"):
                                            if c["choices"][0].get("delta", {}).get("content"):
                                                comp_tok += 1
                                    except json.JSONDecodeError:
                                        pass
                except Exception as e:
                    err = str(e)
                t1 = time.perf_counter()
                return RequestMetrics(
                    request_id=idx, start_time=t0, end_time=t1,
                    ttft=ttft, total_time=t1 - t0,
                    prompt_tokens=0, completion_tokens=comp_tok, error=err,
                )

        return list(await asyncio.gather(*[_one(i, m) for i, m in enumerate(prompts)]))

    async def do_rollout_for_training(model, prompts):
        """Non-streaming rollout that returns real TrajectoryGroups for Megatron."""
        import art
        client = model.openai_client()
        inf_name = model.get_inference_name()

        async def _one(idx, msgs):
            try:
                resp = await client.chat.completions.create(
                    model=inf_name, messages=msgs,
                    max_tokens=256, temperature=1.0, logprobs=True,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                reward = min(len(content) / 200.0, 1.0)
                return art.Trajectory(
                    messages_and_choices=[*msgs, choice],
                    reward=reward,
                )
            except Exception as e:
                logger.warning(f"  train-rollout {idx}: {e}")
                return art.Trajectory(
                    messages_and_choices=[msgs[-1], {"role": "assistant", "content": "err"}],
                    reward=0.0,
                )

        sem = asyncio.Semaphore(8)
        async def _bounded(i, m):
            async with sem:
                return await _one(i, m)

        trajs = await asyncio.gather(*[_bounded(i, m) for i, m in enumerate(prompts)])
        groups = []
        for i in range(0, len(trajs), 4):
            batch = list(trajs[i:i+4])
            if len(batch) >= 2:
                rs = [t.reward for t in batch]
                if len(set(rs)) == 1:
                    for j, t in enumerate(batch):
                        t.reward = t.reward + (j + 1) * 0.01
            groups.append(art.TrajectoryGroup(batch))
        return groups

    async def warmup(base_url, model_name, api_key=None, n=4):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        for _ in range(n):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={"model": model_name,
                              "messages": [{"role": "user", "content": "Hi"}],
                              "max_tokens": 8, "temperature": 0},
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as r:
                        await r.read()
            except Exception:
                pass

    # ---- vLLM + Megatron ------------------------------------------

    async def _run_vllm() -> BenchmarkRun:
        import art
        import shutil
        from art.megatron.backend import MegatronBackend

        # Clean stale checkpoints from previous runs
        stale_dir = os.path.join(".art", "sglang-vs-vllm", "models")
        if os.path.exists(stale_dir):
            shutil.rmtree(stale_dir)
            logger.info(f"[vllm] cleaned stale checkpoints at {stale_dir}")

        run = BenchmarkRun(backend="vllm", model=model_id, dataset=dataset)
        run.start_time = time.perf_counter()

        model = art.TrainableModel(
            name="bench-vllm", project="sglang-vs-vllm",
            base_model=model_id,
        )
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=max_seq_length),
            engine_args=art.dev.EngineArgs(
                model=model_id,
                tensor_parallel_size=tp or min(2, torch.cuda.device_count()),
                gpu_memory_utilization=gpu_mem,
                enable_lora=False,
                max_model_len=max_seq_length,  # Avoids defaulting to 262144
                # NOTE: enforce_eager removed — let vLLM use CUDA graphs for
                # fair comparison with SGLang (which also uses CUDA graphs).
                # Startup is slower (~147s extra) but decode throughput is
                # 15-30% higher, which is what we're benchmarking.
            ),
        )

        bk = MegatronBackend()
        t0 = time.perf_counter()
        await model.register(bk, _openai_client_config={
            "server_args": {"port": vllm_port},
        })
        run.server_startup_time = time.perf_counter() - t0

        base_url = model.inference_base_url
        api_key = model.inference_api_key  # Set by register(), typically "default"
        auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        mname = None
        # Retry /v1/models a few times — the server may need a moment after
        # reporting "ready" before it can serve the models endpoint.
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{base_url}/models",
                        headers=auth_headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r:
                        data = await r.json()
                        if data.get("data"):
                            mname = data["data"][0]["id"]
                            logger.info(f"[vllm] /v1/models returned: {[m['id'] for m in data['data']]}")
                            break
                        else:
                            logger.warning(f"[vllm] /v1/models attempt {attempt+1}/5: {data}")
            except Exception as e:
                logger.warning(f"[vllm] /v1/models query attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                await asyncio.sleep(2 * (attempt + 1))
        if not mname:
            # Fallback to inference_model_name set by register() (the ART
            # served_model_name, e.g. "bench-vllm"), or model.name as last resort.
            mname = model.inference_model_name or model.name
            logger.warning(f"[vllm] /v1/models unavailable, falling back to: {mname}")
        model.inference_model_name = mname
        logger.info(f"[vllm] ready in {run.server_startup_time:.0f}s — {mname} @ {base_url}")

        await warmup(base_url, mname, api_key=api_key)

        prompts = generate_benchmark_prompts(num_rollouts, dataset=dataset, seed=seed)

        for step in range(num_steps):
            logger.info(f"[vllm] step {step+1}/{num_steps}")
            sm = StepMetrics(step=step + 1)
            mem = get_gpu_memory_usage_nvidia_smi()
            sm.gpu_memory_during_rollout = sum(mem.values())

            # Timed rollout (streaming for TTFT measurement)
            sm.rollout_start = time.perf_counter()
            sm.request_metrics = await stream_rollout(
                base_url, mname, prompts, max_output_tokens, concurrency,
                api_key=api_key,
            )
            sm.rollout_end = time.perf_counter()

            errs = [r for r in sm.request_metrics if r.error]
            logger.info(f"  rollout {sm.rollout_time:.1f}s  "
                        f"{sm.rollout_throughput:.0f} tok/s  "
                        f"TTFT={sm.avg_ttft:.4f}s  err={len(errs)}")
            if errs:
                # Log first 3 unique errors for debugging
                unique_errs = list(dict.fromkeys(r.error for r in errs))[:3]
                for i, e in enumerate(unique_errs):
                    logger.error(f"  rollout error [{i+1}]: {e}")

            # Train (real Megatron)
            sm.training_start = time.perf_counter()
            tgroups = await do_rollout_for_training(model, prompts)
            try:
                result = await bk.train(model, tgroups, learning_rate=lr,on_policy_correction=True)
                logger.info(f"  train step={result.step} loss={result.metrics.get('loss','?')}")
            except Exception as e:
                logger.error(f"  train failed: {e}", exc_info=True)
                run.errors.append(str(e))
            sm.training_end = time.perf_counter()

            mname = model.inference_model_name or model.name
            run.steps.append(sm)

        run.end_time = time.perf_counter()
        try:
            await bk.close()
        except Exception:
            pass
        return run

    # ---- SGLang + Megatron (verl-style) ----------------------------

    async def _run_sglang() -> BenchmarkRun:
        """SGLang benchmark using verl-style architecture.

        Key differences from old implementation:
          1. Server starts ONCE and NEVER restarts
          2. sleep(kv_cache+weights)/wake_up(kv_cache) for memory management
          3. Merged LoRA weights saved to disk, reloaded via /update_weights
          4. IDENTICAL stream_rollout as vLLM for fair comparison

        This mirrors verl's RayPPOTrainer.fit() loop:
          generate_sequences() → sleep_replicas() → update_actor() →
          update_weights() (includes wake)
        """
        from benchmarks.sglang_vs_vllm.sglang_megatron_backend import SGLangMegatronBackend
        import art

        run = BenchmarkRun(backend="sglang", model=model_id, dataset=dataset)
        run.start_time = time.perf_counter()

        model = art.TrainableModel(
            name="bench-sglang", project="sglang-vs-vllm",
            base_model=model_id,
        )
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=max_seq_length),
        )

        bk = SGLangMegatronBackend(
            sglang_python=sglang_python,
            port=sglang_port,
            tensor_parallel_size=tp or min(2, torch.cuda.device_count()),
            gpu_memory_utilization=gpu_mem,
        )

        # Phase: Start SGLang server ONCE (verl: launch_server, called once)
        t0 = time.perf_counter()
        await model.register(bk)
        run.server_startup_time = time.perf_counter() - t0

        base_url = model.inference_base_url
        api_key = model.inference_api_key
        mname = model.get_inference_name()
        logger.info(
            f"[sglang] ready in {run.server_startup_time:.0f}s — "
            f"{mname} @ {base_url} (verl-style, will NOT restart)"
        )

        await warmup(base_url, mname, api_key=api_key)

        prompts = generate_benchmark_prompts(num_rollouts, dataset=dataset, seed=seed)

        for step in range(num_steps):
            logger.info(f"[sglang] step {step+1}/{num_steps} (verl-style)")
            sm = StepMetrics(step=step + 1)
            mem = get_gpu_memory_usage_nvidia_smi()
            sm.gpu_memory_during_rollout = sum(mem.values())

            # ---- Rollout phase (verl: generate_sequences) ----
            # IDENTICAL to vLLM: single streaming rollout with include_usage
            # for accurate token counts + TTFT. One rollout, fair comparison.
            sm.rollout_start = time.perf_counter()
            sm.request_metrics = await stream_rollout(
                base_url, mname, prompts, max_output_tokens, concurrency,
                api_key=api_key,
            )
            sm.rollout_end = time.perf_counter()

            errs = [r for r in sm.request_metrics if r.error]
            logger.info(
                f"  rollout {sm.rollout_time:.1f}s  "
                f"{sm.rollout_throughput:.0f} tok/s  "
                f"TTFT={sm.avg_ttft:.4f}s  err={len(errs)}"
            )
            if errs:
                unique_errs = list(dict.fromkeys(r.error for r in errs))[:3]
                for i, e in enumerate(unique_errs):
                    logger.error(f"  rollout error [{i+1}]: {e}")

            # ---- Training phase (verl: sleep → train → update_weights → wake) ----
            sm.training_start = time.perf_counter()
            tgroups = await do_rollout_for_training(model, prompts)
            try:
                # bk.train() internally calls service.train() which does:
                #   sleep(kv_cache+weights) → megatron train →
                #   update_weights(disk) → wake_up(kv_cache)
                # This is the verl-style loop — NO server restart
                result = await bk.train(model, tgroups, learning_rate=lr, on_policy_correction=True)
                logger.info(f"  train step={result.step} loss={result.metrics.get('loss','?')}")
            except Exception as e:
                logger.error(f"  train failed: {e}", exc_info=True)
                run.errors.append(str(e))
            sm.training_end = time.perf_counter()

            # Re-fetch model name — after training, LoRA adapter name is active
            mname = model.get_inference_name()
            run.steps.append(sm)

        run.end_time = time.perf_counter()
        try:
            await bk.close()
        except Exception:
            pass
        return run

    # ---- dispatch --------------------------------------------------

    async def _main():
        fn = _run_vllm if backend == "vllm" else _run_sglang
        result = await fn()
        with open(results_path, "w") as f:
            json.dump(result.summary(), f, indent=2)
        logger.info(f"[{backend}] Results → {results_path}")

    asyncio.run(_main())


# ===================================================================
# Orchestrator
# ===================================================================

def cleanup_gpus() -> None:
    for pat in ["model-service", "megatron-service", "sglang.launch_server",
                "vllm.entrypoints", "vllm.v1", "torchrun"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True, timeout=10)
    # Kill any remaining GPU-holding processes (except this one)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        my_pid = str(os.getpid())
        for pid in r.stdout.strip().split("\n"):
            pid = pid.strip()
            if pid and pid != my_pid:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
    except Exception:
        pass

def spawn_worker(backend: str, cfg: dict, results_path: str) -> int:
    script = os.path.abspath(__file__)
    cfg_file = results_path.replace("_metrics.json", "_config.json")
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    cmd = ["uv", "run", "python", script,
           "--_worker", backend, "--_config", cfg_file, "--_results", results_path]
    logger.info(f"Spawning {backend}: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    # Suppress NCCL/TCPStore noise from Megatron shutdown — send stderr to log file
    stderr_log = results_path.replace("_metrics.json", "_stderr.log")
    with open(stderr_log, "w") as stderr_file:
        proc = subprocess.run(cmd, env=env, stderr=stderr_file)
    logger.info(f"  stderr log: {stderr_log}")

    if proc.returncode in (-9, 137):
        logger.error(f"{backend} OOM-killed. Try --gpu-memory-utilization 0.5")
    elif proc.returncode != 0:
        logger.error(f"{backend} exited with code {proc.returncode}")
    return proc.returncode


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SGLang vs vLLM + Megatron benchmark (verl-style)")
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_results", help=argparse.SUPPRESS)

    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                   help="Qwen3 MoE model (required by Megatron)")
    p.add_argument("--dataset", default="agentic",
                   choices=["gsm8k", "sharegpt", "agentic", "math", "synthetic"])
    p.add_argument("--backends", nargs="+", default=["vllm", "sglang"],
                   choices=["vllm", "sglang"])
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--num-rollouts", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--output", default="benchmark_results")
    p.add_argument("--sglang-python", default="")
    p.add_argument("--vllm-port", type=int, default=8100)
    p.add_argument("--sglang-port", type=int, default=8200)
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Worker mode (subprocess) ---------------------------------
    if args._worker:
        with open(args._config) as f:
            cfg = json.load(f)
        run_worker(args._worker, cfg, args._results)
        return

    # ---- Orchestrator mode ----------------------------------------
    from benchmarks.sglang_vs_vllm.metrics_collector import (
        BenchmarkRun, StepMetrics, RequestMetrics,
        generate_comparison_report,
    )

    os.makedirs(args.output, exist_ok=True)

    # Find SGLang python
    sglang_python = args.sglang_python
    if not sglang_python:
        for candidate in [
            os.path.expanduser("~/.venvs/sglang-bench/bin/python"),
            os.path.expanduser("~/sglang-env/bin/python"),
        ]:
            if os.path.isfile(candidate):
                sglang_python = candidate
                break
        else:
            sglang_python = "python"

    # Config dict passed to workers (flat, JSON-serializable)
    cfg = {
        "model": args.model,
        "dataset": args.dataset,
        "num_steps": args.num_steps,
        "num_rollouts": args.num_rollouts,
        "concurrency": args.concurrency,
        "max_output_tokens": args.max_output_tokens,
        "max_seq_length": args.max_seq_length,
        "tp": args.tp,
        "gpu_mem": args.gpu_memory_utilization,
        "vllm_port": args.vllm_port,
        "sglang_port": args.sglang_port,
        "sglang_python": sglang_python,
        "seed": 42,
        "learning_rate": args.learning_rate,
        "output_dir": args.output,
    }

    logger.info("=" * 60)
    logger.info("  SGLang + Megatron  vs  vLLM + Megatron  (verl-style)")
    logger.info("=" * 60)
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")

    results = {}
    for backend in args.backends:
        results_file = os.path.join(args.output, f"{backend}_metrics.json")
        logger.info(f"\n{'='*60}\n  {backend.upper()} subprocess\n{'='*60}")
        cleanup_gpus()
        rc = spawn_worker(backend, cfg, results_file)
        if rc == 0 and os.path.exists(results_file):
            with open(results_file) as f:
                results[backend] = json.load(f)
            logger.info(f"  {backend} results collected")
        cleanup_gpus()

    # Report
    if "vllm" in results and "sglang" in results:
        vr = _dict_to_run(results["vllm"])
        sr = _dict_to_run(results["sglang"])
        print("\n" + generate_comparison_report(vr, sr, args.output))
    elif results:
        for n, d in results.items():
            print(f"\n{n}: {json.dumps(d, indent=2)}")
    else:
        logger.error("No results!")


def _dict_to_run(d: dict):
    from benchmarks.sglang_vs_vllm.metrics_collector import BenchmarkRun, StepMetrics, RequestMetrics
    run = BenchmarkRun(backend=d["backend"], model=d["model"],
                       dataset=d.get("dataset", ""), server_startup_time=d.get("server_startup_s", 0))
    run.start_time = 0.0
    run.end_time = d.get("total_time_s", 0)
    for sd in d.get("steps", []):
        sm = StepMetrics(step=sd["step"])
        sm.rollout_start = 0.0
        sm.rollout_end = sd.get("rollout_time_s", 0)
        sm.gpu_memory_during_rollout = sd.get("gpu_mem_gb", 0) * 1e9
        n = sd.get("num_requests", 1)
        thru = sd.get("throughput_tok_s", 0)
        rt = sd.get("rollout_time_s", 1)
        for i in range(n):
            sm.request_metrics.append(RequestMetrics(
                request_id=i, start_time=0, end_time=sd.get("avg_latency_s", 0),
                ttft=sd.get("avg_ttft_s", 0), total_time=sd.get("avg_latency_s", 0),
                prompt_tokens=0, completion_tokens=int(thru * rt / max(n, 1)),
            ))
        run.steps.append(sm)
    return run


if __name__ == "__main__":
    main()
