#!/usr/bin/env python3
"""
End-to-end benchmark: Unsloth + SGLang.

Unsloth path uses SGLang for inference + Unsloth for MoE training:
  - verl-style SGLang server (persistent, sleep/wake)
  - ~12x faster MoE training via Unsloth Triton kernels
  - ~35% less VRAM via Split LoRA approach
  - LoRA hot-reload for weight sync (<2s)

Each step: rollout (timed) → train → next rollout with updated weights.
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

    from benchmarks.sglang_benchmarks.metrics_collector import (
        BenchmarkRun, RequestMetrics, StepMetrics,
        get_gpu_memory_usage_nvidia_smi,
    )
    from benchmarks.sglang_benchmarks.config import generate_benchmark_prompts

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
        """Streaming rollout for TTFT measurement.

        Uses stream_options.include_usage=true to get accurate server-side
        token counts in the final SSE chunk, while also measuring TTFT
        from the first content chunk.
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
                                        if not first and c.get("choices"):
                                            if c["choices"][0].get("delta", {}).get("content"):
                                                ttft = time.perf_counter() - t0
                                                first = True
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
        """Non-streaming rollout that returns real TrajectoryGroups for training."""
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

    # ---- Unsloth + SGLang (MoE-optimized, self-contained) -----------

    async def _run_unsloth() -> BenchmarkRun:
        """Unsloth + SGLang benchmark.

        Architecture:
          1. SGLang server starts ONCE (persistent, verl-style)
          2. Rollout via streaming (timed)
          3. do_rollout_for_training creates TrajectoryGroups
          4. ART preprocessing tokenizes/packs into packed tensors
          5. sleep() → Unsloth trains on packed tensors with ART loss →
             wake() → load_lora()
          6. Full memory recovery every step

        Reference: https://unsloth.ai/docs/new/faster-moe
        """
        import math as _math
        from transformers import AutoTokenizer
        from art.preprocessing.tokenize import tokenize_trajectory_groups
        from art.preprocessing.pack import (
            packed_tensors_from_tokenized_results,
            packed_tensors_to_dir,
        )
        from benchmarks.sglang_benchmarks.unsloth_sglang_service import UnslothSGLangService

        unsloth_port = cfg.get("unsloth_port", 8300)
        unsloth_lora_rank = cfg.get("unsloth_lora_rank", 1)
        unsloth_moe_backend = cfg.get("unsloth_moe_backend", "auto")

        # GPU split — None means auto-detect in UnslothSGLangService
        inference_gpus = cfg.get("inference_gpus")
        training_gpus = cfg.get("training_gpus")

        svc = UnslothSGLangService(
            model_name="bench-unsloth",
            base_model=model_id,
            output_dir=os.path.join(output_dir, "unsloth_workdir"),
            sglang_python=sglang_python,
            port=unsloth_port,
            tensor_parallel_size=tp or min(2, torch.cuda.device_count()),
            gpu_memory_utilization=gpu_mem,
            max_running_requests=256,
            lora_rank=unsloth_lora_rank,
            max_seq_length=max_seq_length,
            learning_rate=lr,
            moe_backend=unsloth_moe_backend,
            inference_gpus=inference_gpus,
            training_gpus=training_gpus,
        )

        run = BenchmarkRun(backend="unsloth", model=model_id, dataset=dataset)
        run.start_time = time.perf_counter()

        # Start SGLang ONCE
        t0 = time.perf_counter()
        await svc.start()
        run.server_startup_time = time.perf_counter() - t0

        base_url = svc.base_url
        mname = svc.inference_model_name
        logger.info(
            f"[unsloth] ready in {run.server_startup_time:.0f}s — "
            f"{mname} @ {base_url} (Unsloth MoE + SGLang, verl-style)"
        )

        await warmup(base_url, mname)

        prompts = generate_benchmark_prompts(num_rollouts, dataset=dataset, seed=seed)

        _tokenizer = AutoTokenizer.from_pretrained(model_id)

        class _ModelAdapter:
            def openai_client(self_):
                from openai import AsyncOpenAI
                return AsyncOpenAI(base_url=base_url, api_key="none")
            def get_inference_name(self_):
                return svc.inference_model_name

        _adapter = _ModelAdapter()

        for step in range(num_steps):
            logger.info(f"[unsloth] step {step+1}/{num_steps}")
            sm = StepMetrics(step=step + 1)
            mem = get_gpu_memory_usage_nvidia_smi()
            sm.gpu_memory_during_rollout = sum(mem.values())

            # ---- Rollout (streaming) ----
            mname = svc.inference_model_name
            sm.rollout_start = time.perf_counter()
            sm.request_metrics = await stream_rollout(
                base_url, mname, prompts, max_output_tokens, concurrency,
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

            # ---- Data pipeline (ART preprocessing) ----
            sm.training_start = time.perf_counter()

            tgroups = await do_rollout_for_training(_adapter, prompts)
            n_trajs = sum(len(g.trajectories) for g in tgroups)
            logger.info(f"  collected {n_trajs}/{len(prompts)} trajectories for GRPO training")

            tokenized = list(tokenize_trajectory_groups(
                _tokenizer, tgroups,
                allow_training_without_logprobs=True,
                scale_rewards=True,
            ))

            if not tokenized:
                logger.warning("  no valid tokenized results — skipping training")
                sm.training_end = time.perf_counter()
                run.steps.append(sm)
                continue

            max_tokens = max(len(r.token_ids) for r in tokenized)
            seq_len = min(
                _math.ceil(max_tokens / 2048) * 2048,
                max_seq_length,
            )

            packed = packed_tensors_from_tokenized_results(
                tokenized, seq_len,
                pad_token_id=_tokenizer.eos_token_id or 0,
            )

            pt_dir = os.path.join(
                output_dir, "unsloth_workdir", "packed_tensors", f"step{step+1:04d}",
            )
            disk_info = packed_tensors_to_dir(packed, pt_dir)
            logger.info(
                f"  packed: {disk_info['num_sequences']} seqs × "
                f"{disk_info['sequence_length']} tokens → {pt_dir}"
            )

            # ---- Training (sleep → ART loss on packed tensors → wake → load_lora) ----
            try:
                train_result = await svc.train_step(
                    packed_tensors_dir=pt_dir,
                    num_sequences=disk_info["num_sequences"],
                    sequence_length=disk_info["sequence_length"],
                    lr=lr,
                )
                logger.info(
                    f"  train loss={train_result.get('loss', '?'):.4f}  "
                    f"overhead={train_result.get('total_overhead_s', 0):.1f}s  "
                    f"(ART loss, packed tensors)"
                )
            except Exception as e:
                logger.error(f"  train failed: {e}", exc_info=True)
                run.errors.append(str(e))
            sm.training_end = time.perf_counter()

            run.steps.append(sm)

            # Write partial results after each step so they survive OOM-kill
            run.end_time = time.perf_counter()
            with open(results_path, "w") as f:
                json.dump(run.summary(), f, indent=2)

        try:
            await svc.stop()
        except Exception:
            pass
        return run

    # ---- dispatch --------------------------------------------------

    async def _main():
        if backend != "unsloth":
            raise ValueError(f"Unknown backend: {backend}")
        result = await _run_unsloth()
        with open(results_path, "w") as f:
            json.dump(result.summary(), f, indent=2)
        logger.info(f"[{backend}] Results → {results_path}")

    asyncio.run(_main())


# ===================================================================
# Orchestrator
# ===================================================================

def cleanup_gpus() -> None:
    my_pid = os.getpid()
    my_ppid = os.getppid()
    safe_pids = {my_pid, my_ppid}

    kill_patterns = [
        "sglang.launch_server",
        "torchrun",
        "_worker unsloth",
    ]
    for pat in kill_patterns:
        try:
            r = subprocess.run(
                ["pgrep", "-f", pat], capture_output=True, text=True, timeout=10,
            )
            for pid_str in r.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str and int(pid_str) not in safe_pids:
                    subprocess.run(["kill", "-9", pid_str], capture_output=True, timeout=5)
        except Exception:
            pass

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for pid in r.stdout.strip().split("\n"):
            pid = pid.strip()
            if pid and int(pid) not in safe_pids:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
    except Exception:
        pass

def _find_python_with_torch() -> str:
    """Find a Python interpreter that can import torch."""
    candidates = [
        sys.executable,
        "/usr/bin/python3",
        os.path.expanduser("~/.venvs/art/bin/python"),
        "/opt/conda/bin/python",
    ]
    try:
        r = subprocess.run(
            ["which", "-a", "python3"], capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().split("\n"):
            p = line.strip()
            if p and p not in candidates and ".venv" not in p:
                candidates.append(p)
    except Exception:
        pass

    for python in candidates:
        if not os.path.isfile(python):
            continue
        try:
            r = subprocess.run(
                [python, "-c", "import torch; print(torch.__version__)"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                ver = r.stdout.strip()
                logger.info(f"Worker python: {python} (torch {ver})")
                return python
        except Exception:
            continue

    logger.warning(
        f"No Python with torch found! Falling back to {sys.executable}. "
        f"Install torch: pip install torch"
    )
    return sys.executable


def spawn_worker(backend: str, cfg: dict, results_path: str) -> int:
    script = os.path.abspath(__file__)
    cfg_file = results_path.replace("_metrics.json", "_config.json")
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    python = _find_python_with_torch()
    cmd = [python, script,
           "--_worker", backend, "--_config", cfg_file, "--_results", results_path]
    logger.info(f"Spawning {backend}: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    extra_paths = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))

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
    p = argparse.ArgumentParser(
        description="Unsloth + SGLang benchmark"
    )
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_results", help=argparse.SUPPRESS)

    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                   help="Qwen3 MoE model")
    p.add_argument("--dataset", default="agentic",
                   choices=["gsm8k", "sharegpt", "agentic", "math", "synthetic"])
    p.add_argument("--backends", nargs="+", default=["unsloth"],
                   choices=["unsloth"])
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--num-rollouts", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--output", default="benchmark_results")
    p.add_argument("--sglang-python", default="")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--learning-rate", type=float, default=5e-6)

    # Unsloth-specific options
    p.add_argument("--unsloth-port", type=int, default=8300,
                   help="Port for Unsloth+SGLang inference server")
    p.add_argument("--unsloth-lora-rank", type=int, default=1,
                   help="LoRA rank for Unsloth training (default=1)")
    p.add_argument("--unsloth-moe-backend", default="auto",
                   choices=["auto", "grouped_mm", "unsloth_triton", "native_torch"],
                   help="Unsloth MoE backend: grouped_mm (H100+), unsloth_triton (A100), native_torch (fallback)")

    # GPU split — dedicated inference/training GPUs (auto-detected if not set)
    p.add_argument("--inference-gpus", type=str, default="",
                   help="Comma-separated GPU IDs for SGLang inference (e.g. '0,2,3'). "
                        "Auto-detected if not set: all GPUs except --training-gpu.")
    p.add_argument("--training-gpus", type=str, default="",
                   help="Comma-separated GPU IDs for Unsloth training (e.g. '1,3' for DDP). "
                        "Auto-detected if not set. Use '-1' to force shared mode (sleep/wake).")
    # Backward compat alias
    p.add_argument("--training-gpu", type=int, default=None,
                   help=argparse.SUPPRESS)
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
    from benchmarks.sglang_benchmarks.metrics_collector import (
        BenchmarkRun, StepMetrics, RequestMetrics,
        generate_comparison_report_multi,
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

    # Parse GPU split args
    inference_gpus = None
    if args.inference_gpus:
        inference_gpus = [int(g.strip()) for g in args.inference_gpus.split(",")]

    # training_gpus: new comma-separated arg, or backward-compat single --training-gpu
    training_gpus = None
    if args.training_gpus:
        training_gpus = [int(g.strip()) for g in args.training_gpus.split(",")]
    elif args.training_gpu is not None:
        training_gpus = [args.training_gpu]  # backward compat: single int → list

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
        "sglang_python": sglang_python,
        "seed": 42,
        "learning_rate": args.learning_rate,
        "output_dir": args.output,
        # Unsloth-specific
        "unsloth_port": args.unsloth_port,
        "unsloth_lora_rank": args.unsloth_lora_rank,
        "unsloth_moe_backend": args.unsloth_moe_backend,
        # GPU split (None = auto-detect)
        "inference_gpus": inference_gpus,
        "training_gpus": training_gpus,
    }

    backends_str = " + ".join(b.upper() for b in args.backends)
    logger.info("=" * 60)
    logger.info(f"  {backends_str}  benchmark")
    logger.info("=" * 60)
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")

    results = {}
    for backend in args.backends:
        results_file = os.path.join(args.output, f"{backend}_metrics.json")
        logger.info(f"\n{'='*60}\n  {backend.upper()} subprocess\n{'='*60}")
        cleanup_gpus()
        rc = spawn_worker(backend, cfg, results_file)
        if os.path.exists(results_file):
            with open(results_file) as f:
                results[backend] = json.load(f)
            if rc != 0:
                logger.warning(f"  {backend} exited with code {rc} but results file exists — using it")
            logger.info(f"  {backend} results collected")
        cleanup_gpus()

    # Report
    runs = {name: _dict_to_run(data) for name, data in results.items()}

    if len(runs) >= 2:
        print("\n" + generate_comparison_report_multi(runs, args.output))
    elif len(runs) == 1:
        for n, d in results.items():
            print(f"\n{n}: {json.dumps(d, indent=2)}")
    else:
        logger.error("No results!")


def _dict_to_run(d: dict):
    from benchmarks.sglang_benchmarks.metrics_collector import BenchmarkRun, StepMetrics, RequestMetrics
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
