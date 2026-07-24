#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
from typing import Any

import httpx

import art
from art.megatron.service import MegatronService
from art.megatron.weights.update_replay import inspect_adapter

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"


def _hash_request(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 64:
        raise ValueError(f"Expected 64 pinned request rows, got {len(rows)}")
    ids: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        request = row["request"]
        request["model"] = "bonnie-replay:active"
        expected = {
            "n": 8,
            "temperature": 0.8,
            "max_tokens": 256,
            "logprobs": True,
            "stream": True,
        }
        for key, value in expected.items():
            if request.get(key) != value:
                raise ValueError(
                    f"Request {row.get('id')} has {key}={request.get(key)}"
                )
        if request.get("chat_template_kwargs") != {"enable_thinking": False}:
            raise ValueError(f"Request {row.get('id')} enables thinking")
        row["request_hash"] = _hash_request(request)
        ids.add(str(row["id"]))
        hashes.add(row["request_hash"])
    if len(ids) != 64 or len(hashes) != 64:
        raise ValueError("Pinned request trace contains duplicate IDs or requests")
    return rows


async def _load_worker(
    *,
    client: httpx.AsyncClient,
    requests: list[dict[str, Any]],
    worker: int,
    stop: asyncio.Event,
    samples: list[dict[str, Any]],
    phase: list[str],
) -> None:
    index = worker
    while not stop.is_set():
        row = requests[index % len(requests)]
        index += 8
        started = time.perf_counter()
        error = None
        prompt_tokens = completion_tokens = 0
        ttft_s: float | None = None
        last_token_time_by_choice: dict[int, float] = {}
        inter_token_intervals_s: list[float] = []
        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=row["request"]
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line.removeprefix("data: "))
                    usage = chunk.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(
                        usage.get("completion_tokens", completion_tokens)
                    )
                    for choice in chunk.get("choices") or []:
                        choice_index = int(choice.get("index", 0))
                        logprobs = choice.get("logprobs") or {}
                        emitted = len(logprobs.get("content") or [])
                        if emitted == 0:
                            delta = choice.get("delta") or {}
                            emitted = int(
                                bool(delta.get("content") or delta.get("tool_calls"))
                            )
                        if emitted:
                            now = time.perf_counter()
                            if ttft_s is None:
                                ttft_s = now - started
                            previous = last_token_time_by_choice.get(choice_index)
                            if previous is not None:
                                inter_token_intervals_s.append(now - previous)
                            if emitted > 1:
                                inter_token_intervals_s.extend([0.0] * (emitted - 1))
                            last_token_time_by_choice[choice_index] = now
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        samples.append(
            {
                "worker": worker,
                "phase": phase[0],
                "id": row["id"],
                "request_hash": row["request_hash"],
                "latency_s": time.perf_counter() - started,
                "ttft_s": ttft_s,
                "inter_token_intervals_s": inter_token_intervals_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": error,
            }
        )


async def _record_control(
    *,
    name: str,
    duration_s: float,
    phase: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    phase[0] = name
    sample_start = len(samples)
    started = time.perf_counter()
    await asyncio.sleep(duration_s)
    elapsed_s = time.perf_counter() - started
    sample_end = len(samples)
    lane = samples[sample_start:sample_end]
    completion_tokens = sum(int(row["completion_tokens"]) for row in lane)
    return {
        "name": name,
        "duration_s": elapsed_s,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "request_count": len(lane),
        "request_errors": sum(row["error"] is not None for row in lane),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in lane),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s": completion_tokens / elapsed_s,
    }


def _bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(20260723)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(2000)]
    means.sort()
    return [means[50], means[1949]]


def _gpu_memory_snapshot() -> list[dict[str, int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[dict[str, int | str]] = []
    for line in output.splitlines():
        raw_index, name, raw_used, raw_total = (
            part.strip() for part in line.split(",", maxsplit=3)
        )
        rows.append(
            {
                "gpu_index": int(raw_index),
                "name": name,
                "used_mib": int(raw_used),
                "total_mib": int(raw_total),
            }
        )
    return rows


async def run(args: argparse.Namespace) -> None:
    requests = _load_requests(args.request_jsonl)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(tp=1, pp=1, cp=2, ep=2, etp=1),
        packed_sequence_length=122880,
        streaming_weight_offload=True,
    )
    service = MegatronService(
        model_name="bonnie-replay",
        base_model=BASE_MODEL,
        config={
            "trainer_gpu_ids": [0, 1],
            "inference_gpu_ids": [2, 3],
            "rollout_weights_mode": "lora",
            "rollout_weight_update_mode": "in_flight_lora",
            "allow_unvalidated_arch": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "lora_config": {"rank": 8},
            "engine_args": {
                "tensor_parallel_size": 2,
                "max_model_len": 32769,
                "max_num_batched_tokens": 131072,
                "max_num_seqs": 256,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
            },
        },
        output_dir=str(output / "service"),
    )
    seed = Path(service._resolve_active_lora_path())
    await service.start_openai_server(None)
    base_url = service._vllm_base_url
    headers = service._runtime_headers()
    if service._in_flight_lora_slot != "bonnie-replay:active":
        raise RuntimeError("Replay slot name changed unexpectedly")
    samples: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    phase = ["idle"]
    stop = asyncio.Event()
    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, timeout=300.0
    ) as client:
        runtime_before = (await client.get("/art/metrics")).json()
        if int(runtime_before.get("metrics", {}).get("world_size", 0)) != 2:
            raise RuntimeError(f"Expected vLLM TP2 runtime: {runtime_before!r}")
        load_tasks = (
            [
                asyncio.create_task(
                    _load_worker(
                        client=client,
                        requests=requests,
                        worker=worker,
                        stop=stop,
                        samples=samples,
                        phase=phase,
                    )
                )
                for worker in range(8)
            ]
            if args.mode == "fixed-load"
            else []
        )
        updates: list[dict[str, Any]] = []
        hashes: set[str] = set()
        manifest_sha256: str | None = None
        try:
            if load_tasks:
                controls.append(
                    await _record_control(
                        name="control-before",
                        duration_s=args.control_seconds,
                        phase=phase,
                        samples=samples,
                    )
                )
            phase[0] = "updates"
            for index in range(args.warmups + args.updates):
                policy_version = index + 1
                checkpoint = output / "checkpoints" / f"{policy_version:04d}"
                sample_start = len(samples)
                memory_before = _gpu_memory_snapshot()
                update_started = time.perf_counter()
                metrics = await service.replay_lora_update(
                    source_lora_path=str(seed),
                    output_lora_path=str(checkpoint),
                    policy_version=policy_version,
                    content_version=index,
                )
                update_wall_s = time.perf_counter() - update_started
                expected_topology = {
                    "topology/trainer_tp": 1.0,
                    "topology/trainer_pp": 1.0,
                    "topology/trainer_cp": 2.0,
                    "topology/trainer_ep": 2.0,
                    "topology/trainer_etp": 1.0,
                }
                if {
                    key: metrics.get(key) for key in expected_topology
                } != expected_topology:
                    raise RuntimeError(f"Unexpected trainer topology: {metrics!r}")
                if len(memory_before) != 4:
                    raise RuntimeError(
                        f"Expected four visible H200s, got {memory_before!r}"
                    )
                if any("H200" not in str(row["name"]) for row in memory_before):
                    raise RuntimeError(f"Replay requires H200 GPUs: {memory_before!r}")
                snapshot = inspect_adapter(checkpoint)
                if snapshot.file_sha256 in hashes:
                    raise RuntimeError("Replay produced duplicate update contents")
                if manifest_sha256 is None:
                    manifest_sha256 = snapshot.manifest_sha256
                elif snapshot.manifest_sha256 != manifest_sha256:
                    raise RuntimeError("Replay changed the serving tensor manifest")
                hashes.add(snapshot.file_sha256)
                updates.append(
                    {
                        "index": index,
                        "measured": index >= args.warmups,
                        "policy_version": policy_version,
                        "checkpoint": str(checkpoint),
                        "sha256": snapshot.file_sha256,
                        "manifest_sha256": snapshot.manifest_sha256,
                        "metrics": metrics,
                        "update_wall_s": update_wall_s,
                        "gpu_memory_before": memory_before,
                        "gpu_memory_after": _gpu_memory_snapshot(),
                        "sample_start": sample_start,
                        "sample_end": len(samples),
                    }
                )
                if len(updates) > 2:
                    shutil.rmtree(Path(updates[-3]["checkpoint"]))
            if load_tasks:
                controls.append(
                    await _record_control(
                        name="control-after",
                        duration_s=args.control_seconds,
                        phase=phase,
                        samples=samples,
                    )
                )
                before_rate = float(controls[0]["completion_tokens_per_s"])
                after_rate = float(controls[-1]["completion_tokens_per_s"])
                drift = (
                    abs(after_rate - before_rate) / before_rate
                    if before_rate > 0
                    else float("inf")
                )
                if drift > 0.05:
                    controls.append(
                        await _record_control(
                            name="control-repeat",
                            duration_s=args.control_seconds,
                            phase=phase,
                            samples=samples,
                        )
                    )
        finally:
            stop.set()
            await asyncio.gather(*load_tasks)
        runtime_after = (await client.get("/art/metrics")).json()
    await service.aclose()

    measured = [row for row in updates if row["measured"]]
    totals = [
        float(row["metrics"]["time/weight_update_trainer_publish_s"])
        + float(row["metrics"]["time/weight_update_service_rpc_s"])
        for row in measured
    ]
    measured_sample_rows = [
        sample
        for update in measured
        for sample in samples[update["sample_start"] : update["sample_end"]]
    ]
    (output / "updates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in updates)
    )
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples)
    )
    (output / "runtime-before.json").write_text(json.dumps(runtime_before, indent=2))
    (output / "runtime-after.json").write_text(json.dumps(runtime_after, indent=2))
    topology = {
        "trainer_gpu_ids": [0, 1],
        "inference_gpu_ids": [2, 3],
        "inference_tp": 2,
        "trainer": {"tp": 1, "pp": 1, "cp": 2, "ep": 2, "etp": 1},
    }
    (output / "topology.json").write_text(json.dumps(topology, indent=2))
    reference_snapshot = inspect_adapter(Path(updates[-1]["checkpoint"]))
    tensor_manifest = {
        "schema_version": 1,
        "manifest_sha256": reference_snapshot.manifest_sha256,
        "rank": reference_snapshot.rank,
        "base_model": reference_snapshot.base_model,
        "logical_bytes": reference_snapshot.logical_bytes,
        "transported_bytes": reference_snapshot.transported_bytes,
        "tensor_count": reference_snapshot.tensor_count,
        "tensors": [asdict(tensor) for tensor in reference_snapshot.tensors],
    }
    (output / "tensor-manifest.json").write_text(json.dumps(tensor_manifest, indent=2))
    request_trace = {
        "schema_version": 1,
        "source_artifact": "wb-training/bench/bonnie:v1",
        "source_train_jsonl_sha256": (
            "fd26c97423de94a158d5a54e38485ec55b3fd3504f5f0346850fcfa79ad3ce66"
        ),
        "selection": "first 64 non-empty train.jsonl rows in artifact order",
        "requests": requests,
    }
    (output / "request-trace.json").write_text(
        json.dumps(request_trace, indent=2, sort_keys=True)
    )
    summary = {
        "update_total_mean_s": statistics.fmean(totals),
        "update_total_p50_s": statistics.median(totals),
        "update_total_p95_s": sorted(totals)[round(0.95 * (len(totals) - 1))],
        "update_total_mean_95ci_s": _bootstrap_ci(totals),
        "update_total_stddev_s": statistics.stdev(totals),
        "prompt_tokens": sum(row["prompt_tokens"] for row in samples),
        "completion_tokens": sum(row["completion_tokens"] for row in samples),
        "request_latency_mean_s": (
            statistics.fmean(row["latency_s"] for row in samples) if samples else None
        ),
        "request_errors": sum(row["error"] is not None for row in samples),
        "controls": controls,
    }
    ttfts = [
        float(row["ttft_s"])
        for row in samples
        if row["ttft_s"] is not None and row["error"] is None
    ]
    itls = [
        float(interval)
        for row in samples
        if row["error"] is None
        for interval in row["inter_token_intervals_s"]
    ]
    if samples:
        summary["ttft_p50_s"] = statistics.median(ttfts) if ttfts else None
        summary["ttft_p95_s"] = (
            sorted(ttfts)[round(0.95 * (len(ttfts) - 1))] if ttfts else None
        )
        summary["inter_token_latency_p50_s"] = statistics.median(itls) if itls else None
        summary["inter_token_latency_p95_s"] = (
            sorted(itls)[round(0.95 * (len(itls) - 1))] if itls else None
        )
    if controls:
        invalid_controls = [
            row for row in controls if not row["request_count"] or row["request_errors"]
        ]
        if invalid_controls:
            raise RuntimeError(f"Invalid no-update control lanes: {invalid_controls}")
        control_rate = statistics.fmean(
            float(row["completion_tokens_per_s"]) for row in controls
        )
        measured_lane_s = sum(float(row["update_wall_s"]) for row in measured)
        measured_completion_tokens = sum(
            int(row["completion_tokens"]) for row in measured_sample_rows
        )
        summary["lost_tokens_per_update"] = (
            control_rate * measured_lane_s - measured_completion_tokens
        ) / len(measured)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "art_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "request_trace": str(args.request_jsonl),
        "request_ids_and_hashes": [
            {"id": row["id"], "request_hash": row["request_hash"]} for row in requests
        ],
        "routed_expert_collection": {
            "supported": False,
            "reason": "The replay uses vLLM's OpenAI JSON endpoint.",
        },
        "artifacts": [
            "updates.jsonl",
            "samples.jsonl",
            "runtime-before.json",
            "runtime-after.json",
            "topology.json",
            "tensor-manifest.json",
            "request-trace.json",
            "summary.json",
            "stdout.log",
            "stderr.log",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if summary["request_errors"]:
        raise RuntimeError(
            f"Evidence run observed {summary['request_errors']} request errors"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=("idle", "fixed-load"), required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--control-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
