#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import statistics
import subprocess
import time
from typing import Any

import httpx

import art
from art.megatron.service import MegatronService
from art.megatron.weights.update_replay import (
    evaluate_replay_acceptance,
    inspect_adapter,
    validate_bench_request_trace,
)

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _effective_requests(
    request_jsonl: Path, trace_manifest: Path, *, bench_commit: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, provenance = validate_bench_request_trace(
        request_jsonl, trace_manifest, bench_commit=bench_commit
    )
    generation = provenance["manifest"].get("generation", {})
    expected = {
        "n": 8,
        "temperature": 0.8,
        "max_tokens": 256,
        "logprobs": True,
        "enable_thinking": False,
        "load_concurrency": 8,
    }
    if {key: generation.get(key) for key in expected} != expected:
        raise ValueError(f"Unexpected Bench generation contract: {generation!r}")
    for row in rows:
        effective = dict(row["request"])
        effective.update(
            {
                "model": "bonnie-replay:active",
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        row["effective_request"] = effective
        row["effective_request_sha256"] = _canonical_sha256(effective)
    return rows, provenance


async def _runtime_metrics(client: httpx.AsyncClient) -> dict[str, float]:
    response = await client.get("/art/metrics")
    response.raise_for_status()
    return {
        key: float(value)
        for key, value in response.json().get("metrics", {}).items()
        if isinstance(value, (int, float))
    }


async def _load_worker(
    *,
    client: httpx.AsyncClient,
    requests: list[dict[str, Any]],
    worker: int,
    stop: asyncio.Event,
    samples: list[dict[str, Any]],
    token_events: list[dict[str, Any]],
    phase: list[str],
) -> None:
    sequence = 0
    index = worker
    while not stop.is_set():
        row = requests[index % len(requests)]
        index += 8
        request_id = f"load-{worker}-{sequence:06d}"
        sequence += 1
        started = time.perf_counter()
        phase_at_start = phase[0]
        error = None
        prompt_tokens = completion_tokens = 0
        ttft_s: float | None = None
        last_token_time_by_choice: dict[int, float] = {}
        inter_token_intervals_s: list[float] = []
        spans_by_choice: dict[int, list[dict[str, Any]]] = {}
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=row["effective_request"],
                headers={"X-Request-Id": request_id},
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
                        spans = choice.get("policy_token_spans") or []
                        spans_by_choice.setdefault(choice_index, []).extend(
                            dict(span) for span in spans
                        )
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
                            token_events.append(
                                {
                                    "request_id": request_id,
                                    "monotonic_s": now,
                                    "tokens": emitted,
                                }
                            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
        samples.append(
            {
                "request_id": request_id,
                "worker": worker,
                "phase_at_start": phase_at_start,
                "datapoint_id": row["id"],
                "canonical_request_sha256": row["canonical_request_sha256"],
                "effective_request_sha256": row["effective_request_sha256"],
                "started_monotonic_s": started,
                "ended_monotonic_s": ended,
                "latency_s": ended - started,
                "ttft_s": ttft_s,
                "inter_token_intervals_s": inter_token_intervals_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "policy_token_spans_by_choice": spans_by_choice,
                "error": error,
            }
        )


async def _record_control(
    *,
    client: httpx.AsyncClient,
    name: str,
    duration_s: float,
    phase: list[str],
) -> dict[str, Any]:
    phase[0] = name
    before = await _runtime_metrics(client)
    started = time.perf_counter()
    await asyncio.sleep(duration_s)
    ended = time.perf_counter()
    after = await _runtime_metrics(client)
    elapsed = ended - started
    prompt_tokens = after["prompt_tokens_total"] - before["prompt_tokens_total"]
    completion_tokens = (
        after["generation_tokens_total"] - before["generation_tokens_total"]
    )
    return {
        "name": name,
        "started_monotonic_s": started,
        "ended_monotonic_s": ended,
        "duration_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_per_s": prompt_tokens / elapsed,
        "completion_tokens_per_s": completion_tokens / elapsed,
    }


async def _fixed_input_fingerprint(
    client: httpx.AsyncClient, *, policy_version: int
) -> dict[str, Any]:
    request = {
        "model": "bonnie-replay:active",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 1,
        "n": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    fingerprints: list[str] = []
    for _ in range(2):
        response = await client.post("/v1/chat/completions", json=request)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        spans = choice.get("policy_token_spans") or []
        versions = {int(span["policy_version"]) for span in spans}
        if versions != {policy_version}:
            raise RuntimeError(
                f"Fixed-input request used policies {versions}; expected {policy_version}"
            )
        receipt = {
            "message": choice["message"],
            "logprobs": choice.get("logprobs"),
        }
        fingerprints.append(_canonical_sha256(receipt))
    if len(set(fingerprints)) != 1:
        raise RuntimeError("Fixed-input receiver logit parity failed")
    return {
        "method": "two deterministic receiver requests after commit",
        "sha256": fingerprints[0],
        "policy_version": policy_version,
        "policy_token_spans": spans,
    }


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


def _cpu_memory_snapshot() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", maxsplit=1)
        meminfo[key] = int(raw.strip().split()[0])
    return {
        "system_used_kib": meminfo["MemTotal"] - meminfo["MemAvailable"],
        "runner_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(20260723)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(2000)]
    means.sort()
    return [means[50], means[1949]]


def _attach_policy_ages(
    samples: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> None:
    commits = [
        (
            float(update["committed_monotonic_s"]),
            int(update["receiver"]["committed_state"]["policy_version"]),
        )
        for update in updates
    ]
    for sample in samples:
        current = max(
            (
                version
                for committed_at, version in commits
                if committed_at <= float(sample["ended_monotonic_s"])
            ),
            default=0,
        )
        versions = [
            int(span["policy_version"])
            for spans in sample["policy_token_spans_by_choice"].values()
            for span in spans
        ]
        sample["receiver_policy_version_at_end"] = current
        sample["policy_age_min"] = current - max(versions) if versions else None
        sample["policy_age_max"] = current - min(versions) if versions else None


async def run(args: argparse.Namespace) -> None:
    requests, trace_provenance = _effective_requests(
        args.request_jsonl, args.trace_manifest, bench_commit=args.bench_commit
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.request_jsonl, output / "request-trace.jsonl")
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
    samples: list[dict[str, Any]] = []
    token_events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    phase = ["idle"]
    stop = asyncio.Event()
    async with httpx.AsyncClient(
        base_url=service._vllm_base_url,
        headers=service._runtime_headers(),
        timeout=300.0,
    ) as client:
        runtime_before = await _runtime_metrics(client)
        if int(runtime_before.get("world_size", 0)) != 2:
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
                        token_events=token_events,
                        phase=phase,
                    )
                )
                for worker in range(8)
            ]
            if args.mode == "fixed-load"
            else []
        )
        hashes: set[str] = set()
        manifest_sha256: str | None = None
        try:
            if load_tasks:
                controls.append(
                    await _record_control(
                        client=client,
                        name="control-before",
                        duration_s=args.control_seconds,
                        phase=phase,
                    )
                )
            phase[0] = "updates"
            for index in range(args.warmups + args.updates):
                policy_version = index + 1
                checkpoint = output / "checkpoints" / f"{policy_version:04d}"
                memory_before = {
                    "gpu": _gpu_memory_snapshot(),
                    "cpu": _cpu_memory_snapshot(),
                }
                update_runtime_before = (
                    await _runtime_metrics(client)
                    if index >= args.warmups and load_tasks
                    else None
                )
                update_started = time.perf_counter()
                replay = await service.replay_lora_update(
                    source_lora_path=str(seed),
                    output_lora_path=str(checkpoint),
                    policy_version=policy_version,
                    content_version=index,
                )
                committed_at = time.perf_counter()
                update_runtime_after = (
                    await _runtime_metrics(client)
                    if update_runtime_before is not None
                    else None
                )
                metrics = replay["metrics"]
                receiver = replay["receiver"]
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
                if len(memory_before["gpu"]) != 4 or any(
                    "H200" not in str(row["name"]) for row in memory_before["gpu"]
                ):
                    raise RuntimeError(f"Replay requires four H200s: {memory_before!r}")
                snapshot = inspect_adapter(checkpoint)
                if snapshot.file_sha256 in hashes:
                    raise RuntimeError("Replay produced duplicate update contents")
                if manifest_sha256 is None:
                    manifest_sha256 = snapshot.manifest_sha256
                elif snapshot.manifest_sha256 != manifest_sha256:
                    raise RuntimeError("Replay changed the serving tensor manifest")
                hashes.add(snapshot.file_sha256)
                parity = await _fixed_input_fingerprint(
                    client, policy_version=policy_version
                )
                updates.append(
                    {
                        "index": index,
                        "measured": index >= args.warmups,
                        "requested_policy_version": policy_version,
                        "checkpoint": str(checkpoint),
                        "sha256": snapshot.file_sha256,
                        "manifest_sha256": snapshot.manifest_sha256,
                        "metrics": metrics,
                        "receiver": receiver,
                        "fixed_input_logit_parity": parity,
                        "update_started_monotonic_s": update_started,
                        "committed_monotonic_s": committed_at,
                        "update_wall_s": committed_at - update_started,
                        "runtime_counter_delta": (
                            {
                                "prompt_tokens": (
                                    update_runtime_after["prompt_tokens_total"]
                                    - update_runtime_before["prompt_tokens_total"]
                                ),
                                "completion_tokens": (
                                    update_runtime_after["generation_tokens_total"]
                                    - update_runtime_before["generation_tokens_total"]
                                ),
                            }
                            if update_runtime_before is not None
                            and update_runtime_after is not None
                            else None
                        ),
                        "memory_before": memory_before,
                        "memory_after": {
                            "gpu": _gpu_memory_snapshot(),
                            "cpu": _cpu_memory_snapshot(),
                        },
                    }
                )
                if len(updates) > 2:
                    shutil.rmtree(Path(updates[-3]["checkpoint"]))
            if load_tasks:
                controls.append(
                    await _record_control(
                        client=client,
                        name="control-after",
                        duration_s=args.control_seconds,
                        phase=phase,
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
                            client=client,
                            name="control-repeat",
                            duration_s=args.control_seconds,
                            phase=phase,
                        )
                    )
        finally:
            stop.set()
            await asyncio.gather(*load_tasks)
        runtime_after = await _runtime_metrics(client)
    await service.aclose()

    measured = [row for row in updates if row["measured"]]
    acceptance = evaluate_replay_acceptance([row["metrics"] for row in measured])
    totals = [
        float(row["metrics"]["time/weight_update_trainer_publish_s"])
        + float(row["metrics"]["time/weight_update_service_rpc_s"])
        for row in measured
    ]
    _attach_policy_ages(samples, updates)
    prompt_tokens = sum(
        float((row["runtime_counter_delta"] or {})["prompt_tokens"])
        for row in measured
        if row["runtime_counter_delta"] is not None
    )
    completion_tokens = sum(
        float((row["runtime_counter_delta"] or {})["completion_tokens"])
        for row in measured
        if row["runtime_counter_delta"] is not None
    )
    measured_duration = sum(float(row["update_wall_s"]) for row in measured)
    control_completion_rate = (
        statistics.fmean(row["completion_tokens_per_s"] for row in controls)
        if controls
        else None
    )
    ages = [
        int(sample["policy_age_max"])
        for sample in samples
        if sample["policy_age_max"] is not None
    ]
    gpu_snapshots = [
        gpu
        for update in updates
        for boundary in ("memory_before", "memory_after")
        for gpu in update[boundary]["gpu"]
    ]
    cpu_snapshots = [
        update[boundary]["cpu"]
        for update in updates
        for boundary in ("memory_before", "memory_after")
    ]
    summary: dict[str, Any] = {
        "acceptance": acceptance,
        "update_total_mean_s": statistics.fmean(totals),
        "update_total_p50_s": statistics.median(totals),
        "update_total_p95_s": sorted(totals)[round(0.95 * (len(totals) - 1))],
        "update_total_stddev_s": statistics.stdev(totals),
        "update_total_mean_95ci_s": _bootstrap_ci(totals),
        "measured_lane_duration_s": measured_duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_per_s": prompt_tokens / measured_duration,
        "completion_tokens_per_s": completion_tokens / measured_duration,
        "lost_tokens_per_update": (
            (control_completion_rate * measured_duration - completion_tokens)
            / len(measured)
            if control_completion_rate is not None
            else None
        ),
        "request_errors": sum(row["error"] is not None for row in samples),
        "policy_age_p50": statistics.median(ages) if ages else None,
        "policy_age_p95": (
            sorted(ages)[round(0.95 * (len(ages) - 1))] if ages else None
        ),
        "peak_gpu_used_mib": max(int(row["used_mib"]) for row in gpu_snapshots),
        "peak_system_used_kib": max(row["system_used_kib"] for row in cpu_snapshots),
        "peak_runner_rss_kib": max(row["runner_peak_rss_kib"] for row in cpu_snapshots),
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
    summary.update(
        {
            "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
            "ttft_p95_s": (
                sorted(ttfts)[round(0.95 * (len(ttfts) - 1))] if ttfts else None
            ),
            "inter_token_latency_p50_s": statistics.median(itls) if itls else None,
            "inter_token_latency_p95_s": (
                sorted(itls)[round(0.95 * (len(itls) - 1))] if itls else None
            ),
        }
    )
    (output / "updates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in updates)
    )
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples)
    )
    (output / "token-events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in token_events)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "runtime-before.json").write_text(json.dumps(runtime_before, indent=2))
    (output / "runtime-after.json").write_text(json.dumps(runtime_after, indent=2))
    reference_snapshot = inspect_adapter(Path(updates[-1]["checkpoint"]))
    (output / "tensor-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": reference_snapshot.manifest_sha256,
                "rank": reference_snapshot.rank,
                "base_model": reference_snapshot.base_model,
                "logical_bytes": reference_snapshot.logical_bytes,
                "transported_bytes": reference_snapshot.transported_bytes,
                "tensor_count": reference_snapshot.tensor_count,
                "tensors": [asdict(tensor) for tensor in reference_snapshot.tensors],
            },
            indent=2,
        )
    )
    required_result_files = [
        "manifest.json",
        "request-trace.jsonl",
        "tensor-manifest.json",
        "samples.jsonl",
        "summary.json",
        "stdout.log",
        "stderr.log",
    ]
    result_manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "art_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "bench_trace": trace_provenance,
        "effective_request_transform": {
            "model": "bonnie-replay:active",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        "topology": {
            "trainer_gpu_ids": [0, 1],
            "inference_gpu_ids": [2, 3],
            "inference_tp": 2,
            "trainer": {"tp": 1, "pp": 1, "cp": 2, "ep": 2, "etp": 1},
        },
        "required_result_files": required_result_files,
        "additional_result_files": [
            "updates.jsonl",
            "token-events.jsonl",
            "runtime-before.json",
            "runtime-after.json",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(result_manifest, indent=2))
    if summary["request_errors"]:
        raise RuntimeError(
            f"Evidence run observed {summary['request_errors']} request errors"
        )
    if not acceptance["passed"]:
        raise RuntimeError(f"Replay acceptance gates failed: {acceptance['failures']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-jsonl", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--bench-commit", required=True)
    parser.add_argument("--mode", choices=("idle", "fixed-load"), required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--control-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
