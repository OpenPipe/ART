"""Replay in-flight LoRA updates against an ART vLLM runtime.

The replay input is a sequence of real trainer-produced checkpoints. Keeping the
captured safetensors files intact preserves the serving tensor names, shapes,
dtypes, and byte volume without rerunning Megatron, reward, or judging.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import httpx
from safetensors import safe_open

from art.megatron.model_support.lora_disk import load_adapter_config


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int


@dataclass(frozen=True)
class AdapterSnapshot:
    path: str
    file_sha256: str
    manifest_sha256: str
    logical_bytes: int
    transported_bytes: int
    tensor_count: int
    rank: int
    base_model: str
    tensors: tuple[TensorSpec, ...]


@dataclass
class LoadCounters:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    errors: int = 0

    def __sub__(self, other: LoadCounters) -> LoadCounters:
        return LoadCounters(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            requests=self.requests - other.requests,
            errors=self.errors - other.errors,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_adapter(path: str | Path) -> AdapterSnapshot:
    adapter_dir = Path(path).resolve()
    tensor_path = adapter_dir / "adapter_model.safetensors"
    config = load_adapter_config(adapter_dir)
    rank = int(config.get("r", config.get("rank", 0)))
    base_model = str(config.get("base_model_name_or_path", ""))
    tensors: list[TensorSpec] = []
    logical_bytes = 0
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name)
            tensor_bytes = tensor.numel() * tensor.element_size()
            logical_bytes += tensor_bytes
            tensors.append(
                TensorSpec(
                    name=name,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype=str(tensor.dtype),
                    bytes=tensor_bytes,
                )
            )
    manifest_payload = json.dumps(
        [asdict(tensor) for tensor in tensors],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return AdapterSnapshot(
        path=str(adapter_dir),
        file_sha256=_file_sha256(tensor_path),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        logical_bytes=logical_bytes,
        transported_bytes=tensor_path.stat().st_size,
        tensor_count=len(tensors),
        rank=rank,
        base_model=base_model,
        tensors=tuple(tensors),
    )


def validate_replay_snapshots(
    snapshots: list[AdapterSnapshot],
    *,
    expected_rank: int,
    expected_experts: int,
    expected_model_substring: str,
) -> None:
    if not snapshots:
        raise ValueError("At least one adapter snapshot is required")
    reference = snapshots[0]
    for snapshot in snapshots:
        if snapshot.rank != expected_rank:
            raise ValueError(
                f"{snapshot.path} has rank {snapshot.rank}; expected {expected_rank}"
            )
        if expected_model_substring not in snapshot.base_model:
            raise ValueError(
                f"{snapshot.path} targets {snapshot.base_model!r}; expected a model "
                f"containing {expected_model_substring!r}"
            )
        if snapshot.manifest_sha256 != reference.manifest_sha256:
            raise ValueError(
                f"{snapshot.path} does not have the reference serving tensor layout"
            )
    expert_tensors = [
        tensor
        for tensor in reference.tensors
        if ".mlp.experts." in tensor.name or ".mlp.experts.base_layer." in tensor.name
    ]
    if not expert_tensors:
        raise ValueError("Adapter has no packed Qwen MoE expert tensors")
    mismatched = [
        tensor.name for tensor in expert_tensors if expected_experts not in tensor.shape
    ]
    if mismatched:
        raise ValueError(
            "Packed expert tensors do not preserve the expected expert dimension: "
            + ", ".join(mismatched[:3])
        )


def validate_committed_policy_version(
    response_body: dict[str, Any], *, expected: int
) -> None:
    committed = int(response_body.get("policy_version", -1))
    if committed != expected:
        raise RuntimeError(f"Runtime committed policy {committed}; expected {expected}")


class FixedLoad:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        request: dict[str, Any],
        concurrency: int,
    ) -> None:
        self._client = client
        self._request = request
        self._concurrency = concurrency
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self.counters = LoadCounters()

    async def _worker(self) -> None:
        while self._running:
            try:
                response = await self._client.post(
                    "/v1/chat/completions", json=self._request
                )
                response.raise_for_status()
                usage = response.json().get("usage", {})
                self.counters.prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.counters.completion_tokens += int(
                    usage.get("completion_tokens", 0)
                )
                self.counters.requests += 1
            except BaseException:
                self.counters.errors += 1
                if not self._running:
                    return

    def snapshot(self) -> LoadCounters:
        return LoadCounters(**asdict(self.counters))

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._worker()) for _ in range(self._concurrency)
        ]

    async def stop(self) -> None:
        self._running = False
        await asyncio.gather(*self._tasks, return_exceptions=True)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile), len(ordered) - 1)
    return ordered[index]


async def replay_updates(args: argparse.Namespace) -> dict[str, Any]:
    snapshots = [inspect_adapter(path) for path in args.adapter_path]
    validate_replay_snapshots(
        snapshots,
        expected_rank=args.expected_rank,
        expected_experts=args.expected_experts,
        expected_model_substring=args.expected_model_substring,
    )
    server_paths = args.server_adapter_path or [item.path for item in snapshots]
    if len(server_paths) != len(snapshots):
        raise ValueError("--server-adapter-path must match --adapter-path count")

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(
        base_url=args.server_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:
        metrics_response = await client.get("/art/metrics")
        metrics_response.raise_for_status()
        runtime_metrics = metrics_response.json()
        runtime_world_size = int(
            runtime_metrics.get("metrics", {}).get("world_size", 0)
        )
        if runtime_world_size != args.expected_inference_world_size:
            raise ValueError(
                f"Runtime world size is {runtime_world_size}; expected "
                f"{args.expected_inference_world_size}"
            )

        request: dict[str, Any] | None = None
        if args.mode == "fixed-load":
            if args.request_json is None:
                raise ValueError("--request-json is required in fixed-load mode")
            request = json.loads(Path(args.request_json).read_text())
        fixed_load = (
            FixedLoad(client, request=request, concurrency=args.load_concurrency)
            if request is not None
            else None
        )
        if fixed_load is not None:
            await fixed_load.start()

        update_records: list[dict[str, Any]] = []
        measured_update_s = 0.0
        try:
            total_updates = args.warmups + args.updates
            for update_index in range(total_updates):
                snapshot_index = update_index % len(snapshots)
                policy_version = args.first_policy_version + update_index
                before_load = fixed_load.snapshot() if fixed_load else LoadCounters()
                started = time.perf_counter()
                response = await client.post(
                    "/art/in_flight_lora_update",
                    json={
                        "model_name": args.model_name,
                        "base_model_name": args.base_model_name,
                        "lora_slot": args.lora_slot,
                        "lora_path": server_paths[snapshot_index],
                        "policy_version": policy_version,
                    },
                )
                response.raise_for_status()
                wall_s = time.perf_counter() - started
                body = response.json()
                validate_committed_policy_version(body, expected=policy_version)
                after_load = fixed_load.snapshot() if fixed_load else LoadCounters()
                if update_index >= args.warmups:
                    measured_update_s += wall_s
                    update_records.append(
                        {
                            "update_index": update_index - args.warmups,
                            "policy_version": policy_version,
                            "snapshot_index": snapshot_index,
                            "snapshot_sha256": snapshots[snapshot_index].file_sha256,
                            "wall_s": wall_s,
                            "runtime_timing_s": body.get("timing_s", {}),
                            "load_during_update": asdict(after_load - before_load),
                        }
                    )

            control_load = LoadCounters()
            control_s = 0.0
            if fixed_load is not None and measured_update_s > 0:
                before_control = fixed_load.snapshot()
                control_started = time.perf_counter()
                await asyncio.sleep(measured_update_s)
                control_s = time.perf_counter() - control_started
                control_load = fixed_load.snapshot() - before_control
        finally:
            if fixed_load is not None:
                await fixed_load.stop()

    post_snapshots = [inspect_adapter(path) for path in args.adapter_path]
    if [item.file_sha256 for item in snapshots] != [
        item.file_sha256 for item in post_snapshots
    ]:
        raise RuntimeError("Replay mutated a captured adapter snapshot")

    update_wall = [float(record["wall_s"]) for record in update_records]
    update_completion_tokens = sum(
        int(record["load_during_update"]["completion_tokens"])
        for record in update_records
    )
    control_completion_rate = (
        control_load.completion_tokens / control_s if control_s > 0 else 0.0
    )
    lost_completion_tokens = max(
        control_completion_rate * measured_update_s - update_completion_tokens,
        0.0,
    )
    return {
        "schema_version": 1,
        "mode": args.mode,
        "warmups": args.warmups,
        "measured_updates": args.updates,
        "runtime_world_size": runtime_world_size,
        "snapshots": [
            {key: value for key, value in asdict(snapshot).items() if key != "tensors"}
            for snapshot in snapshots
        ],
        "updates": update_records,
        "control": {
            "wall_s": control_s,
            "load": asdict(control_load),
            "completion_tok_per_s": control_completion_rate,
        },
        "summary": {
            "update_wall_mean_s": statistics.fmean(update_wall),
            "update_wall_p50_s": _percentile(update_wall, 0.50),
            "update_wall_p95_s": _percentile(update_wall, 0.95),
            "measured_update_wall_s": measured_update_s,
            "completion_tokens_during_updates": update_completion_tokens,
            "lost_completion_tokens": lost_completion_tokens,
            "lost_completion_tokens_per_update": lost_completion_tokens
            / max(args.updates, 1),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay captured Qwen3.6 LoRA updates against ART vLLM"
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--adapter-path", action="append", required=True)
    parser.add_argument("--server-adapter-path", action="append")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--lora-slot", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--mode", choices=("idle", "fixed-load"), default="idle")
    parser.add_argument("--request-json")
    parser.add_argument("--load-concurrency", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--first-policy-version", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--expected-rank", type=int, default=8)
    parser.add_argument("--expected-experts", type=int, default=256)
    parser.add_argument("--expected-model-substring", default="Qwen3.6-35B-A3B")
    parser.add_argument("--expected-inference-world-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.warmups < 0 or args.updates < 1:
        raise SystemExit(
            "--warmups must be non-negative and --updates must be positive"
        )
    result = asyncio.run(replay_updates(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
