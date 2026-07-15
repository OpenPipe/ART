"""Exact LoRA state acknowledgement from vLLM model workers."""

from __future__ import annotations

from collections import defaultdict
import os
import socket
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from art_vllm_runtime.engine_core import query_engine_cores

ART_LORA_WORKER_EXTENSION = "art_vllm_runtime.lora_state.ArtLoraStateWorkerExtension"
ART_LORA_WORKER_RPC = "art_lora_worker_state"
ART_LORA_WORKER_RPC_TIMEOUT_S = 30.0
_PROCESS_UUID_ENV = "ART_VLLM_PROCESS_UUID"
_GENERATION_ENV = "ART_VLLM_REPLICA_GENERATION"
_NODE_RANK_ENV = "ART_VLLM_NODE_RANK"
_DATA_PARALLEL_SIZE_ENV = "ART_VLLM_DATA_PARALLEL_SIZE"


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _StrictMessage(_Message):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExpectedLoraWorker(_Message):
    rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    process_uuid: str = Field(min_length=1)
    generation: int = Field(ge=0)


class ExpectedLoraState(_Message):
    lora_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
    policy_version: int = Field(ge=0)


class LoraWorkerStateRequest(_Message):
    expected_workers: tuple[ExpectedLoraWorker, ...] = Field(min_length=1)
    expected_lora: ExpectedLoraState

    @model_validator(mode="after")
    def _validate_membership(self) -> "LoraWorkerStateRequest":
        workers = self.expected_workers
        if tuple(worker.rank for worker in workers) != tuple(range(len(workers))):
            raise ValueError("expected worker ranks must be ordered and contiguous")
        if len({worker.generation for worker in workers}) != 1:
            raise ValueError("expected workers must have one replica generation")

        by_process: dict[str, list[ExpectedLoraWorker]] = defaultdict(list)
        for worker in workers:
            by_process[worker.process_uuid].append(worker)
        node_ranks = []
        local_world_sizes = set()
        for process_workers in by_process.values():
            node_rank = {worker.node_rank for worker in process_workers}
            if len(node_rank) != 1:
                raise ValueError("one process UUID cannot span node ranks")
            node_ranks.extend(node_rank)
            local_ranks = tuple(worker.local_rank for worker in process_workers)
            if local_ranks != tuple(range(len(process_workers))):
                raise ValueError("expected local ranks must be ordered and contiguous")
            local_world_sizes.add(len(process_workers))
        if sorted(node_ranks) != list(range(len(by_process))):
            raise ValueError("expected node ranks must be contiguous")
        if len(local_world_sizes) != 1:
            raise ValueError("native vLLM members must have equal worker counts")
        return self


class WorkerLoraState(_StrictMessage):
    lora_id: int = Field(ge=1)
    lora_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
    policy_version: int = Field(ge=0)
    update_seq: int = Field(ge=1)


class LoraWorkerReport(_StrictMessage):
    rank: int = Field(ge=0)
    executor_rank: int = Field(ge=0)
    executor_world_size: int = Field(ge=1)
    data_parallel_rank: int = Field(ge=0)
    data_parallel_size: int = Field(ge=1)
    local_rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    process_uuid: str = Field(min_length=1)
    generation: int = Field(ge=0)
    worker_pid: int = Field(ge=1)
    hostname: str = Field(min_length=1)
    engine_instance_id: str | None
    world_size: int = Field(ge=1)
    local_world_size: int = Field(ge=1)
    loaded_loras: list[WorkerLoraState]

    @model_validator(mode="after")
    def _validate_loaded_loras(self) -> "LoraWorkerReport":
        if self.executor_rank >= self.executor_world_size:
            raise ValueError("executor rank must be smaller than its world size")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data-parallel rank must be smaller than its size")
        if self.world_size != self.executor_world_size * self.data_parallel_size:
            raise ValueError("worker world size does not include every DP executor")
        if self.rank != (
            self.data_parallel_rank * self.executor_world_size + self.executor_rank
        ):
            raise ValueError("worker global rank does not match its executor rank")
        ids = tuple(lora.lora_id for lora in self.loaded_loras)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("loaded LoRA IDs must be unique and ordered")
        return self


class LoraWorkerStateResponse(_Message):
    status: Literal["acknowledged"] = "acknowledged"
    expected_worker_count: int = Field(ge=1)
    expected_worker_ranks: tuple[int, ...]
    lora: WorkerLoraState
    workers: tuple[LoraWorkerReport, ...]


def configure_lora_worker_identity(
    *, process_uuid: str, generation: int, node_rank: int, data_parallel_size: int
) -> None:
    os.environ[_PROCESS_UUID_ENV] = process_uuid
    os.environ[_GENERATION_ENV] = str(generation)
    os.environ[_NODE_RANK_ENV] = str(node_rank)
    os.environ[_DATA_PARALLEL_SIZE_ENV] = str(data_parallel_size)


class ArtLoraStateWorkerExtension:
    def art_lora_worker_state(self: Any) -> dict[str, Any]:
        from art_vllm_runtime.policy_spans import get_worker_lora_states

        parallel = self.parallel_config
        node_rank = _required_int_env(_NODE_RANK_ENV)
        if node_rank != int(parallel.node_rank):
            raise RuntimeError("ART and vLLM worker node ranks disagree")
        executor_world_size = int(parallel.world_size)
        data_parallel_rank = int(parallel.data_parallel_index)
        data_parallel_size = _required_int_env(_DATA_PARALLEL_SIZE_ENV)
        if data_parallel_size < 1 or data_parallel_rank >= data_parallel_size:
            raise RuntimeError("invalid vLLM worker data-parallel identity")
        world_size = executor_world_size * data_parallel_size
        nnodes = int(parallel.nnodes)
        if world_size % nnodes:
            raise RuntimeError("vLLM worker world size is not divisible by nnodes")
        return {
            "rank": data_parallel_rank * executor_world_size + int(self.rank),
            "executor_rank": int(self.rank),
            "executor_world_size": executor_world_size,
            "data_parallel_rank": data_parallel_rank,
            "data_parallel_size": data_parallel_size,
            "local_rank": int(self.local_rank),
            "node_rank": node_rank,
            "process_uuid": _required_env(_PROCESS_UUID_ENV),
            "generation": _required_int_env(_GENERATION_ENV),
            "worker_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "engine_instance_id": str(self.vllm_config.instance_id) or None,
            "world_size": world_size,
            "local_world_size": world_size // nnodes,
            "loaded_loras": get_worker_lora_states(self.list_loras()),
        }


async def query_lora_worker_state(
    engine_client: Any,
    request: LoraWorkerStateRequest,
) -> LoraWorkerStateResponse:
    replies, data_parallel_size = await _collect_worker_replies(engine_client)
    try:
        reports = tuple(
            LoraWorkerReport.model_validate(reply, strict=True) for reply in replies
        )
    except ValidationError as exc:
        raise RuntimeError("vLLM worker returned malformed LoRA state") from exc

    expected = request.expected_workers
    expected_ranks = tuple(worker.rank for worker in expected)
    if len(reports) != len(expected):
        raise RuntimeError(
            f"vLLM returned {len(reports)} of {len(expected)} worker reports"
        )
    if tuple(report.rank for report in reports) != expected_ranks:
        raise RuntimeError("vLLM worker report ranks are incomplete or out of order")
    if len({(report.process_uuid, report.worker_pid) for report in reports}) != len(
        reports
    ):
        raise RuntimeError("vLLM worker reports contain duplicate process IDs")

    process_counts: dict[str, int] = defaultdict(int)
    for worker in expected:
        process_counts[worker.process_uuid] += 1
    for expected_worker, report in zip(expected, reports, strict=True):
        identity = ExpectedLoraWorker(
            rank=report.rank,
            local_rank=report.local_rank,
            node_rank=report.node_rank,
            process_uuid=report.process_uuid,
            generation=report.generation,
        )
        if identity != expected_worker:
            raise RuntimeError(f"vLLM worker rank {report.rank} identity mismatch")
        if report.world_size != len(expected):
            raise RuntimeError(f"vLLM worker rank {report.rank} world-size mismatch")
        if report.data_parallel_size != data_parallel_size:
            raise RuntimeError(
                f"vLLM worker rank {report.rank} data-parallel-size mismatch"
            )
        if report.local_world_size != process_counts[report.process_uuid]:
            raise RuntimeError(
                f"vLLM worker rank {report.rank} local-world-size mismatch"
            )

    loaded_loras = reports[0].loaded_loras
    if any(report.loaded_loras != loaded_loras for report in reports[1:]):
        raise RuntimeError("vLLM workers report divergent loaded LoRA state")
    requested = request.expected_lora
    matches = tuple(
        lora
        for lora in loaded_loras
        if (
            lora.lora_name,
            lora.lora_path,
            lora.policy_version,
        )
        == (requested.lora_name, requested.lora_path, requested.policy_version)
    )
    if len(matches) != 1:
        raise RuntimeError("requested LoRA state was not loaded on every vLLM worker")
    return LoraWorkerStateResponse(
        expected_worker_count=len(expected),
        expected_worker_ranks=expected_ranks,
        lora=matches[0],
        workers=reports,
    )


async def _collect_worker_replies(engine_client: Any) -> tuple[list[Any], int]:
    parallel = engine_client.vllm_config.parallel_config
    data_parallel_size = int(parallel.data_parallel_size)
    if data_parallel_size == 1:
        replies = await engine_client.collective_rpc(
            ART_LORA_WORKER_RPC, timeout=ART_LORA_WORKER_RPC_TIMEOUT_S
        )
        if type(replies) is not list:
            raise RuntimeError("vLLM collective_rpc returned a non-list reply")
        return replies, data_parallel_size

    groups = await query_engine_cores(
        engine_client,
        "collective_rpc",
        ART_LORA_WORKER_RPC,
        ART_LORA_WORKER_RPC_TIMEOUT_S,
        (),
        None,
    )
    if any(type(group) is not list for group in groups):
        raise RuntimeError("vLLM DP collective_rpc returned a non-list reply")
    return [reply for group in groups for reply in group], data_parallel_size


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required worker identity {name}")
    return value


def _required_int_env(name: str) -> int:
    value = _required_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid worker identity {name}={value!r}") from exc
