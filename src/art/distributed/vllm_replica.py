from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
from typing import Literal, Protocol, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..utils.lifecycle import ChildProcessSupervisor
from ..vllm_runtime import ManagedVllmRuntime, VllmRuntimeLaunchConfig
from .specs import (
    VLLM_KV_EVENT_BUFFER_STEPS,
    VLLM_KV_EVENT_HWM,
    VLLM_PREFIX_HASH_BLOCK_SIZE,
    VLLM_PREFIX_HASH_SEED,
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceReplicaSpec,
    vllm_kv_event_topic,
)

_VLLM_HASH_ENV = {
    "PYTHONHASHSEED": VLLM_PREFIX_HASH_SEED,
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0",
}


def _configure_vllm_hash_environment() -> None:
    for name, expected in _VLLM_HASH_ENV.items():
        configured = os.environ.get(name)
        if configured not in (None, expected):
            raise ValueError(f"{name}={configured!r} conflicts with {expected!r}")
        os.environ[name] = expected


def _prefix_hash_block_size(engine_args: Mapping[str, object]) -> int:
    value = engine_args.get("hash_block_size")
    if value is None:
        value = engine_args.get("block_size")
    if value is None:
        return VLLM_PREFIX_HASH_BLOCK_SIZE
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("vLLM hash_block_size must be a positive integer")
    return value


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


MemberPhase = Literal["starting", "ready", "stopped", "failed"]
ReplicaPhase = Literal[
    "stopped", "starting", "ready", "updating", "quarantined", "closing"
]


class ReplicaLaunchTemplate(_Message):
    served_model_name: str = Field(min_length=1)
    lora_path: str | None = None
    engine_args: dict[str, object] = Field(default_factory=dict)
    server_args: dict[str, object] = Field(default_factory=dict)


class HostMemberLaunchRequest(_Message):
    replica_id: str
    member: ModelServiceMemberSpec
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    process_uuid: str = Field(min_length=1)
    launch_config: VllmRuntimeLaunchConfig


class HostMemberState(_Message):
    replica_id: str
    member_id: str
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    process_uuid: str = Field(min_length=1)
    phase: MemberPhase
    detail: str | None = None


class ReplicaUpdateReport(_Message):
    replica_id: str
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    policy_digest: str = Field(min_length=1)
    update_identity: str = Field(min_length=1)
    ambiguous: bool = False


class ReplicaState(_Message):
    replica_id: str
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    phase: ReplicaPhase
    members: tuple[HostMemberState, ...] = ()
    committed_version: str | None = None
    policy_digest: str | None = None
    update_identity: str | None = None
    quarantine_reason: str | None = None


class ReplicaHostLauncher(Protocol):
    async def start_member(
        self, request: HostMemberLaunchRequest
    ) -> HostMemberState: ...

    async def member_state(
        self, replica_id: str, member_id: str, generation: int
    ) -> HostMemberState: ...

    async def stop_member(
        self, replica_id: str, member_id: str, generation: int
    ) -> None: ...


class _ManagedMember:
    def __init__(self, request: HostMemberLaunchRequest) -> None:
        self.request = request
        self.runtime = ManagedVllmRuntime(host=request.launch_config.host)
        self.failure: RuntimeError | None = None
        self.supervisor = ChildProcessSupervisor(self._failed)

    def _failed(self, error: RuntimeError) -> None:
        self.failure = error


class ManagedVllmHostLauncher:
    """Host-local implementation of the serializable member launch protocol."""

    def __init__(
        self,
        output_root: str,
        *,
        install_parent_cleanup: Callable[[], None] = lambda: None,
        startup_timeout_s: float | None = None,
    ) -> None:
        self._output_root = Path(output_root)
        self._install_parent_cleanup = install_parent_cleanup
        self._startup_timeout_s = startup_timeout_s
        self._members: dict[tuple[str, str, int], _ManagedMember] = {}

    async def start_member(self, request: HostMemberLaunchRequest) -> HostMemberState:
        key = (request.replica_id, request.member.member_id, request.generation)
        if key in self._members:
            raise RuntimeError(f"vLLM member already exists: {key}")
        _configure_vllm_hash_environment()
        managed = _ManagedMember(request)
        self._members[key] = managed
        output_dir = self._output_root / request.replica_id / str(request.generation)
        output_dir /= request.member.member_id
        try:
            await managed.runtime.start(
                launch_config=request.launch_config,
                output_dir=str(output_dir),
                child_processes=managed.supervisor,
                install_parent_cleanup=self._install_parent_cleanup,
                timeout=self._startup_timeout_s,
            )
        except BaseException:
            await self.stop_member(*key)
            raise
        return self._state(managed, "ready")

    async def member_state(
        self, replica_id: str, member_id: str, generation: int
    ) -> HostMemberState:
        managed = self._members.get((replica_id, member_id, generation))
        if managed is None:
            raise RuntimeError(
                f"unknown vLLM member {replica_id}/{member_id}/{generation}"
            )
        process = managed.runtime.process
        failed = managed.failure
        if failed is None and process is not None and process.poll() is not None:
            failed = RuntimeError(f"process exited with code {process.returncode}")
        return self._state(
            managed,
            "failed" if failed is not None else "ready",
            detail=str(failed) if failed is not None else None,
        )

    async def stop_member(
        self, replica_id: str, member_id: str, generation: int
    ) -> None:
        managed = self._members.pop((replica_id, member_id, generation), None)
        if managed is None:
            return
        managed.supervisor.close()
        await asyncio.to_thread(managed.runtime.close)

    async def close(self) -> None:
        keys = tuple(self._members)
        results = await asyncio.gather(
            *(self.stop_member(*key) for key in keys), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("failed to stop vLLM host members", failures)

    @staticmethod
    def _state(
        managed: _ManagedMember, phase: MemberPhase, detail: str | None = None
    ) -> HostMemberState:
        request = managed.request
        return HostMemberState(
            replica_id=request.replica_id,
            member_id=request.member.member_id,
            generation=request.generation,
            generation_digest=request.generation_digest,
            process_uuid=request.process_uuid,
            phase=phase,
            detail=detail,
        )


class ReplicaManager:
    """Owns one native vLLM serving group as an indivisible failure domain."""

    def __init__(
        self,
        spec: ModelServiceReplicaSpec,
        launchers: Mapping[str, ReplicaHostLauncher],
        template: ReplicaLaunchTemplate,
        *,
        port_allocator: Callable[[str], int | Awaitable[int]] | None = None,
        monitor_interval_s: float = 0.25,
    ) -> None:
        missing = {member.host_id for member in spec.members} - launchers.keys()
        if missing:
            raise ValueError(f"replica launchers missing hosts: {sorted(missing)}")
        executor = template.engine_args.get("distributed_executor_backend")
        if executor not in (None, "mp", "multiprocessing"):
            raise ValueError("ART-managed replicas require vLLM multiprocessing")
        for key in ("revision", "tokenizer_revision"):
            configured = template.engine_args.get(key)
            if configured not in (None, spec.model_revision):
                raise ValueError(f"{key} conflicts with the replica model revision")
        if "kv_events_config" in template.engine_args:
            raise ValueError("kv_events_config is owned by the ART replica runtime")
        hash_block_size = _prefix_hash_block_size(template.engine_args)
        managed_args = {
            "enable_prefix_caching": True,
            "prefix_caching_hash_algo": "sha256",
            "hash_block_size": hash_block_size,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
        }
        for key, expected in managed_args.items():
            configured = template.engine_args.get(key)
            if configured not in (None, expected):
                raise ValueError(f"{key} conflicts with ART prefix-cache routing")
        _configure_vllm_hash_environment()
        self._spec = spec
        self._launchers = launchers
        self._template = template
        self._hash_block_size = hash_block_size
        self._port_allocator = port_allocator or self._allocate_port
        self._monitor_interval_s = monitor_interval_s
        self._lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._used_ports = {
            (spec.leader_endpoint.host, spec.leader_endpoint.port),
            (spec.rendezvous.host, spec.rendezvous.port),
        }
        digest = self._generation_digest(spec)
        self._state = ReplicaState(
            replica_id=spec.replica_id,
            generation=spec.generation,
            generation_digest=digest,
            phase="stopped",
        )

    @property
    def spec(self) -> ModelServiceReplicaSpec:
        return self._spec

    @property
    def state(self) -> ReplicaState:
        return self._state

    @property
    def prefix_hash_block_size(self) -> int:
        return self._hash_block_size

    def expected_worker_identities(self) -> tuple[dict[str, int | str], ...]:
        if self._state.phase not in {"ready", "updating"}:
            raise RuntimeError(f"replica workers are not live: {self._state.phase}")
        states = {member.member_id: member for member in self._state.members}
        if states.keys() != {member.member_id for member in self._spec.members}:
            raise RuntimeError("replica worker membership is incomplete")
        local_world_size = len(self._spec.members[0].gpu_ids)
        return tuple(
            {
                "rank": member.node_rank * local_world_size + local_rank,
                "local_rank": local_rank,
                "node_rank": member.node_rank,
                "process_uuid": states[member.member_id].process_uuid,
                "generation": self._state.generation,
            }
            for member in sorted(self._spec.members, key=lambda value: value.node_rank)
            for local_rank in range(local_world_size)
        )

    async def start(self) -> ReplicaState:
        async with self._lock:
            if self._state.phase != "stopped":
                raise RuntimeError(f"cannot start replica in {self._state.phase} state")
            self._state = self._state.model_copy(update={"phase": "starting"})
            requests = tuple(
                self._launch_request(member) for member in self._spec.members
            )
            tasks = [
                asyncio.create_task(
                    self._launchers[request.member.host_id].start_member(request)
                )
                for request in requests
            ]
            try:
                members = await asyncio.gather(*tasks)
                if any(member.phase != "ready" for member in members):
                    raise RuntimeError(f"vLLM gang was not ready: {members!r}")
            except BaseException as error:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await self._stop_members(requests)
                except BaseException as cleanup_error:
                    error = BaseExceptionGroup(
                        "vLLM gang startup and teardown failed",
                        [error, cleanup_error],
                    )
                self._state = self._state.model_copy(
                    update={
                        "phase": "quarantined",
                        "quarantine_reason": f"gang startup failed: {error}",
                    }
                )
                raise error from None
            self._state = self._state.model_copy(
                update={"phase": "ready", "members": tuple(members)}
            )
            self._monitor_task = asyncio.create_task(self._monitor())
            return self._state

    async def stop(self) -> ReplicaState:
        async with self._lock:
            await self._cancel_monitor()
            self._state = self._state.model_copy(update={"phase": "closing"})
            try:
                await self._stop_current_members()
            except BaseException as error:
                self._state = self._state.model_copy(
                    update={
                        "phase": "quarantined",
                        "quarantine_reason": f"replica teardown failed: {error}",
                    }
                )
                raise
            self._state = self._state.model_copy(
                update={"phase": "stopped", "members": ()}
            )
            return self._state

    async def restart(self) -> ReplicaState:
        await self.stop()
        generation = self._spec.generation + 1
        kv_event_port = self._spec.kv_event_base_port
        kv_replay_port = self._spec.kv_replay_base_port
        self._spec = self._spec.model_copy(
            update={
                "generation": generation,
                "kv_event_port": kv_event_port,
                "kv_replay_port": kv_replay_port,
                "leader_endpoint": EndpointSpec(
                    host=self._spec.leader_endpoint.host,
                    port=await self._fresh_port(self._spec.leader_endpoint.host),
                ),
                "rendezvous": EndpointSpec(
                    host=self._spec.rendezvous.host,
                    port=await self._fresh_port(self._spec.rendezvous.host),
                ),
            }
        )
        self._state = ReplicaState(
            replica_id=self._spec.replica_id,
            generation=generation,
            generation_digest=self._generation_digest(self._spec),
            phase="stopped",
        )
        return await self.start()

    def prepare_update(self, *, update_identity: str) -> ReplicaState:
        if self._state.phase != "ready":
            raise RuntimeError(f"cannot update replica in {self._state.phase} state")
        self._state = self._state.model_copy(
            update={"phase": "updating", "update_identity": update_identity}
        )
        return self._state

    def verify_update(self, report: ReplicaUpdateReport) -> ReplicaState:
        expected = self._state
        valid = (
            expected.phase == "updating"
            and report.replica_id == expected.replica_id
            and report.generation == expected.generation
            and report.generation_digest == expected.generation_digest
            and report.update_identity == expected.update_identity
            and not report.ambiguous
        )
        if not valid:
            return self.quarantine(f"ambiguous update report: {report.model_dump()}")
        self._state = expected.model_copy(
            update={
                "phase": "ready",
                "committed_version": report.policy_version,
                "policy_digest": report.policy_digest,
                "quarantine_reason": None,
            }
        )
        return self._state

    def quarantine(self, reason: str) -> ReplicaState:
        self._state = self._state.model_copy(
            update={"phase": "quarantined", "quarantine_reason": reason}
        )
        return self._state

    async def poll(self) -> ReplicaState:
        if self._state.phase not in {"ready", "updating"}:
            return self._state
        states = await asyncio.gather(
            *(
                self._launchers[member.host_id].member_state(
                    self._spec.replica_id, member.member_id, self._spec.generation
                )
                for member in self._spec.members
            ),
            return_exceptions=True,
        )
        failure = next(
            (
                state
                for state in states
                if isinstance(state, BaseException) or state.phase != "ready"
            ),
            None,
        )
        if failure is not None:
            self.quarantine(f"member failure: {failure}")
            try:
                await self._stop_current_members()
            except BaseException as error:
                self.quarantine(f"member failure: {failure}; teardown failure: {error}")
            return self._state
        self._state = self._state.model_copy(update={"members": tuple(states)})  # type: ignore[arg-type]
        return self._state

    async def _monitor(self) -> None:
        try:
            while self._state.phase in {"ready", "updating"}:
                await asyncio.sleep(self._monitor_interval_s)
                await self.poll()
        except asyncio.CancelledError:
            pass

    async def _cancel_monitor(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _stop_current_members(self) -> None:
        await self._stop_calls(
            tuple(
                (
                    self._launchers[member.host_id],
                    self._spec.replica_id,
                    member.member_id,
                    self._spec.generation,
                )
                for member in self._spec.members
            )
        )

    async def _stop_members(
        self, requests: tuple[HostMemberLaunchRequest, ...]
    ) -> None:
        await self._stop_calls(
            tuple(
                (
                    self._launchers[request.member.host_id],
                    request.replica_id,
                    request.member.member_id,
                    request.generation,
                )
                for request in requests
            )
        )

    @staticmethod
    async def _stop_calls(
        calls: tuple[tuple[ReplicaHostLauncher, str, str, int], ...],
    ) -> None:
        pending = calls
        failures: list[BaseException] = []
        for _attempt in range(2):
            results = await asyncio.gather(
                *(
                    launcher.stop_member(replica_id, member_id, generation)
                    for launcher, replica_id, member_id, generation in pending
                ),
                return_exceptions=True,
            )
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if not failures:
                return
            pending = tuple(
                call
                for call, result in zip(pending, results, strict=True)
                if isinstance(result, BaseException)
            )
        raise BaseExceptionGroup("failed to stop vLLM replica members", failures)

    def _launch_request(
        self, member: ModelServiceMemberSpec
    ) -> HostMemberLaunchRequest:
        parallel = self._spec.parallel
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{self._spec.kv_event_base_port}",
            "replay_endpoint": f"tcp://*:{self._spec.kv_replay_base_port}",
            "buffer_steps": VLLM_KV_EVENT_BUFFER_STEPS,
            "hwm": VLLM_KV_EVENT_HWM,
            "max_queue_size": VLLM_KV_EVENT_HWM,
            "topic": vllm_kv_event_topic(
                self._spec.replica_id,
                self._state.generation,
                self._state.generation_digest,
            ),
        }
        engine_args = {
            **self._template.engine_args,
            "revision": self._spec.model_revision,
            "tokenizer_revision": self._spec.model_revision,
            "enable_prefix_caching": True,
            "prefix_caching_hash_algo": "sha256",
            "hash_block_size": self._hash_block_size,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "kv_events_config": kv_events_config,
            "tensor_parallel_size": parallel.tp,
            "pipeline_parallel_size": parallel.pp,
            "data_parallel_size": parallel.dp,
            "enable_expert_parallel": parallel.enable_expert_parallel,
        }
        process_uuid = uuid.uuid4().hex
        launch = VllmRuntimeLaunchConfig(
            base_model=self._spec.base_model,
            port=self._spec.leader_endpoint.port,
            host=self._spec.leader_endpoint.host if member.leader else "127.0.0.1",
            local_gpu_ids=member.gpu_ids,
            lora_path=self._template.lora_path,
            served_model_name=self._template.served_model_name,
            rollout_weights_mode=self._spec.update_mode,
            engine_args=engine_args,
            server_args=self._template.server_args,
            nnodes=len(self._spec.members),
            node_rank=member.node_rank,
            master_addr=self._spec.rendezvous.host
            if len(self._spec.members) > 1
            else None,
            master_port=self._spec.rendezvous.port
            if len(self._spec.members) > 1
            else None,
            headless=not member.leader,
            replica_generation=self._spec.generation,
            process_uuid=process_uuid,
            update_identity=self._state.update_identity,
        )
        return HostMemberLaunchRequest(
            replica_id=self._spec.replica_id,
            member=member,
            generation=self._spec.generation,
            generation_digest=self._state.generation_digest,
            process_uuid=process_uuid,
            launch_config=launch,
        )

    async def _fresh_port(self, host: str) -> int:
        for _ in range(100):
            value = self._port_allocator(host)
            port = (
                await cast(Awaitable[int], value)
                if inspect.isawaitable(value)
                else value
            )
            if (host, port) not in self._used_ports:
                self._used_ports.add((host, port))
                return port
        raise RuntimeError(f"failed to allocate a fresh port on {host}")

    @staticmethod
    def _allocate_port(_host: str) -> int:
        with socket.socket() as listener:
            listener.bind(("", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _generation_digest(spec: ModelServiceReplicaSpec) -> str:
        payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
