from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..utils.lifecycle import ChildProcessSupervisor
from ..vllm_runtime import ManagedVllmRuntime, VllmRuntimeLaunchConfig
from .adapter_transport import AdapterReceiveResult, AdapterTransferTarget
from .specs import ModelServiceMemberSpec, ModelServiceSpec


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


MemberPhase = Literal["starting", "ready", "stopped", "failed"]
ReplicaPhase = Literal[
    "stopped", "starting", "ready", "updating", "quarantined", "closing"
]


class ReplicaLaunchTemplate(_Message):
    served_model_name: str = Field(min_length=1)
    lora_path: str | None = None
    initial_policy_version: int | None = Field(default=None, ge=0)
    initial_generation_id: str | None = Field(default=None, min_length=1)
    engine_args: dict[str, object] = Field(default_factory=dict)
    server_args: dict[str, object] = Field(default_factory=dict)


class HostMemberLaunchRequest(_Message):
    replica_id: str
    member: ModelServiceMemberSpec
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    process_uuid: str = Field(min_length=1)
    startup_timeout_s: float = Field(gt=0)
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


class ReplicaFailure(_Message):
    replica_id: str
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    reason: str = Field(min_length=1)


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

    async def prepare_adapter_receive(
        self,
        generation_id: str,
        template_path: str,
        timeout_s: float,
        transport: Literal["local", "nixl"],
    ) -> AdapterTransferTarget: ...

    async def wait_adapter_receive(
        self, generation_id: str, timeout_s: float
    ) -> AdapterReceiveResult: ...

    async def release_adapter_receive(self, generation_id: str) -> None: ...


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
        managed = _ManagedMember(request)
        self._members[key] = managed
        output_dir = self._output_root / request.process_uuid / request.replica_id
        output_dir /= str(request.generation)
        output_dir /= request.member.member_id
        try:
            await managed.runtime.start(
                launch_config=request.launch_config,
                output_dir=str(output_dir),
                child_processes=managed.supervisor,
                install_parent_cleanup=self._install_parent_cleanup,
                timeout=self._startup_timeout_s or request.startup_timeout_s,
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
        key = (replica_id, member_id, generation)
        managed = self._members.get(key)
        if managed is None:
            return
        managed.supervisor.close()
        await asyncio.to_thread(managed.runtime.close)
        self._members.pop(key, None)

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
        spec: ModelServiceSpec,
        launchers: Mapping[str, ReplicaHostLauncher],
        template: ReplicaLaunchTemplate,
        *,
        on_failure: Callable[[ReplicaFailure], Awaitable[None]] | None = None,
        startup_timeout_s: float = 600.0,
        rpc_timeout_s: float = 60.0,
        monitor_interval_s: float = 0.25,
    ) -> None:
        if min(startup_timeout_s, rpc_timeout_s, monitor_interval_s) <= 0:
            raise ValueError("replica timeouts must be positive")
        missing = {member.host_id for member in spec.members} - launchers.keys()
        if missing:
            raise ValueError(f"replica launchers missing hosts: {sorted(missing)}")
        executor = template.engine_args.get("distributed_executor_backend")
        if executor not in (None, "mp", "multiprocessing"):
            raise ValueError("ART-managed replicas require vLLM multiprocessing")
        for key in ("revision", "tokenizer_revision"):
            configured = template.engine_args.get(key)
            if configured is not None and configured != spec.model_revision:
                raise ValueError(f"{key} conflicts with the replica model revision")
        self._spec = spec
        self._launchers = launchers
        self._host_launchers = tuple(
            launchers[host_id]
            for host_id in dict.fromkeys(member.host_id for member in spec.members)
        )
        self._template = template
        self._on_failure = on_failure
        self._startup_timeout_s = startup_timeout_s
        self._rpc_timeout_s = rpc_timeout_s
        self._monitor_interval_s = monitor_interval_s
        self._lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        digest = self._generation_digest(spec, 0)
        self._state = ReplicaState(
            replica_id=spec.name,
            generation=0,
            generation_digest=digest,
            phase="stopped",
        )

    @property
    def spec(self) -> ModelServiceSpec:
        return self._spec

    @property
    def state(self) -> ReplicaState:
        return self._state

    async def start(self) -> ReplicaState:
        async with self._lock:
            return await self._start_locked()

    async def _start_locked(self) -> ReplicaState:
        if self._state.phase != "stopped":
            raise RuntimeError(f"cannot start replica in {self._state.phase} state")
        self._state = self._state.model_copy(update={"phase": "starting"})
        requests = tuple(self._launch_request(member) for member in self._spec.members)
        tasks = [
            asyncio.create_task(
                self._launchers[request.member.host_id].start_member(request)
            )
            for request in requests
        ]
        try:
            async with asyncio.timeout(self._startup_timeout_s + self._rpc_timeout_s):
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
            return await self._stop_locked()

    async def _stop_locked(self) -> ReplicaState:
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
        self._state = self._state.model_copy(update={"phase": "stopped", "members": ()})
        return self._state

    async def restart(
        self,
        *,
        served_model_name: str,
        lora_path: str | None,
        initial_policy_version: int | None,
        initial_generation_id: str | None = None,
    ) -> ReplicaState:
        async with self._lock:
            await self._stop_locked()
            self._template = self._template.model_copy(
                update={
                    "served_model_name": served_model_name,
                    "lora_path": lora_path,
                    "initial_policy_version": initial_policy_version,
                    "initial_generation_id": initial_generation_id,
                }
            )
            generation = self._state.generation + 1
            self._state = ReplicaState(
                replica_id=self._spec.name,
                generation=generation,
                generation_digest=self._generation_digest(self._spec, generation),
                phase="stopped",
            )
            return await self._start_locked()

    def prepare_update(self, *, update_identity: str) -> ReplicaState:
        if self._state.phase != "ready":
            raise RuntimeError(f"cannot update replica in {self._state.phase} state")
        self._state = self._state.model_copy(
            update={"phase": "updating", "update_identity": update_identity}
        )
        return self._state

    async def prepare_adapter_transfer(
        self,
        generation_id: str,
        template_path: str,
        *,
        transport: Literal["local", "nixl"] = "nixl",
    ) -> tuple[AdapterTransferTarget, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    asyncio.wait_for(
                        launcher.prepare_adapter_receive(
                            generation_id,
                            template_path,
                            max(1.0, self._rpc_timeout_s - 1.0),
                            transport,
                        ),
                        self._rpc_timeout_s,
                    )
                    for launcher in self._host_launchers
                )
            )
        )

    async def wait_adapter_transfer(
        self, generation_id: str
    ) -> tuple[AdapterReceiveResult, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    asyncio.wait_for(
                        launcher.wait_adapter_receive(
                            generation_id, self._rpc_timeout_s
                        ),
                        self._rpc_timeout_s,
                    )
                    for launcher in self._host_launchers
                )
            )
        )

    async def release_adapter_transfer(self, generation_id: str) -> None:
        await asyncio.gather(
            *(
                asyncio.wait_for(
                    launcher.release_adapter_receive(generation_id),
                    self._rpc_timeout_s,
                )
                for launcher in self._host_launchers
            )
        )

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
        failure_event: ReplicaFailure | None = None
        async with self._lock:
            if self._state.phase not in {"ready", "updating"}:
                return self._state
            states = await asyncio.gather(
                *(
                    asyncio.wait_for(
                        self._launchers[member.host_id].member_state(
                            self._spec.name,
                            member.member_id,
                            self._state.generation,
                        ),
                        self._rpc_timeout_s,
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
            if failure is None:
                self._state = self._state.model_copy(
                    update={"members": tuple(states)}  # type: ignore[arg-type]
                )
                return self._state
            reason = f"member failure: {failure}"
            generation = self._state.generation
            generation_digest = self._state.generation_digest
            self.quarantine(reason)
            failure_event = ReplicaFailure(
                replica_id=self._spec.name,
                generation=generation,
                generation_digest=generation_digest,
                reason=reason,
            )
            try:
                await self._stop_current_members()
            except BaseException as error:
                reason += f"; teardown failure: {error}"
                self.quarantine(reason)
                failure_event = failure_event.model_copy(update={"reason": reason})
        if self._on_failure is not None:
            await self._on_failure(failure_event)
        return self._state

    async def _monitor(self) -> None:
        current = asyncio.current_task()
        try:
            while self._monitor_task is current and self._state.phase in {
                "ready",
                "updating",
            }:
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
                    self._spec.name,
                    member.member_id,
                    self._state.generation,
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

    async def _stop_calls(
        self,
        calls: tuple[tuple[ReplicaHostLauncher, str, str, int], ...],
    ) -> None:
        pending = calls
        failures: list[BaseException] = []
        for _attempt in range(2):
            results = await asyncio.gather(
                *(
                    asyncio.wait_for(
                        launcher.stop_member(replica_id, member_id, generation),
                        self._rpc_timeout_s,
                    )
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
        engine_args = {
            **self._template.engine_args,
            "tensor_parallel_size": parallel.tp,
            "pipeline_parallel_size": parallel.pp,
            "data_parallel_size": parallel.dp,
            "enable_expert_parallel": parallel.enable_expert_parallel,
        }
        if self._spec.model_revision is not None:
            engine_args.update(
                revision=self._spec.model_revision,
                tokenizer_revision=self._spec.model_revision,
            )
        process_uuid = uuid.uuid4().hex
        physical_ids = all(isinstance(gpu_id, int) for gpu_id in member.gpu_ids)
        launch = VllmRuntimeLaunchConfig(
            base_model=self._spec.base_model,
            port=self._spec.leader_endpoint.port,
            host=(
                self._spec.leader_endpoint.host
                if member.node_rank == 0
                else "127.0.0.1"
            ),
            cuda_visible_devices=(
                None if physical_ids else ",".join(map(str, member.gpu_ids))
            ),
            local_gpu_ids=(
                tuple(gpu_id for gpu_id in member.gpu_ids if isinstance(gpu_id, int))
                if physical_ids
                else None
            ),
            lora_path=self._template.lora_path,
            served_model_name=self._template.served_model_name,
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
            headless=member.node_rank != 0,
            replica_generation=self._state.generation,
            process_uuid=process_uuid,
            update_identity=self._state.update_identity,
            initial_policy_version=self._template.initial_policy_version,
            initial_generation_id=self._template.initial_generation_id,
        )
        return HostMemberLaunchRequest(
            replica_id=self._spec.name,
            member=member,
            generation=self._state.generation,
            generation_digest=self._state.generation_digest,
            process_uuid=process_uuid,
            startup_timeout_s=self._startup_timeout_s,
            launch_config=launch,
        )

    @staticmethod
    def _generation_digest(spec: ModelServiceSpec, generation: int) -> str:
        payload = json.dumps(
            {"generation": generation, "spec": spec.model_dump(mode="json")},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()
