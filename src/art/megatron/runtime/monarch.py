from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import os
from threading import Event
from typing import Any

from monarch.actor import Actor, Channel, Port, ProcMesh, endpoint
from monarch.spmd import setup_torch_elastic_env_async

from art.distributed.data_plane import PackedBatchLeaseSet

from .data_plane import InMemoryPackedBatch
from .specs import (
    TRAIN_EVENT_ADAPTER,
    AdapterReady,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerRuntimeSpec,
    TrainEvent,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)


class _ActorEventSink:
    def __init__(self, port: Port[dict[str, Any]] | None) -> None:
        self._port = port

    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None:
        if self._port is not None:
            self._port.send(
                {
                    "kind": "progress",
                    "step_index": step_index,
                    "num_steps": num_steps,
                    "metrics": metrics,
                }
            )

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None:
        if self._port is not None:
            self._port.send(
                {
                    "kind": "adapter_ready",
                    "learner_version": learner_version,
                    "adapter_path": adapter_path,
                }
            )


class MonarchTrainerActor(Actor):
    """One warm Megatron rank, spawned once on every trainer ProcMesh process."""

    def __init__(
        self,
        runtime_spec_json: str,
    ) -> None:
        runtime_spec = TrainerRuntimeSpec.model_validate_json(runtime_spec_json)
        topology = runtime_spec.trainer_mesh.topology
        os.environ.update(
            {
                "MODEL_IDENTIFIER": runtime_spec.model_identifier,
                "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE": str(topology.tp),
                "ART_MEGATRON_CONTEXT_PARALLEL_SIZE": str(topology.cp),
                "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE": str(topology.ep),
                "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE": str(topology.pp),
                "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE": str(topology.etp),
                "ART_MEGATRON_LORA_RANK": str(runtime_spec.lora_rank),
                "ART_MEGATRON_LORA_TARGET_MODULES": json.dumps(
                    runtime_spec.lora_target_modules
                ),
                "ART_DISABLE_MEGATRON_COMPILE": (
                    "0" if runtime_spec.compile_enabled else "1"
                ),
                "ART_MEGATRON_ALLOW_UNVALIDATED_ARCH": str(
                    int(runtime_spec.allow_unvalidated_arch)
                ),
                "ART_MEGATRON_ENABLE_MOE_ROUTING_REPLAY": str(
                    int(runtime_spec.enable_moe_routing_replay)
                ),
                "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD": str(
                    int(runtime_spec.streaming_weight_offload)
                ),
                "ART_MEGATRON_OFFLOAD_BETWEEN_JOBS": "0",
            }
        )
        if runtime_spec.random_state is not None:
            os.environ["ART_MEGATRON_RANDOM_STATE"] = str(runtime_spec.random_state)
        if topology.vpp is not None:
            os.environ["ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE"] = str(
                topology.vpp
            )
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != len(runtime_spec.trainer_mesh.ranks):
            raise RuntimeError(
                "Monarch ProcMesh world does not match TrainerRuntimeSpec: "
                f"{world_size} != {len(runtime_spec.trainer_mesh.ranks)}"
            )

        import torch

        rank = int(os.environ["RANK"])
        placement = runtime_spec.trainer_mesh.ranks[rank]
        self._host_id = placement.host_id
        os.environ["LOCAL_RANK"] = str(placement.gpu_id)
        torch.cuda.set_device(placement.gpu_id)
        from art.megatron.train import build_training_runtime

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[runtime_spec.dtype]
        self._runtime = build_training_runtime(
            model_identifier=runtime_spec.model_identifier,
            provider_torch_dtype=dtype,
            print_env=rank == 0,
        )
        if self._runtime.model_support_handler.key != runtime_spec.handler_name:
            raise RuntimeError(
                "resolved model-support handler does not match TrainerRuntimeSpec: "
                f"{self._runtime.model_support_handler.key!r} != "
                f"{runtime_spec.handler_name!r}"
            )
        from art.megatron.training.streaming_weight_offload import (
            streaming_weight_offload_config_from_env,
        )
        from art.megatron.training.weight_offload import WeightOffloadManager

        from .executor import MegatronTrainJobExecutor

        self._executor = MegatronTrainJobExecutor(self._runtime)
        self._weight_offload = WeightOffloadManager.from_config(
            model=self._runtime.model,
            rank=self._runtime.rank,
            compile_enabled=self._runtime.transformer_layers_compiled,
            offload_between_jobs=False,
            streaming_config=streaming_weight_offload_config_from_env(),
        )
        self._weight_offload.install()
        self._valid = True

    @endpoint
    def execute(
        self,
        job_json: str,
        batch_json: str,
        event_port: Port[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._valid:
            raise RuntimeError("trainer actor runtime is invalid")
        job = TrainJobSpec.model_validate_json(job_json)
        leases = PackedBatchLeaseSet.model_validate_json(batch_json)
        batch = InMemoryPackedBatch.open(job.batch, leases.host_refs[self._host_id])
        coordinator = self._runtime.rank == 0
        try:
            with self._weight_offload.job():
                metrics = self._executor.execute(
                    job,
                    batch,
                    _ActorEventSink(event_port if coordinator else None),
                    Event(),
                )
            if coordinator:
                event_port.send({"kind": "actor_completed", "metrics": metrics})
            return {
                "rank": self._runtime.rank,
                "learner_version": job.learner_version,
                "metrics": metrics if coordinator else {},
            }
        except BaseException:
            self._valid = False
            raise
        finally:
            batch.close()

    @endpoint
    def close(self) -> None:
        self._executor.close()

    def __cleanup__(self, exc: Exception | None) -> None:
        if exc is not None:
            self._valid = False
        self._executor.close()


async def spawn_monarch_trainer_actors(
    proc_mesh: ProcMesh,
    runtime_spec: TrainerRuntimeSpec,
) -> Any:
    """Configure torch-elastic first, then initialize exactly one actor per rank."""
    await setup_torch_elastic_env_async(proc_mesh)
    return proc_mesh.spawn(
        "art_megatron_trainer",
        MonarchTrainerActor,
        runtime_spec.model_dump_json(),
    )


class MonarchTrainerRun:
    def __init__(
        self,
        runtime_spec: TrainerRuntimeSpec,
        run_spec: TrainingRunSpec,
        actors: Any,
        proc_mesh: ProcMesh,
    ) -> None:
        if run_spec.runtime_fingerprint != runtime_spec.fingerprint:
            raise ValueError(
                "training run does not match the trainer runtime fingerprint"
            )
        self.runtime_spec = runtime_spec
        self.run_spec = run_spec
        self._actors = actors
        self._proc_mesh = proc_mesh
        self._learner_version = run_spec.initial_learner_version
        self._jobs: dict[str, tuple[str, tuple[TrainEvent, ...]]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._valid = True

    async def train(
        self, job: TrainJobSpec, batch: PackedBatchLeaseSet
    ) -> AsyncIterator[TrainEvent]:
        cached = self._jobs.get(job.job_id)
        if cached is not None and cached[0] == job.fingerprint:
            for event in cached[1]:
                yield event
            return

        async with self._lock:
            cached = self._jobs.get(job.job_id)
            if cached is not None:
                if cached[0] == job.fingerprint:
                    for event in cached[1]:
                        yield event
                    return
                yield TrainAccepted(
                    job_id=job.job_id,
                    run_id=job.run_id,
                    sequence=0,
                    expected_learner_version=job.expected_learner_version,
                )
                yield self._failed(
                    job,
                    1,
                    RuntimeError("job_id was already used with a different job"),
                    False,
                )
                return
            events: list[TrainEvent] = []

            def emit(event: TrainEvent) -> TrainEvent:
                events.append(event)
                return event

            yield emit(
                TrainAccepted(
                    job_id=job.job_id,
                    run_id=job.run_id,
                    sequence=0,
                    expected_learner_version=job.expected_learner_version,
                )
            )
            error = self._validate(job, batch)
            if error is not None:
                yield emit(self._failed(job, len(events), error, not self._valid))
                self._jobs[job.job_id] = (job.fingerprint, tuple(events))
                return

            send_port, receiver = Channel[dict[str, Any]].open()
            collective = asyncio.ensure_future(
                self._actors.execute.call(
                    job.model_dump_json(), batch.model_dump_json(), send_port
                )
            )
            receive = asyncio.ensure_future(receiver.recv())
            try:
                while True:
                    waiters = {receive}
                    if not collective.done():
                        waiters.add(collective)
                    done, _ = await asyncio.wait(
                        waiters, return_when=asyncio.FIRST_COMPLETED
                    )
                    if collective in done:
                        await collective
                        if receive not in done:
                            continue
                    payload = receive.result()
                    if payload["kind"] == "progress":
                        event = TrainProgress(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            step_index=payload["step_index"],
                            num_steps=payload["num_steps"],
                            metrics=payload["metrics"],
                        )
                    elif payload["kind"] == "adapter_ready":
                        event = AdapterReady(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            learner_version=payload["learner_version"],
                            adapter_path=payload["adapter_path"],
                        )
                    else:
                        values = await collective
                        results = list(values.values())
                        versions = {result["learner_version"] for result in results}
                        ranks = {result["rank"] for result in results}
                        if versions != {job.learner_version} or ranks != set(
                            range(len(results))
                        ):
                            raise RuntimeError(
                                "trainer ranks did not agree on job completion"
                            )
                        self._learner_version = job.learner_version
                        yield emit(
                            TrainCompleted(
                                job_id=job.job_id,
                                run_id=job.run_id,
                                sequence=len(events),
                                learner_version=job.learner_version,
                                metrics=payload["metrics"],
                            )
                        )
                        break
                    yield emit(TRAIN_EVENT_ADAPTER.validate_python(event))
                    receive = asyncio.ensure_future(receiver.recv())
            except BaseException as exc:
                self._valid = False
                if not collective.done():
                    collective.cancel()
                if not receive.done():
                    receive.cancel()
                self._closed = True
                await self._proc_mesh.stop()
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    events.append(
                        TrainCancelled(
                            job_id=job.job_id,
                            run_id=job.run_id,
                            sequence=len(events),
                            reason="train stream was cancelled",
                        )
                    )
                    raise
                failure = self._failed(job, len(events), exc, True)
                events.append(failure)
                yield failure
            finally:
                self._jobs[job.job_id] = (job.fingerprint, tuple(events))

    def _validate(
        self, job: TrainJobSpec, batch: PackedBatchLeaseSet
    ) -> BaseException | None:
        if self._closed:
            return RuntimeError("trainer run is closed")
        if not self._valid:
            return RuntimeError("trainer runtime is invalid")
        if job.job_id in self._jobs:
            return RuntimeError("job_id was already used with a different job")
        if job.run_id != self.run_spec.run_id:
            return ValueError("job run_id does not match this training run")
        if job.training_session_id != self.run_spec.training_session_id:
            return ValueError(
                "job training_session_id does not match this training run"
            )
        if job.output.optimizer_state_path != self.run_spec.optimizer_state_path:
            return ValueError(
                "job optimizer state path does not match this training run"
            )
        if job.expected_learner_version != self._learner_version:
            return ValueError(
                "expected learner version mismatch: "
                f"job={job.expected_learner_version}, runtime={self._learner_version}"
            )
        if batch.ref != job.batch:
            return ValueError("job batch ref does not match supplied packed batch")
        if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
            return ValueError(
                "packed batch sequence length does not match the trainer runtime"
            )
        return None

    @staticmethod
    def _failed(
        job: TrainJobSpec,
        sequence: int,
        exc: BaseException,
        invalidated: bool,
    ) -> TrainFailed:
        return TrainFailed(
            job_id=job.job_id,
            run_id=job.run_id,
            sequence=sequence,
            error_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
            runtime_invalidated=invalidated,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._lock:
            try:
                await self._actors.close.call()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
            finally:
                await self._proc_mesh.stop()
