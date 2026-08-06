from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any

from .data_plane import InMemoryPackedBatch, validate_packed_batch
from .specs import TrainerGeneration, TrainJobSpec
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.optimizer_state import OptimizerAdapter


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime, capacity=int(runtime.snapshot_pool_capacity)
        )
        self._closed = False

    def execute(
        self,
        job: TrainJobSpec,
        batch: InMemoryPackedBatch,
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_rl_job

        return execute_megatron_rl_job(
            self.runtime,
            job,
            batch.tensors,
            progress_sink=lambda step_index, num_steps, metrics: sink.progress(
                step_index=step_index,
                num_steps=num_steps,
                metrics=metrics,
            ),
            adapter_ready_sink=lambda: sink.adapter_ready(
                learner_version=job.learner_version,
                adapter_path=job.output_adapter_path,
            ),
            snapshot_sink=self._publisher.submit,
            cancelled=cancelled,
        )

    def advance_without_training(
        self,
        *,
        training_session_id: str,
        expected_learner_version: int,
        learner_version: int,
        optimizer_state_path: str,
        adapter: "OptimizerAdapter | None",
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        if learner_version != expected_learner_version + 1:
            raise ValueError("a no-op learner transition must advance exactly one step")
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != training_session_id
            or runtime.resident_policy_step != expected_learner_version
            or not runtime.optimizer_state_loaded
            or runtime.optimizer is None
        ):
            raise RuntimeError("resident trainer state does not match no-op transition")
        metrics = {}
        if int(runtime.rank) == 0:
            if adapter is None:
                raise RuntimeError("rank zero no-op transition requires an adapter")
            metrics = self._publisher.submit_policy_alias(
                optimizer_state_path=optimizer_state_path,
                expected_step=expected_learner_version,
                adapter=adapter,
            )
        runtime.resident_policy_step = learner_version
        return metrics

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.close()
        controller = getattr(self.runtime, "moe_routing_replay_controller", None)
        if controller is not None:
            controller.remove_router_patches()
            self.runtime.moe_routing_replay_controller = None


class _GenerationPublisher:
    def __init__(self, runtime: Any, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("snapshot pool capacity must be positive")
        self.runtime = runtime
        self.capacity = capacity
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="art-publish")
        self._failures: list[BaseException] = []
        self._in_flight = 0

    def submit(
        self,
        job: TrainJobSpec,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        save_optimizer: bool,
    ) -> dict[str, float]:
        from art.megatron.optimizer_state import snapshot_optimizer_state
        from art.megatron.weights.lora_publish import snapshot_vllm_lora_from_model

        wait_s, in_flight = self._acquire_slot()
        prepare_started = time.perf_counter()
        try:
            lora = snapshot_vllm_lora_from_model(
                model=self.runtime.model,
                adapter_dtypes=adapter_dtypes,
                handler=self.runtime.model_support_handler,
                adapter_config=adapter_config,
                rank=self.runtime.rank,
                world_size=self.runtime.world_size,
            )
            optimizer = (
                snapshot_optimizer_state(
                    self.runtime,
                    generation_id=job.output_generation_id,
                    step=job.learner_version,
                )
                if save_optimizer
                else None
            )
        except BaseException as error:
            try:
                _record_generation_failure(
                    Path(job.output.optimizer_state_path),
                    job.output_generation_id,
                    int(self.runtime.rank),
                    error,
                )
            finally:
                self._release_slot()
            raise
        self._enqueue(
            self._persist_generation,
            generation=job.output.generation,
            optimizer_state_path=job.output.optimizer_state_path,
            staging_adapter_path=job.output.staging_adapter_path,
            lora=lora,
            adapter=None,
            optimizer=optimizer,
        )
        return {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
            "snapshot_prepare_s": time.perf_counter() - prepare_started,
        }

    def submit_policy_alias(
        self,
        *,
        optimizer_state_path: str,
        expected_step: int,
        adapter: "OptimizerAdapter",
    ) -> dict[str, float]:
        wait_s, in_flight = self._acquire_slot()
        self._enqueue(
            self._persist_policy_alias,
            optimizer_state_path,
            expected_step,
            adapter=adapter,
        )
        return {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
        }

    def _acquire_slot(self) -> tuple[float, int]:
        self.raise_if_failed()
        started = time.perf_counter()
        self._slots.acquire()
        wait_s = time.perf_counter() - started
        with self._lock:
            self._in_flight += 1
            return wait_s, self._in_flight

    def _enqueue(self, function: Any, /, *args: Any, **kwargs: Any) -> None:
        try:
            future = self._pool.submit(function, *args, **kwargs)
        except BaseException:
            self._release_slot()
            raise
        future.add_done_callback(self._completed)

    def _persist_policy_alias(
        self,
        optimizer_state_path: str,
        expected_step: int,
        *,
        adapter: "OptimizerAdapter",
    ) -> None:
        from art.megatron.optimizer_state import (
            commit_optimizer_policy_advance,
            read_adapter_publication,
            resolve_committed_optimizer_policy,
            trainer_publication_path,
        )
        from art.utils.output_dirs import get_step_checkpoint_dir

        coordination = trainer_publication_path(
            optimizer_state_path, adapter.generation_id
        )
        try:
            initial = get_step_checkpoint_dir(str(Path(optimizer_state_path).parent), 0)
            policy = resolve_committed_optimizer_policy(
                optimizer_state_path, initial_adapter_path=initial
            )
            if read_adapter_publication(adapter.identity, step=adapter.step) != adapter:
                raise RuntimeError("no-op adapter generation is not immutable")
            if policy.policy_adapter.step > expected_step:
                raise RuntimeError("no-op policy metadata is stale")
            if policy.policy_adapter.step == expected_step:
                commit_optimizer_policy_advance(
                    optimizer_state_path,
                    initial_adapter_path=initial,
                    expected_step=expected_step,
                    adapter=adapter,
                )
                policy = resolve_committed_optimizer_policy(
                    optimizer_state_path, initial_adapter_path=initial
                )
            generation = TrainerGeneration(
                training_session_id=adapter.training_session_id,
                policy_step=adapter.step,
                generation_id=adapter.generation_id,
                adapter_path=adapter.identity,
            )
            _write_json_atomic(
                coordination / "complete.json",
                {
                    "generation": generation.model_dump(mode="json"),
                    "resume_step": policy.policy_adapter.step,
                    "optimizer_step": (
                        0
                        if policy.optimizer_anchor is None
                        else policy.optimizer_anchor.step
                    ),
                },
            )
        except BaseException as error:
            _record_generation_failure(
                Path(optimizer_state_path), adapter.generation_id, 0, error
            )
            raise

    def _persist_generation(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        lora: Any,
        adapter: "OptimizerAdapter | None",
        optimizer: Any,
    ) -> None:
        from art.megatron.optimizer_state import (
            OptimizerAdapter,
            OptimizerShard,
            build_optimizer_manifest,
            commit_optimizer_generation,
            publish_adapter_checkpoint,
            read_committed_optimizer_pointer,
            trainer_publication_path,
            write_optimizer_snapshot_shard,
        )
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        rank = int(self.runtime.rank)
        world_size = int(self.runtime.world_size)
        root = Path(optimizer_state_path)
        coordination = trainer_publication_path(
            optimizer_state_path, generation.generation_id
        )
        coordination.mkdir(parents=True, exist_ok=True)
        try:
            if rank == 0:
                if lora is not None:
                    if staging_adapter_path is None or adapter is not None:
                        raise RuntimeError("new adapter publication is inconsistent")
                    staging = Path(staging_adapter_path)
                    if staging.exists():
                        raise RuntimeError(
                            f"Adapter staging generation exists: {staging}"
                        )
                    save_vllm_lora_snapshot(lora, str(staging))
                    adapter = publish_adapter_checkpoint(
                        staging,
                        step=generation.policy_step,
                        training_session_id=generation.training_session_id,
                        generation_id=generation.generation_id,
                    )
                if adapter is None:
                    raise RuntimeError("rank zero has no immutable adapter")
                if adapter.identity != str(Path(generation.adapter_path).absolute()):
                    raise RuntimeError(
                        "Published adapter path differs from TrainJobSpec"
                    )
                _write_json_atomic(coordination / "adapter.json", adapter.model_dump())

            result: dict[str, Any] = {"rank": rank}
            if optimizer is not None:
                shard = write_optimizer_snapshot_shard(
                    optimizer,
                    optimizer_state_path=optimizer_state_path,
                )
                result.update(
                    shard=shard.model_dump(mode="json"),
                    runtime_sha256=optimizer.runtime_sha256,
                    topology=optimizer.topology.model_dump(mode="json"),
                )
            _write_json_atomic(coordination / f"rank-{rank:08d}.json", result)

            if rank == 0:
                records = _wait_rank_records(coordination, world_size)
                adapter = OptimizerAdapter.model_validate(
                    json.loads((coordination / "adapter.json").read_text("utf-8"))
                )
                if optimizer is not None:
                    runtime_ids = {record["runtime_sha256"] for record in records}
                    topologies = {
                        json.dumps(record["topology"], sort_keys=True)
                        for record in records
                    }
                    if len(runtime_ids) != 1 or len(topologies) != 1:
                        raise RuntimeError(
                            "Trainer ranks produced incompatible optimizer snapshots"
                        )
                    manifest = build_optimizer_manifest(
                        generation=generation.generation_id,
                        step=generation.policy_step,
                        adapter=adapter,
                        runtime_sha256=runtime_ids.pop(),
                        world_size=world_size,
                        shards=[
                            OptimizerShard.model_validate(record["shard"])
                            for record in records
                        ],
                        topology=optimizer.topology,
                    )
                    expected = read_committed_optimizer_pointer(optimizer_state_path)
                    commit_optimizer_generation(
                        optimizer_state_path,
                        manifest,
                        expected_pointer=expected,
                    )
                committed = read_committed_optimizer_pointer(optimizer_state_path)
                optimizer_step = 0 if committed is None else committed.step
                _write_json_atomic(
                    coordination / "complete.json",
                    {
                        "generation": generation.model_dump(mode="json"),
                        "resume_step": (
                            generation.policy_step
                            if optimizer is not None
                            else optimizer_step
                        ),
                        "optimizer_step": optimizer_step,
                    },
                )
            else:
                _wait_for_path(
                    coordination / "complete.json", coordination / "failed.json"
                )
        except BaseException as error:
            _record_generation_failure(root, generation.generation_id, rank, error)
            raise

    def _completed(self, future: Future[None]) -> None:
        try:
            future.result()
        except BaseException as error:
            with self._lock:
                self._failures.append(error)
        finally:
            self._release_slot()

    def _release_slot(self) -> None:
        with self._lock:
            self._in_flight -= 1
        self._slots.release()

    def raise_if_failed(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
        if failures:
            raise BaseExceptionGroup("trainer generation publication failed", failures)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self.raise_if_failed()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _record_generation_failure(
    optimizer_root: Path,
    generation_id: str,
    rank: int,
    error: BaseException,
) -> None:
    from art.megatron.optimizer_state import trainer_publication_path

    coordination = trainer_publication_path(str(optimizer_root), generation_id)
    payload = {"error_type": type(error).__name__, "message": str(error)}
    _write_json_atomic(coordination / f"rank-{rank:08d}.error.json", payload)
    if rank == 0:
        _write_json_atomic(coordination / "failed.json", payload)


def _wait_for_path(path: Path, failed: Path, *, timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if failed.is_file():
            raise RuntimeError(
                f"Peer generation publication failed: {failed.read_text('utf-8')}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for generation publication: {path}")
        time.sleep(0.05)


def _wait_rank_records(path: Path, world_size: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 300.0
    while True:
        errors = tuple(path.glob("rank-*.error.json"))
        if errors:
            raise RuntimeError(
                f"Trainer rank publication failed: {errors[0].read_text('utf-8')}"
            )
        ready = [path / f"rank-{rank:08d}.json" for rank in range(world_size)]
        if all(record.is_file() for record in ready):
            return [json.loads(record.read_text("utf-8")) for record in ready]
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for trainer rank publication records")
        time.sleep(0.05)
