from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING, Any

from .data_plane import InMemoryPackedBatch, validate_packed_batch
from .specs import TrainJobSpec
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.optimizer_state import OptimizerAdapter


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
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
                adapter_path=job.output.adapter_path,
            ),
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
    ) -> None:
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
        if adapter is not None:
            from art.megatron.optimizer_state import (
                save_optimizer_state_under_model_lease,
            )

            save_optimizer_state_under_model_lease(
                runtime,
                optimizer_state_path=optimizer_state_path,
                step=learner_version,
                adapter=adapter,
            )
        runtime.resident_policy_step = learner_version

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        controller = getattr(self.runtime, "moe_routing_replay_controller", None)
        if controller is not None:
            controller.remove_router_patches()
            self.runtime.moe_routing_replay_controller = None
