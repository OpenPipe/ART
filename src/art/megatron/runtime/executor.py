from __future__ import annotations

from threading import Event
from typing import Any

from .data_plane import InMemoryPackedBatch, validate_packed_batch
from .specs import TrainJobSpec
from .trainer_run import EventSink


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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        controller = getattr(self.runtime, "moe_routing_replay_controller", None)
        if controller is not None:
            controller.remove_router_patches()
            self.runtime.moe_routing_replay_controller = None
