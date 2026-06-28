from __future__ import annotations

import asyncio
import time
from typing import Any
import warnings

from .autotune import PipelineAutotuner, build_initial_settings
from .config import (
    PackedGroupObservation,
    PipelineAutotuneConfig,
    PipelineAutotunerProfile,
    PipelineMetric,
)
from .store import PipelineTunerProfileStore


class PipelineAutotunerAttachment:
    def __init__(self, config: PipelineAutotuneConfig) -> None:
        self.config = config
        self.trainer: Any | None = None
        self.store: PipelineTunerProfileStore | None = None
        self.tuner: PipelineAutotuner | None = None
        self.profile_name = config.output_name
        self._sampler_task: asyncio.Task[None] | None = None
        self._started = False

    async def on_start(self, trainer: Any) -> None:
        if self.config.mode == "off":
            return
        self.trainer = trainer
        self.store = PipelineTunerProfileStore.for_model(trainer.model)
        self._validate_weight_update_mode(trainer)
        packed_sequence_length = self._discover_packed_sequence_length()
        inference_gpu_count = self._discover_inference_gpu_count(trainer)
        loaded = self._load_profile_if_requested(packed_sequence_length)
        if loaded is not None:
            settings = loaded.settings
            self.profile_name = self.config.profile or self.config.output_name
            trainer._pipeline_tuner_profile = self.store.resolve(
                self.config.profile
            ).stem
        else:
            settings = build_initial_settings(
                config=self.config,
                inference_gpu_count=inference_gpu_count,
                limit_mean_steps_off_policy=trainer.limit_mean_steps_off_policy,
                max_steps_off_policy=trainer.max_steps_off_policy,
                explicit_workers=trainer.num_rollout_workers,
                explicit_min_batch_size=trainer.min_batch_size,
                explicit_max_batch_size=trainer.max_batch_size,
                explicit_queue_maxsize=trainer.queue_maxsize,
            )
        trainer.apply_pipeline_settings(settings)
        if self.config.mode == "online":
            self.tuner = PipelineAutotuner(
                config=self.config,
                settings=settings,
                model_name=trainer.model.name,
                backend_name=type(trainer.backend).__name__,
                packed_sequence_length=packed_sequence_length,
                inference_gpu_count=inference_gpu_count,
            )
            self._sampler_task = asyncio.create_task(
                self._sample_serving_metrics(),
                name="art_pipeline_autotuner_vllm_sampler",
            )
            self._save_profile()
        self._started = True

    async def on_metric(self, metric: PipelineMetric) -> None:
        if self.tuner is None:
            return
        decision = self.tuner.on_metric(metric)
        if decision is None:
            return
        assert self.trainer is not None
        self.trainer.apply_pipeline_settings(decision.updated)
        self._save_profile()

    async def on_packed_group(self, observation: PackedGroupObservation) -> None:
        if self.tuner is not None:
            self.tuner.on_packed_group(observation)

    async def on_stop(self) -> None:
        if self._sampler_task is not None:
            self._sampler_task.cancel()
            await asyncio.gather(self._sampler_task, return_exceptions=True)
            self._sampler_task = None
        if self._started and self.tuner is not None:
            self._save_profile()

    async def _sample_serving_metrics(self) -> None:
        assert self.trainer is not None
        collector = getattr(
            self.trainer.backend, "collect_train_step_vllm_metrics", None
        )
        if not callable(collector):
            return
        while not self.trainer.state.done:
            try:
                metrics = await collector(self.trainer.model)
                await self._emit_metrics(metrics, step=None)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(self.config.vllm_metric_interval_s)

    async def _emit_metrics(self, metrics: dict[str, float], step: int | None) -> None:
        now = time.monotonic()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                await self.on_metric(
                    PipelineMetric(name=name, value=float(value), step=step, t_s=now)
                )

    def _save_profile(self) -> None:
        if self.tuner is None or self.store is None:
            return
        path = self.store.save(self.config.output_name, self.tuner.profile())
        if self.trainer is not None:
            self.trainer._pipeline_tuner_profile = path.stem

    def _load_profile_if_requested(
        self, active_packed_sequence_length: int
    ) -> PipelineAutotunerProfile | None:
        if self.config.mode == "online" and not self.config.profile:
            return None
        assert self.store is not None
        profile = self.store.load(self.config.profile)
        if (
            profile.packed_sequence_length is not None
            and profile.packed_sequence_length != active_packed_sequence_length
        ):
            warnings.warn(
                "Autotuner profile was produced with packed_sequence_length="
                f"{profile.packed_sequence_length}, but active config uses "
                f"{active_packed_sequence_length}. Applying saved settings, but "
                "retuning is recommended.",
                stacklevel=2,
            )
        return profile

    @staticmethod
    def _validate_weight_update_mode(trainer: Any) -> None:
        internal_config = trainer.model._internal_config or {}
        if internal_config.get("rollout_weight_update_mode") != "in_flight_lora":
            raise ValueError(
                "ART pipeline autotuning is currently designed and profiled only "
                "for in-flight LoRA update semantics. Other rollout weight update "
                "modes change practical policy-age behavior and need dedicated "
                "tuning work before they can be compared."
            )

    @staticmethod
    def _discover_inference_gpu_count(trainer: Any) -> int:
        internal_config = trainer.model._internal_config or {}
        inference_gpu_ids = internal_config.get("inference_gpu_ids")
        if not inference_gpu_ids:
            raise ValueError(
                "Pipeline autotuning requires dedicated inference_gpu_ids."
            )
        return len(inference_gpu_ids)

    @staticmethod
    def _discover_packed_sequence_length() -> int:
        try:
            from art.megatron.runtime_config import get_megatron_runtime_config
        except Exception as exc:
            raise ValueError(
                "Pipeline autotuning requires a backend with fixed packed sequence length."
            ) from exc
        return get_megatron_runtime_config().packed_sequence_length
