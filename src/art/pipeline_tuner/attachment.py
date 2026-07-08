from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any
import warnings

import pydantic

from art.errors import ArtVllmMetricsTimeoutError

from .autotune import PipelineAutotuner, build_initial_settings, recommended_queue_size
from .config import (
    PackedGroupObservation,
    PipelineAutotuneConfig,
    PipelineAutotunerProfile,
    PipelineMetric,
    PipelineTuneSettings,
    TunerDecision,
)
from .store import PipelineTunerProfileStore

_REQUIRED_AUTOTUNE_VLLM_METRICS = frozenset(
    {
        "vllm/num_requests_running",
        "vllm/num_requests_waiting",
        "vllm/num_requests_waiting_capacity",
        "vllm/kv_cache_usage_perc",
        "vllm/num_preemptions_total",
    }
)
_TRAIN_STEP_VLLM_METRICS = frozenset(
    {
        "vllm/prompt_tok_per_s",
        "vllm/completion_tok_per_s",
        "vllm/num_requests_running",
        "vllm/num_requests_waiting",
        "vllm/num_requests_waiting_capacity",
        "vllm/prefix_cache_hit_rate",
        "vllm/kv_cache_usage_perc",
    }
)


class VllmMetricPollHealth(pydantic.BaseModel):
    t_s: float
    timed_out: bool = False


class PipelineAutotunerAttachment:
    def __init__(self, config: PipelineAutotuneConfig) -> None:
        self.config = config
        self.trainer: Any | None = None
        self.store: PipelineTunerProfileStore | None = None
        self.tuner: PipelineAutotuner | None = None
        self.profile_name = config.output_name
        self._sampler_task: asyncio.Task[None] | None = None
        self._poll_health: list[VllmMetricPollHealth] = []
        self._train_step_vllm_metrics: list[PipelineMetric] = []
        self._sampler_error: BaseException | None = None
        self._started = False

    async def on_start(self, trainer: Any) -> None:
        if self.config.mode == "off":
            return
        self.trainer = trainer
        self.store = PipelineTunerProfileStore.for_model(trainer.model)
        self._validate_weight_update_mode(trainer)
        packed_sequence_length = self._discover_packed_sequence_length()
        inference_gpu_count = self._discover_inference_gpu_count(trainer)
        policy_age_limit_steps = self._policy_age_limit_steps(trainer)
        loaded = self._load_profile_if_requested(
            packed_sequence_length, policy_age_limit_steps
        )
        if loaded is not None:
            settings = self._settings_with_current_queue(
                loaded.settings, policy_age_limit_steps
            )
            self.profile_name = self.config.profile or self.config.output_name
            trainer._pipeline_tuner_profile = self.store.resolve(
                self.config.profile
            ).stem
        else:
            settings = build_initial_settings(
                config=self.config,
                inference_gpu_count=inference_gpu_count,
                policy_age_limit_steps=policy_age_limit_steps,
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
                policy_age_limit_steps=policy_age_limit_steps,
            )
            await self._wait_for_initial_serving_metrics()
            self._sampler_task = asyncio.create_task(
                self._sample_serving_metrics(),
                name="art_pipeline_autotuner_vllm_sampler",
            )
            self._save_profile()
        self._started = True

    async def on_metric(self, metric: PipelineMetric) -> None:
        if self.tuner is None:
            return
        self._raise_sampler_error()
        decision = self.tuner.on_metric(metric)
        if decision is None:
            return
        self._raise_if_unhealthy_metric_window(decision)
        assert self.trainer is not None
        self.trainer.apply_pipeline_settings(decision.updated)
        self._save_profile()

    async def on_packed_group(self, observation: PackedGroupObservation) -> None:
        if self.tuner is not None:
            self.tuner.on_packed_group(observation)

    def owns_train_step_vllm_metrics(self) -> bool:
        return self.config.mode == "online"

    async def on_stop(self, *, training_failed: bool = False) -> None:
        if self._sampler_task is not None:
            self._sampler_task.cancel()
            await asyncio.gather(self._sampler_task, return_exceptions=True)
            self._sampler_task = None
        if self._started and self.tuner is not None:
            self._save_profile()
        if not training_failed:
            self._raise_sampler_error()

    async def _wait_for_initial_serving_metrics(self) -> None:
        deadline = time.monotonic() + max(5.0, 2.0 * self.config.vllm_metric_interval_s)
        while True:
            try:
                metrics = await self._collect_required_serving_metrics()
            except ArtVllmMetricsTimeoutError as exc:
                self._record_poll_timeout()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "Pipeline autotuning could not collect an initial ART vLLM "
                        "metrics sample before startup timeout."
                    ) from exc
                await asyncio.sleep(min(self.config.vllm_metric_interval_s, remaining))
                continue
            self._record_poll_success()
            await self._emit_metrics(metrics, step=None, record_train_step=False)
            return

    async def _sample_serving_metrics(self) -> None:
        assert self.trainer is not None
        while not self.trainer.state.done:
            try:
                metrics = await self._collect_required_serving_metrics()
                self._record_poll_success()
                await self._emit_metrics(metrics, step=None)
            except asyncio.CancelledError:
                raise
            except ArtVllmMetricsTimeoutError:
                self._record_poll_timeout()
            except Exception as exc:
                self._sampler_error = exc
                self.trainer.request_stop()
                return
            await asyncio.sleep(self.config.vllm_metric_interval_s)

    async def _collect_required_serving_metrics(self) -> dict[str, float]:
        assert self.trainer is not None
        collector = getattr(
            self.trainer.backend, "collect_train_step_vllm_metrics", None
        )
        if not callable(collector):
            raise RuntimeError(
                "Pipeline autotuning requires ART vLLM metrics collection."
            )
        maybe_metrics = collector(self.trainer.model)
        metrics = (
            await maybe_metrics if inspect.isawaitable(maybe_metrics) else maybe_metrics
        )
        if not isinstance(metrics, dict):
            raise RuntimeError(
                "Pipeline autotuning expected ART vLLM metrics as a dictionary."
            )
        missing = sorted(_REQUIRED_AUTOTUNE_VLLM_METRICS.difference(metrics))
        if missing:
            raise RuntimeError(
                f"Pipeline autotuning requires ART vLLM metrics; missing {missing}."
            )
        for name in _REQUIRED_AUTOTUNE_VLLM_METRICS:
            if not isinstance(metrics[name], (int, float)):
                raise RuntimeError(
                    f"Pipeline autotuning requires numeric ART vLLM metric {name!r}."
                )
        return metrics

    def _record_poll_success(self) -> None:
        self._poll_health.append(VllmMetricPollHealth(t_s=time.monotonic()))

    def _record_poll_timeout(self) -> None:
        self._poll_health.append(
            VllmMetricPollHealth(t_s=time.monotonic(), timed_out=True)
        )

    def _raise_if_unhealthy_metric_window(self, decision: TunerDecision) -> None:
        stats = decision.stats
        if stats is None:
            return
        end_s = max(stats.window_end_s, stats.window_start_s + 1e-6)
        polls = [
            poll
            for poll in self._poll_health
            if stats.window_start_s <= poll.t_s <= end_s
        ]
        if not polls:
            raise RuntimeError(
                "Pipeline autotuning did not collect any ART vLLM metrics polls "
                f"during decision window steps {stats.start_step}-{stats.end_step}."
            )
        timeout_frac = sum(poll.timed_out for poll in polls) / len(polls)
        if timeout_frac > self.config.vllm_metric_timeout_window_frac:
            raise RuntimeError(
                "Pipeline autotuning cannot rely on ART vLLM metrics: "
                f"{timeout_frac:.1%} of metric polls timed out during decision "
                f"window steps {stats.start_step}-{stats.end_step}."
            )

    def _raise_sampler_error(self) -> None:
        if self._sampler_error is not None:
            raise RuntimeError(
                "Pipeline autotuning ART vLLM metrics sampler failed."
            ) from self._sampler_error

    def collect_train_step_metrics(self) -> dict[str, float]:
        samples = self._train_step_vllm_metrics
        self._train_step_vllm_metrics = []
        by_name: dict[str, list[float]] = {}
        for metric in samples:
            by_name.setdefault(metric.name, []).append(metric.value)
        return {
            name: sum(values) / len(values)
            for name, values in by_name.items()
            if values
        }

    async def _emit_metrics(
        self,
        metrics: dict[str, float],
        step: int | None,
        *,
        record_train_step: bool = True,
    ) -> None:
        now = time.monotonic()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                metric = PipelineMetric(
                    name=name, value=float(value), step=step, t_s=now
                )
                if record_train_step and name in _TRAIN_STEP_VLLM_METRICS:
                    self._train_step_vllm_metrics.append(metric)
                await self.on_metric(metric)

    def _save_profile(self) -> None:
        if self.tuner is None or self.store is None:
            return
        path = self.store.save(self.config.output_name, self.tuner.profile())
        if self.trainer is not None:
            self.trainer._pipeline_tuner_profile = path.stem

    def _load_profile_if_requested(
        self, active_packed_sequence_length: int, policy_age_limit_steps: float
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
        if (
            profile.policy_age_limit_steps is not None
            and profile.policy_age_limit_steps != policy_age_limit_steps
        ):
            warnings.warn(
                "Autotuner profile was produced with policy_age_limit_steps="
                f"{profile.policy_age_limit_steps}, but active config uses "
                f"{policy_age_limit_steps}. Recomputing queue size for the "
                "active limit.",
                stacklevel=2,
            )
        return profile

    def _settings_with_current_queue(
        self, settings: PipelineTuneSettings, policy_age_limit_steps: float
    ) -> PipelineTuneSettings:
        return settings.model_copy(
            update={
                "queue_maxsize": recommended_queue_size(
                    target_groups_per_step=settings.min_batch_size,
                    limit_steps_off_policy=policy_age_limit_steps,
                    num_rollout_workers=settings.num_rollout_workers,
                    running_reserve_fraction=self.config.queue_running_reserve_fraction,
                )
            }
        )

    @staticmethod
    def _policy_age_limit_steps(trainer: Any) -> float:
        if trainer.limit_mean_steps_off_policy is not None:
            return float(trainer.limit_mean_steps_off_policy)
        if trainer.max_steps_off_policy is not None:
            return float(trainer.max_steps_off_policy)
        return 1.0

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
