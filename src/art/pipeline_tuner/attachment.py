from __future__ import annotations

import asyncio
import inspect
import math
from queue import Empty, SimpleQueue
import threading
import time
from typing import Any, Literal, NamedTuple
import warnings

import pydantic

from art.errors import ArtVllmMetricsTimeoutError

from .autotune import (
    PipelineAutotuner,
    _vllm_sample_intervals,
    _vllm_sample_max_age_s,
    build_initial_settings,
    freshness_worker_limit,
    recommended_queue_size,
)
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
    scheduled_s: float | None = None
    request_start_s: float | None = None
    outcome: Literal["success", "timeout", "error"] = "success"
    skipped_polls: int = 0


class _VllmMetricPollResult(NamedTuple):
    scheduled_s: float
    request_start_s: float
    completion_s: float
    outcome: Literal["success", "timeout", "error"]
    skipped_polls: int
    metrics: dict[str, float] | None = None
    error: BaseException | None = None


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]


class PipelineAutotunerAttachment:
    def __init__(self, config: PipelineAutotuneConfig) -> None:
        self.config = config
        self.trainer: Any | None = None
        self.store: PipelineTunerProfileStore | None = None
        self.tuner: PipelineAutotuner | None = None
        self.profile_name = config.output_name
        self._sampler_thread: threading.Thread | None = None
        self._sampler_stop = threading.Event()
        self._sampler_results: SimpleQueue[_VllmMetricPollResult] = SimpleQueue()
        self._poll_health: list[VllmMetricPollHealth] = []
        self._train_step_vllm_metrics: dict[str, tuple[float, int]] = {}
        self._sampler_error: BaseException | None = None
        self._started = False

    async def on_start(self, trainer: Any) -> None:
        if self.config.mode == "off":
            return
        self.trainer = trainer
        self.store = PipelineTunerProfileStore.for_model(trainer.model)
        self._validate_weight_update_mode(trainer)
        initial_poll: _VllmMetricPollResult | None = None
        try:
            if self.config.mode == "online":
                self._start_metric_sampler()
                initial_poll = await self._wait_for_initial_serving_metrics()
            packed_sequence_length = self._discover_packed_sequence_length()
            target_packed_sequences = await self._discover_target_packed_sequences(
                trainer
            )
            inference_gpu_count = await self._discover_inference_gpu_count(
                trainer,
                None if initial_poll is None else initial_poll.metrics,
            )
            rollout_worker_capacity = trainer.rollout_worker_capacity
            policy_age_limit_steps = self._policy_age_limit_steps(trainer)
            loaded = self._load_profile_if_requested(
                packed_sequence_length,
                target_packed_sequences,
                policy_age_limit_steps,
                rollout_worker_capacity,
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
                    target_packed_sequences=target_packed_sequences,
                    policy_age_limit_steps=policy_age_limit_steps,
                    rollout_worker_capacity=rollout_worker_capacity,
                )
            trainer.apply_pipeline_settings(settings)
            if self.config.mode == "online":
                self.tuner = PipelineAutotuner(
                    config=self.config,
                    settings=settings,
                    model_name=trainer.model.run_name,
                    backend_name=type(trainer.backend).__name__,
                    packed_sequence_length=packed_sequence_length,
                    target_packed_sequences=target_packed_sequences,
                    inference_gpu_count=inference_gpu_count,
                    policy_age_limit_steps=policy_age_limit_steps,
                    starting_step=trainer.state.next_training_step,
                    rollout_worker_capacity=rollout_worker_capacity,
                )
                assert initial_poll is not None
                self._consume_poll(initial_poll, record_train_step=False)
                self._drain_metric_polls()
                self._save_profile()
            self._started = True
        except BaseException:
            await self._stop_metric_sampler()
            raise

    async def on_metric(self, metric: PipelineMetric) -> None:
        if self.tuner is None:
            return
        self._drain_metric_polls()
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
        await self._stop_metric_sampler()
        if self._started and self.tuner is not None:
            self._save_profile()
        if not training_failed:
            self._raise_sampler_error()

    def _start_metric_sampler(self) -> None:
        if self._sampler_thread is not None:
            raise RuntimeError("ART vLLM metrics sampler is already running")
        self._sampler_stop.clear()
        self._sampler_thread = threading.Thread(
            target=self._metric_sampler_thread_main,
            name="art_pipeline_autotuner_vllm_sampler",
            daemon=True,
        )
        self._sampler_thread.start()

    async def _stop_metric_sampler(self) -> None:
        thread = self._sampler_thread
        if thread is None:
            return
        self._sampler_stop.set()
        await asyncio.to_thread(
            thread.join, max(2.0, 2.0 * self.config.vllm_metric_interval_s)
        )
        if thread.is_alive() and self._sampler_error is None:
            self._sampler_error = RuntimeError(
                "ART vLLM metrics sampler did not stop after its request timeout"
            )
        if not thread.is_alive():
            self._sampler_thread = None
        self._drain_metric_polls()

    def _metric_sampler_thread_main(self) -> None:
        try:
            asyncio.run(self._sample_serving_metrics())
        except BaseException as error:
            now = time.monotonic()
            self._sampler_results.put(
                _VllmMetricPollResult(now, now, now, "error", 0, error=error)
            )

    async def _wait_for_initial_serving_metrics(self) -> _VllmMetricPollResult:
        deadline = time.monotonic() + max(5.0, 2.0 * self.config.vllm_metric_interval_s)
        while True:
            try:
                result = self._sampler_results.get_nowait()
            except Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Pipeline autotuning could not collect an initial ART vLLM "
                        "metrics sample before startup timeout."
                    )
                await asyncio.sleep(min(0.01, self.config.vllm_metric_interval_s))
                continue
            if result.outcome == "success":
                return result
            self._consume_poll(result, record_train_step=False)
            if result.outcome == "error":
                self._raise_sampler_error()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Pipeline autotuning could not collect an initial ART vLLM "
                    "metrics sample before startup timeout."
                ) from result.error

    async def _sample_serving_metrics(self) -> None:
        assert self.trainer is not None
        trainer = self.trainer
        backend = trainer.backend
        factory = getattr(backend, "create_train_step_vllm_metrics_collector", None)
        session = factory(trainer.model) if callable(factory) else None
        if session is not None:
            collector = getattr(session, "collect", None)
        else:
            backend_collector = getattr(
                backend, "collect_train_step_vllm_metrics", None
            )
            collector = (
                None
                if not callable(backend_collector)
                else lambda: backend_collector(trainer.model)
            )
        if not callable(collector):
            raise RuntimeError(
                "Pipeline autotuning requires ART vLLM metrics collection."
            )
        next_s = time.monotonic()
        try:
            while not self._sampler_stop.is_set():
                while not self._sampler_stop.is_set():
                    delay_s = next_s - time.monotonic()
                    if delay_s <= 0.0:
                        break
                    await asyncio.sleep(min(delay_s, 0.05))
                if self._sampler_stop.is_set():
                    break
                scheduled_s = next_s
                request_start_s = time.monotonic()
                metrics: dict[str, float] | None = None
                error: BaseException | None = None
                outcome: Literal["success", "timeout", "error"] = "success"
                try:
                    metrics = await self._collect_required_serving_metrics(collector)
                except ArtVllmMetricsTimeoutError as exc:
                    outcome = "timeout"
                    error = exc
                except Exception as exc:
                    outcome = "error"
                    error = exc
                completion_s = time.monotonic()
                next_s = scheduled_s + self.config.vllm_metric_interval_s
                skipped_polls = 0
                if next_s <= completion_s:
                    skipped_polls = (
                        math.floor(
                            (completion_s - next_s) / self.config.vllm_metric_interval_s
                        )
                        + 1
                    )
                    next_s += skipped_polls * self.config.vllm_metric_interval_s
                self._sampler_results.put(
                    _VllmMetricPollResult(
                        scheduled_s,
                        request_start_s,
                        completion_s,
                        outcome,
                        skipped_polls,
                        metrics,
                        error,
                    )
                )
                if outcome == "error":
                    return
        finally:
            close = (
                getattr(session, "aclose", None)
                if session is not None
                else getattr(backend, "close_train_step_vllm_metrics", None)
            )
            if callable(close):
                maybe_close = close()
                if inspect.isawaitable(maybe_close):
                    await maybe_close

    async def _collect_required_serving_metrics(
        self, collector: Any | None = None
    ) -> dict[str, float]:
        assert self.trainer is not None
        trainer = self.trainer
        if collector is None:
            backend_collector = getattr(
                trainer.backend, "collect_train_step_vllm_metrics", None
            )
            collector = (
                None
                if not callable(backend_collector)
                else lambda: backend_collector(trainer.model)
            )
        if not callable(collector):
            raise RuntimeError(
                "Pipeline autotuning requires ART vLLM metrics collection."
            )
        maybe_metrics = collector()
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

    def _consume_poll(
        self, result: _VllmMetricPollResult, *, record_train_step: bool
    ) -> None:
        self._poll_health.append(
            VllmMetricPollHealth(
                t_s=result.completion_s,
                timed_out=result.outcome == "timeout",
                scheduled_s=result.scheduled_s,
                request_start_s=result.request_start_s,
                outcome=result.outcome,
                skipped_polls=result.skipped_polls,
            )
        )
        if result.outcome == "error":
            self._sampler_error = result.error or RuntimeError(
                "ART vLLM metrics sampler failed without an error"
            )
            if self.trainer is not None:
                self.trainer.request_stop()
            return
        if result.metrics is None:
            return
        if record_train_step:
            for name in _TRAIN_STEP_VLLM_METRICS.intersection(result.metrics):
                value = result.metrics[name]
                if isinstance(value, (int, float)):
                    total, count = self._train_step_vllm_metrics.get(name, (0.0, 0))
                    self._train_step_vllm_metrics[name] = (
                        total + float(value),
                        count + 1,
                    )
        if self.tuner is not None:
            self.tuner.on_vllm_pressure_sample(
                t_s=result.completion_s,
                running=float(result.metrics["vllm/num_requests_running"]),
                waiting_capacity=float(
                    result.metrics["vllm/num_requests_waiting_capacity"]
                ),
            )

    def _drain_metric_polls(self) -> None:
        while True:
            try:
                result = self._sampler_results.get_nowait()
            except Empty:
                return
            self._consume_poll(result, record_train_step=True)

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
        timeouts = sum(poll.timed_out for poll in polls)
        errors = sum(poll.outcome == "error" for poll in polls)
        skipped = sum(poll.skipped_polls for poll in polls)
        poll_slots = len(polls) + skipped
        failed_frac = (timeouts + errors + skipped) / max(poll_slots, 1)
        intervals = _vllm_sample_intervals(
            [
                poll.t_s
                for poll in self._poll_health
                if not poll.timed_out and poll.outcome == "success"
            ],
            window_start_s=stats.window_start_s,
            window_end_s=end_s,
            metric_interval_s=self.config.vllm_metric_interval_s,
        )
        coverage = sum(duration_s for _, duration_s in intervals) / (
            end_s - stats.window_start_s
        )
        min_coverage = 1.0 - self.config.vllm_metric_timeout_window_frac
        decision.stats = stats.model_copy(
            update={
                "vllm_poll_samples": len(polls),
                "vllm_poll_successes": sum(
                    not poll.timed_out and poll.outcome == "success" for poll in polls
                ),
                "vllm_poll_timeouts": timeouts,
                "vllm_poll_errors": errors,
                "vllm_poll_skipped": skipped,
                "vllm_poll_coverage": coverage,
                "vllm_poll_schedule_lag_p99_s": _p99(
                    [
                        max(0.0, poll.request_start_s - poll.scheduled_s)
                        for poll in polls
                        if poll.scheduled_s is not None
                        and poll.request_start_s is not None
                    ]
                ),
                "vllm_poll_request_latency_p99_s": _p99(
                    [
                        max(0.0, poll.t_s - poll.request_start_s)
                        for poll in polls
                        if poll.request_start_s is not None
                    ]
                ),
            }
        )
        if failed_frac > self.config.vllm_metric_timeout_window_frac:
            raise RuntimeError(
                "Pipeline autotuning cannot rely on ART vLLM metrics: "
                f"{failed_frac:.1%} of metric polls timed out, failed, or were "
                f"skipped during decision window steps "
                f"{stats.start_step}-{stats.end_step}."
            )
        if coverage + 1e-9 < min_coverage:
            raise RuntimeError(
                "Pipeline autotuning cannot rely on ART vLLM metrics: successful "
                f"telemetry covered {coverage:.1%} of decision window steps "
                f"{stats.start_step}-{stats.end_step}; requires at least "
                f"{min_coverage:.1%}."
            )
        cutoff_s = end_s - _vllm_sample_max_age_s(self.config.vllm_metric_interval_s)
        self._poll_health = [poll for poll in self._poll_health if poll.t_s >= cutoff_s]

    def _raise_sampler_error(self) -> None:
        if self._sampler_error is not None:
            raise RuntimeError(
                "Pipeline autotuning ART vLLM metrics sampler failed."
            ) from self._sampler_error

    def collect_train_step_metrics(self) -> dict[str, float]:
        self._drain_metric_polls()
        self._raise_sampler_error()
        samples = self._train_step_vllm_metrics
        self._train_step_vllm_metrics = {}
        return {
            name: total / count for name, (total, count) in samples.items() if count > 0
        }

    def _save_profile(self) -> None:
        if self.tuner is None or self.store is None:
            return
        path = self.store.save(self.config.output_name, self.tuner.profile())
        if self.trainer is not None:
            self.trainer._pipeline_tuner_profile = path.stem

    def _load_profile_if_requested(
        self,
        active_packed_sequence_length: int,
        target_packed_sequences: int,
        policy_age_limit_steps: float,
        rollout_worker_capacity: int | None,
    ) -> PipelineAutotunerProfile | None:
        if self.config.mode == "online" and not self.config.profile:
            return None
        assert self.store is not None
        profile = self.store.load(self.config.profile)
        if profile.settings.num_rollout_workers > self.config.max_rollout_workers:
            raise ValueError(
                "Autotuner profile requests "
                f"num_rollout_workers={profile.settings.num_rollout_workers}, which "
                "exceeds the active max_rollout_workers="
                f"{self.config.max_rollout_workers}."
            )
        if (
            rollout_worker_capacity is not None
            and profile.settings.num_rollout_workers > rollout_worker_capacity
        ):
            raise ValueError(
                "Autotuner profile requests "
                f"num_rollout_workers={profile.settings.num_rollout_workers}, above "
                f"current rollout executor capacity {rollout_worker_capacity}."
            )
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
            profile.target_packed_sequences is not None
            and profile.target_packed_sequences != target_packed_sequences
        ):
            warnings.warn(
                "Autotuner profile was produced with target_packed_sequences="
                f"{profile.target_packed_sequences}, but active config uses "
                f"{target_packed_sequences}. Applying saved settings, but "
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
                f"{policy_age_limit_steps}. Recomputing the active worker target for "
                "the active limit.",
                stacklevel=2,
            )
        return profile

    def _settings_with_current_queue(
        self, settings: PipelineTuneSettings, policy_age_limit_steps: float
    ) -> PipelineTuneSettings:
        worker_limit = freshness_worker_limit(
            target_groups_per_step=settings.target_groups_per_step,
            limit_steps_off_policy=policy_age_limit_steps,
            running_reserve_fraction=self.config.queue_running_reserve_fraction,
            worker_step=self.config.worker_step,
        )
        workers = (
            settings.num_rollout_workers
            if worker_limit is None
            else min(settings.num_rollout_workers, worker_limit)
        )
        return settings.model_copy(
            update={
                "min_batch_size": max(
                    settings.min_batch_size,
                    math.ceil(
                        settings.target_groups_per_step
                        * self.config.freshness_min_batch_floor_fraction
                    ),
                ),
                "num_rollout_workers": workers,
                "queue_maxsize": recommended_queue_size(
                    target_groups_per_step=settings.target_groups_per_step,
                ),
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
    async def _discover_inference_gpu_count(
        trainer: Any, serving_metrics: dict[str, float] | None = None
    ) -> int:
        internal_config = trainer.model._internal_config or {}
        inference_gpu_ids = internal_config.get("inference_gpu_ids")
        if inference_gpu_ids:
            return len(inference_gpu_ids)
        metrics = serving_metrics
        if metrics is None:
            collector = getattr(
                trainer.backend, "collect_train_step_vllm_metrics", None
            )
            if not callable(collector):
                raise ValueError(
                    "Pipeline autotuning requires inference_gpu_ids or ART vLLM "
                    "metrics."
                )
            maybe_metrics = collector(trainer.model)
            metrics = (
                await maybe_metrics
                if inspect.isawaitable(maybe_metrics)
                else maybe_metrics
            )
        world_size = metrics.get("vllm/world_size")
        if not isinstance(world_size, (int, float)) or world_size < 1:
            raise ValueError(
                "Pipeline autotuning requires vLLM world size when inference_gpu_ids "
                "are not local to the ART process."
            )
        return int(world_size)

    @staticmethod
    async def _discover_target_packed_sequences(trainer: Any) -> int:
        from art.types import TrainConfig

        backend = trainer.backend
        get_service = getattr(backend, "_get_service", None)
        resolver = getattr(backend, "_resolve_grad_accumulation_sequences", None)
        if callable(get_service) and callable(resolver):
            service = await get_service(trainer.model)
            config = TrainConfig(
                grad_accumulation_sequences=trainer.grad_accumulation_sequences
            )
            return max(1, int(await resolver(service, config)))
        raise ValueError(
            "Pipeline autotuning requires a backend that can resolve global "
            "grad_accumulation_sequences before training starts."
        )

    @staticmethod
    def _discover_packed_sequence_length() -> int:
        try:
            from art.megatron.runtime_config import get_megatron_runtime_config
        except Exception as exc:
            raise ValueError(
                "Pipeline autotuning requires a backend with fixed packed sequence length."
            ) from exc
        return get_megatron_runtime_config().packed_sequence_length
