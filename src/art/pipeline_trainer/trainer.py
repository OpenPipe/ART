from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

from typing_extensions import TypeIs

T = TypeVar("T")

import art
from art import TrajectoryGroup
from art.distributed.rollout import DistributedTrajectoryQueue, RolloutExecutor
from art.distributed.trajectory_store import TrajectoryGroupRef
from art.errors import LocalServingUnavailableError
from art.pipeline_tuner import (
    PackedGroupObservation,
    PackedGroupShape,
    PipelineAutotuneConfig,
    PipelineAutotunerAttachment,
    PipelineMetric,
    PipelineRuntimeConfig,
    PipelineTuneSettings,
    RolloutWorkerController,
)

from .checkpoint_retention import (
    CHECKPOINT_CREATED_AT_METRIC,
    CHECKPOINT_EVAL_COMPLETED_METRIC,
    CHECKPOINT_SAVED_METRIC,
    CheckpointInfo,
    CheckpointRetentionContext,
    CheckpointRetentionStrategy,
)
from .state import PipelineState
from .status import StatusReporter
from .types import ConfigT, EvalFn, RolloutFn, ScenarioT, SingleRolloutFn  # noqa: F401

PIPELINE_STATE_KEY = "_pipeline_trainer"
_ROLLOUT_WALL_TIME_KEY = "_art_rollout_wall_s"
_ACTOR_IDLE_TIME_KEY = "_art_actor_idle_s"
_QUEUE_WAIT_TIME_KEY = "_art_queue_wait_s"
_SCORE_FRESHNESS_TAU_STEPS = 8.0
# Rollout critical batch size from the best current GRPO/RLVR evidence. This is
# grounded in reported experiments, not a well-validated universal constant.
_SCORE_CRITICAL_ROLLOUT_BATCH_SIZE = 300.0


class _ScenarioSourceExhausted(Exception):
    pass


class _ResizableAsyncQueue(asyncio.Queue[T]):
    def resize(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("queue maxsize must be positive")
        grew = maxsize > self.maxsize
        internals = cast(Any, self)
        internals._maxsize = maxsize
        if grew:
            for _ in range(min(maxsize - self.qsize(), len(internals._putters))):
                internals._wakeup_next(internals._putters)


def _is_eval_mapping(
    result: Sequence[art.Trajectory | art.TrajectoryGroup]
    | Mapping[str, Sequence[art.Trajectory | art.TrajectoryGroup]],
) -> TypeIs[Mapping[str, Sequence[art.Trajectory | art.TrajectoryGroup]]]:
    return isinstance(result, Mapping)


def _to_async_iterator(iterable: Iterable[T] | AsyncIterator[T]) -> AsyncIterator[T]:
    """Convert a sync Iterable to an AsyncIterator, or pass through if already async."""
    if isinstance(iterable, AsyncIterator):
        # ty cannot currently preserve T through this runtime generic check.
        return cast(AsyncIterator[T], iterable)

    async def _iter():
        for item in iterable:
            yield item

    return _iter()


def _weighted_percentile(points: list[tuple[float, float]], percentile: float) -> float:
    if not points:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(points, key=lambda item: item[0])
    total = sum(weight for _value, weight in ordered)
    if total <= 0:
        return ordered[-1][0]
    target = percentile * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= target:
            return value
    return ordered[-1][0]


def _policy_age_exp(age: float) -> float:
    exponent = max(age, 0.0) / _SCORE_FRESHNESS_TAU_STEPS
    if exponent >= 700.0:
        return math.inf
    return math.exp(exponent)


def make_group_rollout_fn(
    single_rollout_fn: SingleRolloutFn[ScenarioT, ConfigT],
    n: int = 4,
) -> RolloutFn[ScenarioT, ConfigT]:
    """Create a RolloutFn from a SingleRolloutFn by running it N times in parallel."""

    async def group_rollout(
        model: art.TrainableModel,
        scenario: ScenarioT,
        config: ConfigT,
    ) -> TrajectoryGroup:
        if n <= 0:
            return TrajectoryGroup([])
        results = await asyncio.gather(
            *[single_rollout_fn(model, scenario, config) for _ in range(n)],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, LocalServingUnavailableError):
                raise result
        return TrajectoryGroup(results)

    return group_rollout


class PipelineTrainer(Generic[ScenarioT, ConfigT]):
    """Async 3-stage pipeline for rollouts, training, and eval."""

    def __init__(
        self,
        model: art.TrainableModel,
        backend: art.Backend,
        rollout_fn: RolloutFn[ScenarioT, ConfigT],
        scenarios: AsyncIterator[ScenarioT] | Iterable[ScenarioT],
        config: ConfigT,
        eval_fn: EvalFn[ConfigT] | None = None,
        *,
        pipeline: PipelineRuntimeConfig | None = None,
        autotune: PipelineAutotuneConfig | None = None,
        # Training
        learning_rate: float = 1e-5,
        loss_fn: str = "cispo",
        loss_fn_config: dict | None = None,
        normalize_advantages: bool = True,
        grad_accumulation_sequences: int | None = None,
        adam_params: object | None = None,
        kl_penalty_coef: float = 0.0,
        kl_penalty_step_lag: int | None = None,
        max_steps: int | None = None,
        # Discard handling
        discard_queue_multiplier: int = 100,
        limit_mean_steps_off_policy: float | None = None,
        score_reference_groups_per_step: float | None = None,
        score_reference_rollouts_per_group: float | None = None,
        # Status output
        log_interval_seconds: float = 60.0,
        status_ewa_alpha: float = 0.2,
        total_scenarios: int | None = None,
        # Eval/Checkpointing
        eval_every_n_steps: int = 20,
        eval_at_start: bool = True,
        save_checkpoint: bool = True,
        optimizer_save_interval: int = 5,
        checkpoint_retention_strategy: CheckpointRetentionStrategy | None = None,
        checkpoint_retention_interval: int = 1,
        # Resumption
        resume: bool = True,
        rollout_executor: RolloutExecutor | None = None,
    ) -> None:
        autotune = autotune or PipelineAutotuneConfig()
        rollout_workers_explicit = (
            pipeline is not None and "num_rollout_workers" in pipeline.model_fields_set
        )
        if autotune.mode != "off" and pipeline is not None:
            raise ValueError(
                "Pipeline runtime config cannot be provided when pipeline autotuning "
                "is enabled. The autotuner owns the initial and online pipeline "
                "settings."
            )
        pipeline = pipeline or PipelineRuntimeConfig()
        if eval_every_n_steps < 0:
            raise ValueError("eval_every_n_steps must be >= 0")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be > 0")
        if discard_queue_multiplier <= 0:
            raise ValueError("discard_queue_multiplier must be > 0")
        if limit_mean_steps_off_policy is not None and limit_mean_steps_off_policy < 0:
            raise ValueError("limit_mean_steps_off_policy must be >= 0")
        if checkpoint_retention_interval <= 0:
            raise ValueError("checkpoint_retention_interval must be > 0")
        if optimizer_save_interval <= 0:
            raise ValueError("optimizer_save_interval must be > 0")
        if kl_penalty_step_lag is not None and kl_penalty_step_lag < 1:
            raise ValueError("kl_penalty_step_lag must be >= 1")
        if grad_accumulation_sequences is not None and grad_accumulation_sequences < 1:
            raise ValueError("grad_accumulation_sequences must be >= 1")
        self.model = model
        self.backend = backend
        self.rollout_fn = rollout_fn
        if rollout_executor is None:
            from art.distributed.rollout import LocalRolloutExecutor

            rollout_executor = LocalRolloutExecutor()
        self._rollout_executor = rollout_executor
        self.rollout_worker_capacity = rollout_executor.max_workers
        if self.rollout_worker_capacity is not None:
            if self.rollout_worker_capacity < 1:
                raise ValueError("rollout executor capacity must be >= 1")
            if pipeline.num_rollout_workers > self.rollout_worker_capacity:
                if autotune.mode == "off" and rollout_workers_explicit:
                    raise ValueError(
                        f"num_rollout_workers={pipeline.num_rollout_workers} exceeds "
                        f"rollout executor capacity {self.rollout_worker_capacity}"
                    )
                pipeline = pipeline.model_copy(
                    update={"num_rollout_workers": self.rollout_worker_capacity}
                )
        self.config = config
        self.eval_fn = eval_fn
        self.pipeline = pipeline
        self.autotune = autotune
        self.num_rollout_workers = pipeline.num_rollout_workers
        self.min_batch_size = pipeline.min_batch_size
        self.max_batch_size = (
            pipeline.max_batch_size
            if pipeline.max_batch_size is not None
            else pipeline.min_batch_size
        )
        self.target_groups_per_step = self.max_batch_size
        self.max_steps_off_policy = pipeline.max_steps_off_policy
        self.limit_mean_steps_off_policy = limit_mean_steps_off_policy
        self.queue_maxsize = pipeline.queue_maxsize
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.loss_fn_config = loss_fn_config
        self.normalize_advantages = normalize_advantages
        self.grad_accumulation_sequences = grad_accumulation_sequences
        self.adam_params = adam_params
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_penalty_step_lag = kl_penalty_step_lag
        self.max_steps = max_steps
        self._status_log_interval_seconds = log_interval_seconds
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_at_start = eval_at_start
        self.save_checkpoint = save_checkpoint
        self.optimizer_save_interval = optimizer_save_interval
        self.checkpoint_retention_strategy = checkpoint_retention_strategy
        self.checkpoint_retention_interval = checkpoint_retention_interval
        self.score_reference_groups_per_step = (
            score_reference_groups_per_step
            if score_reference_groups_per_step is not None
            else pipeline.score_reference_groups_per_step
        )
        self.score_reference_rollouts_per_group = (
            score_reference_rollouts_per_group
            if score_reference_rollouts_per_group is not None
            else pipeline.score_reference_rollouts_per_group
        )
        self.resume = resume
        self.discard_queue_multiplier = discard_queue_multiplier
        self._discard_queue: list[TrajectoryGroup] = []
        self._discard_queue_limit = discard_queue_multiplier * self.min_batch_size
        self._collapse_triggered = False
        self._checkpoint_lease_counts: Counter[int] = Counter()
        self._scheduled_eval_steps: set[int] = set()
        self._scheduled_eval_leases: dict[int, AsyncExitStack] = {}

        self.state = PipelineState()
        self._scenario_lock = asyncio.Lock()
        self._scenario_iter: AsyncIterator[ScenarioT] | None = _to_async_iterator(
            scenarios
        )
        self._output_queue: (
            asyncio.Queue[TrajectoryGroup | None] | DistributedTrajectoryQueue | None
        ) = None
        self._eval_queue: asyncio.Queue[int] | None = None
        self._rollout_worker_controller = RolloutWorkerController(
            self, self.num_rollout_workers
        )
        self._rollout_executor.set_target(self.num_rollout_workers)
        self._attachments: list[PipelineAutotunerAttachment] = []
        if self.autotune.mode != "off":
            self._attachments.append(PipelineAutotunerAttachment(self.autotune))
        self._pipeline_tuner_profile: str | None = None
        self._status = StatusReporter(
            get_scenario_offset=lambda: self.state.scenario_offset,
            log_interval_seconds=log_interval_seconds,
            status_ewa_alpha=status_ewa_alpha,
            total_scenarios=total_scenarios,
            num_workers=self.num_rollout_workers,
        )
        self._validate_backend_support()

    async def train(self, *, handle_signals: bool = True) -> None:
        """Run the training pipeline over the configured scenario iterator."""
        start_step = await self.model.get_step()
        pipeline_state = self._read_pipeline_state() if self.resume else {}
        scenario_offset = int(pipeline_state.get("scenario_offset", 0) or 0)
        last_eval_step = int(pipeline_state.get("last_eval_step", 0) or 0)
        stored_step = pipeline_state.get("training_step")

        if stored_step is not None and int(stored_step) != start_step:
            print(
                "Warning: pipeline trainer state step does not match backend step "
                f"({stored_step} != {start_step}); using backend step."
            )

        self.state.policy_version = start_step
        self.state.next_training_step = start_step
        self.state.scenario_offset = scenario_offset
        self.state.total_scenarios_consumed = int(
            pipeline_state.get("total_scenarios_consumed", scenario_offset) or 0
        )
        self.state.accepted_trainable_groups = int(
            pipeline_state.get("accepted_trainable_groups", 0) or 0
        )
        self.state.discarded_stale_groups = int(
            pipeline_state.get("discarded_stale_groups", 0) or 0
        )
        self.state.discarded_zero_variance_groups = int(
            pipeline_state.get("discarded_zero_variance_groups", 0) or 0
        )
        self.state.last_eval_step = last_eval_step
        self.state.completed_eval_steps = {
            int(step) for step in pipeline_state.get("completed_eval_steps", []) or []
        }

        if scenario_offset > 0 and self._scenario_iter is not None:
            skipped = await self._skip_scenarios(self._scenario_iter, scenario_offset)
            self.state.scenario_offset = skipped
            self.state.total_scenarios_consumed = skipped

        try:
            await self._start_attachments()
        except BaseException as primary:
            try:
                await self._stop_attachments(training_failed=True)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Pipeline attachment startup and cleanup failed.",
                    [primary, cleanup],
                ) from None
            raise

        queue_maxsize = (
            self.queue_maxsize
            if self.queue_maxsize is not None
            else max(1, self._freshness_queue_window() * self.target_groups_per_step)
        )
        result_queue_factory = getattr(
            self._rollout_executor, "create_result_queue", None
        )
        if callable(result_queue_factory):
            self._output_queue = result_queue_factory(queue_maxsize)
            await self._output_queue.start()
        else:
            self._output_queue = _ResizableAsyncQueue(maxsize=queue_maxsize)
        self._eval_queue = asyncio.Queue()

        loop = asyncio.get_running_loop()
        stop_requested = False
        installed_handlers: list[tuple[str, signal.Signals]] = []
        original_handlers: dict[signal.Signals, object] = {}

        def _request_stop(sig: signal.Signals) -> None:
            nonlocal stop_requested
            if stop_requested:
                return
            stop_requested = True
            print(f"Shutdown requested ({sig.name}); finishing current work...")
            self.request_stop()

        def _sync_signal_handler(signum: int, _frame: object | None) -> None:
            _request_stop(signal.Signals(signum))

        training_failed = False
        try:
            if self.eval_fn is not None and self.eval_at_start:
                await self._schedule_eval_step(start_step)
                self._persist_state(start_step)

            self._status.start(initial_step=start_step)
            if handle_signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    original_handlers[sig] = signal.getsignal(sig)
                    try:
                        loop.add_signal_handler(sig, _request_stop, sig)
                        installed_handlers.append(("loop", sig))
                    except (NotImplementedError, RuntimeError):
                        try:
                            signal.signal(sig, _sync_signal_handler)
                            installed_handlers.append(("signal", sig))
                        except (ValueError, RuntimeError):
                            continue

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._rollout_stage(), name="rollout_stage")
                    tg.create_task(self._training_stage(), name="training_stage")
                    tg.create_task(self._eval_stage(), name="eval_stage")
                    tg.create_task(self._status_loop(), name="status_loop")
            except* Exception as eg:
                training_failed = True
                self.request_stop()
                for exc in eg.exceptions:
                    if not isinstance(exc, asyncio.CancelledError):
                        print(f"Pipeline stage failed: {exc}")
                raise
        finally:
            primary_failure = sys.exception()
            training_failed = training_failed or primary_failure is not None
            if handle_signals:
                for mode, sig in installed_handlers:
                    if mode == "loop":
                        try:
                            loop.remove_signal_handler(sig)
                        except (NotImplementedError, RuntimeError):
                            pass
                    try:
                        previous = original_handlers.get(sig)
                        if previous is not None:
                            signal.signal(sig, cast(signal.Handlers, previous))
                    except (ValueError, RuntimeError):
                        pass
            cleanup_failures: list[BaseException] = []
            if isinstance(self._output_queue, DistributedTrajectoryQueue):
                try:
                    await self._output_queue.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            try:
                await self._stop_attachments(training_failed=training_failed)
            except BaseException as exc:
                cleanup_failures.append(exc)
            for cleanup in (self._status.flush, self._status.close):
                try:
                    cleanup()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            try:
                await self._release_all_scheduled_eval_leases()
            except BaseException as exc:
                cleanup_failures.append(exc)
            if cleanup_failures:
                if primary_failure is not None:
                    raise BaseExceptionGroup(
                        "Pipeline training and cleanup failed.",
                        [primary_failure, *cleanup_failures],
                    ) from None
                if len(cleanup_failures) == 1:
                    raise cleanup_failures[0]
                raise BaseExceptionGroup(
                    "Pipeline cleanup failed.", cleanup_failures
                ) from None

    def request_stop(self) -> None:
        """Request a clean shutdown of the pipeline stages."""
        if self.state.done:
            return
        self.state.done = True

        async def _notify_policy() -> None:
            async with self.state.policy_updated:
                self.state.policy_updated.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return

        loop.create_task(_notify_policy())
        if self._output_queue is not None:
            if isinstance(self._output_queue, DistributedTrajectoryQueue):
                loop.create_task(self._output_queue.finish())
                return
            try:
                self._output_queue.put_nowait(None)
            except asyncio.QueueFull:
                loop.create_task(self._output_queue.put(None))

    def apply_pipeline_settings(self, settings: PipelineTuneSettings) -> None:
        if (
            self.rollout_worker_capacity is not None
            and settings.num_rollout_workers > self.rollout_worker_capacity
        ):
            raise ValueError(
                f"num_rollout_workers={settings.num_rollout_workers} exceeds rollout "
                f"executor capacity {self.rollout_worker_capacity}"
            )
        self.num_rollout_workers = settings.num_rollout_workers
        self.min_batch_size = settings.min_batch_size
        self.max_batch_size = settings.max_batch_size
        self.target_groups_per_step = settings.target_groups_per_step
        self.queue_maxsize = settings.queue_maxsize
        self._discard_queue_limit = self.discard_queue_multiplier * self.min_batch_size
        self._rollout_worker_controller.set_target(self.num_rollout_workers)
        self._rollout_executor.set_target(self.num_rollout_workers)
        if self._output_queue is not None:
            if isinstance(self._output_queue, DistributedTrajectoryQueue):
                self._output_queue.set_maxsize(self.queue_maxsize)
            else:
                cast(
                    _ResizableAsyncQueue[TrajectoryGroup | None], self._output_queue
                ).resize(self.queue_maxsize)
        self._status._num_workers = self.num_rollout_workers

    async def _start_attachments(self) -> None:
        for attachment in self._attachments:
            await attachment.on_start(self)

    async def _stop_attachments(self, *, training_failed: bool = False) -> None:
        failures: list[BaseException] = []
        for attachment in reversed(self._attachments):
            try:
                await attachment.on_stop(training_failed=training_failed)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("Pipeline attachment cleanup failed.", failures)

    async def _emit_pipeline_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None,
        tags: dict[str, str] | None = None,
    ) -> None:
        if not self._attachments:
            return
        metric = PipelineMetric(
            name=name,
            value=float(value),
            step=step,
            t_s=time.monotonic(),
            tags=tags or {},
        )
        for attachment in self._attachments:
            await attachment.on_metric(metric)

    async def _emit_pipeline_metrics(
        self, metrics: Mapping[str, float], *, step: int | None
    ) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                await self._emit_pipeline_metric(name, float(value), step=step)

    def _collect_attachment_train_step_metrics(self) -> tuple[dict[str, float], bool]:
        metrics: dict[str, float] = {}
        owns_vllm_metrics = False
        for attachment in self._attachments:
            collector = getattr(attachment, "collect_train_step_metrics", None)
            if not callable(collector):
                continue
            attachment_metrics = collector()
            if not isinstance(attachment_metrics, Mapping):
                raise RuntimeError(
                    "Pipeline attachment train-step metrics collector returned a "
                    "non-mapping result."
                )
            metrics.update(
                {
                    name: float(value)
                    for name, value in attachment_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            owns_vllm_metrics = owns_vllm_metrics or bool(
                getattr(attachment, "owns_train_step_vllm_metrics", lambda: False)()
            )
        return metrics, owns_vllm_metrics

    async def _emit_packed_group_observations(
        self, metrics: Mapping[str, float], *, batch: list[TrajectoryGroup], step: int
    ) -> None:
        if not self._attachments:
            return
        observations: list[PackedGroupObservation] = []
        for group in batch:
            shape = group._packed_group_shape
            group._collect_packing_shape = False
            group._packed_group_shape = None
            if shape is None:
                continue
            if not isinstance(shape, PackedGroupShape):
                raise RuntimeError("Backend returned an invalid packed-group shape")
            observations.append(
                PackedGroupObservation(
                    step=step,
                    leaves=shape.leaves,
                )
            )
        groups = int(metrics.get("data/step_num_groups_trainable", 0.0) or 0)
        if groups > 0 and len(observations) != groups:
            raise RuntimeError(
                "Pipeline autotuning requires packed-group token observations "
                "from the backend packer."
            )
        if not observations:
            return
        for observation in observations:
            for attachment in self._attachments:
                await attachment.on_packed_group(observation)

    def _validate_backend_support(self) -> None:
        from art.local.backend import LocalBackend

        if self.eval_fn is not None and not callable(
            getattr(self.backend, "exact_adapter_lease", None)
        ):
            raise ValueError(
                "PipelineTrainer eval requires a backend with exact checkpoint "
                "inference leases."
            )
        if not isinstance(self.backend, LocalBackend):
            return

        model_config = self.model._internal_config or art.dev.InternalModelConfig()
        if not self.backend._supports_concurrent_training_and_inference(self.model):
            raise ValueError(
                "PipelineTrainer only supports LocalBackend in dedicated mode. "
                "Shared LocalBackend pauses inference during training and is not "
                "a supported async PipelineTrainer path. Set both "
                "trainer_gpu_ids and inference_gpu_ids on the TrainableModel "
                "_internal_config to use LocalBackend with PipelineTrainer."
            )
        if (
            self.eval_fn is not None
            and model_config.get("tinker_args") is None
            and model_config.get("rollout_weights_mode", "lora") != "lora"
        ):
            raise ValueError(
                "PipelineTrainer eval requires rollout_weights_mode='lora' so "
                "the requested checkpoint can remain immutable during eval."
            )
        if self.loss_fn not in {"cispo", "ppo"}:
            raise ValueError(
                "PipelineTrainer + LocalBackend(dedicated) only supports "
                "loss_fn='cispo' or loss_fn='ppo'."
            )
        if self.loss_fn_config is not None:
            raise ValueError(
                "PipelineTrainer + LocalBackend(dedicated) requires "
                "loss_fn_config=None."
            )
        if self.adam_params is not None:
            raise ValueError(
                "PipelineTrainer + LocalBackend(dedicated) requires adam_params=None."
            )

    async def _skip_scenarios(
        self, scenarios: AsyncIterator[ScenarioT], count: int
    ) -> int:
        skipped = 0
        while skipped < count:
            try:
                await anext(scenarios)
            except StopAsyncIteration:
                break
            skipped += 1
        if skipped < count:
            print(
                f"Warning: scenario iterator exhausted early while skipping "
                f"(skipped {skipped}/{count})."
            )
        return skipped

    async def _get_next_scenario(self) -> ScenarioT | None:
        if self._scenario_iter is None:
            return None
        async with self._scenario_lock:
            try:
                scenario = await anext(self._scenario_iter)
            except StopAsyncIteration:
                raise _ScenarioSourceExhausted
            self.state.scenario_offset += 1
            self.state.total_scenarios_consumed += 1
            return scenario

    async def _wait_for_policy(self) -> None:
        if self.max_steps_off_policy is None:
            return
        async with self.state.policy_updated:
            while (
                not self.state.done
                and self.state.policy_version
                < self.state.next_training_step - self.max_steps_off_policy
            ):
                await self.state.policy_updated.wait()

    @asynccontextmanager
    async def _checkpoint_lease(self, step: int) -> AsyncIterator[None]:
        self._checkpoint_lease_counts[step] += 1
        try:
            yield
        finally:
            self._release_checkpoint_lease(step)

    @asynccontextmanager
    async def _adapter_retention_lease(self, step: int) -> AsyncIterator[None]:
        async with self._checkpoint_lease(step):
            if not hasattr(type(self.backend), "adapter_retention_lease"):
                yield
                return
            lease = getattr(self.backend, "adapter_retention_lease", None)
            if lease is None:
                yield
                return
            async with lease(self.model, step):
                yield

    @asynccontextmanager
    async def _adapter_lease(self, step: int) -> AsyncIterator[None]:
        if not hasattr(type(self.backend), "adapter_lease"):
            async with self._checkpoint_lease(step):
                yield
            return
        async with self._checkpoint_lease(step):
            lease = getattr(self.backend, "adapter_lease", None)
            if lease is None:
                yield
                return
            async with lease(self.model, step):
                yield

    @asynccontextmanager
    async def _exact_adapter_lease(self, step: int) -> AsyncIterator[None]:
        lease = getattr(self.backend, "exact_adapter_lease")
        async with self._checkpoint_lease(step), lease(self.model, step):
            yield

    def _release_checkpoint_lease(self, step: int) -> None:
        self._checkpoint_lease_counts[step] -= 1
        if self._checkpoint_lease_counts[step] <= 0:
            del self._checkpoint_lease_counts[step]

    async def _schedule_eval_step(self, step: int) -> None:
        if self._eval_queue is None:
            raise RuntimeError("eval queue is not initialized")
        if step in self._scheduled_eval_steps:
            return
        stack = AsyncExitStack()
        await stack.enter_async_context(self._adapter_retention_lease(step))
        try:
            self._scheduled_eval_leases[step] = stack
            self._scheduled_eval_steps.add(step)
            await self._eval_queue.put(step)
            self.state.last_eval_step = step
        except Exception:
            self._scheduled_eval_steps.discard(step)
            self._scheduled_eval_leases.pop(step, None)
            await stack.aclose()
            raise

    async def _release_scheduled_eval_lease(self, step: int) -> None:
        self._scheduled_eval_steps.discard(step)
        stack = self._scheduled_eval_leases.pop(step, None)
        if stack is not None:
            await stack.aclose()

    async def _release_all_scheduled_eval_leases(self) -> None:
        for step in tuple(self._scheduled_eval_leases):
            await self._release_scheduled_eval_lease(step)

    def _retained_adapter_steps(self, current_step: int) -> set[int]:
        min_step = max(0, current_step - self._retention_window_steps())
        return set(range(min_step, current_step + 1))

    def _kl_penalty_reference_step(self, current_step: int) -> int:
        if self.kl_penalty_step_lag is None:
            return 0
        return max(0, current_step - self.kl_penalty_step_lag)

    async def _prune_model_adapters(self, current_step: int) -> None:
        if not hasattr(type(self.backend), "prune_model_adapters"):
            return
        prune = getattr(self.backend, "prune_model_adapters", None)
        if prune is None:
            return
        await prune(
            self.model,
            retain_steps=self._retained_adapter_steps(current_step),
        )

    async def _rollout_worker(self, worker_id: int) -> None:
        assert self._output_queue is not None
        while not self.state.done:
            if not self._rollout_worker_controller.worker_allowed(worker_id):
                break
            try:
                scenario = await self._get_next_scenario()
            except _ScenarioSourceExhausted:
                self.request_stop()
                break
            if scenario is None:
                break
            self._status.note_rollout_started()
            errored = False
            try:
                wait_started = time.monotonic()
                await self._wait_for_policy()
                actor_idle_s = time.monotonic() - wait_started
                if self.state.done:
                    break

                initial_version = self.state.policy_version

                token = self.model.activate_metrics_context("train")
                rollout_started = time.monotonic()
                try:
                    async with self._adapter_lease(initial_version):
                        group = await self._rollout_executor.run(
                            worker_id,
                            self.rollout_fn,
                            self.model,
                            scenario,
                            self.config,
                        )
                finally:
                    token.var.reset(token)
                rollout_wall_s = time.monotonic() - rollout_started
                if not isinstance(group, TrajectoryGroup | TrajectoryGroupRef):
                    errored = True
                    continue
                scenario_metadata = self._scenario_metadata(scenario)
                if isinstance(group, TrajectoryGroup):
                    group.metadata.update(scenario_metadata)
                    self._apply_policy_versions(
                        group,
                        initial_version=initial_version,
                        final_version=self.state.policy_version,
                    )
                if self.state.done:
                    if isinstance(
                        self._output_queue, DistributedTrajectoryQueue
                    ) and isinstance(group, TrajectoryGroupRef):
                        await self._output_queue.discard(group)
                    break
                queue_wait_s = await self._put_output_group(
                    group,
                    metadata=scenario_metadata,
                    initial_policy_version=initial_version,
                    final_policy_version=self.state.policy_version,
                    rollout_wall_s=rollout_wall_s,
                    actor_idle_s=actor_idle_s,
                )
                if isinstance(group, TrajectoryGroup):
                    group.metadata[_ROLLOUT_WALL_TIME_KEY] = rollout_wall_s
                    group.metadata[_QUEUE_WAIT_TIME_KEY] = queue_wait_s
                    group.metadata[_ACTOR_IDLE_TIME_KEY] = actor_idle_s + queue_wait_s
            except asyncio.CancelledError:
                raise
            except LocalServingUnavailableError:
                raise
            except Exception as exc:
                errored = True
                exc_type = f"{type(exc).__module__}.{type(exc).__name__}"
                print(
                    f"Worker {worker_id}: rollout failed ({exc_type}): {exc!r}"
                    f"{self._scenario_error_context(scenario)}"
                )
            finally:
                self._status.note_rollout_finished(errored=errored)

    async def _rollout_stage(self) -> None:
        try:
            await self._rollout_worker_controller.run()
        except _ScenarioSourceExhausted:
            print("Scenario source exhausted; stopping training.")
            self.request_stop()
        if not self.state.done and self._output_queue is not None:
            if isinstance(self._output_queue, DistributedTrajectoryQueue):
                await self._output_queue.finish()
                return
            try:
                self._output_queue.put_nowait(None)
            except asyncio.QueueFull:
                await self._output_queue.put(None)

    async def _training_stage(self) -> None:
        if self._output_queue is None:
            return

        current_step = self.state.next_training_step
        stop_at_step = (
            current_step + self.max_steps if self.max_steps is not None else None
        )
        if stop_at_step is not None and current_step >= stop_at_step:
            self.state.done = True
            self._persist_state(current_step)
            async with self.state.policy_updated:
                self.state.policy_updated.notify_all()
            return
        stop_after_batch = False

        while True:
            if stop_at_step is not None and current_step >= stop_at_step:
                break
            step_start = time.monotonic()
            collect_started = time.monotonic()
            batch, discarded, saw_sentinel = await self._collect_batch(current_step)
            trainer_idle_s = time.monotonic() - collect_started
            self.state.discarded_stale_groups += discarded
            if discarded:
                self._status.note_stale(discarded)
            if not batch:
                break

            actor_wall_s, actor_idle_s, queue_wait_s = (
                self._consume_batch_rollout_timings(batch)
            )

            training_policy_step = current_step
            expected_step = current_step + 1
            should_eval_step = self._should_eval_step(expected_step)
            should_checkpoint = self.save_checkpoint and should_eval_step

            async with self.state.policy_updated:
                self.state.next_training_step = expected_step
                self.state.policy_updated.notify_all()

            self._status.note_training_start(len(batch))
            train_call_start = time.monotonic()
            if os.getenv("ART_TRAIN_STEP_LOG"):
                print(f"[train] step {expected_step} starting (batch={len(batch)})")
            try:
                train_kwargs: dict[str, Any] = {
                    "learning_rate": self.learning_rate,
                    "loss_fn": self.loss_fn,
                    "loss_fn_config": self.loss_fn_config,
                    "normalize_advantages": self.normalize_advantages,
                    "save_checkpoint": should_checkpoint,
                    "adam_params": self.adam_params,
                    "optimizer_save_interval": self.optimizer_save_interval,
                    "final_training_step": stop_at_step,
                }
                if self.grad_accumulation_sequences is not None:
                    train_kwargs["grad_accumulation_sequences"] = (
                        self.grad_accumulation_sequences
                    )
                if self.kl_penalty_coef > 0.0:
                    kl_penalty_reference_step = self._kl_penalty_reference_step(
                        current_step
                    )
                    train_kwargs["kl_penalty_coef"] = self.kl_penalty_coef
                    train_kwargs["kl_penalty_source"] = "sample"
                    train_kwargs["kl_penalty_reference_step"] = (
                        kl_penalty_reference_step
                    )
                if self.autotune.mode != "off":
                    for group in batch:
                        group._collect_packing_shape = True
                result = await self.backend.train(
                    self.model,
                    batch,
                    **train_kwargs,
                )
            except Exception:
                for group in batch:
                    group._collect_packing_shape = False
                    group._packed_group_shape = None
                self._status.note_training_end()
                raise
            finally:
                train_call_elapsed = time.monotonic() - train_call_start
                if os.getenv("ART_TRAIN_STEP_LOG"):
                    print(
                        f"[train] step {expected_step} done in "
                        f"{train_call_elapsed:.1f}s"
                    )

            try:
                current_step = result.step
                self.state.policy_version = current_step
                self.state.next_training_step = current_step
                await self._log_checkpoint_saved(result)
                await self._prune_model_adapters(current_step)
                await self._run_checkpoint_retention(current_step)

                step_seconds = time.monotonic() - step_start
                self._status.note_training_batch(
                    batch, step=current_step, step_seconds=step_seconds
                )

                stale_groups = float(self.state.discarded_stale_groups)
                zero_variance_groups = float(self.state.discarded_zero_variance_groups)
                self.state.accepted_trainable_groups += len(batch)
                generated_groups_cum = (
                    float(self.state.accepted_trainable_groups)
                    + stale_groups
                    + zero_variance_groups
                )
                metrics = {
                    "discarded/cum/stale_groups": stale_groups,
                    "discarded/cum/zero_variance_groups": zero_variance_groups,
                    "discarded/step/stale_groups": float(discarded),
                    "discarded/rate/stale_groups": stale_groups
                    / max(generated_groups_cum, 1.0),
                    "discarded/rate/zero_variance_groups": zero_variance_groups
                    / max(generated_groups_cum, 1.0),
                    "time/step_wall_s": step_seconds,
                    "time/step_collect_batch_s": trainer_idle_s,
                    "time/step_trainer_idle_s": trainer_idle_s,
                }
                metrics.setdefault("time/step_backend_train_s", train_call_elapsed)
                if actor_wall_s > 0:
                    metrics["time/step_rollout_s"] = actor_wall_s
                if actor_idle_s > 0:
                    metrics["time/step_rollout_idle_s"] = actor_idle_s
                if queue_wait_s > 0 and actor_wall_s > 0:
                    metrics["queue/put_wait_frac"] = queue_wait_s / actor_wall_s
                metrics.update(result.metrics)
                attachment_metrics, attachment_owns_vllm_metrics = (
                    self._collect_attachment_train_step_metrics()
                )
                metrics.update(attachment_metrics)
                vllm_metrics_collector = getattr(
                    self.backend, "collect_train_step_vllm_metrics", None
                )
                if (
                    callable(vllm_metrics_collector)
                    and not attachment_owns_vllm_metrics
                    and self.model._serving_capabilities is not None
                    and self.model._serving_capabilities.fast_metrics
                ):
                    maybe_metrics = vllm_metrics_collector(self.model)
                    if inspect.isawaitable(maybe_metrics):
                        metrics.update(await maybe_metrics)
                metrics.update(
                    self._score_metrics(
                        training_policy_step,
                        batch,
                        step_seconds=step_seconds,
                        result_metrics=metrics,
                    )
                )
                metrics.update(await self._queue_freshness_metrics(current_step))
                metrics.update(self._pipeline_settings_metrics())

                await self._emit_packed_group_observations(
                    metrics, batch=batch, step=current_step
                )
                await self._emit_pipeline_metrics(metrics, step=current_step)
                await self.model.log(
                    batch,
                    split="train",
                    step=current_step,
                    metrics=metrics,
                )
                await self._log_zero_variance_groups(current_step)

                if self.eval_fn is not None and should_eval_step:
                    await self._schedule_eval_step(current_step)

                self._persist_state(current_step)
            finally:
                self._status.note_training_end()

            async with self.state.policy_updated:
                self.state.policy_updated.notify_all()

            if saw_sentinel:
                stop_after_batch = True
            if stop_after_batch:
                break

        self.state.done = True
        self._persist_state(current_step)
        async with self.state.policy_updated:
            self.state.policy_updated.notify_all()

    async def _collect_batch(
        self, current_step: int
    ) -> tuple[list[TrajectoryGroup], int, bool]:
        assert self._output_queue is not None
        batch: list[TrajectoryGroup] = []
        discarded = 0
        saw_sentinel = False

        while not saw_sentinel and len(batch) < self.max_batch_size:
            wait = len(batch) < self.min_batch_size
            count = (self.min_batch_size if wait else self.max_batch_size) - len(batch)
            if isinstance(self._output_queue, DistributedTrajectoryQueue):
                items, saw_sentinel = await self._output_queue.get_many(
                    count, wait=wait
                )
                if not items:
                    break
            elif wait:
                item = await self._output_queue.get()
                if item is None:
                    saw_sentinel = True
                    break
                items = [item]
            else:
                try:
                    item = self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    saw_sentinel = True
                    break
                items = [item]
            for item in items:
                self._status.note_group_dequeued()
                self._check_all_failed(item)
                if self._is_group_stale(item, current_step):
                    discarded += 1
                    continue
                if self._group_zero_variance(item):
                    if self._record_zero_variance(item):
                        return [], discarded, saw_sentinel
                    continue
                batch.append(item)

        return batch, discarded, saw_sentinel

    def _check_all_failed(self, group: TrajectoryGroup) -> None:
        """Raise if all rollouts in a group failed with exceptions."""
        if not group.trajectories and group.exceptions:
            first_exc = group.exceptions[0]
            raise RuntimeError(
                f"All {len(group.exceptions)} rollouts in group failed. "
                f"First exception ({first_exc.type}): {first_exc.message}"
            )

    async def _eval_stage(self) -> None:
        if self.eval_fn is None or self._eval_queue is None:
            return

        pending_eval: asyncio.Task[None] | None = None
        while not self.state.done or not self._eval_queue.empty():
            try:
                step = await asyncio.wait_for(self._eval_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if pending_eval is not None and not pending_eval.done():
                try:
                    await pending_eval
                except Exception as exc:
                    print(f"Warning: previous eval failed: {exc}")

            pending_eval = asyncio.create_task(self._run_eval(step))

        if pending_eval is not None and not pending_eval.done():
            try:
                await pending_eval
            except Exception as exc:
                print(f"Warning: final eval failed: {exc}")

    async def _status_loop(self) -> None:
        sleep_seconds = min(1.0, max(0.2, self._status_log_interval_seconds / 10))
        while not self.state.done:
            self._status.log_if_due()
            await asyncio.sleep(sleep_seconds)

    async def _run_eval(self, step: int) -> None:
        assert self.eval_fn is not None
        self._status.note_val_started(step)
        reward: float | None = None
        eval_elapsed = 0.0
        eval_completed = False
        try:
            token = self.model.activate_metrics_context("eval")
            eval_started = time.monotonic()
            try:
                async with self._exact_adapter_lease(step):
                    result = await self.eval_fn(self.model, step, self.config)
            finally:
                token.var.reset(token)
                eval_elapsed = time.monotonic() - eval_started
            splits = result if _is_eval_mapping(result) else {"val": result}

            logged_eval_timing = False
            for split_name, items in splits.items():
                groups, trajectories = self._normalize_eval_items(items)
                self._validate_eval_policy_spans(step, trajectories)
                if split_name == "val":
                    if trajectories:
                        reward = sum(t.reward for t in trajectories) / len(trajectories)
                    else:
                        reward = None
                if groups:
                    metrics = (
                        {"time/step_eval_s": eval_elapsed}
                        if not logged_eval_timing
                        else None
                    )
                    await self.model.log(
                        groups,
                        split=split_name,
                        step=step,
                        metrics=metrics,
                    )
                    logged_eval_timing = True
            if not logged_eval_timing and eval_elapsed > 0:
                await self.model.log(
                    trajectories=None,
                    split="val",
                    step=step,
                    metrics={"time/step_eval_s": eval_elapsed},
                )
            await self._log_checkpoint_eval_completed(step)
            eval_completed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Eval failed at step {step}: {exc}")
        finally:
            await self._release_scheduled_eval_lease(step)
            if eval_completed:
                self.state.completed_eval_steps.add(step)
                self._persist_state(self.state.next_training_step)
            self._status.note_val_finished(step, reward)

    @staticmethod
    def _normalize_eval_items(
        items: Sequence[art.Trajectory | art.TrajectoryGroup],
    ) -> tuple[list[TrajectoryGroup], list[art.Trajectory]]:
        if not items:
            return [], []
        groups: list[TrajectoryGroup] = []
        loose: list[art.Trajectory] = []
        for item in items:
            if isinstance(item, TrajectoryGroup):
                groups.append(item)
            else:
                loose.append(item)
        if loose:
            groups.append(TrajectoryGroup(loose))
        trajectories: list[art.Trajectory] = []
        for group in groups:
            trajectories.extend(group.trajectories)
        return groups, trajectories

    @classmethod
    def _validate_eval_policy_spans(
        cls,
        step: int,
        trajectories: Iterable[art.Trajectory],
    ) -> None:
        for trajectory in trajectories:
            for item in cls._trajectory_messages_and_choices(trajectory):
                extra = getattr(item, "model_extra", None)
                if not isinstance(extra, Mapping) or "policy_token_spans" not in extra:
                    continue
                spans = extra["policy_token_spans"]
                if not isinstance(spans, list):
                    raise RuntimeError("Eval policy_token_spans must be a list")
                for span in spans:
                    if not isinstance(span, Mapping) or "policy_version" not in span:
                        raise RuntimeError("Eval policy token span is malformed")
                    policy_version = int(span["policy_version"])
                    if policy_version != step:
                        raise RuntimeError(
                            f"Eval at step {step} returned policy-{policy_version} tokens"
                        )

    def _apply_policy_versions(
        self,
        group: TrajectoryGroup,
        *,
        initial_version: int,
        final_version: int,
    ) -> None:
        for trajectory in group.trajectories:
            if trajectory.initial_policy_version is None:
                trajectory.initial_policy_version = initial_version
            if trajectory.final_policy_version is None:
                trajectory.final_policy_version = final_version

    def _scenario_metadata(
        self, scenario: ScenarioT
    ) -> dict[str, float | int | str | bool | None]:
        metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
        if metadata is None or not isinstance(metadata, dict):
            return {}

        result: dict[str, float | int | str | bool | None] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if not self._is_scalar_metadata(value):
                continue
            if key == "scenario_id":
                result["scenario_id"] = value
                continue
            result[f"scenario_{key}"] = value
        return result

    @staticmethod
    def _scenario_error_context(scenario: ScenarioT) -> str:
        metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
        if metadata is None or not isinstance(metadata, dict):
            return ""
        fields = (
            f"{key}={metadata[key]!r}"
            for key in ("scenario_id", "epoch", "scenario_index")
            if key in metadata
        )
        context = " ".join(fields)
        return f" [{context}]" if context else ""

    def _is_group_stale(self, group: TrajectoryGroup, current_step: int) -> bool:
        if self.max_steps_off_policy is not None:
            group_version = self._group_initial_version(group)
            if (
                group_version is not None
                and group_version < current_step - self.max_steps_off_policy
            ):
                return True
        if self.limit_mean_steps_off_policy is None:
            return False
        mean_steps = self._group_mean_steps_off_policy(current_step, group)
        return mean_steps is not None and mean_steps > self.limit_mean_steps_off_policy

    def _record_zero_variance(self, group: TrajectoryGroup) -> bool:
        self._discard_queue.append(group)
        self.state.discarded_zero_variance_groups += 1
        self._status.note_zero_variance_discarded(1)
        if len(self._discard_queue) >= self._discard_queue_limit:
            self._trigger_collapse()
            return True
        return False

    def _trigger_collapse(self) -> None:
        if self._collapse_triggered:
            return
        self._collapse_triggered = True
        self.state.done = True
        print(
            "\n"
            "========================================\n"
            "MODEL COLLAPSE DETECTED - Training stopped\n"
            "========================================\n"
            "\n"
            f"Too many trajectory groups ({self._discard_queue_limit}) had zero reward variance,\n"
            "indicating the model may have collapsed to a degenerate policy.\n"
            "\n"
            "To improve training dynamics:\n"
            "  - Lower the learning rate to reduce instability\n"
            "  - Ensure your reward function provides meaningful variance\n"
            "  - Check that prompts are diverse enough to elicit different responses\n"
            "  - Consider using a smaller batch size for more frequent updates\n"
            "\n"
            "To disable this failsafe:\n"
            "  - Increase `discard_queue_multiplier` (currently triggers after\n"
            f"    {self.discard_queue_multiplier} * min_batch_size = {self._discard_queue_limit} zero-variance groups)\n"
            "\n"
        )

    async def _log_zero_variance_groups(self, step: int) -> None:
        if not self._discard_queue:
            return
        discarded = list(self._discard_queue[:50])
        await self.model.log(discarded, split="discarded", step=step)
        self._discard_queue.clear()

    @staticmethod
    def _group_zero_variance(group: TrajectoryGroup) -> bool:
        rewards = [t.reward for t in group.trajectories]
        if len(rewards) <= 1:
            return True
        first = rewards[0]
        return all(abs(r - first) <= 1e-12 for r in rewards[1:])

    def _group_initial_version(self, group: TrajectoryGroup) -> int | None:
        versions = [
            trajectory.initial_policy_version
            for trajectory in group.trajectories
            if trajectory.initial_policy_version is not None
        ]
        if not versions:
            return None
        return min(versions)

    def _average_steps_off_policy(
        self, current_step: int, batch: list[TrajectoryGroup]
    ) -> float:
        steps: list[float] = []
        for group in batch:
            mean_steps = self._group_mean_steps_off_policy(current_step, group)
            if mean_steps is None:
                continue
            steps.append(mean_steps)
        if not steps:
            return 0.0
        return sum(steps) / len(steps)

    def _freshness_queue_window(self) -> int:
        if self.max_steps_off_policy is not None:
            return self.max_steps_off_policy
        if self.limit_mean_steps_off_policy is not None:
            return math.ceil(self.limit_mean_steps_off_policy)
        return 1

    async def _queue_freshness_metrics(self, current_step: int) -> dict[str, float]:
        if self._output_queue is None:
            return {}
        limit_raw = (
            self.limit_mean_steps_off_policy
            if self.limit_mean_steps_off_policy is not None
            else self.max_steps_off_policy
        )
        if limit_raw is None:
            limit_raw = 1.0
        limit = max(float(limit_raw), 1e-9)
        ages: list[float] = []
        capacity_metrics: dict[str, float] = {}
        if isinstance(self._output_queue, DistributedTrajectoryQueue):
            snapshot = await self._output_queue.snapshot()
            for item in snapshot.items:
                descriptor = item.ref.descriptor
                if self.limit_mean_steps_off_policy is not None:
                    if descriptor.policy_token_counts:
                        weight = sum(descriptor.policy_token_counts.values())
                        age = (
                            sum(
                                (current_step - version) * count
                                for version, count in descriptor.policy_token_counts.items()
                            )
                            / weight
                        )
                    else:
                        versions = descriptor.initial_policy_versions or (
                            item.annotations.initial_policy_version,
                        )
                        weights = descriptor.completion_tokens
                        if len(weights) != len(versions) or sum(weights) <= 0:
                            weights = (1.0,) * len(versions)
                        age = sum(
                            (current_step - version) * weight
                            for version, weight in zip(versions, weights, strict=True)
                        ) / sum(weights)
                else:
                    initial = min(
                        descriptor.initial_policy_versions
                        or (item.annotations.initial_policy_version,)
                    )
                    age = float(current_step - initial)
                ages.append(float(age))
            ready = float(len(snapshot.items))
            maxsize = float(snapshot.max_ready_groups)
            put_waiting = float(self._output_queue.put_waiters)
            capacity_metrics = {
                "queue/data_plane_records": float(snapshot.used_records),
                "queue/data_plane_bytes": float(snapshot.used_bytes),
                "queue/data_plane_record_occupancy": snapshot.used_records
                / snapshot.capacity_records,
                "queue/data_plane_byte_occupancy": snapshot.used_bytes
                / snapshot.capacity_bytes,
                "queue/leased_groups": float(snapshot.leased_groups),
            }
        else:
            output_queue = cast(Any, self._output_queue)
            queued = [
                group
                for group in list(output_queue._queue)
                if isinstance(group, TrajectoryGroup)
            ]
            for group in queued:
                if self.limit_mean_steps_off_policy is not None:
                    age = self._group_mean_steps_off_policy(current_step, group)
                else:
                    initial = self._group_initial_version(group)
                    age = None if initial is None else float(current_step - initial)
                if age is not None:
                    ages.append(float(age))
            ready = float(len(queued))
            maxsize = float(self._output_queue.maxsize)
            put_waiting = 0.0
        stale = sum(1 for age in ages if age > limit)
        return {
            "queue/ready_groups_est": ready,
            "queue/completed_backlog_groups": ready,
            "queue/put_waiting_groups": put_waiting,
            "queue/groups_depth": ready,
            "queue/groups_depth_max": maxsize,
            "queue/occupancy": ready / max(maxsize, 1.0),
            "queue/predicted_policy_age_mean_steps": sum(ages) / len(ages)
            if ages
            else 0.0,
            "queue/predicted_policy_age_p95_steps": _weighted_percentile(
                [(age, 1.0) for age in ages], 0.95
            )
            if ages
            else 0.0,
            "queue/freshness_pressure": (sum(ages) / len(ages) / limit)
            if ages
            else 0.0,
            "queue/predicted_stale_fraction": stale / len(ages) if ages else 0.0,
            **capacity_metrics,
        }

    def _pipeline_settings_metrics(self) -> dict[str, float]:
        if self.autotune.mode == "off":
            return {}
        return {
            "pipeline_settings/num_rollout_workers": float(self.num_rollout_workers),
            "pipeline_settings/min_batch_size": float(self.min_batch_size),
            "pipeline_settings/max_batch_size": float(self.max_batch_size),
            "pipeline_settings/target_groups_per_step": float(
                self.target_groups_per_step
            ),
            "pipeline_settings/queue_maxsize": float(self.queue_maxsize or 0),
        }

    def _retention_window_steps(self) -> int:
        if self.max_steps_off_policy is not None:
            return self.max_steps_off_policy
        if self.limit_mean_steps_off_policy is not None:
            return math.ceil(self.limit_mean_steps_off_policy)
        return 0

    def _group_mean_steps_off_policy(
        self, current_step: int, group: TrajectoryGroup
    ) -> float | None:
        weighted_age_sum = 0.0
        weight_sum = 0.0
        for trajectory in group.trajectories:
            stats = self._trajectory_policy_age_stats(current_step, trajectory)
            if stats is None:
                continue
            age_sum, weight, _age_exp_sum = stats
            weighted_age_sum += age_sum
            weight_sum += weight
        if weight_sum <= 0:
            return None
        return weighted_age_sum / weight_sum

    def _score_metrics(
        self,
        current_step: int,
        batch: list[TrajectoryGroup],
        *,
        step_seconds: float,
        result_metrics: dict[str, float],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        accepted_groups = float(len(batch))
        metrics["sample_efficiency/accepted_groups_per_step"] = accepted_groups
        rollouts_per_group = self._batch_rollouts_per_group(batch)
        batch_factor = self._batch_factor(
            accepted_groups=accepted_groups,
            rollouts_per_group=rollouts_per_group,
        )
        metrics["sample_efficiency/batch_factor"] = batch_factor

        age_metrics = self._batch_policy_age_metrics(current_step, batch)
        age_exp_moment = age_metrics.pop("_policy_age_exp_tau8", None)
        metrics.update(age_metrics)
        mean_age = age_metrics.get("offpolicy/token_weighted_policy_age_steps")
        if mean_age is None or age_exp_moment is None:
            return metrics
        freshness = 1.0 / max(float(age_exp_moment), 1e-12)
        metrics["sample_efficiency/freshness_discount"] = freshness

        assistant_tokens = result_metrics.get(
            "data/step_trainable_assistant_tokens",
        )
        if assistant_tokens is not None and step_seconds > 0:
            accepted_tok_per_s = float(assistant_tokens) / step_seconds
            metrics["throughput/accepted_train_tok_per_s"] = accepted_tok_per_s
            metrics["objective/score"] = accepted_tok_per_s * freshness * batch_factor
        return metrics

    def _batch_rollouts_per_group(self, batch: list[TrajectoryGroup]) -> float | None:
        group_sizes = [len(group.trajectories) for group in batch]
        if not group_sizes:
            return None
        return sum(group_sizes) / len(group_sizes)

    def _reference_rollouts_per_group(
        self, rollouts_per_group: float | None
    ) -> float | None:
        if self.score_reference_rollouts_per_group is not None:
            return self.score_reference_rollouts_per_group
        return rollouts_per_group

    def _batch_factor(
        self,
        *,
        accepted_groups: float,
        rollouts_per_group: float | None,
    ) -> float:
        if (
            self.score_reference_groups_per_step is None
            or accepted_groups <= 0
            or rollouts_per_group is None
            or rollouts_per_group <= 0
        ):
            return 1.0
        reference_rollouts = self._reference_rollouts_per_group(rollouts_per_group)
        if reference_rollouts is None or reference_rollouts <= 0:
            return 1.0
        reference_scenario_data = (
            self.score_reference_groups_per_step
            + _SCORE_CRITICAL_ROLLOUT_BATCH_SIZE / reference_rollouts
        )
        current_scenario_data = (
            accepted_groups + _SCORE_CRITICAL_ROLLOUT_BATCH_SIZE / rollouts_per_group
        )
        return reference_scenario_data / current_scenario_data

    def _batch_policy_age_metrics(
        self, current_step: int, batch: list[TrajectoryGroup]
    ) -> dict[str, float]:
        weighted_ages: list[tuple[float, float]] = []
        unweighted_ages: list[float] = []
        age_sum = 0.0
        age_exp_sum = 0.0
        weight_sum = 0.0
        for group in batch:
            for trajectory in group.trajectories:
                stats = self._trajectory_policy_age_stats(current_step, trajectory)
                if stats is None:
                    continue
                trajectory_age_sum, weight, trajectory_age_exp_sum = stats
                if weight <= 0:
                    continue
                age = trajectory_age_sum / weight
                age_sum += trajectory_age_sum
                age_exp_sum += trajectory_age_exp_sum
                weight_sum += weight
                weighted_ages.append((age, weight))
                unweighted_ages.append(age)
        if weight_sum <= 0 or not weighted_ages:
            return {}
        return {
            "offpolicy/token_weighted_policy_age_steps": age_sum / weight_sum,
            "_policy_age_exp_tau8": age_exp_sum / weight_sum,
            "offpolicy/token_weighted_policy_age_p95_steps": _weighted_percentile(
                weighted_ages, 0.95
            ),
        }

    def _trajectory_policy_age_stats(
        self, current_step: int, trajectory: art.Trajectory
    ) -> tuple[float, float, float] | None:
        span_stats = self._trajectory_policy_span_age_stats(current_step, trajectory)
        if span_stats is not None:
            return span_stats
        if trajectory.initial_policy_version is None:
            return None
        weight = self._trajectory_completion_weight(trajectory)
        age = float(current_step - trajectory.initial_policy_version)
        return age * weight, weight, _policy_age_exp(age) * weight

    def _trajectory_policy_span_age_stats(
        self, current_step: int, trajectory: art.Trajectory
    ) -> tuple[float, float, float] | None:
        age_sum = 0.0
        age_exp_sum = 0.0
        weight_sum = 0.0
        for item in self._trajectory_messages_and_choices(trajectory):
            extra = getattr(item, "model_extra", None)
            if not isinstance(extra, Mapping):
                continue
            spans = extra.get("policy_token_spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, Mapping):
                    continue
                try:
                    policy_version = int(span["policy_version"])
                    weight = int(span["end_token"]) - int(span["start_token"])
                except (KeyError, TypeError, ValueError):
                    continue
                if weight <= 0:
                    continue
                age = float(current_step - policy_version)
                age_sum += age * weight
                age_exp_sum += _policy_age_exp(age) * weight
                weight_sum += float(weight)
        if weight_sum <= 0:
            return None
        return age_sum, weight_sum, age_exp_sum

    @staticmethod
    def _trajectory_messages_and_choices(trajectory: art.Trajectory) -> Iterable[Any]:
        yield from trajectory.messages_and_choices
        for history in trajectory.additional_histories:
            yield from history.messages_and_choices

    @staticmethod
    def _trajectory_completion_weight(trajectory: art.Trajectory) -> float:
        value = trajectory.metrics.get("completion_tokens")
        if not isinstance(value, bool) and isinstance(value, int | float) and value > 0:
            return float(value)
        raise RuntimeError(
            "Pipeline training requires a positive completion_tokens metric from "
            "serving response usage; received "
            f"{value!r}"
        )

    def _should_eval_step(self, step: int) -> bool:
        if self.eval_fn is None:
            return False
        if self.eval_every_n_steps <= 0:
            return False
        return (step - self.state.last_eval_step) >= self.eval_every_n_steps

    def _read_pipeline_state(self) -> dict[str, Any]:
        state = self.model.read_state() or {}
        return state.get(PIPELINE_STATE_KEY, {})

    def _persist_state(self, training_step: int) -> None:
        payload = {
            "scenario_offset": self.state.scenario_offset,
            "total_scenarios_consumed": self.state.total_scenarios_consumed,
            "training_step": training_step,
            "last_eval_step": self.state.last_eval_step,
            "completed_eval_steps": sorted(self.state.completed_eval_steps),
            "accepted_trainable_groups": self.state.accepted_trainable_groups,
            "discarded_stale_groups": self.state.discarded_stale_groups,
            "discarded_zero_variance_groups": (
                self.state.discarded_zero_variance_groups
            ),
        }
        if self._pipeline_tuner_profile is not None:
            payload["pipeline_tuner_profile"] = self._pipeline_tuner_profile
        self.model.merge_state({PIPELINE_STATE_KEY: payload})

    def _log_checkpoint_history(self, step: int, metrics: dict[str, float]) -> None:
        row: dict[str, int | float | str] = {
            (key if key.startswith("checkpoint/") else f"checkpoint/{key}"): value
            for key, value in metrics.items()
            if value == value
        }
        if not row:
            return
        row["training_step"] = step
        row["step"] = step
        row["recorded_at"] = datetime.now().isoformat()

        output_dir = self.model._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        with open(Path(output_dir) / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    async def _log_checkpoint_saved(self, result: Any) -> None:
        step = int(result.step)
        checkpoint_path = getattr(result, "checkpoint_path", None)
        path = (
            Path(checkpoint_path)
            if isinstance(checkpoint_path, str) and checkpoint_path
            else Path(self.model._get_output_dir()) / "checkpoints" / f"{step:04d}"
        )
        if not path.exists():
            return
        self._log_checkpoint_history(
            step,
            {
                CHECKPOINT_SAVED_METRIC: 1.0,
                CHECKPOINT_CREATED_AT_METRIC: path.stat().st_ctime,
            },
        )

    async def _log_checkpoint_eval_completed(self, step: int) -> None:
        self._log_checkpoint_history(
            step,
            {CHECKPOINT_EVAL_COMPLETED_METRIC: 1.0},
        )

    def _checkpoint_metrics_by_step(self) -> dict[int, dict[str, float]]:
        history_path = Path(self.model._get_output_dir()) / "history.jsonl"
        if not history_path.exists():
            return {}
        sums: dict[int, dict[str, float]] = {}
        counts: dict[int, dict[str, int]] = {}
        with history_path.open("r", encoding="utf-8") as history_file:
            for line in history_file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                if not isinstance(step, int):
                    continue
                for key, value in row.items():
                    if key in {"step", "recorded_at"}:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    step_sums = sums.setdefault(step, {})
                    step_counts = counts.setdefault(step, {})
                    step_sums[key] = step_sums.get(key, 0.0) + float(value)
                    step_counts[key] = step_counts.get(key, 0) + 1
        return {
            step: {
                key: value / counts[step][key]
                for key, value in step_sums.items()
                if counts[step][key] > 0
            }
            for step, step_sums in sums.items()
        }

    def _checkpoint_infos(self) -> list[CheckpointInfo]:
        checkpoint_dir = Path(self.model._get_output_dir()) / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        metrics_by_step = self._checkpoint_metrics_by_step()
        checkpoints: list[CheckpointInfo] = []
        for path in checkpoint_dir.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            step = int(path.name)
            stat = path.stat()
            metrics = metrics_by_step.get(step, {})
            created_at_unix = metrics.get(CHECKPOINT_CREATED_AT_METRIC)
            created_at = (
                datetime.fromtimestamp(created_at_unix, timezone.utc)
                if created_at_unix is not None
                else datetime.fromtimestamp(stat.st_ctime, timezone.utc)
            )
            checkpoints.append(
                CheckpointInfo(
                    step=step,
                    path=str(path),
                    created_at=created_at,
                    is_eval_step=(
                        step in self.state.completed_eval_steps
                        or metrics.get(CHECKPOINT_EVAL_COMPLETED_METRIC, 0.0) > 0.0
                        or any(key.startswith(("val/", "test/")) for key in metrics)
                    ),
                    metrics=metrics,
                )
            )
        return sorted(checkpoints, key=lambda checkpoint: checkpoint.step)

    def _protected_checkpoint_steps(self, current_step: int) -> set[int]:
        protected_steps = (
            {current_step}
            | set(self._checkpoint_lease_counts)
            | set(self._scheduled_eval_steps)
        )
        if self.kl_penalty_coef > 0.0:
            if self.kl_penalty_step_lag is None:
                protected_steps.add(0)
            else:
                kl_penalty_reference_step = self._kl_penalty_reference_step(
                    current_step
                )
                protected_steps.update(
                    range(kl_penalty_reference_step, current_step + 1)
                )
        return protected_steps

    async def _run_checkpoint_retention(self, current_step: int) -> None:
        strategy = self.checkpoint_retention_strategy
        if strategy is None:
            return
        if current_step % self.checkpoint_retention_interval != 0:
            return
        all_checkpoints = self._checkpoint_infos()
        if not all_checkpoints:
            return
        protected_steps = self._protected_checkpoint_steps(current_step)
        eligible = [
            checkpoint
            for checkpoint in all_checkpoints
            if checkpoint.step not in protected_steps
        ]
        if not eligible:
            return
        context = CheckpointRetentionContext(
            current_step=current_step,
            checkpoints=eligible,
        )
        eligible_steps = {checkpoint.step for checkpoint in eligible}
        keep_eligible_steps = set(strategy(context)) & eligible_steps
        delete_steps = eligible_steps - keep_eligible_steps
        if not delete_steps:
            return
        keep_steps = {checkpoint.step for checkpoint in all_checkpoints} - delete_steps
        await self.backend._delete_checkpoint_files(self.model, sorted(keep_steps))

    @staticmethod
    def _is_scalar_metadata(value: object) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    async def _put_output_group(
        self,
        group: TrajectoryGroup | TrajectoryGroupRef,
        *,
        metadata: dict[str, float | int | str | bool | None],
        initial_policy_version: int,
        final_policy_version: int,
        rollout_wall_s: float,
        actor_idle_s: float,
    ) -> float:
        assert self._output_queue is not None
        queue_wait_started = time.monotonic()
        if isinstance(self._output_queue, DistributedTrajectoryQueue):
            if not isinstance(group, TrajectoryGroupRef):
                raise RuntimeError("distributed result queue requires a stored group")
            accepted, wait_s = await self._output_queue.put(
                group,
                metadata=metadata,
                initial_policy_version=initial_policy_version,
                final_policy_version=final_policy_version,
                rollout_wall_s=rollout_wall_s,
                actor_idle_s=actor_idle_s,
            )
            if accepted:
                self._status.note_group_enqueued()
            return wait_s
        if not isinstance(group, TrajectoryGroup):
            raise RuntimeError("local result queue requires a trajectory group")
        while not self.state.done:
            try:
                await asyncio.wait_for(self._output_queue.put(group), timeout=1.0)
                self._status.note_group_enqueued()
                return time.monotonic() - queue_wait_started
            except asyncio.TimeoutError:
                continue
        return time.monotonic() - queue_wait_started

    def _consume_batch_rollout_timings(
        self, batch: list[TrajectoryGroup]
    ) -> tuple[float, float, float]:
        rollout_wall_s = 0.0
        actor_idle_s = 0.0
        queue_wait_s = 0.0
        for group in batch:
            rollout_wall_s += self._pop_float_metadata(group, _ROLLOUT_WALL_TIME_KEY)
            actor_idle_s += self._pop_float_metadata(group, _ACTOR_IDLE_TIME_KEY)
            queue_wait_s += self._pop_float_metadata(group, _QUEUE_WAIT_TIME_KEY)
        return rollout_wall_s, actor_idle_s, queue_wait_s

    @staticmethod
    def _pop_float_metadata(group: TrajectoryGroup, key: str) -> float:
        value = group.metadata.pop(key, 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
