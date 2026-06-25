from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from typing import Any, AsyncIterator, Generic, Iterable, TypeVar, cast

T = TypeVar("T")

import art
from art import TrajectoryGroup

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


def _to_async_iterator(iterable: Iterable[T] | AsyncIterator[T]) -> AsyncIterator[T]:
    """Convert a sync Iterable to an AsyncIterator, or pass through if already async."""
    if isinstance(iterable, AsyncIterator):
        return iterable

    async def _iter():
        for item in iterable:
            yield item

    return _iter()


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
        # Pipeline settings
        num_rollout_workers: int = 16,
        min_batch_size: int = 4,
        max_batch_size: int | None = None,
        max_steps_off_policy: int = 4,
        queue_maxsize: int | None = None,
        # Training
        learning_rate: float = 1e-5,
        loss_fn: str = "cispo",
        loss_fn_config: dict | None = None,
        normalize_advantages: bool = True,
        adam_params: object | None = None,
        kl_penalty_coef: float = 0.0,
        kl_penalty_step_lag: int | None = None,
        max_steps: int | None = None,
        # Discard handling
        discard_queue_multiplier: int = 100,
        # Status output
        log_interval_seconds: float = 60.0,
        status_ewa_alpha: float = 0.2,
        total_scenarios: int | None = None,
        # Eval/Checkpointing
        eval_every_n_steps: int = 20,
        eval_at_start: bool = True,
        save_checkpoint: bool = True,
        checkpoint_retention_strategy: CheckpointRetentionStrategy | None = None,
        checkpoint_retention_interval: int = 1,
        # Resumption
        resume: bool = True,
    ) -> None:
        if num_rollout_workers <= 0:
            raise ValueError("num_rollout_workers must be > 0")
        if min_batch_size <= 0:
            raise ValueError("min_batch_size must be > 0")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if max_batch_size is not None and max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if max_steps_off_policy < 0:
            raise ValueError("max_steps_off_policy must be >= 0")
        if queue_maxsize is not None and queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be > 0")
        if eval_every_n_steps < 0:
            raise ValueError("eval_every_n_steps must be >= 0")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be > 0")
        if discard_queue_multiplier <= 0:
            raise ValueError("discard_queue_multiplier must be > 0")
        if checkpoint_retention_interval <= 0:
            raise ValueError("checkpoint_retention_interval must be > 0")
        if kl_penalty_step_lag is not None and kl_penalty_step_lag < 1:
            raise ValueError("kl_penalty_step_lag must be >= 1")
        self.model = model
        self.backend = backend
        self.rollout_fn = rollout_fn
        self.config = config
        self.eval_fn = eval_fn
        self.num_rollout_workers = num_rollout_workers
        self.min_batch_size = min_batch_size
        self.max_batch_size = (
            max_batch_size if max_batch_size is not None else 10 * min_batch_size
        )
        self.max_steps_off_policy = max_steps_off_policy
        self.queue_maxsize = queue_maxsize
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.loss_fn_config = loss_fn_config
        self.normalize_advantages = normalize_advantages
        self.adam_params = adam_params
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_penalty_step_lag = kl_penalty_step_lag
        self.max_steps = max_steps
        self._status_log_interval_seconds = log_interval_seconds
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_at_start = eval_at_start
        self.save_checkpoint = save_checkpoint
        self.checkpoint_retention_strategy = checkpoint_retention_strategy
        self.checkpoint_retention_interval = checkpoint_retention_interval
        self.resume = resume
        self.discard_queue_multiplier = discard_queue_multiplier
        self._discard_queue: list[TrajectoryGroup] = []
        self._discard_queue_limit = discard_queue_multiplier * min_batch_size
        self._collapse_triggered = False
        self._checkpoint_lease_counts: Counter[int] = Counter()
        self._scheduled_eval_steps: set[int] = set()
        self._scheduled_eval_leases: dict[int, AsyncExitStack] = {}

        self.state = PipelineState()
        self._scenario_lock = asyncio.Lock()
        self._scenario_iter: AsyncIterator[ScenarioT] | None = _to_async_iterator(
            scenarios
        )
        self._output_queue: asyncio.Queue[TrajectoryGroup | None] | None = None
        self._eval_queue: asyncio.Queue[int] | None = None
        self._status = StatusReporter(
            get_scenario_offset=lambda: self.state.scenario_offset,
            log_interval_seconds=log_interval_seconds,
            status_ewa_alpha=status_ewa_alpha,
            total_scenarios=total_scenarios,
            num_workers=num_rollout_workers,
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
        self.state.last_eval_step = last_eval_step
        self.state.completed_eval_steps = {
            int(step) for step in pipeline_state.get("completed_eval_steps", []) or []
        }

        if scenario_offset > 0 and self._scenario_iter is not None:
            skipped = await self._skip_scenarios(self._scenario_iter, scenario_offset)
            self.state.scenario_offset = skipped
            self.state.total_scenarios_consumed = skipped

        queue_maxsize = (
            self.queue_maxsize
            if self.queue_maxsize is not None
            else max(1, self.max_steps_off_policy * self.max_batch_size)
        )
        self._output_queue = asyncio.Queue(maxsize=queue_maxsize)
        self._eval_queue = asyncio.Queue()

        if self.eval_fn is not None and self.eval_at_start:
            await self._schedule_eval_step(start_step)
            self._persist_state(start_step)

        self._status.start(initial_step=start_step)
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
            for exc in eg.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    print(f"Pipeline stage failed: {exc}")
            raise
        finally:
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
            self._status.flush()
            self._status.close()
            await self._release_all_scheduled_eval_leases()

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
            try:
                self._output_queue.put_nowait(None)
            except asyncio.QueueFull:
                loop.create_task(self._output_queue.put(None))

    def _validate_backend_support(self) -> None:
        from art.dev.validate import is_dedicated_mode
        from art.local.backend import LocalBackend

        if not isinstance(self.backend, LocalBackend):
            return

        model_config = self.model._internal_config or art.dev.InternalModelConfig()
        if not is_dedicated_mode(model_config):
            raise ValueError(
                "PipelineTrainer only supports LocalBackend in dedicated mode. "
                "Shared LocalBackend pauses inference during training and is not "
                "a supported async PipelineTrainer path. Set both "
                "trainer_gpu_ids and inference_gpu_ids on the TrainableModel "
                "_internal_config to use LocalBackend with PipelineTrainer."
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
                return None
            self.state.scenario_offset += 1
            self.state.total_scenarios_consumed += 1
            return scenario

    async def _wait_for_policy(self) -> None:
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
        min_step = max(0, current_step - self.max_steps_off_policy)
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
            scenario = await self._get_next_scenario()
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
                        group = await self.rollout_fn(self.model, scenario, self.config)
                finally:
                    token.var.reset(token)
                rollout_wall_s = time.monotonic() - rollout_started
                if not isinstance(group, TrajectoryGroup):
                    errored = True
                    continue
                self._apply_scenario_metadata(group, scenario)
                self._apply_policy_versions(
                    group,
                    initial_version=initial_version,
                    final_version=self.state.policy_version,
                )
                if self.state.done:
                    break
                queue_wait_s = await self._put_output_group(group)
                group.metadata[_ROLLOUT_WALL_TIME_KEY] = rollout_wall_s
                group.metadata[_ACTOR_IDLE_TIME_KEY] = actor_idle_s + queue_wait_s
            except asyncio.CancelledError:
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
        async with asyncio.TaskGroup() as tg:
            for i in range(self.num_rollout_workers):
                tg.create_task(self._rollout_worker(i))
        if not self.state.done and self._output_queue is not None:
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

            actor_wall_s, actor_idle_s = self._consume_batch_rollout_timings(batch)

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
                }
                if self.kl_penalty_coef > 0.0:
                    kl_penalty_reference_step = self._kl_penalty_reference_step(
                        current_step
                    )
                    train_kwargs["kl_penalty_coef"] = self.kl_penalty_coef
                    train_kwargs["kl_penalty_source"] = "sample"
                    train_kwargs["kl_penalty_reference_step"] = (
                        kl_penalty_reference_step
                    )
                result = await self.backend.train(
                    self.model,
                    batch,
                    **train_kwargs,
                )
            except Exception:
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

                steps_off_policy = self._average_steps_off_policy(current_step, batch)
                metrics = {
                    "discarded_stale_groups": float(self.state.discarded_stale_groups),
                    "steps_off_policy": steps_off_policy,
                    "time/step_wall_s": step_seconds,
                    "throughput/step_trainer_idle_s": trainer_idle_s,
                }
                metrics.setdefault("time/step_trainer_s", train_call_elapsed)
                if actor_wall_s > 0:
                    metrics["time/step_actor_s"] = actor_wall_s
                if actor_idle_s > 0:
                    metrics["throughput/step_actor_idle_s"] = actor_idle_s
                metrics.update(result.metrics)

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
        min_version = current_step - self.max_steps_off_policy

        while len(batch) < self.min_batch_size:
            item = await self._output_queue.get()
            if item is None:
                saw_sentinel = True
                break
            self._status.note_group_dequeued(item)
            self._check_all_failed(item)
            if self._is_group_stale(item, min_version):
                discarded += 1
                continue
            if self._group_zero_variance(item):
                if self._record_zero_variance(item):
                    return [], discarded, saw_sentinel
                continue
            batch.append(item)

        while not saw_sentinel and len(batch) < self.max_batch_size:
            try:
                item = self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                saw_sentinel = True
                break
            self._status.note_group_dequeued(item)
            self._check_all_failed(item)
            if self._is_group_stale(item, min_version):
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
                async with self._adapter_lease(step):
                    result = await self.eval_fn(self.model, step, self.config)
            finally:
                token.var.reset(token)
                eval_elapsed = time.monotonic() - eval_started
            splits: dict[str, list[art.Trajectory | art.TrajectoryGroup]]
            if isinstance(result, dict):
                splits = result
            else:
                splits = {"val": result}

            logged_eval_timing = False
            for split_name, items in splits.items():
                groups, trajectories = self._normalize_eval_items(items)
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
        items: list[art.Trajectory | art.TrajectoryGroup],
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

    def _apply_scenario_metadata(
        self, group: TrajectoryGroup, scenario: ScenarioT
    ) -> None:
        metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
        if metadata is None or not isinstance(metadata, dict):
            return

        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if not self._is_scalar_metadata(value):
                continue
            if key == "scenario_id":
                group.metadata["scenario_id"] = value
                continue
            group.metadata[f"scenario_{key}"] = value

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

    def _is_group_stale(self, group: TrajectoryGroup, min_version: int) -> bool:
        group_version = self._group_initial_version(group)
        if group_version is None:
            return False
        return group_version < min_version

    def _record_zero_variance(self, group: TrajectoryGroup) -> bool:
        self._discard_queue.append(group)
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
        steps: list[int] = []
        for group in batch:
            group_version = self._group_initial_version(group)
            if group_version is None:
                continue
            steps.append(current_step - group_version)
        if not steps:
            return 0.0
        return sum(steps) / len(steps)

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
        }
        self.model.merge_state({PIPELINE_STATE_KEY: payload})

    def _log_checkpoint_history(self, step: int, metrics: dict[str, float]) -> None:
        row = {
            (key if key.startswith("checkpoint/") else f"checkpoint/{key}"): value
            for key, value in metrics.items()
            if value == value
        }
        if not row:
            return
        row["training_step"] = step
        row["time/wall_clock_sec"] = time.time() - self.model._run_start_time
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

    async def _put_output_group(self, group: TrajectoryGroup) -> float:
        assert self._output_queue is not None
        queue_wait_started = time.monotonic()
        while not self.state.done:
            try:
                await asyncio.wait_for(self._output_queue.put(group), timeout=1.0)
                self._status.note_group_enqueued(group)
                return time.monotonic() - queue_wait_started
            except asyncio.TimeoutError:
                continue
        return time.monotonic() - queue_wait_started

    def _consume_batch_rollout_timings(
        self, batch: list[TrajectoryGroup]
    ) -> tuple[float, float]:
        rollout_wall_s = 0.0
        actor_idle_s = 0.0
        for group in batch:
            rollout_wall_s += self._pop_float_metadata(group, _ROLLOUT_WALL_TIME_KEY)
            actor_idle_s += self._pop_float_metadata(group, _ACTOR_IDLE_TIME_KEY)
        return rollout_wall_s, actor_idle_s

    @staticmethod
    def _pop_float_metadata(group: TrajectoryGroup, key: str) -> float:
        value = group.metadata.pop(key, 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
