from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any, AsyncIterator, Generic, Iterable, TypeVar, cast

T = TypeVar("T")

import art
from art import TrajectoryGroup

from .state import PipelineState
from .status import StatusReporter
from .types import ConfigT, EvalFn, RolloutFn, ScenarioT, SingleRolloutFn  # noqa: F401

PIPELINE_STATE_KEY = "_pipeline_trainer"


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
        max_steps: int | None = None,
        # Discard handling
        discard_queue_multiplier: int = 100,
        # Status output
        log_interval_seconds: float = 60.0,
        status_ewa_alpha: float = 0.2,
        total_scenarios: int | None = None,
        # Eval/Checkpointing
        eval_every_n_steps: int = 20,
        eval_step_0: bool = True,
        save_checkpoint: bool = True,
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
        self.max_steps = max_steps
        self._status_log_interval_seconds = log_interval_seconds
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_step_0 = eval_step_0
        self.save_checkpoint = save_checkpoint
        self.resume = resume
        self.discard_queue_multiplier = discard_queue_multiplier
        self._discard_queue: list[TrajectoryGroup] = []
        self._discard_queue_limit = discard_queue_multiplier * min_batch_size
        self._collapse_triggered = False

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

        if self.eval_fn is not None and self.eval_step_0 and start_step == 0:
            await self._eval_queue.put(start_step)
            self.state.last_eval_step = start_step
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

    async def _rollout_worker(self, worker_id: int) -> None:
        assert self._output_queue is not None
        while not self.state.done:
            scenario = await self._get_next_scenario()
            if scenario is None:
                break
            self._status.note_rollout_started()
            errored = False
            try:
                await self._wait_for_policy()
                if self.state.done:
                    break

                initial_version = self.state.policy_version

                group = await self.rollout_fn(self.model, scenario, self.config)
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
                await self._put_output_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errored = True
                print(f"Worker {worker_id}: rollout failed: {exc}")
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
            batch, discarded, saw_sentinel = await self._collect_batch(current_step)
            self.state.discarded_stale_samples += discarded
            if discarded:
                self._status.note_stale(discarded)
            if not batch:
                break

            expected_step = current_step + 1
            should_eval_step = self._should_eval_step(expected_step)
            should_checkpoint = self.save_checkpoint and should_eval_step

            async with self.state.policy_updated:
                self.state.next_training_step = expected_step
                self.state.policy_updated.notify_all()

            self._status.note_training_start(len(batch))
            train_call_start: float | None = None
            if os.getenv("ART_TRAIN_STEP_LOG"):
                print(f"[train] step {expected_step} starting (batch={len(batch)})")
                train_call_start = time.perf_counter()
            try:
                result = await self.backend.train(
                    self.model,
                    batch,
                    learning_rate=self.learning_rate,
                    loss_fn=self.loss_fn,
                    loss_fn_config=self.loss_fn_config,
                    normalize_advantages=self.normalize_advantages,
                    save_checkpoint=should_checkpoint,
                    adam_params=self.adam_params,
                )
            except Exception:
                self._status.note_training_end()
                raise
            finally:
                if train_call_start is not None:
                    train_call_elapsed = time.perf_counter() - train_call_start
                    print(
                        f"[train] step {expected_step} done in "
                        f"{train_call_elapsed:.1f}s"
                    )

            try:
                current_step = result.step
                self.state.policy_version = current_step
                self.state.next_training_step = current_step

                step_seconds = time.monotonic() - step_start
                self._status.note_training_batch(
                    batch, step=current_step, step_seconds=step_seconds
                )

                steps_off_policy = self._average_steps_off_policy(current_step, batch)
                metrics = {
                    "discarded_stale_samples": float(
                        self.state.discarded_stale_samples
                    ),
                    "steps_off_policy": steps_off_policy,
                    "num_groups": float(len(batch)),
                }
                metrics.update(result.metrics)

                await self.model.log(
                    batch,
                    split="train",
                    step=current_step,
                    metrics=metrics,
                )
                await self._log_discarded_groups(current_step)

                if self.eval_fn is not None and should_eval_step:
                    if self._eval_queue is not None:
                        await self._eval_queue.put(current_step)
                    self.state.last_eval_step = current_step

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

        while not saw_sentinel:
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
        try:
            result = await self.eval_fn(self.model, step, self.config)
            splits: dict[str, list[art.Trajectory]]
            if isinstance(result, dict):
                splits = result
            else:
                splits = {"val": result}

            val_trajectories = splits.get("val") or []
            if val_trajectories:
                reward = sum(t.reward for t in val_trajectories) / len(val_trajectories)
            else:
                reward = None

            for split_name, trajectories in splits.items():
                if trajectories:
                    await self.model.log(trajectories, split=split_name, step=step)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Eval failed at step {step}: {exc}")
        finally:
            self._status.note_val_finished(step, reward)

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
            group.metadata[f"scenario_{key}"] = value

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

    async def _log_discarded_groups(self, step: int) -> None:
        if not self._discard_queue:
            return
        discarded = list(self._discard_queue)
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
        }
        self.model.merge_state({PIPELINE_STATE_KEY: payload})

    @staticmethod
    def _is_scalar_metadata(value: object) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    async def _put_output_group(self, group: TrajectoryGroup) -> None:
        assert self._output_queue is not None
        while not self.state.done:
            try:
                await asyncio.wait_for(self._output_queue.put(group), timeout=1.0)
                self._status.note_group_enqueued(group)
                return
            except asyncio.TimeoutError:
                continue
