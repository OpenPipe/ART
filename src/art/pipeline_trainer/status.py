from __future__ import annotations

import math
import shutil
import sys
import time
from typing import Callable, cast

from tqdm import tqdm

from art import TrajectoryGroup


class StatusReporter:
    def __init__(
        self,
        *,
        get_scenario_offset: Callable[[], int],
        log_interval_seconds: float = 60.0,
        status_ewa_alpha: float = 0.2,
        total_scenarios: int | None = None,
        num_workers: int = 1,
    ) -> None:
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be > 0")
        if not 0 < status_ewa_alpha <= 1:
            raise ValueError("status_ewa_alpha must be in (0, 1]")
        if total_scenarios is not None and total_scenarios < 0:
            raise ValueError("total_scenarios must be >= 0")
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self._get_scenario_offset = get_scenario_offset
        self._log_interval_seconds = log_interval_seconds
        self._status_ewa_alpha = status_ewa_alpha
        self._total_scenarios = total_scenarios
        self._num_workers = num_workers

        self._current_step: int | None = None
        self._rolling_out = 0
        self._queued = 0
        self._training = 0
        self._trained = 0
        self._stale = 0
        self._zero_var = 0
        self._errored = 0

        self._train_reward_ewa: float | None = None
        self._seconds_ewa: float | None = None
        self._avg_std_dev_ewa: float | None = None

        self._last_val_step: int | None = None
        self._last_val_reward: float | None = None
        self._val_running_step: int | None = None

        self._tqdm: tqdm | None = None
        self._started = False
        self._last_log_time = 0.0
        self._last_refresh_time = 0.0
        self._refresh_interval_seconds = 0.25

    def start(self, *, initial_step: int | None = None) -> None:
        if self._started:
            return
        self._started = True
        if initial_step is not None:
            self._current_step = initial_step
        self._last_log_time = time.monotonic() - self._log_interval_seconds
        self._last_refresh_time = 0.0
        if sys.stdout.isatty():
            self._tqdm = tqdm(
                total=self._total_scenarios,
                bar_format="{desc}",
                dynamic_ncols=True,
                leave=False,
                file=sys.stdout,
            )
            self._refresh_status(force=True)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        self._started = False

    def flush(self) -> None:
        self.log_if_due(force=True)

    def set_step(self, step: int) -> None:
        self._current_step = step
        self._refresh_status()

    def log_if_due(self, *, force: bool = False) -> None:
        if not self._started:
            return
        now = time.monotonic()
        if not force and (now - self._last_log_time) < self._log_interval_seconds:
            return
        self._last_log_time = now
        self._write_log_line(self._format_full_log())

    def note_rollout_started(self) -> None:
        self._rolling_out += 1
        self._refresh_status()

    def note_rollout_finished(self, *, errored: bool) -> None:
        if self._rolling_out > 0:
            self._rolling_out -= 1
        if errored:
            self._errored += 1
        self._refresh_status()

    def note_group_enqueued(self, _group: TrajectoryGroup) -> None:
        self._queued += 1
        self._refresh_status()

    def note_group_dequeued(self, _group: TrajectoryGroup) -> None:
        if self._queued > 0:
            self._queued -= 1
        self._refresh_status()

    def note_stale(self, count: int) -> None:
        if count > 0:
            self._stale += count
            self._refresh_status()

    def note_zero_variance_discarded(self, count: int) -> None:
        if count > 0:
            self._zero_var += count
            self._refresh_status()

    def note_training_start(self, group_count: int) -> None:
        self._training = group_count
        self._refresh_status()

    def note_training_end(self) -> None:
        self._training = 0
        self._refresh_status()

    def note_training_batch(
        self, batch: list[TrajectoryGroup], *, step: int, step_seconds: float
    ) -> None:
        zero_variance_groups = self._count_zero_variance_groups(batch)
        trainable_groups = len(batch) - zero_variance_groups
        avg_std = self._compute_batch_avg_std_dev(batch)
        avg_reward = self._compute_batch_avg_reward(batch)

        self._current_step = step
        self._trained += trainable_groups
        self._zero_var += zero_variance_groups

        if avg_reward is not None:
            self._train_reward_ewa = self._update_ewa(
                self._train_reward_ewa, avg_reward
            )
        self._seconds_ewa = self._update_ewa(self._seconds_ewa, step_seconds)
        self._avg_std_dev_ewa = self._update_ewa(self._avg_std_dev_ewa, avg_std)
        self._refresh_status(force=True)

    def note_val_started(self, step: int) -> None:
        self._val_running_step = step
        self._refresh_status(force=True)

    def note_val_finished(self, step: int, reward: float | None) -> None:
        self._last_val_step = step
        self._last_val_reward = reward
        if self._val_running_step == step:
            self._val_running_step = None
        self._refresh_status(force=True)

    def _build_snapshot(self) -> dict[str, object]:
        remaining = None
        if self._total_scenarios is not None:
            remaining = max(self._total_scenarios - self._get_scenario_offset(), 0)
        return {
            "step": self._current_step,
            "remaining": remaining,
            "rolling": self._rolling_out,
            "workers": self._num_workers,
            "queued": self._queued,
            "training": self._training,
            "trained": self._trained,
            "zero_var": self._zero_var,
            "stale": self._stale,
            "errored": self._errored,
            "discarded": self._zero_var + self._stale + self._errored,
            "train_reward_ewa": self._train_reward_ewa,
            "train_seconds_ewa": self._seconds_ewa,
            "train_avg_std_ewa": self._avg_std_dev_ewa,
            "val_step": self._last_val_step,
            "val_reward": self._last_val_reward,
            "val_running": self._val_running_step,
        }

    def _format_condensed_line(self) -> str:
        snapshot = self._build_snapshot()
        trained = cast(int, snapshot["trained"])
        discarded = cast(int, snapshot["discarded"])
        remaining = snapshot["remaining"]

        scenarios_fields = [
            "scenarios",
            f"tr={trained}",
        ]
        if remaining is not None:
            scenarios_fields.append(f"r={self._fmt_int_compact(remaining)}")
        scenarios_fields.extend(
            [
                f"q={self._fmt_int_compact(snapshot['queued'])}",
                f"b={self._fmt_int_compact(snapshot['training'])}",
                f"d={discarded}",
            ]
        )

        train_fields = [
            "train",
            f"s={self._fmt_int_compact(snapshot['step'])}",
            f"r={self._fmt_float_compact(snapshot['train_reward_ewa'], 2)}",
            f"dt={self._fmt_float_compact(snapshot['train_seconds_ewa'], 1)}",
            f"sd={self._fmt_float_compact(snapshot['train_avg_std_ewa'], 2)}",
        ]

        val_run = "y" if snapshot["val_running"] is not None else "n"
        val_fields = [
            "val",
            f"r={self._fmt_float_compact(snapshot['val_reward'], 2)}",
            f"act={val_run}",
        ]

        def build_line() -> str:
            return " ".join(
                [
                    f"scenarios[{' '.join(scenarios_fields[1:])}]",
                    f"train[{' '.join(train_fields[1:])}]",
                    f"val[{' '.join(val_fields[1:])}]",
                ]
            )

        line = build_line()
        max_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        if len(line) <= max_width:
            return line

        train_fields = [field for field in train_fields if not field.startswith("sd=")]
        line = build_line()
        if len(line) <= max_width:
            return line

        val_fields = [field for field in val_fields if not field.startswith("act=")]
        line = build_line()
        if len(line) <= max_width:
            return line

        if remaining is not None:
            scenarios_fields = [
                field for field in scenarios_fields if not field.startswith("r=")
            ]
        return build_line()

    def _format_full_log(self) -> str:
        snapshot = self._build_snapshot()
        scenarios_fields = [
            "scenarios",
            f"trained={self._fmt_int(snapshot['trained'])}",
        ]
        if snapshot["remaining"] is not None:
            scenarios_fields.append(f"remaining={self._fmt_int(snapshot['remaining'])}")
        scenarios_fields.extend(
            [
                f"queued={self._fmt_int(snapshot['queued'])}",
                f"training={self._fmt_int(snapshot['training'])}",
                (
                    "discarded["
                    f"total={self._fmt_int(snapshot['discarded'])} "
                    f"0_var={self._fmt_int(snapshot['zero_var'])} "
                    f"stale={self._fmt_int(snapshot['stale'])} "
                    f"errored={self._fmt_int(snapshot['errored'])}"
                    "]"
                ),
                f"rollouts={snapshot['rolling']}/{snapshot['workers']}",
            ]
        )

        train_fields = [
            "train",
            f"step={self._fmt_int(snapshot['step'])}",
            f"reward={self._fmt_float(snapshot['train_reward_ewa'], 3)}",
            f"step_seconds={self._fmt_float(snapshot['train_seconds_ewa'], 2)}",
            f"avg_std={self._fmt_float(snapshot['train_avg_std_ewa'], 3)}",
        ]

        val_run = "yes" if snapshot["val_running"] is not None else "no"
        val_fields = [
            "val",
            f"reward={self._fmt_float(snapshot['val_reward'], 3)}",
            f"active={val_run}",
        ]
        if snapshot["val_step"] is not None:
            val_fields.append(f"step={snapshot['val_step']}")
        if snapshot["val_running"] is not None:
            val_fields.append(f"active_step={snapshot['val_running']}")

        return "[status] " + " ".join(
            [
                f"scenarios[{' '.join(scenarios_fields[1:])}]",
                f"train[{' '.join(train_fields[1:])}]",
                f"val[{' '.join(val_fields[1:])}]",
            ]
        )

    def _write_log_line(self, line: str) -> None:
        if self._tqdm is not None:
            self._tqdm.write(line)
        else:
            print(line)

    def _refresh_status(self, *, force: bool = False) -> None:
        if self._tqdm is None:
            return
        now = time.monotonic()
        if (
            not force
            and (now - self._last_refresh_time) < self._refresh_interval_seconds
        ):
            return
        self._tqdm.set_description_str(self._format_condensed_line())
        self._last_refresh_time = now

    def _count_zero_variance_groups(self, batch: list[TrajectoryGroup]) -> int:
        return sum(1 for group in batch if self._group_zero_variance(group))

    def _group_zero_variance(self, group: TrajectoryGroup) -> bool:
        rewards = [t.reward for t in group.trajectories]
        if len(rewards) <= 1:
            return True
        first = rewards[0]
        return all(abs(r - first) <= 1e-12 for r in rewards[1:])

    def _compute_group_std_dev(self, group: TrajectoryGroup) -> float:
        rewards = [t.reward for t in group.trajectories]
        if len(rewards) <= 1:
            return 0.0
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        return math.sqrt(variance)

    def _compute_batch_avg_std_dev(self, batch: list[TrajectoryGroup]) -> float:
        if not batch:
            return 0.0
        std_devs = [self._compute_group_std_dev(group) for group in batch]
        return sum(std_devs) / len(std_devs)

    def _compute_batch_avg_reward(self, batch: list[TrajectoryGroup]) -> float | None:
        rewards = [t.reward for group in batch for t in group.trajectories]
        if not rewards:
            return None
        return sum(rewards) / len(rewards)

    def _update_ewa(self, previous: float | None, new_value: float) -> float:
        if previous is None:
            return new_value
        alpha = self._status_ewa_alpha
        return alpha * new_value + (1 - alpha) * previous

    @staticmethod
    def _format_count(value: int) -> str:
        if value >= 1_000_000:
            return StatusReporter._format_scaled(value, 1_000_000, "m")
        if value >= 1_000:
            return StatusReporter._format_scaled(value, 1_000, "k")
        return str(value)

    @staticmethod
    def _format_scaled(value: int, scale: int, suffix: str) -> str:
        scaled = value / scale
        text = f"{scaled:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}{suffix}"

    @staticmethod
    def _fmt_int(value: object) -> str:
        if value is None:
            return "n/a"
        return str(value)

    @staticmethod
    def _fmt_int_compact(value: object) -> str:
        if value is None:
            return "na"
        return str(value)

    @staticmethod
    def _fmt_float(value: object, decimals: int) -> str:
        if value is None:
            return "n/a"
        if not isinstance(value, (int, float)):
            return str(value)
        return f"{value:.{decimals}f}"

    @staticmethod
    def _fmt_float_compact(value: object, decimals: int) -> str:
        if value is None:
            return "na"
        if not isinstance(value, (int, float)):
            return str(value)
        return f"{value:.{decimals}f}"
