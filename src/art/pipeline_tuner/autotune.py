from __future__ import annotations

from collections import defaultdict
import math
import statistics

import pydantic

from .config import (
    PackedGroupObservation,
    PipelineAutotuneConfig,
    PipelineAutotunerProfile,
    PipelineMetric,
    PipelineTuneSettings,
    TunerDecision,
    TunerWindowStats,
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ceil_to_multiple(value: float, multiple: int, *, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(value / multiple)) * multiple)


class PackingProjection(pydantic.BaseModel):
    groups: int
    spill_probability: float
    expected_padding_ratio: float


class PipelineAutotuner:
    def __init__(
        self,
        *,
        config: PipelineAutotuneConfig,
        settings: PipelineTuneSettings,
        model_name: str | None,
        backend_name: str | None,
        packed_sequence_length: int,
        inference_gpu_count: int,
        policy_age_limit_steps: float,
    ) -> None:
        self.config = config
        self.settings = settings
        self.model_name = model_name
        self.backend_name = backend_name
        self.packed_sequence_length = packed_sequence_length
        self.inference_gpu_count = inference_gpu_count
        self.policy_age_limit_steps = policy_age_limit_steps
        self.metrics: list[PipelineMetric] = []
        self.packed_groups: list[PackedGroupObservation] = []
        self.decisions: list[TunerDecision] = []
        self._last_decision_step = 0
        self._target_candidate: int | None = None
        self._target_candidate_count = 0

    def on_metric(self, rec: PipelineMetric) -> TunerDecision | None:
        self.metrics.append(rec)
        if rec.name != "objective/score_default" or rec.step is None:
            return None
        return self.maybe_decide(int(rec.step))

    def on_packed_group(self, rec: PackedGroupObservation) -> None:
        self.packed_groups.append(rec)

    def maybe_decide(self, step: int) -> TunerDecision | None:
        if step <= self.config.warmup_ignore_steps:
            return None
        if step - self._last_decision_step < self.config.window_steps:
            return None
        stats = self.window_stats()
        if stats is None or stats.end_step <= self._last_decision_step:
            return None
        decision = self._decide(stats)
        self._last_decision_step = stats.end_step
        self.decisions.append(decision)
        if decision.previous != decision.updated:
            self.settings = decision.updated
        return decision

    def window_stats(self) -> TunerWindowStats | None:
        by_step: dict[int, dict[str, PipelineMetric]] = defaultdict(dict)
        for rec in self.metrics:
            if rec.step is None or int(rec.step) <= self.config.warmup_ignore_steps:
                continue
            current = by_step[int(rec.step)].get(rec.name)
            if current is None or rec.t_s >= current.t_s:
                by_step[int(rec.step)][rec.name] = rec
        steps = sorted(
            step
            for step, values in by_step.items()
            if "objective/score_default" in values
        )
        if len(steps) < self.config.window_steps:
            return None
        window_steps = steps[-self.config.window_steps :]
        t0 = min(by_step[step]["objective/score_default"].t_s for step in window_steps)
        t1 = max(rec.t_s for step in window_steps for rec in by_step[step].values())

        def step_values(name: str) -> list[float]:
            return [
                by_step[step][name].value
                for step in window_steps
                if name in by_step[step]
            ]

        wall_values = step_values("pipeline_trainer/step_wall_s") or step_values(
            "time/step_wall_s"
        )
        collect_values = step_values("pipeline_trainer/step_collect_batch_s")
        wall = sum(wall_values)
        collect = sum(collect_values)
        groups = step_values("data/step_num_groups_trainable")
        discarded_stale = sum(step_values("staleness/discarded_groups"))
        generated_groups = sum(groups) + discarded_stale
        vllm_metrics = [
            rec
            for rec in self.metrics
            if rec.step is None and t0 <= rec.t_s <= max(t1, t0 + 1e-6)
        ]
        pack_samples = [
            float(obs.physical_tokens)
            for obs in self.packed_groups
            if obs.step in set(window_steps)
        ]
        if not pack_samples:
            raise RuntimeError(
                "Pipeline autotuner expected packed-group observations in the "
                "decision window, but none were recorded."
            )
        return TunerWindowStats(
            start_step=window_steps[0],
            end_step=window_steps[-1],
            score_mean=_mean(step_values("objective/score_default")),
            accepted_tok_per_s_mean=_mean(
                step_values("throughput/accepted_train_tok_per_s")
            ),
            trainer_idle_frac=(collect / wall) if wall > 0 else 0.0,
            vllm_capacity_wait_frac=_sample_frac(
                vllm_metrics, ("vllm/num_requests_waiting_capacity",)
            ),
            vllm_active_frac=_sample_frac(vllm_metrics, ("vllm/num_requests_running",)),
            queue_put_wait_frac=_mean(step_values("queue/put_wait_frac")),
            stale_frac=(discarded_stale / generated_groups)
            if generated_groups > 0
            else 0.0,
            predicted_stale_frac=_mean(step_values("queue/predicted_stale_fraction")),
            queue_freshness_pressure=_mean(step_values("queue/freshness_pressure")),
            token_weighted_policy_age_steps_mean=_mean(
                step_values("offpolicy/token_weighted_policy_age_steps")
                or step_values("offpolicy/mean_policy_age_steps")
                or step_values("offpolicy/art_steps_off_policy")
            ),
            freshness_discount_mean=_mean(
                step_values("sample_efficiency/freshness_discount")
            )
            or 1.0,
            padding_ratio_mean=_mean(step_values("data/step_padding_ratio")),
            groups_per_step_mean=_mean(groups),
            step_wall_s_mean=_mean(wall_values),
            collect_batch_s_mean=_mean(collect_values),
            train_work_s_mean=max(0.0, _mean(wall_values) - _mean(collect_values)),
            train_capacity_tokens_mean=_mean(step_values("data/step_train_tokens")),
            group_pack_token_samples=pack_samples,
        )

    def _decide(self, stats: TunerWindowStats) -> TunerDecision:
        capacity_bound = (
            stats.vllm_capacity_wait_frac >= self.config.vllm_capacity_wait_over_frac
        )
        inference_over = capacity_bound
        trainer_under = stats.trainer_idle_frac >= self.config.trainer_idle_under_frac
        trainer_over = stats.trainer_idle_frac <= self.config.trainer_idle_over_frac
        inference_under = not inference_over
        if (
            not inference_over
            and stats.vllm_active_frac < self.config.vllm_active_under_frac
        ):
            inference_under = True
        if inference_under and trainer_under:
            state = "inference_under_train_under"
        elif inference_over and trainer_under:
            state = "inference_over_train_under"
        elif inference_under and trainer_over:
            state = "inference_under_train_over"
        elif inference_over and trainer_over:
            state = "inference_over_train_over"
        else:
            state = (
                "inference_over_train_over"
                if inference_over
                else "inference_under_train_under"
            )

        previous = self.settings
        updated = self._settings_with_recomputed_queue(
            previous, stats, adapt_target=True
        )
        freshness_ratio = max(
            self._policy_age_ratio(stats), stats.queue_freshness_pressure
        )
        age_high = freshness_ratio >= self.config.policy_age_high_fraction
        age_severe = freshness_ratio >= self.config.policy_age_severe_fraction
        predicted_stale_high = stats.predicted_stale_frac >= self.config.stale_high_frac
        action = "hold"
        reason = "inside hysteresis band or already balanced"

        if stats.queue_put_wait_frac >= self.config.queue_put_severe_frac:
            reason = "completed-group queue backpressure is active"
        elif state == "inference_under_train_under" and not age_severe:
            updated = updated.model_copy(
                update={
                    "num_rollout_workers": self._move_workers(
                        updated.num_rollout_workers, +1
                    )
                }
            )
            action = "increase_workers"
            reason = "vLLM and trainer both show idle headroom"
        elif state == "inference_under_train_under":
            reason = "trainer is underfed but queue freshness is near the policy limit"
        elif state == "inference_over_train_under":
            floor = max(
                1,
                math.ceil(
                    updated.target_groups_per_step
                    * self.config.freshness_min_batch_floor_fraction
                ),
            )
            new_min = max(floor, round(updated.min_batch_size * 0.85))
            batch_underfilled = (
                stats.groups_per_step_mean
                < updated.target_groups_per_step * self.config.batch_underfill_frac
            )
            if batch_underfilled and not capacity_bound and not age_severe:
                updated = updated.model_copy(
                    update={
                        "num_rollout_workers": self._move_workers(
                            updated.num_rollout_workers, +1
                        )
                    }
                )
                action = "increase_workers"
                reason = "trainer is waiting on underfilled batches"
            elif age_severe and trainer_under and new_min < updated.min_batch_size:
                updated = updated.model_copy(
                    update={"min_batch_size": min(new_min, updated.max_batch_size)}
                )
                action = "lower_min_batch_size"
                reason = "trainer is underfed and predicted queue freshness is near the policy limit"
        elif state == "inference_under_train_over":
            if updated.min_batch_size < updated.max_batch_size and not age_high:
                updated = updated.model_copy(
                    update={
                        "min_batch_size": min(
                            updated.max_batch_size,
                            max(
                                updated.min_batch_size + 1,
                                round(updated.min_batch_size * 1.15),
                            ),
                        )
                    }
                )
                action = "raise_min_batch_size"
                reason = "trainer is saturated and policy age has headroom"
            elif (
                updated.min_batch_size >= updated.max_batch_size
                and predicted_stale_high
            ):
                updated = updated.model_copy(
                    update={
                        "num_rollout_workers": self._move_workers(
                            updated.num_rollout_workers, -1
                        )
                    }
                )
                action = "decrease_workers"
                reason = "trainer saturated with predicted stale backlog"
        elif state == "inference_over_train_over":
            reason = "both sides are loaded; no throughput-safe online change"

        updated = self._settings_with_recomputed_queue(
            updated, stats, adapt_target=False
        )
        if action == "hold" and updated != previous:
            action = "resize_batch_queue"
            reason = "recomputed target batch size and freshness-bounded queue"
        return TunerDecision(
            step=stats.end_step,
            state=state,
            action=action if updated != previous else "hold",
            reason=reason,
            previous=previous,
            updated=updated,
            stats=stats,
        )

    def _policy_age_ratio(self, stats: TunerWindowStats) -> float:
        limit = max(float(self.policy_age_limit_steps), 1e-9)
        return max(0.0, stats.token_weighted_policy_age_steps_mean) / limit

    def _move_workers(self, current: int, direction: int) -> int:
        raw = max(
            self.config.worker_step,
            _ceil_to_multiple(
                current * self.config.worker_move_fraction, self.config.worker_step
            ),
        )
        cap = _ceil_to_multiple(self.config.max_worker_move, self.config.worker_step)
        return max(self.config.worker_step, current + direction * min(cap, raw))

    def _settings_with_recomputed_queue(
        self,
        settings: PipelineTuneSettings,
        stats: TunerWindowStats | None,
        *,
        adapt_target: bool,
    ) -> PipelineTuneSettings:
        target = (
            self._adaptive_target_groups(settings, stats)
            if adapt_target
            else settings.target_groups_per_step
        )
        was_locked = settings.min_batch_size >= settings.max_batch_size
        min_floor = max(1, math.ceil(target * self.config.min_batch_floor_fraction))
        min_batch = min(settings.min_batch_size, target)
        if adapt_target and target > settings.target_groups_per_step and not was_locked:
            min_batch = max(min_batch, min_floor)
        # Packed sequence length is the user's cap on target/max batch size. If a
        # run should never use larger train batches, lower packed_sequence_length.
        queue = recommended_queue_size(
            target_groups_per_step=min_batch,
            limit_steps_off_policy=self.policy_age_limit_steps,
            num_rollout_workers=settings.num_rollout_workers,
            running_reserve_fraction=self.config.queue_running_reserve_fraction,
        )
        return settings.model_copy(
            update={
                "target_groups_per_step": target,
                "min_batch_size": min_batch,
                "max_batch_size": target,
                "queue_maxsize": queue,
            }
        )

    def _adaptive_target_groups(
        self, settings: PipelineTuneSettings, stats: TunerWindowStats | None
    ) -> int:
        current = settings.target_groups_per_step
        if stats is None:
            return current
        projections = self._packing_projections(settings, stats)
        allowed = [
            projection
            for projection in projections
            if projection.spill_probability <= self.config.target_spill_probability
        ]
        observed = max(allowed, key=lambda p: p.groups).groups if allowed else current
        if observed > current:
            observed = min(
                observed,
                current
                + max(
                    1,
                    min(
                        self.config.target_group_max_increase,
                        math.ceil(current * self.config.target_group_increase_fraction),
                    ),
                ),
            )
        min_delta = max(
            1, math.ceil(current * self.config.target_group_min_relative_change)
        )
        delta = observed - current
        if abs(delta) < min_delta:
            self._target_candidate = None
            self._target_candidate_count = 0
            return current
        immediate_decrease = delta < 0 and abs(delta) >= max(
            1, math.ceil(current * self.config.target_group_immediate_decrease_fraction)
        )
        if immediate_decrease:
            self._target_candidate = None
            self._target_candidate_count = 0
            return observed
        if observed == self._target_candidate:
            self._target_candidate_count += 1
        else:
            self._target_candidate = observed
            self._target_candidate_count = 1
        if self._target_candidate_count >= self.config.target_group_change_windows:
            self._target_candidate = None
            self._target_candidate_count = 0
            return observed
        return current

    def _packing_projections(
        self, settings: PipelineTuneSettings, stats: TunerWindowStats
    ) -> list[PackingProjection]:
        samples = [value for value in stats.group_pack_token_samples if value > 0]
        capacity = float(self.packed_sequence_length)
        if not samples:
            return []
        mean_tokens = _mean(samples)
        center = max(1.0, capacity / max(1.0, mean_tokens))
        current = max(1, settings.target_groups_per_step)
        lo = max(1, int(math.floor(min(center, current) * 0.5)))
        hi = min(
            max(
                current + 2,
                int(math.ceil(max(center, current) * 1.5)),
                int(math.ceil(current * 1.25)),
            ),
            2000,
        )
        sample_count = min(self.config.bootstrap_samples, max(16, len(samples) * 4))
        repeated: list[float] = []
        while len(repeated) < sample_count + hi + 1:
            repeated.extend(samples)
        prefix = [0.0]
        for value in repeated:
            prefix.append(prefix[-1] + value)
        projections: list[PackingProjection] = []
        for groups in range(lo, hi + 1):
            totals = [prefix[idx + groups] - prefix[idx] for idx in range(sample_count)]
            packed_counts = [max(1, math.ceil(total / capacity)) for total in totals]
            expected_capacity = _mean([count * capacity for count in packed_counts])
            expected_tokens = _mean(totals)
            projections.append(
                PackingProjection(
                    groups=groups,
                    spill_probability=_mean(
                        [1.0 if total > capacity else 0.0 for total in totals]
                    ),
                    expected_padding_ratio=max(
                        0.0, (expected_capacity - expected_tokens) / expected_capacity
                    )
                    if expected_capacity > 0
                    else 0.0,
                )
            )
        return projections

    def profile(self) -> PipelineAutotunerProfile:
        return PipelineAutotunerProfile(
            model_name=self.model_name,
            backend=self.backend_name,
            packed_sequence_length=self.packed_sequence_length,
            inference_gpu_count=self.inference_gpu_count,
            policy_age_limit_steps=self.policy_age_limit_steps,
            settings=self.settings,
            config=self.config,
            decisions=self.decisions,
            notes=[
                "The first warmup_ignore_steps are excluded from throughput decisions.",
                "queue_maxsize is bounded so queue_size / target_groups_per_step <= the policy-age limit.",
            ],
        )


def build_initial_settings(
    *,
    config: PipelineAutotuneConfig,
    inference_gpu_count: int,
    policy_age_limit_steps: float,
) -> PipelineTuneSettings:
    workers = _ceil_to_multiple(
        config.initial_model_calls_per_inference_gpu * inference_gpu_count,
        config.worker_step,
        minimum=config.worker_step,
    )
    max_batch = int(config.initial_max_batch_size)
    min_batch = min(int(config.initial_min_batch_size), max_batch)
    queue = recommended_queue_size(
        target_groups_per_step=max_batch,
        limit_steps_off_policy=policy_age_limit_steps,
        num_rollout_workers=workers,
        running_reserve_fraction=config.queue_running_reserve_fraction,
    )
    return PipelineTuneSettings(
        num_rollout_workers=workers,
        min_batch_size=min_batch,
        max_batch_size=max_batch,
        queue_maxsize=queue,
        target_groups_per_step=max_batch,
    )


def recommended_queue_size(
    *,
    target_groups_per_step: int,
    limit_steps_off_policy: float,
    num_rollout_workers: int,
    running_reserve_fraction: float,
) -> int:
    target = max(1, int(target_groups_per_step))
    limit = max(1.0, float(limit_steps_off_policy))
    max_completed = max(1, int(math.floor(target * limit)))
    running_reserve = int(
        math.ceil(max(0, num_rollout_workers) * running_reserve_fraction)
    )
    lower = target
    return max(lower, min(max_completed, max_completed - running_reserve))


def _sample_frac(metrics: list[PipelineMetric], names: tuple[str, ...]) -> float:
    values = [rec.value for rec in metrics if rec.name in names]
    if not values:
        return 0.0
    return _mean([1.0 if value > 0 else 0.0 for value in values])
