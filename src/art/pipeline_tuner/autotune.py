from __future__ import annotations

from collections import defaultdict
import math
import statistics
import warnings

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


def _required_step_values(
    by_step: dict[int, dict[str, PipelineMetric]],
    window_steps: list[int],
    name: str,
) -> list[float]:
    missing = [step for step in window_steps if name not in by_step[step]]
    if missing:
        raise RuntimeError(
            "Pipeline autotuning requires metric "
            f"{name!r} in every decision-window step; missing steps {missing}."
        )
    return [by_step[step][name].value for step in window_steps]


def _ceil_to_multiple(value: float, multiple: int, *, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(value / multiple)) * multiple)


_VLLM_SCRAPE_GROUP_TOLERANCE_S = 0.05
_TRAINER_PADDING_EPSILON = 1e-9


class PackingProjection(pydantic.BaseModel):
    groups: int
    spill_probability: float
    expected_padding_ratio: float
    bootstrap_spill_probability: float = 0.0
    history_spill_probability_upper: float = 0.0
    history_trials: float = 0.0
    history_spills: float = 0.0
    history_bad_padding_events: float = 0.0
    history_bad_padding_probability_upper: float = 0.0


class PackingOutcome(pydantic.BaseModel):
    step: int
    groups: int = pydantic.Field(ge=1)
    packed_sequences: int = pydantic.Field(ge=1)
    padding_ratio: float = pydantic.Field(ge=0.0, le=1.0)
    non_padding_tokens: float = pydantic.Field(ge=0.0)
    train_tokens: float = pydantic.Field(gt=0.0)


class PackingHistoryRisk(pydantic.BaseModel):
    groups: int
    trials: float = 0.0
    spills: float = 0.0
    spill_probability_upper: float = 0.0
    bad_padding_events: float = 0.0
    bad_padding_probability_upper: float = 0.0


class VllmLoadStats(pydantic.BaseModel):
    capacity_wait_frac: float = 0.0
    active_frac: float = 0.0
    capacity_wait_request_s: float = 0.0
    running_request_s: float = 0.0
    pressure: float = 0.0
    capacity_wait_area: float = 0.0
    running_area: float = 0.0
    idle_frac: float = 0.0
    max_num_seqs_mean: float = 0.0


def _trainer_load_score(*, idle_frac: float, padding_ratio: float) -> float:
    denominator = max(
        _TRAINER_PADDING_EPSILON,
        1.0 + _TRAINER_PADDING_EPSILON - max(0.0, min(1.0, padding_ratio)),
    )
    return max(0.0, idle_frac) / denominator


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
        self._packing_outcomes: list[PackingOutcome] = []
        self._packing_outcome_steps: set[int] = set()
        self.decisions: list[TunerDecision] = []
        self._last_decision_step = 0
        self._target_candidate: int | None = None
        self._target_candidate_count = 0
        self._emitted_recommendations: set[str] = set()

    def on_metric(self, rec: PipelineMetric) -> TunerDecision | None:
        self.metrics.append(rec)
        if rec.name != "objective/score" or rec.step is None:
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
        self._emit_stable_recommendations(decision)
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
            step for step, values in by_step.items() if "objective/score" in values
        )
        if len(steps) < self.config.window_steps:
            return None
        window_steps = steps[-self.config.window_steps :]
        t0 = min(by_step[step]["objective/score"].t_s for step in window_steps)
        t1 = max(rec.t_s for step in window_steps for rec in by_step[step].values())

        def step_values(name: str) -> list[float]:
            return [
                by_step[step][name].value
                for step in window_steps
                if name in by_step[step]
            ]

        wall_values = _required_step_values(by_step, window_steps, "time/step_wall_s")
        collect_values = _required_step_values(
            by_step, window_steps, "time/step_collect_batch_s"
        )
        wall = sum(wall_values)
        collect = sum(collect_values)
        groups = _required_step_values(
            by_step, window_steps, "data/step_num_groups_trainable"
        )
        train_capacity_tokens = _required_step_values(
            by_step, window_steps, "data/step_packed_train_tokens"
        )
        discarded_stale = sum(step_values("discarded/step/stale_groups"))
        generated_groups = sum(groups) + discarded_stale
        vllm_metrics = [
            rec
            for rec in self.metrics
            if rec.step is None and t0 <= rec.t_s <= max(t1, t0 + 1e-6)
        ]
        window_step_set = set(window_steps)
        pack_by_step: dict[int, list[float]] = defaultdict(list)
        for obs in self.packed_groups:
            if obs.step in window_step_set:
                pack_by_step[obs.step].append(float(obs.physical_tokens))
        missing_packed_steps = [
            step
            for step, group_count in zip(window_steps, groups)
            if group_count > 0 and not pack_by_step[step]
        ]
        if missing_packed_steps:
            raise RuntimeError(
                "Pipeline autotuner requires packed-group observations in every "
                f"trainable decision-window step; missing steps {missing_packed_steps}."
            )
        pack_samples = [
            sample for step in window_steps for sample in pack_by_step.get(step, [])
        ]
        padding_ratios = []
        for step, capacity in zip(window_steps, train_capacity_tokens):
            if capacity <= 0:
                continue
            non_padding = sum(pack_by_step.get(step, []))
            padding_ratios.append(max(0.0, (capacity - non_padding) / capacity))
        trainer_idle_frac = (collect / wall) if wall > 0 else 0.0
        padding_ratio_mean = _mean(padding_ratios)
        self._record_packing_outcomes(
            by_step=by_step,
            window_steps=window_steps,
            pack_by_step=pack_by_step,
        )
        vllm_load = _vllm_load_stats(vllm_metrics, window_start_s=t0, window_end_s=t1)
        return TunerWindowStats(
            start_step=window_steps[0],
            end_step=window_steps[-1],
            window_start_s=t0,
            window_end_s=t1,
            score_mean=_mean(
                _required_step_values(by_step, window_steps, "objective/score")
            ),
            accepted_tok_per_s_mean=_mean(
                _required_step_values(
                    by_step, window_steps, "throughput/accepted_train_tok_per_s"
                )
            ),
            trainer_idle_frac=trainer_idle_frac,
            trainer_load_score=_trainer_load_score(
                idle_frac=trainer_idle_frac,
                padding_ratio=padding_ratio_mean,
            ),
            vllm_capacity_wait_frac=vllm_load.capacity_wait_frac,
            vllm_active_frac=vllm_load.active_frac,
            vllm_capacity_wait_request_s=vllm_load.capacity_wait_request_s,
            vllm_running_request_s=vllm_load.running_request_s,
            vllm_pressure=vllm_load.pressure,
            vllm_capacity_wait_area=vllm_load.capacity_wait_area,
            vllm_running_area=vllm_load.running_area,
            vllm_idle_frac=vllm_load.idle_frac,
            vllm_max_num_seqs_mean=vllm_load.max_num_seqs_mean,
            queue_put_wait_frac=_mean(step_values("queue/put_wait_frac")),
            stale_frac=(discarded_stale / generated_groups)
            if generated_groups > 0
            else 0.0,
            predicted_stale_frac=_mean(step_values("queue/predicted_stale_fraction")),
            queue_freshness_pressure=_mean(step_values("queue/freshness_pressure")),
            token_weighted_policy_age_steps_mean=_mean(
                _required_step_values(
                    by_step, window_steps, "offpolicy/token_weighted_policy_age_steps"
                )
            ),
            freshness_discount_mean=_mean(
                step_values("sample_efficiency/freshness_discount")
            )
            or 1.0,
            padding_ratio_mean=padding_ratio_mean,
            groups_per_step_mean=_mean(groups),
            step_wall_s_mean=_mean(wall_values),
            collect_batch_s_mean=_mean(collect_values),
            train_work_s_mean=max(0.0, _mean(wall_values) - _mean(collect_values)),
            train_capacity_tokens_mean=_mean(train_capacity_tokens),
            group_pack_token_samples=pack_samples,
        )

    def _record_packing_outcomes(
        self,
        *,
        by_step: dict[int, dict[str, PipelineMetric]],
        window_steps: list[int],
        pack_by_step: dict[int, list[float]],
    ) -> None:
        required = {
            "data/step_num_groups_trainable",
            "data/step_packed_sequences",
            "data/step_packed_train_tokens",
        }
        for step in window_steps:
            if step in self._packing_outcome_steps:
                continue
            values = by_step[step]
            missing = sorted(required.difference(values))
            if missing:
                raise RuntimeError(
                    "Pipeline autotuning requires packing outcome metrics in every "
                    f"decision-window step; missing {missing} at step {step}."
                )
            groups = int(round(values["data/step_num_groups_trainable"].value))
            if groups <= 0:
                continue
            train_tokens = float(values["data/step_packed_train_tokens"].value)
            if train_tokens <= 0.0:
                raise RuntimeError(
                    "Pipeline autotuning requires positive "
                    "data/step_packed_train_tokens "
                    f"at step {step}."
                )
            non_padding_tokens = sum(pack_by_step.get(step, []))
            padding_ratio = max(0.0, (train_tokens - non_padding_tokens) / train_tokens)
            self._packing_outcomes.append(
                PackingOutcome(
                    step=step,
                    groups=groups,
                    packed_sequences=int(
                        round(values["data/step_packed_sequences"].value)
                    ),
                    padding_ratio=min(1.0, padding_ratio),
                    non_padding_tokens=non_padding_tokens,
                    train_tokens=train_tokens,
                )
            )
            self._packing_outcome_steps.add(step)
        if not self._packing_outcomes:
            return
        newest_step = max(outcome.step for outcome in self._packing_outcomes)
        cutoff_step = newest_step - self.config.packing_history_steps + 1
        self._packing_outcomes = [
            outcome for outcome in self._packing_outcomes if outcome.step >= cutoff_step
        ]
        self._packing_outcome_steps = {
            outcome.step for outcome in self._packing_outcomes
        }

    def _decide(self, stats: TunerWindowStats) -> TunerDecision:
        inference_over = stats.vllm_pressure > self.config.vllm_pressure_over_ratio
        trainer_under = stats.trainer_load_score > self.config.trainer_load_under_score
        trainer_over = stats.trainer_load_score <= self.config.trainer_load_over_score
        inference_under = (
            not inference_over
            and stats.vllm_pressure <= self.config.vllm_pressure_under_ratio
        )
        inference_state = (
            "inference_over"
            if inference_over
            else "inference_under"
            if inference_under
            else "inference_balanced"
        )
        trainer_state = (
            "train_under"
            if trainer_under
            else "train_over"
            if trainer_over
            else "train_balanced"
        )
        state = f"{inference_state}_{trainer_state}"

        previous = self.settings
        updated = self._settings_with_recomputed_queue(
            previous, stats, adapt_target=True
        )
        target_changed = (
            updated.target_groups_per_step != previous.target_groups_per_step
        )
        predicted_stale_high = stats.predicted_stale_frac >= self.config.stale_high_frac
        action = "hold"
        reason = "inside hysteresis band or already balanced"

        if stats.queue_put_wait_frac >= self.config.queue_put_severe_frac:
            reason = "completed-group queue backpressure is active"
        elif state in {
            "inference_under_train_under",
            "inference_balanced_train_under",
        }:
            updated = updated.model_copy(
                update={
                    "num_rollout_workers": self._move_workers(
                        updated.num_rollout_workers, +1
                    )
                }
            )
            action = "increase_workers"
            reason = "vLLM pressure is low and trainer is underfed"
        elif state in {
            "inference_under_train_over",
            "inference_balanced_train_over",
        }:
            if (
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

        if not target_changed:
            min_update = self._min_batch_adjustment(updated, stats, state, action)
            if min_update is not None:
                updated, action, reason = min_update

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

    def _min_batch_adjustment(
        self,
        settings: PipelineTuneSettings,
        stats: TunerWindowStats,
        state: str,
        action: str,
    ) -> tuple[PipelineTuneSettings, str, str] | None:
        if (
            action != "increase_workers"
            and stats.trainer_load_score > self.config.trainer_min_batch_lower_score
        ):
            floor = max(
                1,
                math.ceil(
                    settings.target_groups_per_step
                    * self.config.freshness_min_batch_floor_fraction
                ),
            )
            new_min = max(floor, round(settings.min_batch_size * 0.85))
            if new_min < settings.min_batch_size:
                return (
                    settings.model_copy(
                        update={"min_batch_size": min(new_min, settings.max_batch_size)}
                    ),
                    "lower_min_batch_size",
                    "trainer is severely underfed and rollout workers are not being increased",
                )
        should_raise = action == "decrease_workers" or state in {
            "inference_under_train_over",
            "inference_balanced_train_over",
        }
        if should_raise and settings.min_batch_size < settings.max_batch_size:
            new_min = min(
                settings.max_batch_size,
                max(settings.min_batch_size + 1, round(settings.min_batch_size * 1.15)),
            )
            if new_min > settings.min_batch_size:
                return (
                    settings.model_copy(update={"min_batch_size": new_min}),
                    "raise_min_batch_size",
                    "trainer is saturated enough to use denser batches before reducing workers",
                )
        return None

    def _emit_stable_recommendations(self, decision: TunerDecision) -> None:
        recommendations = self._stable_recommendations()
        decision.recommendations.extend(message for _, message in recommendations)
        for key, message in recommendations:
            if key in self._emitted_recommendations:
                continue
            self._emitted_recommendations.add(key)
            warnings.warn(message, UserWarning, stacklevel=2)

    def _stable_recommendations(self) -> list[tuple[str, str]]:
        hold_count = self.config.recommendation_consecutive_holds
        if len(self.decisions) < self.config.recommendation_min_windows:
            return []
        recent = self.decisions[-hold_count:]
        if len(recent) < hold_count or any(
            decision.action != "hold" for decision in recent
        ):
            return []
        current = dict(self._recommendation_candidates(recent[-1]))
        for decision in recent[:-1]:
            current = {
                key: message
                for key, message in current.items()
                if key in dict(self._recommendation_candidates(decision))
            }
        return list(current.items())

    def _recommendation_candidates(
        self, decision: TunerDecision
    ) -> list[tuple[str, str]]:
        stats = decision.stats
        if stats is None:
            return []
        vllm_saturated = stats.vllm_pressure > self.config.vllm_pressure_over_ratio
        vllm_underloaded = stats.vllm_pressure <= self.config.vllm_pressure_under_ratio
        trainer_severely_underloaded = (
            stats.trainer_load_score >= self.config.trainer_load_severe_under_score
        )
        trainer_saturated = (
            stats.trainer_load_score <= self.config.trainer_load_over_score
        )
        recommendations: list[tuple[str, str]] = []
        if vllm_saturated and trainer_severely_underloaded:
            recommendations.append(
                (
                    "increase_inference_gpus",
                    "Pipeline autotuner observes saturated vLLM request pressure "
                    "while Megatron is severely underloaded; increase inference GPUs "
                    "if possible.",
                )
            )
        if vllm_underloaded and trainer_saturated:
            recommendations.append(
                (
                    "increase_group_size_or_training_gpus",
                    "Pipeline autotuner observes severely underloaded vLLM request "
                    "pressure while Megatron is saturated; increase rollout group "
                    "size to use spare inference capacity, or increase training GPUs "
                    "if possible.",
                )
            )
        if (
            stats.padding_ratio_mean >= self.config.padding_high_frac
            and trainer_saturated
            and vllm_saturated
        ):
            recommendations.append(
                (
                    "decrease_packed_sequence_length",
                    "Pipeline autotuner observes high padding while Megatron and vLLM "
                    "are both saturated; decrease packed_sequence_length to reduce "
                    "padding waste.",
                )
            )
        return recommendations

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
        min_batch = min(settings.min_batch_size, target)
        if adapt_target and target > settings.target_groups_per_step:
            ratio = settings.min_batch_size / max(1, settings.max_batch_size)
            min_batch = min(target, max(1, round(target * ratio)))
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
        selected_projection = next(
            (projection for projection in projections if projection.groups == observed),
            None,
        )
        if selected_projection is not None:
            self._record_selected_packing_projection(stats, selected_projection)
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
        samples = [
            value for value in self._packing_projection_samples(stats) if value > 0
        ]
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
        history_risks = self._packing_history_risks(range(lo, hi + 1))
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
            bootstrap_spill_probability = _mean(
                [1.0 if total > capacity else 0.0 for total in totals]
            )
            history_risk = history_risks[groups]
            projections.append(
                PackingProjection(
                    groups=groups,
                    spill_probability=max(
                        bootstrap_spill_probability,
                        history_risk.spill_probability_upper,
                    ),
                    expected_padding_ratio=max(
                        0.0, (expected_capacity - expected_tokens) / expected_capacity
                    )
                    if expected_capacity > 0
                    else 0.0,
                    bootstrap_spill_probability=bootstrap_spill_probability,
                    history_spill_probability_upper=history_risk.spill_probability_upper,
                    history_trials=history_risk.trials,
                    history_spills=history_risk.spills,
                    history_bad_padding_events=history_risk.bad_padding_events,
                    history_bad_padding_probability_upper=(
                        history_risk.bad_padding_probability_upper
                    ),
                )
            )
        return projections

    def _packing_projection_samples(self, stats: TunerWindowStats) -> list[float]:
        cutoff_step = stats.end_step - self.config.packing_history_steps + 1
        prior_samples = [
            float(observation.physical_tokens)
            for observation in self.packed_groups
            if cutoff_step <= observation.step < stats.start_step
        ]
        return prior_samples + stats.group_pack_token_samples

    def _packing_history_risks(
        self, groups_range: range
    ) -> dict[int, PackingHistoryRisk]:
        risks: dict[int, PackingHistoryRisk] = {}
        for groups in groups_range:
            outcomes = [
                outcome
                for outcome in self._packing_outcomes
                if outcome.groups == groups
            ]
            trials = float(len(outcomes))
            spills = float(
                sum(1 for outcome in outcomes if outcome.packed_sequences > 1)
            )
            bad_padding_events = float(
                sum(
                    1
                    for outcome in outcomes
                    if outcome.padding_ratio >= self.config.packing_bad_padding_ratio
                )
            )
            # Zero-spill samples are useful diagnostics but should not block exploration:
            # a beta upper bound with sparse clean samples would make target batches
            # sticky. Actual spills are the hard signal we carry across the horizon.
            risks[groups] = PackingHistoryRisk(
                groups=groups,
                trials=trials,
                spills=spills,
                spill_probability_upper=self._packing_probability_upper(
                    events=spills, trials=trials
                )
                if spills > 0.0
                else 0.0,
                bad_padding_events=bad_padding_events,
                bad_padding_probability_upper=self._packing_probability_upper(
                    events=bad_padding_events, trials=trials
                )
                if bad_padding_events > 0.0
                else 0.0,
            )
        inherited_spill_probability = 0.0
        for groups in sorted(risks):
            inherited_spill_probability = max(
                inherited_spill_probability, risks[groups].spill_probability_upper
            )
            if inherited_spill_probability > risks[groups].spill_probability_upper:
                risks[groups] = risks[groups].model_copy(
                    update={
                        "spill_probability_upper": inherited_spill_probability,
                    }
                )
        return risks

    def _packing_probability_upper(self, *, events: float, trials: float) -> float:
        from scipy.stats import beta as beta_distribution

        non_events = max(0.0, trials - events)
        value = float(
            beta_distribution.ppf(
                self.config.packing_spill_confidence,
                self.config.packing_spill_prior_alpha + events,
                self.config.packing_spill_prior_beta + non_events,
            )
        )
        if not math.isfinite(value):
            raise RuntimeError("Failed to compute packing spill beta posterior.")
        return max(0.0, min(1.0, value))

    @staticmethod
    def _record_selected_packing_projection(
        stats: TunerWindowStats, projection: PackingProjection
    ) -> None:
        stats.packing_history_groups = projection.groups
        stats.packing_history_trials = projection.history_trials
        stats.packing_history_spills = projection.history_spills
        stats.packing_history_spill_probability_upper = (
            projection.history_spill_probability_upper
        )
        stats.packing_history_bad_padding_events = projection.history_bad_padding_events
        stats.packing_history_bad_padding_probability_upper = (
            projection.history_bad_padding_probability_upper
        )

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
                *self._profile_recommendations(),
            ],
        )

    def _profile_recommendations(self) -> list[str]:
        seen: set[str] = set()
        recommendations: list[str] = []
        for decision in self.decisions:
            for recommendation in decision.recommendations:
                if recommendation in seen:
                    continue
                seen.add(recommendation)
                recommendations.append(recommendation)
        return recommendations


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


def _vllm_load_stats(
    metrics: list[PipelineMetric], *, window_start_s: float, window_end_s: float
) -> VllmLoadStats:
    wanted = {
        "vllm/num_requests_running",
        "vllm/num_requests_waiting",
        "vllm/num_requests_waiting_capacity",
        "vllm/max_num_seqs",
    }
    rows: list[tuple[float, str, float]] = []
    for rec in metrics:
        if rec.name in wanted and math.isfinite(rec.value):
            rows.append((rec.t_s, rec.name, rec.value))
    if not rows:
        raise RuntimeError("Pipeline autotuning requires vLLM runtime metric samples.")
    by_time = _group_vllm_metric_rows(rows)
    samples: list[tuple[float, float, float, float]] = []
    complete_times: list[float] = []
    for t_s, values in by_time.items():
        if not {
            "vllm/num_requests_running",
            "vllm/num_requests_waiting",
            "vllm/num_requests_waiting_capacity",
        }.issubset(values):
            continue
        complete_times.append(t_s)
        samples.append(
            (
                values["vllm/num_requests_running"],
                values["vllm/num_requests_waiting"],
                values["vllm/num_requests_waiting_capacity"],
                values.get("vllm/max_num_seqs", 0.0),
            )
        )
    if not samples:
        raise RuntimeError(
            "Pipeline autotuning requires complete vLLM running/waiting/capacity "
            "samples in each decision window."
        )
    times = sorted(complete_times)
    capacity_wait_request_s = 0.0
    running_request_s = 0.0
    total_s = 0.0
    for idx, t_s in enumerate(times):
        values = by_time[t_s]
        if not {
            "vllm/num_requests_running",
            "vllm/num_requests_waiting_capacity",
        }.issubset(values):
            continue
        next_t_s = times[idx + 1] if idx + 1 < len(times) else window_end_s
        start_s = max(t_s, window_start_s)
        end_s = min(next_t_s, window_end_s)
        if end_s <= start_s:
            continue
        duration_s = end_s - start_s
        total_s += duration_s
        capacity_wait_request_s += (
            max(0.0, values["vllm/num_requests_waiting_capacity"]) * duration_s
        )
        running_request_s += max(0.0, values["vllm/num_requests_running"]) * duration_s
    if total_s <= 0.0:
        raise RuntimeError("Pipeline autotuning requires nonzero vLLM sample duration.")
    if running_request_s > 0.0:
        pressure = capacity_wait_request_s / running_request_s
    elif capacity_wait_request_s > 0.0:
        pressure = math.inf
    else:
        pressure = 0.0
    max_num_seqs_values = [
        max_num_seqs for _, _, _, max_num_seqs in samples if max_num_seqs > 0
    ]
    return VllmLoadStats(
        capacity_wait_frac=_mean(
            [1.0 if wait_capacity > 0 else 0.0 for _, _, wait_capacity, _ in samples]
        ),
        active_frac=_mean(
            [1.0 if running > 0 else 0.0 for running, _, _, _ in samples]
        ),
        capacity_wait_request_s=capacity_wait_request_s,
        running_request_s=running_request_s,
        pressure=pressure,
        capacity_wait_area=_mean(
            [
                max(0.0, wait_capacity) / max_num_seqs
                for _, _, wait_capacity, max_num_seqs in samples
                if max_num_seqs > 0
            ]
        ),
        running_area=_mean(
            [
                max(0.0, running) / max_num_seqs
                for running, _, _, max_num_seqs in samples
                if max_num_seqs > 0
            ]
        ),
        idle_frac=_mean(
            [
                1.0 if running <= 0 and waiting <= 0 else 0.0
                for running, waiting, _, _ in samples
            ]
        ),
        max_num_seqs_mean=_mean(max_num_seqs_values),
    )


def _group_vllm_metric_rows(
    rows: list[tuple[float, str, float]],
) -> dict[float, dict[str, float]]:
    rows.sort(key=lambda row: row[0])
    groups: list[list[tuple[float, str, float]]] = []
    current: list[tuple[float, str, float]] = []
    last_t_s: float | None = None
    for row in rows:
        t_s = row[0]
        if last_t_s is None or t_s - last_t_s <= _VLLM_SCRAPE_GROUP_TOLERANCE_S:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
        last_t_s = t_s
    if current:
        groups.append(current)
    by_time: dict[float, dict[str, float]] = {}
    for group in groups:
        values: dict[str, float] = {}
        for _, name, value in group:
            values[name] = value
        by_time[group[0][0]] = values
    return by_time
