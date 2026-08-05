from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
import random
import statistics
from typing import cast
import warnings

import pydantic

from ..preprocessing.pack import PrefixTreePackingPool
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


class PackingOutcome(pydantic.BaseModel):
    step: int
    groups: int = pydantic.Field(ge=1)
    packed_sequences: int = pydantic.Field(ge=1)


def _trainer_underfeed_score(*, idle_frac: float, padding_ratio: float) -> float:
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
        target_packed_sequences: int,
        inference_gpu_count: int,
        policy_age_limit_steps: float,
        rollout_worker_capacity: int | None,
    ) -> None:
        if rollout_worker_capacity is not None and rollout_worker_capacity < 1:
            raise ValueError("rollout_worker_capacity must be >= 1")
        if (
            rollout_worker_capacity is not None
            and settings.num_rollout_workers > rollout_worker_capacity
        ):
            raise ValueError("initial settings exceed rollout worker capacity")
        self.config = config
        self.settings = settings
        self.model_name = model_name
        self.backend_name = backend_name
        self.packed_sequence_length = packed_sequence_length
        self.target_packed_sequences = max(1, int(target_packed_sequences))
        self.inference_gpu_count = inference_gpu_count
        self.policy_age_limit_steps = policy_age_limit_steps
        self.rollout_worker_capacity = rollout_worker_capacity
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
        if self.packed_groups and rec.step > self.packed_groups[-1].step:
            cutoff_step = rec.step - self.config.packing_history_steps + 1
            self.packed_groups = [
                observation
                for observation in self.packed_groups
                if observation.step >= cutoff_step
            ]
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
            by_step, window_steps, "data/step_executed_token_equivalents"
        )
        non_padding_tokens = _required_step_values(
            by_step, window_steps, "data/step_nonpadding_logical_tokens"
        )
        vllm_metrics = [
            rec
            for rec in self.metrics
            if rec.step is None and t0 <= rec.t_s <= max(t1, t0 + 1e-6)
        ]
        window_step_set = set(window_steps)
        packed_group_counts: dict[int, int] = defaultdict(int)
        for obs in self.packed_groups:
            if obs.step in window_step_set:
                packed_group_counts[obs.step] += 1
        missing_packed_steps = [
            step
            for step, group_count in zip(window_steps, groups)
            if group_count > 0 and packed_group_counts[step] != int(round(group_count))
        ]
        if missing_packed_steps:
            raise RuntimeError(
                "Pipeline autotuner requires packed-group observations in every "
                f"trainable decision-window step; missing steps {missing_packed_steps}."
            )
        padding_ratios = []
        for capacity, non_padding in zip(
            train_capacity_tokens, non_padding_tokens, strict=True
        ):
            if capacity <= 0:
                continue
            padding_ratios.append(max(0.0, (capacity - non_padding) / capacity))
        trainer_idle_frac = (collect / wall) if wall > 0 else 0.0
        padding_ratio_mean = _mean(padding_ratios)
        self._record_packing_outcomes(
            by_step=by_step,
            window_steps=window_steps,
        )
        return TunerWindowStats(
            start_step=window_steps[0],
            end_step=window_steps[-1],
            window_start_s=t0,
            window_end_s=t1,
            trainer_underfeed_score=_trainer_underfeed_score(
                idle_frac=trainer_idle_frac,
                padding_ratio=padding_ratio_mean,
            ),
            vllm_pressure=_vllm_pressure(
                vllm_metrics, window_start_s=t0, window_end_s=t1
            ),
            queue_put_wait_frac=_mean(step_values("queue/put_wait_frac")),
            predicted_stale_frac=_mean(step_values("queue/predicted_stale_fraction")),
            padding_ratio_mean=padding_ratio_mean,
        )

    def _record_packing_outcomes(
        self,
        *,
        by_step: dict[int, dict[str, PipelineMetric]],
        window_steps: list[int],
    ) -> None:
        required = {
            "data/step_num_groups_trainable",
            "data/step_packed_sequences",
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
            packed_sequences = int(round(values["data/step_packed_sequences"].value))
            self._packing_outcomes.append(
                PackingOutcome(
                    step=step,
                    groups=groups,
                    packed_sequences=packed_sequences,
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
        trainer_under = (
            stats.trainer_underfeed_score > self.config.trainer_load_under_score
        )
        trainer_over = (
            stats.trainer_underfeed_score <= self.config.trainer_load_over_score
        )
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
        worker_limit = freshness_worker_limit(
            target_groups_per_step=updated.target_groups_per_step,
            limit_steps_off_policy=self.policy_age_limit_steps,
            running_reserve_fraction=self.config.queue_running_reserve_fraction,
            worker_step=self.config.worker_step,
        )
        if (
            worker_limit is not None
            and previous.num_rollout_workers > worker_limit
            and updated.num_rollout_workers <= worker_limit
        ):
            action = "decrease_workers"
            reason = "running rollout reserve exceeded the policy-age budget"
        elif action == "hold" and updated != previous:
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
            and stats.trainer_underfeed_score
            > self.config.trainer_min_batch_lower_score
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
            stats.trainer_underfeed_score >= self.config.trainer_load_severe_under_score
        )
        trainer_saturated = (
            stats.trainer_underfeed_score <= self.config.trainer_load_over_score
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
        floor = min(
            self.config.worker_step,
            self.rollout_worker_capacity or self.config.worker_step,
        )
        moved = max(floor, current + direction * min(cap, raw))
        return min(moved, self.rollout_worker_capacity or moved)

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
        worker_limit = freshness_worker_limit(
            target_groups_per_step=target,
            limit_steps_off_policy=self.policy_age_limit_steps,
            running_reserve_fraction=self.config.queue_running_reserve_fraction,
            worker_step=self.config.worker_step,
        )
        workers = (
            settings.num_rollout_workers
            if worker_limit is None
            else min(settings.num_rollout_workers, worker_limit)
        )
        queue = recommended_queue_size(
            target_groups_per_step=target,
            limit_steps_off_policy=self.policy_age_limit_steps,
            num_rollout_workers=workers,
            running_reserve_fraction=self.config.queue_running_reserve_fraction,
        )
        return settings.model_copy(
            update={
                "num_rollout_workers": workers,
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
        if not projections:
            raise RuntimeError(
                "Pipeline autotuner requires packing shapes before adapting batch size"
            )
        allowed = [
            projection
            for projection in projections
            if projection.spill_probability <= self.config.target_spill_probability
        ]
        observed = (
            max(allowed, key=lambda p: p.groups).groups
            if allowed
            else min(projections, key=lambda p: p.groups).groups
        )
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
        reservoir = self._packing_reservoir(settings, stats)
        if not reservoir:
            return []
        pool = PrefixTreePackingPool(
            [
                [
                    (cast(Sequence[int], leaf.token_ids), leaf.shareable_length)
                    for leaf in observation.leaves
                ]
                for observation in reservoir
            ]
        )
        current = max(1, settings.target_groups_per_step)
        increase = max(
            1,
            min(
                self.config.target_group_max_increase,
                math.ceil(current * self.config.target_group_increase_fraction),
            ),
        )
        lo = max(1, current // 2)
        hi = min(len(reservoir), current + increase)
        history_risks = self._packing_history_risks(range(lo, hi + 1))
        projections: dict[int, PackingProjection] = {}

        def project(groups: int) -> PackingProjection:
            existing = projections.get(groups)
            if existing is not None:
                return existing
            rng = random.Random((stats.end_step << 32) ^ groups)
            spills = 0.0
            for _ in range(self.config.packing_trials):
                selected = rng.sample(range(len(reservoir)), groups)
                after = pool.estimate(selected, seq_len=self.packed_sequence_length)
                spills += float(after.packed_sequences > self.target_packed_sequences)
            trials = float(self.config.packing_trials)
            counterfactual_risk = self._packing_probability_upper(
                events=spills,
                trials=trials,
            )
            projection = PackingProjection(
                groups=groups,
                spill_probability=max(counterfactual_risk, history_risks[groups]),
            )
            projections[groups] = projection
            return projection

        best = lo - 1
        left, right = lo, hi
        while left <= right:
            groups = (left + right) // 2
            if (
                project(groups).spill_probability
                <= self.config.target_spill_probability
            ):
                best = groups
                left = groups + 1
            else:
                right = groups - 1
        for groups in range(max(lo, best - 2), min(hi, best + 2) + 1):
            project(groups)
        monotone_risk = 0.0
        for groups in sorted(projections):
            projection = projections[groups]
            monotone_risk = max(monotone_risk, projection.spill_probability)
            if projection.spill_probability < monotone_risk:
                projections[groups] = projection.model_copy(
                    update={"spill_probability": monotone_risk}
                )
        return [projections[groups] for groups in sorted(projections)]

    def _packing_reservoir(
        self,
        settings: PipelineTuneSettings,
        stats: TunerWindowStats,
    ) -> list[PackedGroupObservation]:
        cutoff_step = stats.end_step - self.config.packing_history_steps + 1
        recent = [
            observation
            for observation in self.packed_groups
            if cutoff_step <= observation.step <= stats.end_step
        ]
        by_step: dict[int, list[PackedGroupObservation]] = defaultdict(list)
        for observation in recent:
            by_step[observation.step].append(observation)
        increase = max(
            1,
            min(
                self.config.target_group_max_increase,
                math.ceil(
                    settings.target_groups_per_step
                    * self.config.target_group_increase_fraction
                ),
            ),
        )
        required = max(
            self.config.packing_reservoir_min_groups,
            self.config.packing_reservoir_multiplier
            * (settings.target_groups_per_step + increase + 1),
        )
        selected: list[PackedGroupObservation] = []
        for step in sorted(by_step, reverse=True):
            selected.extend(by_step[step])
            if len(selected) >= required:
                break
        return selected

    def _packing_history_risks(self, groups_range: range) -> dict[int, float]:
        risks: dict[int, float] = {}
        for groups in groups_range:
            outcomes = [
                outcome
                for outcome in self._packing_outcomes
                if outcome.groups == groups
            ]
            trials = float(len(outcomes))
            spills = float(
                sum(
                    1
                    for outcome in outcomes
                    if outcome.packed_sequences > self.target_packed_sequences
                )
            )
            # Zero-spill samples are useful diagnostics but should not block exploration:
            # a beta upper bound with sparse clean samples would make target batches
            # sticky. Actual spills are the hard signal we carry across the horizon.
            risks[groups] = (
                self._packing_probability_upper(events=spills, trials=trials)
                if spills > 0.0
                else 0.0
            )
        inherited_spill_probability = 0.0
        for groups in sorted(risks):
            inherited_spill_probability = max(
                inherited_spill_probability, risks[groups]
            )
            risks[groups] = inherited_spill_probability
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

    def profile(self) -> PipelineAutotunerProfile:
        return PipelineAutotunerProfile(
            model_name=self.model_name,
            backend=self.backend_name,
            packed_sequence_length=self.packed_sequence_length,
            target_packed_sequences=self.target_packed_sequences,
            inference_gpu_count=self.inference_gpu_count,
            rollout_worker_capacity=self.rollout_worker_capacity,
            policy_age_limit_steps=self.policy_age_limit_steps,
            settings=self.settings,
            config=self.config,
            decisions=self.decisions,
            notes=[
                "The first warmup_ignore_steps are excluded from throughput decisions.",
                "The queued batch and running-rollout reserve are bounded by the policy-age budget.",
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
    target_packed_sequences: int,
    policy_age_limit_steps: float,
    rollout_worker_capacity: int | None,
) -> PipelineTuneSettings:
    target_slots = max(1, int(target_packed_sequences))
    max_batch = int(config.initial_max_groups_per_packed_sequence) * target_slots
    min_batch = min(
        int(config.initial_min_groups_per_packed_sequence) * target_slots,
        max_batch,
    )
    workers = _ceil_to_multiple(
        config.initial_model_calls_per_inference_gpu * inference_gpu_count,
        config.worker_step,
        minimum=config.worker_step,
    )
    worker_limit = freshness_worker_limit(
        target_groups_per_step=max_batch,
        limit_steps_off_policy=policy_age_limit_steps,
        running_reserve_fraction=config.queue_running_reserve_fraction,
        worker_step=config.worker_step,
    )
    if worker_limit is not None:
        workers = min(workers, worker_limit)
    if rollout_worker_capacity is not None:
        workers = min(workers, rollout_worker_capacity)
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


def freshness_worker_limit(
    *,
    target_groups_per_step: int,
    limit_steps_off_policy: float,
    running_reserve_fraction: float,
    worker_step: int,
) -> int | None:
    """Leave one queued batch inside the completed-work freshness budget."""

    if running_reserve_fraction <= 0.0:
        return None
    target = max(1, int(target_groups_per_step))
    max_completed = int(math.floor(target * max(1.0, limit_steps_off_policy)))
    raw_limit = int(math.floor((max_completed - target) / running_reserve_fraction))
    return max(1, (raw_limit // max(1, worker_step)) * max(1, worker_step))


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


def _vllm_pressure(
    metrics: list[PipelineMetric], *, window_start_s: float, window_end_s: float
) -> float:
    wanted = {"vllm/num_requests_running", "vllm/num_requests_waiting_capacity"}
    rows: list[tuple[float, str, float]] = []
    for rec in metrics:
        if rec.name in wanted and math.isfinite(rec.value):
            rows.append((rec.t_s, rec.name, rec.value))
    if not rows:
        raise RuntimeError("Pipeline autotuning requires vLLM runtime metric samples.")
    by_time = _group_vllm_metric_rows(rows)
    times = sorted(t_s for t_s, values in by_time.items() if wanted <= values.keys())
    if not times:
        raise RuntimeError(
            "Pipeline autotuning requires complete vLLM running/capacity samples."
        )
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
        return capacity_wait_request_s / running_request_s
    return math.inf if capacity_wait_request_s > 0.0 else 0.0


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
