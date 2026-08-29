from __future__ import annotations

import math

from .config import (
    InferenceLoadObservation,
    PipelineAutotuneConfig,
    RolloutSupplyEvidence,
    RolloutSupplyObserverState,
    RolloutSupplyTrial,
    RolloutSupplyWindow,
)


class RolloutSupplyInferenceObserver:
    """Infer marginal serving capacity from bounded rollout-worker probes."""

    def __init__(
        self,
        config: PipelineAutotuneConfig,
        state: RolloutSupplyObserverState | None = None,
    ) -> None:
        self.config = config
        self.state = state or RolloutSupplyObserverState()
        self._last: tuple[RolloutSupplyWindow, InferenceLoadObservation] | None = None

    def observe(self, window: RolloutSupplyWindow) -> InferenceLoadObservation:
        if self._last is not None and self._last[0] == window:
            return self._last[1]
        if self.state.windows and window.end_step <= self.state.windows[-1].end_step:
            raise RuntimeError("rollout-supply windows must be strictly ordered")
        previous = self.state.windows[-1] if self.state.windows else None
        self.state.windows.append(window)
        self.state.windows = self.state.windows[
            -self.config.black_box_history_windows :
        ]

        trial = self.state.trial
        if trial is not None and window.workers != trial.trial_workers:
            trial = None
        if trial is None and previous is not None and window.workers > previous.workers:
            trial = RolloutSupplyTrial(
                baseline_workers=previous.workers,
                trial_workers=window.workers,
                started_step=window.end_step,
            )
        self.state.trial = trial

        rate = self._rate(window.workers)
        if trial is not None:
            result = self._trial_observation(trial, window, rate)
        else:
            result = self._evidence_observation(window, rate)
        self._last = window, result
        return result

    def snapshot(self) -> RolloutSupplyObserverState:
        return self.state.model_copy(deep=True)

    def _trial_observation(
        self,
        trial: RolloutSupplyTrial,
        window: RolloutSupplyWindow,
        rate: float,
    ) -> InferenceLoadObservation:
        if window.end_step == trial.started_step:
            return InferenceLoadObservation(
                state="settling", confidence=0.0, supply_groups_per_s=rate
            )
        baseline_rate, baseline_events = self._weighted_rate(
            workers=trial.baseline_workers,
            before_or_at_step=trial.started_step,
        )
        trial_rate, trial_events = self._weighted_rate(
            workers=trial.trial_workers,
            after_step=trial.started_step,
        )
        if baseline_rate <= 0.0 or trial_rate <= 0.0:
            return InferenceLoadObservation(
                state="settling", confidence=0.0, supply_groups_per_s=rate
            )
        ratio = trial_rate / baseline_rate
        threshold = 1.0 + self.config.black_box_min_supply_gain_ratio
        standard_error = math.sqrt(
            1.0 / max(baseline_events, 1.0) + 1.0 / max(trial_events, 1.0)
        )
        z_score = (math.log(ratio) - math.log(threshold)) / standard_error
        gain_probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        decision_probability = self.config.black_box_decision_probability
        if gain_probability >= decision_probability:
            self.state.successful = RolloutSupplyEvidence(
                workers=trial.trial_workers,
                end_step=window.end_step,
                confidence=gain_probability,
            )
            self.state.saturated = None
            self.state.trial = None
            return InferenceLoadObservation(
                state="underloaded",
                confidence=gain_probability,
                supply_groups_per_s=rate,
                trial_ratio=ratio,
            )
        failure_probability = 1.0 - gain_probability
        if failure_probability >= decision_probability:
            self.state.saturated = RolloutSupplyEvidence(
                workers=trial.baseline_workers,
                end_step=window.end_step,
                confidence=failure_probability,
            )
            self.state.trial = None
            return InferenceLoadObservation(
                state="saturated",
                confidence=failure_probability,
                supply_groups_per_s=rate,
                trial_ratio=ratio,
                revert_to_workers=trial.baseline_workers,
            )
        return InferenceLoadObservation(
            state="settling",
            confidence=max(gain_probability, failure_probability),
            supply_groups_per_s=rate,
            trial_ratio=ratio,
        )

    def _evidence_observation(
        self, window: RolloutSupplyWindow, rate: float
    ) -> InferenceLoadObservation:
        saturated = self._decayed_evidence(self.state.saturated, window.end_step)
        if saturated is not None and window.workers >= saturated.workers:
            return InferenceLoadObservation(
                state="saturated",
                confidence=saturated.confidence,
                supply_groups_per_s=rate,
            )
        successful = self._decayed_evidence(self.state.successful, window.end_step)
        if successful is not None and window.workers <= successful.workers:
            return InferenceLoadObservation(
                state="underloaded",
                confidence=successful.confidence,
                supply_groups_per_s=rate,
            )
        return InferenceLoadObservation(
            state="unknown", confidence=0.0, supply_groups_per_s=rate
        )

    def _decayed_evidence(
        self, evidence: RolloutSupplyEvidence | None, end_step: int
    ) -> RolloutSupplyEvidence | None:
        if evidence is None:
            return None
        windows = sum(
            value.end_step > evidence.end_step for value in self.state.windows
        )
        confidence = evidence.confidence * 0.5 ** (
            windows / self.config.black_box_decay_half_life_windows
        )
        if confidence < 0.5:
            return None
        return evidence.model_copy(
            update={"confidence": confidence, "end_step": end_step}
        )

    def _rate(self, workers: int) -> float:
        rate, _events = self._weighted_rate(workers=workers)
        return rate

    def _weighted_rate(
        self,
        *,
        workers: int,
        before_or_at_step: int | None = None,
        after_step: int | None = None,
    ) -> tuple[float, float]:
        selected = [
            (index, window)
            for index, window in enumerate(reversed(self.state.windows))
            if window.workers == workers
            and (before_or_at_step is None or window.end_step <= before_or_at_step)
            and (after_step is None or window.end_step > after_step)
        ]
        weighted_duration = 0.0
        weighted_events = 0.0
        for age, window in selected:
            weight = 0.5 ** (age / self.config.black_box_decay_half_life_windows)
            weighted_duration += weight * window.duration_s
            weighted_events += weight * window.completed_groups
        return (
            weighted_events / weighted_duration if weighted_duration > 0.0 else 0.0,
            weighted_events,
        )
