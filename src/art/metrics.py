from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import time
from typing import Any, Literal

import pydantic

from .api_costs import (
    CostExtractor,
    TokenPricing,
    extract_api_cost,
    normalize_model_name,
    normalize_provider,
)

_active_builder: ContextVar["MetricsBuilder"] = ContextVar("_active_metrics_builder")

_HIERARCHICAL_SECTIONS = {"costs", "time", "data"}
_THROUGHPUT_IDLE_MAPPINGS = {
    "throughput/step_trainer_idle_s": "throughput/cum/trainer_idle_s",
    "throughput/step_actor_idle_s": "throughput/cum/actor_idle_s",
}


class MetricDefinition(pydantic.BaseModel):
    key: str
    title: str
    description: str
    kind: Literal["counter", "duration", "gauge", "rate", "ratio", "score"]
    unit: str | None = None
    higher_is_better: bool | None = None
    dashboard_default: bool = False
    score_component: bool = False


PIPELINE_RL_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="objective/score_default",
        title="Default score",
        description=(
            "accepted trainable assistant tokens per second times freshness "
            "discount times batch-size discount"
        ),
        kind="score",
        higher_is_better=True,
        dashboard_default=True,
        score_component=True,
    ),
    MetricDefinition(
        key="objective/score_raw",
        title="Raw accepted throughput",
        description=(
            "accepted trainable assistant tokens per second before "
            "sample-efficiency discounts"
        ),
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
        score_component=True,
    ),
    MetricDefinition(
        key="objective/score_freshness",
        title="Freshness-adjusted score",
        description="accepted trainable assistant tokens per second times freshness discount",
        kind="score",
        higher_is_better=True,
        score_component=True,
    ),
    MetricDefinition(
        key="sample_efficiency/freshness_discount",
        title="Freshness discount",
        description="1 / (1 + token-weighted mean policy age in train steps)",
        kind="ratio",
        higher_is_better=True,
        score_component=True,
    ),
    MetricDefinition(
        key="sample_efficiency/batch_discount",
        title="Batch-size discount",
        description=(
            "optional penalty for increasing accepted groups per train step "
            "above a configured reference"
        ),
        kind="ratio",
        higher_is_better=True,
        score_component=True,
    ),
    MetricDefinition(
        key="offpolicy/token_weighted_policy_age_steps",
        title="Token-weighted policy age",
        description=(
            "completion-token-weighted mean train-step age of the policy that "
            "generated accepted data"
        ),
        kind="gauge",
        unit="steps",
        higher_is_better=False,
        score_component=True,
    ),
    MetricDefinition(
        key="throughput/accepted_train_tok_per_s",
        title="Accepted train tokens per second",
        description=(
            "accepted trainable assistant tokens divided by PipelineTrainer "
            "step wall time"
        ),
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
        dashboard_default=True,
        score_component=True,
    ),
    MetricDefinition(
        key="data/step_trainer_assistant_tokens",
        title="Accepted assistant tokens",
        description="trainable assistant tokens in accepted groups for this train step",
        kind="counter",
        unit="tokens",
        higher_is_better=None,
        score_component=True,
    ),
    MetricDefinition(
        key="data/step_padding_ratio",
        title="Padding ratio",
        description=(
            "physical packed-token padding divided by packed sequence length "
            "for this step"
        ),
        kind="ratio",
        higher_is_better=False,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="megatron/packed_train_tok_per_s",
        title="Megatron packed train tokens per second",
        description=(
            "physical packed training-token throughput reported by the Megatron worker"
        ),
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="megatron/non_padding_train_tok_per_s",
        title="Megatron non-padding train tokens per second",
        description=(
            "non-padding training-token throughput reported by the Megatron worker"
        ),
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
    ),
    MetricDefinition(
        key="vllm/prompt_tok_per_s",
        title="vLLM prompt tokens per second",
        description="vLLM prompt prefill throughput over time",
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/completion_tok_per_s",
        title="vLLM completion tokens per second",
        description="vLLM decode throughput over time",
        kind="rate",
        unit="tok/s",
        higher_is_better=True,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/num_requests_running",
        title="vLLM running requests",
        description="number of admitted requests currently running in vLLM",
        kind="gauge",
        unit="requests",
        higher_is_better=None,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/num_requests_waiting",
        title="vLLM waiting requests",
        description="number of queued requests waiting for vLLM admission",
        kind="gauge",
        unit="requests",
        higher_is_better=None,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/num_requests_waiting_capacity",
        title="vLLM capacity-waiting requests",
        description="number of queued requests waiting because vLLM capacity is exhausted",
        kind="gauge",
        unit="requests",
        higher_is_better=False,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/max_num_seqs",
        title="vLLM max running sequences",
        description="configured maximum number of sequences vLLM can schedule per iteration",
        kind="gauge",
        unit="requests",
        higher_is_better=None,
    ),
    MetricDefinition(
        key="vllm/max_num_batched_tokens",
        title="vLLM max batched tokens",
        description="configured token scheduling budget per vLLM iteration",
        kind="gauge",
        unit="tokens",
        higher_is_better=None,
    ),
    MetricDefinition(
        key="vllm/max_num_scheduled_tokens",
        title="vLLM max scheduled tokens",
        description="configured maximum scheduled tokens per vLLM iteration",
        kind="gauge",
        unit="tokens",
        higher_is_better=None,
    ),
    MetricDefinition(
        key="vllm/max_model_len",
        title="vLLM max model length",
        description="configured maximum sequence length served by vLLM",
        kind="gauge",
        unit="tokens",
        higher_is_better=None,
    ),
    MetricDefinition(
        key="vllm/prefix_cache_hit_rate",
        title="vLLM prefix cache hit rate",
        description="delta prefix cache hits divided by delta prefix cache queries",
        kind="ratio",
        higher_is_better=True,
        dashboard_default=True,
    ),
    MetricDefinition(
        key="vllm/kv_cache_usage_perc",
        title="vLLM KV cache usage",
        description="fraction of vLLM KV cache blocks in use",
        kind="ratio",
        higher_is_better=None,
    ),
    MetricDefinition(
        key="vllm/num_preemptions_total",
        title="vLLM preemptions",
        description="cumulative vLLM request preemptions",
        kind="counter",
        higher_is_better=False,
    ),
    MetricDefinition(
        key="pipeline_trainer/step_collect_batch_s",
        title="Pipeline collect-batch time",
        description="time spent waiting for enough accepted rollout groups for a train step",
        kind="duration",
        unit="s",
        higher_is_better=False,
    ),
    MetricDefinition(
        key="queue/freshness_pressure",
        title="Queue freshness pressure",
        description="predicted queued policy age divided by the active off-policy limit",
        kind="ratio",
        higher_is_better=False,
    ),
    MetricDefinition(
        key="queue/predicted_stale_fraction",
        title="Predicted stale queue fraction",
        description="fraction of queued groups predicted stale for the next train step",
        kind="ratio",
        higher_is_better=False,
    ),
)

PIPELINE_RL_DASHBOARD_DEFAULT_METRICS = tuple(
    definition.key
    for definition in PIPELINE_RL_METRIC_DEFINITIONS
    if definition.dashboard_default
)
PIPELINE_RL_SCORE_METRICS = tuple(
    definition.key
    for definition in PIPELINE_RL_METRIC_DEFINITIONS
    if definition.score_component
)


def is_cumulative_metric_key(key: str) -> bool:
    parts = key.split("/", 2)
    return len(parts) >= 2 and parts[1] == "cum"


def is_builder_managed_metric(key: str) -> bool:
    return key.startswith(("costs/", "time/step_", "data/step_", "throughput/step_"))


def to_cumulative_metric_key(key: str) -> str:
    if is_cumulative_metric_key(key):
        raise ValueError(f"Metric key '{key}' is already cumulative.")

    section, rest = key.split("/", 1)
    if rest.startswith("step_"):
        rest = rest[len("step_") :]
    return f"{section}/cum/{rest}"


@dataclass
class _PendingMetricsState:
    step_buffer: dict[str, float]
    pending_scenario_ids: set[str]


@dataclass
class _SharedMetricsState:
    lock: asyncio.Lock
    pending_by_scope: dict[str, _PendingMetricsState]
    cum_state: dict[str, float]
    unique_scenario_ids: set[str]
    cost_extractors: dict[str, CostExtractor]
    model_pricing: dict[str, TokenPricing]


def _new_pending_metrics_state() -> _PendingMetricsState:
    return _PendingMetricsState(step_buffer={}, pending_scenario_ids=set())


def _new_shared_metrics_state() -> _SharedMetricsState:
    return _SharedMetricsState(
        lock=asyncio.Lock(),
        pending_by_scope={},
        cum_state={},
        unique_scenario_ids=set(),
        cost_extractors={},
        model_pricing={},
    )


class MetricsBuilder:
    """Build and accumulate step-level metrics for logging."""

    def __init__(
        self,
        cost_context: str,
        *,
        _shared_state: _SharedMetricsState | None = None,
        _buffer_scope: str | None = None,
    ) -> None:
        if not cost_context:
            raise ValueError("cost_context must be non-empty")

        self.cost_context = cost_context
        self._buffer_scope = (
            _buffer_scope if _buffer_scope is not None else cost_context
        )
        self._shared_state = (
            _shared_state if _shared_state is not None else _new_shared_metrics_state()
        )

    def add_cost(self, path: str, usd: float) -> None:
        if not path:
            raise ValueError("Cost path must be non-empty")
        full_key = f"costs/{path}"
        self.add_metric(full_key, float(usd))

    def add_response_cost(
        self,
        source: str,
        response: Any,
        *,
        provider: str,
        model_name: str,
        prompt_price_per_million: float | None = None,
        completion_price_per_million: float | None = None,
        cached_prompt_price_per_million: float | None = None,
        cache_creation_price_per_million: float | None = None,
        cache_read_price_per_million: float | None = None,
    ) -> float | None:
        normalized_source = source.strip("/")
        if not normalized_source:
            raise ValueError("source must be non-empty")

        cost = extract_api_cost(
            response,
            provider=provider,
            model_name=model_name,
            prompt_price_per_million=prompt_price_per_million,
            completion_price_per_million=completion_price_per_million,
            cached_prompt_price_per_million=cached_prompt_price_per_million,
            cache_creation_price_per_million=cache_creation_price_per_million,
            cache_read_price_per_million=cache_read_price_per_million,
            cost_extractors=self._shared_state.cost_extractors,
            model_pricing=self._shared_state.model_pricing,
        )
        if cost is None:
            return None

        self.add_cost(f"{self.cost_context}/{normalized_source}", cost)
        return cost

    def add_metric(self, key: str, value: float) -> None:
        if "/" not in key:
            raise ValueError("Metric key must include a section prefix")
        self._validate_and_add(key, float(value))

    def add_data(
        self,
        step_num_scenarios: int | None = None,
        step_actor_tokens: int | None = None,
        scenario_ids: list[str] | None = None,
    ) -> None:
        if step_num_scenarios is not None:
            self.add_metric("data/step_num_scenarios", float(step_num_scenarios))
        if step_actor_tokens is not None:
            self.add_metric("data/step_actor_tokens", float(step_actor_tokens))
        if scenario_ids is not None:
            self._pending_state().pending_scenario_ids.update(
                str(scenario_id) for scenario_id in scenario_ids
            )

    def add_user_timing(
        self,
        step_wall_s: float | None = None,
        step_actor_s: float | None = None,
        step_eval_s: float | None = None,
    ) -> None:
        if step_wall_s is not None:
            self.add_metric("time/step_wall_s", float(step_wall_s))
        if step_actor_s is not None:
            self.add_metric("time/step_actor_s", float(step_actor_s))
        if step_eval_s is not None:
            self.add_metric("time/step_eval_s", float(step_eval_s))

    def add_idle_times(
        self,
        step_trainer_idle_s: float | None = None,
        step_actor_idle_s: float | None = None,
    ) -> None:
        if step_trainer_idle_s is not None:
            self.add_metric(
                "throughput/step_trainer_idle_s",
                float(step_trainer_idle_s),
            )
        if step_actor_idle_s is not None:
            self.add_metric("throughput/step_actor_idle_s", float(step_actor_idle_s))

    @contextmanager
    def measure(self, key: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add_metric(key, time.monotonic() - started)

    async def flush(self) -> dict[str, float]:
        async with self._shared_state.lock:
            pending_state = self._pending_state()
            result = dict(pending_state.step_buffer)
            pending_scenario_ids = set(pending_state.pending_scenario_ids)
            cost_metrics = {
                key: value for key, value in result.items() if key.startswith("costs/")
            }
            result.update(self._compute_rollups(cost_metrics))

            for key, value in list(result.items()):
                section = key.split("/", 1)[0]
                if section not in _HIERARCHICAL_SECTIONS:
                    continue
                cum_key = to_cumulative_metric_key(key)
                next_value = self._shared_state.cum_state.get(cum_key, 0.0) + value
                self._shared_state.cum_state[cum_key] = next_value
                result[cum_key] = next_value

            if pending_scenario_ids:
                self._shared_state.unique_scenario_ids.update(pending_scenario_ids)
                result["data/cum/num_unique_scenarios"] = float(
                    len(self._shared_state.unique_scenario_ids)
                )

            self._update_throughput_metrics(result)
            pending_state.step_buffer.clear()
            pending_state.pending_scenario_ids.clear()
            return result

    def activate(self) -> Token["MetricsBuilder"]:
        return _active_builder.set(self)

    @contextmanager
    def activate_context(self):
        token = self.activate()
        try:
            yield self
        finally:
            token.var.reset(token)

    @staticmethod
    def get_active() -> "MetricsBuilder":
        return _active_builder.get()

    def for_cost_context(
        self, cost_context: str, *, buffer_scope: str | None = None
    ) -> "MetricsBuilder":
        normalized_cost_context = cost_context.strip()
        if not normalized_cost_context:
            raise ValueError("cost_context must be non-empty")
        normalized_buffer_scope = (
            buffer_scope.strip()
            if buffer_scope is not None
            else normalized_cost_context
        )
        if not normalized_buffer_scope:
            raise ValueError("buffer_scope must be non-empty")
        if (
            normalized_cost_context == self.cost_context
            and normalized_buffer_scope == self._buffer_scope
        ):
            return self
        return MetricsBuilder(
            cost_context=normalized_cost_context,
            _shared_state=self._shared_state,
            _buffer_scope=normalized_buffer_scope,
        )

    def register_cost_extractor(self, provider: str, extractor: CostExtractor) -> None:
        normalized_provider = normalize_provider(provider)
        if normalized_provider is None:
            raise ValueError("provider must be non-empty")
        self._shared_state.cost_extractors[normalized_provider] = extractor

    def register_model_pricing(
        self,
        model_name: str,
        *,
        prompt_per_million: float,
        completion_per_million: float,
        cached_prompt_per_million: float | None = None,
        cache_creation_per_million: float | None = None,
        cache_read_per_million: float | None = None,
    ) -> None:
        normalized_model_name = normalize_model_name(model_name)
        if not normalized_model_name:
            raise ValueError("model_name must be non-empty")
        self._shared_state.model_pricing[normalized_model_name] = TokenPricing(
            prompt_per_million=float(prompt_per_million),
            completion_per_million=float(completion_per_million),
            cached_prompt_per_million=(
                float(cached_prompt_per_million)
                if cached_prompt_per_million is not None
                else None
            ),
            cache_creation_per_million=(
                float(cache_creation_per_million)
                if cache_creation_per_million is not None
                else None
            ),
            cache_read_per_million=(
                float(cache_read_per_million)
                if cache_read_per_million is not None
                else None
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "cum_state": dict(self._shared_state.cum_state),
            "unique_scenario_ids": list(self._shared_state.unique_scenario_ids),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raw_cum_state = state.get("cum_state", {})
        raw_unique_ids = state.get("unique_scenario_ids", [])
        restored_cum_state = {str(k): float(v) for k, v in raw_cum_state.items()}
        restored_unique_ids = {str(v) for v in raw_unique_ids}

        self._shared_state.cum_state.clear()
        self._shared_state.cum_state.update(restored_cum_state)
        self._shared_state.unique_scenario_ids.clear()
        self._shared_state.unique_scenario_ids.update(restored_unique_ids)
        self._shared_state.pending_by_scope.clear()

    def _validate_and_add(self, key: str, value: float) -> None:
        if is_cumulative_metric_key(key):
            raise ValueError(
                f"Metric key '{key}' uses the reserved cumulative namespace."
            )

        pending_state = self._pending_state()
        for existing_key in pending_state.step_buffer:
            if existing_key == key:
                continue
            if existing_key.startswith(f"{key}/"):
                raise ValueError(
                    f"Cannot log '{key}' as a leaf: it is an ancestor of '{existing_key}'."
                )
            if key.startswith(f"{existing_key}/"):
                raise ValueError(
                    f"Cannot log '{key}' as a leaf: '{existing_key}' is already a leaf ancestor."
                )

        pending_state.step_buffer[key] = pending_state.step_buffer.get(key, 0.0) + value

    def _pending_state(self) -> _PendingMetricsState:
        pending_state = self._shared_state.pending_by_scope.get(self._buffer_scope)
        if pending_state is None:
            pending_state = _new_pending_metrics_state()
            self._shared_state.pending_by_scope[self._buffer_scope] = pending_state
        return pending_state

    def _compute_rollups(self, cost_metrics: dict[str, float]) -> dict[str, float]:
        if not cost_metrics:
            return {}

        all_parents: set[str] = set()
        for key in cost_metrics:
            parts = key.split("/")
            for depth in range(2, len(parts)):
                all_parents.add("/".join(parts[:depth]))

        rollups: dict[str, float] = {}
        for parent in all_parents:
            prefix = f"{parent}/"
            rollups[parent] = sum(
                value for key, value in cost_metrics.items() if key.startswith(prefix)
            )

        top_level_children = {key.split("/")[1] for key in cost_metrics}
        costs_all = 0.0
        for child_name in top_level_children:
            child_key = f"costs/{child_name}"
            if child_key in rollups:
                costs_all += rollups[child_key]
            else:
                costs_all += cost_metrics[child_key]
        rollups["costs/all"] = costs_all

        return rollups

    def _update_throughput_metrics(self, result: dict[str, float]) -> None:
        for step_key, cum_key in _THROUGHPUT_IDLE_MAPPINGS.items():
            if step_key not in result:
                continue
            next_value = (
                self._shared_state.cum_state.get(cum_key, 0.0) + result[step_key]
            )
            self._shared_state.cum_state[cum_key] = next_value
            result[cum_key] = next_value

        if (
            "data/step_trainer_tokens" in result
            or "data/step_trainer_assistant_tokens" in result
            or "time/step_trainer_s" in result
        ):
            trainer_tokens = self._shared_state.cum_state.get(
                "data/cum/trainer_assistant_tokens"
            )
            if trainer_tokens is None:
                trainer_tokens = self._shared_state.cum_state.get(
                    "data/cum/trainer_tokens"
                )
            trainer_seconds = self._shared_state.cum_state.get("time/cum/trainer_s")
            if (
                trainer_tokens is not None
                and trainer_seconds is not None
                and trainer_seconds > 0
            ):
                result["throughput/avg_trainer_tok_per_s"] = (
                    trainer_tokens / trainer_seconds
                )

        if "data/step_actor_tokens" in result or "time/step_actor_s" in result:
            actor_tokens = self._shared_state.cum_state.get("data/cum/actor_tokens")
            actor_seconds = self._shared_state.cum_state.get("time/cum/actor_s")
            if (
                actor_tokens is not None
                and actor_seconds is not None
                and actor_seconds > 0
            ):
                result["throughput/avg_actor_tok_per_s"] = actor_tokens / actor_seconds


from .api_costs import track_api_cost
