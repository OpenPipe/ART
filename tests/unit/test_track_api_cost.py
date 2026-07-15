import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art import (
    Model,
    PipelineRuntimeConfig,
    TrainableModel,
    Trajectory,
    TrajectoryGroup,
)
from art.costs import compute_sample_costs, get_model_pricing
from art.metrics import MetricsBuilder, track_api_cost
from art.pipeline_trainer.trainer import PipelineTrainer


class _OpenAIUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = 0,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = type(
            "PromptTokensDetails",
            (),
            {"cached_tokens": cached_tokens},
        )()
        self.cost = cost
        self.model_extra = model_extra


class _OpenAIResponse:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = 0,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
        model: str | None = None,
    ) -> None:
        self.usage = _OpenAIUsage(
            prompt_tokens,
            completion_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            model_extra=model_extra,
        )
        self.model = model


class _AnthropicUsage:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _AnthropicResponse:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        self.usage = _AnthropicUsage(
            input_tokens,
            output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        self.model = model


class TestTrackApiCost:
    def test_compute_sample_costs_uses_tinker_leaf_names(self) -> None:
        pricing = get_model_pricing("openai/gpt-oss-20b")
        assert pricing is not None

        metrics = compute_sample_costs(
            prompt_tokens=1_000,
            completion_tokens=2_000,
            cost_context="train",
            pricing=pricing,
        )

        assert metrics == {
            "costs/train/tinker_prefill": pytest.approx(0.00012),
            "costs/train/tinker_sample": pytest.approx(0.0006),
        }

    @pytest.mark.asyncio
    async def test_openai_cost_extraction_with_explicit_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/correctness"] == pytest.approx(0.0002)

    @pytest.mark.asyncio
    async def test_openai_cost_extraction_accounts_for_cached_tokens(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/cached_openai",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=2.0,
            completion_price_per_million=8.0,
            cached_prompt_price_per_million=0.5,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=2_000,
                completion_tokens=100,
                cached_tokens=1_500,
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/cached_openai"] == pytest.approx(0.00255)

    @pytest.mark.asyncio
    async def test_anthropic_cost_extraction_uses_registered_model_pricing(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.register_model_pricing(
            "anthropic/test-judge",
            prompt_per_million=5.0,
            completion_per_million=7.0,
        )

        @track_api_cost(
            source="llm_judge/faithfulness",
            provider="anthropic",
            model_name="anthropic/test-judge",
        )
        async def _judge() -> _AnthropicResponse:
            return _AnthropicResponse(input_tokens=40, output_tokens=60)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/faithfulness"] == pytest.approx(0.00062)

    @pytest.mark.asyncio
    async def test_anthropic_cost_extraction_accounts_for_cache_write_and_read(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="eval")
        builder.register_model_pricing(
            "anthropic/claude-sonnet-4-6",
            prompt_per_million=3.0,
            completion_per_million=15.0,
            cache_creation_per_million=3.75,
            cache_read_per_million=0.30,
        )

        @track_api_cost(
            source="llm_judge/anthropic_cache",
            provider="anthropic",
            model_name="anthropic/claude-sonnet-4-6",
        )
        async def _judge() -> _AnthropicResponse:
            return _AnthropicResponse(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=1_000,
                cache_read_input_tokens=500,
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/eval/llm_judge/anthropic_cache"] == pytest.approx(0.00495)

    @pytest.mark.asyncio
    async def test_direct_usage_cost_is_used_before_provider_estimation(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/openrouter_usage_cost",
            provider="openrouter",
            model_name="openrouter/openai/gpt-4.1-mini",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=100,
                completion_tokens=50,
                cost=1.68e-05,
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/openrouter_usage_cost"] == pytest.approx(
            1.68e-05
        )

    @pytest.mark.asyncio
    async def test_direct_model_extra_cost_is_used_when_usage_cost_missing(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/openrouter_model_extra_cost",
            provider="openrouter",
            model_name="openrouter/openai/gpt-4.1-mini",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=100,
                completion_tokens=50,
                model_extra={"cost": 1.68e-05},
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics[
            "costs/train/llm_judge/openrouter_model_extra_cost"
        ] == pytest.approx(1.68e-05)

    @pytest.mark.asyncio
    async def test_explicit_model_name_uses_global_pricing(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="train")
        pricing = get_model_pricing("openai/gpt-oss-20b")
        assert pricing is not None

        @track_api_cost(
            source="llm_judge/global_pricing",
            provider="openai",
            model_name="openai/gpt-oss-20b",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=1_000,
                completion_tokens=2_000,
                model="gpt-oss-20b",
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        expected = compute_sample_costs(
            prompt_tokens=1_000,
            completion_tokens=2_000,
            cost_context="train",
            pricing=pricing,
        )
        assert metrics["costs/train/llm_judge/global_pricing"] == pytest.approx(
            expected["costs/train/tinker_prefill"]
            + expected["costs/train/tinker_sample"]
        )

    @pytest.mark.asyncio
    async def test_explicit_model_name_uses_registered_pricing(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="eval")
        builder.register_model_pricing(
            "anthropic/test-judge",
            prompt_per_million=1.5,
            completion_per_million=2.5,
        )

        @track_api_cost(
            source="llm_judge/provider_resolution",
            provider="anthropic",
            model_name="anthropic/test-judge",
        )
        async def _judge() -> _AnthropicResponse:
            return _AnthropicResponse(
                input_tokens=400,
                output_tokens=600,
                model="test-judge",
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/eval/llm_judge/provider_resolution"] == pytest.approx(
            0.0021
        )

    @pytest.mark.asyncio
    async def test_explicit_model_name_uses_litellm_pricing_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        builder = MetricsBuilder(cost_context="train")

        def _fake_get_model_info(model_name: str) -> dict[str, float]:
            assert model_name == "openai/fallback-model"
            return {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 2.5e-07,
            }

        monkeypatch.setattr(litellm, "get_model_info", _fake_get_model_info)

        @track_api_cost(
            source="llm_judge/litellm_fallback",
            provider="openai",
            model_name="openai/fallback-model",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=80,
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        expected = ((20 * 2.5) + (80 * 0.25) + (50 * 15.0)) / 1_000_000
        assert metrics["costs/train/llm_judge/litellm_fallback"] == pytest.approx(
            expected
        )

    @pytest.mark.asyncio
    async def test_explicit_model_name_does_not_depend_on_response_model(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/snapshot",
            provider="openai",
            model_name="openai/gpt-4.1",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(
                prompt_tokens=1_000,
                completion_tokens=100,
                cached_tokens=800,
                model="gpt-4.1-2025-04-14",
            )

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        expected = ((200 * 2.0) + (800 * 0.5) + (100 * 8.0)) / 1_000_000
        assert metrics["costs/train/llm_judge/snapshot"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_decorator_fails_fast_without_model_aware_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/missing_pricing",
            provider="openai",
            model_name="openai/missing-pricing-model",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=10, completion_tokens=20)

        token = builder.activate()
        try:
            with pytest.raises(ValueError, match="No pricing configured"):
                await _judge()
        finally:
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_custom_extractor_takes_precedence(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.register_cost_extractor("openai", lambda _response: 0.75)

        @track_api_cost(
            source="llm_judge/custom",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=1, completion_tokens=1)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/custom"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_decorator_noops_without_active_builder(self) -> None:
        @track_api_cost(
            source="llm_judge/no_context",
            provider="openai",
            model_name="openai/gpt-4.1",
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=10, completion_tokens=20)

        result = await _judge()
        assert isinstance(result, _OpenAIResponse)

    @pytest.mark.asyncio
    async def test_for_cost_context_routes_to_eval_and_shares_state(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        eval_builder = builder.for_cost_context("eval")

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        token = eval_builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await eval_builder.flush()
        assert metrics["costs/eval/llm_judge/correctness"] == pytest.approx(0.0002)


class TestTrackApiCostIntegration:
    @pytest.mark.asyncio
    async def test_model_log_emits_train_and_eval_costs(self, tmp_path: Path) -> None:
        model = Model(
            name="metrics-cost-test",
            project="metrics-cost-test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _train_judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        @track_api_cost(
            source="llm_judge/factuality",
            provider="anthropic",
            model_name="anthropic/claude-sonnet-4-6",
            prompt_price_per_million=3.0,
            completion_price_per_million=4.0,
        )
        async def _eval_judge() -> _AnthropicResponse:
            return _AnthropicResponse(input_tokens=40, output_tokens=10)

        train_token = model.activate_metrics_context("train")
        try:
            await _train_judge()
        finally:
            train_token.var.reset(train_token)

        await model.log(trajectories=None, split="train", step=1, metrics={})

        eval_token = model.activate_metrics_context("eval")
        try:
            await _eval_judge()
        finally:
            eval_token.var.reset(eval_token)

        await model.log(trajectories=None, split="val", step=2, metrics={})

        history_path = (
            tmp_path
            / "metrics-cost-test"
            / "models"
            / "metrics-cost-test"
            / "history.jsonl"
        )
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["costs/train/llm_judge/correctness"] == pytest.approx(0.0002)
        assert second["costs/eval/llm_judge/factuality"] == pytest.approx(0.00016)
        assert second["costs/cum/all"] == pytest.approx(0.00036)

    @pytest.mark.asyncio
    async def test_model_log_keeps_pending_train_and_eval_costs_isolated(
        self, tmp_path: Path
    ) -> None:
        model = Model(
            name="metrics-cost-isolation-test",
            project="metrics-cost-isolation-test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _train_judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        @track_api_cost(
            source="llm_judge/factuality",
            provider="anthropic",
            model_name="anthropic/claude-sonnet-4-6",
            prompt_price_per_million=3.0,
            completion_price_per_million=4.0,
        )
        async def _eval_judge() -> _AnthropicResponse:
            return _AnthropicResponse(input_tokens=40, output_tokens=10)

        train_token = model.activate_metrics_context("train")
        try:
            await _train_judge()
        finally:
            train_token.var.reset(train_token)

        eval_token = model.activate_metrics_context("eval")
        try:
            await _eval_judge()
        finally:
            eval_token.var.reset(eval_token)

        await model.log(trajectories=None, split="val", step=1, metrics={})
        await model.log(trajectories=None, split="train", step=1, metrics={})

        history_path = (
            tmp_path
            / "metrics-cost-isolation-test"
            / "models"
            / "metrics-cost-isolation-test"
            / "history.jsonl"
        )
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert "costs/eval/llm_judge/factuality" in first
        assert "costs/train/llm_judge/correctness" not in first
        assert first["costs/cum/all"] == pytest.approx(0.00016)

        assert "costs/train/llm_judge/correctness" in second
        assert "costs/eval/llm_judge/factuality" not in second
        assert second["costs/cum/all"] == pytest.approx(0.00036)

    @pytest.mark.asyncio
    async def test_pipeline_trainer_activates_train_context_for_rollouts(
        self, tmp_path: Path
    ) -> None:
        model = TrainableModel(
            name="pipeline-context-test",
            project="pipeline-context-test",
            base_model="test-model",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        backend = MagicMock()
        observed_contexts: list[str] = []

        async def rollout_fn(
            _model: TrainableModel,
            _scenario: dict,
            _config: dict,
        ) -> TrajectoryGroup:
            observed_contexts.append(MetricsBuilder.get_active().cost_context)
            return TrajectoryGroup(
                [
                    Trajectory(
                        reward=1.0,
                        messages_and_choices=[
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ],
                    )
                ]
            )

        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=rollout_fn,
            scenarios=[{"metadata": {"scenario_id": "s1"}}],
            config={},
            pipeline=PipelineRuntimeConfig(
                num_rollout_workers=1,
                min_batch_size=1,
                max_batch_size=1,
            ),
            eval_fn=None,
        )
        trainer._output_queue = asyncio.Queue()

        await trainer._rollout_worker(worker_id=0)

        assert observed_contexts == ["train"]

    @pytest.mark.asyncio
    async def test_pipeline_trainer_activates_eval_context_for_eval_fn(
        self, tmp_path: Path
    ) -> None:
        model = TrainableModel(
            name="pipeline-eval-context-test",
            project="pipeline-eval-context-test",
            base_model="test-model",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        backend = MagicMock()
        observed_contexts: list[str] = []

        @asynccontextmanager
        async def exact_adapter_lease(*_args: object):
            yield

        backend.exact_adapter_lease = exact_adapter_lease

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            model_name="openai/gpt-4.1",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge_call() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        async def eval_fn(
            _model: TrainableModel,
            _step: int,
            _config: dict,
        ) -> list[Trajectory]:
            observed_contexts.append(MetricsBuilder.get_active().cost_context)
            await _judge_call()
            return [
                Trajectory(
                    reward=1.0,
                    messages_and_choices=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                )
            ]

        async def rollout_fn(
            _model: TrainableModel,
            _scenario: dict,
            _config: dict,
        ) -> TrajectoryGroup:
            return TrajectoryGroup([])

        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=rollout_fn,
            scenarios=[],
            config={},
            pipeline=PipelineRuntimeConfig(
                num_rollout_workers=1,
                min_batch_size=1,
                max_batch_size=1,
            ),
            eval_fn=eval_fn,
        )

        await trainer._run_eval(step=1)

        assert observed_contexts == ["eval"]

        history_path = (
            tmp_path
            / "pipeline-eval-context-test"
            / "models"
            / "pipeline-eval-context-test"
            / "history.jsonl"
        )
        with open(history_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert any("costs/eval/llm_judge/correctness" in row for row in rows)
        assert any("time/step_eval_s" in row for row in rows)
