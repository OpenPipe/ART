import importlib
from typing import Any

import pytest

from art import TrainableModel
from art.costs import build_cost_calculator, get_model_pricing


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        num_choices: int = 1,
    ) -> None:
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self.choices = [object() for _ in range(num_choices)]


class _FakeCompletions:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def create(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return self._response


def _patch_async_openai(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> None:
    model_module = importlib.import_module("art.model")

    class _FakeAsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = type(
                "FakeChat",
                (),
                {"completions": _FakeCompletions(response)},
            )()

        def with_options(self, *args: Any, **kwargs: Any) -> "_FakeAsyncOpenAI":
            return self

    monkeypatch.setattr(model_module, "AsyncOpenAI", _FakeAsyncOpenAI)


def _build_model() -> TrainableModel:
    pricing = get_model_pricing("openai/gpt-oss-20b")
    assert pricing is not None

    model = TrainableModel(
        name="test-run",
        project="test-project",
        base_model="openai/gpt-oss-20b",
    )
    model.inference_api_key = "test-key"
    model.inference_base_url = "http://example.test/v1"
    model.set_cost_calculator(build_cost_calculator(pricing))
    return model


class TestModelOpenAIClientCosts:
    @pytest.mark.asyncio
    async def test_openai_client_automatically_logs_train_tinker_costs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_async_openai(monkeypatch, _FakeResponse(1_000, 2_000))
        model = _build_model()
        builder = model.metrics_builder("train")

        with builder.activate_context():
            await model.openai_client().chat.completions.create(
                model=model.get_inference_name(),
                messages=[{"role": "user", "content": "hello"}],
            )

        metrics = await builder.flush()
        assert metrics["costs/train/tinker_prefill"] == pytest.approx(0.00012)
        assert metrics["costs/train/tinker_sample"] == pytest.approx(0.0006)
        assert metrics["costs/train"] == pytest.approx(0.00072)

    @pytest.mark.asyncio
    async def test_openai_client_automatically_logs_eval_tinker_costs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_async_openai(monkeypatch, _FakeResponse(500, 250))
        model = _build_model()
        builder = model.metrics_builder("eval")

        with builder.activate_context():
            await model.openai_client().chat.completions.create(
                model=model.get_inference_name(),
                messages=[{"role": "user", "content": "hello"}],
            )

        metrics = await builder.flush()
        assert metrics["costs/eval/tinker_prefill"] == pytest.approx(0.00006)
        assert metrics["costs/eval/tinker_sample"] == pytest.approx(0.000075)
        assert metrics["costs/eval"] == pytest.approx(0.000135)

    @pytest.mark.asyncio
    async def test_openai_client_does_not_log_costs_without_active_metrics_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_async_openai(monkeypatch, _FakeResponse(1_000, 2_000))
        model = _build_model()
        builder = model.metrics_builder("train")

        await model.openai_client().chat.completions.create(
            model=model.get_inference_name(),
            messages=[{"role": "user", "content": "hello"}],
        )

        metrics = await builder.flush()
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_multiple_choices_scale_prefill_cost_once_per_sample(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_async_openai(monkeypatch, _FakeResponse(1_000, 2_000, num_choices=3))
        model = _build_model()
        builder = model.metrics_builder("train")

        with builder.activate_context():
            await model.openai_client().chat.completions.create(
                model=model.get_inference_name(),
                messages=[{"role": "user", "content": "hello"}],
                n=3,
            )

        metrics = await builder.flush()
        assert metrics["costs/train/tinker_prefill"] == pytest.approx(0.00036)
        assert metrics["costs/train/tinker_sample"] == pytest.approx(0.0006)

    def test_manual_cost_calculator_still_returns_tinker_metrics(self) -> None:
        model = _build_model()

        metrics = model.cost_calculator(1_000, 2_000, "train")

        assert metrics["costs/train/tinker_prefill"] == pytest.approx(0.00012)
        assert metrics["costs/train/tinker_sample"] == pytest.approx(0.0006)
