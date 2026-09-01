import asyncio
import importlib
import json

from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
import pytest

from art.metrics import MetricsBuilder

ruler_module = importlib.import_module("art.rewards.ruler")


class _FakePromptTokenDetails:
    def __init__(self, *, cached_tokens: int = 0) -> None:
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakePromptTokenDetails(
            cached_tokens=cached_tokens
        )
        self.cost = cost
        self.model_extra = model_extra


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(
        self,
        *,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
    ) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            model_extra=model_extra,
        )


def _score_content_for_ids(trajectory_ids: list[str]) -> str:
    return json.dumps(
        {
            "scores": [
                {
                    "trajectory_id": trajectory_id,
                    "explanation": f"Trajectory {trajectory_id}.",
                    "score": int(trajectory_id) / 10,
                }
                for trajectory_id in trajectory_ids
            ]
        }
    )


def _score_content(count: int) -> str:
    return _score_content_for_ids([str(index) for index in range(1, count + 1)])


def _response(count: int, *, cost: float | None = None) -> _FakeResponse:
    return _FakeResponse(
        content=_score_content(count),
        prompt_tokens=100,
        completion_tokens=50,
        cost=cost,
    )


_TWO_TRAJECTORIES: list[list[ChatCompletionMessageParam]] = [
    [{"role": "user", "content": "first"}],
    [{"role": "user", "content": "second"}],
]


@pytest.mark.asyncio
async def test_ruler_records_builder_cost_for_supported_judges(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Best answer.",
                            "score": 0.9,
                        }
                    ]
                }
            ),
            prompt_tokens=100,
            completion_tokens=50,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="openai/gpt-4.1",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.9)
    assert metrics["costs/train/judge/ruler"] == pytest.approx(0.0006)
    assert metrics["costs/train/judge"] == pytest.approx(0.0006)
    assert metrics["costs/train"] == pytest.approx(0.0006)
    assert metrics["costs/all"] == pytest.approx(0.0006)
    assert metrics["costs/cum/train/judge/ruler"] == pytest.approx(0.0006)


@pytest.mark.asyncio
async def test_ruler_skips_cost_when_pricing_is_unavailable(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Good enough.",
                            "score": 0.7,
                        }
                    ]
                }
            ),
            prompt_tokens=80,
            completion_tokens=20,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="ollama/qwen3:32b",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.7)
    assert not any(key.startswith("costs/") for key in metrics)


@pytest.mark.asyncio
async def test_ruler_records_direct_cost_for_openrouter_judges(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Good enough.",
                            "score": 0.8,
                        }
                    ]
                }
            ),
            prompt_tokens=80,
            completion_tokens=20,
            cost=1.68e-05,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="openrouter/openai/gpt-4.1-mini",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.8)
    assert metrics["costs/train/judge/ruler"] == pytest.approx(1.68e-05)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_count", [1, 3])
async def test_ruler_retries_missing_or_extra_scores(monkeypatch, invalid_count):
    responses = iter([_response(invalid_count), _response(2)])
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    scores = await ruler_module.ruler(_TWO_TRAJECTORIES)

    assert len(calls) == 2
    assert [score.trajectory_id for score in scores] == ["1", "2"]
    initial_messages = calls[0]["messages"]
    assert len(initial_messages) == 2
    assert "exactly 2 score objects" in initial_messages[0]["content"]
    assert "1, 2" in initial_messages[0]["content"]
    retry_messages = calls[1]["messages"]
    assert retry_messages[:2] == initial_messages
    assert retry_messages[2] == {
        "role": "assistant",
        "content": _score_content(invalid_count),
    }
    correction = retry_messages[3]["content"]
    assert f"had {invalid_count} score object" in correction
    assert (
        "Missing trajectory IDs: 2"
        if invalid_count == 1
        else "Missing trajectory IDs: none"
    ) in correction
    assert "Duplicate trajectory IDs: none" in correction
    assert (
        "Unexpected trajectory IDs: 3"
        if invalid_count == 3
        else "Unexpected trajectory IDs: none"
    ) in correction
    assert "using each trajectory ID exactly once: 1, 2" in correction


@pytest.mark.asyncio
async def test_ruler_raises_after_structural_attempts_are_exhausted(monkeypatch):
    # Reproduce the 046 capture: two trajectories requested, but both otherwise-valid
    # judge responses return only trajectory 1.
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _response(1)

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    with pytest.raises(ValueError, match="Expected 2 scores, but got 1"):
        await ruler_module.ruler(_TWO_TRAJECTORIES)

    assert len(calls) == 2
    retry_messages = calls[1]["messages"]
    assert retry_messages[2] == {
        "role": "assistant",
        "content": _score_content(1),
    }
    assert "Missing trajectory IDs: 2" in retry_messages[3]["content"]


@pytest.mark.asyncio
async def test_ruler_structural_retry_propagates_cancellation(monkeypatch):
    calls = 0

    async def _fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(1)
        raise asyncio.CancelledError

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    with pytest.raises(asyncio.CancelledError):
        await ruler_module.ruler(_TWO_TRAJECTORIES)

    assert calls == 2


@pytest.mark.asyncio
async def test_ruler_records_cost_for_every_structural_attempt(monkeypatch):
    responses = iter([_response(1, cost=0.01), _response(2, cost=0.02)])

    async def _fake_acompletion(**_kwargs):
        return next(responses)

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            _TWO_TRAJECTORIES,
            judge_model="openrouter/openai/gpt-4.1-mini",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert len(scores) == 2
    assert metrics["costs/train/judge/ruler"] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_ruler_retries_duplicate_trajectory_ids(monkeypatch):
    responses = iter(
        [
            _FakeResponse(
                content=_score_content_for_ids(["1", "1"]),
                prompt_tokens=100,
                completion_tokens=50,
            ),
            _response(2),
        ]
    )
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    scores = await ruler_module.ruler(_TWO_TRAJECTORIES)

    assert len(calls) == 2
    assert [score.trajectory_id for score in scores] == ["1", "2"]
    correction = calls[1]["messages"][3]["content"]
    assert "Missing trajectory IDs: 2" in correction
    assert "Duplicate trajectory IDs: 1" in correction


@pytest.mark.asyncio
async def test_ruler_orders_scores_by_trajectory_id(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=_score_content_for_ids(["2", "1"]),
            prompt_tokens=100,
            completion_tokens=50,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    scores = await ruler_module.ruler(_TWO_TRAJECTORIES)

    assert [score.trajectory_id for score in scores] == ["1", "2"]
    assert [score.score for score in scores] == pytest.approx([0.1, 0.2])
