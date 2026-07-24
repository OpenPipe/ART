from __future__ import annotations

import asyncio
import copy
import json
import math
from typing import Any

import httpx
from pydantic import ValidationError
import pytest
import torch

from art.distill.reference_loss import topk_plus_tail_forward_kl
from art.distill.scorer import TeacherScoringRequest
from art.distill.types import TeacherView, TopK
from art.distill.vllm import (
    _VLLM_FLOAT32_MASS_OVERFLOW_ALLOWANCE,
    VLLMRetryableScoringError,
    VLLMScoringError,
    VLLMTeacherScorer,
    _parse_position,
)
from art.serving_capabilities import ServingCapabilities


def _capabilities(**updates: Any) -> ServingCapabilities:
    values = {
        "runtime": "art_vllm",
        "protocol_version": 1,
        "prompt_token_distributions": True,
        "prompt_token_distribution_version": 1,
        "max_prompt_logprobs": 20,
        "full_prompt_distribution": False,
        "prompt_distribution_temperature": "unit_only",
        "token_space_fingerprint": "token-space",
        "logical_vocab_size": 5,
    }
    values.update(updates)
    return ServingCapabilities.model_validate(values)


_MISSING = object()


def _request(
    *,
    k: int = 2,
    temperature: float = 1.0,
    truncate_prompt_tokens: Any = _MISSING,
) -> TeacherScoringRequest:
    teacher_request: dict[str, Any] = {
        "messages": [
            {"content": "Choose a card.", "role": "user"},
            {
                "content": "Hint: the private label is blue.",
                "role": "system",
            },
        ],
        "model": "student",
    }
    if truncate_prompt_tokens is not _MISSING:
        teacher_request["truncate_prompt_tokens"] = truncate_prompt_tokens
    return TeacherScoringRequest.create(
        generation_id="generation-1",
        teacher_view=TeacherView.from_request(
            "chat_completions",
            teacher_request,
        ),
        forced_token_ids=(2, 3),
        selected_positions=(0, 1),
        teacher_name="teacher",
        teacher_revision="revision-7",
        token_space_fingerprint="token-space",
        logical_vocab_size=5,
        target=TopK(k=k, temperature=temperature),
    )


def _wire_response(request: TeacherScoringRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "model": "served-teacher",
        "revision": request.teacher_revision,
        "choices": [{"finish_reason": "length", "index": 0, "token_ids": [4]}],
        "prompt_logprobs": [
            None,
            {"10": {"logprob": math.log(0.9), "rank": 1}},
            {
                "0": {"logprob": math.log(0.5), "rank": 1},
                "1": {"logprob": math.log(0.3), "rank": 2},
                # vLLM includes the forced token even when it is outside top-k.
                "2": {"logprob": math.log(0.1), "rank": 3},
            },
            {
                "3": {"logprob": math.log(0.6), "rank": 1},
                "4": {"logprob": math.log(0.25), "rank": 2},
            },
        ],
    }


def test_distribution_capability_requires_token_space_and_capacity() -> None:
    with pytest.raises(ValidationError, match="version, capacity"):
        ServingCapabilities(
            runtime="art_vllm",
            protocol_version=1,
            prompt_token_distributions=True,
        )


@pytest.mark.asyncio
async def test_scores_exact_forced_tokens_and_retains_residual_tail() -> None:
    request = _request()
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        calls.append((http_request.url.path, body))
        if http_request.url.path == "/v1/chat/completions/render":
            return httpx.Response(
                200,
                json={
                    "request_id": "render-id",
                    "token_ids": [10, 11],
                    "sampling_params": {"max_tokens": 99, "temperature": 0.7},
                    "model": "base-teacher",
                },
            )
        return httpx.Response(200, json=_wire_response(request))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=_capabilities(),
            model_name="served-teacher",
            render_model_name="base-teacher",
            headers={"Authorization": "Bearer secret"},
            client=client,
        )
        result = await scorer.score(request)

    assert [path for path, _ in calls] == [
        "/v1/chat/completions/render",
        "/inference/v1/generate",
    ]
    render = calls[0][1]
    teacher_view = request.teacher_view.request()
    assert isinstance(teacher_view, dict)
    assert render["messages"] == teacher_view["messages"]
    assert render["model"] == "base-teacher"
    assert render["stream"] is False

    generate = calls[1][1]
    assert generate["model"] == "served-teacher"
    assert generate["token_ids"] == [10, 11, 2, 3]
    assert generate["request_id"] == request.request_id
    assert generate["sampling_params"]["prompt_logprobs"] == 2
    assert generate["sampling_params"]["skip_reading_prefix_cache"] is True
    assert generate["sampling_params"]["max_tokens"] == 1
    assert generate["sampling_params"]["temperature"] == 0.0

    first, second = result.positions
    assert tuple(entry.token_id for entry in first.entries) == (0, 1)
    assert 2 not in tuple(entry.token_id for entry in first.entries)
    assert math.exp(first.tail_logprob or 0.0) == pytest.approx(0.2)
    assert tuple(entry.token_id for entry in second.entries) == (3, 4)
    assert math.exp(second.tail_logprob or 0.0) == pytest.approx(0.15)


_OBSERVED_VLLM_TOP32 = (
    (902, -0.16369986534118652),
    (9834, -1.9136998653411865),
    (7196, -5.663700103759766),
    (21639, -16.038700103759766),
    (59565, -18.038700103759766),
    (1231, -19.163700103759766),
    (18581, -19.663700103759766),
    (8365, -20.288700103759766),
    (537, -20.413700103759766),
    (30315, -20.538700103759766),
    (72104, -20.538700103759766),
    (18341, -21.288700103759766),
    (220, -22.163700103759766),
    (6857, -22.288700103759766),
    (11891, -22.288700103759766),
    (66479, -22.413700103759766),
    (4658, -22.538700103759766),
    (2581, -23.413700103759766),
    (2152, -23.413700103759766),
    (84198, -23.538700103759766),
    (39205, -23.788700103759766),
    (2578, -24.163700103759766),
    (21502, -24.163700103759766),
    (13866, -24.413700103759766),
    (308, -24.538700103759766),
    (33100, -24.538700103759766),
    (4302, -24.538700103759766),
    (6536, -24.663700103759766),
    (9693, -24.788700103759766),
    (834, -24.788700103759766),
    (28366, -24.913700103759766),
    (3204, -24.913700103759766),
)


def _observed_vllm_wire_distribution() -> dict[str, Any]:
    return {
        str(token_id): {"logprob": logprob, "rank": rank}
        for rank, (token_id, logprob) in enumerate(_OBSERVED_VLLM_TOP32, start=1)
    }


def test_accepts_observed_vllm_overflow_and_normalizes_deterministically() -> None:
    wire = _observed_vllm_wire_distribution()
    observed_mass = math.fsum(math.exp(entry["logprob"]) for entry in wire.values())
    assert observed_mass - 1.0 == pytest.approx(1.113570438e-7, rel=1e-8)

    first = _parse_position(
        wire,
        position=0,
        forced_token_id=9693,
        k=32,
        logical_vocab_size=151936,
    )
    second = _parse_position(
        wire,
        position=0,
        forced_token_id=9693,
        k=32,
        logical_vocab_size=151936,
    )

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.tail_logprob is not None
    assert math.exp(first.tail_logprob) > 0.0
    masses = [
        *(math.exp(entry.logprob) for entry in first.entries),
        math.exp(first.tail_logprob),
    ]
    assert math.fsum(masses) == pytest.approx(1.0, abs=2e-15)
    assert all(
        left.logprob >= right.logprob
        for left, right in zip(first.entries, first.entries[1:])
    )


def _unit_plus_overflow_wire(total_mass: float) -> dict[str, Any]:
    return {
        "0": {"logprob": 0.0, "rank": 1},
        "1": {"logprob": math.log(total_mass - 1.0), "rank": 2},
    }


def test_accepts_exact_vllm_float32_overflow_boundary() -> None:
    total_mass = 1.0 + _VLLM_FLOAT32_MASS_OVERFLOW_ALLOWANCE
    result = _parse_position(
        _unit_plus_overflow_wire(total_mass),
        position=0,
        forced_token_id=0,
        k=2,
        logical_vocab_size=3,
    )

    assert result.tail_logprob is not None
    assert math.fsum(
        [
            *(math.exp(entry.logprob) for entry in result.entries),
            math.exp(result.tail_logprob),
        ]
    ) == pytest.approx(1.0, abs=2e-15)


@pytest.mark.parametrize(
    "total_mass",
    [
        math.nextafter(
            1.0 + _VLLM_FLOAT32_MASS_OVERFLOW_ALLOWANCE,
            math.inf,
        ),
        1.001,
    ],
)
def test_rejects_beyond_boundary_and_material_sparse_overflow(
    total_mass: float,
) -> None:
    with pytest.raises(VLLMScoringError, match="invalid residual"):
        _parse_position(
            _unit_plus_overflow_wire(total_mass),
            position=0,
            forced_token_id=0,
            k=2,
            logical_vocab_size=3,
        )


def test_sparse_overflow_still_requires_forced_token_observation() -> None:
    with pytest.raises(VLLMScoringError, match="omitted the forced token"):
        _parse_position(
            _observed_vllm_wire_distribution(),
            position=0,
            forced_token_id=42,
            k=32,
            logical_vocab_size=151936,
        )


def test_synthetic_tail_has_negligible_finite_forward_kl_effect() -> None:
    parsed = _parse_position(
        _observed_vllm_wire_distribution(),
        position=0,
        forced_token_id=9693,
        k=32,
        logical_vocab_size=151936,
    )
    explicit = torch.tensor(
        [[math.exp(entry.logprob) for entry in parsed.entries]],
        dtype=torch.float64,
    )
    tail = torch.tensor([math.exp(parsed.tail_logprob or 0.0)], dtype=torch.float64)
    ids = torch.tensor(
        [[entry.token_id for entry in parsed.entries]],
        dtype=torch.int64,
    )
    logits = torch.zeros((1, 151936), dtype=torch.float64)
    repaired = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=ids,
        teacher_topk_probs=explicit,
        teacher_tail_prob=tail,
        mask=torch.tensor([True]),
    )
    no_tail = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=ids,
        teacher_topk_probs=explicit / explicit.sum(),
        teacher_tail_prob=torch.zeros(1, dtype=torch.float64),
        mask=torch.tensor([True]),
    )

    assert torch.isfinite(repaired.loss_sum)
    assert tail.item() > 0.0
    assert abs(repaired.loss_sum.item() - no_tail.loss_sum.item()) < 1e-12


def test_positive_sparse_residual_preserves_explicit_logprobs() -> None:
    first_logprob = math.log(0.6)
    second_logprob = math.log(0.3)
    result = _parse_position(
        {
            "0": {"logprob": first_logprob, "rank": 1},
            "1": {"logprob": second_logprob, "rank": 2},
        },
        position=0,
        forced_token_id=0,
        k=2,
        logical_vocab_size=5,
    )

    assert tuple(entry.logprob for entry in result.entries) == (
        first_logprob,
        second_logprob,
    )
    expected_tail = 1.0 - math.fsum((math.exp(first_logprob), math.exp(second_logprob)))
    assert result.tail_logprob == math.log(expected_tail)


def test_full_distribution_normalizes_without_synthesizing_a_tail() -> None:
    result = _parse_position(
        {
            "0": {"logprob": math.log(0.6), "rank": 1},
            "1": {"logprob": math.log(0.40000004), "rank": 2},
        },
        position=0,
        forced_token_id=0,
        k=2,
        logical_vocab_size=2,
    )

    assert result.tail_logprob is None
    assert math.fsum(math.exp(entry.logprob) for entry in result.entries) == (
        pytest.approx(1.0, abs=2e-15)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scoring_request", "capabilities", "message"),
    [
        (
            _request(),
            ServingCapabilities.openai_compatible(),
            "requires serving capability",
        ),
        (_request(temperature=2.0), _capabilities(), "temperature 1.0 only"),
        (_request(k=3), _capabilities(max_prompt_logprobs=2), "exceeds"),
        (
            _request(),
            _capabilities(token_space_fingerprint="different"),
            "fingerprints",
        ),
        (
            _request(),
            _capabilities(logical_vocab_size=6),
            "vocabulary sizes",
        ),
    ],
)
async def test_rejects_unsupported_capabilities_before_network(
    scoring_request: TeacherScoringRequest,
    capabilities: ServingCapabilities,
    message: str,
) -> None:
    async def unexpected(_: httpx.Request) -> httpx.Response:
        raise AssertionError("capability rejection must happen before I/O")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=capabilities,
            client=client,
        )
        with pytest.raises(VLLMScoringError, match=message):
            await scorer.score(scoring_request)


@pytest.mark.asyncio
async def test_classifies_transient_http_status_for_bounded_retry() -> None:
    async def unavailable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "temporarily unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=_capabilities(),
            client=client,
        )
        with pytest.raises(VLLMRetryableScoringError, match="transiently"):
            await scorer.score(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize("truncate_prompt_tokens", [1, 0])
async def test_rejects_prompt_truncation_before_network(
    truncate_prompt_tokens: int,
) -> None:
    async def unexpected(_: httpx.Request) -> httpx.Response:
        raise AssertionError("prompt-truncation rejection must happen before I/O")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=_capabilities(),
            client=client,
        )
        with pytest.raises(VLLMScoringError, match="does not allow prompt truncation"):
            await scorer.score(_request(truncate_prompt_tokens=truncate_prompt_tokens))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda response: response["prompt_logprobs"].pop(),
            "count does not match",
        ),
        (
            lambda response: response["prompt_logprobs"][2]["1"].pop("rank"),
            "malformed rank",
        ),
        (
            lambda response: response["prompt_logprobs"][2].update(
                {"5": {"logprob": math.log(0.1), "rank": 2}}
            ),
            "invalid token ID",
        ),
        (
            lambda response: response["prompt_logprobs"][2]["1"].update({"rank": 1}),
            "duplicate ranks",
        ),
        (
            lambda response: response["prompt_logprobs"][2].pop("2"),
            "omitted the forced token",
        ),
        (
            lambda response: response["prompt_logprobs"][2]["0"].update(
                {"logprob": float("inf")}
            ),
            "non-finite",
        ),
        (
            lambda response: response["prompt_logprobs"][2].update(
                {
                    "0": {"logprob": math.log(0.8), "rank": 1},
                    "1": {"logprob": math.log(0.4), "rank": 2},
                }
            ),
            "invalid residual",
        ),
        (
            lambda response: response.update({"model": "wrong"}),
            "model does not match",
        ),
        (
            lambda response: response.update({"revision": "wrong"}),
            "revision does not match",
        ),
    ],
)
async def test_rejects_malformed_or_mismatched_generate_responses(
    mutate: Any,
    message: str,
) -> None:
    request = _request()
    generated = copy.deepcopy(_wire_response(request))
    mutate(generated)

    async def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path.endswith("/render"):
            return httpx.Response(
                200,
                json={
                    "token_ids": [10, 11],
                    "sampling_params": {},
                    "model": "served-teacher",
                },
            )
        return httpx.Response(
            200,
            content=json.dumps(generated).encode(),
            headers={"Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=_capabilities(),
            model_name="served-teacher",
            client=client,
        )
        with pytest.raises(VLLMScoringError, match=message):
            await scorer.score(request)


@pytest.mark.asyncio
async def test_cancellation_propagates_without_a_partial_result() -> None:
    async def cancel(_: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancel)) as client:
        scorer = VLLMTeacherScorer(
            base_url="http://teacher.test",
            capabilities=_capabilities(),
            client=client,
        )
        with pytest.raises(asyncio.CancelledError):
            await scorer.score(_request())
