from __future__ import annotations

import math

from pydantic import ValidationError
import pytest

from art.distill.scorer import (
    RankedTokenLogprob,
    ScoredPosition,
    TeacherDistributionScorer,
    TeacherScoringRequest,
    TeacherScoringResult,
)
from art.distill.types import TeacherView, TopK


def _request(*, temperature: float = 1.0) -> TeacherScoringRequest:
    return TeacherScoringRequest.create(
        generation_id="generation-1",
        teacher_view=TeacherView.from_request(
            "chat_completions",
            {"messages": [{"content": "Choose", "role": "user"}]},
        ),
        forced_token_ids=(4, 3, 2),
        selected_positions=(0, 2),
        teacher_name="teacher",
        teacher_revision="revision-7",
        token_space_fingerprint="token-space",
        logical_vocab_size=5,
        target=TopK(k=2, temperature=temperature),
    )


def _row(position: int, forced_token_id: int) -> ScoredPosition:
    return ScoredPosition(
        position=position,
        forced_token_id=forced_token_id,
        entries=(
            RankedTokenLogprob(rank=0, token_id=0, logprob=math.log(0.6)),
            RankedTokenLogprob(rank=1, token_id=1, logprob=math.log(0.3)),
        ),
        tail_logprob=math.log(0.1),
        logical_vocab_size=5,
        temperature=1.0,
    )


def _result(request: TeacherScoringRequest | None = None) -> TeacherScoringResult:
    request = request or _request()
    return TeacherScoringResult.create(
        request=request,
        positions=(_row(0, 4), _row(2, 2)),
    )


def test_request_identity_is_deterministic_and_models_round_trip() -> None:
    first = _request()
    second = _request()

    assert first.request_id == second.request_id
    assert first.request_sha256 == second.request_sha256
    assert first.forced_token_sha256 == second.forced_token_sha256
    assert TeacherScoringRequest.model_validate_json(first.model_dump_json()) == first

    result = _result(first)
    assert TeacherScoringResult.model_validate_json(result.model_dump_json()) == result

    changed = TeacherScoringRequest.create(
        generation_id=first.generation_id,
        teacher_view=first.teacher_view,
        forced_token_ids=first.forced_token_ids,
        selected_positions=(2,),
        teacher_name=first.teacher_name,
        teacher_revision=first.teacher_revision,
        token_space_fingerprint=first.token_space_fingerprint,
        logical_vocab_size=first.logical_vocab_size,
        target=first.target,
    )
    assert changed.request_id != first.request_id


def test_rank_order_width_and_normalization_are_validated() -> None:
    with pytest.raises(ValidationError, match="contiguous rank order"):
        _row(0, 4).model_copy(
            update={
                "entries": (
                    RankedTokenLogprob(rank=1, token_id=0, logprob=math.log(0.6)),
                    RankedTokenLogprob(rank=0, token_id=1, logprob=math.log(0.3)),
                )
            }
        ).model_validate(
            _row(0, 4)
            .model_copy(
                update={
                    "entries": (
                        RankedTokenLogprob(rank=1, token_id=0, logprob=math.log(0.6)),
                        RankedTokenLogprob(rank=0, token_id=1, logprob=math.log(0.3)),
                    )
                }
            )
            .model_dump()
        )

    with pytest.raises(ValidationError, match="descending probability"):
        ScoredPosition(
            position=0,
            forced_token_id=4,
            entries=(
                RankedTokenLogprob(rank=0, token_id=0, logprob=math.log(0.3)),
                RankedTokenLogprob(rank=1, token_id=1, logprob=math.log(0.6)),
            ),
            tail_logprob=math.log(0.1),
            logical_vocab_size=5,
            temperature=1.0,
        )

    with pytest.raises(ValidationError, match="normalized"):
        _row(0, 4).model_copy(update={"tail_logprob": math.log(0.2)}).model_validate(
            _row(0, 4).model_copy(update={"tail_logprob": math.log(0.2)}).model_dump()
        )

    with pytest.raises(ValidationError, match="exact top-k width"):
        TeacherScoringResult.create(
            request=_request(),
            positions=(
                _row(0, 4).model_copy(
                    update={
                        "entries": (_row(0, 4).entries[0],),
                        "tail_logprob": math.log(0.4),
                    }
                ),
                _row(2, 2),
            ),
        )


def test_result_rejects_incomplete_positions_and_echo_mismatch() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="completely match"):
        TeacherScoringResult(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            generation_id=request.generation_id,
            teacher_view_fingerprint=request.teacher_view.fingerprint,
            forced_token_sha256=request.forced_token_sha256,
            selected_positions=request.selected_positions,
            teacher_name=request.teacher_name,
            teacher_revision=request.teacher_revision,
            token_space_fingerprint=request.token_space_fingerprint,
            logical_vocab_size=request.logical_vocab_size,
            target=request.target,
            positions=(_row(0, 4),),
        )

    result = _result(request)
    mismatched = result.model_copy(update={"teacher_revision": "revision-8"})
    with pytest.raises(ValueError, match="teacher revision"):
        mismatched.validate_for(request)

    wrong_temperature_request = _request(temperature=2.0)
    wrong_temperature_echo = result.model_copy(
        update={
            "request_id": wrong_temperature_request.request_id,
            "request_sha256": wrong_temperature_request.request_sha256,
        }
    )
    with pytest.raises(ValueError, match="target"):
        wrong_temperature_echo.validate_for(wrong_temperature_request)


def test_forced_token_may_be_outside_top_k_without_changing_width() -> None:
    request = _request()
    result = _result(request)

    assert request.forced_token_ids[0] == 4
    assert 4 not in {entry.token_id for entry in result.positions[0].entries}
    assert len(result.positions[0].entries) == request.target.k
    assert result.validate_for(request) is result

    wrong_forced_token = result.model_copy(
        update={
            "positions": (
                result.positions[0].model_copy(update={"forced_token_id": 3}),
                result.positions[1],
            )
        }
    )
    with pytest.raises(ValueError, match="forced token"):
        wrong_forced_token.validate_for(request)


async def test_protocol_accepts_and_calls_a_recording_fake() -> None:
    request = _request()
    expected = _result(request)

    class RecordingScorer:
        calls: list[TeacherScoringRequest]

        def __init__(self) -> None:
            self.calls = []

        async def score(
            self,
            request: TeacherScoringRequest,
        ) -> TeacherScoringResult:
            self.calls.append(request)
            return expected

    fake = RecordingScorer()
    scorer: TeacherDistributionScorer = fake

    actual = await scorer.score(request)

    assert actual is expected
    assert fake.calls == [request]
