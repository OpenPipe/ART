from datetime import datetime, timezone
import json
import math
from typing import Any

import numpy as np
from openai.types.chat import ChatCompletion
import pytest
import torch

from art import trajectories as tr
from art.preprocessing.dynamo_tokens import choice_completion_logprobs
from art.trajectories._protocols import _chat_response
from art.trajectories._routed_experts import choice_routes, history_routes


def _response():
    return ChatCompletion.model_validate(
        {
            "id": "capture",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "prompt_token_ids": [10, 11],
            "prompt_routed_experts": [[[0, 1]], [[1, 2]]],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "yes"},
                    "token_ids": [20, 21],
                    "compact_logprobs": [-0.1, -0.2],
                    "routed_experts": [[[2, 3]], [[-1, -1]]],
                }
            ],
        }
    )


def _trajectory(response=None):
    now = datetime.now(timezone.utc)
    exchange = tr.ChatCompletionsExchange(
        request={"model": "policy", "messages": [{"role": "user", "content": "hi"}]},
        response=response or _response(),
        start_time=now,
        end_time=now,
    )
    return tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]), reward=0
    )


def test_compact_logprobs_and_routes_survive_tokenize_tensorize_and_serialization():
    trajectory = _trajectory()
    expected = [[[0, 1]], [[1, 2]], [[2, 3]], [[-1, -1]]]
    tokenized = trajectory.tokenize()
    assert tokenized.tokens == [10, 11, 20, 21]
    assert tokenized.logprobs[2:] == [-0.1, -0.2]
    assert tokenized.routed_experts == expected
    restored = tr.Trajectory.model_validate_json(
        trajectory.model_dump_json()
    ).tokenize()
    assert restored.routed_experts == expected
    tensorized = tokenized.tensorize()
    assert tensorized.routed_experts.dtype == torch.int32
    assert tensorized.routed_experts.shape == (4, 1, 2)
    assert tensorized.routed_experts.tolist() == expected
    moved = tensorized.to("cpu")
    moved.routed_experts[0, 0, 0] = 99
    assert tensorized.routed_experts[0, 0, 0] == 0
    restored_tensor = type(tensorized).model_validate_json(tensorized.model_dump_json())
    assert restored_tensor.routed_experts.tolist() == expected
    for value in (tokenized, tensorized):
        restored = tr.compact_validate(value.compact_dump())
        assert isinstance(restored, (tr.TokenizedTrajectory, tr.TensorizedTrajectory))
        assert restored.routed_experts is not None


@pytest.mark.parametrize(
    "mode,flag",
    [
        ("raw_logprobs", tr.TokenFlag.RAW_LOGPROBS),
        ("processed_logprobs", tr.TokenFlag.PROCESSED_LOGPROBS),
    ],
)
def test_logprob_modes_survive_tokenization_tensorization_and_compact(mode, flag):
    response = _response()
    response.model_extra["logprobs_mode"] = mode
    tokenized = _trajectory(response).tokenize()
    assert [bool(value & flag) for value in tokenized.flags] == [
        False,
        False,
        True,
        True,
    ]
    for value in (tokenized, tokenized.tensorize()):
        restored = tr.compact_validate(value.compact_dump())
        assert isinstance(restored, (tr.TokenizedTrajectory, tr.TensorizedTrajectory))
        assert [bool(int(item) & flag) for item in restored.flags] == [
            False,
            False,
            True,
            True,
        ]


def test_legacy_binary_routes_align_and_leave_last_token_missing():
    response = _response()
    extra = response.choices[0].model_extra
    response.model_extra.pop("prompt_routed_experts")
    extra.pop("routed_experts")
    extra["art_moe_routing"] = {
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [20, 21],
        "num_experts": 4,
        "routed_experts": np.array([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=np.uint8),
    }
    tokenized = _trajectory(response).tokenize()
    assert tokenized.routed_experts == [[[0, 1]], [[1, 2]], [[2, 3]], [[-1, -1]]]


def test_routes_do_not_follow_equal_tokens_after_a_changed_causal_prefix():
    history = _trajectory().histories()[0]
    assert history_routes(history, [10, 999, 20, 21]) == [
        [[0, 1]],
        [[-1, -1]],
        [[-1, -1]],
        [[-1, -1]],
    ]


@pytest.mark.parametrize("values", [[-0.1], [math.nan, -0.2], [True, -0.2]])
def test_malformed_compact_logprobs_fail_even_with_classic_fallback(values):
    response = _response()
    response.choices[0].model_extra["compact_logprobs"] = values
    with pytest.raises(ValueError, match="matching finite logprobs"):
        choice_completion_logprobs(response.choices[0])


def test_duplicate_compact_fields_must_agree():
    response = _response()
    response.choices[0].model_extra["art_completion_logprobs"] = [-0.1, -0.3]
    with pytest.raises(ValueError, match="disagree"):
        choice_completion_logprobs(response.choices[0])


def test_stream_aggregates_logprob_deltas_and_final_routing_snapshot():
    chunks: list[dict[str, Any]] = []
    response = _response().model_dump()
    for i, token in enumerate([20, 21]):
        chunks.append(
            {
                "id": "capture",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "policy",
                "prompt_token_ids": [10, 11],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "yes" if i == 0 else ""},
                        "token_ids": [token],
                        "compact_logprobs": [-0.1 if i == 0 else -0.2],
                        "compact_top_logprobs": {
                            "token_ids": [[token, 99]],
                            "logprobs": [[-0.1, -0.9]],
                        },
                        "finish_reason": "stop" if i else None,
                    }
                ],
            }
        )
    chunks[-1]["prompt_routed_experts"] = response["prompt_routed_experts"]
    chunks[-1]["choices"][0]["routed_experts"] = response["choices"][0][
        "routed_experts"
    ]
    body = (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    ).encode()
    restored = _chat_response(body, stream=True)
    assert restored.choices[0].model_extra is not None
    assert restored.choices[0].model_extra["compact_logprobs"] == [-0.1, -0.2]
    assert restored.choices[0].model_extra["compact_top_logprobs"]["token_ids"] == [
        [20, 99],
        [21, 99],
    ]
    assert _trajectory(restored).tokenize().routed_experts == [
        [[0, 1]],
        [[1, 2]],
        [[2, 3]],
        [[-1, -1]],
    ]


@pytest.mark.parametrize(
    "routes",
    [[[[1]]], [[[True]], [[1]], [[1]], [[1]]], [[[1]], [[1, 2]], [[1]], [[1]]]],
)
def test_routes_reject_bad_lengths_ids_and_shapes(routes):
    value = _trajectory().tokenize().model_dump()
    value["routed_experts"] = routes
    with pytest.raises(ValueError):
        tr.TokenizedTrajectory.model_validate(value)


def test_topk_has_sampled_token_alignment_and_survives_tensor_and_json_roundtrips():
    response = _response()
    response.choices[0].model_extra["compact_top_logprobs"] = {
        "token_ids": [[20, 99], [98]],
        "logprobs": [[-0.1, -0.9], [-0.05]],
    }
    tokenized = _trajectory(response).tokenize()
    assert tokenized.top_k.tokens == [[-1, -1], [-1, -1], [20, 99], [98, -1]]
    assert math.isnan(tokenized.top_k.logprobs[3][1])
    assert tokenized.top_k.logprobs[2] == [-0.1, -0.9]
    # Row 3 describes the distribution that sampled token 21, even when 21
    # is outside top-k. TrainerRank predicts that token at forward row 2.
    assert tokenized.logprobs[3] == -0.2
    tensorized = tokenized.tensorize()
    assert isinstance(tensorized.top_k, tr.TensorizedTopK)
    assert tensorized.top_k.tokens.dtype == torch.int64
    assert tensorized.top_k.logprobs.dtype == torch.float32
    assert tensorized.top_k.tokens.shape == (4, 2)
    for value in (tokenized, tensorized):
        restored = type(value).model_validate_json(value.model_dump_json())
        compact = tr.compact_validate(value.compact_dump())
        for candidate in (restored, compact):
            assert isinstance(
                candidate, (tr.TokenizedTrajectory, tr.TensorizedTrajectory)
            )
            assert candidate.top_k is not None
            ids = candidate.top_k.tokens
            assert (
                ids.tolist() if isinstance(ids, torch.Tensor) else ids
            ) == tokenized.top_k.tokens
    moved = tensorized.to("cpu")
    moved.top_k.tokens[2, 0] = 123
    assert tensorized.top_k.tokens[2, 0] == 20
    tensorized.to_("cpu")
    assert tensorized.top_k.tokens[2, 0] == 20


def test_equivalent_legacy_and_native_routes_can_coexist_but_token_ids_cannot_conflict():
    response = _response()
    extra = response.choices[0].model_extra
    extra["art_moe_routing"] = {
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [20, 21],
        "routed_experts": np.array([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=np.uint8),
    }
    capture = choice_routes(response.choices[0], response)
    assert capture is not None and capture[1][-1] == [[-1, -1]]
    extra["art_moe_routing"]["completion_token_ids"] = [99, 21]
    with pytest.raises(ValueError, match="response token IDs"):
        choice_routes(response.choices[0], response)


def test_legacy_choice_retains_response_prompt_routes_after_finalization():
    from art.openai import finalize_chat_completion

    response = finalize_chat_completion(_response())
    choice = response.choices[0]
    capture = choice_routes(choice, None)
    assert capture is not None and capture[1] == [
        [[0, 1]],
        [[1, 2]],
        [[2, 3]],
        [[-1, -1]],
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_text_completions_preserve_compact_and_routing_metadata(stream):
    from art.trajectories._protocols import _completion_response

    response = _response()
    choice = response.choices[0].model_dump(exclude={"message"})
    choice.update(
        text="yes",
        prompt_token_ids=[10, 11],
        prompt_routed_experts=[[[0, 1]], [[1, 2]]],
    )
    choice["compact_top_logprobs"] = {
        "token_ids": [[20], [21]],
        "logprobs": [[-0.1], [-0.2]],
    }
    payload = {
        "id": "text",
        "object": "text_completion",
        "created": 0,
        "model": "policy",
        "choices": [choice],
    }
    body = json.dumps(payload)
    if stream:
        body = "data: " + body + "\n\ndata: [DONE]\n\n"
    now = datetime.now(timezone.utc)
    exchange = tr.CompletionsExchange(
        request={"model": "policy", "prompt": [10, 11]},
        response=_completion_response(body.encode(), stream=stream),
        start_time=now,
        end_time=now,
    )
    tokenized = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(completions=[exchange]), reward=0
    ).tokenize()
    assert tokenized.tokens == [10, 11, 20, 21]
    assert tokenized.logprobs[2:] == [-0.1, -0.2]
    assert tokenized.routed_experts == [[[0, 1]], [[1, 2]], [[2, 3]], [[-1, -1]]]
    assert tokenized.top_k is not None and tokenized.top_k.tokens == [
        [-1],
        [-1],
        [20],
        [21],
    ]
