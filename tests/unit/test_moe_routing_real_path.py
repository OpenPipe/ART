from __future__ import annotations

import math
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
import pytest

from art.megatron.routing_replay import (
    build_moe_routing_replay_bundle_from_packed_tensors,
)
from art.preprocessing.moe_routing import (
    ART_MOE_ROUTING_METADATA_KEY,
    align_choice_routes_to_tokenized_result,
    attach_moe_routing_metadata_to_choice,
)
from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import TokenizedResult
from art.trajectories import Trajectory


class _FakeTokenizer:
    def decode(self, token_id: int) -> str:
        return str(token_id)


def _choice(metadata: dict[str, Any]) -> Choice:
    return Choice.model_validate(
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "x"},
            ART_MOE_ROUTING_METADATA_KEY: metadata,
        }
    )


def _route(seed: int) -> list[list[int]]:
    return [[seed, seed + 1], [seed + 2, seed + 3]]


def test_align_choice_routes_to_tokenized_result_maps_vllm_routes() -> None:
    routes, stats = align_choice_routes_to_tokenized_result(
        token_ids=[10, 11, 20, 21],
        choices=[
            _choice(
                {
                    "prompt_token_ids": [10, 11],
                    "completion_token_ids": [20, 21],
                    "prompt_routed_experts": [_route(0), _route(10)],
                    "completion_routed_experts": [_route(20), _route(30)],
                }
            )
        ],
        choice_offsets=[2],
        choice_token_lengths=[2],
    )

    assert routes == [_route(0), _route(10), _route(20), _route(30)]
    assert stats.choices_with_routing == 1
    assert stats.routed_tokens == 4


def test_align_choice_routes_to_tokenized_result_uses_current_vllm_contract() -> None:
    response_payload = {
        "prompt_token_ids": [10, 11],
        "prompt_routed_experts": [_route(0), _route(10)],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "x"},
                "token_ids": [20, 21],
                "routed_experts": [_route(20), _route(30)],
            }
        ],
    }
    choice = Choice.model_validate(response_payload["choices"][0])
    attach_moe_routing_metadata_to_choice(
        choice=choice,
        response_payload=response_payload,
        choice_index=0,
    )

    routes, stats = align_choice_routes_to_tokenized_result(
        token_ids=[10, 11, 20, 21],
        choices=[choice],
        choice_offsets=[2],
        choice_token_lengths=[2],
    )

    assert routes == [_route(0), _route(10), _route(20), _route(30)]
    assert stats.choices_with_routing == 1
    assert stats.routed_tokens == 4


def test_align_choice_routes_to_tokenized_result_rejects_token_mismatch() -> None:
    with pytest.raises(RuntimeError, match="prompt token ids do not match"):
        align_choice_routes_to_tokenized_result(
            token_ids=[10, 12, 20],
            choices=[
                _choice(
                    {
                        "prompt_token_ids": [10, 11],
                        "completion_token_ids": [20],
                        "prompt_routed_experts": [_route(0), _route(10)],
                        "completion_routed_experts": [_route(20)],
                    }
                )
            ],
            choice_offsets=[2],
            choice_token_lengths=[1],
        )


def _tokenized(
    token_ids: list[int],
    routes: list[list[list[int]]],
    *,
    prompt_id: int,
    prompt_length: int,
) -> TokenizedResult:
    return TokenizedResult(
        advantage=1.0,
        chat="",
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=[0] * prompt_length + [1] * (len(token_ids) - prompt_length),
        logprobs=[math.nan] * prompt_length + [-1.0] * (len(token_ids) - prompt_length),
        pixel_values=None,
        image_grid_thw=None,
        trajectory=Trajectory(),
        choice_offsets=[prompt_length],
        extra_logprobs={},
        _tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        moe_routed_experts=cast(list[list[list[int]] | None], routes),
        prompt_id=prompt_id,
        prompt_length=prompt_length,
    )


def test_pack_carries_routes_through_shared_prefix_splicing() -> None:
    first = _tokenized(
        [10, 11, 20, 21],
        [_route(0), _route(10), _route(20), _route(30)],
        prompt_id=123,
        prompt_length=2,
    )
    second = _tokenized(
        [10, 11, 22, 23],
        [_route(0), _route(99), _route(40), _route(50)],
        prompt_id=123,
        prompt_length=2,
    )

    packed = packed_tensors_from_tokenized_results(
        [first, second],
        seq_len=8,
        pad_token_id=0,
        truncate_long_results=False,
        include_moe_routing=True,
    )

    assert packed["tokens"].tolist()[0][:6] == [10, 11, 20, 21, 22, 23]
    routing_replay = packed["moe_routing_replay"]
    assert routing_replay is not None
    assert routing_replay.expert_indices.tolist()[0][:6] == [
        _route(0),
        _route(10),
        _route(20),
        _route(30),
        _route(40),
        _route(50),
    ]
    stats = routing_replay.pack_stats
    assert stats.shared_prefix_rows == 2
    assert stats.shared_prefix_conflict_rows == 1
    assert stats.shared_prefix_conflict_slots == 4


def test_build_replay_bundle_uses_packed_sequence_sample_calls() -> None:
    result = _tokenized(
        [10, 11, 20],
        [_route(0), _route(10), _route(20)],
        prompt_id=456,
        prompt_length=2,
    )
    packed = packed_tensors_from_tokenized_results(
        [result],
        seq_len=4,
        pad_token_id=0,
        truncate_long_results=False,
        include_moe_routing=True,
    )

    bundle = build_moe_routing_replay_bundle_from_packed_tensors(
        packed_tensors=packed,
        global_grad_accumulation_sequences=1,
    )

    route = bundle.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    assert route.sample_index == 0
    assert route.expert_indices.tolist()[:3] == [[0, 1], [10, 11], [20, 21]]
    assert len(set(route.expert_indices.tolist()[3])) == 2
