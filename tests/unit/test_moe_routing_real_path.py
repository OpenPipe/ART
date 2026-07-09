from __future__ import annotations

import math
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
import pytest
import torch

from art.megatron.prefix_tree import parse_prefix_tree_row
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
    trainable_start: int | None = None,
    advantage: float = 1.0,
    weight: float = 1.0,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> TokenizedResult:
    trainable_start = prompt_length if trainable_start is None else trainable_start
    return TokenizedResult(
        advantage=advantage,
        chat="",
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=[0] * trainable_start + [1] * (len(token_ids) - trainable_start),
        logprobs=[math.nan] * trainable_start
        + [-1.0] * (len(token_ids) - trainable_start),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        trajectory=Trajectory(),
        choice_offsets=[trainable_start],
        extra_logprobs={},
        _tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        moe_routed_experts=cast(list[list[list[int]] | None], routes),
        prompt_id=prompt_id,
        prompt_length=prompt_length,
        weight=weight,
    )


def test_pack_carries_routes_through_prefix_tree_splicing() -> None:
    first = _tokenized(
        [10, 11, 20, 21],
        [_route(0), _route(10), _route(20), _route(30)],
        prompt_id=123,
        prompt_length=1,
        trainable_start=2,
    )
    second = _tokenized(
        [10, 11, 22, 23],
        [_route(99), _route(10), _route(40), _route(50)],
        prompt_id=123,
        prompt_length=1,
        trainable_start=2,
    )

    packed = packed_tensors_from_tokenized_results(
        [first, second],
        seq_len=8,
        pad_token_id=0,
        truncate_long_results=False,
        include_moe_routing=True,
    )

    assert packed["tokens"].tolist()[0][:7] == [10, 11, 20, 21, 11, 22, 23]
    routing_replay = packed["moe_routing_replay"]
    assert routing_replay is not None
    assert routing_replay.expert_indices.tolist()[0][:7] == [
        _route(0),
        _route(10),
        _route(20),
        _route(30),
        _route(10),
        _route(40),
        _route(50),
    ]
    stats = routing_replay.pack_stats
    assert stats.prefix_tree_rows == 1
    assert stats.prefix_tree_conflict_rows == 1
    assert stats.prefix_tree_conflict_slots == 4


def test_prefix_tree_pack_keeps_trainable_duplicates_in_leaf_metadata() -> None:
    first = _tokenized(
        [10, 11, 20, 21],
        [_route(0), _route(10), _route(20), _route(30)],
        prompt_id=123,
        prompt_length=1,
        trainable_start=2,
        advantage=2.0,
        weight=0.5,
        pixel_values=torch.ones(1, 2),
        image_grid_thw=torch.tensor([[1, 2, 3]]),
    )
    second = _tokenized(
        [10, 11, 20, 22],
        [_route(0), _route(10), _route(40), _route(50)],
        prompt_id=123,
        prompt_length=1,
        trainable_start=2,
        advantage=4.0,
        weight=0.25,
        pixel_values=torch.full((1, 2), 2.0),
        image_grid_thw=torch.tensor([[4, 5, 6]]),
    )

    packed = packed_tensors_from_tokenized_results(
        [first, second],
        seq_len=8,
        pad_token_id=0,
        truncate_long_results=False,
    )

    assert packed["tokens"].tolist()[0][:7] == [10, 11, 20, 21, 11, 20, 22]
    assert packed["input_pos"].tolist()[0][:7] == [0, 1, 2, 3, 1, 2, 3]
    assert packed["assistant_mask"].tolist()[0][:7] == [
        False,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert math.isnan(float(packed["logprobs"][0, 1]))
    assert float(packed["logprobs"][0, 2]) == -1.0
    assert float(packed["logprobs"][0, 5]) == -1.0
    assert float(packed["advantages"][0, 2]) != float(packed["advantages"][0, 5])
    assert float(packed["weights"][0, 2]) != float(packed["weights"][0, 5])
    assert int(packed["group_ids"][0, 2]) != int(packed["group_ids"][0, 5])
    pixel_values = packed["pixel_values"][0]
    image_grid_thw = packed["image_grid_thw"][0]
    assert pixel_values is not None
    assert image_grid_thw is not None
    assert torch.equal(pixel_values, torch.ones(1, 2))
    assert torch.equal(image_grid_thw, torch.tensor([[1, 2, 3]]))


def test_prefix_tree_pack_public_api_emits_nested_metadata() -> None:
    results = [
        _tokenized(
            [10, 11, 20, 101, 201],
            [_route(0), _route(10), _route(20), _route(30), _route(40)],
            prompt_id=1,
            prompt_length=4,
            trainable_start=4,
        ),
        _tokenized(
            [10, 11, 20, 102, 202],
            [_route(0), _route(10), _route(20), _route(50), _route(60)],
            prompt_id=2,
            prompt_length=4,
            trainable_start=4,
        ),
        _tokenized(
            [10, 12, 30, 103, 203],
            [_route(0), _route(70), _route(80), _route(90), _route(100)],
            prompt_id=3,
            prompt_length=4,
            trainable_start=4,
        ),
    ]

    packed = packed_tensors_from_tokenized_results(
        results,
        seq_len=16,
        pad_token_id=0,
        truncate_long_results=False,
    )
    tree = parse_prefix_tree_row(
        group_ids=packed["group_ids"][0],
        parent_ids=packed["parent_ids"][0],
    )

    assert packed["tokens"].tolist()[0][: tree.valid_tokens] == [
        10,
        11,
        20,
        101,
        201,
        102,
        202,
        12,
        30,
        103,
        203,
    ]
    assert max(segment.depth for segment in tree.segments) == 2
    assert not packed["assistant_mask"][0, 2]
    assert not packed["assistant_mask"][0, 3]
    assert packed["assistant_mask"][0, 4]
    assert packed["assistant_mask"][0, 6]
    assert int(packed["group_ids"][0, 4]) != int(packed["group_ids"][0, 6])


def test_pack_infers_at_least_topk_experts_from_sparse_routes() -> None:
    result = _tokenized(
        [10, 20],
        [[[0, 0, 0, 0]], [[0, 0, 0, 0]]],
        prompt_id=456,
        prompt_length=1,
    )

    packed = packed_tensors_from_tokenized_results(
        [result],
        seq_len=4,
        pad_token_id=0,
        truncate_long_results=False,
        include_moe_routing=True,
    )

    routing_replay = packed["moe_routing_replay"]
    assert routing_replay is not None
    assert routing_replay.topk == 4
    assert routing_replay.num_experts == 4


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
