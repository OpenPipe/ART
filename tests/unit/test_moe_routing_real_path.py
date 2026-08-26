from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import numpy as np
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
import pytest
import torch

from art.distributed.data_plane import SharedMemoryPackedBatchStore
from art.distributed.packing import TrajectoryPayload
from art.megatron.prefix_tree import parse_prefix_tree_row
from art.megatron.routing_replay import (
    build_moe_routing_replay_bundle_from_packed_tensors,
)
from art.preprocessing.moe_routing import (
    ART_MOE_ROUTING_METADATA_KEY,
    NUM_EXPERTS_KEY,
    ROUTED_EXPERTS_KEY,
    MoeRouteArray,
    MoeRouteSegments,
    align_choice_routes_to_tokenized_result,
)
from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import TokenizedResult
from art.trajectories import ChatCompletionsExchange, Trajectory


class _FakeTokenizer:
    def decode(self, token_id: int) -> str:
        return str(token_id)


def _choice(metadata: dict[str, Any]) -> Choice:
    metadata.setdefault("num_experts", 256)
    return Choice.model_validate(
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "x"},
            ART_MOE_ROUTING_METADATA_KEY: metadata,
        }
    )


def _route(seed: int) -> list[list[int]]:
    seed %= 240
    return [[seed, seed + 1], [seed + 2, seed + 3]]


def _routes_to_list(routes: Any) -> list[Any]:
    if hasattr(routes, "segments"):
        output: list[Any] = []
        for segment in routes.segments:
            output.extend(segment.tolist())
        return output
    return routes.tolist()


def test_align_choice_routes_keeps_binary_route_views() -> None:
    combined = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    combined.flags.writeable = False
    routes, _ = align_choice_routes_to_tokenized_result(
        token_ids=[10, 11, 20, 21],
        choices=[
            _choice(
                {
                    "prompt_token_ids": [10, 11],
                    "completion_token_ids": [20, 21],
                    "routed_experts": combined,
                }
            )
        ],
        choice_offsets=[2],
        choice_token_lengths=[2],
    )

    assert isinstance(routes, MoeRouteSegments)
    assert all(np.shares_memory(segment, combined) for segment in routes.segments)


def test_align_choice_routes_to_tokenized_result_rejects_token_mismatch() -> None:
    with pytest.raises(RuntimeError, match="prompt token ids do not match"):
        align_choice_routes_to_tokenized_result(
            token_ids=[10, 12, 20],
            choices=[
                _choice(
                    {
                        "prompt_token_ids": [10, 11],
                        "completion_token_ids": [20],
                        "routed_experts": np.asarray(
                            [_route(0), _route(10), _route(20)], dtype=np.uint8
                        ),
                    }
                )
            ],
            choice_offsets=[2],
            choice_token_lengths=[1],
        )


def test_align_choice_routes_materializes_missing_terminal_route() -> None:
    routes, _stats = align_choice_routes_to_tokenized_result(
        token_ids=[10, 20],
        choices=[
            _choice(
                {
                    "prompt_token_ids": [10],
                    "completion_token_ids": [20],
                    "routed_experts": np.asarray([_route(0)], dtype=np.uint8),
                }
            )
        ],
        choice_offsets=[1],
        choice_token_lengths=[1],
    )

    assert routes is not None
    materialized = _routes_to_list(routes)
    assert materialized[0] == _route(0)
    assert all(len(set(layer)) == 2 for layer in materialized[1])


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
    num_experts: int = 256,
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
        moe_routed_experts=MoeRouteArray(
            np.asarray(
                routes,
                dtype=np.uint8 if num_experts <= 256 else np.uint16,
            ),
            num_experts=num_experts,
        ),
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
        [_route(0), _route(10), _route(40), _route(50)],
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
        min_prefix_tree_shared_segment_length=0,
    )

    assert packed["tokens"].tolist()[0][:7] == [10, 11, 20, 21, 11, 22, 23]
    routing_replay = packed["moe_routing_replay"]
    assert routing_replay is not None
    assert torch.movedim(routing_replay.expert_indices[:, 0], 0, 1).tolist()[:7] == [
        _route(0),
        _route(10),
        _route(20),
        _route(30),
        _route(10),
        _route(40),
        _route(50),
    ]
    assert routing_replay.pack_stats.packed_tokens == 7


def test_pack_uses_reference_routes_for_shared_prefix() -> None:
    first = _tokenized(
        [10, 11, 20],
        [_route(0), _route(10), _route(20)],
        prompt_id=123,
        prompt_length=2,
    )
    second = _tokenized(
        [10, 11, 21],
        [_route(90), _route(10), _route(30)],
        prompt_id=123,
        prompt_length=2,
    )

    packed = packed_tensors_from_tokenized_results(
        [first, second],
        seq_len=8,
        truncate_long_results=False,
        include_moe_routing=True,
        min_prefix_tree_shared_segment_length=0,
    )

    replay = packed["moe_routing_replay"]
    assert replay is not None
    routes = torch.movedim(replay.expert_indices[:, 0], 0, 1).tolist()
    assert routes[:5] == [
        _route(0),
        _route(10),
        _route(20),
        _route(10),
        _route(30),
    ]


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
        min_prefix_tree_shared_segment_length=0,
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
        min_prefix_tree_shared_segment_length=0,
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


def test_prefix_tree_pack_best_fit_combines_independent_small_groups() -> None:
    results = []
    for group in range(4):
        prompt = [10 + group, 100, 200 + group]
        for sample in range(2):
            token_ids = [*prompt, 300 + group * 10 + sample]
            results.append(
                _tokenized(
                    token_ids,
                    [_route(token) for token in token_ids],
                    prompt_id=group,
                    prompt_length=3,
                    trainable_start=3,
                )
            )

    packed = packed_tensors_from_tokenized_results(
        results,
        seq_len=12,
        pad_token_id=0,
        truncate_long_results=False,
        min_prefix_tree_shared_segment_length=0,
    )

    assert packed["tokens"].shape[0] == 2
    assert int((packed["group_ids"] != -1).sum().item()) == 24


@pytest.mark.parametrize(
    ("num_experts", "dtype"),
    [(256, torch.uint8), (257, torch.uint16)],
)
def test_pack_preserves_exact_expert_count_and_smallest_dtype(
    num_experts: int, dtype: torch.dtype
) -> None:
    result = _tokenized(
        [10, 20],
        [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
        prompt_id=456,
        prompt_length=1,
        num_experts=num_experts,
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
    assert routing_replay.num_experts == num_experts
    assert routing_replay.expert_indices.dtype == dtype
    assert routing_replay.expert_indices.shape == (1, 1, 4, 4)
    assert all(
        len(set(row)) == 4 for row in routing_replay.expert_indices[0, 0].tolist()
    )


def test_build_replay_bundle_retains_layer_major_storage() -> None:
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

    replay = packed["moe_routing_replay"]
    assert replay is not None
    assert bundle.tensor_backed
    assert bundle.steps == {}
    assert bundle.expert_indices is replay.expert_indices
    assert bundle.expert_indices[0, 0].tolist()[:3] == [
        [0, 1],
        [10, 11],
        [20, 21],
    ]
    assert len(set(bundle.expert_indices[0, 0, 3].tolist())) == 2


def test_trajectory_route_roundtrip_preserves_exact_contract() -> None:
    routes = MoeRouteArray(
        np.asarray([[[0, 256]], [[255, 1]]], dtype=np.uint16),
        num_experts=257,
    )
    trajectory = Trajectory(
        messages_and_choices=[
            _choice(
                {
                    "prompt_token_ids": [10],
                    "completion_token_ids": [20],
                    ROUTED_EXPERTS_KEY: routes,
                    NUM_EXPERTS_KEY: 257,
                }
            )
        ]
    )

    restored = TrajectoryPayload.from_trajectory(trajectory).build()
    choice = restored.messages_and_choices[0]
    assert isinstance(choice, Choice)
    metadata = (choice.model_extra or {})[ART_MOE_ROUTING_METADATA_KEY]
    restored_routes = metadata[ROUTED_EXPERTS_KEY]
    assert isinstance(restored_routes, MoeRouteArray)
    assert restored_routes.num_experts == 257
    assert restored_routes.dtype == np.dtype(np.uint16)
    assert np.array_equal(restored_routes, routes)


def test_exchange_route_roundtrip_preserves_exact_contract() -> None:
    routes = MoeRouteArray(
        np.asarray([[[0, 256]], [[255, 1]]], dtype=np.uint16),
        num_experts=257,
    )
    response = ChatCompletion(
        id="route-test",
        choices=[_choice({ROUTED_EXPERTS_KEY: routes, NUM_EXPERTS_KEY: 257})],
        created=0,
        model="test-model",
        object="chat.completion",
    )
    now = datetime.now()
    trajectory = Trajectory(
        exchanges={
            "chat_completions": [
                ChatCompletionsExchange(
                    request={"model": "test-model", "messages": []},
                    response=response,
                    start_time=now,
                    end_time=now,
                )
            ]
        }
    )

    restored = TrajectoryPayload.from_trajectory(trajectory).build()
    choice = restored.exchanges.chat_completions[0].response.choices[0]
    restored_routes = (choice.model_extra or {})[ART_MOE_ROUTING_METADATA_KEY][
        ROUTED_EXPERTS_KEY
    ]
    assert restored_routes.num_experts == 257
    assert np.array_equal(restored_routes, routes)


def test_shm_replay_is_one_layer_major_uint16_tensor() -> None:
    packed = packed_tensors_from_tokenized_results(
        [
            _tokenized(
                [10, 20],
                [[[0, 256]], [[255, 1]]],
                prompt_id=456,
                prompt_length=1,
                num_experts=257,
            )
        ],
        seq_len=4,
        pad_token_id=0,
        truncate_long_results=False,
        include_moe_routing=True,
    )
    store = SharedMemoryPackedBatchStore(
        owner_actor_id="test-owner", capacity_bytes=1 << 20
    )
    try:
        ref = store.create(packed, batch_id="route-batch")
        replay_specs = [
            spec for spec in ref.tensors if spec.name.startswith("moe_routing_replay/")
        ]
        assert [spec.name for spec in replay_specs] == [
            "moe_routing_replay/expert_indices"
        ]
        assert replay_specs[0].dtype == "uint16"
        assert ref.moe_routing_replay is not None
        assert ref.moe_routing_replay.num_experts == 257
        assert ref.moe_routing_replay.packed_tokens == 2

        with store.map(ref) as mapped:
            replay = mapped.tensors["moe_routing_replay"]
            assert replay.expert_indices.shape == (1, 1, 4, 2)
            assert replay.expert_indices.dtype == torch.uint16
            bundle = build_moe_routing_replay_bundle_from_packed_tensors(
                packed_tensors=mapped.tensors,
                global_grad_accumulation_sequences=1,
            )
            assert bundle.expert_indices is replay.expert_indices
    finally:
        store.close()
