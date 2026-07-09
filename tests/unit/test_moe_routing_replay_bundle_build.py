from __future__ import annotations

import pytest
import torch

from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    build_moe_routing_replay_bundle_from_packed_tensors,
)
from art.preprocessing.moe_routing import (
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
)


def _packed(
    expert_indices: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    num_experts: int,
) -> dict[str, object]:
    num_sequences, sequence_length, num_layers, topk = expert_indices.shape
    group_ids = torch.full((num_sequences, sequence_length), -1, dtype=torch.int64)
    parent_ids = torch.full_like(group_ids, -1)
    group_ids[token_mask] = 11
    parent_ids[token_mask] = 11
    return {
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "moe_routing_replay": PackedMoeRoutingReplay(
            expert_indices=expert_indices,
            token_mask=token_mask,
            num_layers=num_layers,
            topk=topk,
            num_experts=num_experts,
            pack_stats=MoeRoutingPackStats(packed_tokens=int(token_mask.sum().item())),
        ),
    }


def test_fast_bundle_build_roundtrips_expanded_full_masks(tmp_path) -> None:
    expert_indices = torch.tensor(
        [[[[0, 1], [2, 3]], [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[0, 0], [0, 0]]]],
        dtype=torch.int32,
    )
    token_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    bundle = build_moe_routing_replay_bundle_from_packed_tensors(
        packed_tensors=_packed(expert_indices, token_mask, num_experts=8),  # type: ignore[arg-type]
        global_grad_accumulation_sequences=1,
    )

    route = bundle.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    assert route.expert_indices[:3].tolist() == [[0, 1], [1, 2], [2, 3]]
    assert route.expert_mask.shape == route.expert_indices.shape
    assert bool(route.expert_mask.all().item())
    assert all(
        0 <= expert_id < route.num_experts
        for expert_id in route.expert_indices[3].tolist()
    )
    assert len(set(route.expert_indices[3].tolist())) == route.max_topk

    bundle.to_dir(tmp_path)
    loaded = MoeRoutingReplayBundle.from_dir(tmp_path)
    loaded_route = loaded.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    assert torch.equal(loaded_route.expert_indices, route.expert_indices)
    assert bool(loaded_route.expert_mask.all().item())


def test_fast_bundle_build_rejects_invalid_routed_expert_id() -> None:
    expert_indices = torch.tensor(
        [[[[0, 1]], [[1, 9]], [[2, 3]], [[0, 0]]]],
        dtype=torch.int32,
    )
    token_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    with pytest.raises(RuntimeError, match="outside \\[0, num_experts\\)"):
        build_moe_routing_replay_bundle_from_packed_tensors(
            packed_tensors=_packed(expert_indices, token_mask, num_experts=8),  # type: ignore[arg-type]
            global_grad_accumulation_sequences=1,
        )
