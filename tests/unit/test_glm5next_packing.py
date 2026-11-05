"""The GLM MVP cannot admit prefix trees, even under memory pressure."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank
import torch

from art.megatron.glm5next.packing import pad_single_roots
from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER
from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import ForwardInput


@pytest.mark.parametrize("cp", [1, 2, 8])
@pytest.mark.parametrize("memory_minimal", [False, True])
def test_planner_emits_independent_aligned_roots(monkeypatch, cp, memory_minimal):
    rank = _rank()
    rank.runtime.model_support_handler = GLM5NEXT_HANDLER
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 1, cp, 1))
    monkeypatch.setattr(rank, "_forced_test_anchor", lambda: "full_sharing")
    rows = [torch.tensor([154820, 2, 3, 4, 5]), torch.tensor([154820, 2, 3])]
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=row, hidden_states=True) for row in rows],
        memory_minimal=memory_minimal,
    )
    packed = plan.groups[0].packed
    alignment = 4 if cp == 1 else 8 * cp
    assert plan.packed_tokens == sum(len(row) + (-len(row) % alignment) for row in rows)
    assert len(packed.segments) == 2
    assert torch.equal(packed.group_ids, packed.parent_ids)
    cursor = 0
    for segment in packed.segments:
        assert len(segment.sequence_indices) == 1 and segment.start == 0
        i = segment.sequence_indices[0]
        assert segment.packed_start == cursor
        positions = packed.positions_by_sequence[i]
        assert torch.equal(packed.tokens[0, positions], rows[i])
        assert packed.position_ids[0, positions[0]] == 0
        assert len(positions) == len(rows[i])  # tails never select outputs/loss
        padded = len(rows[i]) + (-len(rows[i]) % alignment)
        tail = packed.tokens[0, cursor + len(rows[i]) : cursor + padded]
        assert bool((tail == 0).all())
        cursor += padded


@pytest.mark.parametrize("cp", [False, True])
def test_tree_never_reaches_dsa(cp):
    groups, parents = torch.tensor([[1, 1, 2, 2]]), torch.tensor([[1, 1, 1, 1]])
    context = SimpleNamespace(
        input_pos=torch.arange(4)[None],
        group_ids=groups if not cp else torch.tensor([[2, 2]]),
        parent_ids=parents if not cp else torch.tensor([[2, 2]]),
        context_parallel_state=SimpleNamespace(
            group_ids=groups[0], parent_ids=parents[0]
        )
        if cp
        else None,
    )
    with patch("art.megatron.glm5next.dsa_cp.build_glm5next_dsa_state") as dsa:
        with pytest.raises(ValueError, match="single-root"):
            GLM5NEXT_HANDLER.build_prefix_tree_model_state(context)
        dsa.assert_not_called()


def test_unaligned_datums_build_native_dsa_state():
    from art.megatron.glm5next.dsa_cp import build_glm5next_dsa_state

    rows = [torch.tensor([154820, 2, 3, 4, 5]), torch.tensor([154820, 2, 3])]
    packed = pad_single_roots(prefix_tree_pack(rows, max_depth=0), cp_size=1)
    state = build_glm5next_dsa_state(
        position_ids=packed.position_ids,
        group_ids=packed.group_ids,
        parent_ids=packed.parent_ids,
        device=torch.device("cpu"),
    )
    assert state is not None


@pytest.mark.parametrize("layer_number", [1, 4])
def test_branching_never_reaches_kda_or_dsa_layer(layer_number):
    from art.megatron.glm5next.layer import Glm5NextTransformerLayer

    layer = Glm5NextTransformerLayer.__new__(Glm5NextTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = layer_number
    layer.config = SimpleNamespace(cuda_graph_impl="none")

    class ForbiddenAttention(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("tree reached attention")

    layer.self_attention = ForbiddenAttention()
    state = SimpleNamespace(
        group_ids=torch.tensor([[0, 0, 1, 1]]),
        parent_ids=torch.tensor([[-1, -1, 0, 0]]),
    )
    with pytest.raises(ValueError, match="single-root"):
        layer(torch.zeros(4, 1, 4, 16), attention_bias=state)
