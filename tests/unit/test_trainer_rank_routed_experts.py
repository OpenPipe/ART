from contextlib import nullcontext
from dataclasses import replace
from enum import Enum
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.megatron.routed_experts import (
    CURRENT_ROUTES,
    RoutingContext,
    _routing,
    capture_routes,
    prepare_routes,
    set_routing_layout,
    use_routes,
    validate_routes,
)
from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput, TrainerRank
from art.trainer_rank._impl import _ForwardItem


def router(topk=1, layers=1):
    return SimpleNamespace(
        topk=topk,
        config=SimpleNamespace(
            num_moe_experts=4, num_layers=layers, sequence_parallel=False
        ),
        router_replay=SimpleNamespace(target_topk_idx=None, router_replay_action=None),
    )


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(3, 1, 2),
        torch.full((3, 1, 2), -1),
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.tensor([[[0, 4]]] * 3),
        torch.tensor([[[1, 1]]] * 3),
    ],
)
def test_replay_rejects_missing_misaligned_and_invalid_experts(bad):
    with pytest.raises(ValueError):
        validate_routes(bad, 3, 1, [(router(topk=2), 0)])


def test_replay_uses_all_model_layers_and_owns_input_snapshot():
    routes = torch.tensor([[[0, 0], [1, 2]], [[0, 0], [2, 3]]])
    owned = validate_routes(routes, 2, 2, [(router(topk=2), 1)])
    routes.fill_(-1)
    assert owned[:, 1].tolist() == [[1, 2], [2, 3]]


def test_forward_routes_align_with_inputs_without_shift_and_group_separately():
    rank = TrainerRank.__new__(TrainerRank)
    rank.runtime = SimpleNamespace(provider=SimpleNamespace(num_layers=1))
    rank._routing_bindings = [(router(), 0)]
    rank._ensure_checkpoint_slots_for = lambda *_args, **_kwargs: None
    rank._resolve_slot_ref = lambda *_args, **_kwargs: None
    tokens = torch.tensor([1, 2, 3])
    routes = torch.tensor([[[0]], [[1]], [[2]]])
    replay = ForwardInput(
        input_tokens=tokens, routed_experts=routes, hidden_states=True
    )
    normal = ForwardInput(input_tokens=tokens, hidden_states=True)
    item = rank._forward_item(replay)
    assert torch.equal(item.routed_experts, routes)
    strict = replace(replay, options=ForwardOptions(max_gradient_staleness=0))
    groups = rank._group_active_request_indices([replay, normal, replay, strict])
    assert [indices for _key, indices in groups] == [(0, 2), (1,), (3,)]


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
def test_cached_forward_and_replay_preserve_owned_routes(monkeypatch, retention):
    from trainer_rank_test_support import checkpoint_runtime

    model = torch.nn.Linear(1, 1, bias=False)
    rank = TrainerRank(checkpoint_runtime(model))
    binding, seen = router(), []
    rank._routing_bindings = [(binding, 0)]
    monkeypatch.setattr(rank, "_resolve_slot_ref", lambda *_a, **_kw: None)
    lora = SimpleNamespace(use_lora_slot=lambda _slot, **_kwargs: nullcontext())
    monkeypatch.setitem(sys.modules, "art.megatron.lora", lora)
    monkeypatch.setattr(
        rank, "_topology", lambda: SimpleNamespace(dp=1, tp=1, cp=1, pp=1)
    )
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_configure_hybridep", lambda *a, **kw: None)
    monkeypatch.setattr(
        rank,
        "_prepare_packed_forward",
        lambda _: SimpleNamespace(token_uids=None, attention_state=None),
    )

    def forward(items, prepared):
        context = CURRENT_ROUTES.get()
        assert context is not None
        routes = context.targets[id(binding)]["attention"].flatten()
        seen.append(routes.tolist())
        return [ForwardOutput(None, None, None, model.weight.sum() * routes.float())]

    monkeypatch.setattr(rank, "_forward_packed", forward)
    routes = torch.tensor([[[1]], [[2]]])
    output = rank.forward(
        ForwardInput(
            input_tokens=torch.tensor([1, 2]),
            routed_experts=routes,
            hidden_states=True,
            options=ForwardOptions(backward_state=retention),
        )
    )
    assert CURRENT_ROUTES.get() is None
    routes.fill_(3)
    rank.backward(output.hidden_states.sum())
    assert seen == [[1, 2]] * (2 if retention == "replay" else 1)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[3.0]]))
    assert CURRENT_ROUTES.get() is None and not rank._forward_graph_cache().handles()


def test_prefix_routes_use_reference_sequence_and_cp_padding_is_explicit():
    tokens = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])]
    routes = [torch.tensor([[[0]], [[1]], [[2]]]), torch.tensor([[[2]], [[3]], [[3]]])]
    items = [
        _ForwardItem(ForwardInput(input_tokens=t), t, None, r)
        for t, r in zip(tokens, routes)
    ]
    packed = prefix_tree_pack(tokens, max_depth=2)
    binding = router()
    prepared = SimpleNamespace(
        token_uids=torch.tensor([[2, 0, -1, 3, 1]]),
        attention_state=SimpleNamespace(
            gdn_execution_plan=SimpleNamespace(gdn_token_indices=[3, 0, 1])
        ),
    )
    context = prepare_routes(items, packed, prepared, [(binding, 0)], "cpu")
    assert context is not None
    assert context.targets[id(binding)]["attention"].flatten().tolist() == [
        2,
        0,
        0,
        3,
        1,
    ]
    assert context.targets[id(binding)]["gdn"].flatten().tolist() == [3, 0, 1]


@pytest.mark.parametrize("reentrant", [False, True])
def test_two_live_graphs_recompute_their_own_routes_and_layout(monkeypatch, reentrant):
    class Action(Enum):
        REPLAY_FORWARD = 1

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.moe.router_replay",
        SimpleNamespace(RouterReplayAction=Action),
    )
    binding = router()

    def route(logits):
        replay = binding.router_replay
        assert replay.router_replay_action is Action.REPLAY_FORWARD
        return logits.softmax(-1).gather(1, replay.target_topk_idx).square().sum()

    binding._art_rank_original_routing = route
    values = torch.tensor([[0.3, 0.7, 0.1, 0.2]], requires_grad=True)
    outputs = []
    for expert in (0, 2):
        context = RoutingContext(
            {
                id(binding): {
                    "attention": torch.tensor([[1]]),
                    "gdn": torch.tensor([[expert]]),
                }
            }
        )
        with use_routes(context):
            set_routing_layout("gdn")
            function = capture_routes(lambda x: _routing(binding, x))
            # Later layout changes must not alter the captured checkpoint.
            set_routing_layout("attention")
            outputs.append(checkpoint(function, values, use_reentrant=reentrant))
    assert CURRENT_ROUTES.get() is None
    assert binding.router_replay.target_topk_idx is None
    sum(outputs).backward()
    expected = values.detach().clone().requires_grad_()
    expected.softmax(-1)[:, [0, 2]].square().sum().backward()
    torch.testing.assert_close(values.grad, expected.grad)
    assert binding.router_replay.target_topk_idx is None
    assert binding.router_replay.router_replay_action is None
