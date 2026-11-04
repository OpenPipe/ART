from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron import prefix_tree_state
from art.megatron.segmented_expert_lora import _attention_forward
from art.megatron.segmented_lora import (
    _CURRENT_LORA_SEGMENTS,
    LoRASegments,
    use_lora_segments,
)


@pytest.mark.parametrize("counts", [(5, 0, 4), (2, 0, 7)])
def test_attention_isolates_multiple_independent_roots_per_adapter(monkeypatch, counts):
    groups = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3]])
    state = SimpleNamespace(group_ids=groups, parent_ids=groups.clone(), model_state={})
    segments = LoRASegments(cast(Any, ("first", "idle", "second")), counts)
    calls = []

    def build(groups, parents, **kwargs):
        assert torch.equal(kwargs["input_pos"], torch.arange(groups.numel())[None])
        return SimpleNamespace(group_ids=groups, parent_ids=parents)

    monkeypatch.setattr(prefix_tree_state, "create_prefix_tree_state", build)

    def original(q, k, v, mask, *, attention_bias, **kwargs):
        assert torch.equal(attention_bias.group_ids, attention_bias.parent_ids)
        assert attention_bias.group_ids.unique().numel() == 1
        current = _CURRENT_LORA_SEGMENTS.get()
        assert current is not None and sum(n > 0 for n in current.counts) == 1
        calls.append(len(q))
        return (q + k + v).cumsum(0)

    q = torch.arange(9.0).reshape(9, 1, 1, 1).requires_grad_()
    k, v = torch.ones_like(q), torch.ones_like(q)
    expected = torch.cat([part.cumsum(0) for part in (q + k + v).split((2, 3, 2, 2))])
    with use_lora_segments(segments):
        actual = _attention_forward(
            SimpleNamespace(), original, q, k, v, None, attention_bias=state
        )
        again = _attention_forward(
            SimpleNamespace(), original, q, k, v, None, attention_bias=state
        )
        assert _CURRENT_LORA_SEGMENTS.get() is segments
    assert calls == [2, 3, 2, 2] * 2
    assert torch.equal(actual, expected) and torch.equal(again, expected)
    cotangent = torch.cat([torch.zeros(5), torch.ones(4)]).reshape_as(actual)
    (actual_grad,) = torch.autograd.grad(actual, q, cotangent, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, q, cotangent)
    assert torch.equal(actual_grad, expected_grad)
    assert not actual_grad[:5].count_nonzero()
    assert not torch.cuda.is_initialized()


def test_attention_refuses_root_shared_across_adapters(monkeypatch):
    groups = torch.tensor([[0, 0, 1, 1]])
    state = SimpleNamespace(group_ids=groups, parent_ids=groups.clone(), model_state={})
    segments = LoRASegments(cast(Any, ("first", "second")), (1, 3))
    monkeypatch.setattr(
        prefix_tree_state, "create_prefix_tree_state", lambda *a, **k: None
    )
    q = torch.ones(4, 1, 1, 1)
    with (
        use_lora_segments(segments),
        pytest.raises(ValueError, match="independent unshared"),
    ):
        _attention_forward(
            SimpleNamespace(), lambda *a, **k: q, q, q, q, None, attention_bias=state
        )
