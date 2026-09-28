"""Bound list materialization independently of the tensor element inventory."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
from test_planner_input_snapshots import _plan, _snapshot
import torch


def _labels(plan, tensor):
    group = plan.groups[0]
    return replace(
        plan, groups=(replace(group, items=(replace(group.items[0], labels=tensor),)),)
    )


def test_zero_element_shape_does_not_materialize_unbounded_lists(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path)
    tensor = torch.empty((4097, 0), dtype=torch.long)
    assert tensor.numel() == 0
    payload = _snapshot(rank, _labels(plan, tensor))()
    value = payload["requests"][0]["target_tokens"]
    assert isinstance(value, dict), f"zero elements emitted {len(value)} outer lists"
    assert value["unavailable"] == "token_structure_over_limit"
    assert payload["layouts"]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize(
    "shape, admitted",
    [
        ((), True),
        ((0,), True),
        ((0, 4097), True),
        ((1, 0, 4097), True),
        ((4094, 0), True),
        ((4095, 0), False),
        ((4097, 0), False),
        ((2, 2046, 0), True),
        ((2, 2047, 0), False),
        ((1,) * 32, True),
        ((8, 8, 8, 8, 0), False),
    ],
)
def test_list_structure_budget_preserves_supported_values(
    monkeypatch, tmp_path, inference, shape, admitted
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    with torch.inference_mode() if inference else nullcontext():
        tensor = torch.zeros(shape, dtype=torch.long)
    payload = _snapshot(rank, _labels(plan, tensor))()
    value = payload["requests"][0]["target_tokens"]
    if admitted:
        assert value == tensor.tolist()
    else:
        assert value["unavailable"] == "token_structure_over_limit"
        assert value["shape"] == list(shape)
        assert "token_structure_over_limit" in payload["incomplete_reasons"]
    assert payload["layouts"]  # Valid input IDs remain usable independently.


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("shape", [(2**32, 0), (2**31, 2**31, 0)])
def test_structure_refusal_precedes_materialization(
    monkeypatch, tmp_path, inference, shape
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    with torch.inference_mode() if inference else nullcontext():
        tensor = torch.empty(shape, dtype=torch.long)
    assert tensor.numel() == tensor.untyped_storage().nbytes() == 0
    replay = _snapshot(rank, _labels(plan, tensor))
    tolist = torch.Tensor.tolist

    def bounded(value, *args, **kwargs):
        assert tuple(value.shape) != shape, "unbounded list materialization"
        return tolist(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", bounded)
    payload = replay()
    assert (
        payload["requests"][0]["target_tokens"]["unavailable"]
        == "token_structure_over_limit"
    )


def test_list_structure_budget_is_shared_across_requests(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path)
    group = plan.groups[0]
    item = replace(group.items[0], labels=torch.empty((2047, 0), dtype=torch.long))
    plan = replace(plan, groups=(replace(group, items=(item, item)),))
    payload = _snapshot(rank, plan)()
    first, second = payload["requests"]
    assert first["target_tokens"] == [[]] * 2047
    assert second["target_tokens"]["unavailable"] == "token_structure_over_limit"
