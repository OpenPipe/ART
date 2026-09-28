"""Nested tensor diagnostics must not discard the surrounding planner report."""

from contextlib import nullcontext
from dataclasses import replace
import sys

import pytest
from test_planner_input_snapshots import _plan, _snapshot
import torch

from art.trainer_rank._planner_misses import validate_report


def with_labels(plan, labels):
    group = plan.groups[0]
    return replace(
        plan,
        groups=(replace(group, items=(replace(group.items[0], labels=labels),)),),
    )


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("depth", [32, 64, 128, 129, 1024, 2000])
def test_tensor_nesting_preserves_complete_surrounding_report(
    monkeypatch, tmp_path, inference, depth
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    original = plan.groups[0].items[0].input_ids.tolist()
    with torch.inference_mode() if inference else nullcontext():
        labels = torch.ones((1,) * depth, dtype=torch.long)
    assert labels.numel() == 1 and labels.untyped_storage().nbytes() == 8
    before_limit = sys.getrecursionlimit()
    replay = _snapshot(rank, with_labels(plan, labels))
    path = rank._planner_reporter.report(
        predicted_peak_bytes=1000,
        observed_peak_bytes=2000,
        phase="forward",
        replay_factory=replay,
    )
    assert path is not None
    report = validate_report(path.read_bytes())
    assert report["replay"] is not None, report["incomplete_reasons"]
    payload = report["replay"]
    assert payload["requests"][0]["input_tokens"] == original
    assert payload["layouts"]
    value = payload["requests"][0]["target_tokens"]
    if depth <= 128:
        assert value == labels.tolist()
        assert "token_nesting_over_limit" not in report["incomplete_reasons"]
    else:
        assert value == {
            "unavailable": "token_nesting_over_limit",
            "shape": [1] * depth,
            "dtype": "torch.int64",
            "device": "cpu",
        }
        assert "token_nesting_over_limit" in report["incomplete_reasons"]
        assert report["replay_complete"] is False
    assert sys.getrecursionlimit() == before_limit
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("shape", [(), (3,), (0,) + (1,) * 1999, (1, 0) + (1,) * 1998])
def test_scalar_flat_and_early_empty_dimensions_remain_available(
    monkeypatch, tmp_path, shape
):
    rank, plan = _plan(monkeypatch, tmp_path)
    labels = torch.zeros(shape, dtype=torch.long)
    payload = _snapshot(rank, with_labels(plan, labels))()
    assert payload["requests"][0]["target_tokens"] == labels.tolist()
    assert "token_nesting_over_limit" not in payload["incomplete_reasons"]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("field", ["input_ids", "labels"])
def test_depth_refusal_precedes_tolist(monkeypatch, tmp_path, inference, field):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    with torch.inference_mode() if inference else nullcontext():
        labels = torch.ones((1,) * 1024, dtype=torch.long)
    group = plan.groups[0]
    plan = replace(
        plan,
        groups=(replace(group, items=(replace(group.items[0], **{field: labels}),)),),
    )
    replay = _snapshot(rank, plan)
    tolist = torch.Tensor.tolist

    def guarded(tensor, *args, **kwargs):
        assert tensor.ndim != 1024, "deep tensor materialized before refusal"
        return tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", guarded)
    key = "input_tokens" if field == "input_ids" else "target_tokens"
    assert replay()["requests"][0][key]["unavailable"] == ("token_nesting_over_limit")
