"""Bounded planner diagnostics use actual CPU plans, never a device readback."""

from contextlib import nullcontext
from dataclasses import replace
import gc
import weakref

import pytest
from test_trainer_rank_split import _rank, _request
import torch

from art.trainer_rank import _impl as tr
from art.trainer_rank._planner_misses import Reporter, validate_report


def _plan(monkeypatch, tmp_path, *, inference=False):
    rank = _rank(monkeypatch)
    rank._planner_reporter = Reporter(10, spool_dir=tmp_path)
    with torch.inference_mode() if inference else nullcontext():
        request = _request(0)
    return rank, rank._plan_flat_forward([request])


def _snapshot(rank, plan):
    rank._begin_planner_observation(plan, tr._MemoryCheck(1000, 2000, True))
    observation = rank._planner_observation
    assert observation is not None and observation["comparable"]
    return observation["replay"]


@pytest.mark.parametrize("active_inference", [False, True])
def test_inference_cpu_inputs_retain_original_layout(
    monkeypatch, tmp_path, active_inference
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=True)
    original = plan.groups[0].items[0].input_ids.tolist()
    with torch.inference_mode(active_inference):
        replay = _snapshot(rank, plan)
    payload = replay()
    assert payload["requests"][0]["input_tokens"] == original
    assert payload["requests"][0]["target_tokens"] == original
    assert len(payload["layouts"]) == 1
    assert (
        payload["layouts"][0]["expected_fingerprint"]
        == plan.groups[0].layout.fingerprint
    )
    assert payload["incomplete_reasons"] == [
        "immutable runtime group/slot, head, checkpoint and GDN facts"
    ]
    rank._planner_reporter.report(
        predicted_peak_bytes=1000,
        observed_peak_bytes=2000,
        phase="forward",
        replay_factory=replay,
    )
    [path] = list(tmp_path.glob("*.json"))
    assert validate_report(path.read_bytes())["replay_complete"] is False
    rank.finish_planner_observation()


@pytest.mark.parametrize("inference", [False, True])
def test_modified_input_is_not_replaced_by_captured_or_later_values(
    monkeypatch, tmp_path, inference
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    replay = _snapshot(rank, plan)
    with torch.inference_mode(inference):
        plan.groups[0].items[0].input_ids[0] += 1
    payload = replay()
    assert payload["requests"][0]["input_tokens"]["unavailable"] == "modified_input"
    assert payload["layouts"] == []
    assert "layout_inputs_unavailable" in payload["incomplete_reasons"]


@pytest.mark.parametrize("inference", [False, True])
def test_request_replacement_does_not_substitute_planner_input(
    monkeypatch, tmp_path, inference
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    item = plan.groups[0].items[0]
    original = item.input_ids.tolist()
    replay = _snapshot(rank, plan)
    item.request.input_tokens = torch.tensor([88, 99])
    item.request.target_tokens = torch.tensor([99, 88])
    payload = replay()
    assert payload["requests"][0]["input_tokens"] == original
    assert payload["requests"][0]["target_tokens"] == original


def test_device_inputs_are_identified_without_copy_or_materialization(
    monkeypatch, tmp_path
):
    rank, plan = _plan(monkeypatch, tmp_path)
    group = plan.groups[0]
    item = group.items[0]
    device = torch.empty_like(item.input_ids, device="meta")
    plan = replace(
        plan,
        groups=(
            replace(group, items=(replace(item, input_ids=device, labels=device),)),
        ),
    )
    replay = _snapshot(rank, plan)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("D2H"))
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda *_a, **_k: pytest.fail("device read")
    )
    payload = replay()
    value = payload["requests"][0]["input_tokens"]
    assert value == {
        "unavailable": "device_input",
        "shape": [10],
        "dtype": "torch.int64",
        "device": "meta",
    }
    assert payload["layouts"] == []


def test_snapshot_budget_and_lifetime_are_owned(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path, inference=True)
    # The diagnostic input inventory is tested independently of plan pricing;
    # two 600,000-element fields must not each consume a fresh million budget.
    group = plan.groups[0]
    with torch.inference_mode():
        tensor = torch.zeros(600_000, dtype=torch.long)
    plan = replace(
        plan,
        groups=(
            replace(
                group, items=(replace(group.items[0], input_ids=tensor, labels=tensor),)
            ),
        ),
    )
    clone = torch.Tensor.clone
    copies = []

    def tracked(value, *args, **kwargs):
        result = clone(value, *args, **kwargs)
        copies.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", tracked)
    replay = _snapshot(rank, plan)
    assert len(copies) == 1 and copies[0]() is not None
    payload = replay()
    assert len(payload["requests"][0]["input_tokens"]) == 600_000
    assert (
        payload["requests"][0]["target_tokens"]["unavailable"]
        == "token_inventory_over_limit"
    )
    rank.finish_planner_observation()
    del replay
    gc.collect()
    assert copies[0]() is None


def test_failed_cpu_snapshot_stays_incomplete_and_does_not_abort(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path, inference=True)

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic copy refused")

    monkeypatch.setattr(torch.Tensor, "clone", fail)
    payload = _snapshot(rank, plan)()
    assert (
        payload["requests"][0]["input_tokens"]["unavailable"]
        == "input_snapshot_unavailable"
    )
    assert payload["layouts"] == []


def test_versioned_cpu_inputs_do_not_require_a_snapshot_copy(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path)
    original = plan.groups[0].items[0].input_ids.tolist()
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda *_a, **_k: pytest.fail("unnecessary input copy")
    )
    payload = _snapshot(rank, plan)()
    assert payload["requests"][0]["input_tokens"] == original
    assert len(payload["layouts"]) == 1
