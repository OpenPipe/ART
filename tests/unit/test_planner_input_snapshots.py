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
from art.trainer_rank._planner_retention import report_retention_scope


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
    assert payload["incomplete_reasons"] == []
    rank._planner_reporter.report(
        predicted_peak_bytes=1000,
        observed_peak_bytes=2000,
        phase="forward",
        replay_factory=replay,
    )
    [path] = list(tmp_path.glob("*.json"))
    assert validate_report(path.read_bytes())["replay_complete"] is True
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
    assert payload["requests"][0]["input_tokens"]["unavailable"] == (
        "selected_layout_input_mismatch"
    )
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


@pytest.mark.parametrize("nested", [False, True])
def test_suppressed_scope_captures_no_cpu_inputs(monkeypatch, tmp_path, nested):
    rank, plan = _plan(monkeypatch, tmp_path, inference=True)
    clone = torch.Tensor.clone
    copies = []

    def tracked(value, *args, **kwargs):
        copies.append(value.numel())
        return clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", tracked)
    with report_retention_scope(None, capture=False):
        with report_retention_scope(None, capture=True) if nested else nullcontext():
            rank._begin_planner_observation(plan, tr._MemoryCheck(1000, 2000, True))
    assert copies == []
    assert rank._planner_observation is not None
    assert rank._planner_observation["comparable"] is False
    rank.finish_planner_observation()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("inference", [False, True])
def test_input_changed_before_observation_is_not_claimed_as_selected(
    monkeypatch, tmp_path, inference
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    tensor = plan.groups[0].items[0].input_ids
    with torch.inference_mode(inference):
        tensor[0] += 1
    payload = _snapshot(rank, plan)()
    value = payload["requests"][0]["input_tokens"]
    assert isinstance(value, dict), value
    assert value["unavailable"] == "selected_layout_input_mismatch"
    assert payload["layouts"] == []
    assert "selected_layout_input_mismatch" in payload["incomplete_reasons"]


@pytest.mark.parametrize("inference", [False, True])
def test_device_swap_after_capture_is_not_read(monkeypatch, tmp_path, inference):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    replay = _snapshot(rank, plan)
    tensor = plan.groups[0].items[0].input_ids
    with torch.inference_mode(inference):
        replacement = torch.empty_like(tensor, device="meta")
    torch.utils.swap_tensors(tensor, replacement)
    original = torch.Tensor.tolist

    def no_device_read(value, *args, **kwargs):
        assert value.device.type == "cpu", "device materialization attempted"
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", no_device_read)
    payload = replay()
    assert payload["requests"][0]["input_tokens"]["unavailable"] == "device_input"
    assert payload["layouts"] == []


@pytest.mark.parametrize("inference", [False, True])
def test_emission_canonicalization_ignores_default_device(
    monkeypatch, tmp_path, inference
):
    rank, plan = _plan(monkeypatch, tmp_path, inference=inference)
    original = plan.groups[0].items[0].input_ids.tolist()
    replay = _snapshot(rank, plan)
    cpu = torch.Tensor.cpu
    cuda_initialized = torch.cuda.is_initialized()

    def no_readback(value, *args, **kwargs):
        assert value.device.type == "cpu", "Diagnostic device readback"
        return cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", no_readback)
    with torch.device("meta"):
        payload = replay()
    assert payload["requests"][0]["input_tokens"] == original
    assert len(payload["layouts"]) == 1
    assert payload["layouts"][0]["expected_fingerprint"] == (
        plan.groups[0].layout.fingerprint
    )
    assert torch.cuda.is_initialized() == cuda_initialized
    rank.finish_planner_observation()


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("changed_before_observation", [False, True])
def test_partial_group_inputs_are_not_claimed_as_selected(
    monkeypatch, tmp_path, inference, changed_before_observation
):
    rank = _rank(monkeypatch)
    rank._planner_reporter = Reporter(10, spool_dir=tmp_path)
    with torch.inference_mode() if inference else nullcontext():
        requests = [_request(0), _request(1), replace(_request(2), no_grad=True)]
    plan = rank._plan_flat_forward(requests)
    assert [len(group.items) for group in plan.groups] == [2, 1]
    assert plan.groups[1].layout is not None
    first, second = plan.groups[0].items
    unaffected = plan.groups[1].items[0].input_ids.tolist()
    if changed_before_observation:
        with torch.inference_mode(inference):
            first.input_ids[0] += 1
    replay = _snapshot(rank, plan)
    with torch.inference_mode(inference):
        second.input_ids[0] += 1
    payload = replay()
    assert payload["requests"][0]["input_tokens"] == {
        "unavailable": "selected_layout_input_unverified"
    }
    assert payload["requests"][1]["input_tokens"]["unavailable"] == "modified_input"
    assert payload["requests"][2]["input_tokens"] == unaffected
    assert len(payload["layouts"]) == 1
    assert payload["layouts"][0]["expected_fingerprint"] == (
        plan.groups[1].layout.fingerprint
    )
    assert "layout_inputs_unavailable" in payload["incomplete_reasons"]
    assert "selected_layout_input_unverified" in payload["incomplete_reasons"]
    rank.finish_planner_observation()


def test_missing_layout_does_not_certify_observed_inputs(monkeypatch, tmp_path):
    rank, plan = _plan(monkeypatch, tmp_path)
    plan = replace(plan, groups=(replace(plan.groups[0], layout=None),))
    payload = _snapshot(rank, plan)()
    assert payload["requests"][0]["input_tokens"] == {
        "unavailable": "selected_layout_input_unverified"
    }
    assert payload["layouts"] == []
    assert "selected_layout_unavailable" in payload["incomplete_reasons"]
    rank.finish_planner_observation()


@pytest.mark.parametrize("inference", [False, True])
def test_empty_storage_replacement_retains_other_replay_metadata(
    monkeypatch, tmp_path, inference
):
    rank = _rank(monkeypatch)
    rank._planner_reporter = Reporter(10, spool_dir=tmp_path)
    with torch.inference_mode() if inference else nullcontext():
        requests = [_request(0), replace(_request(1), no_grad=True)]
    plan = rank._plan_flat_forward(requests)
    assert len(plan.groups) == 2
    replay = _snapshot(rank, plan)
    tensor = plan.groups[0].items[0].input_ids
    with torch.inference_mode(inference):
        torch.utils.swap_tensors(tensor, torch.empty(0, dtype=torch.long))
    rank._planner_reporter.report(
        predicted_peak_bytes=1000,
        observed_peak_bytes=2000,
        phase="forward",
        replay_factory=replay,
    )
    [path] = list(tmp_path.glob("*.json"))
    report = validate_report(path.read_bytes())
    assert report["replay_complete"] is False
    retained = report["replay"]
    assert retained["memory_replay"]
    assert len(retained["layouts"]) == 1
    assert retained["requests"][0]["input_tokens"]["unavailable"] == (
        "modified_input" if inference else "selected_layout_input_mismatch"
    )
    assert retained["requests"][1]["input_tokens"] == (
        plan.groups[1].items[0].input_ids.tolist()
    )
    assert not any("runtime_facts" in reason for reason in report["incomplete_reasons"])
    rank.finish_planner_observation()
