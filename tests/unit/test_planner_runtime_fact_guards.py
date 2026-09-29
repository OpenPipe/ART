"""Unsupported estimators and oversized primitive facts must decline replay."""

from copy import deepcopy

import pytest
from test_grouped_planner_replay import emitted, head_rank, request

from art.trainer_rank import _planner_misses as reports
from art.trainer_rank import _planner_replay as runtime


@pytest.mark.parametrize(
    "method", ["_split_required_memory", "_gdn_segment_layer_bytes"]
)
def test_changed_custom_cost_is_not_reported_replay_complete(method, tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    rank._recompute_granularity = "selective"
    rank._gdn_layers = 1
    original = getattr(rank, method)
    setattr(rank, method, lambda *a, **k: original(*a, **k) + 10**12)
    plan = rank._plan_flat_forward([request(65, grad=True)])
    report, _ = emitted(rank, plan, tmp_path)
    assert not report["replay_complete"]
    assert (
        "runtime_facts_unavailable:custom_runtime_estimator_unsupported"
        in report["incomplete_reasons"]
    )
    with pytest.raises(ValueError, match="incomplete replay"):
        reports.replay(report)


@pytest.mark.parametrize(
    "method",
    [
        "_one_layer_recompute",
        "_topology_key",
        "_physical_tokens",
        "_plan_group_rows",
        "_plan_retained_tokens",
    ],
)
def test_custom_shared_estimator_readers_decline(method):
    rank = head_rank()
    plan = rank._plan_flat_forward([request(65, grad=True)])
    original = getattr(rank, method)
    setattr(rank, method, lambda *a, **k: original(*a, **k))
    with pytest.raises(ValueError, match="custom_runtime_estimator_unsupported"):
        runtime.capture(rank, plan)


@pytest.mark.parametrize("inventory", ["stages", "slots", "requests"])
def test_cumulative_fact_budget_precedes_json_encoding(inventory, monkeypatch):
    rank = head_rank()
    facts = runtime.capture(rank, rank._plan_flat_forward([request(65, grad=True)]))
    group = facts["groups"][0]
    group["gdn"] = None
    if inventory == "stages":
        group["forward"] = [0, [[0, 0]] * 4096, 0]
        group["gradient"] = deepcopy(group["forward"])
        facts["groups"] = [deepcopy(group) for _ in range(8)]
    elif inventory == "slots":
        group["slot"] = "\U0001f600" * 4096
        facts["groups"] = [deepcopy(group) for _ in range(8)]
    else:
        group["request_indices"] = [2**62] * 4096
        facts["groups"] = [deepcopy(group) for _ in range(8)]
    monkeypatch.setattr(
        runtime.json, "dumps", lambda *a, **k: pytest.fail("encoded oversized facts")
    )
    with pytest.raises(ValueError, match="runtime_facts_over_limit"):
        runtime.validate(facts)


def test_stock_methods_and_small_escaped_identity_remain_supported(tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    assert report["replay_complete"]
    assert reports.replay(report)["aggregate"]["matches"]
    facts = deepcopy(report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"])
    facts["groups"][0]["slot"] = '\U0001f600\n"'
    runtime.validate(facts)
