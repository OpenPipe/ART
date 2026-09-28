"""Report preflight must share capture's ownership and inventory boundaries."""

import pytest
from test_grouped_planner_replay import emitted, head_rank, request

from art.trainer_rank import _planner_misses as reports
from art.trainer_rank import _planner_replay as runtime
from art.trainer_rank import _prefix_tree_planner as trees


def test_foreign_rank_bound_estimator_is_incomplete(tmp_path):
    rank, donor = head_rank(), head_rank()
    donor._hidden_size *= 1_000_000
    rank._estimate_required_memory_bytes_from_values = (
        donor._estimate_required_memory_bytes_from_values
    )
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    assert not report["replay_complete"]
    assert (
        "runtime_facts_unavailable:custom_runtime_estimator_unsupported"
        in report["incomplete_reasons"]
    )


@pytest.mark.parametrize(
    "case", ["shared_requests", "layout_values", "invalid_facts", "unused_layout"]
)
def test_shared_inventory_preflight_precedes_layout_construction(
    case, monkeypatch, tmp_path
):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    items = (
        [request(5, grad=True), request(3)]
        if case == "shared_requests"
        else [request(2, grad=True)]
    )
    plan = rank._plan_flat_forward(items)
    report, _ = emitted(rank, plan, tmp_path)
    if case == "unused_layout":
        report["replay"]["layouts"].append(report["replay"]["layouts"][0])
        reason = "unused grouped replay metadata"
    elif case == "invalid_facts":
        report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
            "version"
        ] = 2
        reason = "unsupported runtime facts version"
    else:
        monkeypatch.setattr(runtime, "_MAX_INPUT_VALUES", 10)
        reason = "runtime_token_inventory_over_limit"
        if case == "shared_requests":
            with pytest.raises(ValueError, match=reason):
                runtime.capture(rank, plan)
        else:
            report["replay"]["layouts"][0]["input_tokens"] = [list(range(11))]
    monkeypatch.setattr(
        trees,
        "plan_prefix_tree_layout",
        lambda *a, **k: pytest.fail("constructed before preflight"),
    )
    with pytest.raises(ValueError, match=reason):
        reports.replay(report)


def test_combined_inventory_at_limit_still_replays(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_MAX_INPUT_VALUES", 16)
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(5, grad=True), request(3)]), tmp_path
    )
    assert report["replay_complete"]
    assert reports.replay(report)["aggregate"]["matches"]
