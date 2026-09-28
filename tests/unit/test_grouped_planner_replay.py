"""Real CPU plans and the emitted report path; never allocator/model execution."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_trainer_rank_head_memory import rank as head_rank
from test_trainer_rank_head_memory import request
from test_trainer_rank_pending_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
from test_trainer_rank_split import _rank, _request
import torch

from art.trainer_rank import _impl as tr
from art.trainer_rank import _planner_misses as reports


def emitted(rank, plan, tmp_path):
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    children = plan.subforwards if isinstance(plan, tr._SplitForwardPlan) else (plan,)
    costs = [rank._plan_cost(child) for child in children]
    required = rank._split_required_memory(costs)
    rank._begin_planner_observation(
        plan, tr._MemoryCheck(required, required + 1000, True)
    )
    observation = rank._planner_observation
    assert observation is not None and observation["comparable"]
    path = rank._planner_reporter.report(
        predicted_peak_bytes=observation["predicted"],
        observed_peak_bytes=required * 2,
        phase="forward",
        admission_peak_bytes=required,
        replay_factory=observation["replay"],
    )
    assert path is not None
    rank.finish_planner_observation()
    return reports.validate_report(path.read_bytes()), costs


@pytest.mark.parametrize("family", ["zero", "head", "checkpoint", "gdn"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_selected_grouped_cost_recomputed(
    family, calibrated, monkeypatch, tmp_path, pending_rank
):
    rank = (
        _rank(monkeypatch)
        if family == "zero"
        else pending_rank
        if family == "gdn"
        else head_rank()
    )
    if family == "checkpoint":
        setattr(rank.runtime.model[0], "output_layer", None)
    items = (
        [_request(1)]
        if family == "zero"
        else [
            request(65, grad=True),
            replace(request(33), input_tokens=torch.arange(33) + 100),
        ]
    )
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(items)
    if calibrated:
        rank._memory_profiles[plan.signature] = tr._MemoryProfile(
            bytes_per_token=100,
            packed_tokens=plan.packed_tokens,
            logical_per_packed=plan.active_logical_tokens / plan.packed_tokens,
            retained_compute_bytes_per_token=10,
        )
    report, costs = emitted(rank, plan, tmp_path)
    assert report["replay_complete"], report["incomplete_reasons"]
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert all(item["matches"] for item in result["estimates"] + result["layouts"])
    assert result["estimates"][0]["required_bytes"] == costs[0].required
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("split", [False, True])
def test_placement_admission_with_runtime_facts_stays_incomplete(
    split, monkeypatch, tmp_path
):
    from art.trainer_rank import ForwardOptions, _planner_evidence

    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda *_args: 10**12)
    monkeypatch.setattr(rank, "_available_cpu_memory_bytes", lambda: 10**12)
    options = ForwardOptions(
        backward_state="replay" if split else "gpu",
        output_device="cpu" if split else "model",
    )
    items = [replace(request(rows, grad=True), options=options) for rows in (65, 33)]
    plan = (
        tr._SplitForwardPlan(
            tuple(rank._plan_flat_forward([item]) for item in items), ((0,), (1,)), 2
        )
        if split
        else rank._plan_flat_forward(items)
    )
    children = plan.subforwards if isinstance(plan, tr._SplitForwardPlan) else (plan,)
    unplaced = rank._split_required_memory([rank._plan_cost(p) for p in children])
    with _planner_evidence.scope(
        _planner_evidence.Decision("forward", sync_across_dp=False, owner=rank)
    ):
        plan, check = rank._admit_graph_memory(plan)
    assert check.fits and check.cpu_fits and check.cpu_required_bytes > 0
    assert check.sample is not None
    local = check.sample.local_required_bytes
    assert local is not None
    assert local == check.estimated_required_bytes and local != unplaced
    rank._begin_planner_observation(
        plan, replace(check, estimated_required_bytes=local + 123)
    )
    observation = rank._planner_observation
    assert observation is not None and observation["comparable"]
    path = rank._planner_reporter.report(
        predicted_peak_bytes=observation["predicted"],
        observed_peak_bytes=local * 2,
        phase="forward",
        admission_peak_bytes=local + 123,
        replay_factory=observation["replay"],
    )
    assert path is not None
    rank.finish_planner_observation()
    report = reports.validate_report(path.read_bytes())
    payload = report["replay"]
    assert payload["local_admission_peak_bytes"] == local
    assert payload["reduced_admission_peak_bytes"] == local + 123
    assert report["predicted_peak_bytes"] == round(local / tr._MEMORY_SAFETY_FACTOR)
    assert payload["requests"][0]["options"]["backward_state"] == options.backward_state
    assert payload["requests"][0]["options"]["output_device"] == options.output_device
    assert [r["target_tokens"] for r in payload["requests"]] == [
        list(range(65)),
        list(range(33)),
    ]
    assert all(
        e["runtime_facts"] is not None for e in payload["memory_replay"]["estimates"]
    )
    # Without the placement boundary, unchanged-source replay reconstructs every
    # model estimate but incorrectly claims it can reproduce placement admission.
    assert not report["replay_complete"], reports.replay(report)
    assert report["incomplete_reasons"] == ["graph_placement_admission_unavailable"]
    with pytest.raises(ValueError, match="graph_placement_admission_unavailable"):
        reports.replay(report)


def test_runtime_dimensions_change_recomputed_cost_not_expected_answer(tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([request(65, grad=True)])
    original, _ = emitted(rank, plan, tmp_path)
    for field, value in [("head_vocabulary", 1), ("checkpoint_layers", 400)]:
        changed = deepcopy(original)
        changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][field] = (
            value
        )
        result = reports.replay(changed)
        assert result["estimates"][0]["matches"] is False
        assert result["aggregate"]["matches"] is False


def test_legacy_grouped_report_cannot_enable_replay_by_flipping_flag(
    monkeypatch, tmp_path
):
    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    original, _ = emitted(rank, rank._plan_flat_forward([_request(1)]), tmp_path)
    item = original["replay"]["memory_replay"]["estimates"][0]
    del item["runtime_facts"]
    with pytest.raises(ValueError, match="immutable runtime"):
        reports.replay(original)


def test_custom_estimator_and_hybridep_explicitly_incomplete(monkeypatch):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    original = rank._moe_workspace_bytes
    rank._moe_workspace_bytes = lambda *args, **kwargs: original(*args, **kwargs)
    with pytest.raises(ValueError, match="custom_runtime_estimator"):
        _planner_replay.capture(rank, plan)
    del rank._moe_workspace_bytes
    rank.runtime.provider.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="hybridep"):
        _planner_replay.capture(rank, plan)


def test_snapshot_owned_before_runtime_changes(monkeypatch, tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([request(65, grad=True)])
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    rank._begin_planner_observation(plan, tr._MemoryCheck(1, 2, True))
    observation = rank._planner_observation
    assert observation is not None and observation["comparable"]
    original = deepcopy(observation["replay"]()["memory_replay"])
    rank._moe_forward_stages = ((1, 999999),)
    rank.runtime.model[0].decoder.eval()
    rank._padded_vocab_size = 2
    assert observation["replay"]()["memory_replay"] == original
    rank.finish_planner_observation()


def test_selected_slot_terms_are_replayed_and_frozen(layer, tmp_path):
    from test_trainer_rank_converted_memory import weights
    from test_trainer_rank_pending_memory import rank_with_moe
    from test_trainer_rank_slot_memory import load_slot
    from test_trainer_rank_slot_memory import request as slot_request

    rank, _ = rank_with_moe(weights(layer, 8))
    load_slot(rank, "small", 1)
    load_slot(rank, "large", 64)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(
        [slot_request("small", rows=2), slot_request("large", rows=65, grad=True)],
        ensure_slots=False,
    )
    original, _ = emitted(rank, plan, tmp_path)
    actual = reports.replay(original)
    assert actual["aggregate"]["matches"]
    groups = original["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
        "groups"
    ]
    assert groups[0]["forward"] != groups[1]["forward"]
    load_slot(rank, "large", 1)
    assert reports.replay(original) == actual
    changed = deepcopy(original)
    changed_group = changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
        "groups"
    ][1]
    changed_group["gradient"][1][0][1] += 10**12
    result = reports.replay(changed)
    assert not result["estimates"][0]["matches"]
    assert (
        result["estimates"][0]["required_bytes"]
        > actual["estimates"][0]["required_bytes"]
    )


@pytest.mark.parametrize(
    "change", ["version", "group", "layout", "gdn_segment", "budget"]
)
def test_fact_validation_rejects_inconsistent_or_unbounded_input(
    change, pending_rank, tmp_path
):
    rank = pending_rank
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    facts = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    if change == "version":
        facts["version"] = 2
    elif change == "group":
        facts["groups"][0]["rows"] += 1
    elif change == "layout":
        facts["groups"][0]["layout_fingerprint"] = "wrong"
    elif change == "gdn_segment":
        facts["groups"][0]["gdn"]["segments"][0]["end"] += 1
    else:
        facts["groups"] *= 1025
    with pytest.raises(ValueError):
        reports.replay(report)


def test_missing_layout_stays_incomplete(monkeypatch, tmp_path):
    rank = _rank(monkeypatch)
    # Reporting was disabled while this plan was selected; do not invent its layout.
    plan = rank._plan_flat_forward([_request(1)])
    report, _ = emitted(rank, plan, tmp_path)
    assert not report["replay_complete"]
    assert "selected_layout_unavailable" in report["incomplete_reasons"]
    with pytest.raises(ValueError, match="incomplete replay"):
        reports.replay(report)


@pytest.mark.parametrize(
    "lengths,packed,segments",
    [
        ([10136, 10247, 10985, 10365, 9880, 9912, 12602, 10529], 49439, 9),
        ([11675, 12486, 11444, 10256, 11859, 11747, 10084], 49365, 8),
        ([11592], 11592, 1),
    ],
)
def test_retained_060_geometry_with_synthetic_token_values(
    lengths, packed, segments, pending_rank, tmp_path
):
    # These dimensions come from the three retained 060 reports. Their token
    # values/runtime facts were unavailable: this is representative geometry,
    # explicitly not a recovered replay of the old reports or their costs.
    rank = pending_rank
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    shared = 5031 if len(lengths) > 1 else 0
    items = [
        tr.ForwardInput(
            input_tokens=torch.cat(
                (torch.arange(shared), torch.arange(n - shared) + 100000 * (i + 1))
            ),
            target_tokens=torch.arange(n),
        )
        for i, n in enumerate(lengths)
    ]
    plan = rank._plan_flat_forward(items, memory_minimal=True)
    assert (
        plan.packed_tokens,
        plan.active_logical_tokens,
        plan.grad_segment_count,
    ) == (packed, sum(lengths), segments)
    record, _ = emitted(rank, plan, tmp_path)
    result = reports.replay(record)
    assert result["aggregate"]["matches"]
    assert all(row["matches"] for row in result["estimates"] + result["layouts"])


def test_capture_bounds_do_not_materialize_device_positions(monkeypatch, tmp_path):
    from art.trainer_rank import _planner_replay

    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([request(2, grad=True)])
    group = plan.groups[0]
    packed = replace(
        group.packed,
        positions_by_sequence=tuple(
            torch.empty_like(p, device="meta")
            for p in group.packed.positions_by_sequence
        ),
    )
    device_plan = replace(plan, groups=(replace(group, packed=packed),))
    with pytest.raises(ValueError, match="head_positions_unavailable"):
        _planner_replay.capture(rank, device_plan)
    monkeypatch.setattr(_planner_replay, "_MAX_SEGMENTS", 0)
    with pytest.raises(ValueError, match="segment_inventory_over_limit"):
        _planner_replay.capture(rank, plan)


def test_capture_rejects_constructor_stage_inventory_before_iteration(monkeypatch):
    from art.trainer_rank import _memory, _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    rank._moe_forward_stages = ((1, 0),) * 4097
    monkeypatch.setattr(
        _memory, "_moe_workspace_terms", lambda *a, **k: pytest.fail("iterated terms")
    )
    with pytest.raises(ValueError, match="stage_inventory_over_limit"):
        _planner_replay.capture(rank, plan)


def test_capture_budget_rejects_before_json_materialization(monkeypatch):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    monkeypatch.setattr(_planner_replay, "_MAX_BYTES", 256)
    monkeypatch.setattr(
        _planner_replay.json,
        "dumps",
        lambda *a, **k: pytest.fail("encoded over budget"),
    )
    with pytest.raises(ValueError, match="runtime_facts_over_limit"):
        _planner_replay.capture(rank, plan)


def test_runtime_reader_error_retains_no_arbitrary_text(monkeypatch, tmp_path):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([_request(1)])

    def fail(*args):
        raise ValueError("PRIVATE_RUNTIME_TEXT" * 100000)

    monkeypatch.setattr(_planner_replay, "capture", fail)
    report, _ = emitted(rank, plan, tmp_path)
    assert report["incomplete_reasons"] == ["runtime_facts_unavailable:ValueError"]
    assert not report["replay_complete"]


@pytest.mark.parametrize("field", ["labels", "gdn_origin"])
def test_runtime_provenance_is_joined_to_request_and_layout(
    field, pending_rank, tmp_path
):
    rank = head_rank() if field == "labels" else pending_rank
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    if field == "labels":
        report["replay"]["requests"][0]["target_tokens"] = [-100] * 65
    else:
        segment = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
            "groups"
        ][0]["gdn"]["segments"][0]
        segment["start"] += 1
        segment["end"] += 1
    with pytest.raises(ValueError, match="disagree"):
        reports.replay(report)


def test_capture_and_replay_share_exact_element_budget(monkeypatch, tmp_path):
    from art.trainer_rank import _planner_replay

    monkeypatch.setattr(_planner_replay, "_MAX_INPUT_VALUES", 4)
    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([_request(1, length=2)])
    report, _ = emitted(rank, plan, tmp_path)
    assert report["replay_complete"]
    assert reports.replay(report)["aggregate"]["matches"]
    monkeypatch.setattr(_planner_replay, "_MAX_INPUT_VALUES", 3)
    with pytest.raises(ValueError, match="runtime_token_inventory_over_limit"):
        _planner_replay.capture(rank, plan)
    with pytest.raises(ValueError, match="runtime_token_inventory_over_limit"):
        reports.replay(report)


@pytest.mark.parametrize(
    "field", ["requests", "layouts", "checkpoint_slots", "subforward_group_counts"]
)
def test_grouped_replay_refuses_unconsumed_metadata(field, monkeypatch, tmp_path):
    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(rank, rank._plan_flat_forward([_request(1)]), tmp_path)
    report["replay"][field].append(deepcopy(report["replay"][field][0]))
    with pytest.raises(ValueError, match="unused"):
        reports.replay(report)
