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


def adapter_report(layer, tmp_path):
    """A grouped report whose gradient slot has pending adapter gradients."""
    from test_trainer_rank_adapter_gradient_memory import lora, parameter
    from test_trainer_rank_converted_memory import weights
    from test_trainer_rank_pending_memory import rank_with_moe
    from test_trainer_rank_slot_memory import load_slot
    from test_trainer_rank_slot_memory import request as slot_request

    rank, _ = rank_with_moe(weights(layer, 8))
    load_slot(rank, "small", 1)
    load_slot(rank, "large", 64)
    # Unallocated slot gradients the recompute backward will allocate: small,
    # distinct per-layer sizes (the fixture has 40 layers; keep this bounded).
    layers = tr._language_model(rank.runtime.model[0]).decoder.layers
    assert len(layers) <= 64
    sizes = [16 * (index + 1) for index in range(len(layers))]
    assert sum(sizes) * 2 <= 66_560  # BF16 bytes, checked before allocating.
    params = []
    for size, block in zip(sizes, layers, strict=True):
        params.append(parameter(size))
        block.add_module("adapter", lora(large=[params[-1]]))
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(
        [slot_request("small", rows=2), slot_request("large", rows=65, grad=True)],
        ensure_slots=False,
    )
    original, costs = emitted(rank, plan, tmp_path)
    return original, costs, params


def test_selected_adapter_gradients_are_replayed_and_frozen(layer, tmp_path):
    original, costs, params = adapter_report(layer, tmp_path)
    assert costs[0].checkpoint_adapter_gradient > 0
    groups = original["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
        "groups"
    ]
    assert groups[0]["adapter"] is None
    assert groups[1]["adapter"]["name"] == "large" and any(
        groups[1]["adapter"]["pending"]
    )
    actual = reports.replay(original)
    assert actual["aggregate"]["matches"]
    assert all(item["matches"] for item in actual["estimates"])
    # Gradients allocated after selection cannot change the replayed answer.
    for param in params:
        param.grad = torch.zeros_like(param)
    assert reports.replay(original) == actual
    changed = deepcopy(original)
    changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]["groups"][1][
        "adapter"
    ]["pending"][0] += 10**12
    result = reports.replay(changed)
    assert not result["estimates"][0]["matches"]
    assert (
        result["estimates"][0]["required_bytes"]
        > actual["estimates"][0]["required_bytes"]
    )


@pytest.mark.parametrize(
    "change",
    ["gradient", "length", "value", "kind_length", "kindless_pending", "mixed_kinds"],
)
def test_adapter_fact_validation_rejects_forged_input(change, layer, tmp_path):
    report, _, _ = adapter_report(layer, tmp_path)
    group = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
        "groups"
    ][1]
    adapter = group["adapter"]
    if change == "gradient":
        group["grad"] = False
        group["moe_covered"] = False
    elif change == "length":
        adapter["pending"] = [0] * 1026
    elif change == "value":
        adapter["pending"][0] = 1.5
    elif change == "kind_length":
        adapter["kind"] = "k" * 65
    elif change == "kindless_pending":
        adapter["kind"] = None
    else:
        groups = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
            "groups"
        ]
        groups[0]["grad"] = True
        groups[0]["adapter"] = {"kind": None, "name": "base", "pending": []}
    # Name the refusal: a forged fact must fail validation, not a later check.
    message = (
        "invalid runtime dimension"
        if change == "value"
        else "invalid adapter gradient facts"
    )
    with pytest.raises(ValueError, match=message):
        reports.replay(report)


def test_recomputed_layer_and_head_stage_facts_are_replayed_and_frozen(
    monkeypatch, tmp_path
):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    original, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    item = original["replay"]["memory_replay"]["estimates"][0]
    assert item["runtime_facts"]["groups"][0]["moe_covered"] is True
    assert item["arguments"]["head_backward_traced"] is False
    actual = reports.replay(original)
    assert actual["aggregate"]["matches"]
    # The replaying process's settings and TE state do not enter the answer.
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", "1")
    monkeypatch.setattr(tr, "_TE_CUBLAS_WORKSPACE_BYTES", 0)
    assert reports.replay(original) == actual
    for field in (
        "moe_checkpoint_state_bytes_per_token",
        "te_workspace_growth_bytes",
        "backward_row_state_bytes",
        "triton_min_rows",
    ):
        changed = deepcopy(original)
        changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][field] += (
            10**9
        )
        result = reports.replay(changed)
        assert not result["estimates"][0]["matches"], field
        assert (
            result["estimates"][0]["required_bytes"]
            > actual["estimates"][0]["required_bytes"]
        ), field
    # Coverage selects the input-gradient charge and whether the head stages.
    changed = deepcopy(original)
    changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]["groups"][0][
        "moe_covered"
    ] = False
    result = reports.replay(changed)
    assert not result["estimates"][0]["matches"]
    assert (
        result["estimates"][0]["required_bytes"]
        != actual["estimates"][0]["required_bytes"]
    )


@pytest.mark.parametrize(
    "change",
    ["shared", "covered_no_grad", "triton_min_rows", "routed_rows", "head_staging"],
)
def test_recomputed_layer_fact_validation_rejects_forged_input(change, layer, tmp_path):
    report, _, _ = adapter_report(layer, tmp_path)
    item = report["replay"]["memory_replay"]["estimates"][0]
    facts = item["runtime_facts"]
    if change == "shared":
        terms = facts["groups"][1]["gradient"]
        terms[2] = terms[0] + 1
        message = "invalid MoE terms"
    elif change == "covered_no_grad":
        facts["groups"][0]["moe_covered"] = True
        message = "invalid MoE recompute coverage"
    elif change == "triton_min_rows":
        facts["triton_min_rows"] = 64.0
        message = "invalid runtime dimension"
    elif change == "routed_rows":
        item["arguments"]["group_routed_rows"][1] -= 1
        message = "routed rows"
    else:
        del item["arguments"]["head_backward_traced"]
        message = "head backward staging"
    with pytest.raises(ValueError, match=message):
        reports.replay(report)


@pytest.mark.parametrize("value", [None, 1.5, -1])
def test_replay_refuses_invalid_recorded_geometry(value, tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    # The recomputed mixer multiplies these; refuse before any pricing.
    report["replay"]["memory_replay"]["rank"]["geometry"]["kv_channels"] = value
    with pytest.raises(ValueError, match="geometry"):
        reports.replay(report)


@pytest.mark.parametrize("setting", ["0", "-3"])
def test_non_positive_triton_threshold_is_captured_and_replayed(
    setting, monkeypatch, tmp_path
):
    # Live pricing accepts it (the fallback chunk then prices no rows).
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", setting)
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, costs = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    facts = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    assert facts["triton_min_rows"] == int(setting)
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert result["estimates"][0]["required_bytes"] == costs[0].required


def test_recompute_readers_are_unread_without_a_checkpointed_decoder(
    monkeypatch, tmp_path
):
    from art.trainer_rank import _planner_replay

    # Live pricing reads them only for a checkpointed decoder's recompute;
    # capture must not refuse a plan over a setting live never reads.
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", "not-a-number")
    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(rank, rank._plan_flat_forward([_request(1)]), tmp_path)
    facts = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    assert facts["checkpoint_layers"] == 0
    assert all(facts[name] is None for name in _planner_replay._RECOMPUTE_READERS)
    assert reports.replay(report)["aggregate"]["matches"]
    forged = deepcopy(report)
    forged["replay"]["memory_replay"]["estimates"][0]["runtime_facts"][
        "te_workspace_growth_bytes"
    ] = 0
    with pytest.raises(ValueError, match="invalid runtime dimension"):
        reports.replay(forged)


def test_staged_head_needs_recorded_cp2_target_backward(tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    # This CP1 plan cannot stage its head; a forged flag must not price it so.
    report["replay"]["memory_replay"]["estimates"][0]["arguments"][
        "head_backward_traced"
    ] = True
    with pytest.raises(ValueError, match="staging disagrees"):
        reports.replay(report)


def test_staged_cp2_head_is_replayed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tr,
        "_TRITON_STATS_STATE",
        {"succeeded": {"local_logsumexp_stats"}, "failed": False},
    )
    from types import SimpleNamespace

    from art.megatron.context_parallel.types import ParallelTopology

    # CP2 through the stock topology reader (as capture requires), over a CPU
    # CP2 topology and planning config.
    monkeypatch.setattr(tr.TrainerRank, "_topology_key", lambda self: (1, 1, 2, 1))
    rank = head_rank()
    rank._topology = lambda: ParallelTopology(tp=1, cp=2)
    for name, value in {
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "params_dtype": torch.bfloat16,
    }.items():
        setattr(rank.runtime.provider, name, value)
    rank.runtime.model_support_handler = SimpleNamespace(
        build_gdn_execution_spec=True,
        context_parallel_workload_profile=lambda provider: None,
    )
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([request(512, grad=True)])
    assert plan.signature.topology[2] == 2
    report, costs = emitted(rank, plan, tmp_path)
    item = report["replay"]["memory_replay"]["estimates"][0]
    assert item["arguments"]["head_backward_traced"] is True
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert result["estimates"][0]["required_bytes"] == costs[0].required


@pytest.mark.parametrize("change", ["omit_default", "omit_required", "extra"])
def test_recorded_geometry_fields_follow_its_schema(change, tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    geometry = report["replay"]["memory_replay"]["rank"]["geometry"]
    if change == "omit_default":
        # Older reports may omit fields that default to zero.
        (name,) = [n for n, v in geometry.items() if n == "gdn_conv_kernel" and v == 0]
        del geometry[name]
        assert reports.replay(report)["aggregate"]["matches"]
        return
    if change == "omit_required":
        del geometry["kv_channels"]
    else:
        geometry["unknown_width"] = 1
    with pytest.raises(ValueError, match="geometry"):
        reports.replay(report)


@pytest.mark.parametrize(
    "field,value", [("hidden_size", "8"), ("gdn_layers", -1), ("sequence_parallel", 0)]
)
def test_replay_refuses_invalid_recorded_rank_dimensions(field, value, tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    report["replay"]["memory_replay"]["rank"][field] = value
    with pytest.raises(ValueError, match="rank dimensions"):
        reports.replay(report)


@pytest.mark.parametrize(
    "name",
    [
        "_te_workspace_growth_bytes",
        "_moe_recompute_covered_for",
        "_triton_min_rows",
        "_plan_group_routed_rows",
        "_plan_head_backward_traced",
        "_head_backward_traced",
    ],
)
def test_custom_recomputed_layer_reader_is_explicitly_incomplete(name, monkeypatch):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    original = getattr(rank, name)
    monkeypatch.setattr(rank, name, lambda *args: original(*args))
    with pytest.raises(ValueError, match="custom_runtime_estimator"):
        _planner_replay.capture(rank, plan)


@pytest.mark.parametrize("layers", [2**10 + 1, 0, 40.0])
def test_replay_bounds_the_recorded_layer_count(layers, pending_rank, tmp_path):
    rank = pending_rank
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    # Replay sizes per-layer tuples from this field; refuse it before costing.
    report["replay"]["memory_replay"]["rank"]["num_layers"] = layers
    with pytest.raises(ValueError, match="layer count"):
        reports.replay(report)


def test_custom_adapter_gradient_reader_is_explicitly_incomplete(monkeypatch):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    original = rank._pending_adapter_gradient_bytes
    monkeypatch.setattr(
        rank, "_pending_adapter_gradient_bytes", lambda refs: original(refs)
    )
    with pytest.raises(ValueError, match="custom_runtime_estimator"):
        _planner_replay.capture(rank, plan)


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
        facts["version"] += 1
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


def layout_report(monkeypatch, tmp_path):
    """A grouped CP2 report priced on every rank's own layouts."""
    from test_trainer_rank_checkpoint_memory import rank as checkpoint_rank
    from test_trainer_rank_layout_memory import _requests, art_cp

    rank = art_cp(checkpoint_rank(), monkeypatch)
    del rank._topology_key  # the stock reader, over the fixture's CP2 topology
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(_requests())
    assert rank._plan_group_layouts(plan) is not None
    report, costs = emitted(rank, plan, tmp_path)
    return report, costs, rank


def test_cp_layouts_are_replayed_and_frozen(monkeypatch, tmp_path):
    original, costs, rank = layout_report(monkeypatch, tmp_path)
    facts = original["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    (group,) = facts["groups"]
    assert len(group["layout"]["attention_rows"]) == 2
    assert len(facts["layer_gdn_inputs"]) == facts["checkpoint_layers"] == 40
    assert any(facts["layer_gdn_inputs"]) and not all(facts["layer_gdn_inputs"])
    actual = reports.replay(original)
    assert actual["aggregate"]["matches"]
    assert actual["estimates"][0]["required_bytes"] == costs[0].required
    # The live model's layers cannot change the replayed answer.
    for layer in rank.runtime.model[0].decoder.layers:
        layer._art_gdn_island_boundary = None
    assert reports.replay(original) == actual

    def changed(edit):
        report = deepcopy(original)
        edit(report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"])
        return reports.replay(report)["estimates"][0]

    # Each frozen layout fact reaches the per-rank pricing.
    for edit in (
        lambda f: f["groups"][0]["layout"]["attention_rows"].__setitem__(1, 10**5),
        lambda f: f["groups"][0]["layout"]["attention_retained"].__setitem__(0, 10**12),
        lambda f: f.__setitem__("layer_gdn_inputs", [True] * 40),
    ):
        result = changed(edit)
        assert not result["matches"]
        assert result["required_bytes"] != actual["estimates"][0]["required_bytes"]


@pytest.mark.parametrize(
    "change",
    [
        "no_grad",
        "gdn_length",
        "value",
        "inputs_length",
        "inputs_type",
        "ranks",
        "argument",
        "empty_gdn",
    ],
)
def test_cp_layout_fact_validation_rejects_forged_input(change, monkeypatch, tmp_path):
    report, _, _ = layout_report(monkeypatch, tmp_path)
    item = report["replay"]["memory_replay"]["estimates"][0]
    facts = item["runtime_facts"]
    layout = facts["groups"][0]["layout"]
    if change == "no_grad":
        facts["groups"][0]["grad"] = False
        facts["groups"][0]["moe_covered"] = False
        facts["groups"][0]["adapter"] = None
        message = "invalid CP layout facts"
    elif change == "gdn_length":
        layout["gdn_rows"].append(1)
        message = "invalid CP layout facts"
    elif change == "value":
        layout["attention_retained"][0] = 1.5
        message = "invalid runtime dimension"
    elif change == "inputs_length":
        facts["layer_gdn_inputs"].pop()
        message = "invalid layer input layout facts"
    elif change == "inputs_type":
        facts["layer_gdn_inputs"][0] = 1
        message = "invalid layer input layout facts"
    elif change == "ranks":
        for key in ("attention_rows", "gdn_rows", "attention_retained"):
            layout[key].append(1)
        message = "recorded topology"
    elif change == "argument":
        item["arguments"]["group_layouts"] = [dict(layout)]
        message = "immutable runtime"
    else:
        layout["gdn_rows"] = []
        message = "invalid CP layout facts"
    with pytest.raises(ValueError, match=message):
        reports.replay(report)


def test_cp_layouts_need_the_live_cp2_topology(tmp_path):
    # Layout pricing is CP2-only: layouts forged onto a CP1 report are refused.
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(
        rank, rank._plan_flat_forward([request(65, grad=True)]), tmp_path
    )
    facts = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    (group,) = facts["groups"]
    group["layout"] = {
        "attention_rows": [group["rows"]],
        "gdn_rows": None,
        "attention_retained": [0],
    }
    facts["layer_gdn_inputs"] = [False] * facts["checkpoint_layers"]
    with pytest.raises(ValueError, match="recorded topology"):
        reports.replay(report)


@pytest.mark.parametrize("name", ["_plan_group_layouts", "_layer_gdn_inputs"])
def test_custom_layout_reader_is_explicitly_incomplete(name, monkeypatch):
    from art.trainer_rank import _planner_replay

    rank = _rank(monkeypatch)
    plan = rank._plan_flat_forward([_request(1)])
    original = getattr(rank, name)
    monkeypatch.setattr(rank, name, lambda *args: original(*args))
    with pytest.raises(ValueError, match="custom_runtime_estimator"):
        _planner_replay.capture(rank, plan)
