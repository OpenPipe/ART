from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import stat
from typing import Any

import pytest

from art.trainer_rank import _planner_misses as reports


@pytest.fixture(autouse=True)
def clear_sink():
    reports.set_report_sink(None)
    yield
    reports.set_report_sink(None)


def report(tmp_path: Path, **kwargs):
    args: dict[str, Any] = dict(
        predicted_peak_bytes=100,
        observed_peak_bytes=120,
        phase="forward_backward",
        replay_factory=lambda: {"memory_replay": {"test": True}},
    )
    args.update(kwargs)
    return reports.Reporter(5, spool_dir=tmp_path / "spool").report(**args)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "", "five"])
def test_invalid_threshold_refuses(value):
    with pytest.raises(ValueError):
        reports.parse_options({reports.MISS_THRESHOLD_ENV: value})


def test_options_are_independent_and_default_off():
    assert reports.parse_options({}) == reports.Options()
    assert reports.parse_options({reports.ALLOW_OVERSIZED_ENV: "1"}) == reports.Options(
        True
    )
    assert reports.parse_options({reports.MISS_THRESHOLD_ENV: "5"}) == reports.Options(
        False, 5
    )
    with pytest.raises(ValueError):
        reports.parse_options({reports.ALLOW_OVERSIZED_ENV: "true"})


@pytest.mark.parametrize(
    "observed,emitted",
    [(94, True), (95, False), (100, False), (105, False), (106, True)],
)
def test_threshold_strict_both_directions(tmp_path, observed, emitted):
    called = []
    path = report(
        tmp_path,
        observed_peak_bytes=observed,
        replay_factory=lambda: called.append(True) or {},
    )
    assert (path is not None) == emitted
    assert bool(called) == emitted
    if emitted:
        record = reports.validate_report(path.read_bytes())
        assert record["error_pct"] == abs(observed - 100)
    else:
        assert not (tmp_path / "spool").exists()


def test_disabled_oom_never_serializes_or_creates_spool(tmp_path):
    reporter = reports.Reporter(spool_dir=tmp_path / "spool")
    assert (
        reporter.report(
            predicted_peak_bytes=1,
            observed_peak_bytes=None,
            oom=True,
            phase="backward",
            replay_factory=lambda: pytest.fail("serialized"),
        )
        is None
    )
    assert not reporter.spool_dir.exists()


def test_oom_always_records_partial_separately(tmp_path):
    reporter = reports.Reporter(100_000, spool_dir=tmp_path / "spool")
    path = reporter.report(
        predicted_peak_bytes=100,
        observed_peak_bytes=110,
        oom=True,
        partial_peak_bytes=110,
        phase="backward",
        replay_factory=lambda: {},
    )
    assert path is not None
    record = reports.validate_report(path.read_bytes())
    assert record["oom"] is True
    assert record["observed_peak_bytes"] is record["error_pct"] is None
    assert record["partial_peak_bytes"] == 110


def test_sink_sees_original_durable_file_and_failure_preserves_it(tmp_path, caplog):
    seen = []

    def sink(path):
        seen.append(path.read_bytes())
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        raise OSError("secret text must not be logged")

    reports.set_report_sink(sink)
    path = report(tmp_path)
    assert seen == [path.read_bytes()]
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert "sink failed" in caplog.text
    assert "secret text" not in caplog.text
    assert len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("failure", ["serialize", "factory", "overflow"])
def test_incomplete_replay_is_retained_and_honest(tmp_path, monkeypatch, failure):
    def factory():
        if failure == "factory":
            raise RuntimeError("private payload")
        if failure == "serialize":
            return {"opaque": object()}
        return {"large": "x" * 4000}

    if failure == "overflow":
        monkeypatch.setattr(reports, "MAX_REPORT_BYTES", 2048)
    path = report(tmp_path, replay_factory=factory)
    record = reports.validate_report(path.read_bytes())
    assert record["replay"] is None
    assert record["replay_complete"] is False
    assert record["incomplete_reasons"]
    if failure == "overflow":
        assert record["incomplete_reasons"] == ["replay unavailable: ValueError"]
    assert b"private payload" not in path.read_bytes()


def test_atomic_persistence_is_idempotent_and_conflicts_refuse(tmp_path):
    original = report(tmp_path)
    raw = original.read_bytes()
    destination = tmp_path / "driver"
    delivered = reports.persist_report(raw, destination)
    assert reports.persist_report(raw, destination) == delivered
    changed = json.loads(raw)
    changed["phase"] = "backward"
    with pytest.raises(ValueError, match="different bytes"):
        reports.persist_report(reports._encode(changed), destination)
    assert delivered.read_bytes() == raw


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "nan",
        "traversal",
        "noncanonical",
        "oom_peak",
        "extra",
        "wrong_percent",
        "below_threshold",
    ],
)
def test_transport_rejects_invalid_reports(tmp_path, mutation):
    raw = report(tmp_path).read_bytes()
    record = json.loads(raw)
    if mutation == "duplicate":
        raw = raw.replace(b'"format":2', b'"format":2,"format":2')
    elif mutation == "nan":
        raw = raw.replace(b'"threshold_pct":5', b'"threshold_pct":NaN')
    elif mutation == "noncanonical":
        raw = json.dumps(record).encode()
    else:
        if mutation == "traversal":
            record["id"] = "../../elsewhere"
        elif mutation == "oom_peak":
            record["oom"] = True
        elif mutation == "wrong_percent":
            record["error_pct"] = 5
        elif mutation == "below_threshold":
            record["threshold_pct"] = 20
        else:
            record["unknown"] = "unregistered"
        raw = reports._encode(record)
    with pytest.raises(ValueError):
        reports.persist_report(raw, tmp_path / "driver")
    assert not (tmp_path / "driver").exists()


def test_spool_full_preserves_first_report_and_does_not_fail_training(
    tmp_path, monkeypatch, caplog
):
    path = report(tmp_path)
    raw = path.read_bytes()
    monkeypatch.setattr(reports, "MAX_SPOOL_REPORTS", 1)
    assert report(tmp_path) is None
    assert path.read_bytes() == raw
    assert "local persistence failed" in caplog.text


def test_uploaded_report_eviction_during_scan_keeps_new_report(tmp_path, monkeypatch):
    old = report(tmp_path)
    assert old is not None
    actual_lstat = Path.lstat
    evicted = []

    def lstat(path, *args, **kwargs):
        if path == old and not evicted:
            # The uploader has acknowledged the old file and unlinks it after
            # persistence enumerates it, but before its quota stat completes.
            path.unlink()
            evicted.append(path)
        return actual_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(reports, "MAX_SPOOL_REPORTS", 1)
    delivered = []
    reports.set_report_sink(lambda path: delivered.append(path.read_bytes()))
    new = report(tmp_path)
    assert new is not None
    assert evicted == [old]
    assert not old.exists()
    assert delivered == [new.read_bytes()]
    assert reports.validate_report(delivered[0])["id"] == new.stem
    assert list(new.parent.iterdir()) == [new]


def test_spool_symlink_refuses(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (tmp_path / "spool").symlink_to(target, target_is_directory=True)
    assert report(tmp_path) is None
    assert not list(target.iterdir())


def test_replay_reruns_real_memory_estimator_and_prefix_layout(tmp_path):
    from art.trainer_rank._prefix_tree_planner import (
        build_canonical_prefix_tree,
        plan_prefix_tree_layout,
    )

    tokens = [[1, 2, 3], [1, 2, 4]]
    tree = build_canonical_prefix_tree(tokens)
    layout = plan_prefix_tree_layout(tree, tree.decision_indices)
    state: dict[str, Any] = {
        "rank": {
            "num_layers": 4,
            "hidden_size": 8,
            "param_dtype_size": 2,
            "recompute_granularity": "full",
            "one_layer_recompute": True,
            "sequence_parallel": False,
            "attention_output_gate": False,
            "mlp_activation_factor": 3,
            "gdn_layers": 0,
            "checkpointed_moe_layers": 0,
            "recompute_modules": [],
            "moe_output_bytes_per_token": 0,
            "moe_forward_stages": [],
            "geometry": {
                "hidden_size": 8,
                "ffn_hidden_size": 32,
                "num_attention_heads": 2,
                "num_query_groups": 2,
                "kv_channels": 4,
            },
            "topology": [1, 1, 1, 1],
        },
        "estimates": [
            {
                "signature": {
                    "topology": [1, 1, 1, 1],
                    "planner_coefficients": [2, None],
                    "slot_group_count": 1,
                    "request_mix": ["hidden_states"],
                    "grad_enabled": True,
                    "grad_modes": [True],
                    "slot_shapes": [[True, [[4, 8], []]], [False, []]],
                },
                "profile": {"bytes_per_token": 100.0, "packed_tokens": 4},
                "arguments": {
                    "packed_tokens": 4,
                    "output_bytes": 0,
                    "logical_tokens": 4,
                    "gdn_segments": 0,
                    "retained_tokens": 4,
                    "group_rows": [],
                },
                "expected_required_bytes": 440,
                "retained_bytes": 440,
                "cost_components": {
                    "required": 440,
                    "retained": 440,
                    "checkpoint_retained": 0,
                    "checkpoint_workspace": 0,
                    "checkpoint_input_gradient": 0,
                    "checkpoint_peak_increment": 0,
                    "hybridep_growth": 0,
                },
            }
        ],
    }
    payload = {
        "memory_replay": state,
        "split_memory_floor_bytes": 0,
        "local_admission_peak_bytes": 440,
        "reduced_admission_peak_bytes": 9999,
        "safety_factor": 1.1,
        "layouts": [
            {
                "input_tokens": tokens,
                "selected_decisions": list(tree.decision_indices),
                "expected_packed_tokens": layout.packed_tokens,
                "expected_fingerprint": layout.fingerprint,
            }
        ],
    }
    path = report(
        tmp_path,
        predicted_peak_bytes=400,
        observed_peak_bytes=500,
        admission_peak_bytes=9999,
        replay_factory=lambda: payload,
    )
    result = reports.replay(reports.validate_report(path.read_bytes()))
    assert result["estimates"] == [
        {"required_bytes": 440, "retained_bytes": 440, "matches": True}
    ]
    assert result["aggregate"] == {
        "local_admission_peak_bytes": 440,
        "predicted_peak_bytes": 400,
        "matches": True,
    }
    assert result["layouts"][0]["matches"]
    assert result["source_matches"] is True
    for field in (
        "local_admission_peak_bytes",
        "split_memory_floor_bytes",
        "safety_factor",
    ):
        changed = json.loads(path.read_bytes())
        changed["replay"][field] += 1000
        assert reports.replay(changed)["aggregate"]["matches"] is False
    for field in ("predicted_peak_bytes", "admission_peak_bytes"):
        changed = json.loads(path.read_bytes())
        changed[field] += 1
        assert reports.replay(changed)["aggregate"]["matches"] is False
    unrecorded = reports.validate_report(path.read_bytes())
    unrecorded["replay"]["memory_replay"]["rank"]["one_layer_recompute"] = None
    with pytest.raises(ValueError, match="recompute mode is not recorded"):
        reports.replay(unrecorded)
    for name in ("_impl.py", "_memory_policy.py", "_options.py"):
        drifted = reports.validate_report(path.read_bytes())
        drifted["replay"]["source_files"][name]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="source differs"):
            reports.replay(drifted)
        assert (
            reports.replay(drifted, allow_source_drift=True)["source_matches"] is False
        )
    assert "_gdn_memory.py" in reports._source_files()
    # Frozen stages are independent inputs, not a recorded total substituted
    # for the estimator. Changing a fixed stage changes actual recomputation.
    staged = json.loads(path.read_bytes())
    staged["replay"]["memory_replay"]["rank"]["moe_forward_stages"] = [[1, 1000]]
    actual = reports.replay(staged)
    assert actual["estimates"][0]["required_bytes"] == 1104
    assert actual["estimates"][0]["matches"] is False
    del staged["replay"]["memory_replay"]["rank"]["moe_forward_stages"]
    with pytest.raises(ValueError, match="rank fields differ"):
        reports.replay(staged)
    for rows in (None, [[0, False]], [[4, True]]):
        changed = json.loads(path.read_bytes())
        changed["replay"]["memory_replay"]["estimates"][0]["arguments"][
            "group_rows"
        ] = rows
        with pytest.raises(ValueError, match="immutable runtime"):
            reports.replay(changed)
    for field in (
        "checkpoint_input_gradient",
        "checkpoint_workspace",
        "checkpoint_peak_increment",
        "hybridep_growth",
    ):
        altered = json.loads(path.read_bytes())
        altered["replay"]["memory_replay"]["estimates"][0]["cost_components"][
            field
        ] += 1
        assert reports.replay(altered)["estimates"][0]["matches"] is False
    state["estimates"][0]["profile"]["bytes_per_token"] = 200.0
    changed = report(
        tmp_path,
        predicted_peak_bytes=400,
        observed_peak_bytes=500,
        admission_peak_bytes=9999,
        replay_factory=lambda: payload,
    )
    changed_result = reports.replay(reports.validate_report(changed.read_bytes()))
    assert changed_result["estimates"] == [
        {"required_bytes": 880, "retained_bytes": 880, "matches": False}
    ]
    assert changed_result["aggregate"]["matches"] is False


def test_incomplete_replay_refuses(tmp_path):
    path = report(tmp_path, replay_factory=lambda: {})
    with pytest.raises(ValueError, match="incomplete replay"):
        reports.replay(reports.validate_report(path.read_bytes()))


@pytest.mark.parametrize("floor", [0, 10_000])
def test_actual_emitted_split_retains_evidence_but_refuses_missing_runtime_facts(
    monkeypatch, tmp_path, floor
):
    from test_trainer_rank_split import _rank
    import torch

    from art.trainer_rank import _impl as tr

    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path / "emitted")
    children = []
    for suffix in (3, 5):
        requests = [
            tr.ForwardInput(
                input_tokens=torch.tensor([1, 2, token]), hidden_states=True
            )
            for token in (suffix, suffix + 1)
        ]
        child = rank._plan_flat_forward(requests)
        rank._memory_profiles[child.signature] = tr._MemoryProfile(
            bytes_per_token=100,
            packed_tokens=20,
            logical_per_packed=2,
            retained_compute_bytes_per_token=10,
        )
        children.append(child)
    plan = tr._SplitForwardPlan(tuple(children), ((0, 1), (2, 3)), 4)
    key = rank._split_memory_key(plan)
    assert key is not None
    rank._split_memory_floors[key] = floor
    costs = [rank._plan_cost(child) for child in children]
    local = max(
        rank._split_required_memory(costs),
        int(floor * tr._MEMORY_SAFETY_FACTOR),
    )
    rank._begin_planner_observation(
        plan, tr._MemoryCheck(local + 123, local + 1000, True)
    )
    observation = rank._planner_observation
    assert observation is not None
    assert observation["comparable"] is True
    rank._moe_forward_stages = ((1, 1000),)
    # Close the real profiling window with fake allocator counters only;
    # CPU planning and the emitted replay snapshot are otherwise maintained code.
    observation["baseline"] = 0
    observation["peak"] = local * 2
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: local * 2)
    rank._complete_planner_observation(phase="forward")
    [path] = list(rank._planner_reporter.spool_dir.glob("*.json"))
    original = reports.validate_report(path.read_bytes())
    assert original["replay_complete"] is False
    assert "immutable runtime" in original["incomplete_reasons"][0]
    snapshot = original["replay"]
    assert snapshot["memory_replay"]["rank"]["moe_forward_stages"] == []
    assert len(snapshot["layouts"]) == 2
    assert snapshot["local_admission_peak_bytes"] == local
    assert snapshot["reduced_admission_peak_bytes"] == local + 123
    assert [
        item["cost_components"] for item in snapshot["memory_replay"]["estimates"]
    ] == [asdict(cost) for cost in costs]
    with pytest.raises(ValueError, match="immutable runtime"):
        reports.replay(original)
    monkeypatch.setattr("sys.argv", ["planner-replay", str(path)])
    with pytest.raises(ValueError, match="immutable runtime"):
        reports.main()
    # Flipping the top-level completeness bit cannot authorize missing facts,
    # including a family whose observed checkpoint floor happened to be zero.
    original["replay_complete"] = True
    for item in snapshot["memory_replay"]["estimates"]:
        item["missing_inputs"] = []
    with pytest.raises(ValueError, match="immutable runtime"):
        reports.replay(original)


@pytest.mark.parametrize("slots", [[], [[True, [[3, 8], []]], [False, []]]])
@pytest.mark.parametrize("placements", [[], [["gpu", "model"], ["replay", "cpu"]]])
def test_signature_json_roundtrip_is_immutable(slots, placements):
    from art.trainer_rank._impl import _MemorySignature

    values: dict[str, Any] = dict(
        topology=[1, 1, 1, 1],
        planner_coefficients=[2, None],
        slot_group_count=1,
        request_mix=["hidden_states"],
        grad_enabled=True,
        grad_modes=[True],
        slot_shapes=slots,
        memory_placement=placements,
    )
    old = dict(values)
    for name in ("topology", "planner_coefficients", "request_mix", "grad_modes"):
        old[name] = tuple(old[name])
    with pytest.raises(TypeError, match="unhashable"):
        hash(_MemorySignature(**old))
    key = _MemorySignature(**reports._signature_values(json.loads(json.dumps(values))))
    assert {key: 1}[key] == 1
    assert key.slot_shapes == tuple(
        (enabled, tuple(map(tuple, shapes))) for enabled, shapes in slots
    )
    assert key.memory_placement == tuple(map(tuple, placements))


@pytest.mark.parametrize(
    "slots",
    [None, [[1, []]], [[True, [[True]]]], [[False, [[-1]]]], [[True]], [[True, "bad"]]],
)
def test_invalid_signature_shapes_refuse(slots):
    with pytest.raises(ValueError, match="slot_shapes"):
        reports._signature_values(
            dict(
                topology=[],
                planner_coefficients=[],
                request_mix=[],
                grad_modes=[],
                slot_shapes=slots,
            )
        )


def test_observation_uses_checkpoint_aware_aggregate(monkeypatch, tmp_path):
    from test_trainer_rank_split import _rank
    import torch

    from art.trainer_rank import _impl as tr

    rank = _rank(monkeypatch)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path / "aggregate")
    children = tuple(
        rank._plan_flat_forward(
            [tr.ForwardInput(input_tokens=torch.tensor([i, 2]), hidden_states=True)]
        )
        for i in (1, 3)
    )
    costs = [
        tr._SubforwardCost(
            required=220,
            retained=110,
            checkpoint_retained=100,
            checkpoint_workspace=0,
            checkpoint_input_gradient=100,
        )
        for _ in children
    ]
    old = sum(c.retained for c in costs) + max(c.ephemeral for c in costs)
    assert old == 330
    assert rank._split_required_memory(costs) == 440
    monkeypatch.setattr(rank, "_plan_cost", lambda child: costs[0])
    split = tr._SplitForwardPlan(children, ((0,), (1,)), 2)
    rank._begin_planner_observation(split, tr._MemoryCheck(440, 500, True))
    observation = rank._planner_observation
    assert observation is not None
    assert observation["predicted"] == 400
    assert observation["replay"]()["local_admission_peak_bytes"] == 440
