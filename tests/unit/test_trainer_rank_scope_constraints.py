"""Profile-scope constraints and pricing-source drift; synthetic CPU readings."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import pickle
from types import MethodType
from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_cp_output_families import anchors, native_rank, output_requests
import torch

from art.trainer_rank import _impl
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_family_scope_transition_preserves_generic_observation(
    monkeypatch, tmp_path, kind, grad, bounded
):
    rank = native_rank(kind)
    with (
        memory.fixed_slope_memory_profiles(),
        torch.set_grad_enabled(grad),
        memory.bounded_forward_peaks() if bounded else nullcontext(),
    ):
        anchors(rank, grad)
        inputs = output_requests(2971, grad, "none", True, False, None, False)
        original = rank._plan_flat_forward(inputs)
        rank._update_memory_profile(
            original,
            10_000_000_000 + original.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )
        before = rank._memory_check(rank._plan_flat_forward(inputs))
        assert before.estimated_required_bytes >= 10_000_000_000
        fake_device(monkeypatch, rank, 1_000_000_000)
        with rank.calibration_rung(
            output_requests(4095, grad, "none", True, False, None, False),
            no_grad=not grad,
            seed_uncovered_cp_family=True,
        ) as rung:
            assert rung.decision.seed and rung.decision.fits
            list(rung.forward_micro_batches())
        assert original.signature != rank._plan_flat_forward(inputs).signature
        # Profile dictionaries retain both evidence channels through reload.
        json_profiles = {
            reports._memory_signature(
                json.loads(json.dumps(asdict(key)))
            ): reports._memory_profile(json.loads(json.dumps(asdict(value))))
            for key, value in rank._memory_profiles.items()
        }
        for restored in (
            json_profiles,
            pickle.loads(pickle.dumps(rank._memory_profiles)),
        ):
            rank._memory_profiles = restored
            plan = rank._plan_flat_forward(inputs)
            assert rank._memory_check(plan).estimated_required_bytes >= 10_000_000_000
            assert rank._plan_cost(plan).required >= 10_000_000_000
        rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        repeated = rank._plan_flat_forward(inputs)
        rank._available_memory_bytes = lambda sample=None: 5_000_000_000
        after = rank._memory_check(repeated)
        print(
            dict(
                kind=kind,
                grad=grad,
                old_signature=repr(original.signature),
                new_signature=repr(repeated.signature),
                old_peak=before.estimated_required_bytes,
                new_peak=after.estimated_required_bytes,
                fits=after.fits,
            )
        )
        assert not after.fits, (
            "scope transition must preserve the known 10 GB observation"
        )
        from test_grouped_planner_replay import emitted

        report, _ = emitted(rank, repeated, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        assert reports.replay(report)["aggregate"]["matches"]


def test_optimizer_source_drift_changes_fingerprint(monkeypatch, tmp_path):
    from pathlib import Path
    import shutil

    from art.trainer_rank import _planner_misses as reports

    source_dir = Path(memory.__file__).parent
    # Copy the fingerprint's own source inventory and its new math dependency.
    for name in (*reports._SOURCE_NAMES, "_optimizer.py"):
        shutil.copyfile(source_dir / name, tmp_path / name)
    monkeypatch.setattr(reports, "__file__", str(tmp_path / "_planner_misses.py"))
    before = reports._source_files()
    optimizer_path = tmp_path / "_optimizer.py"
    old = optimizer_path.read_text()
    new = old.replace(
        "_OPTIMIZER_BYTES_PER_ELEMENT = 36", "_OPTIMIZER_BYTES_PER_ELEMENT = 72"
    )
    assert new != old
    optimizer_path.write_text(new)
    assert reports._source_files() != before, (
        "changing optimizer admission math must be source drift"
    )


@pytest.mark.parametrize(
    "change", ["operation", "output", "adapter", "partition", "topology", "short"]
)
def test_scope_floor_does_not_borrow_from_another_request_class(change):
    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        plan = rank._plan_flat_forward(
            output_requests(2971, False, "none", True, False, None, False)
        )
        rank._update_memory_profile(
            plan,
            10_000_000_000 + plan.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )
        signature = plan.signature
        if change == "operation":
            signature = replace(signature, grad_enabled=True, grad_modes=(True,))
        elif change == "output":
            signature = replace(signature, request_mix=("target:single",))
        elif change == "adapter":
            signature = replace(signature, slot_shapes=((False, ((2, 16, 2, 2, 16),)),))
        elif change == "partition":
            signature = memory._ScopedMemorySignature(
                **vars(signature), checkpoint_partition=(8, (1, 1, 1))
            )
        elif change == "topology":
            signature = replace(signature, topology=(1, 1, 4, 1))
        else:
            signature = replace(signature, short_requests=True)
        floor, _ = memory._cp_fixed_phase_bytes(
            rank, signature, 0, None, include_optimizer=False, packed_tokens=2971
        )
        with patch.object(rank, "_memory_profiles", {}):
            native_floor, _ = memory._cp_fixed_phase_bytes(
                rank, signature, 0, None, include_optimizer=False, packed_tokens=2971
            )
        assert floor == native_floor


@pytest.mark.parametrize("source", ["_optimizer.py", "_prefix_tree_materializer.py"])
def test_pricing_dependency_drift_refuses_replay(monkeypatch, tmp_path, source):
    from pathlib import Path
    import shutil

    from test_grouped_planner_replay import emitted
    from test_trainer_rank_calibration_trust_region import request
    from test_trainer_rank_cp_request_floor import prepared_rank, record_minimal_caller

    from art.trainer_rank import _optimizer

    source_dir = Path(memory.__file__).parent
    for name in reports._SOURCE_NAMES:
        shutil.copyfile(source_dir / name, tmp_path / name)
    monkeypatch.setattr(reports, "__file__", str(tmp_path / "_planner_misses.py"))
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = prepared_rank("moe")
        rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
        record_minimal_caller(rank, True, 100_000_000)
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        plan = rank._plan_flat_forward([request(72, grad=True)])
        report, costs = emitted(rank, plan, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        assert reports.replay(report)["source_matches"]
        path = tmp_path / source
        text = path.read_text()
        if source == "_optimizer.py":
            # Make the changed phase exceed the independent caller/retention
            # bound, so the test also proves the recomputed cost changes.
            text = text.replace(
                "_OPTIMIZER_BYTES_PER_ELEMENT = 36",
                "_OPTIMIZER_BYTES_PER_ELEMENT = 360",
            )
            monkeypatch.setattr(_optimizer, "_OPTIMIZER_BYTES_PER_ELEMENT", 360)
        else:
            text += "\n# Changed layout materialization dependency.\n"
        path.write_text(text)
        with pytest.raises(ValueError, match="source differs"):
            reports.replay(report)
        changed = reports.replay(report, allow_source_drift=True)
        assert not changed["source_matches"]
        if source == "_optimizer.py":
            assert changed["estimates"][0]["required_bytes"] > costs[0].required
            assert not changed["aggregate"]["matches"]
