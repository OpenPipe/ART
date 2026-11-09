"""CPU witnesses for retained graphs and qualified cross-profile borrowing."""

import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_fixed_slope import signature
from test_trainer_rank_unanchored_floor import observe, required
import torch.multiprocessing as mp

from art.trainer_rank import _impl
from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports


def runtime_observe(monkeypatch, rank, key, n, peak, retained=None):
    plan = SimpleNamespace(
        signature=key, packed_tokens=n, active_logical_tokens=n, output_bytes=0
    )
    monkeypatch.setattr(_impl.torch.cuda, "max_memory_allocated", lambda *_: peak)
    m._update_peak_memory_profile(rank, cast(Any, plan), 0, retained_after=retained)


def child_cost(rank, key, n=8192):
    return rank._subforward_cost(
        packed_tokens=n,
        logical_tokens=n,
        output_bytes=0,
        signature=key,
        group_rows=((n // 8, True),),
    )


def test_ineligible_observation_cannot_discount_observed_split_retention(monkeypatch):
    rank, key = _rank(), signature((0, "forward_backward", 8))
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 20_000_000_000
    )
    seed = SimpleNamespace(
        signature=key, packed_tokens=512, active_logical_tokens=512, output_bytes=0
    )
    with m.fixed_slope_memory_profiles():
        m._update_memory_profile(
            rank, cast(Any, seed), 512_000_000, retained_bytes=512_000
        )
        assert required(rank, key, 8192) < 20_000_000_000
        runtime_observe(monkeypatch, rank, key, 8192, 8_000_000_000, 8_000_000_000)
        assert m._trusted_extent(rank._memory_profiles[key]) == 512
        cost = child_cost(rank, key)
        assert cost.retained >= 8_000_000_000
        split = rank._split_required_memory([cost] * 3)
        assert split >= 24_000_000_000
        assert not rank._memory_check_required(split).fits


def test_later_qualified_rates_cannot_erase_observed_retention(monkeypatch):
    rank, key = _rank(), signature((0, "forward_backward", 8))
    with m.fixed_slope_memory_profiles():
        runtime_observe(monkeypatch, rank, key, 256, 300_000_000)
        runtime_observe(monkeypatch, rank, key, 512, 512_000_000, 512_000)
        runtime_observe(monkeypatch, rank, key, 8192, 8_000_000_000, 8_000_000_000)
        runtime_observe(monkeypatch, rank, key, 2048, 1_000_000_000, 1_000_000)
        runtime_observe(monkeypatch, rank, key, 4096, 2_000_000_000, 2_000_000)
        for n in (8192, 16384, 65536):
            assert child_cost(rank, key, n).retained >= 8_000_000_000


def caller_borrow(rank):
    from test_moe_caller_cp_coverage import workload
    from test_moe_caller_cp_invariants import sig

    root = sig(8)
    target = _impl._MemorySignature(
        **{
            name: getattr(root, name)
            for name in _impl._MemorySignature.__dataclass_fields__
        }
    )
    measured, wanted = workload(8, 8320), workload(8, 8192)
    plan = SimpleNamespace(
        signature=root,
        packed_tokens=8320,
        active_logical_tokens=8320,
        output_bytes=0,
        caller_cp_workload=measured,
    )
    m._update_memory_profile(
        rank, cast(Any, plan), 16 * 2**30, retained_bytes=None, caller_phase=True
    )
    observe(rank, target, 3, 300_000_000)
    return root, target, measured, wanted


def test_borrowed_cp_caller_must_cover_ownership_and_workspace():
    rank = _rank()
    with m.fixed_slope_memory_profiles():
        root, target, measured, wanted = caller_borrow(rank)
        assert measured[1][6][0] == 512 and wanted[1][6][0] == 1024
        source = rank._memory_profiles[root]
        assert (
            source.compute_peak(8192, context_parallel=True, cp_workload=wanted)
            >= m._MEMORY_DEFERRED_BYTES
        )
        demand = rank._estimate_required_memory_bytes_from_values(
            packed_tokens=8192,
            logical_tokens=8192,
            output_bytes=0,
            signature=target,
            caller_cp_workload=wanted,
        )
        assert demand >= 32 * 2**30
        assert demand > 33_800_000_000
        covered = rank._estimate_required_memory_bytes_from_values(
            packed_tokens=8320,
            logical_tokens=8320,
            output_bytes=0,
            signature=target,
            caller_cp_workload=measured,
        )
        assert 16 * 2**30 <= covered < 100_000_000_000
        # Logical-row pricing cannot bypass the source's covering token extent.
        assert (
            rank._estimate_required_memory_bytes_from_values(
                packed_tokens=8320,
                logical_tokens=8320 * 64,
                output_bytes=0,
                signature=target,
                caller_cp_workload=measured,
            )
            >= m._MEMORY_DEFERRED_BYTES
        )


@pytest.mark.parametrize(
    "mismatch", ["adapter", "checkpoint", "forward_only", "groups", "caller_interval"]
)
def test_model_fallback_rejects_incompatible_geometry_and_scope(mismatch):
    rank, target = _rank(), signature((0, "forward_backward", 32))
    source = replace(target, request_mix=("hidden",))
    if mismatch == "adapter":
        source = replace(signature((0, "forward_backward", 1)), request_mix=("hidden",))
    elif mismatch == "checkpoint":
        source = m._ScopedMemorySignature(
            **vars(source), checkpoint_partition=(1, (1, 1, 8))
        )
    elif mismatch == "forward_only":
        source = m._ScopedMemorySignature(**vars(source), bounded_forward_peak=True)
    elif mismatch == "groups":
        source = replace(source, slot_group_count=2)
    with m.fixed_slope_memory_profiles():
        if mismatch == "caller_interval":
            plan = SimpleNamespace(
                signature=source,
                packed_tokens=8192,
                active_logical_tokens=8192,
                output_bytes=0,
            )
            m._update_memory_profile(
                rank,
                cast(Any, plan),
                1_000_000_000,
                retained_bytes=None,
                caller_phase=False,
            )
        else:
            observe(rank, source, 8192, 1_000_000_000)
        observe(rank, target, 3, 300_000_000)
        assert required(rank, target, 32768) >= m._MEMORY_DEFERRED_BYTES
        assert required(rank, target, 131072) >= m._MEMORY_DEFERRED_BYTES


def test_global_eligibility_does_not_depend_on_rank_local_hull(monkeypatch, tmp_path):
    from test_moe_caller_cp_ownership import admission_worker

    key, demands = signature(), []
    with m.fixed_slope_memory_profiles():
        for i in range(8):
            rank = _rank()
            middle = 400_000_000 if i % 2 else 350_000_000
            for n, peak in ((3, 300_000_000), (256, middle), (8192, 2_000_000_000)):
                runtime_observe(monkeypatch, rank, key, n, peak)
            assert m._trusted_extent(rank._memory_profiles[key]) == 256
            runtime_observe(monkeypatch, rank, key, 512, 500_000_000)
            assert rank._memory_profiles[key].rate_observations == ((512, 500_000_000),)
            demands.append(required(rank, key, 32768))
    assert demands == [65_450_000_000] * 8
    mp.start_processes(
        admission_worker,
        args=(8, str(tmp_path / "gloo"), str(tmp_path), demands, [70_000_000_000] * 8),
        nprocs=8,
        join=True,
        start_method="fork",
    )
    results = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(8)]
    assert all(row["fits"] for row in results)
    assert len({row["required"] for row in results}) == 1


def test_workspace_fixture_has_synthetic_provenance():
    path = Path(m.__file__).parents[3] / "tests/unit/test_trainer_rank_fixed_slope.py"
    source = path.read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_tiny_forward_does_not_poison_large_forward"
    )
    text = ast.get_source_segment(source, node)
    assert text is not None
    assert "production" not in text.lower()
    assert "synthetic" in text.lower()


def test_independent_extents_and_retained_floors_merge_and_roundtrip():
    from art.trainer_rank import _planner_misses as reports

    key = signature((0, "forward_backward", 8))
    profiles = []
    with m.fixed_slope_memory_profiles():
        for rows in (
            ((3, 300_000_000, 200_000_000, True), (512, 512_000_000, 512_000, True)),
            (
                (256, 400_000_000, 300_000_000, True),
                (8192, 8_000_000_000, 8_000_000_000, False),
            ),
        ):
            rank = _rank()
            for n, peak, retained, eligible in rows:
                plan = SimpleNamespace(
                    signature=key,
                    packed_tokens=n,
                    active_logical_tokens=n,
                    output_bytes=0,
                )
                m._update_memory_profile(
                    rank,
                    cast(Any, plan),
                    peak,
                    retained_bytes=retained,
                    rate_eligible=eligible,
                )
            profiles.append(rank._memory_profiles[key])
        fields = m._merge_memory_profile_model_fields(*profiles)
        assert fields == m._merge_memory_profile_model_fields(*reversed(profiles))
        assert fields["eligible_packed_tokens"] == 512
        assert fields["subfloor_packed_tokens"] == 256
        assert fields["retained_rate_tokens"] == 512
        for n, peak in (
            (3, 200_000_000),
            (256, 300_000_000),
            (8192, 8_000_000_000),
        ):
            assert m._retained_floor_peak(fields["retained_compute_peaks"], n) >= peak
        merged = replace(profiles[0], **fields, packed_tokens=8192)
        restored = reports._memory_profile(json.loads(json.dumps(asdict(merged))))
        assert restored == merged and pickle.loads(pickle.dumps(merged)) == merged
        legacy = asdict(merged)
        legacy["profile_format_version"] = 3
        for field in (
            "eligible_packed_tokens",
            "subfloor_packed_tokens",
            "retained_rate_tokens",
            "retained_compute_peaks",
        ):
            legacy.pop(field)
        old = m._memory_profile_from_values(legacy)
        for profile in (restored, old):
            m._validate_memory_profile_model(profile)
            rank._memory_profiles[key] = profile
            assert child_cost(rank, key).retained >= 8_000_000_000
        # Updating an old encoding cannot revive its discount after discarding
        # the earlier untrusted retained reading's unavailable exact extent.
        plan = SimpleNamespace(
            signature=key,
            packed_tokens=4096,
            active_logical_tokens=4096,
            output_bytes=0,
        )
        m._update_memory_profile(
            rank, cast(Any, plan), 2_000_000_000, retained_bytes=2_000_000
        )
        assert child_cost(rank, key).retained >= 8_000_000_000


def test_promoting_incomplete_caller_preserves_its_forward_floor():
    rank = _rank()
    key = replace(signature((0, "forward_backward", 8)), topology=(1, 1, 1, 1))
    root = m._ScopedMemorySignature(**vars(key), bounded_caller_peak=True)
    # Earlier caller-v2 records keep incomplete peaks only in the safety floor.
    previous = m._CallerMemoryProfileV2(
        bytes_per_token=20 * 2**30 / 32768,
        packed_tokens=32768,
        compute_peak_floor=20 * 2**30,
    )
    m._validate_memory_profile_model(previous)
    rank._memory_profiles[root] = reports._memory_profile(
        json.loads(json.dumps(asdict(previous)))
    )
    with m.fixed_slope_memory_profiles():
        observe(rank, root, 8192, 16 * 2**30)
        profile = rank._memory_profiles[root]
        assert profile.absolute_compute_floor == 20 * 2**30
        assert profile.compute_slope == (16 * 2**30, 8192)
        assert m._trusted_extent(profile) == 8192
        reference = m._rate_reference(rank, key, None, packed_tokens=8192)
        assert reference is not None and reference.source == "calibration_root"
        assert reference.fixed >= 20 * 2**30
