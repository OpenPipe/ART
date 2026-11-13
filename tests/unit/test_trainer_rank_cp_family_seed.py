"""Recorded CP caller evidence; additional seed readings are synthetic CPU input."""

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_calibration_seed import fake_device, metadata_rank
from test_trainer_rank_calibration_trust_region import request
import torch

from art.megatron.context_parallel.runtime import context_parallel_memory_workload
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.megatron.gdn.gdn_prefix_tree import GdnPlannerConfig
from art.trainer_rank import TrainerRankMemoryError
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports
from art.trainer_rank._planner_cost import ModelGeometry

DATA = json.loads(
    (
        Path(__file__).parent / "fixtures/memory_fixed_slope/cp_caller_families.json"
    ).read_text()
)


def recorded(kind, lora):
    row = next(r for r in DATA["rows"] if r["kind"] == kind and r["lora_rank"] == lora)
    return (
        row,
        reports._memory_signature(row["signature"]),
        reports._memory_profile(row["profile"]),
    )


@lru_cache
def workload(template, n):
    value = json.loads(template)
    config = dict(value["attention_config"])
    config["attention_sparse_block_size"] = tuple(config["attention_sparse_block_size"])
    metadata = torch.ones((1, n), dtype=torch.int64)
    family, loads = context_parallel_memory_workload(
        group_ids=metadata,
        parent_ids=metadata,
        topology=ParallelTopology(cp=8),
        config=ContextParallelConfig(**config),
        original_seq_len=n,
        build_gdn_execution_spec=True,
        gdn_planner_config=GdnPlannerConfig(**json.loads(value["gdn_config"])),
    )
    requested = json.loads(family)
    requested["execution_qualification"] = value["execution_qualification"]
    return json.dumps(requested, sort_keys=True, separators=(",", ":")), loads


def estimator(key, profile):
    rank = _rank()
    if isinstance(profile, memory._CallerMemoryProfile):
        qualification = json.loads(profile.caller_cp_peaks[0][2])[
            "execution_qualification"
        ]
        rank._geometry = ModelGeometry(**qualification["geometry"])
        rank._hidden_size = rank._geometry.hidden_size
        rank._num_layers = qualification["layers"]
        rank._gdn_layers = qualification["gdn_layers"]
        rank._moe_layers = qualification["moe_layers"]
    rank._memory_profiles = {key: profile}
    return rank


def required(rank, key, n, work=None):
    return rank._estimate_required_memory_bytes_from_values(
        packed_tokens=n,
        logical_tokens=n,
        output_bytes=4 * n,
        gdn_segments=1,
        signature=key,
        caller_cp_workload=work,
    )


@pytest.mark.parametrize("lora", [1, 8, 32])
def test_recorded_short_recurrence_is_uncovered_despite_identical_kernels(lora):
    row, key, profile = recorded("moe", lora)
    rank = estimator(key, profile)
    template = profile.caller_cp_peaks[0][2]
    short = workload(template, 512)
    with memory.fixed_slope_memory_profiles():
        assert not memory._unanchored(profile)
        assert memory._rate_reference(rank, key, profile).source == "signature"
        assert memory._rate_floor_tokens(key) == 512
        assert not memory._uncovered_subfloor_cp(profile, key, 512, short)
        assert required(rank, key, 512, short) == row["required"] == 5072854620278311936
        for n in (512, 2048, 4096, 8192, 16384):
            work = workload(template, n)
            want = json.loads(work[0])
            assert (
                want["execution_qualification"]
                == json.loads(template)["execution_qualification"]
            )
            assert want["gdn"][1] == [n >= 4096]
            assert (profile.covering_cp_peak(n, work) is None) == (n < 4096)
        assert all(
            w <= h
            for ws, hs in zip(short[1], profile.caller_cp_peaks[1][3], strict=True)
            for w, h in zip(ws, hs, strict=True)
        )


def family_rank(monkeypatch, lora=1):
    _, key, profile = recorded("moe", lora)
    rank = metadata_rank("moe")
    rank._memory_profiles = {key: profile}
    rank._calibration_memory_rungs = {key: (profile, profile.compute_observations)}
    rank._available_memory_bytes = lambda sample=None: 44_000_000_000
    original = rank._plan_flat_forward
    template = profile.caller_cp_peaks[0][2]

    def plan(*args, **kwargs):
        value = original(*args, **kwargs)
        return replace(
            value,
            signature=key,
            caller_cp_workload=workload(template, value.packed_tokens),
        )

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    return rank, key, profile


@pytest.mark.parametrize("lora", [1, 8, 32])
def test_explicit_measured_family_seed_covers_smaller_waves_without_teaching_rates(
    monkeypatch, lora
):
    with memory.fixed_slope_memory_profiles():
        rank, key, before = family_rank(monkeypatch, lora)
        template = before.caller_cp_peaks[0][2]
        plans = {
            n: rank._plan_flat_forward([request(n, grad=True)])
            for n in (512, 2048, 4096, 8192)
        }
        prices = {
            n: rank._memory_check(p).estimated_required_bytes for n, p in plans.items()
        }
        assert prices[512] > 1 << 62 and prices[2048] > 1 << 62
        # This is an injected new measurement, never a recorded short-family peak.
        fake_device(monkeypatch, rank, 300_000_000 + 4 * 2048)
        with rank.calibration_rung(
            [request(2048, grad=True)], no_grad=False, seed_uncovered_cp_family=True
        ) as rung:
            assert (
                rung.decision.seed and rung.decision.extrapolated and rung.decision.fits
            )
            assert rung.decision.seed_margin == 2
            assert rank._memory_profiles[key] is before
            assert not rank._memory_check(plans[512]).fits
            list(rung.forward_micro_batches())
        after = rank._memory_profiles[key]
        assert after.rate_observations == before.rate_observations
        assert after.compute_slope == before.compute_slope
        assert after.warm_bytes_per_token == before.warm_bytes_per_token
        assert after.eligible_packed_tokens == before.eligible_packed_tokens
        for n in (512, 2048):
            assert after.covering_cp_peak(n, workload(template, n)) == 300_000_000
            assert rank._memory_check(plans[n]).fits
            assert rank._memory_check(plans[n]).estimated_required_bytes <= prices[8192]
        assert rank._memory_check(plans[4096]).estimated_required_bytes == prices[4096]
        # Changing actual kernel identity still requires a new measurement.
        family, loads = workload(template, 512)
        values = json.loads(family)
        values["execution_qualification"]["kernels"] = ["unknown"]
        unknown = replace(plans[512], caller_cp_workload=(json.dumps(values), loads))
        assert not rank._memory_check(unknown).fits


@pytest.mark.parametrize("kind", ["moe", "dense"])
@pytest.mark.parametrize("lora", [1, 8, 32])
def test_ordinary_pricing_is_unchanged_without_new_family_measurement(kind, lora):
    _, key, profile = recorded(kind, lora)
    rank = estimator(key, profile)
    template = profile.caller_cp_peaks[0][2] if kind == "moe" else None
    with memory.fixed_slope_memory_profiles():
        for n in (512, 2048, 4096):
            work = None if template is None else workload(template, n)
            price = required(rank, key, n, work)
            assert (price >= 1 << 62) == (kind == "moe" and n < 4096)
        assert rank._memory_profiles[key] is profile


@pytest.mark.parametrize(
    "invalid",
    ["tiny", "large", "covered", "stale", "unanchored", "revoked", "disabled", "scope"],
)
def test_family_seed_cannot_override_other_gates(monkeypatch, invalid):
    rank, key, profile = family_rank(monkeypatch)
    n = {"tiny": 511, "large": 8192, "covered": 4096}.get(invalid, 2048)
    if invalid == "stale":
        rank._calibration_memory_rungs = {}
    if invalid in ("unanchored", "revoked", "disabled"):
        profile = replace(
            profile,
            **{
                "unanchored": {"rate_observations": ()},
                "revoked": {"non_dominant_cp_pairs": (("a", "b"),)},
                "disabled": {"dominance_disabled": True},
            }[invalid],
        )
        rank._memory_profiles[key] = profile
        rank._calibration_memory_rungs[key] = (profile, profile.compute_observations)
    with memory.fixed_slope_memory_profiles(invalid != "scope"):
        with pytest.raises(ValueError, match="CP family seed"):
            with rank.calibration_rung(
                [request(n, grad=True)], no_grad=False, seed_uncovered_cp_family=True
            ):
                pass


def test_incomplete_family_seed_teaches_no_coverage_and_cannot_retry(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank, key, profile = family_rank(monkeypatch)
        fake_device(monkeypatch, rank, 300_000_000)
        with rank.calibration_rung(
            [request(2048, grad=True)], no_grad=False, seed_uncovered_cp_family=True
        ) as rung:
            iterator = rung.forward_micro_batches()
            next(iterator)
            iterator.close()
        plan = rank._plan_flat_forward([request(512, grad=True)])
        assert not rank._memory_check(plan).fits
        assert rank._memory_profiles[key].rate_observations == profile.rate_observations
        with pytest.raises(ValueError, match="CP family seed"):
            with rank.calibration_rung(
                [request(2048, grad=True)], no_grad=False, seed_uncovered_cp_family=True
            ):
                pass
        with rank.calibration_rung([request(512, grad=True)], no_grad=False) as rung:
            with pytest.raises(TrainerRankMemoryError):
                rung.forward_micro_batches()
