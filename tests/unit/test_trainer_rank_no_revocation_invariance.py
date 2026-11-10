"""Recorded CPU bounds without revocation, including fixed request costs."""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path

import pytest
from test_trainer_rank_calibration_seed import fake_device, metadata_rank
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_subfloor_cp import requests, short_kernel
import torch

from art.trainer_rank import _impl
from art.trainer_rank import _memory as memory


def seed_profiles(monkeypatch, kind):
    rank = metadata_rank(kind)
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    fake_device(monkeypatch, rank, 2_500_000_000)
    for grad in (False, True):
        with torch.set_grad_enabled(grad):
            with rank.calibration_rung(
                [request(16384, grad=grad)], no_grad=not grad
            ) as rung:
                list(rung.forward_micro_batches())
    from test_trainer_rank_cp_output_families import fixture_floor

    for grad in (False, True):
        fixture_floor(rank, grad)
        with short_kernel():
            fixture_floor(rank, grad)
    return rank


def admission_record(rank, case):
    lengths, grads = case["lengths"], case["grads"]
    inputs = [
        replace(value, no_grad=not grad)
        for value, grad in zip(requests(lengths, True), grads, strict=True)
    ]
    rank._available_memory_bytes = lambda sample=None: case["budget_bytes"]
    with (
        torch.set_grad_enabled(any(grads)),
        short_kernel() if case["kernel"] == "short" else nullcontext(),
    ):
        flat = rank._plan_flat_forward(inputs)
        unsplit = rank._memory_check(flat)
        result = rank._find_admissible_forward(
            inputs, checkpoint=None, refusal_prefix="recorded CPU admission"
        )
    if isinstance(result, _impl._ForwardRefusal):
        plan, check, decision = result.plan, result.check, "refused"
    else:
        plan, check = result
        decision = "split" if isinstance(plan, _impl._SplitForwardPlan) else "flat"
    children = (
        plan.subforwards if isinstance(plan, _impl._SplitForwardPlan) else (plan,)
    )
    return {
        "decision": decision,
        "fits": check.fits,
        "required_bytes": check.estimated_required_bytes,
        "available_bytes": check.available_bytes,
        "unsplit_required_bytes": unsplit.estimated_required_bytes,
        "unsplit_fits": unsplit.fits,
        "packed_tokens": [child.packed_tokens for child in children],
        "request_counts": [child.request_count for child in children],
    }


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_no_revocation_admission_matches_recorded_baseline(monkeypatch, kind):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/memory_fixed_slope/cp_no_revocation.json"
        ).read_text()
    )
    overrides = json.loads(
        (
            Path(__file__).parent
            / "fixtures/memory_fixed_slope/cp_request_floor_overrides.json"
        ).read_text()
    )
    expected = {
        (row["kind"], row["id"]): row["expected"] for row in overrides["overrides"]
    }
    with memory.fixed_slope_memory_profiles():
        rank = seed_profiles(monkeypatch, kind)
        for case in fixture["cases"]:
            if case["kind"] != kind:
                continue
            assert all(
                not getattr(p, "non_dominant_cp_pairs", ())
                and not getattr(p, "dominance_disabled", False)
                for p in rank._memory_profiles.values()
            )
            record = admission_record(rank, case)
            assert record == expected.get((kind, case["id"]), case["expected"]), case
            assert (
                record["unsplit_required_bytes"]
                >= case["expected"]["unsplit_required_bytes"]
            )
            if case["expected"]["decision"] == "refused":
                assert record["decision"] == "refused"
