"""Fixed per-request CP costs; allocator peaks are sanitized scalar fixtures."""

from typing import Any, cast

import pytest
from test_trainer_rank_calibration_trust_region import observe, request
from test_trainer_rank_cp_output_families import native_rank
import torch

from art.trainer_rank import _memory as memory

ROWS = (
    (16, True, 1_684_410_368),
    (1, False, 62_968_320),
    (4, False, 33_579_008),
    (72, False, 73_775_616),
)


def prepared_rank(kind, *, include_floor=True):
    rank = native_rank(kind)
    from test_trainer_rank_cp_family_seed import recorded

    # Reuse the sanitized native rank-eight physical adapter metadata. No
    # parameter storage or accelerator execution is needed for these shapes.
    _, signature, _ = recorded(kind, 8)
    shapes = signature.slot_shapes[0][1]
    rank._slot_memory_shapes = lambda ref: shapes
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            # Complete product-shaped ladder extents, with sanitized scalar
            # readings. These establish coverage, not new GPU evidence.
            for n, peak in (
                (16384, 2_200_000_000),
                (8192, 1_300_000_000),
                (512, 431_280_128),
                (2048, 897_256_448),
                (4096, 853_261_312),
                (32768, 4_100_000_000),
                (65536, 8_700_000_000),
                (131072, 18_200_000_000),
            ):
                observe(rank, n, peak, grad=grad)
            if not grad:
                observe(rank, 262144, 37_200_000_000, grad=False)
    if include_floor:
        from test_trainer_rank_cp_output_families import fixture_floor

        for grad in (False, True):
            fixture_floor(rank, grad)
    return rank


def record_minimal_caller(rank, grad, peak):
    token = memory._CP_FAMILY_CALIBRATION.set(True)
    try:
        plan = rank._plan_flat_forward([request(1, grad=grad)])
    finally:
        memory._CP_FAMILY_CALIBRATION.reset(token)
    plan.__dict__["_calibration_cp_family_seed"] = True
    plan.__dict__["_exclude_subfloor_memory_sample"] = True
    rank._update_memory_profile(
        plan,
        peak + plan.output_bytes,
        retained_bytes=None,
        caller_phase=True,
        rate_eligible=False,
    )
    return plan


@pytest.mark.parametrize("kind", ["moe", "dense"])
@pytest.mark.parametrize("n,grad,measured", ROWS)
def test_tiny_request_including_following_optimizer_covers_observed_peak(
    kind, n, grad, measured
):
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        rank = prepared_rank(kind)
        record_minimal_caller(rank, grad, 311_326_208 if grad else 62_968_320)
        plan = rank._plan_flat_forward([request(n, grad=grad)])
        estimate = rank._memory_check(plan).estimated_required_bytes
        assert estimate < memory._MEMORY_DEFERRED_BYTES
        assert estimate >= measured, (kind, n, estimate, measured)


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_minimal_family_trial_is_once_complete_and_rate_isolated(
    monkeypatch, kind, grad
):
    from dataclasses import asdict
    import pickle

    from test_trainer_rank_calibration_seed import fake_device

    from art.trainer_rank import _planner_misses as reports

    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        rank = prepared_rank(kind, include_floor=False)
        before = {k: p.rate_observations for k, p in rank._memory_profiles.items()}
        fake_device(monkeypatch, rank, 100_000_000)
        with rank.calibration_rung(
            [request(1, grad=grad)],
            no_grad=not grad,
            seed_uncovered_cp_family=True,
        ) as rung:
            assert rung.decision.seed and rung.decision.seed_margin == 2
            assert rung.decision.fits
            assert rung.decision.predicted_peak_bytes == (
                2 * rung.decision.static_compute_bytes + rung.decision.output_bytes
            )
            iterator = rung.forward_micro_batches()
            next(iterator)
            assert not any(
                getattr(p, "cp_request_floors", ())
                for p in rank._memory_profiles.values()
            )
            assert list(iterator) == []
        profile = rank._memory_profiles[rung._candidate.plan.signature]
        assert profile.cp_request_floors and profile.profile_format_version == 8
        for k, p in rank._memory_profiles.items():
            assert p.rate_observations == before.get(k, ())
        assert reports._memory_profile(asdict(profile)) == profile
        assert pickle.loads(pickle.dumps(profile)) == profile
        with pytest.raises(ValueError, match="CP family seed"):
            with rank.calibration_rung(
                [request(1, grad=grad)], no_grad=not grad, seed_uncovered_cp_family=True
            ):
                pass
        plan = rank._plan_flat_forward([request(72, grad=grad)])
        floor = memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
        assert floor >= 99_000_000
        check = rank._memory_check(plan)
        assert check.estimated_required_bytes >= floor
        assert rank._plan_cost(plan).required >= floor


def test_optimizer_guard_refuses_before_any_execution():
    from types import SimpleNamespace

    from art.trainer_rank import _impl, _optimizer

    parameter = torch.nn.Parameter(torch.empty(1000, device="meta"))
    demands = []
    rank = SimpleNamespace(
        _checkpoint_slots={
            "first": SimpleNamespace(params=(parameter,)),
            "second": SimpleNamespace(params=(parameter,)),
        },
    )

    def check(required, **kwargs):
        demands.append((required, kwargs))
        return SimpleNamespace(fits=False)

    rank._memory_check_required = check
    with pytest.raises(_impl.TrainerRankMemoryError, match="optim_step"):
        _optimizer._guard_optimizer_memory(cast(Any, rank), ("first", "second"))
    assert demands == [
        (int((1000 * 36 + 64 * 2**20) * 1.1 + 0.999999), {"sync_across_dp": True})
    ]


@pytest.mark.parametrize("fits", [False, True])
def test_optimizer_admission_precedes_reduction_and_step(fits):
    from types import SimpleNamespace

    from art.trainer_rank import _impl, _optimizer

    parameter = torch.nn.Parameter(torch.empty(1000, device="meta"))
    events = []
    rank = SimpleNamespace(
        device=torch.device("cuda"),
        _topology_key=lambda: (1, 1, 8, 1),
        _guard_forward_collective=lambda name: None,
        _guard_optim_step_configuration=lambda *args: None,
        _selected_dynamic_checkpoints=lambda names: ("selected",),
        _checkpoint_slots={"selected": SimpleNamespace(params=(parameter,))},
    )

    def check(required, **kwargs):
        events.append("admission")
        return SimpleNamespace(fits=fits)

    def execute(*args, **kwargs):
        events.append("execution")
        return {"update_successful": 1.0}

    rank._memory_check_required = check
    rank._dynamic_optim_step = execute
    if fits:
        assert _optimizer.optim_step(
            cast(Any, rank), params=_impl.AdamParams(learning_rate=1e-4)
        )
        assert events == ["admission", "execution"]
    else:
        with pytest.raises(_impl.TrainerRankMemoryError, match="optim_step"):
            _optimizer.optim_step(
                cast(Any, rank), params=_impl.AdamParams(learning_rate=1e-4)
            )
        assert events == ["admission"]
    assert parameter.grad is None


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_request_floor_isolated_monotone_and_revocation_safe(kind):
    from dataclasses import replace

    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank(kind)
        floor_plan = record_minimal_caller(rank, False, 250_000_000)
        profile = rank._memory_profiles[floor_plan.signature]
        assert profile.caller_cp_peaks == ()
        assert profile.rate_observations == ()
        assert profile.eligible_packed_tokens == profile.retained_rate_tokens == 0
        import json

        prices = {}
        for n in (1, 4, 16, 63, 64, 72, 128, 512, 2048, 2971, 4095, 4096, 8191):
            plan = rank._plan_flat_forward([request(n)])
            family = json.loads(plan.caller_cp_workload[0])
            family["execution_qualification"].pop("signature", None)
            key = json.dumps(family, sort_keys=True)
            prices.setdefault(key, []).append(
                rank._memory_check(plan).estimated_required_bytes
            )
        assert all(values == sorted(values) for values in prices.values())
        hidden = request(72)
        hidden = replace(hidden, target_tokens=None, hidden_states=True)
        token = memory._CP_FAMILY_CALIBRATION.set(True)
        try:
            other = rank._plan_flat_forward([hidden])
        finally:
            memory._CP_FAMILY_CALIBRATION.reset(token)
        assert (
            memory._cp_measured_request_floor(
                rank, other.signature, other.caller_cp_workload
            )
            == 0
        )
        plan = rank._plan_flat_forward([request(72)])
        rank._memory_profiles[plan.signature] = replace(
            profile, dominance_disabled=True
        )
        assert (
            rank._memory_check(plan).estimated_required_bytes
            >= memory._MEMORY_DEFERRED_BYTES
        )


def test_request_floor_merge_and_invalid_restore():
    from dataclasses import asdict, replace

    from art.trainer_rank import _planner_misses as reports

    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank("moe")
        plan = record_minimal_caller(rank, False, 150_000_000)
        profile = rank._memory_profiles[plan.signature]
        key = profile.cp_request_floors[0][0]
        larger = replace(profile, cp_request_floors=((key, 300_000_000),))
        merged = replace(
            profile, **memory._merge_memory_profile_model_fields(profile, larger)
        )
        assert merged.cp_request_floors == ((key, 300_000_000),)
        assert reports._memory_profile(asdict(merged)) == merged
        for changes in (
            {"profile_format_version": 7},
            {"cp_request_floors": ((key, -1),)},
            {"cp_request_floors": ((key, 1), (key, 2))},
            {"cp_request_floors": (("unknown", 1),)},
        ):
            with pytest.raises(ValueError):
                reports._memory_profile(asdict(replace(profile, **changes)))


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_minimal_trial_covers_each_output_class(monkeypatch, kind, grad):
    from test_trainer_rank_calibration_seed import fake_device
    from test_trainer_rank_cp_output_families import output_modes, output_requests

    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        rank = prepared_rank(kind)
        long_before = rank._memory_check(
            rank._plan_flat_forward([request(16384, grad=grad)])
        )
        for mode in output_modes(kind):
            fake_device(monkeypatch, rank, 100_000_000)
            inputs = output_requests(1, grad, *mode, layers=rank._num_layers)
            with rank.calibration_rung(
                inputs, no_grad=not grad, seed_uncovered_cp_family=True
            ) as rung:
                assert rung.decision.seed and rung.decision.fits
                list(rung.forward_micro_batches())
            plan = rank._plan_flat_forward(
                output_requests(72, grad, *mode, layers=rank._num_layers)
            )
            assert (
                memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
                >= 99_000_000
            )
            assert rank._memory_check(plan).estimated_required_bytes >= 99_000_000
        long_after = rank._memory_check(
            rank._plan_flat_forward([request(16384, grad=grad)])
        )
        assert long_after.fits == long_before.fits
        assert (
            long_after.estimated_required_bytes >= long_before.estimated_required_bytes
        )


@pytest.mark.parametrize("change", ["topology", "requests"])
def test_minimal_seed_requires_one_known_native_request(change):
    from dataclasses import replace

    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = prepared_rank("moe")
        plan = rank._plan_flat_forward([request(1, grad=True)])
        if change == "topology":
            rank._parallel_shape = replace(rank._parallel_shape, cp=4)
        else:
            plan = replace(plan, request_count=2)
        with pytest.raises(ValueError, match="CP family seed"):
            memory._calibration_estimate(rank, plan, seed_uncovered_cp_family=True)


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_fixed_request_floor_survives_reference_change(kind):
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank(kind)
        record_minimal_caller(rank, False, 4_000_000_000)
        before_plan = rank._plan_flat_forward([request(8191)])
        after_plan = rank._plan_flat_forward([request(8192)])
        before = rank._memory_check(before_plan).estimated_required_bytes
        after = rank._memory_check(after_plan).estimated_required_bytes
        assert after >= before
        # Both flat and split children apply the fixed component once, outside
        # the token cold multiplier. Lack of family detail retains the floor.
        assert rank._plan_cost(before_plan).required >= before
        assert rank._plan_cost(after_plan).required >= after
        assert (
            memory._cp_request_floor(rank, after_plan.signature, None) == 4_000_000_000
        )


def test_fixed_floor_does_not_prune_an_unrelated_family_split():
    from test_trainer_rank_subfloor_cp import requests, short_kernel

    from art.trainer_rank import _impl

    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank("moe")
        record_minimal_caller(rank, False, 4_000_000_000)
        with short_kernel():
            from test_trainer_rank_cp_output_families import fixture_floor

            fixture_floor(rank, False)
            inputs = requests([256, 256], False)
            children = [rank._plan_flat_forward([r]) for r in inputs]
            assert all(
                memory._cp_measured_request_floor(
                    rank, p.signature, p.caller_cp_workload
                )
                == 0
                for p in children
            )
            rank._available_memory_bytes = lambda sample=None: 300_000_000
            exact = rank._split_required_memory([rank._plan_cost(p) for p in children])
            assert exact < 300_000_000
            assert not rank._memory_check(rank._plan_flat_forward(inputs)).fits
            result = rank._find_admissible_forward(
                inputs, checkpoint=None, refusal_prefix="fixed floor"
            )
            assert not isinstance(result, _impl._ForwardRefusal)


@pytest.mark.parametrize("kind", ["moe", "dense"])
@pytest.mark.parametrize("grad", [False, True])
def test_minimum_floor_cannot_discount_existing_price(kind, grad):
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        rank = prepared_rank(kind)
        sizes = (1, 4, 16, 72, 512, 2048, 2971, 4095, 4096, 8191)
        before = [
            rank._memory_check(
                rank._plan_flat_forward([request(n, grad=grad)])
            ).estimated_required_bytes
            for n in sizes
        ]
        record_minimal_caller(rank, grad, 100_000_000)
        after = [
            rank._memory_check(
                rank._plan_flat_forward([request(n, grad=grad)])
            ).estimated_required_bytes
            for n in sizes
        ]
        assert all(a >= b for a, b in zip(after, before)), list(
            zip(sizes, before, after)
        )
