"""CPU admission for native CP8 operation/output families; peaks are synthetic."""

from dataclasses import replace

import pytest
from test_trainer_rank_calibration_seed import metadata_rank
from test_trainer_rank_calibration_trust_region import observe, request
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank import _memory as memory


def test_unprofiled_forward_target_2971_after_backward_seed_is_finite():
    rank = metadata_rank("moe")
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    with memory.fixed_slope_memory_profiles():
        observe(rank, 8192, 1_300_000_000, grad=True)
        observe(rank, 16384, 2_200_000_000, grad=True)
        fixture_floor(rank, True)
        with torch.no_grad():
            plan = rank._plan_flat_forward([request(2971)])
            check = rank._memory_check(plan)
        assert check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES
        assert check.fits
        from art.trainer_rank import _impl

        with torch.no_grad():
            selected = rank._select_next_micro_batch(
                tuple(_impl._materialize(r) for r in [request(2971)]),
                0,
                checkpoint=None,
            )
        assert selected.check.fits


def native_rank(kind):
    """Native checkpoint/owner metadata with the model's hybrid CP planner."""
    from art.megatron.model_support.handlers.qwen3_5 import (
        Qwen35DenseHandler,
        Qwen35MoeHandler,
    )
    from art.trainer_rank._planner_cost import ModelGeometry

    rank = metadata_rank(kind)
    cfg = rank.runtime.provider
    cfg.ffn_hidden_size = 512 if kind == "moe" else 17408
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 32 if kind == "moe" else 48
    cfg.linear_key_head_dim = cfg.linear_value_head_dim = 128
    cfg.linear_conv_kernel_dim = 4
    rank._geometry = ModelGeometry.from_config(cfg, has_gdn=True, has_moe=kind == "moe")
    rank._gdn_layers = rank._num_layers * 3 // 4
    rank._mixer_top_gaps = (0, 1)
    rank.runtime.model_support_handler = (
        Qwen35MoeHandler() if kind == "moe" else Qwen35DenseHandler()
    )
    if kind == "moe":
        router = rank.runtime.model[0].decoder.layers[0].router
        router.config = cfg
        rank._routing_bindings = [(router, i) for i in range(rank._num_layers)]
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    return rank


def fixture_floor(rank, grad):
    """Restore a completed synthetic minimum reading, with no rate/coverage data."""
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        token = memory._CP_FAMILY_CALIBRATION.set(True)
        try:
            plan = rank._plan_flat_forward(
                output_requests(1, grad, "none", True, True, 2, False)
            )
        finally:
            memory._CP_FAMILY_CALIBRATION.reset(token)
        plan.__dict__["_calibration_cp_family_seed"] = True
        rank._update_memory_profile(
            plan,
            1_000_000 + plan.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )


def anchors(rank, grad, *, include_floor=True):
    with torch.set_grad_enabled(grad):
        observe(rank, 8192, 10_000_000, grad=grad)
        observe(rank, 16384, 20_000_000, grad=grad)
    if include_floor:
        fixture_floor(rank, grad)


def test_local_measurement_at_2048_does_not_cover_gradient_target_2971(monkeypatch):
    from test_trainer_rank_calibration_seed import fake_device

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        anchors(rank, True)
        fake_device(monkeypatch, rank, 200_000 + 4 * 2048)
        with rank.calibration_rung(
            [request(2048, grad=True)], no_grad=False, seed_uncovered_cp_family=True
        ) as rung:
            assert rung.decision.seed and rung.decision.fits
            list(rung.forward_micro_batches())
        plan = rank._plan_flat_forward([request(2971, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        assert (
            profile.covering_cp_peak(plan.packed_tokens, plan.caller_cp_workload)
            is None
        )
        check = rank._memory_check(plan)
        assert check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES
        assert check.fits
        from art.trainer_rank import _impl

        selected = rank._select_next_micro_batch(
            tuple(_impl._materialize(r) for r in [request(2971, grad=True)]),
            0,
            checkpoint=None,
        )
        assert selected.check.fits


def output_requests(
    n, grad, target, hidden, logits, topk, routes, *, mixed=False, layers=40
):
    """Generators for every _request_mix_key output component and its unions."""
    targets = (
        None
        if target == "none"
        else torch.arange(n)
        if target == "single"
        else torch.zeros((n, 1 if target == "column" else 3), dtype=torch.int64)
    )
    request = ForwardInput(
        input_tokens=torch.arange(n) % 1024,
        target_tokens=targets,
        hidden_states=hidden,
        logits=logits,
        top_k=topk,
        routed_experts=torch.arange(8, dtype=torch.int64)
        .reshape(1, 1, 8)
        .expand(n, layers, 8)
        if routes
        else None,
        no_grad=not grad,
    )
    return [request, replace(request, routed_experts=None)] if mixed else [request]


def output_modes(kind="moe"):
    import itertools

    return tuple(
        values
        for values in itertools.product(
            ("none", "single", "column", "multiple"),
            (False, True),
            (False, True),
            (None, 1, 7),
            (False, True),
        )
        if (values[0] != "none" or any(values[1:4]))
        and (kind == "moe" or not values[-1])
    )


def static_seed_bound(rank, plan):
    from art.trainer_rank import _impl

    estimate = memory._CalibrationEstimate(
        signature_id="",
        candidate=plan.packed_tokens,
        seed=True,
        seed_uncovered_cp_family=True,
    )
    return rank._estimate_required_memory_bytes_from_values(
        packed_tokens=plan.packed_tokens,
        request_count=plan.request_count,
        output_bytes=plan.output_bytes,
        signature=plan.signature,
        logical_tokens=plan.active_logical_tokens,
        gdn_segments=plan.grad_segment_count,
        group_rows=rank._plan_group_rows(plan),
        slot_refs=tuple(g.slot_ref for g in plan.groups),
        head_workspace_bytes=rank._plan_head_workspace_bytes(plan),
        checkpoint_floor=_impl._gdn_memory.plan_floor(rank, plan),
        retained_tokens=rank._plan_retained_tokens(plan),
        caller_cp_workload=plan.caller_cp_workload,
        _calibration_compute_bytes=estimate,
    )


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_every_output_mode_has_a_finite_short_cold_bound(kind, grad):
    rank = native_rank(kind)
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        anchors(rank, grad)
        for mode in output_modes(kind):
            for n in (72, 512, 2971, 4095, 4096, 8191):
                inputs = output_requests(n, grad, *mode, layers=rank._num_layers)
                plan = rank._plan_flat_forward(inputs)
                check = rank._memory_check(plan)
                bound = static_seed_bound(rank, plan)
                assert check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES, (
                    kind,
                    grad,
                    mode,
                    n,
                )
                assert check.estimated_required_bytes <= bound, (kind, grad, mode, n)


@pytest.mark.parametrize("grad", [False, True])
def test_every_output_mode_can_measure_local_and_chained_families(monkeypatch, grad):
    from test_trainer_rank_calibration_seed import fake_device

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        anchors(rank, grad)
        for mode in output_modes():
            for n in (4095, 8191):
                inputs = output_requests(n, grad, *mode, layers=rank._num_layers)
                plan = rank._plan_flat_forward(inputs)
                profile = rank._memory_profiles.get(plan.signature)
                covering = getattr(profile, "covering_cp_peak", None)
                if (
                    callable(covering)
                    and covering(plan.packed_tokens, plan.caller_cp_workload)
                    is not None
                ):
                    continue  # The product's covered-family preparation is a no-op.
                fake_device(monkeypatch, rank, 200_000 + plan.output_bytes)
                with rank.calibration_rung(
                    inputs, no_grad=not grad, seed_uncovered_cp_family=True
                ) as rung:
                    plan = rung._candidate.plan
                    profile = rank._memory_profiles.get(plan.signature)
                    rates = () if profile is None else profile.rate_observations
                    d = rung.decision
                    assert d.seed and d.fits and d.seed_margin == 2
                    assert (
                        d.predicted_peak_bytes
                        == 2 * d.static_compute_bytes + d.output_bytes
                    )
                    list(rung.forward_micro_batches())
                measured = rank._memory_profiles[plan.signature]
                assert measured.rate_observations == rates
                with rank.calibration_rung(inputs, no_grad=not grad) as ordinary:
                    assert ordinary.decision.eligible and ordinary.decision.fits
                    assert not ordinary.decision.extrapolated
                assert measured.covering_cp_peak(n, plan.caller_cp_workload) is not None
                assert not memory._uncovered_cp_family(
                    rank,
                    measured,
                    plan.signature,
                    n,
                    plan.caller_cp_workload,
                    request_count=1,
                )
                assert (
                    rank._memory_check(plan).estimated_required_bytes
                    < memory._MEMORY_DEFERRED_BYTES
                )
                with pytest.raises(ValueError, match="CP family seed"):
                    with rank.calibration_rung(
                        inputs, no_grad=not grad, seed_uncovered_cp_family=True
                    ):
                        pass


def test_measured_forward_family_survives_codec_and_scope_exit(monkeypatch):
    from dataclasses import asdict

    from test_trainer_rank_calibration_seed import fake_device

    from art.trainer_rank import _planner_misses as reports

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        anchors(rank, False)
        inputs = output_requests(4095, False, "none", True, False, None, False)
        plan = rank._plan_flat_forward(inputs)
        fake_device(monkeypatch, rank, 200_000 + plan.output_bytes)
        with rank.calibration_rung(
            inputs, no_grad=True, seed_uncovered_cp_family=True
        ) as rung:
            list(rung.forward_micro_batches())
        plan = rank._plan_flat_forward(inputs)
        key = plan.signature
        saved = rank._memory_profiles[key]
        rank._memory_profiles[key] = reports._memory_profile(asdict(saved))
        assert rank._memory_profiles[key] == saved
        before = rank._memory_check(rank._plan_flat_forward(inputs))
    with torch.no_grad():
        after = rank._plan_flat_forward(inputs)
        assert after.signature == key
        assert rank._memory_check(after) == before


@pytest.mark.parametrize("grad", [False, True])
def test_revoked_and_unknown_kernel_families_remain_refused(monkeypatch, grad):
    import json

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        anchors(rank, grad)
        if not grad:
            from test_trainer_rank_calibration_seed import fake_device

            fake_device(monkeypatch, rank, 200_000 + 4 * 2048)
            with rank.calibration_rung(
                [request(2048)], no_grad=True, seed_uncovered_cp_family=True
            ) as rung:
                list(rung.forward_micro_batches())
        plan = rank._plan_flat_forward([request(2971, grad=grad)])
        key = plan.signature
        original = rank._memory_profiles[key]
        family, loads = plan.caller_cp_workload
        value = json.loads(family)
        value["execution_qualification"]["kernels"] = ["unknown"]
        unknown = replace(plan, caller_cp_workload=(json.dumps(value), loads))
        assert (
            rank._memory_check(unknown).estimated_required_bytes
            >= memory._MEMORY_DEFERRED_BYTES
        )
        pair = memory._cp_pair(family, family)
        for change in (
            {"non_dominant_cp_pairs": (pair,)},
            {"dominance_disabled": True},
        ):
            rank._memory_profiles[key] = replace(original, **change)
            check = rank._memory_check(plan)
            assert (
                not check.fits
                and check.estimated_required_bytes >= memory._MEMORY_DEFERRED_BYTES
            )


def test_forward_evidence_cannot_license_an_unmeasured_backward_family():
    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        plan = rank._plan_flat_forward([request(2971, grad=True)])
        before = rank._memory_check(plan).estimated_required_bytes
        anchors(rank, False)
        after = rank._memory_check(plan).estimated_required_bytes
        assert after >= before >= memory._MEMORY_DEFERRED_BYTES
