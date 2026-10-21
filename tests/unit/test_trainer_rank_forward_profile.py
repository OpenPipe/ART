"""Cold fixed scratch must not poison known single-root hidden-only inference."""

from contextlib import ExitStack
from dataclasses import fields, replace

import pytest
from test_trainer_rank_checkpoint_memory import dense_rank
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
import torch

from art.trainer_rank import ForwardInput, _impl, _memory, bounded_forward_peaks


@pytest.fixture
def bounded_rank(pending_rank):
    with bounded_forward_peaks():
        yield pending_rank


def request(tokens):
    return ForwardInput(
        input_tokens=torch.full((tokens,), 198, dtype=torch.long),
        hidden_states=True,
        no_grad=True,
    )


def observe(rank, plan, compute):
    rank._update_memory_profile(
        plan, plan.output_bytes + compute, retained_bytes=plan.output_bytes
    )


@pytest.mark.parametrize("cp", (2, 4, 8))
@pytest.mark.parametrize("grad", (False, True))
@pytest.mark.parametrize("kind", ("dense", "moe"))
def test_native_cp_bounded_calibration_keeps_physical_workspace_and_full_outputs(
    bounded_rank, cp, grad, kind
):
    rank = dense_rank() if kind == "dense" else bounded_rank
    inputs = [replace(request(128), no_grad=not grad)]
    with bounded_forward_peaks(), _memory.direct_forward_profile():
        control = rank._plan_flat_forward(inputs)
        rank._topology_key = lambda: (1, 1, cp, 1)
        rank._parallel_shape = replace(rank._parallel_shape, cp=cp)
        signature = rank._memory_signature_from_requests(
            inputs, slot_group_count=1, grad_modes=(grad,)
        )
    assert signature.bounded_forward_peak and signature != control.signature
    observe(rank, replace(control, signature=signature), 470221312)
    profile = rank._memory_profiles[signature]
    rows = 32768 // cp + 128  # Uneven ownership must use actual rows.
    arguments = dict(
        packed_tokens=32768,
        logical_tokens=32768,
        output_bytes=32768 * rank._hidden_size * 2,
        signature=signature,
        group_rows=((rows, grad),),
    )
    cold = rank._estimate_required_memory_bytes_from_values(**arguments)
    rank._memory_profiles[signature] = replace(
        profile, forward_compute_peaks=((8192, 470221312),)
    )
    calibrated = rank._estimate_required_memory_bytes_from_values(**arguments)
    assert calibrated < cold
    retained, workspace = rank._checkpoint_memory_floor(((rows, grad),))
    if grad:
        assert retained == rows * rank._num_layers * rank._hidden_size * 2
        if kind == "dense":
            assert workspace >= rank._sequence_parallel_workspace_bytes(rows)
            assert rank._checkpoint_input_gradient_bytes(((rows, grad),), retained) == (
                rows * rank._hidden_size * 2
            )
    assert calibrated >= int(
        (retained + workspace + 2 * arguments["output_bytes"]) * 1.1
    )
    rank._memory_profiles[signature] = replace(
        profile, forward_compute_peaks=((8192, 0),)
    )
    assert calibrated >= rank._estimate_required_memory_bytes_from_values(**arguments)


@pytest.mark.parametrize("cp,actual", ((2, 1), (3, 3), (16, 16), (4, 8)))
def test_dense_bounded_cp_requires_supported_matching_runtime_geometry(cp, actual):
    rank = dense_rank()
    rank._topology_key = lambda: (1, 1, cp, 1)
    rank._parallel_shape = replace(rank._parallel_shape, cp=actual)
    with bounded_forward_peaks():
        signature = rank._memory_signature_from_requests(
            [request(128)], slot_group_count=1, grad_modes=(False,)
        )
    assert not isinstance(signature, _memory._ScopedMemorySignature)


def test_bounded_dense_cp_no_grad_charges_the_worst_rank_fc1_stage():
    rank = dense_rank()
    rank._topology_key = lambda: (1, 1, 8, 1)
    rank._parallel_shape = replace(rank._parallel_shape, cp=8)
    with bounded_forward_peaks():
        signature = rank._memory_signature_from_requests(
            [request(128)], slot_group_count=1, grad_modes=(False,)
        )
    required = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=32768,
        output_bytes=0,
        signature=signature,
        group_rows=((32768, False),),
    )
    assert required >= int(32768 * (6 * 17408 + 4 * 5120) * 2 * 1.1)


def test_three_token_cold_peak_refuses_uncalibrated_81920(bounded_rank, monkeypatch):
    rank = bounded_rank
    short = rank._plan_flat_forward([request(3)])
    long = rank._plan_flat_forward([request(81920)])
    assert short.signature == long.signature
    assert short.signature.bounded_forward_peak
    cold_required = rank._memory_check(long).estimated_required_bytes
    # The real H200 three-token reproduction observed this compute delta.
    fixed = 470221312
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 63 * 2**30)
    observe(rank, short, fixed)
    profile = rank._memory_profiles[short.signature]
    assert profile.caller_plans == 0 and profile.warm_bytes_per_token is None
    assert profile.forward_compute_peaks == ((3, fixed),)
    assert profile.bytes_per_token * 81920 > 10_000 * 2**30
    # A lone tiny observation cannot bound an unseen long root safely.
    assert rank._memory_check(long).estimated_required_bytes > 63 * 2**30
    assert not rank._memory_check(long).fits
    # Calibrate progressively as the growing estimates permit. Cold scratch
    # no longer poisons the slope after longer roots are observed.
    for tokens in (24, 192, 1536, 8192):
        plan = rank._plan_flat_forward([request(tokens)])
        assert rank._memory_check(plan).fits
        observe(rank, plan, fixed)
    assert rank._memory_check(long).fits
    assert rank._memory_check(long).estimated_required_bytes == cold_required
    assert cold_required < 63 * 2**30


def test_observed_peak_still_refuses_insufficient_memory(bounded_rank, monkeypatch):
    rank = bounded_rank
    short = rank._plan_flat_forward([request(3)])
    fixed = 470221312
    observe(rank, short, fixed)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: fixed - 1)
    for tokens in (3, 24, 25, 81920):
        check = rank._memory_check(rank._plan_flat_forward([request(tokens)]))
        assert not check.fits
        assert check.estimated_required_bytes >= fixed


def test_costs_are_monotone_and_static_floors_survive(bounded_rank):
    rank = bounded_rank
    short = rank._plan_flat_forward([request(3)])
    observe(rank, short, 470221312)
    observe(rank, rank._plan_flat_forward([request(8192)]), 470221312)
    required = []
    for tokens in (1, 2, 3, 4, 8, 23, 24, 25, 128, 8192, 81920):
        plan = rank._plan_flat_forward([request(tokens)])
        profiled = rank._memory_check(plan).estimated_required_bytes
        saved, rank._memory_profiles = rank._memory_profiles, {}
        unprofiled = rank._memory_check(plan).estimated_required_bytes
        rank._memory_profiles = saved
        assert profiled >= unprofiled
        required.append(profiled)
    assert required == sorted(required)
    long = rank._plan_flat_forward([request(81920)])
    observe(rank, long, 32 * 2**30)
    prior = rank._memory_check(long).estimated_required_bytes
    observe(rank, short, 1)
    assert (
        rank._memory_profiles[long.signature].forward_compute_peaks[-1][1] == 32 * 2**30
    )
    assert rank._memory_check(long).estimated_required_bytes >= prior


def test_direct_gradient_profile_keeps_static_and_retained_graph_floors(bounded_rank):
    rank = bounded_rank
    inputs = [replace(request(3), no_grad=False)]
    yielded = rank._plan_flat_forward(inputs)
    assert type(yielded.signature) is _impl._MemorySignature
    with _memory.direct_forward_profile():
        short = rank._plan_flat_forward(inputs)
        long = rank._plan_flat_forward([replace(request(81920), no_grad=False)])
    assert short.signature == long.signature and short.signature.bounded_forward_peak
    assert short.signature != yielded.signature
    # Existing policy drops a one-time recompute allowance once profiled;
    # compare the token-dependent floors with that same observation state.
    observe(rank, short, 0)
    static = rank._memory_check(long).estimated_required_bytes
    rank._memory_profiles.clear()
    observe(rank, long, 4058112)
    rank._update_memory_profile(
        short,
        short.output_bytes + 4058112,
        retained_bytes=short.output_bytes + 516608,
    )
    required = rank._memory_check(long).estimated_required_bytes
    assert static <= required < 63 * 2**30
    assert (
        rank._plan_cost(long).retained
        >= rank._checkpoint_memory_floor(((81920, True),))[0]
    )
    assert rank._memory_profiles[short.signature].caller_plans == 0
    assert rank._plan_flat_forward(inputs).signature == yielded.signature


def test_direct_scope_resets_after_nested_scope_and_exception(bounded_rank):
    rank = bounded_rank
    inputs = [replace(request(3), no_grad=False)]
    with pytest.raises(ValueError, match="scope"):
        with _memory.direct_forward_profile():
            with _memory.direct_forward_profile():
                assert rank._plan_flat_forward(inputs).signature.bounded_forward_peak
            assert rank._plan_flat_forward(inputs).signature.bounded_forward_peak
            raise ValueError("scope")
    assert type(rank._plan_flat_forward(inputs).signature) is _impl._MemorySignature


@pytest.mark.parametrize(
    "mode",
    ["grad", "target", "logits", "topk", "multiple", "matrix", "cp", "unsupported"],
)
def test_other_requests_keep_existing_profile_growth(bounded_rank, mode):
    rank = bounded_rank
    observed = rank._plan_flat_forward([request(3)])
    value = request(3)
    if mode == "grad":
        value = replace(value, no_grad=False)
    elif mode == "target":
        value = replace(value, target_tokens=torch.ones(3, dtype=torch.long))
    elif mode == "logits":
        value = replace(value, logits=True)
    elif mode == "topk":
        value = replace(value, top_k=1)
    elif mode == "matrix":
        value = replace(value, input_tokens=value.input_tokens.reshape(1, 3))
    elif mode == "cp":
        rank._topology_key = lambda: (1, 1, 2, 1)
    elif mode == "unsupported":
        rank.runtime.model[0].decoder.config.recompute_num_layers = 2
    signature = rank._memory_signature_from_requests(
        [value, value] if mode == "multiple" else [value],
        slot_group_count=1,
        grad_modes=(mode == "grad",),
    )
    assert not isinstance(signature, _memory._ScopedMemorySignature)
    observe(rank, replace(observed, signature=signature), 470221312)
    assert type(rank._memory_profiles[signature]) is _impl._MemoryProfile
    required = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=81920,
        output_bytes=0,
        signature=signature,
        group_rows=((81920, mode == "grad"),),
    )
    assert required > 10_000 * 2**30


@pytest.mark.parametrize("grad", [False, True])
def test_default_signature_profile_and_estimate_match_main(pending_rank, grad):
    rank = pending_rank
    inputs = [replace(request(3), no_grad=not grad)]
    # Main's field schema and request identity, pinned at e18c570.
    expected = _impl._MemorySignature(
        topology=rank._topology_key(),
        planner_coefficients=(rank._coefficient_version, rank._coefficient_table),
        slot_group_count=1,
        request_mix=("hidden",),
        grad_enabled=grad,
        grad_modes=(grad,),
    )
    plan = rank._plan_flat_forward(inputs)
    assert type(plan.signature) is _impl._MemorySignature
    assert plan.signature == expected
    assert tuple(f.name for f in fields(plan.signature)) == (
        "topology",
        "planner_coefficients",
        "slot_group_count",
        "request_mix",
        "grad_enabled",
        "grad_modes",
        "slot_shapes",
        "short_requests",
    )
    # Default and opted-in observations never share profiles. The original
    # all-observations slope still prices this unseen long root on main.
    observe(rank, plan, 470221312)
    profile = rank._memory_profiles[plan.signature]
    assert type(profile) is _impl._MemoryProfile
    assert "forward_compute_peaks" not in vars(profile)
    long = rank._plan_flat_forward([replace(request(81920), no_grad=not grad)])
    assert long.signature == plan.signature
    expected_required = int((long.output_bytes + int(470221312 / 3 * 81920)) * 1.1)
    assert rank._memory_check(long).estimated_required_bytes == expected_required
    with bounded_forward_peaks(), _memory.direct_forward_profile():
        scoped = rank._plan_flat_forward(inputs)
        assert scoped.signature != plan.signature
        assert scoped.signature not in rank._memory_profiles
    assert rank._plan_flat_forward(inputs).signature == expected


def test_cap_grows_beyond_longest_observed_root(bounded_rank):
    rank = bounded_rank
    observed = rank._plan_flat_forward([request(8192)])
    peak = 2**40  # Dominate the analytic floors and outputs in this witness.
    observe(rank, observed, peak)
    # One observation grows proportionally until a second root fits a rate.
    for tokens, growth in ((4096, 1), (8192, 1), (16384, 2), (32768, 4), (65536, 8)):
        plan = rank._plan_flat_forward([request(tokens)])
        assert rank._memory_check(plan).estimated_required_bytes == int(
            (plan.output_bytes + peak * growth) * 1.1
        )


@pytest.mark.parametrize("grad", [False, True])
def test_fixed_intercept_admits_within_horizon_and_keeps_steepest_rate(
    bounded_rank, monkeypatch, grad
):
    rank = bounded_rank
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 63 * 2**30)
    with _memory.direct_forward_profile():
        short, medium, long = [
            rank._plan_flat_forward([replace(request(tokens), no_grad=not grad)])
            for tokens in (128, 1024, 4096)
        ]
    observe(rank, short, 10 * 2**30)
    observe(rank, medium, 21 * 2**29)
    slope = 2**29 / (1024 - 128)
    compute = 21 * 2**29 * 1.75
    check = rank._memory_check(long)
    assert check.fits
    assert check.estimated_required_bytes == int(
        (long.output_bytes + int(compute)) * 1.1
    )
    # A flatter later interval must not replace the steepest qualifying pair.
    with _memory.direct_forward_profile():
        later = rank._plan_flat_forward([replace(request(8192), no_grad=not grad)])
        long = rank._plan_flat_forward([replace(request(32768), no_grad=not grad)])
    observe(rank, later, 11 * 2**30)
    compute = 11 * 2**30 + slope * (32768 - 8192)
    assert rank._memory_check(long).estimated_required_bytes == int(
        (long.output_bytes + int(compute)) * 1.1
    )


@pytest.mark.parametrize("grad", [False, True])
def test_fixed_intercept_ladder_admits_each_step_within_horizon(
    bounded_rank, monkeypatch, grad
):
    rank = bounded_rank
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 63 * 2**30)
    with _memory.direct_forward_profile():
        for tokens in (128, 512, 2048, 8192, 32768, 81920):
            plan = rank._plan_flat_forward([replace(request(tokens), no_grad=not grad)])
            check = rank._memory_check(plan)
            assert check.fits
            compute = 10 * 2**30 + (tokens - 128) * 2**29 // (1024 - 128)
            if tokens > 128:
                assert check.estimated_required_bytes >= plan.output_bytes + compute
            observe(rank, plan, compute)


def test_close_observations_grow_proportionally_beyond_horizon(
    bounded_rank, monkeypatch
):
    rank = bounded_rank
    for tokens, peak in ((128, 10 * 2**30), (200, 12 * 2**30)):
        observe(rank, rank._plan_flat_forward([request(tokens)]), peak)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 63 * 2**30)
    long = rank._plan_flat_forward([request(8192)])
    check = rank._memory_check(long)
    assert not check.fits
    assert check.estimated_required_bytes == int(
        (long.output_bytes + int(12 * 2**30 * 8192 / 200)) * 1.1
    )


def test_marginal_growth_can_refuse_more_than_origin_fit(bounded_rank, monkeypatch):
    rank = bounded_rank
    for tokens, peak in ((128, 10 * 2**30), (256, 30 * 2**30)):
        observe(rank, rank._plan_flat_forward([request(tokens)]), peak)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 70 * 2**30)
    # The measured slope predicts 70 GiB, while the origin fit predicts 60.
    for tokens, gib in ((512, 70), (1024, 150), (2048, 300)):
        long = rank._plan_flat_forward([request(tokens)])
        check = rank._memory_check(long)
        assert not check.fits
        assert check.estimated_required_bytes == int(
            (long.output_bytes + gib * 2**30) * 1.1
        )


def test_observed_range_uses_smallest_covering_peak(bounded_rank):
    rank = bounded_rank
    for tokens, peak in ((8192, 12 * 2**30), (128, 10 * 2**30), (1024, 21 * 2**29)):
        observe(rank, rank._plan_flat_forward([request(tokens)]), peak)
    for tokens, peak in (
        (128, 10 * 2**30),
        (129, 21 * 2**29),
        (1024, 21 * 2**29),
        (1025, 12 * 2**30),
        (8192, 12 * 2**30),
    ):
        plan = rank._plan_flat_forward([request(tokens)])
        assert rank._memory_check(plan).estimated_required_bytes == int(
            (plan.output_bytes + peak) * 1.1
        )


def test_observations_stay_bounded_and_keep_peak_envelope(bounded_rank):
    rank = bounded_rank
    for exponent in range(13):
        plan = rank._plan_flat_forward([request(2**exponent)])
        observe(rank, plan, (12 if exponent == 4 else 10) * 2**30)
    profile = rank._memory_profiles[plan.signature]
    assert len(profile.forward_compute_peaks) == 8
    assert profile.forward_compute_peaks[0] == (1, 10 * 2**30)
    assert profile.forward_compute_peaks[-1] == (4096, 12 * 2**30)
    # Evicting the intermediate high peak cannot lower the covering envelope.
    for tokens in (32, 64, 4096):
        covered = rank._plan_flat_forward([request(tokens)])
        assert rank._memory_check(covered).estimated_required_bytes == int(
            (covered.output_bytes + 12 * 2**30) * 1.1
        )
    observe(rank, plan, 1)
    assert (
        rank._memory_profiles[plan.signature].forward_compute_peaks
        == profile.forward_compute_peaks
    )


@pytest.mark.parametrize("later_peak", [2**40, 2**39])
def test_plateau_grows_then_becomes_proportional(bounded_rank, later_peak):
    rank = bounded_rank
    observe(rank, rank._plan_flat_forward([request(128)]), 2**40)
    observe(rank, rank._plan_flat_forward([request(256)]), later_peak)
    profile = rank._memory_profiles[rank._plan_flat_forward([request(256)]).signature]
    assert profile.compute_peak(512) > profile.compute_peak(256)
    # The horizon joins continuously to proportional growth at longer roots.
    assert profile.compute_peak(1025) == profile.compute_peak(1024) * 1025 / 1024
    assert profile.compute_peak(81920) == 2 * profile.compute_peak(40960)
    for tokens, growth in ((256, 1), (512, 1.25), (1024, 1.75), (81920, 140)):
        plan = rank._plan_flat_forward([request(tokens)])
        assert rank._memory_check(plan).estimated_required_bytes == int(
            (plan.output_bytes + 2**40 * growth) * 1.1
        )


def test_public_scope_resets_after_nested_exception(pending_rank):
    rank = pending_rank
    default = rank._plan_flat_forward([request(3)]).signature
    with pytest.raises(ValueError, match="scope"):
        with bounded_forward_peaks():
            with bounded_forward_peaks():
                assert rank._plan_flat_forward([request(3)]).signature != default
            assert _memory._BOUNDED_FORWARD_PEAKS.get()
            raise ValueError("scope")
    assert not _memory._BOUNDED_FORWARD_PEAKS.get()
    assert rank._plan_flat_forward([request(3)]).signature == default


@pytest.mark.parametrize("enabled", [False, True])
def test_dp_rank_forward_enters_direct_profile_only_when_opted_in(
    pending_rank, monkeypatch, enabled
):
    from contextlib import nullcontext

    rank = pending_rank
    calls = []
    original = _memory.direct_forward_profile

    def direct():
        calls.append(True)
        return original()

    monkeypatch.setattr(_memory, "direct_forward_profile", direct)
    monkeypatch.setattr(rank, "_guard_forward_collective", lambda *a: None)
    monkeypatch.setattr(rank, "_backward_work", lambda: None)
    monkeypatch.setattr(rank, "_plan_admissible_forward", lambda *a, **k: (None, None))
    monkeypatch.setattr(rank, "_execute_admitted_plan", lambda *a, **k: [object()])
    with bounded_forward_peaks() if enabled else nullcontext():
        rank.dp_rank_forward([request(3)])
    assert calls == ([True] if enabled else [])


@pytest.mark.parametrize("cp", (2, 4, 8))
@pytest.mark.parametrize("grad", (False, True))
def test_native_moe_cp_bounded_profiles_keep_static_floors(bounded_rank, cp, grad):
    rank = bounded_rank
    inputs = [replace(request(128), no_grad=not grad)]
    with _memory.direct_forward_profile():
        control = rank._plan_flat_forward(inputs)
        rank._topology_key = lambda: (1, 1, cp, 1)
        rank._parallel_shape = replace(rank._parallel_shape, cp=cp)
        signature = rank._memory_signature_from_requests(
            inputs, slot_group_count=1, grad_modes=(grad,)
        )
    assert signature.bounded_forward_peak
    assert signature.topology == (1, 1, cp, 1)
    assert signature != control.signature
    # Fixed scratch is an intercept. Preserve the existing physical-row and
    # full-output floors; the calibration never divides a CP1 prediction.
    observe(rank, replace(control, signature=signature), 470221312)
    profile = rank._memory_profiles[signature]
    assert profile.forward_compute_peaks == ((128, 470221312),)
    arguments = dict(
        packed_tokens=32768,
        logical_tokens=32768,
        output_bytes=32768 * 2048 * 2,
        signature=signature,
        group_rows=((32768 // cp, grad),),
    )
    required = rank._estimate_required_memory_bytes_from_values(**arguments)
    rank._memory_profiles[signature] = replace(
        profile, forward_compute_peaks=((8192, 470221312),)
    )
    calibrated = rank._estimate_required_memory_bytes_from_values(**arguments)
    assert calibrated < required
    rank._memory_profiles[signature] = replace(
        profile, forward_compute_peaks=((8192, 0),)
    )
    floor = rank._estimate_required_memory_bytes_from_values(**arguments)
    assert calibrated >= floor
    rank._memory_profiles.clear()
    assert signature.topology == (1, 1, cp, 1)


# Values from origin/main e720a9e9, before bounded CP2/4/8 calibration, for one
# 32,768-token hidden root: cold required, required after one 128-token
# observation, checkpoint retained and workspace bytes, and the input-gradient
# term. The CP extension must leave every CP1 prediction where it was.
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize(
    "kind,grad,expected",
    (
        ("dense", False, (9375108300, 132783420211, 0, 1342177280, 0)),
        ("dense", True, (47687558758, 132783420211, 21474836480, 0, 21474836480)),
        ("moe", False, (7529614540, 132561960960, 0, 6710886400, 0)),
        ("moe", True, (21066437427, 132561960960, 5368709120, 6178209792, 5368709120)),
    ),
)
def test_cp1_predictions_are_unchanged_by_bounded_cp(
    pending_rank, kind, grad, expected, scoped
):
    rank = dense_rank() if kind == "dense" else pending_rank
    assert rank._topology_key()[2] == rank._parallel_shape.cp == 1

    def root(tokens):
        ids = torch.arange(tokens) % 1900 + 100
        return ForwardInput(input_tokens=ids, hidden_states=True, no_grad=not grad)

    with ExitStack() as stack:
        if scoped:
            stack.enter_context(bounded_forward_peaks())
            stack.enter_context(_memory.direct_forward_profile())
        long, short = (rank._plan_flat_forward([root(n)]) for n in (32768, 128))
        assert getattr(long.signature, "bounded_forward_peak", False) is scoped
        rows = rank._plan_group_rows(long)
        retained, workspace = rank._checkpoint_memory_floor(rows)
        cold = rank._memory_check(long).estimated_required_bytes
        observe(rank, short, 470221312)
        warm = rank._memory_check(long).estimated_required_bytes
        gradient = rank._checkpoint_input_gradient_bytes(rows, retained)
    assert (cold, warm, retained, workspace, gradient) == expected
