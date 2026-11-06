"""CPU learning contract; no claim about unobserved CUDA workspace."""

from dataclasses import asdict, replace
import pickle
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank import _memory as m


def signature():
    return m._ScopedMemorySignature(
        topology=(1, 1, 1, 1),
        planner_coefficients=(2, "test"),
        slot_group_count=1,
        request_mix=("target:single",),
        grad_enabled=True,
        grad_modes=(True,),
        bounded_caller_peak=True,
    )


def classify(rank, requests=None, *, mode=True, groups=1):
    if requests is None:
        requests = [
            ForwardInput(
                input_tokens=torch.arange(128), target_tokens=torch.arange(128)
            )
        ]
    return m._memory_signature_from_requests(
        cast(Any, rank), requests, slot_group_count=groups, grad_modes=(mode,)
    )


@pytest.fixture(params=(1, 2, 4, 8), ids=lambda cp: f"cp{cp}")
def caller_rank(request, monkeypatch):
    cp = request.param
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.context_parallel.checkpoint",
        SimpleNamespace(current_checkpoint_partition=lambda: None),
    )
    monkeypatch.setattr(m, "_checkpoint_layers", lambda *args: 40)
    rank = memory_rank()
    rank._topology_key = lambda: (1, 1, cp, 1)
    rank._coefficient_version = 2
    rank._coefficient_table = "test"
    rank._slot_memory_shapes = lambda _: ()
    rank._parallel_shape = SimpleNamespace(ep=1, etp=1, cp=cp)
    rank._moe_layers = 40
    rank._moe_memory_supported = True
    rank._dense_fc1_adapted = False
    return rank


@pytest.fixture
def caller_signature(caller_rank):
    sig = classify(caller_rank)
    assert isinstance(sig, m._ScopedMemorySignature) and sig.bounded_caller_peak
    assert not sig.bounded_forward_peak
    return sig


def observe(rank, sig, rows, peak, caller=True):
    plan = cast(
        Any,
        SimpleNamespace(
            packed_tokens=rows,
            active_logical_tokens=rows,
            output_bytes=0,
            signature=sig,
        ),
    )
    m._update_memory_profile(rank, plan, peak, retained_bytes=None, caller_phase=caller)


def test_caller_seed_and_reload_transients_remain_bounded(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 128, 1_075_000_000, caller=False)
    assert r._memory_profiles[sig].caller_compute_peaks == ()
    observe(r, sig, 128, 1_075_000_000)
    for rows in (2048, 8192, 32768):
        observe(r, sig, rows, 1_036_600_000 + 300_000 * rows)
    p = r._memory_profiles[sig]
    assert p.bytes_per_token * 32768 > 200_000_000_000
    assert p.compute_peak(32768) < 12_000_000_000
    # Reload/first-use transient at a short root is kept as a fixed floor.
    observe(r, sig, 128, 12_000_000_000)
    p = r._memory_profiles[sig]
    assert p.compute_peak(32768) == 12_000_000_000
    assert p.compute_peak(131072) >= p.compute_peak(32768)
    assert p.compute_peak(131073) >= p.compute_peak(131072)


def test_higher_forward_cannot_be_lost_after_caller_seed(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 8192, 4_000_000_000)
    observe(r, sig, 8192, 6_000_000_000, caller=False)
    assert r._memory_profiles[sig].compute_peak(8192) == 6_000_000_000
    # Multi-root/ordinary profiles keep their existing max rate.
    other = replace(sig, bounded_caller_peak=False)
    observe(r, other, 128, 1_075_000_000)
    assert not isinstance(r._memory_profiles[other], m._BoundedMemoryProfile)


def test_only_supported_single_root_training_gets_a_caller_profile(caller_rank):
    request = ForwardInput(
        input_tokens=torch.arange(128), target_tokens=torch.arange(128)
    )

    assert classify(caller_rank, [request]).bounded_caller_peak
    for requests, mode, groups in (
        ([request], False, 1),
        ([request, request], True, 1),
        ([request], True, 2),
        ([replace(request, input_tokens=torch.arange(128).reshape(1, -1))], True, 1),
        ([ForwardInput(input_tokens=torch.arange(128), hidden_states=True)], True, 1),
    ):
        assert not getattr(
            classify(caller_rank, requests, mode=mode, groups=groups),
            "bounded_caller_peak",
            False,
        )


@pytest.mark.parametrize(
    "scope",
    (
        "dp",
        "tp",
        "pp",
        "ep",
        "etp",
        "cp-mismatch",
        "cp-zero",
        "dense",
        "unsupported",
        "recompute",
    ),
)
def test_unsupported_geometry_keeps_ordinary_profile(caller_rank, monkeypatch, scope):
    rank = caller_rank
    topology = list(rank._topology_key())
    if scope in ("dp", "tp", "pp"):
        topology[{"dp": 0, "tp": 1, "pp": 3}[scope]] = 2
    elif scope in ("ep", "etp"):
        setattr(rank._parallel_shape, scope, 2)
    elif scope == "cp-mismatch":
        rank._parallel_shape.cp = topology[2] * 2
    elif scope == "cp-zero":
        topology[2] = rank._parallel_shape.cp = 0
    elif scope == "dense":
        rank._moe_layers = 0
    elif scope == "unsupported":
        rank._moe_memory_supported = False
    else:
        monkeypatch.setattr(m, "_checkpoint_layers", lambda *args: 0)
    rank._topology_key = lambda: tuple(topology)
    assert not getattr(classify(rank), "bounded_caller_peak", False)


@pytest.mark.parametrize("cp", (3, 16))
def test_caller_coverage_does_not_need_a_fixed_cp_degree(caller_rank, cp):
    caller_rank._topology_key = lambda: (1, 1, cp, 1)
    caller_rank._parallel_shape.cp = cp
    assert classify(caller_rank).bounded_caller_peak


def memory_rank():
    r = SimpleNamespace(
        _memory_profiles={},
        _num_layers=40,
        _hidden_size=2048,
        _param_dtype_size=2,
        _recompute_granularity="full",
        _geometry=SimpleNamespace(moe_experts=256),
        _moe_workspace_bytes=lambda *args, **kwargs: 0,
        _sequence_parallel_lora_floor=lambda floor, *args: floor,
        _checkpoint_memory_floor=lambda *args: (8 * 2**30, 16 * 2**30),
        _checkpoint_input_gradient_bytes=lambda *args: 0,
        _checkpoint_adapter_gradient_bytes=lambda *args: 0,
        _checkpoint_gradient_groups=lambda *args: (),
        _one_layer_recompute=lambda: True,
        _topology_key=lambda: (1, 1, 1, 1),
        _gdn_segment_layer_bytes=lambda: 0,
    )
    return r


def test_static_floor_and_safety_factor_are_retained(caller_rank, caller_signature):
    r, sig = caller_rank, caller_signature
    observe(r, sig, 8192, 4_000_000_000)
    required = m._estimate_required_memory_bytes_from_values(
        cast(Any, r),
        packed_tokens=8192,
        logical_tokens=8192,
        output_bytes=0,
        signature=sig,
        group_rows=((8192 // sig.topology[2] + 128, True),),
    )
    assert required == int(24 * 2**30 * 1.1)


@pytest.mark.parametrize("forward_peak", [1_000_000_000, 20 * 2**30])
def test_forward_only_long_root_cannot_lower_requirement(
    forward_peak, caller_signature
):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 8192, 16 * 2**30)
    before = r._memory_profiles[sig].compute_peak(32768)
    observe(r, sig, 32768, forward_peak, caller=False)
    assert r._memory_profiles[sig].compute_peak(32768) >= before


def test_first_incomplete_forward_survives_complete_seed(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 8192, 12_000_000_000, caller=False)
    observe(r, sig, 8192, 4_000_000_000)
    assert r._memory_profiles[sig].compute_peak(8192) >= 12_000_000_000


def test_short_reload_never_lowers_extrapolated_requirement(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 2048, 4_000_000_000)
    observe(r, sig, 8192, 12_000_000_000)
    before = r._memory_profiles[sig].compute_peak(32768)
    observe(r, sig, 128, 16_000_000_000)
    assert r._memory_profiles[sig].compute_peak(32768) >= before


def test_no_discount_beyond_complete_caller_extent(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    pinned = replace(sig, bounded_caller_peak=False)
    for rows, peak, caller in (
        (128, 1_075_000_000, False),
        (128, 1_075_000_000, True),
        (2048, 1_700_000_000, True),
        (8192, 4_000_000_000, True),
        (32768, 8_000_000_000, False),
    ):
        for key in (sig, pinned):
            observe(r, key, rows, peak, caller=caller)
    baseline = r._memory_profiles[pinned]
    for rows in (8193, 16384, 32768, 65536, 262144):
        assert (
            r._memory_profiles[sig].compute_peak(rows)
            >= baseline.bytes_per_token * rows
        )


@pytest.mark.parametrize(
    "termination", ["direct", "split-child", "close", "throw", "reset"]
)
def test_incomplete_caller_never_extends_backward_curve(monkeypatch, termination):
    from test_trainer_rank_split_peak import _counter_split

    r, requests, counters = _counter_split(monkeypatch)
    sig = signature()
    monkeypatch.setattr(r, "_memory_signature_from_requests", lambda *a, **k: sig)
    short = r._plan_flat_forward(requests[:1])
    m._update_memory_profile(r, short, 20_000, retained_bytes=None, caller_phase=True)
    before = r._memory_profiles[sig].compute_peak(32768)
    long = replace(short, packed_tokens=32768, logical_tokens=32768, output_bytes=0)
    if termination == "direct":
        r._run_flat_plan_with_memory_tracking(
            long, check=r._memory_check(long), context="test"
        )
    elif termination == "split-child":
        from art.trainer_rank import _impl

        split = _impl._SplitForwardPlan((long, long), ((0,), (1,)), 2)
        r._execute_split_plan_with_memory_tracking(
            split, check=r._memory_check(long), context="test"
        )
    else:
        monkeypatch.setattr(r, "_available_memory_bytes", lambda: 10**12)
        select = r._select_next_micro_batch
        monkeypatch.setattr(
            r,
            "_select_next_micro_batch",
            lambda *a, **k: replace(select(*a, **k), plan=long),
        )
        iterator = r.forward_micro_batches([requests[:1]], yield_empty=True)
        next(iterator)
        if termination == "close":
            iterator.close()
        elif termination == "throw":
            with pytest.raises(ValueError, match="abandoned"):
                iterator.throw(ValueError("abandoned"))
        else:
            r._peak_resets += 1
            assert list(iterator) == []
    assert r._memory_profiles[sig].compute_peak(32768) >= before


def test_forward_only_root_cannot_turn_refusal_into_admission(
    caller_rank, caller_signature
):
    r, sig = caller_rank, caller_signature

    def required():
        return m._estimate_required_memory_bytes_from_values(
            cast(Any, r),
            packed_tokens=32768,
            logical_tokens=32768,
            output_bytes=0,
            signature=sig,
            group_rows=((32768 // sig.topology[2] + 128, True),),
        )

    observe(r, sig, 8192, 16 * 2**30)
    before = required()
    assert before > 40 * 2**30
    observe(r, sig, 32768, 20 * 2**30, caller=False)
    assert required() >= before
    assert required() > 40 * 2**30
    # Only a complete covering interval may replace the conservative rate.
    observe(r, sig, 32768, 24 * 2**30)
    assert int(24 * 2**30 * 1.1) <= required() < 40 * 2**30


def test_observation_pruning_preserves_all_peak_floors(caller_signature):
    r, sig = SimpleNamespace(_memory_profiles={}), caller_signature
    observe(r, sig, 8192, 50_000_000_000, caller=False)
    for length in range(128, 32769, 128):
        observe(r, sig, length, 1_000_000_000 + length * 300_000)
    profile = r._memory_profiles[sig]
    assert len(profile.caller_compute_peaks) == 8
    assert profile.caller_compute_peaks[-1][0] == 32768
    values = [profile.compute_peak(n) for n in range(128, 131073, 128)]
    assert min(values) >= 50_000_000_000
    assert values == sorted(values)


def test_tiny_callers_cannot_poison_a_measured_long_root(caller_rank, caller_signature):
    r, sig = caller_rank, caller_signature
    cp, tokens = sig.topology[2], 9216

    def required():
        return m._estimate_required_memory_bytes_from_values(
            cast(Any, r),
            packed_tokens=tokens,
            logical_tokens=tokens,
            output_bytes=0,
            signature=sig,
            # Ownership can be uneven; packed length is still the global 9K.
            group_rows=((tokens // cp + 128, True),),
        )

    for _ in range(2):
        observe(r, sig, 128, 1_075_000_000)
    assert required() > 40 * 2**30
    observe(r, sig, tokens, 4_000_000_000, caller=False)
    assert required() > 40 * 2**30
    # Measuring 1.1K packed tokens on CP8 cannot cover a 9K packed root.
    observe(r, sig, tokens // cp, 4_000_000_000)
    if cp > 1:
        assert required() > 40 * 2**30
    observe(r, sig, tokens, 4_000_000_000)
    assert required() == int(24 * 2**30 * 1.1) < 40 * 2**30
    # A later tiny caller/reload remains a fixed floor within this coverage.
    observe(r, sig, 128, 12_000_000_000)
    profile = r._memory_profiles[sig]
    assert profile.compute_peak(tokens) == 12_000_000_000
    assert required() < 40 * 2**30
    assert profile.compute_peak(tokens + 1) >= profile.bytes_per_token * (tokens + 1)


def test_profile_roundtrip_preserves_topology_coverage_and_floors(
    caller_rank, caller_signature, tmp_path
):
    r, sig = caller_rank, caller_signature
    observe(r, sig, 128, 1_075_000_000)
    observe(r, sig, 8192, 4_000_000_000)
    observe(r, sig, 32768, 6_000_000_000, caller=False)
    # Profiles are rank-local in memory; exercise their serialization without
    # claiming ART saves/loads caller calibration with model checkpoints.
    path = tmp_path / "profiles.pkl"
    path.write_bytes(pickle.dumps(r._memory_profiles))
    restored = pickle.loads(path.read_bytes())
    assert restored == r._memory_profiles
    profile = restored[sig]
    assert isinstance(profile, m._CallerMemoryProfile)
    assert asdict(sig)["topology"] == r._topology_key()
    assert asdict(profile)["caller_compute_peaks"] == (
        (128, 1_075_000_000),
        (8192, 4_000_000_000),
    )
    assert profile.compute_peak(8192) == 6_000_000_000
    assert profile.compute_peak(8193) >= profile.bytes_per_token * 8193
    # No curve crosses into another CP cohort or an ordinary signature.
    assert replace(sig, topology=(1, 1, sig.topology[2] * 2, 1)) not in restored
    assert replace(sig, bounded_caller_peak=False) not in restored
    r._memory_profiles = restored
    observe(r, sig, 8192, 7_000_000_000)
    assert r._memory_profiles[sig].compute_peak(8192) == 7_000_000_000


def test_split_digest_distinguishes_cp_and_caller_scope(caller_signature, monkeypatch):
    from test_trainer_rank_split_peak import _counter_split

    from art.trainer_rank import _impl

    r, requests, _ = _counter_split(monkeypatch)
    flat = replace(r._plan_flat_forward(requests[:1]), signature=caller_signature)

    def key(sig):
        child = replace(flat, signature=sig)
        return m._split_memory_key(_impl._SplitForwardPlan((child,), ((0,),), 1))

    original = key(caller_signature)
    assert original is not None and len(original) == 32
    assert key(pickle.loads(pickle.dumps(caller_signature))) == original
    assert key(replace(caller_signature, bounded_caller_peak=False)) != original
    assert key(replace(caller_signature, bounded_forward_peak=True)) != original
    assert (
        key(
            replace(
                caller_signature, topology=(1, 1, caller_signature.topology[2] * 2, 1)
            )
        )
        != original
    )
