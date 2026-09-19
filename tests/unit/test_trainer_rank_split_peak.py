"""CPU split-admission regressions; counter observations are not native peaks."""

from collections import namedtuple
from contextlib import nullcontext
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import _impl as tr


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(hidden_size=8, num_layers=4, padded_vocab_size=32)
        self.decoder = object()

    def _preprocess(self, *args, **kwargs):
        return None


def _rank():
    return tr.TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[_Model()],
                optimizer=None,
                provider=SimpleNamespace(
                    hidden_size=8, num_layers=4, recompute_granularity="full"
                ),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )


def _requests(count=2, length=100):
    return [
        tr.ForwardInput(
            input_tokens=torch.tensor([10_000 + i, *range(1, length)]),
            target_tokens=torch.tensor([10_000 + i, *range(1, length)]),
        )
        for i in range(count)
    ]


def _native_slot_fields(monkeypatch, rank):
    # Isolate other regressions from the separate CPU fallback test. These are
    # the native LoRASlotRef's two scalar fields, without importing Megatron.
    NativeSlotFields = namedtuple("NativeSlotFields", "kind name")
    monkeypatch.setattr(
        rank, "_slot_ref", lambda name: NativeSlotFields("checkpoint", name)
    )


def _split(rank, requests, chunks, *, memory_minimal=False):
    return tr._SplitForwardPlan(
        tuple(
            rank._plan_flat_forward(
                [requests[i] for i in chunk], memory_minimal=memory_minimal
            )
            for chunk in chunks
        ),
        tuple(chunks),
        len(requests),
    )


def test_local_checkpoint_fallback_still_admits_split(monkeypatch):
    rank = _rank()
    # This is the existing _slot_ref outcome when optional Megatron is absent.
    monkeypatch.setattr(rank, "_slot_ref", lambda name: tr._LocalLoRASlotRef(name))
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda *, packed_tokens, **_: packed_tokens,
    )
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 100)
    requests = _requests()
    plan, check = rank._admit_split_rung(
        ((0,), (1,)),
        requests,
        [r.input_tokens for r in requests],
        checkpoint=tr.Unset,
    )
    assert isinstance(plan, tr._SplitForwardPlan)
    assert check.fits
    assert all(isinstance(g.slot_ref, tr._LocalLoRASlotRef) for g in plan.groups)


@pytest.mark.parametrize("shared_prefix", [False, True])
def test_128_real_request_fields_have_a_bounded_split_key(monkeypatch, shared_prefix):
    rank = _rank()
    _native_slot_fields(monkeypatch, rank)
    requests = _requests(128, 3)
    if shared_prefix:
        requests = [
            replace(r, input_tokens=torch.tensor([1, 2, i + 3]))
            for i, r in enumerate(requests)
        ]
    # Exercise actual request mix, modes, output metadata, segment construction,
    # coefficient selection and tensor shapes rather than hand-built plans.
    requests = [
        replace(r, no_grad=bool(i % 2), hidden_states=True)
        for i, r in enumerate(requests)
    ]
    plan = _split(
        rank,
        requests,
        (tuple(range(64)), tuple(range(64, 128))),
        memory_minimal=shared_prefix,
    )
    key = rank._split_memory_key(plan)
    assert isinstance(key, bytes) and len(key) == 32
    assert key == rank._split_memory_key(plan)
    changed = replace(
        plan.subforwards[0], output_bytes=plan.subforwards[0].output_bytes + 4
    )
    assert (
        rank._split_memory_key(
            replace(plan, subforwards=(changed, plan.subforwards[1]))
        )
        != key
    )


def test_profile_order_change_cannot_drop_completed_split_floor(monkeypatch):
    rank = _rank()
    _native_slot_fields(monkeypatch, rank)
    requests = _requests()
    requests[1] = replace(requests[1], no_grad=True)
    plan = _split(rank, requests, ((0,), (1,)))
    a, b = plan.subforwards
    assert a.signature != b.signature
    # Distinct real signatures can learn distinct rates. Keep the static model
    # floor small to isolate profile-driven reordering, not kernel accounting.
    rank._hidden_size = rank._param_dtype_size = 1
    rank._num_layers = 0
    rank._memory_profiles[a.signature] = tr._MemoryProfile(
        20, 100, retained_compute_bytes_per_token=1
    )
    rank._memory_profiles[b.signature] = tr._MemoryProfile(
        10, 100, retained_compute_bytes_per_token=1
    )
    assert rank._plan_cost(a).ephemeral > rank._plan_cost(b).ephemeral
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 10_100)
    rank._record_split_memory_floor(plan, 100, 1_200)
    # The real profile update changes only cost/order, not requests or geometry.
    rank._update_memory_profile(b, 3_000, retained_bytes=500)
    assert rank._plan_cost(b).ephemeral > rank._plan_cost(a).ephemeral
    before = dict(rank._memory_profiles)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 10_000)
    accepted, check = rank._admit_split_rung(
        ((0,), (1,)),
        requests,
        [r.input_tokens for r in requests],
        checkpoint=tr.Unset,
    )
    assert accepted is None
    assert check.estimated_required_bytes >= 11_000
    assert rank._memory_profiles == before


def _counter_split(monkeypatch):
    rank = _rank()
    # This executor injects allocator counters without creating cached graphs.
    monkeypatch.setattr(rank, "_graph_memory_policy_enabled", lambda: False)
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    _native_slot_fields(monkeypatch, rank)
    requests = _requests()
    child = rank._plan_flat_forward(requests[:1])
    rank._memory_profiles[child.signature] = tr._MemoryProfile(
        0,
        100,
        retained_compute_bytes_per_token=0,
    )
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 10_000)
    counters: dict[str, Any] = dict(allocated=100, peak=100, resets=[], executed=0)
    monkeypatch.setattr(tr, "_telemetry_phase", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: counters["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: counters["peak"])

    def reset(_):
        counters["resets"].append(counters["allocated"])
        counters["peak"] = counters["allocated"]

    def execute(plan):
        counters["executed"] += 1
        if counters["executed"] == counters.get("fail_at"):
            counters["profiles_before_failure"] = dict(rank._memory_profiles)
            raise counters["error"]
        counters["peak"] = counters["allocated"] + 8_000
        counters["allocated"] += 500
        return [
            tr.ForwardOutput(torch.zeros(100), None, None, None)
            for _ in range(plan.request_count)
        ]

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    rank.device = torch.device("cuda")  # Only injected counters; no CUDA tensors.
    return rank, requests, counters


def test_completed_iterator_preserves_caller_peak_for_next_admission(monkeypatch):
    rank, requests, counters = _counter_split(monkeypatch)
    # This fixture's 10,000-byte admission budget is synthetic, not a physical
    # deficit. Keep its learned-floor refusal independent of cache recovery.
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (1_000_000, 1_000_000))
    releases = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: releases.append(True))
    iterator = rank.forward_batches([requests], yield_empty=True)
    batch = next(iterator)
    assert batch.stats.subforward_count == counters["executed"] == 2
    assert counters["resets"] == [100, 600]
    children = dict(rank._memory_profiles)
    counters["peak"] = 21_000
    assert list(iterator) == []
    assert rank._memory_profiles == children
    del batch
    counters["allocated"] = 100
    with pytest.raises(tr.TrainerRankMemoryError):
        next(rank.forward_batches([requests], yield_empty=True))
    assert counters["executed"] == 2

    assert releases == []


@pytest.mark.parametrize("termination", ["throw", "close"])
def test_incomplete_caller_does_not_learn_split_peak(monkeypatch, termination):
    rank, requests, counters = _counter_split(monkeypatch)
    iterator = rank.forward_batches([requests], yield_empty=True)
    batch = next(iterator)
    assert batch.stats.subforward_count == counters["executed"] == 2
    children = dict(rank._memory_profiles)
    counters["peak"] = 21_000
    if termination == "throw":
        original = RuntimeError("caller failed after partial work")
        with pytest.raises(RuntimeError) as caught:
            iterator.throw(original)
        assert caught.value is original
    else:
        assert iterator.close() is None
    assert list(iterator) == []
    assert rank._split_memory_floors == {}
    assert rank._memory_profiles == children
    assert counters["executed"] == 2


def test_partial_forward_does_not_learn_split_peak(monkeypatch):
    rank, requests, counters = _counter_split(monkeypatch)
    original = torch.cuda.OutOfMemoryError("second split child allocation")
    counters.update(fail_at=2, error=original)
    iterator = rank.forward_batches([requests], yield_empty=True)
    with pytest.raises(tr.TrainerRankPartialExecutionError) as caught:
        next(iterator)
    assert "1 of 2 completed" in str(caught.value)
    assert isinstance(caught.value.__cause__, tr.TrainerRankMemoryError)
    assert caught.value.__cause__.__cause__ is original
    assert list(iterator) == []
    assert counters["executed"] == 2
    assert rank._split_memory_floors == {}
    assert rank._memory_profiles == counters["profiles_before_failure"]


def test_empty_dp_rank_retains_global_selection_collective_sequence(monkeypatch):
    # Real selection/find/rung methods with explicit scalar reductions. This is
    # not native distributed convergence or a model-execution test.
    traces = []
    for dp_rank in (0, 1):
        with monkeypatch.context() as patch:
            rank = _rank()
            _native_slot_fields(patch, rank)
            trace = []
            patch.setattr(rank, "_dp_rank_and_size", lambda: (dp_rank, 2))
            patch.setattr(rank, "_forward_memory_group", lambda: f"tp-cp-{dp_rank}")
            patch.setattr(rank, "_available_memory_bytes", lambda: 100)
            patch.setattr(
                rank,
                "_estimate_required_memory_bytes_from_values",
                lambda *, packed_tokens, **_: packed_tokens,
            )
            patch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
            patch.setattr(rank, "_ensure_checkpoint_slots_for", lambda *a, **k: None)

            # Mode/profile agreements are already collective in production;
            # retain their sequence while controlling this two-rank witness.
            def agree(value):
                trace.append(("global", "agree", bool(value)))
                return bool(value)

            def profiled(**_):
                trace.append(("global", "profile", False))
                return False

            patch.setattr(rank, "_all_ranks_true", agree)
            patch.setattr(rank, "_all_ranks_have_memory_profile", profiled)

            search = rank._search_next_micro_batch
            search_finished = []

            def searched(*args, **kwargs):
                result = search(*args, **kwargs)
                search_finished.append(True)
                return result

            patch.setattr(rank, "_search_next_micro_batch", searched)

            def reduce(value, op, group=None):
                trace.append(("global" if group is None else "local", str(op)))
                if group is None:
                    value.fill_(
                        max(value.item(), 100 if search_finished else 200)
                        if op == tr.dist.ReduceOp.MAX
                        else min(value.item(), 100)
                    )

            patch.setattr(tr.dist, "is_available", lambda: True)
            patch.setattr(tr.dist, "is_initialized", lambda: True)
            patch.setattr(tr.dist, "all_reduce", reduce)
            candidate = rank._select_next_micro_batch([_requests()], 0)
            assert candidate.check.fits
            assert len(candidate.inputs) == (1 if dp_rank == 0 else 0)
            assert isinstance(
                candidate.plan,
                tr._SplitForwardPlan if dp_rank == 0 else tr._FlatForwardPlan,
            )
            traces.append([event for event in trace if event[0] == "global"])
            assert traces[-1][-2:] == [
                ("global", str(tr.dist.ReduceOp.MAX)),
                ("global", str(tr.dist.ReduceOp.MIN)),
            ]
    assert traces[0] == traces[1]


def test_order_normalization_keeps_mapping_partition_and_shapes(monkeypatch):
    rank = _rank()
    _native_slot_fields(monkeypatch, rank)
    requests = _requests(3, 4)
    requests[1] = replace(requests[1], no_grad=True)
    plan = _split(rank, requests, ((0,), (1,), (2,)))
    key = rank._split_memory_key(plan)
    for order in permutations(range(3)):
        q = replace(
            plan,
            subforwards=tuple(plan.subforwards[i] for i in order),
            request_indices=tuple(plan.request_indices[i] for i in order),
        )
        assert rank._split_memory_key(q) == key
    # Mapping moves without the corresponding child must remain distinguishable.
    remapped = replace(plan, request_indices=((1,), (0,), (2,)))
    assert rank._split_memory_key(remapped) != key
    assert rank._split_memory_key(_split(rank, requests, ((0, 1), (2,)))) != key
    a = plan.subforwards[0]
    variants = [
        replace(a, packed_tokens=a.packed_tokens + 1),
        replace(a, output_bytes=a.output_bytes + 4),
        replace(a, signature=replace(a.signature, topology=(1, 2, 1, 1))),
        replace(a, output_metadata=(("different-checkpoint", False),)),
    ]
    assert all(
        rank._split_memory_key(replace(plan, subforwards=(v, *plan.subforwards[1:])))
        != key
        for v in variants
    )


def test_native_and_known_local_reference_preserve_same_floor(monkeypatch):
    rank = _rank()
    _native_slot_fields(monkeypatch, rank)
    requests = _requests()
    plan = _split(rank, requests, ((0,), (1,)))
    local = replace(
        plan,
        subforwards=tuple(
            replace(
                p,
                groups=tuple(
                    replace(g, slot_ref=tr._LocalLoRASlotRef(g.slot_ref.name))
                    for g in p.groups
                ),
            )
            for p in plan.subforwards
        ),
    )
    key = rank._split_memory_key(plan)
    assert rank._split_memory_key(local) == key
    peak = [6100]
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: peak[0])
    original = dict(rank._memory_profiles)
    rank._record_split_memory_floor(plan, 100, 200)
    reversed_plan = replace(
        local,
        subforwards=tuple(reversed(local.subforwards)),
        request_indices=tuple(reversed(local.request_indices)),
    )
    peak[0] = 3100
    rank._record_split_memory_floor(reversed_plan, 100, 200)
    assert rank._split_memory_floors[key] == 6000
    peak[0] = 9100
    rank._record_split_memory_floor(reversed_plan, 100, 200)
    assert rank._split_memory_floors[key] == 9000
    assert rank._memory_profiles == original


def test_large_unsupported_keys_and_full_cache_remain_explicit(monkeypatch):
    rank = _rank()
    _native_slot_fields(monkeypatch, rank)
    plan = _split(rank, _requests(), ((0,), (1,)))
    huge = replace(
        plan, subforwards=(plan.subforwards[0],) * 1025, request_indices=((0,),) * 1025
    )
    assert rank._split_memory_key(huge) is None
    key = rank._split_memory_key(plan)
    rank._split_memory_floors = {i.to_bytes(32, "big"): i for i in range(1024)}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1000)
    before = dict(rank._split_memory_floors)
    rank._record_split_memory_floor(plan, 0, 0)
    assert rank._split_memory_floor_status == "cache_full_not_learned"
    assert rank._split_memory_floors == before and key not in rank._split_memory_floors
    # A field-size limit applies before repr copies arbitrarily long strings.
    a = replace(plan.subforwards[0], output_metadata=(("x" * 4097, False),))
    assert (
        rank._split_memory_key(replace(plan, subforwards=(a, plan.subforwards[1])))
        is None
    )
