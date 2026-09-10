"""CPU admission contracts; injected observations are not GPU peak measurements."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRank, Unset


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(hidden_size=8, num_layers=4, padded_vocab_size=32)
        self.decoder = object()

    def _preprocess(self, *args, **kwargs):
        return None


def _rank():
    return TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[_Model()],
                optimizer=None,
                provider=SimpleNamespace(hidden_size=8, num_layers=4),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )


def _requests(output="hidden_states", inactive_length=1):
    tokens = torch.arange(8)
    active = {
        "hidden_states": ForwardInput(input_tokens=tokens, hidden_states=True),
        "target_tokens": ForwardInput(input_tokens=tokens, target_tokens=tokens),
        "logits": ForwardInput(input_tokens=tokens, logits=True),
        "top_k": ForwardInput(input_tokens=tokens, top_k=2),
    }[output]
    return [
        active,
        ForwardInput(input_tokens=torch.arange(inactive_length)),
    ]


@pytest.mark.parametrize(
    "output", ["hidden_states", "target_tokens", "logits", "top_k"]
)
@pytest.mark.parametrize("no_grad", [False, True])
def test_inactive_length_preserves_warm_cost_and_profile(monkeypatch, output, no_grad):
    rank = _rank()
    short = [replace(r, no_grad=no_grad) for r in _requests(output)]
    long = [replace(r, no_grad=no_grad) for r in _requests(output, 8001)]
    plans = [rank._plan_flat_forward(requests) for requests in (short, long)]
    first, second = plans
    assert first.signature == second.signature
    assert first.packed_tokens == second.packed_tokens == 8
    assert first.output_bytes == second.output_bytes
    assert first.output_metadata == second.output_metadata
    assert (first.logical_tokens, second.logical_tokens) == (9, 8009)
    assert len(first.groups) == len(second.groups) == 1
    a, b = first.groups[0], second.groups[0]
    assert a.request_indices == b.request_indices == (0,)
    for field in ("tokens", "group_ids", "parent_ids", "position_ids"):
        assert torch.equal(getattr(a.packed, field), getattr(b.packed, field))
    rank._update_memory_profile(first, 10_000, retained_bytes=1000)
    profile = rank._memory_profiles[first.signature]
    budget = rank._memory_check(first).estimated_required_bytes
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: budget)
    assert rank._memory_check(first).fits
    assert rank._memory_check(second) == rank._memory_check(first)
    assert rank._plan_cost(first) == rank._plan_cost(second)
    for requests, plan in zip((short, long), plans, strict=True):
        lower = rank._split_chunk_lower_cost(
            requests, [r.input_tokens for r in requests], checkpoint=Unset
        )
        assert lower == rank._plan_cost(plan)
    rank._update_memory_profile(second, 10_000, retained_bytes=1000)
    assert rank._memory_profiles[first.signature] == profile


def test_public_pair_avoids_inactive_only_split_and_keeps_total_telemetry(monkeypatch):
    rank = _rank()
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    observed = rank._plan_flat_forward(_requests())
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    budget = rank._memory_check(observed).estimated_required_bytes
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: budget)
    executed = []

    def run(plan, **kwargs):
        executed.append(plan)
        # Only model execution is replaced. Admission, split search, trust and
        # public iterator reconstruction all run their production paths.
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ], None

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    batches = list(rank.forward_micro_batches([_requests(inactive_length=8001)]))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.indices == (0,)
    assert batch.stats.global_count == batch.stats.local_count == 1
    assert batch.stats.logical_tokens == 8009
    assert batch.stats.packed_tokens == 8
    assert len(batch.outputs) == 1 and len(batch.outputs[0]) == 2
    assert batch.stats.subforward_count == len(executed) == 1
    assert batch.stats.estimated_required_bytes == budget
    assert not batch.stats.cold_start


def test_active_length_still_increases_warm_cost():
    rank = _rank()
    short = _requests()
    observed = rank._plan_flat_forward(short)
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    longer = rank._plan_flat_forward(
        [replace(short[0], input_tokens=torch.arange(16)), short[1]]
    )
    assert observed.signature == longer.signature
    assert rank._memory_check(longer).estimated_required_bytes == 22_000
    assert rank._plan_cost(longer).retained > rank._plan_cost(observed).retained


def test_inactive_observation_cannot_discount_later_shared_active_work(monkeypatch):
    ranks = [_rank(), _rank()]
    checks = []
    for rank, inactive_length in zip(ranks, (1, 8001), strict=True):
        requests = _requests(inactive_length=inactive_length)
        observed = rank._plan_flat_forward(requests, memory_minimal=True)
        rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
        candidate = rank._plan_flat_forward(
            [requests[0]] * 8 + _requests()[1:], memory_minimal=True
        )
        assert candidate.signature == observed.signature
        assert candidate.packed_tokens == observed.packed_tokens == 8
        assert candidate.logical_tokens == 65
        monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 20_000)
        checks.append(rank._memory_check(candidate))
    # Identical observed GPU work must produce identical future admission.
    # Total-input ratios formerly discounted the second estimate to 11,985,
    # admitting a plan that the equivalent short calibration refused.
    assert checks[0] == checks[1]
    assert checks[0].estimated_required_bytes == 88_000
    assert not checks[0].fits


def test_warm_admission_rechecks_current_residency(monkeypatch):
    rank = _rank()
    plan = rank._plan_flat_forward(_requests())
    rank._update_memory_profile(plan, 10_000, retained_bytes=1000)
    required = rank._memory_check(plan).estimated_required_bytes
    profile = rank._memory_profiles[plan.signature]
    state = {"allocated": 10_000, "reserved": 10_000}
    total = 100_000
    monkeypatch.setattr(rank, "device", torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _: (total - state["reserved"], total)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: state["allocated"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: state["reserved"])
    monkeypatch.delenv("ART_TRAINER_RANK_TEST_HOOKS", raising=False)
    # Fresh memory accounting observes newly resident state without discarding
    # a valid incremental profile; cached free blocks remain reusable.
    for allocated, reserved, fits in [
        (10_000, 10_000, True),
        (90_000, 90_000, False),
        (10_000, 90_000, True),
    ]:
        state.update(allocated=allocated, reserved=reserved)
        check = rank._memory_check(plan)
        assert check.estimated_required_bytes == required
        assert check.available_bytes == total - allocated - 3000
        assert check.fits == fits
        assert rank._memory_profiles[plan.signature] == profile


def test_distinct_output_and_pure_grad_modes_require_their_own_profile():
    rank = _rank()
    requests = _requests()
    observed = rank._plan_flat_forward(requests)
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    for changed in (
        [replace(r, no_grad=True) for r in requests],
        _requests("logits"),
        _requests("target_tokens"),
        _requests("top_k"),
    ):
        plan = rank._plan_flat_forward(changed)
        assert plan.signature != observed.signature
        assert not rank._all_ranks_have_memory_profile(
            packed_tokens=plan.packed_tokens, signature=plan.signature
        )
