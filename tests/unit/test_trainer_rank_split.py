"""Landing contract for best-effort internal splitting in TrainerRank.

Written before the implementation (test-first, as for the automatic planner)
and expected to FAIL on the pre-split tree. Contract, as agreed:

- ``dp_rank_forward`` should try not to raise when splitting the call into
  sequential subforwards would make execution feasible. The split ladder is
  bounded and deterministic: the fewest subforwards that fit, cutting the
  requests in prefix-local depth-first order so most sharing stays inside one
  chunk.
- All returned autograd graphs remain live together, so admission of
  subforward ``j`` must account for the retained memory of every earlier
  subforward plus the current transient peak. Until a retained-bytes profile
  exists, retained memory is conservatively the full estimate; a profile is
  trusted only near the scale it was observed at.
- Failed rungs are rejected with cheap bounds; the planner runs only for the
  rung that executes. Ensuring checkpoint slots is a collective, so it happens
  exactly once per call regardless of how many rungs this rank tries.
- Split execution creates an independent slot-graph sentinel per subforward,
  and any execution-time failure of an admitted split is reported as
  ``TrainerRankPartialExecutionError``.
- Outputs are reconstructed in the caller's order exactly as an unsplit call
  would return them.
- Refusing is acceptable when the ladder is exhausted (a single request alone
  cannot fit) — confident refusal over expensive search.
- The same machinery applies inside ``forward_micro_batches`` when even the
  minimum wave cannot fit unsplit.
- Telemetry reports ``subforward_count`` (``last_forward_telemetry`` and
  ``MicroBatchStats``); it is 1 for unsplit calls.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TrainerRank,
    TrainerRankMemoryError,
    TrainerRankPartialExecutionError,
    TrainerRankSlotStateError,
)
from art.trainer_rank._impl import _FlatForwardPlan, _MemoryCheck, _MemoryProfile

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef
    from art.megatron.train import TrainingRuntime


class _FakeGPT(torch.nn.Module):
    def __init__(self, *, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_layers=4,
            padded_vocab_size=vocab_size,
        )
        self.decoder = object()

    def _preprocess(self, *args: object, **kwargs: object) -> None:
        return None


def _runtime() -> "TrainingRuntime":
    return SimpleNamespace(
        model=[_FakeGPT()],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=8, num_layers=4),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )  # type: ignore


def _request(marker: int, length: int = 10) -> ForwardInput:
    # Unique leading token: no shareable prefix, so packed tokens equal
    # logical tokens and packed-token budgets map directly onto splits. The
    # marker doubles as the trailing token so executed outputs can be traced
    # back to their request.
    tokens = torch.tensor(
        [10_000 + marker, *range(1, length - 1), marker], dtype=torch.long
    )
    return ForwardInput(input_tokens=tokens, target_tokens=tokens)


def _packed_budget(
    monkeypatch: pytest.MonkeyPatch,
    rank: TrainerRank,
    available: int | Callable[[], int],
) -> None:
    """Express memory purely in packed tokens, bypassing the live model."""

    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda *, packed_tokens, **_kwargs: packed_tokens,
    )

    def check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        limit = available if isinstance(available, int) else available()
        return _MemoryCheck(required, limit, required <= limit)

    monkeypatch.setattr(rank, "_memory_check_required", check)


def _recording_executor(
    monkeypatch: pytest.MonkeyPatch, rank: TrainerRank
) -> list[_FlatForwardPlan]:
    """Replace execution with a recorder that emits traceable outputs."""

    executed: list[_FlatForwardPlan] = []

    def run(plan: _FlatForwardPlan, **_kwargs: object) -> tuple[list, None]:
        executed.append(plan)
        outputs: list[ForwardOutput | None] = [None] * plan.request_count
        for group in plan.groups:
            for index, item in zip(group.request_indices, group.items, strict=True):
                marker = int(item.input_ids[-1])
                outputs[index] = ForwardOutput(
                    torch.tensor([float(marker)]), None, None, None
                )
        assert all(output is not None for output in outputs)
        return outputs, None

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    return executed


def _rank(monkeypatch: pytest.MonkeyPatch) -> TrainerRank:
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    return rank


def test_dp_rank_forward_splits_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    executed = _recording_executor(monkeypatch, rank)
    inputs = [_request(marker) for marker in range(4)]
    # Unsplit: 40 packed tokens. Budget admits 20, so two subforwards of two
    # requests fit once a retained profile says earlier graphs cost nothing
    # extra here (the cumulative test covers the conservative default).
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    _packed_budget(monkeypatch, rank, 20)

    outputs = rank.dp_rank_forward(inputs)

    assert [int(output.target_logprobs.item()) for output in outputs] == [0, 1, 2, 3]
    assert len(executed) == 2
    # Each subforward is a self-contained flat plan (its own local request
    # indices); identify what it executed by the requests' trailing markers.
    assert [
        tuple(int(item.input_ids[-1]) for group in plan.groups for item in group.items)
        for plan in executed
    ] == [(0, 1), (2, 3)]
    telemetry = rank.last_forward_telemetry()
    assert telemetry["subforward_count"] == 2
    assert telemetry["subforward_request_indices"] == ((0, 1), (2, 3))


def test_unsplit_call_reports_a_single_subforward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    _packed_budget(monkeypatch, rank, 1_000)

    rank.dp_rank_forward([_request(marker) for marker in range(4)])

    telemetry = rank.last_forward_telemetry()
    assert telemetry["subforward_count"] == 1
    assert telemetry["subforward_request_indices"] == ((0, 1, 2, 3),)


def test_split_outputs_preserve_nested_caller_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    nested = [
        [_request(0), _request(1)],
        [_request(2)],
        [_request(3), _request(4), _request(5)],
    ]
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    _packed_budget(monkeypatch, rank, 20)

    outputs = rank.dp_rank_forward(nested)

    assert [
        [int(output.target_logprobs.item()) for output in group] for group in outputs
    ] == [[0, 1], [2], [3, 4, 5]]
    assert rank.last_forward_telemetry()["subforward_count"] >= 3


def test_split_ladder_is_bounded_and_refuses_when_one_request_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    plan_calls = 0
    original_plan = rank._plan_flat_forward

    def plan(requests, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests, **kwargs)

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    inputs = [_request(marker) for marker in range(8)]
    # Even a single 10-token request exceeds the budget: refuse after the
    # bounded ladder (2, 4, 8 subforwards), whose failed rungs are rejected
    # with cheap bounds — the planner runs only for the unsplit attempts.
    _packed_budget(monkeypatch, rank, 9)

    with pytest.raises(TrainerRankMemoryError) as exc_info:
        rank.dp_rank_forward(inputs)

    assert exc_info.value.predicted_peak_bytes > exc_info.value.usable_limit_bytes
    assert "smaller" in exc_info.value.suggestion
    # Confident refusal, honestly worded: the bounded ladder was unable to
    # find a feasible split — not a claim that none exists.
    message = str(exc_info.value).lower()
    assert "split" in message
    assert "unable to find" in message or "could not find" in message
    assert "no feasible" not in message and "infeasible" not in message
    assert plan_calls == 2


def test_split_admission_accounts_for_live_graphs_cumulatively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned graphs stay live together; later subforwards pay for earlier ones.

    Four 10-token requests under a 25-token budget: two halves would each fit
    alone (20), but the second half must also carry the first half's retained
    graphs (20 + 20 = 40 > 25); four singles fail the same way at the third
    subforward (10 + 10 + 10 = 30 > 25). With retained memory conservatively
    equal to the estimate (no retained profile yet), the only correct outcome
    is a refusal — admitting the halves would be unsafe.
    """

    rank = _rank(monkeypatch)
    executed = _recording_executor(monkeypatch, rank)
    _packed_budget(monkeypatch, rank, 25)

    with pytest.raises(TrainerRankMemoryError):
        rank.dp_rank_forward([_request(marker) for marker in range(4)])

    assert executed == []


def test_split_admission_uses_a_retained_profile_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once retained bytes are profiled below the transient peak, splits fit.

    Same 4x10 case with a 25-token budget, but a retained profile says only
    10% of the estimate stays live after a subforward returns: two halves are
    20 transient + 2 retained = 22 <= 25, so the call must split in two.
    """

    rank = _rank(monkeypatch)
    executed = _recording_executor(monkeypatch, rank)
    _packed_budget(monkeypatch, rank, 25)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.1)

    outputs = rank.dp_rank_forward([_request(marker) for marker in range(4)])

    assert len(outputs) == 4
    assert len(executed) == 2


def test_forward_micro_batches_splits_the_minimum_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    # One top-level item holding four requests (40 tokens) under a 20-token
    # budget: the minimum wave cannot fit unsplit and must split, not raise.
    items = [[_request(marker) for marker in range(4)]]
    _packed_budget(monkeypatch, rank, 20)

    batches = list(rank.forward_micro_batches(items))

    assert len(batches) == 1
    assert batches[0].stats.global_count == 1
    assert batches[0].stats.subforward_count == 2
    assert [
        [int(output.target_logprobs.item()) for output in group]
        for group in batches[0].outputs
    ] == [[0, 1, 2, 3]]


def test_split_decisions_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    rank = _rank(monkeypatch)
    executed = _recording_executor(monkeypatch, rank)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    _packed_budget(monkeypatch, rank, 30)
    inputs = [_request(marker) for marker in range(8)]

    def partition() -> list[tuple[int, ...]]:
        return [
            tuple(
                int(item.input_ids[-1]) for group in plan.groups for item in group.items
            )
            for plan in executed
        ]

    rank.dp_rank_forward(inputs)
    first = partition()
    executed.clear()
    rank.dp_rank_forward(inputs)
    second = partition()

    assert first == second
    assert len(first) == 4


def test_split_ladder_ensures_checkpoint_slots_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensuring slots is a world collective, so its count must not depend on
    this rank's inputs: DP ranks whose ladders stop at different rungs would
    otherwise deadlock. Exactly one ensure per call, however many rungs run."""

    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    ensured = 0
    original = rank._ensure_checkpoint_slots

    def ensure(names):
        nonlocal ensured
        ensured += 1
        return original(names)

    monkeypatch.setattr(rank, "_ensure_checkpoint_slots", ensure)
    _packed_budget(monkeypatch, rank, 30)

    rank.dp_rank_forward([_request(marker) for marker in range(8)])

    assert rank.last_forward_telemetry()["subforward_count"] == 4
    assert ensured == 1


@pytest.mark.parametrize(
    ("observed_packed_tokens", "expect_split"), ((1, False), (20, True))
)
def test_retained_profile_is_trusted_only_near_its_observed_scale(
    monkeypatch: pytest.MonkeyPatch,
    observed_packed_tokens: int,
    expect_split: bool,
) -> None:
    """A 10% retained fraction observed on a 1-token forward must not authorize
    20-token subforwards (same 4x10 / 25 case as the profile test); observed at
    the subforwards' scale it does."""

    rank = _rank(monkeypatch)
    executed = _recording_executor(monkeypatch, rank)
    _packed_budget(monkeypatch, rank, 25)
    inputs = [_request(marker) for marker in range(4)]
    signature = rank._plan_flat_forward(inputs).signature
    rank._memory_profiles[signature] = _MemoryProfile(
        bytes_per_token=1.0,
        packed_tokens=observed_packed_tokens,
        retained_fraction=0.1,
    )

    if expect_split:
        rank.dp_rank_forward(inputs)
        assert len(executed) == 2
    else:
        with pytest.raises(TrainerRankMemoryError):
            rank.dp_rank_forward(inputs)
        assert executed == []


def test_retained_observations_are_max_merged_once_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _packed_budget(monkeypatch, rank, 1_000)
    plan = rank._plan_flat_forward([_request(marker) for marker in range(4)])

    def fraction() -> float:
        return rank._retained_fraction(
            plan.signature,
            packed_tokens=plan.packed_tokens,
            logical_tokens=plan.logical_tokens,
        )

    # Observations are retained bytes over the same forward's peak delta (40).
    assert fraction() == 1.0  # never observed
    rank._update_memory_profile(plan, 40, retained_bytes=40)
    assert fraction() == 1.0  # observed: everything retained
    rank._update_memory_profile(plan, 40, retained_bytes=4)
    assert fraction() == 1.0  # a lower observation cannot replace it

    rank._memory_profiles.clear()
    rank._update_memory_profile(plan, 40, retained_bytes=20)
    assert fraction() == pytest.approx(0.5)  # first observation taken as is
    rank._update_memory_profile(plan, 40, retained_bytes=28)
    assert fraction() == pytest.approx(0.7)
    rank._update_memory_profile(plan, 40, retained_bytes=12)
    assert fraction() == pytest.approx(0.7)
    rank._update_memory_profile(plan, 40, retained_bytes=None)
    assert fraction() == pytest.approx(0.7)  # peak-only update keeps it


@pytest.mark.parametrize("failing_ordinal", (0, 1))
def test_split_execution_failure_is_reported_as_partial_execution(
    monkeypatch: pytest.MonkeyPatch, failing_ordinal: int
) -> None:
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    _packed_budget(monkeypatch, rank, 20)
    runs = 0

    def run(plan: _FlatForwardPlan, **_kwargs: object) -> tuple[list, None]:
        nonlocal runs
        ordinal, runs = runs, runs + 1
        if ordinal == failing_ordinal:
            raise TrainerRankMemoryError("simulated CUDA OOM")
        return [
            ForwardOutput(torch.zeros(1), None, None, None)
        ] * plan.request_count, None

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)

    with pytest.raises(TrainerRankPartialExecutionError) as exc_info:
        rank.dp_rank_forward([_request(marker) for marker in range(4)])

    message = str(exc_info.value)
    assert f"subforward {failing_ordinal + 1} of 2 failed during execution" in message
    assert f"({failing_ordinal} of 2 completed)" in message
    assert "simulated CUDA OOM" in message


@dataclass(frozen=True)
class _SlotRef:
    name: str | None


def test_split_subforwards_track_independent_slot_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two subforwards on one slot carry independent slot-graph sentinels:
    releasing the first subforward's graph keeps slot load/step blocked until
    the second is released too."""

    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_fraction", lambda *_args, **_kwargs: 0.0)
    _packed_budget(monkeypatch, rank, 20)
    ref = cast("LoRASlotRef", _SlotRef("teacher"))
    monkeypatch.setattr(rank, "_slot_ref", lambda name: _SlotRef(name))
    monkeypatch.setattr(rank, "_resolve_slot_ref", lambda request, **_kwargs: ref)
    monkeypatch.setattr(rank, "_validate_hybridep_topology", lambda: None)
    monkeypatch.setattr(rank, "_topology", lambda: object())
    monkeypatch.setattr(rank, "_configure_hybridep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rank, "_prepare_packed_forward", lambda _packed: None)

    def forward(items: object, _prepared: object) -> list[ForwardOutput]:
        return [
            ForwardOutput(
                torch.ones(1, requires_grad=True) * int(item.input_ids[-1]),
                None,
                None,
                None,
            )
            for item in cast(Any, items)
        ]

    monkeypatch.setattr(rank, "_forward_packed", forward)
    lora = ModuleType("art.megatron.lora")
    cast(Any, lora).use_lora_slot = lambda _slot: nullcontext()
    monkeypatch.setitem(sys.modules, "art.megatron.lora", lora)

    outputs = rank.dp_rank_forward([_request(marker) for marker in range(4)])
    first, second = rank.last_forward_telemetry()["subforward_request_indices"]

    def loss(indices: tuple[int, ...]) -> torch.Tensor:
        return torch.stack([outputs[index].target_logprobs for index in indices]).sum()

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        rank._guard_slot_can_load(ref)
    loss(first).backward()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        rank._guard_slot_can_load(ref)
    with pytest.raises(TrainerRankSlotStateError, match="Cannot optim_step"):
        rank._guard_checkpoint_can_step("teacher")
    loss(second).backward()
    rank._guard_slot_can_load(ref)
    rank._guard_checkpoint_can_step("teacher")
