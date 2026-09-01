"""Landing contract for best-effort internal splitting in TrainerRank.

Written before the implementation (test-first, as for the automatic planner)
and expected to FAIL on the pre-split tree. Contract, as agreed:

- ``dp_rank_forward`` should try not to raise when splitting the call into
  sequential subforwards would make execution feasible. The split ladder is
  bounded and deterministic: the fewest subforwards that fit, partitioning
  along the prefix tree so shared-prefix siblings stay together.
- All returned autograd graphs remain live together, so admission of
  subforward ``j`` must account for the retained memory of every earlier
  subforward plus the current transient peak. Until a retained-bytes profile
  exists, retained memory is conservatively the full estimate.
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
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TrainerRank,
    TrainerRankMemoryError,
)
from art.trainer_rank._impl import _FlatForwardPlan, _MemoryCheck

if TYPE_CHECKING:
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
    # requests fit (each 20 transient; the second must also carry the first's
    # retained graphs — see the cumulative test for that constraint).
    _packed_budget(monkeypatch, rank, 20)

    outputs = rank.dp_rank_forward(inputs)

    assert [int(output.target_logprobs.item()) for output in outputs] == [0, 1, 2, 3]
    assert len(executed) == 2
    assert [
        tuple(index for group in plan.groups for index in group.request_indices)
        for plan in executed
    ] == [(0, 1), (2, 3)]
    assert rank.last_forward_telemetry()["subforward_count"] == 2


def test_unsplit_call_reports_a_single_subforward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    _packed_budget(monkeypatch, rank, 1_000)

    rank.dp_rank_forward([_request(marker) for marker in range(4)])

    assert rank.last_forward_telemetry()["subforward_count"] == 1


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
    # Even a single 10-token request exceeds the budget: refuse, but only
    # after a bounded ladder (1, 2, 4, 8 subforwards; two layout modes each).
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
    assert plan_calls <= 2 * (8).bit_length() + 2


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
    monkeypatch.setattr(rank, "_retained_fraction", lambda plan: 0.1)

    outputs = rank.dp_rank_forward([_request(marker) for marker in range(4)])

    assert len(outputs) == 4
    assert len(executed) == 2


def test_forward_micro_batches_splits_the_minimum_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = _rank(monkeypatch)
    _recording_executor(monkeypatch, rank)
    monkeypatch.setattr(rank, "_retained_fraction", lambda plan: 0.0)
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
    monkeypatch.setattr(rank, "_retained_fraction", lambda plan: 0.0)
    _packed_budget(monkeypatch, rank, 30)
    inputs = [_request(marker) for marker in range(8)]

    rank.dp_rank_forward(inputs)
    first = [
        tuple(index for group in plan.groups for index in group.request_indices)
        for plan in executed
    ]
    executed.clear()
    rank.dp_rank_forward(inputs)
    second = [
        tuple(index for group in plan.groups for index in group.request_indices)
        for plan in executed
    ]

    assert first == second
    assert len(first) == 4
