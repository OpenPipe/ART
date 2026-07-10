from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

import pytest
import torch

from art.megatron.prefix_tree_packing import (
    estimate_prefix_tree_packed_tokens,
    prefix_tree_pack,
)
from art.trainer_rank import (
    AdapterSelection,
    ForwardInput,
    ForwardOutput,
    TopK,
    TrainerRank,
    TrainerRankMemoryError,
    Unset,
    _flatten,
    _MemoryCheck,
    _MemoryProfile,
)


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


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        model=[_FakeGPT()],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=8, num_layers=4),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )


def _tokens(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _target_request(
    tokens: torch.Tensor,
    *,
    target_count: int = 1,
    top_k: int | None = None,
    logits: bool = False,
    hidden_states: bool = False,
    checkpoint: AdapterSelection = Unset,
    lora: AdapterSelection = Unset,
) -> ForwardInput:
    labels = (
        tokens
        if target_count == 1
        else torch.stack(
            tuple(tokens + offset for offset in range(target_count)),
            dim=-1,
        )
    )
    return ForwardInput(
        input_tokens=tokens,
        target_tokens=labels,
        top_k=top_k,
        logits=logits,
        hidden_states=hidden_states,
        checkpoint=checkpoint,
        lora=lora,
    )


def _set_packed_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    rank: TrainerRank,
    available: int | Callable[[], int],
) -> None:
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda **kwargs: kwargs["packed_tokens"],
    )

    def check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        limit = available() if callable(available) else available
        return _MemoryCheck(required, limit, required <= limit)

    monkeypatch.setattr(rank, "_memory_check_required", check)


def _ternary_tree_sequences() -> tuple[torch.Tensor, ...]:
    # Shape: shared root, two continuation branches, and terminal nodes at
    # several depths. This mirrors prompt -> continuation A/B -> terminal data.
    root = [10, 11, 12]
    left = root + [20, 21]
    right = root + [30, 31, 32]
    return (
        _tokens(*(root + [1])),
        _tokens(*(left + [2])),
        _tokens(*(left + [3, 4])),
        _tokens(*(right + [5])),
        _tokens(*(right + [6, 7])),
        _tokens(80, 81),
    )


def _vineppo_like_inputs() -> list[list[ForwardInput]]:
    groups: list[list[ForwardInput]] = []
    for prompt_index in range(4):
        prompt = [100 + prompt_index, 200 + prompt_index, 201 + prompt_index]
        trajectories = []
        for branch_index, completion_len in enumerate((1, 2, 4)):
            completion = [300 + branch_index] * completion_len
            tokens = _tokens(*(prompt + completion))
            trajectories.append(
                _target_request(
                    tokens,
                    target_count=2 if branch_index == 2 else 1,
                    top_k=5 if branch_index == 1 else None,
                    hidden_states=branch_index == 0,
                )
            )
        groups.append(trajectories)
    return groups


def _random_tree_sequences(seed: int, *, max_depth: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    def segment(depth: int) -> list[int]:
        return [depth * 100 + randint(1, 40) for _ in range(randint(1, 4))]

    def walk(prefix: list[int], depth: int) -> None:
        if depth >= max_depth or randint(0, 2) == 0:
            out.append(_tokens(*(prefix + segment(depth))))
            return
        shared = prefix + segment(depth)
        out.append(_tokens(*shared))
        walk(shared + [10 + depth], depth + 1)
        walk(shared + [20 + depth], depth + 1)

    walk([], 0)
    return tuple(out)


@pytest.mark.parametrize("max_depth", (0, 1, 2, 4))
def test_pack_estimator_matches_ternary_and_random_trees(max_depth: int) -> None:
    cases = [
        _ternary_tree_sequences(),
        _random_tree_sequences(3, max_depth=4),
        _random_tree_sequences(99, max_depth=5),
    ]

    for sequences in cases:
        pack = prefix_tree_pack(sequences, max_depth=max_depth)

        assert estimate_prefix_tree_packed_tokens(
            sequences, max_depth=max_depth
        ) == int(pack.tokens.numel())
        for sequence, positions in zip(
            sequences, pack.positions_by_sequence, strict=True
        ):
            torch.testing.assert_close(pack.tokens.reshape(-1)[positions], sequence)


def test_shared_trainable_tokens_accumulate_independent_output_gradients() -> None:
    sequences = (
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([1, 2, 3], dtype=torch.long),
    )
    pack = prefix_tree_pack(sequences, max_depth=4)
    hidden = torch.randn(int(pack.tokens.numel()), 3, requires_grad=True)
    weights = (2.0, 5.0)

    loss = sum(
        weight * hidden.index_select(0, positions).sum()
        for weight, positions in zip(weights, pack.positions_by_sequence, strict=True)
    )
    loss.backward()

    expected = torch.zeros_like(hidden)
    for weight, positions in zip(weights, pack.positions_by_sequence, strict=True):
        expected.index_add_(
            0,
            positions,
            torch.full((int(positions.numel()), 3), weight, dtype=hidden.dtype),
        )
    torch.testing.assert_close(hidden.grad, expected)


def test_planner_handles_vineppo_nested_shape_and_request_mix() -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=3)  # type: ignore[arg-type]
    inputs = _vineppo_like_inputs()
    flat = list(_flatten(inputs))

    plan = rank._plan_flat_forward(flat)
    estimate = rank._estimate_flat_forward(flat)

    assert estimate is not None
    packed_tokens, output_bytes, signature = estimate
    assert packed_tokens == plan.packed_tokens
    assert output_bytes == plan.output_bytes
    assert signature == plan.signature
    assert plan.request_count == 12
    assert plan.signature.request_mix == (
        "target:(2,)",
        "target:single+hidden",
        "target:single+topk:5",
    )


def test_forward_micro_batches_preserves_nested_vineppo_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=2)  # type: ignore[arg-type]
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    monkeypatch.setattr(
        rank,
        "_memory_check",
        lambda plan, *, sync_across_dp=False: _MemoryCheck(
            plan.packed_tokens, 10_000, True
        ),
    )
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )
    groups = _vineppo_like_inputs()

    micro_batches = list(rank.forward_micro_batches(groups))

    assert [batch.indices for batch in micro_batches] == [(0, 1, 2, 3)]
    assert micro_batches[0].select(groups) == groups
    assert len(micro_batches[0].outputs) == 4
    assert all(
        isinstance(group_outputs, list) and len(group_outputs) == 3
        for group_outputs in micro_batches[0].outputs
    )


def test_adaptive_planner_materializes_only_final_large_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=3)  # type: ignore[arg-type]
    rank._last_global_micro_batch_size = 32
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    plan_calls = 0
    estimate_calls = 0
    original_plan = rank._plan_flat_forward
    original_estimate = rank._estimate_flat_forward
    inputs = [
        _target_request(
            _tokens(1, 2, 3, index % 7, index),
            target_count=2 if index % 5 == 0 else 1,
            top_k=3 if index % 4 == 0 else None,
            hidden_states=index % 9 == 0,
        )
        for index in range(96)
    ]
    limit = rank._estimate_flat_forward(inputs[:40])
    assert limit is not None
    limit_packed_tokens = limit[0]

    def plan(requests):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests)

    def estimate(requests):
        nonlocal estimate_calls
        estimate_calls += 1
        return original_estimate(requests)

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    monkeypatch.setattr(rank, "_estimate_flat_forward", estimate)
    _set_packed_token_budget(monkeypatch, rank, limit_packed_tokens)

    candidate = rank._select_next_micro_batch(inputs, 0)

    assert candidate.stats_global_count == 40
    assert plan_calls == 1
    assert estimate_calls <= 10
    assert candidate.rejected_candidates <= 8


def test_adaptive_planner_globally_falls_back_when_one_rank_cannot_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 2))
    monkeypatch.setattr(rank, "_all_ranks_true", lambda _local: False)
    plans = 0
    original = rank._plan_flat_forward

    def plan(requests):
        nonlocal plans
        plans += 1
        return original(requests)

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    candidate = rank._select_next_micro_batch(
        [_target_request(_tokens(index)) for index in range(4)], 0
    )

    assert candidate.stats_global_count == 2
    assert plans == 1


def test_adaptive_planner_probes_new_heterogeneous_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_resolve_slot_ref", lambda request: request.checkpoint)
    inputs = [
        _target_request(_tokens(index), checkpoint=f"S{index % 4}")
        for index in range(16)
    ]

    first = rank._select_next_micro_batch(inputs, 0)
    rank._memory_profiles[first.plan.signature] = _MemoryProfile(0.0, 1_000_000)
    rank._last_global_micro_batch_size = 1
    second = rank._select_next_micro_batch(inputs, 1)
    rank._memory_profiles[second.plan.signature] = _MemoryProfile(0.0, 1_000_000)
    rank._last_global_micro_batch_size = 2
    third = rank._select_next_micro_batch(inputs, 3)

    assert [
        first.stats_global_count,
        second.stats_global_count,
        third.stats_global_count,
    ] == [
        1,
        2,
        4,
    ]


def test_adaptive_planner_grows_stable_window_to_largest_aligned_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=1)  # type: ignore[arg-type]
    rank._last_global_micro_batch_size = 512
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    _set_packed_token_budget(monkeypatch, rank, 700)

    candidate = rank._select_next_micro_batch(
        [_target_request(_tokens(index)) for index in range(900)],
        0,
    )

    assert candidate.stats_global_count == 672
    assert candidate.rejected_candidates <= 2


def test_forward_micro_batches_shrinks_when_memory_budget_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=2)  # type: ignore[arg-type]
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    inputs = [_target_request(_tokens(1, 2, 3, index)) for index in range(14)]
    first_limit = rank._estimate_flat_forward(inputs[:8])
    tail_limit = rank._estimate_flat_forward(inputs[8:11])
    assert first_limit is not None
    assert tail_limit is not None
    first_limit_packed_tokens = first_limit[0]
    tail_limit_packed_tokens = tail_limit[0]
    available = {"packed_tokens": first_limit_packed_tokens}
    plan_calls = 0
    original_plan = rank._plan_flat_forward

    def plan(requests):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests)

    def run(plan, **_kwargs):
        if available["packed_tokens"] == first_limit_packed_tokens:
            available["packed_tokens"] = tail_limit_packed_tokens
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ]

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    _set_packed_token_budget(monkeypatch, rank, lambda: available["packed_tokens"])
    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)

    batches = list(rank.forward_micro_batches(inputs))

    assert [batch.stats.global_count for batch in batches] == [8, 3, 3]
    assert [batch.stats.available_bytes for batch in batches] == [
        first_limit_packed_tokens,
        tail_limit_packed_tokens,
        tail_limit_packed_tokens,
    ]
    assert [batch.indices for batch in batches] == [
        tuple(range(8)),
        (8, 9, 10),
        (11, 12, 13),
    ]
    assert plan_calls == len(batches)


def test_heterogeneous_slots_split_packing_without_losing_output_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime(), shared_prefix_max_depth=4)  # type: ignore[arg-type]
    monkeypatch.setattr(
        TrainerRank,
        "_slot_ref",
        staticmethod(lambda kind, name: (kind, name)),
    )
    rank.set_checkpoint("student")
    requests = [
        _target_request(_tokens(1, 2, 3), top_k=3),
        _target_request(_tokens(1, 2, 4), checkpoint=None, logits=True),
        _target_request(_tokens(1, 2, 5), lora="teacher", hidden_states=True),
        _target_request(_tokens(1, 2, 6), checkpoint="critic", target_count=4),
    ]

    plan = rank._plan_flat_forward(requests)
    estimate = rank._estimate_flat_forward(requests)

    assert estimate is not None
    packed_tokens, output_bytes, signature = estimate
    assert packed_tokens == plan.packed_tokens
    assert output_bytes == plan.output_bytes
    assert signature == plan.signature
    assert plan.signature.slot_group_count == 4
    assert {group.slot_ref for group in plan.groups} == {
        ("checkpoint", "student"),
        ("checkpoint", None),
        ("lora", "teacher"),
        ("checkpoint", "critic"),
    }


@pytest.mark.parametrize("api", ("dp_rank_forward", "forward_micro_batches"))
def test_forward_raises_before_expected_oom_with_actionable_context(
    api: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())  # type: ignore[arg-type]
    if api == "dp_rank_forward":
        monkeypatch.setattr(
            rank,
            "_memory_check",
            lambda plan, **_kwargs: _MemoryCheck(plan.output_bytes + 1, 0, False),
        )
    else:
        monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
        monkeypatch.setattr(
            rank,
            "_estimate_required_memory_bytes_from_values",
            lambda **_kwargs: 99,
        )
        monkeypatch.setattr(
            rank,
            "_memory_check_required",
            lambda required, **_kwargs: _MemoryCheck(required, 1, False),
        )
    request = [_target_request(_tokens(1, 2, 3), logits=True)]

    with pytest.raises(TrainerRankMemoryError) as exc_info:
        (
            rank.dp_rank_forward(request)
            if api == "dp_rank_forward"
            else next(iter(rank.forward_micro_batches(request)))
        )

    message = str(exc_info.value)
    assert api in message
    assert "packed_tokens=" in message
    assert "logical_tokens=" in message
    assert "output_gb=" in message
    assert "Use smaller top-level items" in message


def test_flatten_rejects_dicts_to_avoid_silent_top_level_shape_changes() -> None:
    with pytest.raises(TypeError, match="dict was passed directly"):
        list(_flatten({"bad": _target_request(_tokens(1, 2))}))  # type: ignore[arg-type]


def test_no_output_requests_do_not_pack_or_consume_compute_memory() -> None:
    rank = TrainerRank(_runtime())  # type: ignore[arg-type]
    requests: Iterable[ForwardInput] = [
        ForwardInput(input_tokens=_tokens(1, 2, 3)),
        ForwardInput(input_tokens=_tokens(1, 2, 4)),
    ]

    plan = rank._plan_flat_forward(list(requests))

    assert plan.groups == ()
    assert plan.packed_tokens == 0
    assert rank._memory_check(plan).estimated_required_bytes == 0
