from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING

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
)
from art.trainer_rank._impl import (
    _CheckpointSlot,
    _flatten,
    _MemoryCheck,
    _MemoryProfile,
)
from art.trainer_rank._prefix_tree_planner import build_canonical_prefix_tree

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
    # Deliberately lightweight structural fake; importing/constructing the real
    # Megatron runtime would make these CPU-only unit tests require Megatron.
    return SimpleNamespace(
        model=[_FakeGPT()],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=8, num_layers=4),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )  # type: ignore


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
    )


def _set_packed_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    rank: TrainerRank,
    available: int | Callable[[], int],
) -> None:
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda *, packed_tokens, **_kwargs: packed_tokens,
    )

    def check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        limit = available if isinstance(available, int) else available()
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

    loss = torch.stack(
        [
            weight * hidden.index_select(0, positions).sum()
            for weight, positions in zip(
                weights, pack.positions_by_sequence, strict=True
            )
        ]
    ).sum()
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
    rank = TrainerRank(_runtime())
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
    rank = TrainerRank(_runtime())
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
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
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


def test_forward_micro_batches_prewarms_next_wave_during_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    inputs = _unshared_requests(8)
    limit = rank._estimate_flat_forward(inputs[:4])
    assert limit is not None
    _set_packed_token_budget(monkeypatch, rank, lambda: limit[0])
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )

    generator = rank.forward_micro_batches(inputs)
    first = next(generator)
    assert first.stats.global_count == 4

    # While the generator is suspended at the yield (the caller's GPU time),
    # the predicted next wave must be planned in the background so the next
    # wave's selection is a cache hit.
    future = rank._speculative_planning_future
    assert future is not None
    future.result(timeout=30)
    next_rows = tuple(
        request.input_tokens.detach().reshape(-1).to(dtype=torch.long)
        for request in inputs[4:8]
    )
    assert rank._cached_group_layout(rank._layout_cache_key(next_rows)) is not None

    remaining = list(generator)
    assert [batch.stats.global_count for batch in remaining] == [4]


def _unshared_requests(count: int) -> list[ForwardInput]:
    """Rows with no shareable prefix, so packed tokens equal logical tokens
    under every layout and packed-token budgets translate directly to waves."""

    return [
        _target_request(_tokens(1_000 + index, 2_000 + index, 3_000 + index, index))
        for index in range(count)
    ]


def _prewarmed_rank(
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[ForwardInput],
    budget_rows: list[ForwardInput],
    *,
    dp: tuple[int, int] = (0, 1),
) -> TrainerRank:
    """Rank whose packed-token budget admits exactly ``budget_rows`` per wave."""

    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: dp)
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    limit = rank._estimate_flat_forward(budget_rows)
    assert limit is not None
    _set_packed_token_budget(monkeypatch, rank, lambda: limit[0])
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )
    return rank


def _rows(requests: Iterable[ForwardInput]) -> tuple[torch.Tensor, ...]:
    return tuple(
        request.input_tokens.detach().reshape(-1).to(dtype=torch.long)
        for request in requests
    )


def test_speculative_planning_uses_immutable_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating caller tensors after the yield must not poison the cache."""

    inputs = _unshared_requests(8)
    rank = _prewarmed_rank(monkeypatch, inputs, inputs[:4])
    original_rows = tuple(row.clone() for row in _rows(inputs[4:8]))
    original_key = rank._layout_cache_key(original_rows)

    generator = rank.forward_micro_batches(inputs)
    next(generator)
    # The caller mutates its (aliased) input tensors while suspended.
    for request in inputs[4:8]:
        request.input_tokens.fill_(999)
    future = rank._speculative_planning_future
    assert future is not None
    future.result(timeout=30)

    cached = rank._cached_group_layout(original_key)
    assert cached is not None
    expected_tree = build_canonical_prefix_tree(
        tuple(tuple(row.tolist()) for row in original_rows)
    )
    assert cached[0].content_fingerprint == expected_tree.content_fingerprint
    assert cached[0].fingerprint == expected_tree.fingerprint


def test_speculative_planning_warms_this_dp_ranks_local_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _unshared_requests(16)
    # Local budget of 4 items per rank -> global waves of 8 at DP2.
    rank = _prewarmed_rank(monkeypatch, inputs, inputs[0:8:2], dp=(0, 2))

    generator = rank.forward_micro_batches(inputs)
    first = next(generator)
    assert first.stats.global_count == 8
    future = rank._speculative_planning_future
    assert future is not None
    future.result(timeout=30)

    # Real planning on DP rank 0 uses the strided local slice of the next
    # global wave [8, 16): items 8, 10, 12, 14 — not the whole global slice.
    local_key = rank._layout_cache_key(_rows(inputs[8:16:2]))
    global_key = rank._layout_cache_key(_rows(inputs[8:16]))
    assert rank._cached_group_layout(local_key) is not None
    assert rank._cached_group_layout(global_key) is None


def test_width_search_lets_prefix_sharing_widen_the_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-sharing upper bound may accept a width, never reject one."""

    # 2,000 shared tokens: under the fitted cost model a shared level on the
    # GDN model pays for itself from roughly 1,500 saved tokens per layer.
    shared = tuple(range(10_000, 12_000))
    inputs = [
        _target_request(_tokens(*shared, 1)),
        _target_request(_tokens(*shared, 2)),
    ]
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )
    plan = rank._plan_flat_forward(inputs)
    assert plan.packed_tokens < 4_002, "planner must share the common prefix"
    # Budget fits the shared plan (2,002 packed) but not the no-sharing bound
    # (4,002); the wave must still take both requests.
    _set_packed_token_budget(monkeypatch, rank, 2_400)

    batches = list(rank.forward_micro_batches(inputs))

    assert [batch.stats.global_count for batch in batches] == [2]


def _attention_runtime() -> "TrainingRuntime":
    # Same structural fake as _runtime(), but an attention-only model support
    # handler so the cost model applies no GDN sharing penalty.
    return SimpleNamespace(
        model=[_FakeGPT()],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=8, num_layers=4),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )  # type: ignore


def test_width_search_survives_non_monotone_cost_optimal_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharing can be declined at width 2 yet accepted at width 3.

    Cost-optimal packed tokens are therefore not monotone in width (fits at
    1, fails at 2, fits at 3 in this case). Feasibility must instead be judged
    by the memory-minimal layout, which is monotone, so the search reaches
    width 3 instead of stopping at the spurious failure.
    """

    # 500 shared tokens on the attention model: one saved copy (width 2) does
    # not pay for the extra segment under the fitted cost model, two saved
    # copies (width 3) do.
    shared = tuple(range(10_000, 10_500))
    inputs = [_target_request(_tokens(*shared, tail)) for tail in (1, 2, 3)]
    rank = TrainerRank(_attention_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )
    # Confirm the non-monotone premise under the production cost model.
    two = rank._plan_flat_forward(inputs[:2])
    three = rank._plan_flat_forward(inputs)
    assert two.packed_tokens == 1_002, two.packed_tokens
    assert three.packed_tokens == 503, three.packed_tokens
    _set_packed_token_budget(monkeypatch, rank, 700)

    batches = list(rank.forward_micro_batches(inputs))

    assert [batch.stats.global_count for batch in batches] == [3]
    assert batches[0].stats.packed_tokens <= 700


def test_dp_rank_forward_falls_back_to_memory_minimal_layout_before_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tuple(range(10_000, 10_040))
    inputs = [_target_request(_tokens(*shared, tail)) for tail in (1, 2)]
    rank = TrainerRank(_attention_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    executed: list[int] = []
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            executed.append(plan.packed_tokens)
            or [
                ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
            ],
            None,
        ),
    )
    # Cost-optimal layout declines sharing (82 tokens); full sharing (42) fits.
    _set_packed_token_budget(monkeypatch, rank, 60)

    outputs = rank.dp_rank_forward(inputs)

    assert len(outputs) == 2
    assert executed == [42]
    assert rank.last_forward_telemetry()["selected_max_depth"] == 2


def test_profiled_steady_state_keeps_the_wide_shared_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile trust must judge the selected layout, not the no-sharing bound.

    GRPO-16: sixteen rows sharing a 1,000-token prompt. The selected
    (full-sharing) width-16 plan packs 1,016 tokens; the no-sharing bound is
    16,016. With an existing profile of 1,016 tokens (trust growth 8x =>
    8,128) and a 1,100-token budget, width 16 fits and is inside the profiled
    regime, so a steady-state call must not regress to width 8 because the
    stale bound looked untrusted.
    """

    prompt = tuple(range(20_000, 21_000))
    inputs = [_target_request(_tokens(*prompt, tail)) for tail in range(16)]
    rank = TrainerRank(_attention_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )
    plan = rank._plan_flat_forward(inputs)
    assert plan.packed_tokens == 1_016, plan.packed_tokens
    # Steady state: a prior call profiled exactly this shape.
    rank._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=1.0, packed_tokens=1_016
    )
    _set_packed_token_budget(monkeypatch, rank, 1_100)

    batches = list(rank.forward_micro_batches(inputs))

    assert [batch.stats.global_count for batch in batches] == [16]
    assert not batches[0].stats.cold_start


def test_forward_micro_batches_telemetry_reports_hidden_speculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _unshared_requests(8)
    rank = _prewarmed_rank(monkeypatch, inputs, inputs[:4])
    list(rank.forward_micro_batches(inputs))
    telemetry = rank.last_forward_telemetry()
    assert telemetry["planning_ms"] > 0.0
    assert "speculative_planning_ms" in telemetry


@pytest.mark.parametrize("api", ("dp_rank_forward", "forward_micro_batches"))
def test_forward_preserves_caller_owned_nested_input_tensors(
    api: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_: True)
    monkeypatch.setattr(
        rank,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        ),
    )
    groups = _vineppo_like_inputs()
    tensors = [
        (request, request.input_tokens, request.target_tokens)
        for group in groups
        for request in group
    ]
    snapshots = [
        (inputs.clone(), None if targets is None else targets.clone())
        for _request, inputs, targets in tensors
    ]

    if api == "dp_rank_forward":
        rank.dp_rank_forward(groups)
    else:
        list(rank.forward_micro_batches(groups))

    for (request, inputs, targets), (expected_inputs, expected_targets) in zip(
        tensors, snapshots, strict=True
    ):
        assert request.input_tokens is inputs
        assert request.target_tokens is targets
        assert inputs.device.type == "cpu"
        torch.testing.assert_close(inputs, expected_inputs)
        if targets is not None and expected_targets is not None:
            assert targets.device.type == "cpu"
            torch.testing.assert_close(targets, expected_targets)


def test_adaptive_planner_materializes_only_final_large_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())
    rank._last_global_micro_batch_size = 32
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    plan_calls = 0
    estimate_calls = 0
    original_plan = rank._plan_flat_forward
    original_estimate = rank._estimate_flat_forward
    # Unique leading tokens: no shareable prefix, so packed tokens equal
    # logical tokens under every layout and the budget maps directly to width.
    inputs = [
        _target_request(
            _tokens(1_000 + index, 2, 3, index % 7, index),
            target_count=2 if index % 5 == 0 else 1,
            top_k=3 if index % 4 == 0 else None,
            hidden_states=index % 9 == 0,
        )
        for index in range(96)
    ]
    limit = rank._estimate_flat_forward(inputs[:40])
    assert limit is not None
    limit_packed_tokens = limit[0]

    def plan(requests, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests, **kwargs)

    def estimate(requests, **kwargs):
        nonlocal estimate_calls
        estimate_calls += 1
        return original_estimate(requests, **kwargs)

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
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 2))
    monkeypatch.setattr(rank, "_all_ranks_true", lambda _local: False)
    plans = 0
    original = rank._plan_flat_forward

    def plan(requests, **kwargs):
        nonlocal plans
        plans += 1
        return original(requests, **kwargs)

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    candidate = rank._select_next_micro_batch(
        [_target_request(_tokens(index)) for index in range(4)], 0
    )

    assert candidate.stats_global_count == 2
    assert plans == 1


def test_adaptive_planner_probes_new_heterogeneous_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        rank,
        "_resolve_slot_ref",
        lambda request, **_kwargs: request.checkpoint,
    )
    for index in range(4):
        rank._checkpoint_slots.setdefault(f"S{index}", _CheckpointSlot()).params = ()
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
    rank = TrainerRank(_runtime())
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
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **_kwargs: True)
    inputs = _unshared_requests(14)
    first_limit = rank._estimate_flat_forward(inputs[:8])
    tail_limit = rank._estimate_flat_forward(inputs[8:11])
    assert first_limit is not None
    assert tail_limit is not None
    first_limit_packed_tokens = first_limit[0]
    tail_limit_packed_tokens = tail_limit[0]
    available = {"packed_tokens": first_limit_packed_tokens}
    plan_calls = 0
    original_plan = rank._plan_flat_forward

    def plan(requests, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests, **kwargs)

    def run(plan, **_kwargs):
        if available["packed_tokens"] == first_limit_packed_tokens:
            available["packed_tokens"] = tail_limit_packed_tokens
        return (
            [ForwardOutput(None, None, None, None) for _ in range(plan.request_count)],
            None,
        )

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
    class SlotRef(str):
        @property
        def name(self) -> str:
            return str(self)

    def slot_ref(name: str | None) -> SlotRef | None:
        return None if name is None else SlotRef(name)

    rank = TrainerRank(_runtime())
    monkeypatch.setattr(
        TrainerRank,
        "_slot_ref",
        staticmethod(slot_ref),
    )
    rank._default_slot_ref = rank._slot_ref("student")
    for name in ("student", "teacher", "critic"):
        rank._checkpoint_slots.setdefault(name, _CheckpointSlot()).params = ()
    requests = [
        _target_request(_tokens(1, 2, 3), top_k=3),
        _target_request(_tokens(1, 2, 4), checkpoint=None, logits=True),
        _target_request(_tokens(1, 2, 5), checkpoint="teacher", hidden_states=True),
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
        "student",
        None,
        "teacher",
        "critic",
    }


@pytest.mark.parametrize("api", ("dp_rank_forward", "forward_micro_batches"))
def test_forward_raises_before_expected_oom_with_actionable_context(
    api: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = TrainerRank(_runtime())
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
    assert "predicted_peak_gb=" in message
    assert "usable_limit_gb=" in message
    assert "Use smaller top-level items" in message
    assert exc_info.value.predicted_peak_bytes >= 0
    assert exc_info.value.usable_limit_bytes >= 0
    assert "smaller" in exc_info.value.suggestion


def test_flatten_rejects_dicts_to_avoid_silent_top_level_shape_changes() -> None:
    with pytest.raises(TypeError, match="dict was passed directly"):
        list(_flatten({"bad": _target_request(_tokens(1, 2))}))  # type: ignore[arg-type]


def test_no_output_requests_do_not_pack_or_consume_compute_memory() -> None:
    rank = TrainerRank(_runtime())
    requests: Iterable[ForwardInput] = [
        ForwardInput(input_tokens=_tokens(1, 2, 3)),
        ForwardInput(input_tokens=_tokens(1, 2, 4)),
    ]

    plan = rank._plan_flat_forward(list(requests))

    assert plan.groups == ()
    assert plan.packed_tokens == 0
    assert rank._memory_check(plan).estimated_required_bytes == 0
