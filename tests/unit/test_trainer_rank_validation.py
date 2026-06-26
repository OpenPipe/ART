from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TopK,
    TrainerRank,
    TrainerRankMemoryError,
    Unset,
    _anchor_disconnected_outputs,
    _MemoryCheck,
    _MemoryProfile,
    _validate_top_k,
)


class _Model:
    vocab_size = 8


def _runtime(model: torch.nn.Module | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=[model or torch.nn.Linear(1, 1)],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=4, num_layers=1),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )


def _target_request(token: int) -> ForwardInput[torch.Tensor, None, None, None]:
    tokens = torch.tensor([token, token + 1], dtype=torch.long)
    return ForwardInput(input_tokens=tokens, target_tokens=tokens)


def test_forward_input_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ForwardInput(input_tokens=torch.tensor([1]), top_k=0)


def test_forward_input_adapter_selection_defaults_to_unset() -> None:
    request = ForwardInput(input_tokens=torch.tensor([1]))

    assert request.checkpoint is Unset
    assert request.lora is Unset


def test_forward_input_accepts_explicit_base_checkpoint() -> None:
    request = ForwardInput(input_tokens=torch.tensor([1]), checkpoint=None)

    assert request.checkpoint is None
    assert request.lora is Unset


def test_forward_input_rejects_checkpoint_and_lora_together() -> None:
    with pytest.raises(ValueError, match="cannot set both checkpoint and lora"):
        ForwardInput(input_tokens=torch.tensor([1]), checkpoint="a", lora="b")


def test_validate_top_k_rejects_values_above_vocab_size() -> None:
    with pytest.raises(ValueError, match="top_k=9 exceeds vocabulary size 8"):
        _validate_top_k(9, _Model())  # type: ignore[arg-type]


def test_trainer_rank_accepts_nested_shared_prefix_for_gdn_runtime() -> None:
    trainer = TrainerRank(_runtime(), shared_prefix_max_depth=2)  # type: ignore[arg-type]

    assert trainer.shared_prefix_max_depth == 2


def test_trainer_rank_accepts_zero_depth_shared_prefix_for_gdn_runtime() -> None:
    trainer = TrainerRank(_runtime(), shared_prefix_max_depth=0)  # type: ignore[arg-type]

    assert trainer.shared_prefix_max_depth == 0


def test_trainer_rank_pop_rejects_empty_adapter_stack() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="No pushed LoRA or checkpoint"):
        trainer.pop_pushed_lora_or_checkpoint()


def test_dp_rank_forward_preserves_nested_shape_for_inactive_requests() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    request_a = ForwardInput(input_tokens=torch.tensor([1]))
    request_b = ForwardInput(input_tokens=torch.tensor([2]))

    outputs = trainer.dp_rank_forward([[request_a], [request_b]])

    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].target_logprobs is None
    assert outputs[1][0].target_logprobs is None
    assert not hasattr(trainer, "forward")
    assert not hasattr(trainer, "micro_batches")


def test_forward_micro_batches_uses_deterministic_dp_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (1, 2))
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(5)])
    )

    assert [batch.indices for batch in batches] == [(1,), (3,), ()]
    assert [len(batch.outputs) for batch in batches] == [1, 1, 0]


def test_forward_micro_batches_syncs_fit_decision_across_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (1, 2))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    sync_flags: list[bool] = []

    def memory_check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        sync_flags.append(sync_across_dp)
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=1 << 30,
            fits=True,
        )

    monkeypatch.setattr(trainer, "_memory_check_required", memory_check)
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )

    next(iter(trainer.forward_micro_batches([_target_request(i) for i in range(6)])))

    assert sync_flags
    assert all(sync_flags)


def test_forward_micro_batches_outputs_match_top_level_nested_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )

    nested = [[_target_request(1), _target_request(3)]]
    batch = next(iter(trainer.forward_micro_batches(nested)))

    assert batch.inputs == nested
    assert len(batch.outputs) == 1
    assert len(batch.outputs[0]) == 2


def test_forward_micro_batches_ramps_after_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))

    def run(plan, **_kwargs):
        trainer._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=0.0,
            packed_tokens=plan.packed_tokens,
        )
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ]

    monkeypatch.setattr(trainer, "_run_flat_plan_with_memory_tracking", run)

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(8)])
    )

    assert batches[0].stats.global_count == 1
    assert batches[0].stats.cold_start
    assert batches[1].stats.global_count > 1
    assert not batches[1].stats.cold_start


def test_forward_micro_batches_does_not_overtrust_tiny_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    inputs = [_target_request(i) for i in range(64)]
    tiny_plan = trainer._plan_flat_forward([inputs[0]])
    trainer._memory_profiles[tiny_plan.signature] = _MemoryProfile(
        bytes_per_token=0.0,
        packed_tokens=tiny_plan.packed_tokens,
    )

    candidate = trainer._select_next_micro_batch(inputs, 0)

    assert candidate.stats_global_count == 8
    assert candidate.plan.packed_tokens == 16


def test_forward_micro_batches_shrinks_to_largest_fitting_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._last_global_micro_batch_size = 4
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )

    def required_memory(**kwargs):
        return kwargs["packed_tokens"]

    def memory_check(required, *, sync_across_dp=False):
        assert sync_across_dp
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=6,
            fits=required <= 6,
        )

    monkeypatch.setattr(
        trainer, "_estimate_required_memory_bytes_from_values", required_memory
    )
    monkeypatch.setattr(trainer, "_memory_check_required", memory_check)
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )

    batch = next(
        iter(trainer.forward_micro_batches([_target_request(i) for i in range(8)]))
    )

    assert batch.stats.global_count == 3
    assert batch.stats.rejected_candidates >= 1


def test_forward_micro_batches_tail_does_not_reset_stable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._last_global_micro_batch_size = 64
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **kwargs: kwargs["packed_tokens"],
    )
    monkeypatch.setattr(
        trainer,
        "_memory_check_required",
        lambda required, *, sync_across_dp=False: _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=128,
            fits=required <= 128,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(130)])
    )

    assert [batch.stats.global_count for batch in batches] == [64, 64, 2]
    assert trainer._last_global_micro_batch_size == 64


def test_forward_micro_batches_grows_small_stable_window_when_work_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._last_global_micro_batch_size = 64
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **kwargs: kwargs["packed_tokens"],
    )
    monkeypatch.setattr(
        trainer,
        "_memory_check_required",
        lambda required, *, sync_across_dp=False: _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=512,
            fits=required <= 512,
        ),
    )

    candidate = trainer._select_next_micro_batch(
        [_target_request(i) for i in range(512)],
        0,
    )

    assert candidate.stats_global_count == 256


def test_forward_micro_batches_avoids_packing_rejected_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )
    original_plan = trainer._plan_flat_forward
    plan_calls = 0
    memory_checks = 0

    def plan(requests):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests)

    def memory_check(plan, *, sync_across_dp=False):
        assert sync_across_dp
        nonlocal memory_checks
        memory_checks += 1
        return _MemoryCheck(
            estimated_required_bytes=plan.packed_tokens,
            available_bytes=10,
            fits=True,
        )

    monkeypatch.setattr(trainer, "_plan_flat_forward", plan)
    monkeypatch.setattr(trainer, "_memory_check", memory_check)
    inputs = [_target_request(i) for i in range(8)]

    batches = list(trainer.forward_micro_batches(inputs))

    assert [batch.stats.global_count for batch in batches] == [8]
    assert plan_calls == 1
    assert memory_checks == 0


def test_forward_micro_batches_replans_reused_input_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    original_plan = trainer._plan_flat_forward
    plan_calls = 0

    def plan(requests):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(requests)

    monkeypatch.setattr(trainer, "_plan_flat_forward", plan)
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_kwargs: [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ],
    )
    inputs = [_target_request(1)]

    list(trainer.forward_micro_batches(inputs))
    inputs[0] = _target_request(10)
    list(trainer.forward_micro_batches(inputs))

    assert plan_calls == 2


def test_cached_adaptive_estimate_rechecks_current_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **kwargs: kwargs["packed_tokens"],
    )
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    original_estimate = trainer._estimate_flat_forward
    estimate_calls = 0
    available = [1 << 30, 1]

    def estimate(requests):
        nonlocal estimate_calls
        estimate_calls += 1
        return original_estimate(requests)

    def memory_check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        assert sync_across_dp
        current = available.pop(0)
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=current,
            fits=required <= current,
        )

    monkeypatch.setattr(trainer, "_estimate_flat_forward", estimate)
    monkeypatch.setattr(trainer, "_memory_check_required", memory_check)
    inputs = [_target_request(1), _target_request(2)]

    first = trainer._cached_adaptive_estimate((0, 1), inputs)
    second = trainer._cached_adaptive_estimate((0, 1), inputs)

    assert first is not None and first[0].fits
    assert second is not None and not second[0].fits
    assert estimate_calls == 1


def test_forward_micro_batches_raises_when_smallest_batch_will_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **_kwargs: 4,
    )
    monkeypatch.setattr(
        trainer,
        "_memory_check_required",
        lambda required, *, sync_across_dp=False: _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=3,
            fits=False,
        ),
    )
    with pytest.raises(TrainerRankMemoryError, match="smallest DP microbatch"):
        next(iter(trainer.forward_micro_batches([_target_request(1)])))


def test_forward_micro_batches_rejects_mismatched_replicated_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    import art.trainer_rank as trainer_rank

    monkeypatch.setattr(trainer_rank.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_rank.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_rank.dist, "get_world_size", lambda: 2)

    def gather(output, value):
        output[:] = [value, value + 1]

    monkeypatch.setattr(trainer_rank.dist, "all_gather_object", gather)

    with pytest.raises(ValueError, match="same top-level input count"):
        list(trainer.forward_micro_batches([_target_request(1)]))


def test_forward_plan_estimates_output_memory_for_request_combo() -> None:
    class FakeGPT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                hidden_size=4,
                num_layers=1,
                padded_vocab_size=10,
            )
            self.decoder = object()

        def _preprocess(self, *args: object, **kwargs: object) -> None:
            return None

    trainer = TrainerRank(_runtime(FakeGPT()))  # type: ignore[arg-type]
    tokens = torch.tensor([1, 2, 3], dtype=torch.long)
    labels = torch.stack((tokens, tokens + 1), dim=1)

    plan = trainer._plan_flat_forward(
        [
            ForwardInput(
                input_tokens=tokens,
                target_tokens=labels,
                top_k=5,
                logits=True,
                hidden_states=True,
            )
        ]
    )

    target_bytes = 3 * 2 * 4
    topk_bytes = 3 * 5 * (4 + 8)
    logits_bytes = 3 * 10 * 4
    hidden_bytes = 3 * 4 * 4
    assert plan.output_bytes == target_bytes + topk_bytes + logits_bytes + hidden_bytes


def test_disconnected_outputs_keep_zero_graph_anchor() -> None:
    hidden = torch.randn(2, 3, requires_grad=True)
    disconnected = torch.zeros(4)
    top_k = TopK(logprobs=torch.zeros(4, 2), tokens=torch.ones(4, 2, dtype=torch.long))

    (anchored,), (anchored_top_k,) = _anchor_disconnected_outputs(
        [disconnected],
        [top_k],
        hidden,
    )

    assert anchored is not None
    assert anchored.requires_grad
    assert anchored_top_k is not None
    assert anchored_top_k.logprobs.requires_grad
    torch.testing.assert_close(anchored, disconnected)
    torch.testing.assert_close(anchored_top_k.logprobs, top_k.logprobs)
    (anchored.sum() + anchored_top_k.logprobs.sum()).backward()
    assert hidden.grad is not None
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
