from __future__ import annotations

from dataclasses import dataclass
import gc
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import (
    AdamParams,
    ForwardInput,
    ForwardOutput,
    TopK,
    TrainerRank,
    TrainerRankMemoryError,
    TrainerRankSlotStateError,
    Unset,
    _anchor_disconnected_outputs,
    _MemoryCheck,
    _MemoryProfile,
    _validate_top_k,
)


class _Model:
    vocab_size = 8


class _FakeLoRASite(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.A_T = torch.nn.Parameter(torch.zeros(4, 2, device=device, dtype=dtype))
        self.B_T = torch.nn.Parameter(torch.zeros(2, 5, device=device, dtype=dtype))

    def _expected_weight_keys(self, suffix: str) -> list[str]:
        return [f"{self.prefix}.{suffix}.weight"]


class _NativeOptimizer:
    config = None
    param_groups: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> tuple[bool, float, int | None]:
        self.step_calls += 1
        raise AssertionError("TrainerRank must not step the native optimizer")

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1


@dataclass(frozen=True)
class _SlotRef:
    kind: str
    name: str | None


def _runtime(
    model: torch.nn.Module | None = None,
    *,
    optimizer: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=[model or torch.nn.Linear(1, 1)],
        optimizer=optimizer,
        provider=SimpleNamespace(hidden_size=4, num_layers=1),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )


def _target_request(token: int) -> ForwardInput[torch.Tensor, None, None, None]:
    tokens = torch.tensor([token, token + 1], dtype=torch.long)
    return ForwardInput(input_tokens=tokens, target_tokens=tokens)


def _indexed_outputs(plan: object, **_kwargs: object) -> list[ForwardOutput]:
    return [
        ForwardOutput(torch.tensor([index], dtype=torch.float32), None, None, None)
        for index in range(int(getattr(plan, "request_count")))
    ]


def _output_values(outputs: object) -> list[int]:
    if isinstance(outputs, ForwardOutput):
        target_logprobs = outputs.target_logprobs
        assert isinstance(target_logprobs, torch.Tensor)
        return [int(target_logprobs.item())]
    values: list[int] = []
    for item in outputs:  # type: ignore[union-attr]
        values.extend(_output_values(item))
    return values


def _output_shape(outputs: object) -> object:
    if isinstance(outputs, ForwardOutput):
        return "output"
    return [_output_shape(item) for item in outputs]  # type: ignore[union-attr]


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


def test_trainer_rank_load_rejects_active_adapter_stack() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._slot_stack.append(object())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="Cannot load a LoRA/checkpoint"):
        trainer.load_checkpoint_slot("teacher", {})
    with pytest.raises(RuntimeError, match="Cannot load a LoRA/checkpoint"):
        trainer.load_lora_slot("teacher", {})


def test_trainer_rank_rejects_adapter_keys_without_installed_lora_site() -> None:
    trainer = TrainerRank(_runtime(_FakeLoRASite("base.layer")))  # type: ignore[arg-type]
    valid = {
        "base.layer.lora_A.weight": torch.empty(1),
        "base.layer.lora_B.weight": torch.empty(1),
    }
    trainer._validate_adapter_slot_keys("checkpoint", "student", valid)

    with pytest.raises(ValueError, match="matching LoRA target modules"):
        trainer._validate_adapter_slot_keys(
            "checkpoint",
            "student",
            {**valid, "base.other.lora_A.weight": torch.empty(1)},
        )


def test_trainer_rank_normalizes_adapter_tensors_to_installed_site() -> None:
    site = _FakeLoRASite("base.layer", dtype=torch.bfloat16)
    trainer = TrainerRank(_runtime(site))  # type: ignore[arg-type]
    adapter = {
        "base.layer.lora_A.weight": torch.ones(3, 4, dtype=torch.float32),
        "base.layer.lora_B.weight": torch.ones(5, 3, dtype=torch.float32),
    }

    normalized = trainer._normalize_adapter_model(adapter)

    assert all(tensor.device == site.A_T.device for tensor in normalized.values())
    assert all(tensor.dtype == torch.bfloat16 for tensor in normalized.values())


def test_trainer_rank_default_forward_uses_explicit_base_slot() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]

    plan = trainer._plan_flat_forward([_target_request(1)])

    assert len(plan.groups) == 1
    slot = plan.groups[0].slot_ref
    assert slot is not None
    assert getattr(slot, "kind") == "checkpoint"
    assert getattr(slot, "name") is None


def test_optim_step_requires_loaded_checkpoint_slot() -> None:
    optimizer = _NativeOptimizer()
    trainer = TrainerRank(_runtime(optimizer=optimizer))  # type: ignore[arg-type]

    with pytest.raises(TrainerRankSlotStateError, match="loaded checkpoint slot"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3))

    assert optimizer.step_calls == 0


def test_optim_step_rejects_loaded_slots_without_grads() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._checkpoint_slot_params_by_name["student"] = (
        torch.nn.Parameter(torch.ones(2)),
    )

    with pytest.raises(TrainerRankSlotStateError, match="none have gradients"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3))
    with pytest.raises(TrainerRankSlotStateError, match="no gradients"):
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-3),
            checkpoints=["student"],
        )


def test_optim_step_rejects_explicit_slot_subset_with_missing_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ready = torch.nn.Parameter(torch.ones(2))
    missing = torch.nn.Parameter(torch.ones(2))
    ready.grad = torch.ones_like(ready)
    trainer._checkpoint_slot_params_by_name["ready"] = (ready,)
    trainer._checkpoint_slot_params_by_name["missing"] = (missing,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    with pytest.raises(TrainerRankSlotStateError, match="missing"):
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-3),
            checkpoints=["ready", "missing"],
        )


def test_optim_step_implicitly_steps_only_slots_with_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ready = torch.nn.Parameter(torch.ones(2))
    untouched = torch.nn.Parameter(torch.ones(2))
    ready.grad = torch.ones_like(ready)
    trainer._checkpoint_slot_params_by_name["ready"] = (ready,)
    trainer._checkpoint_slot_params_by_name["untouched"] = (untouched,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    before_ready = ready.detach().clone()
    before_untouched = untouched.detach().clone()
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )

    assert "ready" in trainer._dynamic_optimizers
    assert "untouched" not in trainer._dynamic_optimizers
    assert not torch.equal(before_ready, ready)
    torch.testing.assert_close(untouched, before_untouched)


def test_checkpoint_slot_optimizer_state_round_trips_same_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([0.5, -0.25])
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )
    state = trainer.checkpoint_slot_optimizer_state("student")

    assert state is not None
    restored = TrainerRank(_runtime())  # type: ignore[arg-type]
    restored._checkpoint_slot_params_by_name["student"] = (
        torch.nn.Parameter(torch.ones(2)),
    )
    restored._dynamic_optimizers["student"] = restored._restore_dynamic_optimizer(
        "student", state
    )

    restored_state = restored.checkpoint_slot_optimizer_state("student")
    assert restored_state is not None
    assert restored_state["optimizer"]
    assert restored_state["master_params"]


def test_checkpoint_slot_optimizer_state_reproduces_exact_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adam = AdamParams(
        learning_rate=3e-4,
        beta1=0.8,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip_norm=10.0,
    )

    def configure(value: torch.Tensor) -> tuple[TrainerRank, torch.nn.Parameter]:
        trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
        param = torch.nn.Parameter(value.clone())
        trainer._checkpoint_slot_params_by_name["student"] = (param,)
        monkeypatch.setattr(
            trainer,
            "_reduce_dynamic_grads",
            lambda params, **_kwargs: tuple(item.grad.float() for item in params),
        )
        return trainer, param

    original, original_param = configure(
        torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    )
    original_param.grad = torch.tensor([0.2, -0.4], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    state = original.checkpoint_slot_optimizer_state("student")
    assert state is not None

    restored, restored_param = configure(original_param.detach())
    restored._dynamic_optimizers["student"] = restored._restore_dynamic_optimizer(
        "student", state
    )
    for param in (original_param, restored_param):
        param.grad = torch.tensor([-0.3, 0.1], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    restored.optim_step(params=adam)

    torch.testing.assert_close(restored_param, original_param, atol=0, rtol=0)
    original_state = original.checkpoint_slot_optimizer_state("student")
    restored_state = restored.checkpoint_slot_optimizer_state("student")
    assert original_state is not None and restored_state is not None
    _assert_nested_tensors_equal(restored_state, original_state)


def test_dynamic_optimizer_keeps_fp32_master_weight_and_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.bfloat16))
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )

    for _ in range(100):
        param.grad = torch.ones_like(param)
        trainer.optim_step(
            params=AdamParams(
                learning_rate=1e-5,
                weight_decay=0.0,
                grad_clip_norm=10.0,
            )
        )

    dynamic = trainer._dynamic_optimizers["student"]
    assert dynamic.master_params[0].dtype == torch.float32
    assert param.item() < torch.tensor(0.1, dtype=torch.bfloat16).item()
    state = dynamic.optimizer.state[dynamic.master_params[0]]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_checkpoint_slot_optimizer_state_rejects_layout_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )
    state = trainer.checkpoint_slot_optimizer_state("student")
    assert state is not None
    state["layout"] = {"different": True}

    restored = TrainerRank(_runtime())  # type: ignore[arg-type]
    restored._checkpoint_slot_params_by_name["student"] = (
        torch.nn.Parameter(torch.ones(2)),
    )
    with pytest.raises(TrainerRankSlotStateError, match="topology or parameter layout"):
        restored._restore_dynamic_optimizer("student", state)


def test_checkpoint_slot_optimizer_state_rejects_missing_master_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    trainer.optim_step(params=AdamParams(learning_rate=1e-2))
    state = trainer.checkpoint_slot_optimizer_state("student")
    assert state is not None
    state["master_params"] = ()

    restored = TrainerRank(_runtime())  # type: ignore[arg-type]
    restored._checkpoint_slot_params_by_name["student"] = (
        torch.nn.Parameter(torch.ones(2)),
    )
    with pytest.raises(TrainerRankSlotStateError, match="master parameters"):
        restored._restore_dynamic_optimizer("student", state)


def test_checkpoint_slot_optimizer_state_rejects_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )
    state = trainer.checkpoint_slot_optimizer_state("student")
    assert state is not None

    restored = TrainerRank(_runtime())  # type: ignore[arg-type]
    restored._checkpoint_slot_params_by_name["student"] = (
        torch.nn.Parameter(torch.ones(3)),
    )

    with pytest.raises(TrainerRankSlotStateError, match="topology or parameter layout"):
        restored._restore_dynamic_optimizer("student", state)


def test_trainer_rank_load_rejects_pending_checkpoint_graph() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    output = ForwardOutput(torch.ones(1, requires_grad=True) * 2, None, None, None)

    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]

    with pytest.raises(TrainerRankSlotStateError, match="Cannot load checkpoint slot"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    assert tracked[0].target_logprobs is not None
    tracked[0].target_logprobs.sum().backward()

    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_step_rejects_pending_checkpoint_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_slot_ref", lambda kind, name: _SlotRef(kind, name))
    ref = _SlotRef("checkpoint", "student")
    output = ForwardOutput(torch.ones(1, requires_grad=True) * 2, None, None, None)

    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]

    with pytest.raises(TrainerRankSlotStateError, match="Cannot optim_step"):
        trainer._guard_checkpoint_can_step("student")

    assert tracked[0].target_logprobs is not None
    tracked[0].target_logprobs.sum().backward()

    trainer._guard_checkpoint_can_step("student")


def test_trainer_rank_step_allows_missing_slot_graph_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank.__new__(TrainerRank)
    monkeypatch.setattr(trainer, "_slot_ref", lambda kind, name: _SlotRef(kind, name))

    trainer._guard_checkpoint_can_step("student")


def test_trainer_rank_zero_grad_does_not_clear_live_slot_graphs() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("lora", "teacher")
    output = ForwardOutput(
        None,
        TopK(
            torch.ones(1, requires_grad=True) * 2,
            torch.ones(1, dtype=torch.long),
        ),
        None,
        None,
    )

    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]
    trainer.zero_grad()

    assert tracked[0].top_k is not None
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_retained_backward_keeps_slot_graph_guard() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    output = ForwardOutput(torch.ones(1, requires_grad=True) * 2, None, None, None)
    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]
    target = tracked[0].target_logprobs
    assert target is not None

    target.sum().backward(retain_graph=True)
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    target.sum().backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_tracks_each_independent_output_graph() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    outputs = [
        ForwardOutput(torch.ones(1, requires_grad=True) * scale, None, None, None)
        for scale in (2, 3)
    ]
    tracked = trainer._track_slot_graph_outputs(ref, outputs)  # type: ignore[arg-type]
    first = tracked[0].target_logprobs
    second = tracked[1].target_logprobs
    assert first is not None and second is not None

    first.sum().backward()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    second.sum().backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_tracks_graph_after_output_is_replaced_by_loss() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    output = ForwardOutput(torch.ones(1, requires_grad=True) * 2, None, None, None)
    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]
    target = tracked[0].target_logprobs
    assert target is not None
    loss = target.sum()
    del output, tracked, target
    gc.collect()

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    loss.backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_releases_abandoned_output_graph() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    output = ForwardOutput(torch.ones(1, requires_grad=True) * 2, None, None, None)
    tracked = trainer._track_slot_graph_outputs(ref, [output])  # type: ignore[arg-type]
    del output, tracked
    gc.collect()

    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


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


def test_dp_rank_forward_supports_arbitrary_nested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(
        trainer, "_run_flat_plan_with_memory_tracking", _indexed_outputs
    )
    nested = [
        [[[[[_target_request(1)]]]]],
        [[[[[_target_request(3), _target_request(5)]]]]],
    ]

    outputs = cast(Any, trainer).dp_rank_forward(nested)

    assert _output_shape(outputs) == [
        [[[[["output"]]]]],
        [[[[["output", "output"]]]]],
    ]
    assert _output_values(outputs) == [0, 1, 2]


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


def test_forward_micro_batches_supports_arbitrary_nested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer, "_all_ranks_have_memory_profile", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        trainer, "_run_flat_plan_with_memory_tracking", _indexed_outputs
    )
    nested = [
        [[[[[_target_request(1)]]]]],
        [[[[[_target_request(3), _target_request(5)]]]]],
    ]

    batches = list(cast(Any, trainer).forward_micro_batches(nested))

    assert len(batches) == 1
    assert batches[0].inputs == nested
    assert batches[0].select(nested) == nested
    assert _output_shape(batches[0].outputs) == [
        [[[[["output"]]]]],
        [[[[["output", "output"]]]]],
    ]
    assert _output_values(batches[0].outputs) == [0, 1, 2]


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


def _assert_nested_tensors_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        actual_dict = cast(dict[Any, object], actual)
        expected_dict = cast(dict[Any, object], expected)
        for key in expected_dict:
            _assert_nested_tensors_equal(actual_dict[key], expected_dict[key])
    elif isinstance(expected, tuple | list):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_tensors_equal(actual_item, expected_item)
    else:
        assert actual == expected
