from __future__ import annotations

from dataclasses import dataclass
import gc
import inspect
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


def _empty_outputs(plan: object, **_kwargs: object) -> list[ForwardOutput]:
    return [ForwardOutput(None, None, None, None)] * int(getattr(plan, "request_count"))


def _stub_forward(mp, rank, out=_empty_outputs, dp=(0, 1), profiled=False) -> None:
    mp.setattr(rank, "_dp_rank_and_size", lambda: dp)
    mp.setattr(rank, "_run_flat_plan_with_memory_tracking", out)
    if profiled:
        mp.setattr(rank, "_all_ranks_have_memory_profile", lambda **_: True)


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


def _trainer_with_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    value: torch.Tensor,
) -> tuple[TrainerRank, torch.nn.Parameter]:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    param = torch.nn.Parameter(value.clone())
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    return trainer, param


def _tracked_targets(
    trainer: TrainerRank, ref: _SlotRef, *scales: float
) -> list[torch.Tensor]:
    tracked = trainer._track_slot_graph_outputs(
        ref,  # type: ignore[arg-type]
        [
            ForwardOutput(torch.ones(1, requires_grad=True) * scale, None, None, None)
            for scale in scales
        ],
    )
    return [cast(torch.Tensor, output.target_logprobs) for output in tracked]


def test_forward_input_validation() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ForwardInput(input_tokens=torch.tensor([1]), top_k=0)
    with pytest.raises(ValueError, match="cannot set both checkpoint and lora"):
        ForwardInput(input_tokens=torch.tensor([1]), checkpoint="a", lora="b")
    with pytest.raises(ValueError, match="top_k=9 exceeds vocabulary size 8"):
        _validate_top_k(9, _Model())  # type: ignore[arg-type]


@pytest.mark.parametrize(("checkpoint", "expected"), ((Unset, Unset), (None, None)))
def test_forward_input_distinguishes_unset_and_base_checkpoint(
    checkpoint: object, expected: object
) -> None:
    request = ForwardInput(input_tokens=torch.tensor([1]), checkpoint=checkpoint)  # type: ignore[arg-type]

    assert request.checkpoint is expected
    assert request.lora is Unset


def test_forward_input_preserves_public_runtime_shape() -> None:
    fields = tuple(ForwardInput.__dataclass_fields__)
    assert tuple(inspect.signature(ForwardInput).parameters) == fields
    assert ForwardInput.__match_args__ == fields


@pytest.mark.parametrize("depth", (0, 2))
def test_trainer_rank_accepts_shared_prefix_depth(depth: int) -> None:
    trainer = TrainerRank(_runtime(), shared_prefix_max_depth=depth)  # type: ignore[arg-type]

    assert trainer.shared_prefix_max_depth == depth


def test_trainer_rank_adapter_stack_errors() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="No pushed LoRA or checkpoint"):
        trainer.pop_pushed_lora_or_checkpoint()
    trainer._slot_stack.append(object())  # type: ignore[arg-type]
    for load in (trainer.load_checkpoint_slot, trainer.load_lora_slot):
        with pytest.raises(RuntimeError, match="Cannot load a LoRA/checkpoint"):
            load("teacher", {})


def test_trainer_rank_rejects_adapter_keys_without_installed_lora_site() -> None:
    trainer = TrainerRank(_runtime(_FakeLoRASite("base.layer")))  # type: ignore[arg-type]
    valid = {
        "base.layer.lora_A.weight": torch.empty(1),
        "base.layer.lora_B.weight": torch.empty(1),
    }
    trainer._prepare_adapter_model("checkpoint", "student", valid)

    with pytest.raises(ValueError, match="matching LoRA target modules"):
        trainer._prepare_adapter_model(
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

    normalized = trainer._prepare_adapter_model("checkpoint", "student", adapter)

    assert all(tensor.device == site.A_T.device for tensor in normalized.values())
    assert all(tensor.dtype == torch.bfloat16 for tensor in normalized.values())


def test_checkpoint_slot_adapter_config_is_validated_and_copied() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }

    retained = trainer._validate_checkpoint_slot_adapter_config(
        "student", config, alpha=16
    )

    assert retained == config
    config["target_modules"].append("v_proj")  # type: ignore[union-attr]
    assert retained is not None
    assert retained["target_modules"] == ["q_proj"]
    with pytest.raises(ValueError, match="conflicts"):
        trainer._validate_checkpoint_slot_adapter_config("student", config, alpha=32)
    with pytest.raises(ValueError, match="missing"):
        trainer._validate_checkpoint_slot_adapter_config(
            "student", {"r": 8}, alpha=None
        )


def test_checkpoint_slot_adapter_config_rejects_cross_rank_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    monkeypatch.setattr("art.trainer_rank.dist.is_initialized", lambda: True)
    monkeypatch.setattr("art.trainer_rank.dist.get_world_size", lambda: 2)

    def gather(output: list[object], value: object) -> None:
        output[:] = [value, {"different": True}]

    monkeypatch.setattr("art.trainer_rank.dist.all_gather_object", gather)

    with pytest.raises(ValueError, match="differs across ranks"):
        trainer._validate_checkpoint_slot_adapter_config("student", None, alpha=None)


def test_load_checkpoint_slot_retains_config_and_uses_its_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        trainer,
        "_load_slot",
        lambda *_args, **kwargs: seen.update(kwargs) or 1,
    )
    monkeypatch.setattr(trainer, "_validate_dynamic_slot_consistency", lambda *_: ())
    monkeypatch.setattr(
        trainer, "_validate_loaded_checkpoint_slot_config", lambda *_: None
    )
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }

    trainer.load_checkpoint_slot("student", {}, adapter_config=config)

    assert seen["alpha"] == 16
    assert trainer._checkpoint_slot_adapter_configs["student"] == config
    trainer.load_checkpoint_slot("student", {}, alpha=7)
    assert seen["alpha"] == 7
    assert "student" not in trainer._checkpoint_slot_adapter_configs


def test_checkpoint_slot_publish_requires_retained_adapter_config() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown checkpoint slot"):
        trainer.save_checkpoint_slot_lora("missing", "/unused")

    trainer._checkpoint_slot_params_by_name["student"] = ()

    with pytest.raises(TrainerRankSlotStateError, match="adapter_config"):
        trainer.save_checkpoint_slot_lora("student", "/unused")


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

    original, original_param = _trainer_with_checkpoint(
        monkeypatch, torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    )
    original_param.grad = torch.tensor([0.2, -0.4], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    state = original.checkpoint_slot_optimizer_state("student")
    assert state is not None

    restored, restored_param = _trainer_with_checkpoint(
        monkeypatch, original_param.detach()
    )
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
    trainer, param = _trainer_with_checkpoint(
        monkeypatch, torch.tensor([0.1], dtype=torch.bfloat16)
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


@pytest.mark.parametrize(
    ("corruption", "error"),
    (
        ("layout", "topology or parameter layout"),
        ("missing_master", "master parameters"),
        ("shape", "topology or parameter layout"),
    ),
)
def test_checkpoint_slot_optimizer_state_rejects_incompatible_state(
    corruption: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, param = _trainer_with_checkpoint(monkeypatch, torch.ones(2))
    param.grad = torch.ones_like(param)
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )
    state = trainer.checkpoint_slot_optimizer_state("student")
    assert state is not None
    if corruption == "layout":
        state["layout"] = {"different": True}
    elif corruption == "missing_master":
        state["master_params"] = ()
    restored, _ = _trainer_with_checkpoint(
        monkeypatch, torch.ones(3 if corruption == "shape" else 2)
    )
    with pytest.raises(TrainerRankSlotStateError, match=error):
        restored._restore_dynamic_optimizer("student", state)


@pytest.mark.parametrize("operation", ("load", "step"))
def test_trainer_rank_rejects_mutating_slot_with_pending_graph(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    monkeypatch.setattr(trainer, "_slot_ref", lambda kind, name: _SlotRef(kind, name))
    target = _tracked_targets(trainer, ref, 2)[0]
    guard = (
        (lambda: trainer._guard_slot_can_load(ref))  # type: ignore[arg-type]
        if operation == "load"
        else (lambda: trainer._guard_checkpoint_can_step("teacher"))
    )

    with pytest.raises(TrainerRankSlotStateError, match="Cannot"):
        guard()

    target.sum().backward()
    guard()


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
    target = _tracked_targets(trainer, ref, 2)[0]

    target.sum().backward(retain_graph=True)
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    target.sum().backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_tracks_each_independent_output_graph() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    first, second = _tracked_targets(trainer, ref, 2, 3)

    first.sum().backward()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    second.sum().backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_tracks_graph_after_output_is_replaced_by_loss() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    target = _tracked_targets(trainer, ref, 2)[0]
    loss = target.sum()
    del target
    gc.collect()

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]

    loss.backward()
    trainer._guard_slot_can_load(ref)  # type: ignore[arg-type]


def test_trainer_rank_releases_abandoned_output_graph() -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    ref = _SlotRef("checkpoint", "teacher")
    target = _tracked_targets(trainer, ref, 2)[0]
    del target
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
    _stub_forward(monkeypatch, trainer, _indexed_outputs)
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
    _stub_forward(monkeypatch, trainer, dp=(1, 2))

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(5)])
    )

    assert [batch.indices for batch in batches] == [(1,), (3,), ()]
    assert [len(batch.outputs) for batch in batches] == [1, 1, 0]


def test_forward_micro_batches_syncs_fit_decision_across_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    _stub_forward(monkeypatch, trainer, dp=(1, 2), profiled=True)
    sync_flags: list[bool] = []

    def memory_check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        sync_flags.append(sync_across_dp)
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=1 << 30,
            fits=True,
        )

    monkeypatch.setattr(trainer, "_memory_check_required", memory_check)
    next(iter(trainer.forward_micro_batches([_target_request(i) for i in range(6)])))

    assert sync_flags
    assert all(sync_flags)


def test_forward_micro_batches_supports_arbitrary_nested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    _stub_forward(monkeypatch, trainer, _indexed_outputs, profiled=True)
    expected = [
        [[[[[_target_request(1)]]]]],
        [[[[[_target_request(3), _target_request(5)]]]]],
    ]
    nested = [(child for child in item) for item in expected]

    batches = list(cast(Any, trainer).forward_micro_batches(nested))

    assert batches[0].inputs == expected
    assert _output_shape(batches[0].outputs) == [
        [[[[["output"]]]]],
        [[[[["output", "output"]]]]],
    ]
    assert _output_values(batches[0].outputs) == [0, 1, 2]


def test_forward_micro_batches_ramps_after_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]

    def run(plan, **_kwargs):
        trainer._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=0.0,
            packed_tokens=plan.packed_tokens,
        )
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ]

    _stub_forward(monkeypatch, trainer, run)

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


def test_forward_micro_batches_tail_does_not_reset_stable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())  # type: ignore[arg-type]
    trainer._last_global_micro_batch_size = 64
    _stub_forward(monkeypatch, trainer, profiled=True)
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
    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(130)])
    )

    assert [batch.stats.global_count for batch in batches] == [64, 64, 2]
    assert trainer._last_global_micro_batch_size == 64


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

    monkeypatch.setattr(trainer_rank.dist, "is_initialized", lambda: False)
    _stub_forward(monkeypatch, trainer, dp=(1, 2))
    invalid = ForwardInput(
        input_tokens=torch.tensor([1, 2]), target_tokens=torch.tensor([1, 2, 3])
    )
    with pytest.raises(ValueError, match="target_tokens"):
        next(iter(trainer.forward_micro_batches([invalid, _target_request(1)])))


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
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    labels = torch.stack((tokens, tokens + 1), dim=-1)

    request = ForwardInput(
        input_tokens=tokens,
        target_tokens=labels,
        top_k=5,
        logits=True,
        hidden_states=True,
    )
    plan = trainer._plan_flat_forward([request])
    estimate = trainer._estimate_flat_forward([request])

    target_bytes = 3 * 2 * 4
    topk_bytes = 3 * 5 * (4 + 8)
    logits_bytes = 3 * 10 * 4
    hidden_bytes = 3 * 4 * 4
    assert estimate is not None and estimate[0] == plan.packed_tokens
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
