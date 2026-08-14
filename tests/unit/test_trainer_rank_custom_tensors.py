from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, TypeVar, cast

import pytest
import torch

from art.trainer_rank import (
    AdamParams,
    TrainerRank,
    TrainerRankSlotStateError,
    Unset,
)
from art.trainer_rank._checkpoint import (
    _file_digest,
    _manifest_digest,
    materialize_lora,
    prepare_checkpoint,
)
from art.trainer_rank._impl import _CheckpointSlot

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


class _CustomTensorAPI(Protocol):
    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: str | object = Unset,
    ) -> ModuleT: ...

    def parameter(
        self,
        name: str,
        factory: Callable[[], torch.Tensor | torch.nn.Parameter],
        *,
        checkpoint: str | object = Unset,
    ) -> torch.nn.Parameter: ...

    def buffer(
        self,
        name: str,
        factory: Callable[[], torch.Tensor],
        *,
        checkpoint: str | object = Unset,
    ) -> torch.Tensor: ...


class _ClassFactoryHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


class _ValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, 1, dtype=dtype)
        self.register_buffer("offset", torch.arange(1, dtype=dtype))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden) + self.offset


def _runtime(model: torch.nn.Module | None = None) -> Any:
    return SimpleNamespace(
        model=[model or torch.nn.Linear(1, 1)],
        optimizer=None,
        provider=SimpleNamespace(
            hidden_size=4,
            num_layers=1,
            kv_channels=2,
            art_flex_sliding_windows=(16,),
        ),
        model_support_handler=SimpleNamespace(
            build_gdn_execution_spec=True,
            canonicalize_loaded_lora_state=lambda state, _model: state,
            from_vllm_lora_tensors=lambda state, **_kwargs: state,
            to_vllm_lora_tensors=lambda state, **kwargs: (
                state,
                kwargs["adapter_config"],
            ),
            zero_internal_padding_grads=lambda _model: None,
            zero_internal_padding_params=lambda _model: None,
        ),
        rank=0,
        world_size=1,
    )


def _config() -> dict[str, object]:
    return {
        "base_model_name_or_path": "test/model",
        "r": 2,
        "lora_alpha": 2,
        "target_modules": ["q_proj"],
    }


def _trainer(*names: str) -> tuple[TrainerRank, _CustomTensorAPI]:
    trainer = TrainerRank(_runtime())
    for name in names:
        trainer._checkpoint_slots[name] = _CheckpointSlot(config=cast(Any, _config()))
    return trainer, cast(_CustomTensorAPI, trainer)


def test_module_accepts_class_and_lambda_factories_and_is_idempotent() -> None:
    _trainer_rank, rank = _trainer("student")
    class_calls = 0
    lambda_calls = 0

    class CountingHead(_ClassFactoryHead):
        def __init__(self) -> None:
            nonlocal class_calls
            class_calls += 1
            super().__init__()

    def value_head() -> _ValueHead:
        nonlocal lambda_calls
        lambda_calls += 1
        return _ValueHead(3)

    class_head = rank.module("class_head", CountingHead, checkpoint="student")
    lambda_head = rank.module("lambda_head", value_head, checkpoint="student")

    assert isinstance(class_head, CountingHead)
    assert isinstance(lambda_head, _ValueHead)
    assert rank.module("class_head", CountingHead, checkpoint="student") is class_head
    assert rank.module("lambda_head", value_head, checkpoint="student") is lambda_head
    assert class_calls == 1
    assert lambda_calls == 1


def test_checkpoint_resolution_uses_explicit_then_pushed_then_default() -> None:
    trainer, rank = _trainer("student", "teacher")
    trainer._set_default_slot(trainer._slot_ref("student"))

    student = rank.parameter("bias", lambda: torch.tensor([1.0]))
    with trainer.push_checkpoint("teacher"):
        teacher = rank.parameter("bias", lambda: torch.tensor([2.0]))
        assert rank.parameter("bias", lambda: torch.tensor([9.0])) is teacher
        assert (
            rank.parameter("bias", lambda: torch.tensor([9.0]), checkpoint="student")
            is student
        )

    assert rank.parameter("bias", lambda: torch.tensor([9.0])) is student
    assert teacher is not student

    _unselected_trainer, unselected = _trainer("student")
    with pytest.raises(
        TrainerRankSlotStateError, match="active.*checkpoint|checkpoint"
    ):
        unselected.buffer("missing_default", lambda: torch.zeros(1))
    with pytest.raises(TrainerRankSlotStateError, match="missing|loaded checkpoint"):
        rank.buffer("missing_slot", lambda: torch.zeros(1), checkpoint="missing")


def test_registration_is_idempotent_within_kind_and_rejects_kind_collisions() -> None:
    _trainer_rank, rank = _trainer("student")
    calls = 0

    def parameter() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones(2)

    first = rank.parameter("shared", parameter, checkpoint="student")
    assert rank.parameter("shared", parameter, checkpoint="student") is first
    assert calls == 1

    for register in (
        lambda: rank.buffer("shared", lambda: torch.zeros(2), checkpoint="student"),
        lambda: rank.module("shared", _ClassFactoryHead, checkpoint="student"),
    ):
        with pytest.raises(TrainerRankSlotStateError, match="shared.*parameter|kind"):
            register()


def test_registration_moves_to_rank_device_without_changing_factory_dtype() -> None:
    trainer, rank = _trainer("student")
    head = rank.module(
        "head",
        lambda: _ValueHead(3, dtype=torch.float64),
        checkpoint="student",
    )
    parameter = rank.parameter(
        "temperature",
        lambda: torch.tensor(0.5, dtype=torch.float64),
        checkpoint="student",
    )
    buffer = rank.buffer(
        "count",
        lambda: torch.tensor(3, dtype=torch.int64),
        checkpoint="student",
    )

    assert next(head.parameters()).device == trainer.device
    assert next(head.parameters()).dtype == torch.float64
    assert parameter.device == trainer.device
    assert parameter.dtype == torch.float64
    assert buffer.device == trainer.device
    assert buffer.dtype == torch.int64


@pytest.mark.parametrize(
    ("register", "message"),
    (
        (lambda rank: rank.module("bad", lambda: torch.ones(1)), "Module"),
        (lambda rank: rank.parameter("bad", lambda: object()), "Tensor"),
        (lambda rank: rank.buffer("bad", lambda: object()), "Tensor"),
    ),
)
def test_invalid_factories_do_not_partially_register(
    register: Callable[[Any], object], message: str
) -> None:
    trainer, rank = _trainer("student")
    with pytest.raises(TypeError, match=message):
        with trainer.push_checkpoint("student"):
            register(rank)
    assert trainer._checkpoint_slots["student"].custom == {}


def test_frozen_module_parameters_are_persisted_but_not_optimized() -> None:
    trainer, rank = _trainer("student")

    def factory() -> _ValueHead:
        head = _ValueHead(3)
        head.proj.weight.requires_grad_(False)
        return head

    head = rank.module("head", factory, checkpoint="student")
    slot = trainer._checkpoint_slots["student"]
    assert tuple(slot.params) == (head.proj.bias,)
    assert set(head.state_dict()) == {"proj.weight", "proj.bias", "offset"}


def test_custom_module_outputs_participate_in_checkpoint_graph_guards() -> None:
    trainer, rank = _trainer("student")
    head = rank.module("head", lambda: _ValueHead(3), checkpoint="student")
    output = head.proj(torch.ones(1, 3, requires_grad=True))
    ref = trainer._slot_ref("student")

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)

    output.sum().backward()
    trainer.zero_grad()
    trainer._guard_slot_can_load(ref)

    with torch.no_grad():
        detached = head(torch.tensor([[4.0, 5.0, 6.0]]))
    assert not detached.requires_grad
    trainer._guard_slot_can_load(ref)


def test_custom_module_graph_tracking_allows_in_place_layers() -> None:
    _trainer_rank, rank = _trainer("student")
    head = rank.module(
        "head",
        lambda: torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.ReLU(inplace=True),
        ),
        checkpoint="student",
    )

    output = head(torch.ones(1, 3, requires_grad=True))
    output.sum().backward()

    assert head[0].weight.grad is not None


def test_custom_parameter_outputs_participate_in_checkpoint_graph_guards() -> None:
    trainer, rank = _trainer("student")
    parameter = rank.parameter(
        "temperature", lambda: torch.tensor(2.0), checkpoint="student"
    )
    output = parameter * torch.tensor(3.0, requires_grad=True)
    ref = trainer._slot_ref("student")

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)

    output.backward()
    trainer.zero_grad()
    trainer._guard_slot_can_load(ref)


def test_checkpoint_load_rejects_custom_grads_and_stale_objects() -> None:
    trainer, rank = _trainer("student")
    head = rank.module("head", _ClassFactoryHead, checkpoint="student")
    parameter = rank.parameter(
        "temperature", lambda: torch.tensor(2.0), checkpoint="student"
    )
    (head(torch.ones(1)) + parameter).sum().backward()
    ref = trainer._slot_ref("student")

    with pytest.raises(TrainerRankSlotStateError, match="accumulated gradients"):
        trainer._guard_slot_can_load(ref)

    trainer.zero_grad()
    trainer._guard_slot_can_load(ref)
    trainer._checkpoint_slots["student"] = _CheckpointSlot(config=cast(Any, _config()))
    with pytest.raises(TrainerRankSlotStateError, match="is stale"):
        head(torch.ones(1))
    with pytest.raises(TrainerRankSlotStateError, match="is stale"):
        parameter * 2


def test_no_grad_custom_objects_do_not_create_graph_guards() -> None:
    trainer, rank = _trainer("student")
    head = rank.module("head", _ClassFactoryHead, checkpoint="student")
    parameter = rank.parameter(
        "temperature", lambda: torch.tensor(2.0), checkpoint="student"
    )
    with torch.no_grad():
        output = head(torch.tensor([3.0])) + torch.mul(
            input=torch.tensor([4.0]), other=parameter
        )
    assert not output.requires_grad
    trainer._guard_slot_can_load(trainer._slot_ref("student"))


def test_selected_optimizer_step_updates_only_its_custom_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, rank = _trainer("A", "B")
    a = rank.parameter("gain", lambda: torch.tensor(1.0), checkpoint="A")
    b = rank.parameter("gain", lambda: torch.tensor(2.0), checkpoint="B")
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(
            torch.zeros_like(param, dtype=torch.float32)
            if param.grad is None
            else param.grad.float()
            for param in params
        ),
    )
    (a * 3).backward()
    before_b = b.detach().clone()
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0),
        checkpoints=["A"],
    )
    assert not torch.equal(a, torch.tensor(1.0))
    torch.testing.assert_close(b, before_b, atol=0, rtol=0)


def test_parameter_can_be_added_after_checkpoint_optimizer_exists() -> None:
    trainer, rank = _trainer("student")
    slot = trainer._checkpoint_slots["student"]
    existing = torch.nn.Parameter(torch.tensor(1.0))
    trainer._tag_custom_parameters((existing,))
    slot.params = (existing,)
    slot.optimizer = trainer._new_dynamic_optimizer(
        "student", AdamParams(learning_rate=1e-3)
    )

    added = rank.parameter("added", lambda: torch.tensor(2.0), checkpoint="student")

    assert slot.params == (existing, added)
    assert slot.optimizer is not None
    assert len(slot.optimizer.master_params) == 2
    assert len(slot.optimizer.optimizer.param_groups) == 2
    assert slot.optimizer.optimizer.param_groups[0]["lr"] == pytest.approx(
        slot.optimizer.optimizer.param_groups[1]["lr"]
    )


def test_failed_optimizer_extension_does_not_partially_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, rank = _trainer("student")
    slot = trainer._checkpoint_slots["student"]
    existing = torch.nn.Parameter(torch.tensor(1.0))
    trainer._tag_custom_parameters((existing,))
    slot.params = (existing,)
    slot.optimizer = trainer._new_dynamic_optimizer(
        "student", AdamParams(learning_rate=1e-3)
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("invalid state")

    monkeypatch.setattr(trainer, "_extend_dynamic_optimizer", fail)

    with pytest.raises(RuntimeError, match="invalid state"):
        rank.parameter("added", lambda: torch.tensor(2.0), checkpoint="student")

    assert "added" not in slot.custom
    assert slot.params == (existing,)
    assert slot.optimizer is not None
    assert len(slot.optimizer.master_params) == 1


def test_custom_parameters_join_slot_optimizer_with_replicated_metadata() -> None:
    trainer, rank = _trainer("student")
    head = rank.module("head", lambda: _ValueHead(3), checkpoint="student")
    standalone = rank.parameter(
        "temperature", lambda: torch.tensor(1.0), checkpoint="student"
    )
    buffer = rank.buffer("running_mean", lambda: torch.zeros(3), checkpoint="student")

    expected = (*head.parameters(), standalone)
    slot = trainer._checkpoint_slots["student"]
    assert slot.params == expected
    assert all(bool(getattr(param, "allreduce")) for param in expected)
    assert all(getattr(param, "grad_sync_domain") == "tp_default" for param in expected)
    assert all(getattr(param, "grad_sync_op") == "avg" for param in expected)
    assert all(buffer is not param for param in slot.params)
    assert all(
        not mask.any() for mask in trainer._dynamic_optimizer_padding_masks("student")
    )


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
@pytest.mark.parametrize("payload", ("custom", "optimizer"))
def test_custom_checkpoint_payload_keys_are_strictly_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    trainer, rank = _real_lora_trainer()
    head, temperature, _buffer = _register_custom_tensors(rank)
    _step_custom_tensors(trainer, head, temperature, monkeypatch)
    output = tmp_path / "checkpoint"
    trainer.save_checkpoint(str(output), "student")

    from safetensors.torch import load_file, save_file

    relative = (
        "custom_tensors.safetensors"
        if payload == "custom"
        else "optimizer/custom.safetensors"
    )
    tensors = load_file(output / relative)
    tensors.pop(next(iter(tensors)))
    save_file(tensors, output / relative)
    manifest = json.loads((output / "checkpoint.json").read_text())
    manifest["files"][relative] = _file_digest(output / relative)
    manifest["digest"] = _manifest_digest(manifest)
    (output / "checkpoint.json").write_text(json.dumps(manifest))

    expected = (
        "custom tensor payload" if payload == "custom" else "custom optimizer payload"
    )
    with pytest.raises(RuntimeError, match=expected):
        prepare_checkpoint(str(output))


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_custom_tensor_checkpoint_artifacts_and_lora_export_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, rank = _real_lora_trainer()
    head, temperature, _running_mean = _register_custom_tensors(rank)
    _step_custom_tensors(trainer, head, temperature, monkeypatch)

    output = tmp_path / "checkpoint"
    trainer.save_checkpoint(str(output), "student")

    from safetensors import safe_open

    with safe_open(output / "adapter_model.safetensors", framework="pt") as handle:
        assert all("value_head" not in key for key in handle.keys())
        assert all("temperature" not in key for key in handle.keys())
        assert all("running_mean" not in key for key in handle.keys())
    with safe_open(output / "custom_tensors.safetensors", framework="pt") as handle:
        assert set(handle.keys()) == {
            "running_mean",
            "temperature",
            "value_head.offset",
            "value_head.proj.bias",
            "value_head.proj.weight",
        }
    with safe_open(output / "optimizer/custom.safetensors", framework="pt") as handle:
        assert set(handle.keys()) == {
            f"{component}/{key}"
            for key in (
                "temperature",
                "value_head.proj.bias",
                "value_head.proj.weight",
            )
            for component in ("master", "exp_avg", "exp_avg_sq", "step")
        }

    manifest = json.loads((output / "checkpoint.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["custom_tensors"] == {
        "running_mean": {
            "kind": "buffer",
            "tensor_keys": ["running_mean"],
            "trainable_keys": [],
        },
        "temperature": {
            "kind": "parameter",
            "tensor_keys": ["temperature"],
            "trainable_keys": ["temperature"],
        },
        "value_head": {
            "kind": "module",
            "tensor_keys": [
                "value_head.offset",
                "value_head.proj.bias",
                "value_head.proj.weight",
            ],
            "trainable_keys": [
                "value_head.proj.bias",
                "value_head.proj.weight",
            ],
        },
    }
    assert "custom_tensors.safetensors" in manifest["files"]
    assert "optimizer/custom.safetensors" in manifest["files"]

    monkeypatch.setattr(
        "art.megatron.model_support.lora_disk.normalize_lora_checkpoint_to_vllm",
        lambda _path: None,
    )
    exported = tmp_path / "lora"
    materialize_lora(output, exported, require_optimizer=True)
    assert {path.name for path in exported.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_custom_tensors_and_optimizer_restore_lazily_and_survive_unmaterialized_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import load_file

    from art.trainer_rank import _checkpoint

    original, original_api = _real_lora_trainer()
    original_head, original_temperature, original_running_mean = (
        _register_custom_tensors(original_api)
    )
    original_running_mean.fill_(7)
    _step_custom_tensors(original, original_head, original_temperature, monkeypatch)
    saved = tmp_path / "saved"
    original.save_checkpoint(str(saved), "student")

    restored, restored_api = _empty_real_lora_trainer()
    _checkpoint.load_checkpoint(restored, prepare_checkpoint(str(saved)), "student")

    resaved = tmp_path / "resaved-before-registration"
    restored.save_checkpoint(str(resaved), "student")
    _assert_tensor_dict_equal(
        load_file(resaved / "custom_tensors.safetensors"),
        load_file(saved / "custom_tensors.safetensors"),
    )
    _assert_tensor_dict_equal(
        load_file(resaved / "optimizer/custom.safetensors"),
        load_file(saved / "optimizer/custom.safetensors"),
    )

    calls = 0

    def head_factory() -> _ValueHead:
        nonlocal calls
        calls += 1
        return _ValueHead(3)

    restored_head = restored_api.module(
        "value_head", head_factory, checkpoint="student"
    )
    restored_temperature = restored_api.parameter(
        "temperature", lambda: torch.tensor(-99.0), checkpoint="student"
    )
    restored_running_mean = restored_api.buffer(
        "running_mean", lambda: torch.full((3,), -99.0), checkpoint="student"
    )
    assert calls == 1
    for actual, expected in zip(
        restored_head.state_dict().values(),
        original_head.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        restored_temperature, original_temperature, atol=0, rtol=0
    )
    torch.testing.assert_close(
        restored_running_mean, original_running_mean, atol=0, rtol=0
    )

    for trainer, head, temperature in (
        (original, original_head, original_temperature),
        (restored, restored_head, restored_temperature),
    ):
        _step_custom_tensors(trainer, head, temperature, monkeypatch, scale=-0.5)
    for actual, expected in zip(
        restored._checkpoint_slots["student"].params,
        original._checkpoint_slots["student"].params,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    incompatible, incompatible_api = _empty_real_lora_trainer()
    _checkpoint.load_checkpoint(incompatible, prepare_checkpoint(str(saved)), "student")
    with pytest.raises(TrainerRankSlotStateError, match="shape/dtype"):
        incompatible_api.parameter(
            "temperature",
            lambda: torch.zeros(2, dtype=torch.float64),
            checkpoint="student",
        )


def _real_lora_trainer() -> tuple[TrainerRank, _CustomTensorAPI]:
    trainer, api = _empty_real_lora_trainer()
    adapter = {
        "layer.q_proj.lora_A.weight": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "layer.q_proj.lora_B.weight": torch.tensor(
            [[0.2, 0.1], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        ),
    }
    loaded = trainer._load_checkpoint_slot("student", adapter, alpha=2)
    params = trainer._validate_checkpoint_consistency("student", loaded, set(adapter))
    trainer._checkpoint_slots["student"] = _CheckpointSlot(params, cast(Any, _config()))
    return trainer, api


def _empty_real_lora_trainer() -> tuple[TrainerRank, _CustomTensorAPI]:
    from art.megatron.lora import LoRA

    lora = LoRA("layer.q_proj", 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    trainer = TrainerRank(_runtime(lora))
    return trainer, cast(_CustomTensorAPI, trainer)


def _register_custom_tensors(
    rank: _CustomTensorAPI,
) -> tuple[_ValueHead, torch.nn.Parameter, torch.Tensor]:
    head = rank.module("value_head", lambda: _ValueHead(3), checkpoint="student")
    temperature = rank.parameter(
        "temperature", lambda: torch.tensor(0.5), checkpoint="student"
    )
    running_mean = rank.buffer(
        "running_mean", lambda: torch.zeros(3), checkpoint="student"
    )
    return head, temperature, running_mean


def _step_custom_tensors(
    trainer: TrainerRank,
    head: _ValueHead,
    temperature: torch.nn.Parameter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scale: float = 1.0,
) -> None:
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(
            torch.zeros_like(param, dtype=torch.float32)
            if param.grad is None
            else param.grad.float()
            for param in params
        ),
    )
    hidden = torch.tensor([[0.25, -0.5, 1.0]])
    (head(hidden).sum() + temperature * scale).backward()
    trainer.optim_step(
        params=AdamParams(
            learning_rate=1e-3,
            beta1=0.8,
            beta2=0.95,
            weight_decay=0.0,
            grad_clip_norm=10.0,
        ),
        checkpoints=["student"],
    )


def _assert_tensor_dict_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert set(actual) == set(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
