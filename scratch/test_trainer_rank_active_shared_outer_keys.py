from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron import lora as lora_module
from art.megatron.lora import (
    EXPERT_TP_GRAD_SYNC_DOMAIN,
    GRAD_SYNC_OP_NONE,
    GRAD_SYNC_OP_SUM,
    LoRA,
    LoraFactor,
    LoRAParallelSpec,
    MoeLoraParameterization,
)
from art.trainer_rank import AdamParams, TrainerRank, _checkpoint
import art.trainer_rank._impl as trainer_rank_impl
from art.trainer_rank._impl import _CheckpointSlot

PREFIX = "model.layers.0.mlp.experts.{expert}"
EXPERTS = 3
IN_FEATURES = 4
OUT_FEATURES = 6


@pytest.fixture(autouse=True)
def _cpu_single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0,
    )

    def build_optimizer(
        params: Iterable[torch.nn.Parameter], config: AdamParams
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    monkeypatch.setattr(trainer_rank_impl, "_build_dynamic_optimizer", build_optimizer)


def _site(
    projection: str,
    shared_factor: LoraFactor,
    parameterization: MoeLoraParameterization,
    rank: int,
) -> LoRA:
    row = shared_factor == "B"
    return LoRA(
        adapter_model_prefix=f"{PREFIX}.{projection}",
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rank=rank,
        alpha=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=EXPERTS,
        moe_parameterization=parameterization,
        shared_factor=shared_factor,
        a_parallel_spec=LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=row,
            shard_dim=-2 if row else None,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_NONE if row else GRAD_SYNC_OP_SUM,
        ),
        b_parallel_spec=LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=not row,
            shard_dim=None if row else -1,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM if row else GRAD_SYNC_OP_NONE,
        ),
        allreduce=False,
    )


def _sites(parameterization: MoeLoraParameterization, rank: int) -> tuple[LoRA, LoRA]:
    return (
        _site("gate_proj", "A", parameterization, rank),
        _site("down_proj", "B", parameterization, rank),
    )


def _adapter(
    parameterization: MoeLoraParameterization,
    rank: int,
    targets: Sequence[str],
    *,
    offset: int,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for site_index, site in enumerate(_sites(parameterization, rank)):
        if not any(target in site.adapter_model_prefix for target in targets):
            continue
        with torch.no_grad():
            for factor_index, parameter in enumerate((site.A_T, site.B_T)):
                values = torch.arange(parameter.numel(), dtype=parameter.dtype)
                parameter.copy_(
                    values.reshape(parameter.shape)
                    + offset
                    + 1000 * site_index
                    + 100 * factor_index
                    + 1
                )
        state.update(site.sharded_lora_state_dict())
    return state


def _trainer(
    monkeypatch: pytest.MonkeyPatch,
    parameterization: MoeLoraParameterization = "per_expert",
) -> TrainerRank:
    model = torch.nn.Sequential(*_sites(parameterization, rank=2))
    runtime = SimpleNamespace(
        model=[model],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=4, num_layers=1),
        model_identifier="test/model",
        model_support_spec=None,
        model_support_handler=SimpleNamespace(
            canonicalize_loaded_lora_state=lambda state, _model: state,
            zero_internal_padding_grads=lambda _model: None,
            zero_internal_padding_params=lambda _model: None,
        ),
        rank=0,
        world_size=1,
    )
    trainer = TrainerRank(cast(Any, runtime))
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, *, scale_grads: tuple(
            (
                torch.zeros_like(parameter, dtype=torch.float32)
                if parameter.grad is None
                else parameter.grad.detach().float().mul(scale_grads)
            )
            for parameter in params
        ),
    )
    return trainer


def _config(
    parameterization: MoeLoraParameterization, rank: int, targets: Sequence[str]
) -> dict[str, object]:
    return {
        "base_model_name_or_path": "test/model",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": list(targets),
        "moe_parameterization": parameterization,
    }


def _install(
    trainer: TrainerRank,
    name: str,
    adapter: dict[str, torch.Tensor],
    config: dict[str, object],
) -> tuple[torch.nn.Parameter, ...]:
    loaded = trainer._load_checkpoint_slot(
        name, adapter, alpha=float(cast(int, config["lora_alpha"]))
    )
    params = trainer._validate_checkpoint_consistency(name, loaded, set(adapter))
    trainer._checkpoint_slots[name] = _CheckpointSlot(
        params,
        cast(Any, config),
    )
    trainer._validate_loaded_checkpoint_config(name, cast(Any, config))
    return params


def _active_sites(trainer: TrainerRank, name: str) -> list[tuple[LoRA, Any]]:
    ref = trainer._slot_ref(name)
    return [
        (module, slot)
        for chunk in trainer.runtime.model
        for module in chunk.modules()
        if isinstance(module, LoRA)
        if (slot := module._slot(ref)) is not None
    ]


def _adapter_state(trainer: TrainerRank, name: str) -> dict[str, torch.Tensor]:
    ref = trainer._slot_ref(name)
    return {
        key: value.detach().clone()
        for module, _slot in _active_sites(trainer, name)
        for key, value in module.sharded_lora_state_dict(ref).items()
    }


def _optimizer_state_by_key(
    trainer: TrainerRank, name: str
) -> dict[tuple[str, str], torch.Tensor]:
    slot = trainer._checkpoint_slots[name]
    dynamic = slot.optimizer
    assert dynamic is not None
    assert {id(param) for param in dynamic.master_params} == {
        id(param) for param in dynamic.optimizer.param_groups[0]["params"]
    }
    masters = {
        id(parameter): master
        for parameter, master in zip(slot.params, dynamic.master_params, strict=True)
    }
    result: dict[tuple[str, str], torch.Tensor] = {}
    ref = trainer._slot_ref(name)
    for module, _slot in _active_sites(trainer, name):
        for key, parameter, expert in module._export_items(ref):
            master = masters[id(parameter)]
            state = dynamic.optimizer.state[master]
            for component, value in (
                ("master", master),
                ("exp_avg", cast(torch.Tensor, state["exp_avg"])),
                ("exp_avg_sq", cast(torch.Tensor, state["exp_avg_sq"])),
            ):
                result[(key, component)] = (
                    (value if expert is None else value[expert]).detach().clone()
                )
            result[(key, "step")] = cast(torch.Tensor, state["step"]).clone()
    return result


def _assert_tensor_maps_equal(
    actual: dict[Any, torch.Tensor], expected: dict[Any, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def _set_grads(trainer: TrainerRank, name: str, offset: int) -> None:
    for index, parameter in enumerate(trainer._checkpoint_slots[name].params):
        parameter.grad = (
            torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(
                parameter.shape
            )
            + offset
            + 100 * index
            + 1
        )


def test_rank_target_and_parameterization_switching_is_slot_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(monkeypatch)
    cases: tuple[tuple[str, MoeLoraParameterization, int, tuple[str, ...]], ...] = (
        ("per_first", "per_expert", 2, ("gate_proj", "down_proj")),
        ("shared", "shared_outer", 3, ("gate_proj",)),
        ("per_second", "per_expert", 1, ("down_proj",)),
    )
    for index, (name, parameterization, rank, targets) in enumerate(cases):
        adapter = _adapter(parameterization, rank, targets, offset=10000 * index)
        params = _install(
            trainer,
            name,
            adapter,
            _config(parameterization, rank, targets),
        )
        trainer._set_default_slot(trainer._slot_ref(name))
        assert trainer._default_slot_ref == trainer._slot_ref(name)
        sites = _active_sites(trainer, name)
        assert len(sites) == len(targets)
        assert {slot.parameterization for _module, slot in sites} == {parameterization}
        assert {slot.rank for _module, slot in sites} == {rank}
        assert tuple(
            id(parameter)
            for parameter in trainer._iter_slot_parameters(trainer._slot_ref(name))
        ) == tuple(id(parameter) for parameter in params)
        assert {
            key for group in trainer._local_parameter_key_groups(name) for key in group
        } == set(adapter)

    shared = _adapter("shared_outer", 2, ("gate_proj",), offset=70000)
    per_expert = _adapter("per_expert", 2, ("gate_proj",), offset=80000)
    with pytest.raises(KeyError, match="Incomplete or mixed"):
        trainer._load_checkpoint_slot("mixed", {**shared, **per_expert}, alpha=2)
    incomplete = dict(shared)
    incomplete.pop(next(key for key in incomplete if ".2." in key))
    with pytest.raises(KeyError, match="Incomplete or mixed"):
        trainer._load_checkpoint_slot("incomplete", incomplete, alpha=2)

    mixed_targets = {
        **_adapter("shared_outer", 2, ("gate_proj",), offset=90000),
        **_adapter("per_expert", 2, ("down_proj",), offset=100000),
    }
    with pytest.raises(ValueError, match="loaded weights use"):
        _install(
            _trainer(monkeypatch),
            "mixed_targets",
            mixed_targets,
            _config("shared_outer", 2, ("gate_proj", "down_proj")),
        )


def test_shared_outer_checkpoint_and_optimizer_roundtrip_are_key_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "run"
    targets = ("gate_proj", "down_proj")
    adapter = _adapter("shared_outer", 3, targets, offset=90000)
    adam = AdamParams(
        learning_rate=3e-4,
        beta1=0.8,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip_norm=0.0,
    )
    original = _trainer(monkeypatch, "per_expert")
    _install(original, name, adapter, _config("shared_outer", 3, targets))
    _set_grads(original, name, 10)
    original.optim_step(params=adam, checkpoints=[name])
    expected_adapter = _adapter_state(original, name)
    expected_optimizer = _optimizer_state_by_key(original, name)

    output = tmp_path / "checkpoint"
    original.save_checkpoint(str(output), name)
    prepared = _checkpoint.prepare_checkpoint(str(output))
    assert set(prepared.keys) == set(expected_adapter)

    restored = _trainer(monkeypatch, "per_expert")
    _checkpoint.load_checkpoint(restored, prepared, name)
    assert {
        slot.parameterization for _module, slot in _active_sites(restored, name)
    } == {"shared_outer"}
    _assert_tensor_maps_equal(_adapter_state(restored, name), expected_adapter)
    _assert_tensor_maps_equal(
        _optimizer_state_by_key(restored, name), expected_optimizer
    )

    for trainer in (original, restored):
        _set_grads(trainer, name, 20)
        trainer.optim_step(params=adam, checkpoints=[name])
    _assert_tensor_maps_equal(
        _adapter_state(restored, name), _adapter_state(original, name)
    )
    _assert_tensor_maps_equal(
        _optimizer_state_by_key(restored, name),
        _optimizer_state_by_key(original, name),
    )
