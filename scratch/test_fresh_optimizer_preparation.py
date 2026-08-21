from __future__ import annotations

from collections.abc import Sequence
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
    LoRASlotRef,
    MoeLoraParameterization,
)
from art.trainer_rank import AdamParams, TrainerRank, _impl
from art.trainer_rank._checkpoint import (
    PreparedCheckpoint,
    PreparedCheckpointSlotInstall,
)
from art.trainer_rank._impl import _CheckpointSlot

PREFIX = "model.layers.0.mlp.experts.{expert}"
EXPERTS = 3
IN_FEATURES = 4
OUT_FEATURES = 6


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0,
    )


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
        dtype=torch.bfloat16,
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
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for site_index, site in enumerate(_sites(parameterization, rank)):
        if not any(target in site.adapter_model_prefix for target in targets):
            continue
        with torch.no_grad():
            for factor_index, parameter in enumerate((site.A_T, site.B_T)):
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(values + 1000 * site_index + 100 * factor_index + 1)
        state.update(site.sharded_lora_state_dict())
    return state


def _prepared_checkpoint(
    parameterization: MoeLoraParameterization,
    rank: int,
    targets: Sequence[str],
) -> tuple[TrainerRank, PreparedCheckpointSlotInstall]:
    modules = _sites("per_expert", rank=2)
    model = torch.nn.Sequential(*modules)
    trainer = object.__new__(TrainerRank)
    trainer.runtime = cast(
        Any,
        SimpleNamespace(
            model=[model],
            model_support_handler=SimpleNamespace(
                canonicalize_loaded_lora_state=lambda state, _model: state
            ),
        ),
    )
    adapter = _adapter(parameterization, rank, targets)
    ref = LoRASlotRef("checkpoint", "run")
    sites = tuple(
        (module, slot)
        for module in modules
        if (
            slot := module.prepare_lora_slot(
                ref, adapter, alpha=rank, requires_grad=True
            )
        )
        is not None
    )
    config = {
        "base_model_name_or_path": "test/model",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": list(targets),
        "moe_parameterization": parameterization,
    }
    source = PreparedCheckpoint(
        path=Path("."),
        config=config,
        keys=tuple(adapter),
        manifest=None,
        digest="test",
    )
    return trainer, PreparedCheckpointSlotInstall(
        name="run",
        source=source,
        config=config,
        sites=sites,
        expected_keys=frozenset(adapter),
    )


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes()


def _storage_nbytes(tensors: Sequence[torch.Tensor]) -> int:
    storages = {_storage_key(tensor) for tensor in tensors}
    return sum(nbytes for _device, _index, _pointer, nbytes in storages)


@pytest.mark.parametrize(
    ("parameterization", "rank", "targets"),
    (
        ("per_expert", 1, ("gate_proj",)),
        ("shared_outer", 3, ("gate_proj", "down_proj")),
    ),
)
def test_fresh_optimizer_is_complete_exact_cpu_payload(
    monkeypatch: pytest.MonkeyPatch,
    parameterization: MoeLoraParameterization,
    rank: int,
    targets: tuple[str, ...],
) -> None:
    trainer, checkpoint = _prepared_checkpoint(parameterization, rank, targets)
    monkeypatch.setattr(
        _impl,
        "_build_dynamic_optimizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh preparation constructed an optimizer")
        ),
    )
    cuda_initialized = torch.cuda.is_initialized()

    prepared = trainer.prepare_fresh_checkpoint_slot_optimizer_for_residency(checkpoint)

    assert torch.cuda.is_initialized() == cuda_initialized
    assert len(prepared.master_params) == len(checkpoint.parameters)
    assert len(prepared.state) == len(checkpoint.parameters)
    assert prepared.valid_ranges == tuple(None for _ in checkpoint.parameters)
    for parameter, master, values in zip(
        checkpoint.parameters,
        prepared.master_params,
        prepared.state,
        strict=True,
    ):
        assert master.device.type == "cpu"
        assert master.dtype == torch.float32
        assert master.requires_grad
        assert _storage_key(master) != _storage_key(parameter)
        torch.testing.assert_close(master, parameter.float())
        assert set(values) == {"step", "exp_avg", "exp_avg_sq"}
        step = cast(torch.Tensor, values["step"])
        exp_avg = cast(torch.Tensor, values["exp_avg"])
        exp_avg_sq = cast(torch.Tensor, values["exp_avg_sq"])
        assert step.shape == () and step.dtype == torch.float32
        assert exp_avg.shape == exp_avg_sq.shape == master.shape
        assert exp_avg.dtype == exp_avg_sq.dtype == torch.float32
        assert not torch.count_nonzero(step)
        assert not torch.count_nonzero(exp_avg)
        assert not torch.count_nonzero(exp_avg_sq)
    assert prepared.param_group == {
        "lr": 0.0,
        "bias_correction": True,
        "betas": (0.9, 0.99),
        "eps": 1e-13,
        "weight_decay": 0.1,
    }

    tensor_storage_keys = tuple(_storage_key(tensor) for tensor in prepared.tensors)
    assert len(set(tensor_storage_keys)) == len(tensor_storage_keys)
    parameter_elements = sum(param.numel() for param in checkpoint.parameters)
    steps = tuple(cast(torch.Tensor, values["step"]) for values in prepared.state)
    scalar_bytes = _storage_nbytes(steps)
    assert scalar_bytes == 4 * len(checkpoint.parameters)
    assert _storage_nbytes(prepared.tensors) == 12 * parameter_elements + scalar_bytes

    snapshot = prepared.snapshot_source()
    optimizer = cast(dict[str, object], snapshot.state["optimizer"])
    state = cast(dict[int, dict[str, object]], optimizer["state"])
    groups = cast(list[dict[str, object]], optimizer["param_groups"])
    assert len(state) == len(checkpoint.parameters)
    assert all(
        set(values) == {"step", "exp_avg", "exp_avg_sq"} for values in state.values()
    )
    assert groups == [{**prepared.param_group, "params": list(range(len(state)))}]


def test_first_step_reuses_every_prepared_optimizer_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, checkpoint = _prepared_checkpoint(
        "shared_outer", rank=3, targets=("gate_proj", "down_proj")
    )
    prepared = trainer.prepare_fresh_checkpoint_slot_optimizer_for_residency(checkpoint)
    for module, slot in checkpoint.sites:
        module.install_lora_slot(slot)
    trainer._checkpoint_slots = {
        "run": _CheckpointSlot(
            params=checkpoint.parameters,
            config=cast(Any, checkpoint.config),
        )
    }
    assert trainer._dynamic_optimizer_layout("run") == prepared.layout

    optimizer = torch.optim.AdamW(prepared.master_params, lr=9.0)
    trainer._bind_prepared_checkpoint_slot_optimizer("run", prepared, optimizer)
    optimizer.state.default_factory = lambda: (_ for _ in ()).throw(
        AssertionError("optimizer lazily allocated state")
    )
    monkeypatch.setattr(
        trainer,
        "_new_dynamic_optimizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("first step created a dynamic optimizer")
        ),
    )
    monkeypatch.setattr(trainer, "_prune_slot_graphs", lambda *_args: None)
    monkeypatch.setattr(trainer, "_guard_checkpoint_can_step", lambda *_args: None)
    tensor_ids = tuple(map(id, prepared.tensors))
    storage_keys = tuple(map(_storage_key, prepared.tensors))
    params = AdamParams(
        learning_rate=0.25,
        beta1=0.7,
        beta2=0.8,
        eps=1e-7,
        weight_decay=0.05,
        grad_clip_norm=0.0,
    )

    result = trainer.optim_step_reduced(
        "run",
        params=params,
        grads=tuple(
            torch.ones_like(param, dtype=torch.float32)
            for param in checkpoint.parameters
        ),
    )

    assert result["update_successful"] == 1.0
    live = trainer.checkpoint_slot_residency_tensors("run").optimizer
    assert tuple(map(id, live)) == tensor_ids
    assert tuple(map(_storage_key, live)) == storage_keys
    assert optimizer.param_groups[0]["lr"] == params.learning_rate
    assert optimizer.param_groups[0]["betas"] == (params.beta1, params.beta2)
    assert optimizer.param_groups[0]["eps"] == params.eps
    assert optimizer.param_groups[0]["weight_decay"] == params.weight_decay
    assert all(
        optimizer.state[master] is values
        for master, values in zip(prepared.master_params, prepared.state, strict=True)
    )
