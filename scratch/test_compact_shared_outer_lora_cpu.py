from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import art.megatron.lora as lora_module
from art.megatron.lora import (
    EXPERT_TP_GRAD_SYNC_DOMAIN,
    GRAD_SYNC_OP_NONE,
    GRAD_SYNC_OP_SUM,
    LoRA,
    LoraFactor,
    LoRAParallelSpec,
    LoRAPublishPlanner,
    LoRASlotRef,
    MoeLoraParameterization,
)

PREFIX = "base_model.model.model.layers.0.mlp.experts.{expert}"
EXPERTS = 3
IN_FEATURES = 4
OUT_FEATURES = 6
RANK = 2


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0,
    )


def _make_lora(
    shared_factor: LoraFactor,
    parameterization: MoeLoraParameterization,
) -> LoRA:
    row = shared_factor == "B"
    a_spec = LoRAParallelSpec(
        shard_domain="expert_tp",
        sharded=row,
        shard_dim=-2 if row else None,
        grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_NONE if row else GRAD_SYNC_OP_SUM,
    )
    b_spec = LoRAParallelSpec(
        shard_domain="expert_tp",
        sharded=not row,
        shard_dim=None if row else -1,
        grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_SUM if row else GRAD_SYNC_OP_NONE,
    )
    projection = "gate_proj" if shared_factor == "A" else "down_proj"
    return LoRA(
        adapter_model_prefix=f"{PREFIX}.{projection}",
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rank=RANK,
        alpha=RANK,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=EXPERTS,
        moe_parameterization=parameterization,
        shared_factor=shared_factor,
        a_parallel_spec=a_spec,
        b_parallel_spec=b_spec,
        allreduce=False,
    )


def _fill(lora: LoRA, offset: int = 0) -> None:
    with torch.no_grad():
        for index, param in enumerate((lora.A_T, lora.B_T)):
            values = torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape)
            param.copy_(values + offset + index * 100 + 1)


def test_parameterization_requires_explicit_typed_provider_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ART_MEGATRON_LORA_MOE_PARAMETERIZATION", "shared_outer")
    with pytest.raises(ValueError, match="must be explicitly set"):
        lora_module._configured_lora_moe_parameterization(SimpleNamespace())
    provider = SimpleNamespace(_art_lora_moe_parameterization="shared_outer")
    assert lora_module._configured_lora_moe_parameterization(provider) == "shared_outer"


def test_per_expert_positional_constructor_contract_is_unchanged() -> None:
    spec = LoRAParallelSpec(shard_domain="expert_tp")
    lora = LoRA(
        PREFIX,
        IN_FEATURES,
        OUT_FEATURES,
        RANK,
        RANK,
        torch.float32,
        torch.device("cpu"),
        EXPERTS,
        spec,
        spec,
        False,
    )
    assert tuple(lora.A_T.shape) == (EXPERTS, IN_FEATURES, RANK)
    assert tuple(lora.B_T.shape) == (EXPERTS, RANK, OUT_FEATURES)
    assert not getattr(lora.A_T, "allreduce")
    assert not getattr(lora.B_T, "allreduce")


def _assert_roundtrip(source: LoRA, target: LoRA, ref: LoRASlotRef) -> None:
    expected = source.sharded_lora_state_dict()
    assert target.load_lora_slot(ref, expected, alpha=RANK, requires_grad=True)
    slot = target._slot(ref)
    assert slot is not None
    assert slot.parameterization == source.moe_parameterization
    assert tuple(slot.A_T.shape) == tuple(source.A_T.shape)
    assert tuple(slot.B_T.shape) == tuple(source.B_T.shape)
    assert getattr(slot.A_T, "lora_is_expert") == (slot.A_T.ndim == 3)
    assert getattr(slot.B_T, "lora_is_expert") == (slot.B_T.ndim == 3)
    assert getattr(slot.A_T, "lora_moe_parameterization") == source.moe_parameterization
    assert getattr(slot.B_T, "lora_moe_parameterization") == source.moe_parameterization
    assert getattr(slot.A_T, "allreduce") == (slot.A_T.ndim == 2)
    assert getattr(slot.B_T, "allreduce") == (slot.B_T.ndim == 2)

    state = target.sharded_lora_state_dict(ref)
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[key], expected[key]) for key in state)
    active_keys = {
        key
        for suffix, param in target._lora_params(ref)
        for key in target._expected_weight_keys_for_param(
            suffix.removesuffix(".weight"),
            param,
        )
    }
    assert active_keys == set(state)
    manifest = target.sharded_lora_manifest(ref)
    assert manifest.keys() == state.keys()
    shared_suffix = "lora_A" if source.shared_factor == "A" else "lora_B"
    shared_keys = [
        key for key in state if ".shared." in key and f".{shared_suffix}." in key
    ]
    assert len(shared_keys) == (source.moe_parameterization == "shared_outer")
    if shared_keys:
        assert manifest[shared_keys[0]]["domain"] == "expert_tp"

    for param in (slot.A_T, slot.B_T):
        setattr(
            param,
            "main_grad",
            torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) + 1,
        )
    grads = target.sharded_lora_grad_dict(ref)
    assert grads.keys() == state.keys()
    assert {key: tuple(value.shape) for key, value in grads.items()} == {
        key: tuple(value.shape) for key, value in state.items()
    }

    metadata = LoRAPublishPlanner(
        [torch.nn.Sequential(target)], slot_ref=ref
    ).global_metadata({key: value.dtype for key, value in state.items()})
    by_key = {item.key: item for item in metadata}
    assert by_key.keys() == state.keys()
    assert all(by_key[key].shape == tuple(state[key].shape) for key in state)


@pytest.mark.parametrize(
    ("shared_factor", "a_shape", "b_shape", "shared_suffix"),
    (
        ("A", (IN_FEATURES, RANK), (EXPERTS, RANK, OUT_FEATURES), "lora_A"),
        ("B", (EXPERTS, IN_FEATURES, RANK), (RANK, OUT_FEATURES), "lora_B"),
    ),
)
def test_compact_shapes_metadata_layout_and_exact_keys(
    shared_factor: LoraFactor,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    shared_suffix: str,
) -> None:
    lora = _make_lora(shared_factor, "shared_outer")
    assert tuple(lora.A_T.shape) == a_shape
    assert tuple(lora.B_T.shape) == b_shape
    assert lora.num_local_experts == EXPERTS
    assert getattr(lora.A_T, "lora_shard_domain") == "expert_tp"
    assert getattr(lora.B_T, "lora_shard_domain") == "expert_tp"
    assert getattr(lora.A_T, "grad_sync_domain") == EXPERT_TP_GRAD_SYNC_DOMAIN
    assert getattr(lora.B_T, "grad_sync_domain") == EXPERT_TP_GRAD_SYNC_DOMAIN
    assert getattr(lora.A_T, "grad_sync_op") == (
        GRAD_SYNC_OP_SUM if shared_factor == "A" else GRAD_SYNC_OP_NONE
    )
    assert getattr(lora.B_T, "grad_sync_op") == (
        GRAD_SYNC_OP_SUM if shared_factor == "B" else GRAD_SYNC_OP_NONE
    )
    assert getattr(lora.A_T, "allreduce") == (shared_factor == "A")
    assert getattr(lora.B_T, "allreduce") == (shared_factor == "B")

    lora.A_T.data.fill_(1)
    lora.B_T.data.fill_(1)
    lora.bind_expert_layout((2, None, 0), (2, None, 0))
    shared_param = lora.A_T if shared_factor == "A" else lora.B_T
    expert_param = lora.B_T if shared_factor == "A" else lora.A_T
    assert torch.count_nonzero(shared_param) == shared_param.numel()
    assert torch.count_nonzero(expert_param[1]) == 0

    state = lora.sharded_lora_state_dict()
    shared_keys = [key for key in state if f".{shared_suffix}.weight" in key]
    assert shared_keys == [
        f"{PREFIX.replace('{expert}', 'shared')}."
        f"{'gate_proj' if shared_factor == 'A' else 'down_proj'}."
        f"{shared_suffix}.weight"
    ]
    assert all(".{expert}." not in key for key in state)
    assert not any(".1." in key for key in state)
    assert len(state) == 3

    metadata = LoRAPublishPlanner([torch.nn.Sequential(lora)]).global_metadata(
        {key: value.dtype for key, value in state.items()}
    )
    by_key = {item.key: item for item in metadata}
    assert by_key.keys() == state.keys()
    assert by_key[shared_keys[0]].shape == tuple(state[shared_keys[0]].shape)

    restored = _make_lora(shared_factor, "shared_outer")
    restored.bind_expert_layout((2, None, 0), (2, None, 0))
    restored.load_lora(state)
    restored_state = restored.sharded_lora_state_dict()
    assert restored_state.keys() == state.keys()
    assert all(torch.equal(restored_state[key], state[key]) for key in state)

    expanded = dict(state)
    compact_key = shared_keys[0]
    compact = expanded.pop(compact_key)
    projection = "gate_proj" if shared_factor == "A" else "down_proj"
    for expert in (0, 2):
        expanded[
            f"{PREFIX.format(expert=expert)}.{projection}.{shared_suffix}.weight"
        ] = compact
    with pytest.raises(KeyError, match="Mixed LoRA parameterization"):
        lora.load_lora(expanded)

    partially_mixed = dict(state)
    partially_mixed[
        f"{PREFIX.format(expert=0)}.{projection}.{shared_suffix}.weight"
    ] = compact
    with pytest.raises(KeyError, match="Incomplete or mixed"):
        lora.prepare_lora_slot(
            LoRASlotRef("checkpoint", "mixed"),
            partially_mixed,
            requires_grad=False,
        )


@pytest.mark.parametrize("shared_factor", ("A", "B"))
@pytest.mark.parametrize(
    ("source_parameterization", "target_parameterization"),
    (("shared_outer", "per_expert"), ("per_expert", "shared_outer")),
)
def test_dynamic_slot_compact_export_roundtrip_and_planner(
    shared_factor: LoraFactor,
    source_parameterization: MoeLoraParameterization,
    target_parameterization: MoeLoraParameterization,
) -> None:
    source = _make_lora(shared_factor, source_parameterization)
    target = _make_lora(shared_factor, target_parameterization)
    _fill(source, offset=1000)
    _assert_roundtrip(source, target, LoRASlotRef("checkpoint", "run"))


@pytest.mark.parametrize("shared_factor", ("A", "B"))
def test_dynamic_shared_factor_uses_dense_dp_cp_and_etp_groups(
    shared_factor: LoraFactor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.training import finalize_grads
    from art.trainer_rank._impl import TrainerRank

    class Group:
        def __init__(self, name: str) -> None:
            self.name = name

        def size(self) -> int:
            return 2

    dense_dp_cp = Group("dense_dp_cp_including_ep")
    expert_dp = Group("expert_dp")
    expert_tp = Group("expert_tp")
    dense_requests: list[bool] = []
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_group",
        lambda *, with_context_parallel: (
            dense_requests.append(with_context_parallel) or dense_dp_cp
        ),
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_data_parallel_group",
        lambda: expert_dp,
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_tensor_parallel_group",
        lambda *, check_initialized: expert_tp,
    )

    source = _make_lora(shared_factor, "shared_outer")
    target = _make_lora(shared_factor, "per_expert")
    _fill(source)
    ref = LoRASlotRef("checkpoint", "groups")
    assert target.load_lora_slot(
        ref,
        source.sharded_lora_state_dict(),
        requires_grad=True,
    )
    params = target.lora_slot_params(ref)
    for param in params:
        param.grad = torch.ones_like(param)

    reductions: dict[Group, list[tuple[int, ...]]] = {}

    def record_reduce(grads: list[torch.Tensor], *, group: Group, op: object) -> None:
        del op
        reductions[group] = [tuple(grad.shape) for grad in grads]

    monkeypatch.setattr(finalize_grads, "coalesced_all_reduce", record_reduce)
    trainer = object.__new__(TrainerRank)
    trainer._reduce_dynamic_grads(
        params,
        scale_grads=1.0,
    )

    shared_param = params[0] if shared_factor == "A" else params[1]
    expert_param = params[1] if shared_factor == "A" else params[0]
    assert dense_requests == [True]
    assert reductions[dense_dp_cp] == [tuple(shared_param.shape)]
    assert reductions[expert_dp] == [tuple(expert_param.shape)]
    assert reductions[expert_tp] == [tuple(shared_param.shape)]


@pytest.mark.parametrize("shared_factor", ("A", "B"))
def test_dynamic_optimizer_masks_use_active_slot_parameterization(
    shared_factor: LoraFactor,
) -> None:
    from art.trainer_rank._impl import TrainerRank, _CheckpointSlot

    source = _make_lora(shared_factor, "shared_outer")
    target = _make_lora(shared_factor, "per_expert")
    _fill(source)
    ref = LoRASlotRef("checkpoint", "optimizer")
    slot = target.prepare_lora_slot(
        ref,
        source.sharded_lora_state_dict(),
        requires_grad=True,
    )
    assert slot is not None
    trainer = object.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        model=[torch.nn.Sequential(target)],
        model_support_handler=SimpleNamespace(
            canonicalize_loaded_lora_state=lambda state, _model: state
        ),
    )

    prepared = SimpleNamespace(
        parameters=(slot.A_T, slot.B_T),
        sites=((target, slot),),
    )
    prepared_masks = trainer._prepared_optimizer_padding_masks(prepared)
    assert [tuple(mask.shape) for mask in prepared_masks] == [
        tuple(slot.A_T.shape),
        tuple(slot.B_T.shape),
    ]
    assert not any(torch.count_nonzero(mask) for mask in prepared_masks)

    target.install_lora_slot(slot)
    trainer._checkpoint_slots = {
        "optimizer": _CheckpointSlot(params=(slot.A_T, slot.B_T))
    }
    installed_masks = trainer._dynamic_optimizer_padding_masks("optimizer")
    assert [tuple(mask.shape) for mask in installed_masks] == [
        tuple(slot.A_T.shape),
        tuple(slot.B_T.shape),
    ]
    assert not any(torch.count_nonzero(mask) for mask in installed_masks)
