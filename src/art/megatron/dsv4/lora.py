from __future__ import annotations

import math
from typing import Any

from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
import torch

from art.megatron.lora import (
    GRAD_SYNC_OP_NONE,
    GRAD_SYNC_OP_SUM,
    LORA_ALPHA,
    TP_DEFAULT_GRAD_SYNC_DOMAIN,
    LoRA,
    LoRAParallelSpec,
)


def _weight_device_dtype(weight: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return weight.device, weight.dtype


def replicated_lora(
    *,
    adapter_model_prefix: str,
    weight: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    alpha: int = LORA_ALPHA,
) -> LoRA:
    device, dtype = _weight_device_dtype(weight)
    sync = LoRAParallelSpec(
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_NONE,
    )
    return LoRA(
        adapter_model_prefix=adapter_model_prefix,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dtype=dtype,
        device=device,
        a_parallel_spec=sync,
        b_parallel_spec=sync,
        allreduce=True,
    )


def column_parallel_lora(
    *,
    adapter_model_prefix: str,
    weight: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    alpha: int = LORA_ALPHA,
) -> LoRA:
    device, dtype = _weight_device_dtype(weight)
    a_spec = LoRAParallelSpec(
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_SUM,
    )
    b_spec = LoRAParallelSpec(
        sharded=True,
        shard_dim=-1,
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_NONE,
    )
    return LoRA(
        adapter_model_prefix=adapter_model_prefix,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dtype=dtype,
        device=device,
        a_parallel_spec=a_spec,
        b_parallel_spec=b_spec,
        allreduce=True,
    )


def row_parallel_lora(
    *,
    adapter_model_prefix: str,
    weight: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    alpha: int = LORA_ALPHA,
) -> LoRA:
    device, dtype = _weight_device_dtype(weight)
    a_spec = LoRAParallelSpec(
        sharded=True,
        shard_dim=-2,
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_NONE,
    )
    b_spec = LoRAParallelSpec(
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_SUM,
    )
    return LoRA(
        adapter_model_prefix=adapter_model_prefix,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dtype=dtype,
        device=device,
        a_parallel_spec=a_spec,
        b_parallel_spec=b_spec,
        allreduce=True,
    )


class Dsv4GroupedOutputLoRA(torch.nn.Module):
    def __init__(
        self,
        *,
        adapter_model_prefix: str,
        weight: torch.Tensor,
        in_features: int,
        out_features: int,
        local_groups: int,
        rank: int,
        alpha: int = LORA_ALPHA,
    ) -> None:
        super().__init__()
        if out_features % local_groups != 0:
            raise ValueError(
                f"{adapter_model_prefix}: out_features={out_features} is not "
                f"divisible by local_groups={local_groups}"
            )
        self.local_groups = local_groups
        self.out_per_group = out_features // local_groups
        self.lora = column_parallel_lora(
            adapter_model_prefix=adapter_model_prefix,
            weight=weight,
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
        )

    @property
    def adapter_model_prefix(self) -> str:
        return self.lora.adapter_model_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = self.lora.A_T.shape[-1]
        hidden = x @ self.lora.A_T
        weight = self.lora.B_T.view(rank, self.local_groups, self.out_per_group)
        out = torch.einsum("bsgr,rgo->bsgo", hidden, weight)
        return out if self.lora.scale == 1.0 else out * self.lora.scale


class Dsv4RowOutputLoRA(torch.nn.Module):
    def __init__(
        self,
        *,
        adapter_model_prefix: str,
        weight: torch.Tensor,
        in_features: int,
        out_features: int,
        tp_group: Any,
        rank: int,
        alpha: int = LORA_ALPHA,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.lora = row_parallel_lora(
            adapter_model_prefix=adapter_model_prefix,
            weight=weight,
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
        )

    @property
    def adapter_model_prefix(self) -> str:
        return self.lora.adapter_model_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora(x)
        if self.tp_group is not None and self.tp_group.size() > 1:
            out = reduce_from_tensor_model_parallel_region(out, group=self.tp_group)
        return out


class Dsv4PlainLoRA(torch.nn.Module):
    def __init__(self, lora: LoRA, *, sync_input_grad_group: Any | None = None) -> None:
        super().__init__()
        self.lora = lora
        self.sync_input_grad_group = sync_input_grad_group

    @property
    def adapter_model_prefix(self) -> str:
        return self.lora.adapter_model_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.sync_input_grad_group is not None
            and self.sync_input_grad_group.size() > 1
        ):
            x = copy_to_tensor_model_parallel_region(
                x, group=self.sync_input_grad_group
            )
        return self.lora(x)


def _targets_include(target_modules: set[str], *names: str) -> bool:
    return not target_modules or any(name in target_modules for name in names)


def _attach_lora(
    module: torch.nn.Module, attr_name: str, lora: torch.nn.Module
) -> None:
    if hasattr(module, attr_name):
        raise RuntimeError(f"{module.__class__.__name__}.{attr_name} is already set")
    setattr(module, attr_name, lora)


def apply_dsv4_attention_lora(
    attention: Any,
    *,
    adapter_model_prefix: str,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    if _targets_include(target_modules, "q_a_proj"):
        _attach_lora(
            attention,
            "wq_a_lora",
            Dsv4PlainLoRA(
                replicated_lora(
                    adapter_model_prefix=f"{adapter_model_prefix}.self_attn.q_a_proj",
                    weight=attention.wq_a.weight,
                    in_features=attention.wq_a.weight.shape[1],
                    out_features=attention.wq_a.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                )
            ),
        )
    if _targets_include(target_modules, "q_b_proj"):
        _attach_lora(
            attention,
            "wq_b_lora",
            Dsv4PlainLoRA(
                column_parallel_lora(
                    adapter_model_prefix=f"{adapter_model_prefix}.self_attn.q_b_proj",
                    weight=attention.wq_b.weight,
                    in_features=attention.wq_b.weight.shape[1],
                    out_features=attention.wq_b.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                ),
                sync_input_grad_group=attention.tp_group,
            ),
        )
    if _targets_include(target_modules, "kv_proj"):
        _attach_lora(
            attention,
            "wkv_lora",
            Dsv4PlainLoRA(
                replicated_lora(
                    adapter_model_prefix=f"{adapter_model_prefix}.self_attn.kv_proj",
                    weight=attention.wkv.weight,
                    in_features=attention.wkv.weight.shape[1],
                    out_features=attention.wkv.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                )
            ),
        )
    if _targets_include(target_modules, "o_a_proj"):
        _attach_lora(
            attention,
            "wo_a_lora",
            Dsv4GroupedOutputLoRA(
                adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_a_proj",
                weight=attention.wo_a.weight,
                in_features=attention.wo_a.weight.shape[1],
                out_features=attention.wo_a.weight.shape[0],
                local_groups=attention.n_local_groups,
                rank=rank,
                alpha=alpha,
            ),
        )
    if _targets_include(target_modules, "o_b_proj"):
        _attach_lora(
            attention,
            "wo_b_lora",
            Dsv4RowOutputLoRA(
                adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_b_proj",
                weight=attention.wo_b.weight,
                in_features=attention.wo_b.weight.shape[1],
                out_features=attention.wo_b.weight.shape[0],
                tp_group=attention.tp_group,
                rank=rank,
                alpha=alpha,
            ),
        )
    compressor = getattr(attention, "compressor", None)
    if compressor is None:
        return
    if _targets_include(target_modules, "compressor.kv_proj"):
        _attach_lora(
            compressor,
            "kv_proj_lora",
            Dsv4PlainLoRA(
                replicated_lora(
                    adapter_model_prefix=(
                        f"{adapter_model_prefix}.self_attn.compressor.kv_proj"
                    ),
                    weight=compressor.wkv.weight,
                    in_features=compressor.wkv.weight.shape[1],
                    out_features=compressor.wkv.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                )
            ),
        )
    if _targets_include(target_modules, "compressor.gate_proj"):
        _attach_lora(
            compressor,
            "gate_proj_lora",
            Dsv4PlainLoRA(
                replicated_lora(
                    adapter_model_prefix=(
                        f"{adapter_model_prefix}.self_attn.compressor.gate_proj"
                    ),
                    weight=compressor.wgate.weight,
                    in_features=compressor.wgate.weight.shape[1],
                    out_features=compressor.wgate.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                )
            ),
        )


def _unwrap_lora(module: torch.nn.Module | None) -> LoRA | None:
    if isinstance(module, LoRA):
        return module
    nested = getattr(module, "lora", None) if module is not None else None
    return nested if isinstance(nested, LoRA) else None


def _adapter_alpha_dim(lora: LoRA) -> tuple[int, int]:
    dim = int(lora.A_T.shape[-1])
    alpha = float(lora.scale) * dim
    rounded = round(alpha)
    if not math.isclose(alpha, rounded):
        raise RuntimeError(f"{lora.adapter_model_prefix}: non-integral alpha={alpha}")
    return rounded, dim


def _adapter_param_prefix(base_prefix: str, adapter_key: str | None) -> str:
    if adapter_key is None:
        return f"{base_prefix}.adapter"
    return f"{base_prefix}.adapter.{adapter_key}"


def _adapter_weight(
    base_prefix: str,
    lora: LoRA,
    *,
    adapter_key: str | None = None,
) -> Any:
    from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple
    from megatron.bridge.models.conversion.peft_bridge import AdapterWeight

    alpha, dim = _adapter_alpha_dim(lora)
    linear_in = lora.A_T.transpose(-1, -2).contiguous()
    linear_out = lora.B_T.transpose(-1, -2).contiguous()
    param_prefix = _adapter_param_prefix(base_prefix, adapter_key)
    return AdapterWeight(
        global_base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=alpha,
        dim=dim,
        linear_in_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_in.weight",
            weight=linear_in,
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_out.weight",
            weight=linear_out,
            vp_stage=0,
        ),
    )


def _tensor_parallel_group() -> Any | None:
    try:
        return ps.get_tensor_model_parallel_group()
    except Exception:
        return None


def _gather_tp_input_shards(
    linear_in: torch.Tensor, tp_group: Any | None
) -> torch.Tensor:
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return linear_in
    tp_group = tp_group or _tensor_parallel_group()
    world_size = tp_group.size() if tp_group is not None else dist.get_world_size()
    if world_size <= 1:
        return linear_in
    gathered = [torch.empty_like(linear_in) for _ in range(world_size)]
    dist.all_gather(gathered, linear_in.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=-1)


def _row_parallel_adapter_weight(
    base_prefix: str,
    lora: LoRA,
    *,
    tp_group: Any | None,
) -> Any:
    from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple
    from megatron.bridge.models.conversion.peft_bridge import AdapterWeight

    alpha, dim = _adapter_alpha_dim(lora)
    linear_in = _gather_tp_input_shards(
        lora.A_T.transpose(-1, -2).contiguous(), tp_group
    )
    linear_out = lora.B_T.transpose(-1, -2).contiguous()
    param_prefix = _adapter_param_prefix(base_prefix, None)
    return AdapterWeight(
        global_base_prefix=base_prefix,
        adapter_key=None,
        alpha=alpha,
        dim=dim,
        linear_in_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_in.weight",
            weight=linear_in,
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_out.weight",
            weight=linear_out,
            vp_stage=0,
        ),
    )


def _add_adapter(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    base_prefix: str,
    lora_module: torch.nn.Module | None,
) -> None:
    lora = _unwrap_lora(lora_module)
    if lora is None:
        return
    adapter_weights_by_base[f"{base_prefix}.weight"] = [
        _adapter_weight(base_prefix, lora)
    ]


def _add_row_parallel_adapter(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    base_prefix: str,
    lora_module: torch.nn.Module | None,
) -> None:
    lora = _unwrap_lora(lora_module)
    if lora is None:
        return
    adapter_weights_by_base[f"{base_prefix}.weight"] = [
        _row_parallel_adapter_weight(
            base_prefix,
            lora,
            tp_group=getattr(lora_module, "tp_group", None),
        )
    ]


def add_dsv4_attention_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    attention: Any,
) -> None:
    attn_prefix = f"{layer_prefix}.self_attention"
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.wq_a",
        lora_module=getattr(attention, "wq_a_lora", None),
    )
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.wq_b",
        lora_module=getattr(attention, "wq_b_lora", None),
    )
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.wkv",
        lora_module=getattr(attention, "wkv_lora", None),
    )
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.wo_a",
        lora_module=getattr(attention, "wo_a_lora", None),
    )
    _add_row_parallel_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.wo_b",
        lora_module=getattr(attention, "wo_b_lora", None),
    )
    compressor = getattr(attention, "compressor", None)
    if compressor is None:
        return
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.compressor.wkv",
        lora_module=getattr(compressor, "kv_proj_lora", None),
    )
    _add_adapter(
        adapter_weights_by_base,
        base_prefix=f"{attn_prefix}.compressor.wgate",
        lora_module=getattr(compressor, "gate_proj_lora", None),
    )


def add_dsv4_shared_experts_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    shared_experts: Any,
) -> None:
    from art.megatron.lora import SharedExpertsLinearFC1LoRA, SharedExpertsLinearFC2LoRA

    linear_fc1 = getattr(shared_experts, "linear_fc1", None)
    if isinstance(linear_fc1, SharedExpertsLinearFC1LoRA):
        base_prefix = f"{layer_prefix}.mlp.shared_experts.linear_fc1"
        adapter_weights_by_base[f"{base_prefix}.weight"] = [
            _adapter_weight(
                base_prefix,
                linear_fc1.gate_lora,
                adapter_key="adapter_gate",
            ),
            _adapter_weight(
                base_prefix,
                linear_fc1.up_lora,
                adapter_key="adapter_up",
            ),
        ]

    linear_fc2 = getattr(shared_experts, "linear_fc2", None)
    if isinstance(linear_fc2, SharedExpertsLinearFC2LoRA):
        base_prefix = f"{layer_prefix}.mlp.shared_experts.linear_fc2"
        adapter_weights_by_base[f"{base_prefix}.weight"] = [
            _row_parallel_adapter_weight(
                base_prefix,
                linear_fc2.row_parallel_lora.lora,
                tp_group=_tensor_parallel_group(),
            )
        ]


def disable_dsv4_etp_shared_expert_lora_compile(shared_experts: Any) -> None:
    """Keep DSV4 ETP shared-expert down LoRA outside compiled layer graphs.

    Torch 2.11 can spend unbounded time compiling the row-parallel shared-expert
    LoRA path when DSV4 runs with expert tensor parallelism. The barrier is
    instance-local and leaves non-ETP and generic MoE shared experts unchanged.
    """
    from art.megatron.lora import SharedExpertsLinearFC2LoRA

    linear_fc2 = getattr(shared_experts, "linear_fc2", None)
    if not isinstance(linear_fc2, SharedExpertsLinearFC2LoRA):
        return
    if bool(getattr(linear_fc2, "_art_dsv4_compile_disabled", False)):
        return
    linear_fc2.forward = torch.compiler.disable(linear_fc2.forward)
    linear_fc2._art_dsv4_compile_disabled = True
