from collections.abc import Sequence
import math
from typing import Any, Literal, cast

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
from pydantic import BaseModel, ConfigDict
import torch

from .cute_grouped_lora_quack import quack_grouped_lora, quack_grouped_lora_dual

LORA_RANK = 1
LORA_ALPHA = 32

ShardDomain = Literal["tp", "expert_tp"]
GradSyncDomain = Literal["tp_default", "expert_tp"]
GradSyncOp = Literal["none", "sum", "avg"]

TP_DEFAULT_GRAD_SYNC_DOMAIN: GradSyncDomain = "tp_default"
EXPERT_TP_GRAD_SYNC_DOMAIN: GradSyncDomain = "expert_tp"
GRAD_SYNC_OP_NONE: GradSyncOp = "none"
GRAD_SYNC_OP_SUM: GradSyncOp = "sum"
GRAD_SYNC_OP_AVG: GradSyncOp = "avg"


class LoRAParallelSpec(BaseModel):
    # This spec only describes TP / expert-TP behavior.
    # DP/CP vs expert-DP behavior is selected separately via `allreduce`.
    model_config = ConfigDict(frozen=True)

    shard_domain: ShardDomain = "tp"
    sharded: bool = False
    shard_dim: int | None = None
    grad_sync_domain: GradSyncDomain = TP_DEFAULT_GRAD_SYNC_DOMAIN
    grad_sync_op: GradSyncOp = GRAD_SYNC_OP_NONE


def _distributed_initialized() -> bool:
    is_initialized = getattr(torch.distributed, "is_initialized", None)
    return (
        torch.distributed.is_available()
        and callable(is_initialized)
        and bool(is_initialized())
    )


def _get_shard_world_size(domain: ShardDomain) -> int:
    if not _distributed_initialized():
        return 1
    if domain == "tp":
        return ps.get_tensor_model_parallel_world_size()
    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None:
        return 1
    return group.size()


def _get_shard_rank(domain: ShardDomain) -> int:
    if not _distributed_initialized():
        return 0
    if domain == "tp":
        return ps.get_tensor_model_parallel_rank()
    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None:
        return 0
    return group.rank()


def _get_shard_group(domain: ShardDomain) -> Any | None:
    if not _distributed_initialized():
        return None
    if domain == "tp":
        return ps.get_tensor_model_parallel_group()
    return ps.get_expert_tensor_parallel_group(check_initialized=False)


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid shard axis {axis} for tensor ndim={ndim}")
    return axis


def _linear_disables_tensor_parallel_comm(linear: Any) -> bool:
    # Shared experts can keep TP-sharded weights while deferring TP comm to the
    # overlap path by setting parallel_mode=None / explicit_expert_comm=True.
    return getattr(linear, "parallel_mode", "") is None or getattr(
        linear, "explicit_expert_comm", False
    )


def _set_lora_parallel_metadata(
    param: torch.nn.Parameter,
    *,
    parallel_spec: LoRAParallelSpec,
    allreduce: bool,
) -> None:
    replicated = not parallel_spec.sharded
    setattr(param, "lora_shard_domain", parallel_spec.shard_domain)
    setattr(param, "lora_tp_sharded", parallel_spec.sharded)
    setattr(param, "lora_tp_replicated", replicated)
    setattr(param, "lora_tp_shard_dim", parallel_spec.shard_dim)
    setattr(param, "grad_sync_domain", parallel_spec.grad_sync_domain)
    setattr(param, "grad_sync_op", parallel_spec.grad_sync_op)
    # Megatron DDP routing flag:
    # - allreduce=True: sync with regular DP/CP replicas.
    # - allreduce=False: sync with expert-DP replicas.
    # TP / expert-TP replica handling is controlled by grad_sync_* metadata.
    setattr(param, "allreduce", allreduce)

    # Megatron's native TP finalize path consumes this attr.
    setattr(
        param,
        "average_gradients_across_tp_domain",
        (
            replicated
            and parallel_spec.grad_sync_domain == TP_DEFAULT_GRAD_SYNC_DOMAIN
            and parallel_spec.grad_sync_op == GRAD_SYNC_OP_AVG
        ),
    )

    # Megatron optimizer and checkpoint logic rely on tensor model-parallel metadata
    # to distinguish true shards from TP-duplicate params.
    if parallel_spec.sharded:
        shard_dim = parallel_spec.shard_dim
        if shard_dim is None:
            raise ValueError("LoRAParallelSpec.shard_dim must be set when sharded=True")
        setattr(param, "tensor_model_parallel", True)
        setattr(param, "partition_dim", _normalize_axis(shard_dim, param.ndim))
        # stride > 1 means the dim is split into blocks and each tp rank holds a shard of the block
        # this might happen for fused e.g. gate_(up|proj), but loras are individual per module
        setattr(param, "partition_stride", 1)
    else:
        setattr(param, "tensor_model_parallel", False)
        setattr(param, "partition_dim", -1)
        setattr(param, "partition_stride", 1)


class LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
        num_local_experts: int = 1,
        a_parallel_spec: LoRAParallelSpec = LoRAParallelSpec(),
        b_parallel_spec: LoRAParallelSpec = LoRAParallelSpec(),
        allreduce: bool = True,
    ) -> None:
        super().__init__()
        assert num_local_experts == 1 or "{expert}" in adapter_model_prefix, (
            "adapter_model_prefix must contain the '{expert}' format placeholder if num_local_experts > 1"
        )
        self.adapter_model_prefix = adapter_model_prefix
        self.scale = alpha / rank
        self.A_T = torch.nn.Parameter(
            torch.zeros(
                num_local_experts, in_features, rank, dtype=dtype, device=device
            ).squeeze(0)
        )
        self.B_T = torch.nn.Parameter(
            torch.zeros(
                num_local_experts, rank, out_features, dtype=dtype, device=device
            ).squeeze(0)
        )
        _set_lora_parallel_metadata(
            self.A_T,
            parallel_spec=a_parallel_spec,
            allreduce=allreduce,
        )
        _set_lora_parallel_metadata(
            self.B_T,
            parallel_spec=b_parallel_spec,
            allreduce=allreduce,
        )
        self._expert_offset = ps.get_expert_model_parallel_rank() * num_local_experts
        self.reset_lora_parameters()

    @property
    def num_local_experts(self) -> int:
        return self.A_T.shape[0] if self.A_T.ndim == 3 else 1

    def _broadcast_if_replicated(self, param: torch.nn.Parameter) -> None:
        if not param.lora_tp_replicated:  # ty: ignore[unresolved-attribute]
            return
        domain = param.lora_shard_domain  # ty: ignore[unresolved-attribute]
        world_size = _get_shard_world_size(domain)
        if world_size <= 1:
            return
        group = _get_shard_group(domain)
        if group is None:
            raise RuntimeError(
                f"{self.adapter_model_prefix}: missing process group for replicated parameter domain={domain}"
            )
        src = torch.distributed.get_global_rank(  # ty: ignore[possibly-missing-attribute]
            group, 0
        )
        torch.distributed.broadcast(  # ty: ignore[possibly-missing-attribute]
            param.data,
            src=src,
            group=group,
        )

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA weights (A=Kaiming, B=zeros) like PEFT defaults."""
        if self.A_T.ndim == 3:
            for expert in range(self.A_T.shape[0]):
                torch.nn.init.kaiming_uniform_(self.A_T[expert].T, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.A_T.T, a=math.sqrt(5))
        torch.nn.init.zeros_(self.B_T)
        self._broadcast_if_replicated(self.A_T)
        self._broadcast_if_replicated(self.B_T)

    def _expected_weight_keys(self, suffix: str) -> list[str]:
        if self.num_local_experts > 1:
            return [
                f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{suffix}.weight"
                for expert in range(self.num_local_experts)
            ]
        return [f"{self.adapter_model_prefix}.{suffix}.weight"]

    def load_lora(self, adapter_model: dict[str, torch.Tensor]) -> None:
        missing_keys = [
            key
            for suffix in ("lora_A", "lora_B")
            for key in self._expected_weight_keys(suffix)
            if key not in adapter_model
        ]
        if missing_keys:
            raise KeyError(
                f"Missing LoRA adapter keys for {self.adapter_model_prefix}: {sorted(missing_keys)}"
            )
        self.load_weights(
            adapter_model,
            suffix="lora_A",
            into=self.A_T,
        )
        self.load_weights(
            adapter_model,
            suffix="lora_B",
            into=self.B_T,
        )

    def load_weights(
        self,
        adapter_model: dict[str, torch.Tensor],
        *,
        suffix: str,
        into: torch.nn.Parameter,
    ) -> None:
        keys = self._expected_weight_keys(suffix)
        if self.num_local_experts > 1:
            weight = torch.stack([adapter_model[key].T for key in keys])
        else:
            weight = adapter_model[keys[0]].T
        self.load_weight(weight, into=into)

    def load_weight(self, weight: torch.Tensor, *, into: torch.nn.Parameter) -> None:
        domain = into.lora_shard_domain  # ty: ignore[unresolved-attribute]
        if into.lora_tp_sharded:  # ty: ignore[unresolved-attribute]
            axis = into.lora_tp_shard_dim  # ty: ignore[unresolved-attribute]
            axis = _normalize_axis(axis, weight.ndim)
            world_size = _get_shard_world_size(domain)
            rank = _get_shard_rank(domain)
            if weight.shape[axis] % world_size != 0:
                raise ValueError(
                    f"{self.adapter_model_prefix}: weight shape {tuple(weight.shape)} is not divisible by world size "
                    f"{world_size} on axis {axis}"
                )
            local_size = weight.shape[axis] // world_size
            if into.shape[axis] != local_size:
                raise ValueError(
                    f"{self.adapter_model_prefix}: expected local shard size {into.shape[axis]}, got {local_size}"
                )
            weight = weight.narrow(axis, rank * local_size, local_size)
        elif tuple(weight.shape) != tuple(into.shape):
            raise ValueError(
                f"{self.adapter_model_prefix}: unsharded load shape mismatch, got {tuple(weight.shape)} "
                f"expected {tuple(into.shape)}"
            )
        into.data.copy_(weight)
        into.requires_grad = True

    def _should_export_parameter(self, param: torch.nn.Parameter) -> bool:
        """
        Determine if the given LoRA param should be exported in the sharded LoRA state dict
        (drop replicated ranks/params).
        """
        if self.num_local_experts > 1:  # self is a MoE layer
            if ps.get_expert_data_parallel_rank() != 0:
                return False
        else:  # self is a non-MoE layer
            # dp x cp rank 0 participates
            if ps.get_data_parallel_rank(with_context_parallel=True) != 0:
                return False

        # this param is fully sharded, all shard ranks participate
        if param.lora_tp_sharded:  # ty: ignore[unresolved-attribute]
            return True
        # param is replicated, tp rank 0 or etp rank 0 participates
        return _get_shard_rank(param.lora_shard_domain) == 0  # ty: ignore[unresolved-attribute]

    def _manifest_for_param(self, param: torch.nn.Parameter) -> dict[str, Any]:
        return {
            "domain": param.lora_shard_domain,  # ty: ignore[unresolved-attribute]
            "sharded": param.lora_tp_sharded,  # ty: ignore[unresolved-attribute]
            "shard_dim": param.lora_tp_shard_dim,  # ty: ignore[unresolved-attribute]
            "shard_world_size": _get_shard_world_size(param.lora_shard_domain)  # ty: ignore[unresolved-attribute]
            if param.lora_tp_sharded  # ty: ignore[unresolved-attribute]
            else 1,
            "shard_rank": _get_shard_rank(param.lora_shard_domain)  # ty: ignore[unresolved-attribute]
            if param.lora_tp_sharded  # ty: ignore[unresolved-attribute]
            else 0,
        }

    def _lora_params(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [
            ("lora_A.weight", self.A_T),
            ("lora_B.weight", self.B_T),
        ]

    def _export_items(
        self,
    ) -> list[tuple[str, torch.nn.Parameter, int | None]]:
        export_items: list[tuple[str, torch.nn.Parameter, int | None]] = []
        for key, param in self._lora_params():
            if not self._should_export_parameter(param):
                continue
            if self.num_local_experts > 1:
                for expert in range(self.num_local_experts):
                    full_key = f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{key}"
                    export_items.append((full_key, param, expert))
            else:
                export_items.append((f"{self.adapter_model_prefix}.{key}", param, None))
        return export_items

    def sharded_lora_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            key: self._manifest_for_param(param)
            for key, param, _expert in self._export_items()
        }

    def sharded_lora_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key, param, expert in self._export_items():
            state[key] = param.data[expert].T if expert is not None else param.data.T
        return state

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        if tokens_per_expert is not None:
            assert self.num_local_experts > 1, (
                "tokens_per_expert is only supported if num_local_experts > 1"
            )
            bsz = tokens_per_expert
            if isinstance(bsz, list):
                bsz = torch.tensor(bsz, dtype=torch.int64, device="cpu")
            # If no tokens routed locally, return zeros.
            if isinstance(bsz, torch.Tensor) and int(torch.count_nonzero(bsz)) == 0:
                return x.new_zeros((x.shape[0], self.B_T.shape[-1]))
            return quack_grouped_lora(x, self.A_T, self.B_T, bsz, scale=self.scale)
        out = (x @ self.A_T) @ self.B_T
        if self.scale == 1.0:
            return out
        return out * self.scale


class SelfAttentionLinearProjLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_proj: TERowParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.linear_proj = linear_proj
        self.reduce_output = reduce_output
        assert isinstance(linear_proj.weight, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="tp",
            sharded=True,
            shard_dim=-2,
            grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_NONE,  # only need DP-type reductions
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": False,
                "shard_dim": None,
                "grad_sync_op": GRAD_SYNC_OP_SUM,  # sum replicated TP contributions
            }
        )
        self.lora = LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_proj.in_features,
            out_features=linear_proj.out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_proj.weight.dtype,
            device=linear_proj.weight.device,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            # Non-expert LoRA params use Megatron's dense DP/CP gradient buckets.
            allreduce=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_output, bias_output = self.linear_proj(x)
        assert isinstance(base_output, torch.Tensor)
        assert isinstance(bias_output, (torch.Tensor, type(None)))

        lora_output = self.lora(x)
        if self.reduce_output and self.provider.tensor_model_parallel_size > 1:
            if self.provider.sequence_parallel:
                lora_output = reduce_scatter_to_sequence_parallel_region(lora_output)
            else:
                lora_output = reduce_from_tensor_model_parallel_region(lora_output)
        return base_output + lora_output, bias_output


class SelfAttentionLinearQKVLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_qkv: TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
    ) -> None:
        super().__init__()
        self.provider = provider
        linear_qkv.return_layernorm_output = True
        linear_qkv.return_layernorm_output_gathered = True
        self.linear_qkv = linear_qkv
        assert self.provider.kv_channels is not None
        assert self.provider.num_query_groups is not None
        assert self.provider.num_attention_heads is not None
        if self.provider.num_attention_heads % self.provider.num_query_groups != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_query_groups for QKV LoRA"
            )
        linear_qkv_weight = cast(torch.Tensor, linear_qkv.weight)
        total_out_features_per_rank = linear_qkv_weight.shape[0]
        kv_out_features = self.provider.kv_channels * self.provider.num_query_groups
        tp_world_size = ps.get_tensor_model_parallel_world_size()
        assert kv_out_features % tp_world_size == 0, (
            "kv_out_features must be divisible by tensor parallel size"
        )
        q_out_features = self.provider.kv_channels * self.provider.num_attention_heads
        assert q_out_features % tp_world_size == 0, (
            "q_out_features must be divisible by tensor parallel size"
        )
        q_out_features_per_rank = q_out_features // tp_world_size
        kv_out_features_per_rank = kv_out_features // tp_world_size
        self.attention_output_gate = bool(
            getattr(self.provider, "attention_output_gate", False)
        )
        q_and_gate_out_features_per_rank = total_out_features_per_rank - (
            2 * kv_out_features_per_rank
        )
        expected_q_out_features_per_rank = q_out_features_per_rank * (
            2 if self.attention_output_gate else 1
        )
        assert q_and_gate_out_features_per_rank == expected_q_out_features_per_rank, (
            "Unexpected per-rank QKV packing for this attention layout"
        )
        self.num_query_groups_per_partition = (
            self.provider.num_query_groups // tp_world_size
        )
        self.num_attention_heads_per_group = (
            self.provider.num_attention_heads // self.provider.num_query_groups
        )
        self.hidden_size_per_attention_head = self.provider.kv_channels
        assert isinstance(linear_qkv.weight, torch.Tensor)
        self.q_proj_lora = self._build_qkv_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.q_proj",
            linear_qkv=linear_qkv,
            rank=rank,
            alpha=alpha,
            out_features=q_and_gate_out_features_per_rank,
        )
        self.k_proj_lora = self._build_qkv_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.k_proj",
            linear_qkv=linear_qkv,
            rank=rank,
            alpha=alpha,
            out_features=kv_out_features_per_rank,
        )
        self.v_proj_lora = self._build_qkv_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.v_proj",
            linear_qkv=linear_qkv,
            rank=rank,
            alpha=alpha,
            out_features=kv_out_features_per_rank,
        )

    @staticmethod
    def _build_qkv_lora(
        *,
        adapter_model_prefix: str,
        linear_qkv: TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
        out_features: int,
    ) -> LoRA:
        assert isinstance(linear_qkv.weight, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="tp",
            sharded=False,
            shard_dim=None,
            grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM,  # sum replicated TP contributions
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": True,
                "shard_dim": -1,
                "grad_sync_op": GRAD_SYNC_OP_NONE,  # only need DP-type reductions
            }
        )
        return LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_qkv.in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_qkv.weight.dtype,
            device=linear_qkv.weight.device,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            # Non-expert LoRA params use Megatron's dense DP/CP gradient buckets.
            allreduce=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        (
            linear_output_and_layernorm_output,
            bias,
        ) = self.linear_qkv(x)
        linear_output, layernorm_output = linear_output_and_layernorm_output
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(layernorm_output, torch.Tensor)
        assert isinstance(bias, (torch.Tensor, type(None)))

        query_and_gate = self.q_proj_lora(layernorm_output)
        key = self.k_proj_lora(layernorm_output)
        value = self.v_proj_lora(layernorm_output)
        # Qwen3 packs [query, key, value] per group, while Qwen3.5 packs
        # [query, gate, key, value].
        query_and_gate_5d = query_and_gate.reshape(
            query_and_gate.shape[0],
            query_and_gate.shape[1],
            self.num_query_groups_per_partition,
            self.num_attention_heads_per_group
            * (2 if self.attention_output_gate else 1),
            self.hidden_size_per_attention_head,
        )
        key_5d = key.reshape(
            key.shape[0],
            key.shape[1],
            self.num_query_groups_per_partition,
            1,
            self.hidden_size_per_attention_head,
        )
        value_5d = value.reshape(
            value.shape[0],
            value.shape[1],
            self.num_query_groups_per_partition,
            1,
            self.hidden_size_per_attention_head,
        )
        qkv_5d = torch.cat([query_and_gate_5d, key_5d, value_5d], dim=3)
        adapter_output = qkv_5d.reshape(qkv_5d.shape[0], qkv_5d.shape[1], -1)

        return linear_output + adapter_output, bias


class GatedDeltaNetInProjLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        in_proj: TELayerNormColumnParallelLinear,
        gated_delta_net: GatedDeltaNet,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        in_proj.return_layernorm_output = True
        in_proj.return_layernorm_output_gathered = True
        self.in_proj = in_proj
        self.num_value_heads_per_partition = (
            gated_delta_net.num_value_heads // ps.get_tensor_model_parallel_world_size()
        )
        qkv_out_features_per_partition = (
            gated_delta_net.qk_dim * 2 + gated_delta_net.v_dim
        ) // ps.get_tensor_model_parallel_world_size()
        z_out_features_per_partition = (
            gated_delta_net.v_dim // ps.get_tensor_model_parallel_world_size()
        )
        assert isinstance(in_proj.weight, torch.Tensor)
        self.qkv_lora = self._build_in_proj_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.in_proj_qkv",
            in_proj=in_proj,
            rank=rank,
            alpha=alpha,
            out_features=qkv_out_features_per_partition,
        )
        self.z_lora = self._build_in_proj_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.in_proj_z",
            in_proj=in_proj,
            rank=rank,
            alpha=alpha,
            out_features=z_out_features_per_partition,
        )

    @staticmethod
    def _build_in_proj_lora(
        *,
        adapter_model_prefix: str,
        in_proj: TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
        out_features: int,
    ) -> LoRA:
        assert isinstance(in_proj.weight, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="tp",
            sharded=False,
            shard_dim=None,
            grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM,
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": True,
                "shard_dim": -1,
                "grad_sync_op": GRAD_SYNC_OP_NONE,
            }
        )
        return LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=in_proj.in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dtype=in_proj.weight.dtype,
            device=in_proj.weight.device,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            allreduce=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        (
            linear_output_and_layernorm_output,
            bias,
        ) = self.in_proj(x)
        linear_output, layernorm_output = linear_output_and_layernorm_output
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(layernorm_output, torch.Tensor)
        assert isinstance(bias, (torch.Tensor, type(None)))

        qkv = self.qkv_lora(layernorm_output)
        z = self.z_lora(layernorm_output)
        beta = qkv.new_zeros(
            qkv.shape[0],
            qkv.shape[1],
            self.num_value_heads_per_partition,
        )
        alpha = beta.clone()
        adapter_output = torch.cat([qkv, z, beta, alpha], dim=-1)
        return linear_output + adapter_output, bias


class MLPExpertsLinearFC1LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelGroupedLinear,
        rank: int,
        alpha: float,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        assert linear_fc1 is not None
        self.linear_fc1 = linear_fc1
        self.gate_lora = self._build_fc1_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.gate_proj",
            linear_fc1=linear_fc1,
            rank=rank,
            alpha=alpha,
            num_local_experts=num_local_experts,
        )
        self.up_lora = self._build_fc1_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.up_proj",
            linear_fc1=linear_fc1,
            rank=rank,
            alpha=alpha,
            num_local_experts=num_local_experts,
        )
        self.uses_direct_quack_grouped_lora_dual = True

    @staticmethod
    def _build_fc1_lora(
        *,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelGroupedLinear,
        rank: int,
        alpha: float,
        num_local_experts: int,
    ) -> LoRA:
        assert linear_fc1 is not None
        assert isinstance(linear_fc1.weight0, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=False,
            shard_dim=None,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM,  # we handle this with extended finalize_grads
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": True,
                "shard_dim": -1,
                "grad_sync_domain": EXPERT_TP_GRAD_SYNC_DOMAIN,
                "grad_sync_op": GRAD_SYNC_OP_NONE,  # only need DP-type reductions
            }
        )
        return LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_fc1.in_features,
            out_features=linear_fc1.out_features // 2,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc1.weight0.dtype,
            device=linear_fc1.weight0.device,
            num_local_experts=num_local_experts,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            # Expert LoRA params use Megatron's expert-DP gradient buckets.
            allreduce=False,
        )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = self.linear_fc1(x, tokens_per_expert)
        counts = tokens_per_expert
        if isinstance(counts, list):
            counts = torch.tensor(counts, dtype=torch.int64, device="cpu")
        if isinstance(counts, torch.Tensor) and int(torch.count_nonzero(counts)) == 0:
            adapter_out = x.new_zeros((x.shape[0], self.linear_fc1.out_features))
        else:
            adapter_out = quack_grouped_lora_dual(
                x,
                self.gate_lora.A_T,
                self.gate_lora.B_T,
                self.up_lora.A_T,
                self.up_lora.B_T,
                counts,
                scale_gate=self.gate_lora.scale,
                scale_up=self.up_lora.scale,
            )
        return base_out + adapter_out, bias_out


class SharedExpertsLinearFC1LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelLinear | TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.linear_fc1 = linear_fc1
        self.gate_lora = self._build_fc1_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.gate_proj",
            linear_fc1=linear_fc1,
            rank=rank,
            alpha=alpha,
        )
        self.up_lora = self._build_fc1_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.up_proj",
            linear_fc1=linear_fc1,
            rank=rank,
            alpha=alpha,
        )

    @staticmethod
    def _build_fc1_lora(
        *,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelLinear | TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
    ) -> LoRA:
        assert isinstance(linear_fc1.weight, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="tp",
            sharded=False,
            shard_dim=None,
            grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM,
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": True,
                "shard_dim": -1,
                "grad_sync_op": GRAD_SYNC_OP_NONE,
            }
        )
        return LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_fc1.in_features,
            out_features=linear_fc1.out_features // 2,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc1.weight.dtype,
            device=linear_fc1.weight.device,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            allreduce=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = self.linear_fc1(x)
        adapter_out = torch.cat(
            [self.gate_lora(x), self.up_lora(x)],
            dim=-1,
        )
        return base_out + adapter_out, bias_out


class MLPExpertsLinearFC2LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc2: TERowParallelGroupedLinear,
        rank: int,
        alpha: float,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        assert linear_fc2 is not None
        assert isinstance(linear_fc2.weight0, torch.Tensor)
        self.linear_fc2 = linear_fc2
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=True,
            shard_dim=-2,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_NONE,  # only need DP-type reductions
        )
        b_parallel_spec = a_parallel_spec.model_copy(
            update={
                "sharded": False,
                "shard_dim": None,
                "grad_sync_domain": EXPERT_TP_GRAD_SYNC_DOMAIN,
                "grad_sync_op": GRAD_SYNC_OP_SUM,  # we handle this with extended finalize_grads
            }
        )
        self.lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.down_proj",
            in_features=linear_fc2.in_features,
            out_features=linear_fc2.out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc2.weight0.dtype,
            device=linear_fc2.weight0.device,
            num_local_experts=num_local_experts,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            # Expert LoRA params use Megatron's expert-DP gradient buckets.
            allreduce=False,
        )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = self.linear_fc2(x, tokens_per_expert)
        adapter_out = self.lora(x, tokens_per_expert=tokens_per_expert)
        # the reason there is no TP comm here is because the MoE token routing handles
        # expert TP comm externally
        return base_out + adapter_out, bias_out


class SharedExpertsLinearFC2LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc2: TERowParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
    ) -> None:
        super().__init__()
        self.row_parallel_lora = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.down_proj",
            linear_proj=linear_fc2,
            rank=rank,
            alpha=alpha,
            provider=provider,
            reduce_output=not _linear_disables_tensor_parallel_comm(linear_fc2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.row_parallel_lora(x)


def apply_lora_adapters(
    model: Sequence[torch.nn.Module],
    provider: GPTModelProvider,
) -> list[torch.nn.Module]:
    def _is_language_transformer_layer(module_name: str) -> bool:
        while module_name.startswith("module."):
            module_name = module_name.removeprefix("module.")
        return module_name.startswith(
            ("decoder.layers.", "language_model.decoder.layers.")
        )

    def _unwrap_attr(
        value: Any,
        attr_name: str,
        expected_type: type[Any] | tuple[type[Any], ...],
    ) -> Any:
        if isinstance(value, expected_type):
            return value
        unwrapped = getattr(value, attr_name)
        assert isinstance(unwrapped, expected_type)
        return unwrapped

    for chunk in model:
        for module_name, module in chunk.named_modules():
            if isinstance(module, TransformerLayer):
                if not _is_language_transformer_layer(module_name):
                    continue
                adapter_model_prefix = (
                    f"base_model.model.model.layers.{module.layer_number - 1}"
                )
                if isinstance(module.self_attention, SelfAttention):
                    self_attention_linear_proj = _unwrap_attr(
                        module.self_attention.linear_proj,
                        "linear_proj",
                        TERowParallelLinear,
                    )
                    module.self_attention.linear_proj = SelfAttentionLinearProjLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_proj",
                        linear_proj=self_attention_linear_proj,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        provider=provider,
                    )
                    self_attention_linear_qkv = _unwrap_attr(
                        module.self_attention.linear_qkv,
                        "linear_qkv",
                        TELayerNormColumnParallelLinear,
                    )
                    module.self_attention.linear_qkv = SelfAttentionLinearQKVLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.self_attn",
                        linear_qkv=self_attention_linear_qkv,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        provider=provider,
                    )
                elif isinstance(module.self_attention, GatedDeltaNet):
                    gated_delta_net_out_proj = _unwrap_attr(
                        module.self_attention.out_proj,
                        "out_proj",
                        TERowParallelLinear,
                    )
                    module.self_attention.out_proj = SelfAttentionLinearProjLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.linear_attn.out_proj",
                        linear_proj=gated_delta_net_out_proj,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        provider=provider,
                    )
                    gated_delta_net_in_proj = _unwrap_attr(
                        module.self_attention.in_proj,
                        "in_proj",
                        TELayerNormColumnParallelLinear,
                    )
                    module.self_attention.in_proj = GatedDeltaNetInProjLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.linear_attn",
                        in_proj=gated_delta_net_in_proj,
                        gated_delta_net=module.self_attention,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                    )
                else:
                    raise TypeError(
                        "Unsupported self_attention module type for Megatron LoRA: "
                        f"{type(module.self_attention)}"
                    )
                experts = getattr(module.mlp, "experts", None)
                if experts is not None:
                    assert isinstance(experts, TEGroupedMLP)
                    mlp_experts_linear_fc1 = _unwrap_attr(
                        experts.linear_fc1,
                        "linear_fc1",
                        TEColumnParallelGroupedLinear,  # type: ignore[arg-type]
                    )
                    experts.linear_fc1 = MLPExpertsLinearFC1LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                        linear_fc1=mlp_experts_linear_fc1,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        num_local_experts=experts.num_local_experts,
                    )
                    mlp_experts_linear_fc2 = _unwrap_attr(
                        experts.linear_fc2,
                        "linear_fc2",
                        TERowParallelGroupedLinear,  # type: ignore[arg-type]
                    )
                    experts.linear_fc2 = MLPExpertsLinearFC2LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                        linear_fc2=mlp_experts_linear_fc2,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        num_local_experts=experts.num_local_experts,
                    )
                else:
                    mlp_linear_fc1 = _unwrap_attr(
                        module.mlp.linear_fc1,
                        "linear_fc1",
                        (TEColumnParallelLinear, TELayerNormColumnParallelLinear),
                    )
                    module.mlp.linear_fc1 = SharedExpertsLinearFC1LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp",
                        linear_fc1=mlp_linear_fc1,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                    )
                    mlp_linear_fc2 = _unwrap_attr(
                        module.mlp.linear_fc2,
                        "linear_fc2",
                        TERowParallelLinear,
                    )
                    module.mlp.linear_fc2 = SharedExpertsLinearFC2LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp",
                        linear_fc2=mlp_linear_fc2,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        provider=provider,
                    )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    assert isinstance(shared_experts, SharedExpertMLP)
                    shared_experts_linear_fc1 = _unwrap_attr(
                        shared_experts.linear_fc1,
                        "linear_fc1",
                        (TEColumnParallelLinear, TELayerNormColumnParallelLinear),
                    )
                    shared_experts.linear_fc1 = SharedExpertsLinearFC1LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.shared_expert",
                        linear_fc1=shared_experts_linear_fc1,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                    )
                    shared_experts_linear_fc2 = _unwrap_attr(
                        shared_experts.linear_fc2,
                        "linear_fc2",
                        TERowParallelLinear,
                    )
                    shared_experts.linear_fc2 = SharedExpertsLinearFC2LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.shared_expert",
                        linear_fc2=shared_experts_linear_fc2,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        provider=provider,
                    )
    return list(model)
