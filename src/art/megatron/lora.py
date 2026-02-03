import math
from typing import Sequence

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe import grouped_gemm_util
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
import torch


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
        self._expert_offset = ps.get_expert_model_parallel_rank() * num_local_experts
        self.reset_lora_parameters()

    @property
    def num_local_experts(self) -> int:
        return self.A_T.shape[0] if self.A_T.ndim == 3 else 1

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA weights (A=Kaiming, B=zeros) like PEFT defaults."""
        if self.A_T.ndim == 3:
            for expert in range(self.A_T.shape[0]):
                torch.nn.init.kaiming_uniform_(self.A_T[expert].T, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.A_T.T, a=math.sqrt(5))
        torch.nn.init.zeros_(self.B_T)

    def load_lora(self, adapter_model: dict[str, torch.Tensor]) -> None:
        try:
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
        except KeyError:
            print("Unable to find LoRA weights for", self.adapter_model_prefix)
            self.reset_lora_parameters()

    def load_weights(
        self,
        adapter_model: dict[str, torch.Tensor],
        *,
        suffix: str,
        into: torch.nn.Parameter,
    ) -> None:
        self.load_weight(
            (
                torch.stack(
                    [
                        adapter_model[
                            f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{suffix}.weight"
                        ].T
                        for expert in range(self.num_local_experts)
                    ]
                )
                if self.num_local_experts > 1
                else adapter_model[f"{self.adapter_model_prefix}.{suffix}.weight"].T
            ),
            into=into,
        )

    def load_weight(self, weight: torch.Tensor, *, into: torch.nn.Parameter) -> None:
        setattr(into, "sharded", False)
        tp_world_size = ps.get_tensor_model_parallel_world_size()
        tp_rank = ps.get_tensor_model_parallel_rank()
        for axis in (-2, -1):
            if weight.shape[axis] == into.shape[axis]:
                continue
            # assume our param is tensor sharded along this axis
            assert weight.shape[axis] // tp_world_size == into.shape[axis], (
                f"Weight shape {weight.shape} does not match into shape {into.shape} along axis {axis}"
            )
            s = into.shape[axis]
            weight = weight.narrow(axis, tp_rank * s, s)
            setattr(into, "sharded", True)
        into.data.copy_(weight)
        into.requires_grad = True

    def sharded_lora_state_dict(self) -> dict[str, torch.Tensor]:
        if self.num_local_experts > 1:
            if ps.get_expert_data_parallel_rank() != 0:
                return {}
            return {
                f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{key}": param.data[
                    expert
                ].T
                for expert in range(self.num_local_experts)
                for key, param in (
                    ("lora_A.weight", self.A_T),
                    ("lora_B.weight", self.B_T),
                )
            }
        if ps.get_data_parallel_rank() != 0 or torch.all(self.A_T == 0):
            return {}
        return {
            f"{self.adapter_model_prefix}.{key}": param.data.T
            for key, param in (
                ("lora_A.weight", self.A_T),
                ("lora_B.weight", self.B_T),
            )
            if getattr(param, "sharded", False)
            or ps.get_tensor_model_parallel_rank() == 0
        }

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
            # If no tokens routed locally, return zeros
            if isinstance(bsz, torch.Tensor) and int(torch.count_nonzero(bsz)) == 0:
                return x.new_zeros((x.shape[0], self.B_T.shape[-1]))
            tmp = grouped_gemm_util.ops.gmm(x, self.A_T, bsz, trans_b=False)  # type: ignore[attr-defined]
            out = grouped_gemm_util.ops.gmm(tmp, self.B_T, bsz, trans_b=False)  # type: ignore[attr-defined]
            return out * self.scale
        else:
            return ((x @ self.A_T) @ self.B_T) * self.scale


class SelfAttentionLinearProjLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_proj: TERowParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.linear_proj = linear_proj
        assert isinstance(linear_proj.weight, torch.Tensor)
        self.lora = LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_proj.in_features,
            out_features=linear_proj.out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_proj.weight.dtype,
            device=linear_proj.weight.device,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_output, bias_output = self.linear_proj(x)
        assert isinstance(base_output, torch.Tensor)
        assert isinstance(bias_output, (torch.Tensor, type(None)))
        lora_output = self.lora(x)
        if (
            self.provider.sequence_parallel
            and self.provider.tensor_model_parallel_size > 1
        ):
            tp_rank = ps.get_tensor_model_parallel_rank()
            tokens_per_rank = base_output.shape[0]
            start = tp_rank * tokens_per_rank
            end = start + tokens_per_rank
            lora_output = lora_output[start:end]
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
        q_out_features = self.provider.kv_channels * self.provider.num_attention_heads
        kv_out_features = self.provider.kv_channels * self.provider.num_query_groups
        tp_world_size = ps.get_tensor_model_parallel_world_size()
        assert kv_out_features % tp_world_size == 0, (
            "kv_out_features must be divisible by tensor parallel size"
        )
        assert q_out_features % tp_world_size == 0, (
            "q_out_features must be divisible by tensor parallel size"
        )
        q_out_features_per_rank = q_out_features // tp_world_size
        kv_out_features_per_rank = kv_out_features // tp_world_size
        assert isinstance(linear_qkv.weight, torch.Tensor)
        self.q_proj_lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.q_proj",
            in_features=linear_qkv.in_features,
            out_features=q_out_features_per_rank,
            rank=rank,
            alpha=alpha,
            dtype=linear_qkv.weight.dtype,
            device=linear_qkv.weight.device,
        )
        self.k_proj_lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.k_proj",
            in_features=linear_qkv.in_features,
            out_features=kv_out_features_per_rank,
            rank=rank,
            alpha=alpha,
            dtype=linear_qkv.weight.dtype,
            device=linear_qkv.weight.device,
        )
        self.v_proj_lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.v_proj",
            in_features=linear_qkv.in_features,
            out_features=kv_out_features_per_rank,
            rank=rank,
            alpha=alpha,
            dtype=linear_qkv.weight.dtype,
            device=linear_qkv.weight.device,
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

        query = self.q_proj_lora(layernorm_output)
        key = self.k_proj_lora(layernorm_output)
        value = self.v_proj_lora(layernorm_output)

        assert isinstance(self.linear_qkv.config.kv_channels, int)
        query_4d = query.reshape(
            query.shape[0], query.shape[1], -1, self.linear_qkv.config.kv_channels
        )
        key_4d = key.reshape(
            key.shape[0], key.shape[1], -1, self.linear_qkv.config.kv_channels
        )
        value_4d = value.reshape(
            value.shape[0], value.shape[1], -1, self.linear_qkv.config.kv_channels
        )

        qkv_4d = torch.cat([query_4d, key_4d, value_4d], dim=2)
        adapter_output = qkv_4d.reshape(qkv_4d.shape[0], qkv_4d.shape[1], -1)

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
        assert isinstance(linear_fc1.weight0, torch.Tensor)
        self.gate_lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.gate_proj",
            in_features=linear_fc1.in_features,
            out_features=linear_fc1.out_features // 2,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc1.weight0.dtype,
            device=linear_fc1.weight0.device,
            num_local_experts=num_local_experts,
        )
        self.up_lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.up_proj",
            in_features=linear_fc1.in_features,
            out_features=linear_fc1.out_features // 2,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc1.weight0.dtype,
            device=linear_fc1.weight0.device,
            num_local_experts=num_local_experts,
        )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = self.linear_fc1(x, tokens_per_expert)
        gate_out = self.gate_lora(x, tokens_per_expert=tokens_per_expert)
        up_out = self.up_lora(x, tokens_per_expert=tokens_per_expert)
        adapter_out = torch.cat([gate_out, up_out], dim=1)
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
        self.lora = LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.down_proj",
            in_features=linear_fc2.in_features,
            out_features=linear_fc2.out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_fc2.weight0.dtype,
            device=linear_fc2.weight0.device,
            num_local_experts=num_local_experts,
        )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = self.linear_fc2(x, tokens_per_expert)
        adapter_out = self.lora(x, tokens_per_expert=tokens_per_expert)
        return base_out + adapter_out, bias_out


def apply_lora_adapters(
    model: Sequence[torch.nn.Module],
    provider: GPTModelProvider,
) -> None:
    with torch.no_grad():
        for chunk in model:
            for module in chunk.modules():
                if isinstance(module, TransformerLayer):
                    adapter_model_prefix = (
                        f"base_model.model.model.layers.{module.layer_number - 1}"
                    )
                    assert isinstance(module.self_attention, SelfAttention)
                    self_attention_linear_proj = module.self_attention.linear_proj
                    if not isinstance(self_attention_linear_proj, TERowParallelLinear):
                        self_attention_linear_proj = (
                            self_attention_linear_proj.linear_proj
                        )
                        assert isinstance(
                            self_attention_linear_proj, TERowParallelLinear
                        )
                    module.self_attention.linear_proj = SelfAttentionLinearProjLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_proj",
                        linear_proj=self_attention_linear_proj,
                        rank=1,
                        alpha=32,
                        provider=provider,
                    )
                    self_attention_linear_qkv = module.self_attention.linear_qkv
                    if not isinstance(
                        self_attention_linear_qkv, TELayerNormColumnParallelLinear
                    ):
                        self_attention_linear_qkv = self_attention_linear_qkv.linear_qkv
                        assert isinstance(
                            self_attention_linear_qkv, TELayerNormColumnParallelLinear
                        )
                    module.self_attention.linear_qkv = SelfAttentionLinearQKVLoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.self_attn",
                        linear_qkv=self_attention_linear_qkv,
                        rank=1,
                        alpha=32,
                        provider=provider,
                    )
                    assert isinstance(module.mlp.experts, TEGroupedMLP)
                    mlp_experts_linear_fc1 = module.mlp.experts.linear_fc1
                    if not isinstance(
                        mlp_experts_linear_fc1,
                        TEColumnParallelGroupedLinear,  # type: ignore
                    ):
                        mlp_experts_linear_fc1 = mlp_experts_linear_fc1.linear_fc1
                        assert isinstance(
                            mlp_experts_linear_fc1,
                            TEColumnParallelGroupedLinear,  # type: ignore
                        )
                    module.mlp.experts.linear_fc1 = MLPExpertsLinearFC1LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                        linear_fc1=mlp_experts_linear_fc1,
                        rank=1,
                        alpha=32,
                        num_local_experts=module.mlp.experts.num_local_experts,
                    )
                    mlp_experts_linear_fc2 = module.mlp.experts.linear_fc2
                    if not isinstance(
                        mlp_experts_linear_fc2,
                        TERowParallelGroupedLinear,  # type: ignore
                    ):
                        mlp_experts_linear_fc2 = mlp_experts_linear_fc2.linear_fc2
                        assert isinstance(
                            mlp_experts_linear_fc2,
                            TERowParallelGroupedLinear,  # type: ignore
                        )
                    module.mlp.experts.linear_fc2 = MLPExpertsLinearFC2LoRA(
                        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                        linear_fc2=mlp_experts_linear_fc2,
                        rank=1,
                        alpha=32,
                        num_local_experts=module.mlp.experts.num_local_experts,
                    )
