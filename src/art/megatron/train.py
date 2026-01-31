# isort: off
import os

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
# isort: on

import gc
import json
import math
import shutil
import time
from typing import Any, cast

from megatron.bridge import AutoBridge
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
import torch

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from art.utils.group_aggregate import group_aggregate

_pinned_buffers: dict[str, torch.Tensor] = {}
_is_offloaded = False

QWEN3_235B_INSTRUCT = "Qwen/Qwen3-235B-A22B-Instruct-2507"

model_identifier = os.environ.get("MODEL_IDENTIFIER", QWEN3_235B_INSTRUCT)
bridge = AutoBridge.from_hf_pretrained(
    model_identifier,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
provider = bridge.to_megatron_provider()
provider.attention_backend = AttnBackend.fused
provider.recompute_granularity = "full"
provider.recompute_method = "uniform"
provider.recompute_num_layers = 1
provider.tensor_model_parallel_size = min(2, torch.cuda.device_count())
provider.context_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = torch.cuda.device_count()
provider.expert_tensor_parallel_size = 1
provider.moe_shared_expert_overlap = True
provider.moe_router_dtype = "fp32"

if provider.tensor_model_parallel_size > 1:
    provider.sequence_parallel = True


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


provider.register_pre_wrap_hook(lambda x: freeze_model(x) or x)

model = provider.provide_distributed_model(
    ddp_config=DistributedDataParallelConfig(),
    data_parallel_random_init=False,
)

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

for module in model:
    while not isinstance(module, GPTModel) and hasattr(module, "module"):
        module = module.module
    if isinstance(module, GPTModel):
        _preprocess = module._preprocess

        def _preprocess_hook(*args, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore
            table = preproc_output[1]  # [S,B,1,D] type: ignore
            D = table.size(-1)  # type: ignore
            table_flat = table.view(table.size(0), D)  # type: ignore
            # position_ids: [B, S]
            position_ids = kwargs["position_ids"]
            B, S = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))  # [B*S, D]
            gathered = gathered.view(B, S, D).permute(1, 0, 2).contiguous()  # [S, B, D]
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = _preprocess_hook  # type: ignore[attr-defined]


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
    ) -> None:
        super().__init__()
        self.linear_proj = linear_proj
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
        if provider.sequence_parallel and provider.tensor_model_parallel_size > 1:
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
    ) -> None:
        super().__init__()
        linear_qkv.return_layernorm_output = True
        linear_qkv.return_layernorm_output_gathered = True
        self.linear_qkv = linear_qkv
        assert provider.kv_channels is not None
        assert provider.num_query_groups is not None
        assert provider.num_attention_heads is not None
        q_out_features = provider.kv_channels * provider.num_attention_heads
        kv_out_features = provider.kv_channels * provider.num_query_groups
        tp_world_size = ps.get_tensor_model_parallel_world_size()
        assert kv_out_features % tp_world_size == 0, (
            "kv_out_features must be divisible by tensor parallel size"
        )
        assert q_out_features % tp_world_size == 0, (
            "q_out_features must be divisible by tensor parallel size"
        )
        q_out_features_per_rank = q_out_features // tp_world_size
        kv_out_features_per_rank = kv_out_features // tp_world_size
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
        self.linear_fc1 = linear_fc1
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
                    self_attention_linear_proj = self_attention_linear_proj.linear_proj
                    assert isinstance(self_attention_linear_proj, TERowParallelLinear)
                module.self_attention.linear_proj = SelfAttentionLinearProjLoRA(
                    adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_proj",
                    linear_proj=self_attention_linear_proj,
                    rank=1,
                    alpha=32,
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
                )
                assert isinstance(module.mlp.experts, TEGroupedMLP)
                mlp_experts_linear_fc1 = module.mlp.experts.linear_fc1
                if not isinstance(
                    mlp_experts_linear_fc1, TEColumnParallelGroupedLinear
                ):
                    mlp_experts_linear_fc1 = mlp_experts_linear_fc1.linear_fc1
                    assert isinstance(
                        mlp_experts_linear_fc1, TEColumnParallelGroupedLinear
                    )
                module.mlp.experts.linear_fc1 = MLPExpertsLinearFC1LoRA(
                    adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                    linear_fc1=mlp_experts_linear_fc1,
                    rank=1,
                    alpha=32,
                    num_local_experts=module.mlp.experts.num_local_experts,
                )
                mlp_experts_linear_fc2 = module.mlp.experts.linear_fc2
                if not isinstance(mlp_experts_linear_fc2, TERowParallelGroupedLinear):
                    mlp_experts_linear_fc2 = mlp_experts_linear_fc2.linear_fc2
                    assert isinstance(
                        mlp_experts_linear_fc2, TERowParallelGroupedLinear
                    )
                module.mlp.experts.linear_fc2 = MLPExpertsLinearFC2LoRA(
                    adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
                    linear_fc2=mlp_experts_linear_fc2,
                    rank=1,
                    alpha=32,
                    num_local_experts=module.mlp.experts.num_local_experts,
                )

optimizer = get_megatron_optimizer(
    config=OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
    ),
    model_chunks=model,  # type: ignore
)

if rank == 0:
    # Print the number of parameters in the optimizer, nicely formatted
    num_params = sum(
        p.numel()
        for group in optimizer.param_groups
        if not group["is_decoupled_lr"]
        for p in group["params"]
    )
    print(f"Number of parameters in optimizer: {num_params:,}")
    total_params = sum(p.numel() for m in model for p in m.parameters())
    percent = (num_params / total_params) * 100 if total_params > 0 else 0
    print(f"Optimizer parameters as percent of total: {percent:0.2f}%")


class TrainingJob(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig


def print0(*values: Any) -> None:
    if rank == 0:
        print(*values)


def offload_to_cpu() -> None:
    """Offload model params and optimizer state to CPU pinned memory."""
    global _is_offloaded, _pinned_buffers
    if _is_offloaded:
        return

    for chunk in model:
        for module in chunk.modules():
            for attr in ["A_T", "B_T"]:
                if not hasattr(module, attr):
                    continue
                param = getattr(module, attr)
                if (
                    not isinstance(param, torch.nn.Parameter)
                    or param.device.type != "cuda"
                ):
                    continue
                key = f"{id(module)}_{attr}"
                if (
                    key not in _pinned_buffers
                    or _pinned_buffers[key].shape != param.shape
                    or _pinned_buffers[key].dtype != param.dtype
                ):
                    _pinned_buffers[key] = torch.empty(
                        param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                    )
                _pinned_buffers[key].copy_(param.data, non_blocking=True)
                param.data = _pinned_buffers[key]

    # Offload remaining model parameters (including base weights).
    for chunk in model:
        for param in chunk.parameters():
            if not isinstance(param, torch.nn.Parameter) or param.device.type != "cuda":
                continue
            key = f"param_{id(param)}"
            if (
                key not in _pinned_buffers
                or _pinned_buffers[key].shape != param.shape
                or _pinned_buffers[key].dtype != param.dtype
            ):
                _pinned_buffers[key] = torch.empty(
                    param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                )
            _pinned_buffers[key].copy_(param.data, non_blocking=True)
            param.data = _pinned_buffers[key]

    for param_id, state in optimizer.optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                key = f"opt_{id(param_id)}_{k}"
                if (
                    key not in _pinned_buffers
                    or _pinned_buffers[key].shape != v.shape
                    or _pinned_buffers[key].dtype != v.dtype
                ):
                    _pinned_buffers[key] = torch.empty(
                        v.shape, dtype=v.dtype, device="cpu", pin_memory=True
                    )
                _pinned_buffers[key].copy_(v, non_blocking=True)
                state[k] = _pinned_buffers[key]

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    _is_offloaded = True
    if rank == 0:
        print("Offloaded model params and optimizer to CPU")


def reload_to_gpu(device: torch.device | str | None = None) -> None:
    """Reload model params and optimizer state to GPU."""
    global _is_offloaded
    if not _is_offloaded:
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    for chunk in model:
        for module in chunk.modules():
            for attr in ["A_T", "B_T"]:
                if not hasattr(module, attr):
                    continue
                param = getattr(module, attr)
                if (
                    not isinstance(param, torch.nn.Parameter)
                    or param.device.type != "cpu"
                ):
                    continue
                gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                gpu_tensor.copy_(param.data, non_blocking=True)
                param.data = gpu_tensor

    # Reload remaining model parameters (including base weights).
    for chunk in model:
        for param in chunk.parameters():
            if not isinstance(param, torch.nn.Parameter) or param.device.type != "cpu":
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

    for state in optimizer.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                gpu_tensor = torch.empty(v.shape, dtype=v.dtype, device=device)
                gpu_tensor.copy_(v, non_blocking=True)
                state[k] = gpu_tensor

    torch.cuda.synchronize()
    _is_offloaded = False
    if rank == 0:
        print("Reloaded LoRA params and optimizer to GPU")


def calculate_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> torch.Tensor:
    causal_mask = (
        torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )
        .unsqueeze(0)
        .expand(batch_size, seq_len, seq_len)
    )
    group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    parent_mask = parent_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    mask = causal_mask & (group_mask | parent_mask)
    return mask


offload_to_cpu()

while True:
    torch.distributed.barrier()
    jobs_dir = "/tmp/megatron_training_jobs"
    os.makedirs(jobs_dir, exist_ok=True)
    job_names = sorted(
        job_name for job_name in os.listdir(jobs_dir) if job_name.endswith(".json")
    )
    if not job_names:
        time.sleep(1)
        continue

    wake_lock_path = "/tmp/megatron_vllm_waking"
    while os.path.exists(wake_lock_path):
        time.sleep(0.2)

    reload_to_gpu()

    job_name = job_names[0]
    job_path = os.path.join(jobs_dir, job_name)
    with open(job_path, "rb") as f:
        job = TrainingJob.model_validate_json(f.read())
    config = job.config
    experimental_config = job.experimental_config
    print0("Loaded job from", job_path)
    print0("Job:", job)
    adapter_model_path = f"{job.lora_path}/adapter_model.safetensors"
    if os.path.exists(adapter_model_path):
        print0("Loading adapter model from", adapter_model_path)
        adapter_model = load_file(adapter_model_path)
        with torch.no_grad():
            for chunk in model:
                for module in chunk.modules():
                    if hasattr(module, "load_lora"):
                        module.load_lora(adapter_model)  # type: ignore
    else:
        print0("No adapter model found at", adapter_model_path)
        adapter_model = {}
        with torch.no_grad():
            for chunk in model:
                for module in chunk.modules():
                    if hasattr(module, "reset_lora_parameters"):
                        module.reset_lora_parameters()  # type: ignore
    optimizer_shard_path = os.path.join(
        job.optimizer_state_path, f"{rank + 1:02d}-of-{world_size:02d}.pt"
    )
    if os.path.exists(optimizer_shard_path):
        print(
            "Loading optimizer state from",
            optimizer_shard_path,
        )
        optimizer.load_state_dict(torch.load(optimizer_shard_path))
    else:
        # No checkpoint for this run; reset optimizer state to avoid cross-run leakage
        print(
            "No optimizer state found at",
            optimizer_shard_path,
            "— resetting optimizer for new run",
        )
        optimizer.optimizer.state.clear()
        optimizer.reload_model_params()
    print0("Loading packed tensors from", job.disk_packed_tensors["dir"])
    packed_tensors = packed_tensors_from_dir(**job.disk_packed_tensors)
    num_sequences = job.disk_packed_tensors["num_sequences"]
    dp_rank = ps.get_data_parallel_rank()
    dp_world_size = ps.get_data_parallel_world_size()
    indices = list(
        range(
            dp_rank,
            num_sequences,
            dp_world_size,
        )
    )
    # pad indices
    if num_sequences % dp_world_size <= dp_rank > 0:
        indices.append(
            (list(range(num_sequences)) * (dp_world_size // num_sequences + 1))[dp_rank]
        )
    for index in indices:
        inputs = PackedTensors(  # type: ignore
            **{
                key: value[index : index + 1]
                for key, value in packed_tensors.items()
                if isinstance(value, torch.Tensor)
            },
            pixel_values=[None],
            image_grid_thw=[None],
        )
        ref_logprobs = None
        device = next(model[0].parameters()).device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)  # type: ignore
        attention_mask = ~calculate_mask(
            batch_size=inputs["tokens"].shape[0],
            seq_len=inputs["tokens"].shape[1],
            device=device,
            group_ids=inputs["group_ids"],
            parent_ids=inputs["parent_ids"],
        ).unsqueeze(1)  # add head dimension [B, H=1, S, S]
        attention_bias = torch.where(
            attention_mask,
            torch.tensor(
                float("-inf"), dtype=next(model[0].parameters()).dtype, device=device
            ),
            torch.tensor(0.0, dtype=next(model[0].parameters()).dtype, device=device),
        )
        new_logprobs: torch.Tensor = -model[0](
            input_ids=inputs["tokens"],
            position_ids=inputs["input_pos"],
            attention_mask=attention_mask,
            labels=shift_tensor(inputs["tokens"], 0),
            extra_block_kwargs={"attention_bias": attention_bias},
        )
        loss = loss_fn(
            inputs,  # type: ignore
            new_logprobs,
            ref_logprobs,
            None,
            experimental_config,
        )
        probs_corr = loss.probs_corr.item()
        print0("Correlation between old and new probabilities:", probs_corr)
        loss = loss.mean_policy_loss + config.beta * loss.mean_kl
        loss.backward()
        # Reduce LoRA grads
        start = time.perf_counter()
        num_grads = 0
        for chunk in model:
            for param in chunk.parameters():
                if param.grad is None:
                    continue
                torch.distributed.all_reduce(
                    param.grad,
                    op=torch.distributed.ReduceOp.AVG,
                    group=ps.get_data_parallel_group(),
                )
                num_grads += 1
        print0(
            f"Reduced {num_grads} LoRA grads in {(time.perf_counter() - start) * 1e3:.1f} ms"
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate
        update_successful, grad_norm, num_zeros_in_grad = cast(
            tuple[bool, float, int | None], optimizer.step()
        )
        optimizer.zero_grad()

        # Mean reduce loss across all ranks for logging
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            with open("/tmp/megatron_training_log.jsonl", "a+") as log_file:
                log_msg = json.dumps(
                    {
                        "loss": loss.item(),
                        "grad_norm": grad_norm,
                        "probs_corr": probs_corr,
                    }
                )
                print("Logging", log_msg)
                log_file.write(log_msg + "\n")

    sharded_state_dict = {}
    for chunk in model:
        for module in chunk.modules():
            if hasattr(module, "sharded_lora_state_dict"):
                module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                    module.sharded_lora_state_dict()  # type: ignore
                )
                for key, value in module_sharded_lora_state_dict.items():
                    target_dtype = (
                        adapter_model[key].dtype
                        if key in adapter_model
                        else value.dtype
                    )
                    sharded_state_dict[key] = value.to(target_dtype)
    shard_path = os.path.join(
        job.lora_path,
        f"adapter_model-{rank + 1:02d}-of-{world_size:02d}.safetensors",
    )
    print("Saving adapter shard to", shard_path)
    save_file(sharded_state_dict, shard_path)
    print("Saving optimizer shard to", optimizer_shard_path)
    os.makedirs(job.optimizer_state_path, exist_ok=True)
    torch.save(optimizer.state_dict(), optimizer_shard_path)
    offload_to_cpu()
    # Ensure all ranks have finished saving before signaling completion
    torch.distributed.barrier()
    if rank == 0:
        os.remove(job_path)
        with open("/tmp/megatron_training_log.jsonl", "a+") as log_file:
            log_file.write("all done\n")
        shutil.rmtree(job.disk_packed_tensors["dir"])
