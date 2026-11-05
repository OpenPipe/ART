"""GLM-5.3 Flash Kimi Delta Attention for MCore 0.17 and FLA 0.5.

The two-stage gates follow HF and Megatron's KDA, including the corrected
q/k/v-only input width and sigmoid output gate. Separate projections expose
exactly q/k/v/o to ART's LoRA wrappers. TP1/PP1 is the GLM MVP topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class KimiDeltaAttentionSubmodules:
    q_proj: Any = None
    k_proj: Any = None
    v_proj: Any = None
    o_proj: Any = None
    b_proj: Any = None
    f_a_proj: Any = None
    f_b_proj: Any = None
    g_a_proj: Any = None
    g_b_proj: Any = None


def _tensor(value: Any) -> Tensor:
    return value[0] if isinstance(value, tuple) else value


class _HeadNorm(nn.Module):
    def __init__(self, width: int, eps: float, **kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, **kwargs))
        self.variance_epsilon = eps

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:  # ty: ignore[invalid-method-override]
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        work = value.to(dtype)
        work = work * torch.rsqrt(
            work.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (work * self.weight.to(dtype) * gate.to(dtype).sigmoid()).to(value.dtype)


class KimiDeltaAttention(nn.Module):
    """MCore self-attention interface: [S,B,D] -> (output, None).

    Accepts HF or MCore config field names. ``backend='reference'`` is an
    explicit CPU oracle; CUDA execution always uses FLA. CP packed_seq_params
    carry *global*, unpadded sequence boundaries and contiguous local shards.
    ART attention_bias prefix trees may use arbitrary attention token layouts.
    """

    lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    q_proj: Any
    k_proj: Any
    v_proj: Any
    f_a_proj: Any
    f_b_proj: Any
    g_a_proj: Any
    g_b_proj: Any
    q_conv1d: nn.Conv1d
    k_conv1d: nn.Conv1d
    v_conv1d: nn.Conv1d

    def __init__(
        self,
        config: Any,
        submodules: KimiDeltaAttentionSubmodules | None = None,
        layer_number: int | None = None,
        pg_collection: Any = None,
        cp_comm_type: str | None = None,
        *,
        backend: str = "fla",
        device: Any = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del cp_comm_type, kwargs
        self.config = config
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.cp_group = getattr(pg_collection, "cp", None)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(
            getattr(config, "linear_num_heads", None) or config.linear_num_key_heads
        )
        self.head_dim = int(
            getattr(config, "linear_head_dim", None) or config.linear_key_head_dim
        )
        self.projection_size = self.num_heads * self.head_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.gate_lower_bound = getattr(
            config, "linear_lower_bound", getattr(config, "kda_gate_lower_bound", -5.0)
        )
        self.norm_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "layernorm_epsilon", 1e-6))
        )
        if int(getattr(config, "tensor_model_parallel_size", 1)) != 1 or getattr(
            config, "sequence_parallel", False
        ):
            raise ValueError(
                "GLM KDA currently requires TP1 without sequence parallelism"
            )
        for field, expected in (
            ("linear_num_value_heads", self.num_heads),
            ("linear_value_head_dim", self.head_dim),
        ):
            if getattr(config, field, expected) != expected:
                raise ValueError(
                    f"KDA requires equal q/k/v heads and dimensions: {field}"
                )
        if self.conv_kernel_size < 1:
            raise ValueError("KDA convolution kernel size must be positive")
        if backend not in ("fla", "reference"):
            raise ValueError(f"Unknown KDA backend: {backend}")
        self.backend = backend
        factory = {
            "device": device,
            "dtype": dtype or getattr(config, "params_dtype", torch.float32),
        }
        specs = submodules or KimiDeltaAttentionSubmodules()

        def linear(name: str, incoming: int, outgoing: int) -> nn.Module:
            spec = getattr(specs, name)
            if spec is None:
                return nn.Linear(incoming, outgoing, bias=False, **factory)
            from megatron.core.transformer.spec_utils import build_module

            options = dict(
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            if name == "o_proj":
                options.update(
                    input_is_parallel=True, tp_group=getattr(pg_collection, "tp", None)
                )
            elif name in ("f_a_proj", "g_a_proj"):
                options.update(
                    parallel_mode="duplicated", skip_weight_param_allocation=False
                )
            else:
                options.update(
                    gather_output=False, tp_group=getattr(pg_collection, "tp", None)
                )
            return build_module(spec, incoming, outgoing, **options)

        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(self, name, linear(name, self.hidden_size, self.projection_size))
        self.o_proj = linear("o_proj", self.projection_size, self.hidden_size)
        self.b_proj = linear("b_proj", self.hidden_size, self.num_heads)
        for prefix in ("f", "g"):
            setattr(
                self,
                f"{prefix}_a_proj",
                linear(f"{prefix}_a_proj", self.hidden_size, self.head_dim),
            )
            setattr(
                self,
                f"{prefix}_b_proj",
                linear(f"{prefix}_b_proj", self.head_dim, self.projection_size),
            )
        # HF marks conv1d, A_log and dt_bias as strict FP32 parameters.
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(
                self,
                name,
                nn.Conv1d(
                    self.projection_size,
                    self.projection_size,
                    self.conv_kernel_size,
                    groups=self.projection_size,
                    padding=self.conv_kernel_size - 1,
                    bias=False,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads, device=device, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(self.projection_size, device=device, dtype=torch.float32)
        )
        self.o_norm = _HeadNorm(self.head_dim, self.norm_eps, **factory)
        for parameter in self._fp32_parameters():
            setattr(parameter, "keep_in_fp32", True)

    def _fp32_parameters(self) -> tuple[nn.Parameter, ...]:
        return cast(
            tuple[nn.Parameter, ...],
            (
                self.A_log,
                self.dt_bias,
                self.q_conv1d.weight,
                self.k_conv1d.weight,
                self.v_conv1d.weight,
            ),
        )

    def _apply(self, fn: Any, recurse: bool = True) -> KimiDeltaAttention:
        # Preserve values as well as dtype (casting to BF16 and back loses bits).
        names = (
            "A_log",
            "dt_bias",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
        )
        saved = [
            (name, self.get_parameter(name).data, self.get_parameter(name).grad)
            for name in names
        ]
        result = super()._apply(fn, recurse)
        for name, value, grad in saved:
            parameter = self.get_parameter(name)
            parameter.data = (
                parameter.data.float()
                if value.is_meta
                else value.to(device=parameter.device, dtype=torch.float32)
            )
            setattr(parameter, "keep_in_fp32", True)
            if grad is not None:
                parameter.grad = (
                    parameter.grad.float()
                    if grad.is_meta and parameter.grad is not None
                    else grad.to(device=parameter.device, dtype=torch.float32)
                )
        return result

    def backward_dw(self) -> None:
        """MCore delayed weight-gradient hook (ordinary PyTorch needs no hook)."""
        for name in (
            *self.lora_targets,
            "b_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
        ):
            callback = getattr(getattr(self, name), "backward_dw", None)
            if callback is not None:
                callback()

    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        inference_context: Any = None,
        attention_bias: Any = None,
        packed_seq_params: Any = None,
        sequence_len_offset: Any = None,
        *,
        inference_params: Any = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None]:
        del attention_mask, sequence_len_offset, kwargs
        if (
            inference_context is not None
            or inference_params is not None
            or key_value_states is not None
        ):
            raise NotImplementedError(
                "KDA supports training self-attention without an inference cache"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("KDA hidden states must be [S,B,hidden_size]")
        group = self.cp_group or getattr(attention_bias, "cp_group", None)
        cp_size = dist.get_world_size(group) if group is not None else 1
        if cp_size == 1 and int(getattr(self.config, "context_parallel_size", 1)) > 1:
            raise ValueError(
                "KDA CP requires an explicit pg_collection.cp or attention_bias.cp_group"
            )
        if getattr(attention_bias, "group_ids", None) is not None:
            if packed_seq_params is not None:
                raise ValueError("Use either ART prefix metadata or PackedSeqParams")
            from .kda_cp import prefix_tree_forward

            return prefix_tree_forward(self, hidden_states, attention_bias, group), None
        cu_seqlens = None
        if packed_seq_params is not None:
            if packed_seq_params.qkv_format != "thd" or hidden_states.shape[1] != 1:
                raise ValueError("Packed KDA requires THD with batch dimension one")
            cu_seqlens = packed_seq_params.cu_seqlens_q
            other = packed_seq_params.cu_seqlens_kv
            if (
                cu_seqlens is None
                or other is None
                or not torch.equal(cu_seqlens, other)
            ):
                raise ValueError(
                    "Packed KDA requires identical Q/KV sequence boundaries"
                )
            padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
            if padded is not None and not torch.equal(padded, cu_seqlens):
                raise ValueError(
                    "KDA requires compact packed tokens without internal padding"
                )
        if cp_size > 1 and cu_seqlens is None:
            if hidden_states.shape[1] != 1:
                raise ValueError("Dense KDA CP requires batch dimension one")
            cu_seqlens = torch.tensor(
                [0, hidden_states.shape[0] * cp_size],
                device=hidden_states.device,
                dtype=torch.int32,
            )
        return self._forward_packed(hidden_states, cu_seqlens, group), None

    def _forward_packed(
        self, hidden: Tensor, cu: Tensor | None, group: Any = None
    ) -> Tensor:
        from .kda_cp import build_context, validate_boundaries

        cp_size = dist.get_world_size(group) if group is not None else 1
        if cu is not None:
            validate_boundaries(cu, hidden.shape[0] * cp_size)
        # Clone boundaries on every call: FLA's tensor-identity cache must behave
        # identically during activation-checkpoint replay (prime-rl).
        cu = cu.clone() if cu is not None else None
        context = (
            build_context(cu, group, self.conv_kernel_size)
            if cp_size > 1 and self.backend == "fla"
            else None
        )
        x = hidden.transpose(0, 1).contiguous()
        mixed = torch.cat(
            [
                _tensor(getattr(self, name)(x))
                for name in ("q_proj", "k_proj", "v_proj")
            ],
            dim=-1,
        )
        weight = torch.cat(
            [
                getattr(self, name).weight
                for name in ("q_conv1d", "k_conv1d", "v_conv1d")
            ]
        ).squeeze(1)
        if self.backend == "reference":
            from .kda_cp import reference_gather
            from .kda_reference import causal_conv

            if cp_size > 1:
                mixed = reference_gather(mixed, group)
            convolved = causal_conv(mixed, weight, cu)
        else:
            from fla.modules.convolution import causal_conv1d

            convolution = dict(
                x=mixed,
                weight=weight,
                bias=None,
                activation="silu",
                output_final_state=False,
            )
            if context is None:
                convolved, _ = causal_conv1d(**convolution, cu_seqlens=cu)  # ty: ignore[call-non-callable]
            else:
                convolved, _ = _cp_convolution(**convolution, cp_context=context)
        shape = (*convolved.shape[:2], self.num_heads, self.head_dim)
        q, k, v = [
            part.reshape(shape).contiguous() for part in convolved.chunk(3, dim=-1)
        ]
        raw_g = _tensor(self.f_b_proj(_tensor(self.f_a_proj(x))))
        beta = _tensor(self.b_proj(x)).float().sigmoid()
        if self.backend == "reference" and cp_size > 1:
            from .kda_cp import reference_gather

            raw_g = reference_gather(raw_g, group)
            beta = reference_gather(beta, group)
        work_dtype = torch.float64 if raw_g.dtype == torch.float64 else torch.float32
        raw_g = (raw_g.to(work_dtype) + self.dt_bias.to(work_dtype)).reshape(shape)
        rate = self.A_log.to(work_dtype).exp().view(1, 1, self.num_heads, 1)
        if self.gate_lower_bound is None:
            decay = -rate * F.softplus(raw_g)
        else:
            decay = float(self.gate_lower_bound) * (rate * raw_g).sigmoid()
        if self.backend == "reference":
            from .kda_reference import recurrent_kda

            output = recurrent_kda(q, k, v, decay, beta, cu_seqlens=cu)
            if cp_size > 1:
                rank = dist.get_rank(group)
                output = output[:, rank * x.shape[1] : (rank + 1) * x.shape[1]]
        else:
            from fla.ops.kda import chunk_kda

            output, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=decay.contiguous(),
                beta=beta.contiguous(),
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                # 0.5.0's optimized safe_gate path predates the causality fix.
                safe_gate=False,
                cu_seqlens=context.cu_seqlens if context is not None else cu,
                cp_context=context,
                initial_state=None,
                output_final_state=False,
            )
        gate = _tensor(self.g_b_proj(_tensor(self.g_a_proj(x))))
        output = self.o_norm(output, gate.reshape_as(output)).flatten(-2)
        return _tensor(self.o_proj(output)).transpose(0, 1).contiguous()


@torch.compiler.disable
def _cp_convolution(**kwargs: Any) -> Any:
    # torch.compile currently lowers CP all-gather into the wrong output shape.
    from fla.modules.convolution import causal_conv1d

    return causal_conv1d(**kwargs)  # ty: ignore[call-non-callable]
