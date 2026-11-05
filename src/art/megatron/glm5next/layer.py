from __future__ import annotations

from typing import Any

from megatron.core.transformer.moe.router import TopKRouter
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from art.megatron.dsv4.hyper_connection import DeepSeekV4HyperConnectionUtil
from art.megatron.dsv4.layer import Dsv4TransformerLayer


class Glm5NextRMSNorm(nn.Module):
    """HF's cast-before-weight RMSNorm (also used after the mean stream head)."""

    def __init__(self, config: Any, hidden_size: int, eps: float, **_: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(dtype)


class Glm5NextFinalNorm(Glm5NextRMSNorm):
    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim == 4:
            # GLM's HF HyperHead and MCore 0.21's learned_output_contract=False.
            hidden_states = hidden_states.mean(dim=2)
        return super().forward(hidden_states)


class Glm5NextRouter(TopKRouter):
    """Keep the gate GEMM in FP32, including on Hopper's TE path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keep_fp32_parameters = ("weight",)
        setattr(self.weight, "_keep_fp32", True)
        self.weight.data = self.weight.data.float()

    def gating(self, input: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.float()
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(input.float(), self.weight.float(), bias)


class _Glm5NextHyperConnectionUtil(DeepSeekV4HyperConnectionUtil):
    def hc_pre_raw(
        self, x: Tensor, hc_fn: Tensor, hc_scale: Tensor, hc_base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().hc_pre_raw(x, hc_fn, hc_scale, hc_base)


class Glm5NextTransformerLayer(Dsv4TransformerLayer):
    """One HF layer: attention mHC followed by dense/MoE mHC.

    DSV4's layer owns the same six FP32 parameters and pre/post equations.
    Its learned final head and hash router are deliberately absent here.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hc_util = _Glm5NextHyperConnectionUtil(self.config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Any = None,
        context: Any = None,
        context_mask: Any = None,
        rotary_pos_emb: Any = None,
        rotary_pos_cos: Any = None,
        rotary_pos_sin: Any = None,
        rotary_pos_cos_sin: Any = None,
        attention_bias: Any = None,
        inference_context: Any = None,
        packed_seq_params: Any = None,
        sequence_len_offset: Any = None,
        padding_mask: Any = None,
        input_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Also fence direct model calls: KDA is the first layer, before any DSA.
        if attention_bias is not None and hasattr(attention_bias, "group_ids"):
            from art.megatron.glm5next.packing import validate_single_roots

            validate_single_roots(attention_bias.group_ids, attention_bias.parent_ids)
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            input_ids=input_ids,
            **kwargs,
        )
