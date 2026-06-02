from typing import Any, cast

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor
import torch
from torch import Tensor
import torch.nn.functional as F

from art.megatron.dsv4.hyper_connection import (
    DeepSeekV4HyperConnectionUtil,
    HCHeadParams,
)


def _first_tensor(value: Any) -> Tensor:
    return value[0] if isinstance(value, tuple) else value


def _input_ids_sbd(
    input_ids: Tensor, hidden_states: Tensor, tp_group: Any | None
) -> Tensor:
    if input_ids.ndim != 2:
        raise ValueError(
            f"DSV4 hash routing expects 2D input_ids, got {tuple(input_ids.shape)}."
        )
    if input_ids.shape == hidden_states.shape[:2]:
        return input_ids
    if input_ids.t().shape == hidden_states.shape[:2]:
        return input_ids.t().contiguous()
    input_ids_sbd = input_ids.t().contiguous()
    if (
        tp_group is not None
        and tp_group.size() > 1
        and input_ids_sbd.shape[1] == hidden_states.shape[1]
        and input_ids_sbd.shape[0] % tp_group.size() == 0
    ):
        local_s = input_ids_sbd.shape[0] // tp_group.size()
        start = tp_group.rank() * local_s
        local_input_ids = input_ids_sbd[start : start + local_s]
        if local_input_ids.shape == hidden_states.shape[:2]:
            return local_input_ids
    raise ValueError(
        "DSV4 hash routing input_ids do not match hidden states: "
        f"input_ids={tuple(input_ids.shape)} hidden={tuple(hidden_states.shape)}."
    )


class Dsv4Router(TopKRouter):
    def __init__(self, config: TransformerConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self._dsv4_input_ids: Tensor | None = None

    def set_layer_number(self, layer_number: int):
        super().set_layer_number(layer_number)
        cfg = cast(Any, self.config)
        if self._is_hash_layer():
            if "tid2eid" not in self._buffers:
                vocab_size = int(getattr(cfg, "vocab_size", cfg.padded_vocab_size))
                self.register_buffer(
                    "tid2eid",
                    torch.zeros(vocab_size, self.topk, dtype=torch.long),
                    persistent=True,
                )
        elif "e_score_correction_bias" not in self._buffers:
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(int(cfg.num_moe_experts), dtype=torch.float32),
                persistent=True,
            )

    def set_input_ids(self, input_ids: Tensor | None) -> None:
        self._dsv4_input_ids = input_ids

    def _is_hash_layer(self) -> bool:
        cfg = cast(Any, self.config)
        return self.layer_number is not None and self.layer_number <= int(
            cfg.dsv4_n_hash_layers
        )

    def _scores(self, logits: Tensor) -> Tensor:
        if self.score_function == "sigmoid":
            return torch.sigmoid(logits)
        if self.score_function == "softmax":
            return F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        if self.score_function == "sqrtsoftplus":
            return F.softplus(logits.float()).sqrt().to(logits.dtype)
        raise ValueError(
            f"Unsupported DSV4 router score function {self.score_function!r}."
        )

    def routing(self, logits: Tensor, padding_mask: Tensor | None = None):
        cfg = cast(Any, self.config)
        num_moe_experts = int(cfg.num_moe_experts)
        logits = logits.view(-1, num_moe_experts)
        scores = self._scores(logits)
        if self._is_hash_layer():
            if self._dsv4_input_ids is None:
                raise RuntimeError(
                    "DSV4 hash router requires input_ids for hash-moe layers."
                )
            tid2eid = cast(Tensor, self.tid2eid)
            indices = tid2eid[self._dsv4_input_ids.reshape(-1)].long()
        else:
            selection_scores = scores
            e_score_correction_bias = getattr(self, "e_score_correction_bias", None)
            if e_score_correction_bias is not None:
                selection_scores = selection_scores + e_score_correction_bias
            indices = selection_scores.topk(self.topk, dim=-1, sorted=False).indices
        if indices.shape[-1] != self.topk:
            raise RuntimeError(
                "DSV4 router selected an invalid number of experts: "
                f"selected={indices.shape[-1]} expected={self.topk}."
            )
        probs = scores.gather(1, indices)
        if self.score_function != "softmax":
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
        probs = probs * float(cfg.moe_router_topk_scaling_factor)
        routing_map = F.one_hot(indices, num_classes=num_moe_experts).sum(dim=1).bool()
        if padding_mask is not None:
            valid = padding_mask.reshape(-1).bool()
            probs = torch.where(valid.unsqueeze(-1), probs, torch.zeros_like(probs))
            routing_map = routing_map & valid.unsqueeze(-1)
        return probs, routing_map


class Dsv4MoELayer(MoELayer):
    _dsv4_input_ids: Tensor | None = None

    def set_input_ids(self, input_ids: Tensor | None) -> None:
        self._dsv4_input_ids = input_ids

    def forward(
        self,
        hidden_states: Tensor,
        intermediate_tensors=None,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
    ):
        if isinstance(self.router, Dsv4Router):
            input_ids = input_ids if input_ids is not None else self._dsv4_input_ids
            router_input_ids = None
            if input_ids is not None:
                router_input_ids = _input_ids_sbd(
                    input_ids, hidden_states, getattr(self, "attn_tp_group", None)
                )
            self.router.set_input_ids(router_input_ids)
        try:
            return super().forward(
                hidden_states,
                intermediate_tensors=intermediate_tensors,
                padding_mask=padding_mask,
            )
        finally:
            if isinstance(self.router, Dsv4Router):
                self.router.set_input_ids(None)


class Dsv4TransformerLayer(TransformerLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = cast(Any, self.config)
        hc = int(cfg.dsv4_hc_mult)
        hc_dim = hc * self.config.hidden_size
        mix = (2 + hc) * hc
        self.hc_attn_fn = torch.nn.Parameter(
            torch.empty(mix, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = torch.nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.hc_attn_scale = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = torch.nn.Parameter(
            torch.empty(mix, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_base = torch.nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.hc_ffn_scale = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
        for param in (
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
        ):
            setattr(param, "_keep_fp32", True)
        self.hc_util = DeepSeekV4HyperConnectionUtil(self.config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context: Any = None,
        packed_seq_params: Any = None,
        sequence_len_offset: Tensor | None = None,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        **_: Any,
    ):
        if hidden_states.ndim == 3:
            hidden_states = self.hc_util.block_expand(hidden_states)

        attn_input, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        attn_input = _first_tensor(apply_module(self.input_layernorm)(attn_input))
        attn_output = self.self_attention(
            attn_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        hidden_states = self.hc_util.layer_post(attn_output, hidden_states, post, comb)

        mlp_input, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        mlp_input = _first_tensor(apply_module(self.pre_mlp_layernorm)(mlp_input))
        if isinstance(self.mlp, Dsv4MoELayer):
            mlp_output = self.mlp(
                mlp_input, padding_mask=padding_mask, input_ids=input_ids
            )
        else:
            mlp_output = self.mlp(mlp_input, padding_mask=padding_mask)
        hidden_states = self.hc_util.layer_post(mlp_output, hidden_states, post, comb)
        return make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        ), context


class Dsv4FinalNorm(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float,
    ) -> None:
        super().__init__(config=config)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hc_head_params = HCHeadParams(config)
        self.hc_util = DeepSeekV4HyperConnectionUtil(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim == 4:
            hidden_states = self.hc_util.block_head(
                hidden_states,
                self.hc_head_params.hc_head_fn,
                self.hc_head_params.hc_head_scale,
                self.hc_head_params.hc_head_base,
            )
        dtype = hidden_states.dtype
        normed = hidden_states.float()
        normed = normed * torch.rsqrt(normed.square().mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype)
