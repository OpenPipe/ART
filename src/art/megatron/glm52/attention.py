from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import get_pg_size
import torch

from art.megatron.glm52.cp_attention import context_parallel_sparse_mla
from art.megatron.glm52.indexer import (
    Glm52RoutedTopk,
    context_parallel_tree_topk,
    indexer_rope,
    streaming_tree_topk,
)
from art.megatron.glm52.sparse_mla import sparse_mla
from art.megatron.glm52.state import Glm52PrefixTreeState, require_glm52_state


def _tensor(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _latent_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)
    return weight * normalized.to(x.dtype)


def _half_split_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)


class Glm52Indexer(torch.nn.Module):
    def __init__(
        self,
        config: MLATransformerConfig,
        *,
        linear_builder: Any,
        norm_builder: Any,
        tp_group: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.heads = int(not_none(config.dsa_indexer_n_heads))
        self.head_dim = int(not_none(config.dsa_indexer_head_dim))
        self.topk = int(not_none(config.dsa_indexer_topk))
        linear_kwargs = {
            "config": config,
            "init_method": config.init_method,
            "bias": False,
            "skip_bias_add": False,
            "skip_weight_param_allocation": False,
            "parallel_mode": "duplicated",
        }
        self.linear_wq_b = build_module(
            linear_builder,
            config.q_lora_rank,
            self.heads * self.head_dim,
            tp_comm_buffer_name="glm52_index_q",
            **linear_kwargs,
        )
        self.linear_wk = build_module(
            linear_builder,
            config.hidden_size,
            self.head_dim,
            tp_comm_buffer_name="glm52_index_k",
            **linear_kwargs,
        )
        norm_config = deepcopy(config)
        norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            norm_builder,
            config=norm_config,
            hidden_size=self.head_dim,
            eps=1e-6,
        )
        self.linear_weights_proj = build_module(
            linear_builder,
            config.hidden_size,
            self.heads,
            tp_comm_buffer_name="glm52_index_weights",
            **linear_kwargs,
        )
        self.requires_grad_(False)

    def _gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.config.sequence_parallel or get_pg_size(self.tp_group) == 1:
            return tensor
        return gather_from_sequence_parallel_region(
            tensor,
            tensor_parallel_output_grad=False,
            group=self.tp_group,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        state: Glm52PrefixTreeState,
    ) -> torch.Tensor | Glm52RoutedTopk:
        q = _tensor(self.linear_wq_b(q_residual)).view(
            q_residual.shape[0], q_residual.shape[1], self.heads, self.head_dim
        )
        k = _tensor(self.linear_wk(hidden_states))
        k = _tensor(apply_module(self.k_norm)(k)).to(q.dtype)
        weights = _tensor(self.linear_weights_proj(hidden_states)).float()
        q = self._gather_sequence(q)
        k = self._gather_sequence(k)
        weights = self._gather_sequence(weights)
        expected = (q.shape[1], q.shape[0])
        if state.position_ids.shape != expected:
            raise RuntimeError(
                "GLM-5.2 indexer state/token shape mismatch: "
                f"state={tuple(state.position_ids.shape)} tokens={expected}."
            )
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        q, k = indexer_rope(q, k, state.rope_cos, state.rope_sin)
        weights = weights.permute(1, 0, 2).contiguous()
        weights *= (self.heads * self.head_dim) ** -0.5
        if state.context_parallel_state is not None:
            return context_parallel_tree_topk(q, k, weights, state, topk=self.topk)
        return streaming_tree_topk(
            q.contiguous(),
            k.contiguous(),
            weights,
            state.indexer_rows,
            topk=self.topk,
        )


class Glm52SparseCore(torch.nn.Module):
    def __init__(
        self,
        *,
        config: MLATransformerConfig,
        layer_number: int,
        pg_collection: ProcessGroupCollection,
        linear_builder: Any,
        norm_builder: Any,
        **_: Any,
    ) -> None:
        super().__init__()
        pattern = tuple(getattr(config, "glm52_indexer_types"))
        layer_index = int(layer_number) - 1
        if not 0 <= layer_index < len(pattern):
            raise ValueError(
                f"GLM-5.2 layer index {layer_index} is outside its index pattern."
            )
        full_layers = [
            index for index in range(layer_index + 1) if pattern[index] == "full"
        ]
        if not full_layers:
            raise ValueError(
                f"GLM-5.2 shared index layer {layer_index} has no preceding full layer."
            )
        self.full_layer_index = full_layers[-1]
        self.indexer = (
            Glm52Indexer(
                config,
                linear_builder=linear_builder,
                norm_builder=norm_builder,
                tp_group=pg_collection.tp,
            )
            if pattern[layer_index] == "full"
            else None
        )

    def topk(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        state: Glm52PrefixTreeState,
    ) -> torch.Tensor | Glm52RoutedTopk:
        if self.indexer is not None:
            indices = self.indexer(hidden_states.detach(), q_residual.detach(), state)
            state.topk_by_full_layer[self.full_layer_index] = indices
            return indices
        indices = state.topk_by_full_layer.get(self.full_layer_index)
        if indices is None:
            raise RuntimeError(
                "GLM-5.2 shared index layer ran before its full index layer "
                f"{self.full_layer_index}."
            )
        return indices


def glm52_core_builder(linear_builder: Any, norm_builder: Any):
    return partial(
        Glm52SparseCore,
        linear_builder=linear_builder,
        norm_builder=norm_builder,
    )


class Glm52SelfAttention(Attention):
    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: Any,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str = "self",
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(
            config,
            submodules,
            layer_number,
            attn_mask_type,
            attention_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.config: MLATransformerConfig
        q_down_kwargs = {
            "parallel_mode": "duplicated",
            "skip_weight_param_allocation": False,
        }
        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            config.hidden_size,
            config.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            tp_comm_buffer_name="q_down_proj",
            **q_down_kwargs,
        )
        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            config.q_lora_rank,
            config.num_attention_heads
            * (config.qk_head_dim + config.qk_pos_emb_head_dim),
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up_proj",
            tp_group=self.tp_group,
        )
        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            config.hidden_size,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            tp_comm_buffer_name="kv_down_proj",
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
        )
        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_head_dim + config.v_head_dim),
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
            tp_group=self.tp_group,
        )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            config=config,
            hidden_size=config.q_lora_rank,
            eps=1e-6,
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            config=config,
            hidden_size=config.kv_lora_rank,
            eps=1e-6,
        )
        self.softmax_scale = (config.qk_head_dim + config.qk_pos_emb_head_dim) ** -0.5
        self.q_a_lora: Any = None
        self.q_b_lora: Any = None
        self.kv_a_lora: Any = None

    def get_query_key_value_tensors(self, *args: Any, **kwargs: Any):
        del args, kwargs
        raise RuntimeError("GLM-5.2 uses its absorbed sparse-MLA forward path.")

    def _gather_replicated_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.config.sequence_parallel or get_pg_size(self.tp_group) == 1:
            return tensor
        return gather_from_sequence_parallel_region(
            tensor,
            tensor_parallel_output_grad=False,
            group=self.tp_group,
        )

    def _column_lora_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if get_pg_size(self.tp_group) == 1:
            return tensor
        if self.config.sequence_parallel:
            return gather_from_sequence_parallel_region(tensor, group=self.tp_group)
        return copy_to_tensor_model_parallel_region(tensor, group=self.tp_group)

    @torch.compiler.disable
    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_bias: Any = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask
        state = require_glm52_state(attention_bias)
        q_compressed = _tensor(self.linear_q_down_proj(hidden_states))
        if self.q_a_lora is not None:
            q_compressed = q_compressed + self.q_a_lora(hidden_states)
        q_residual = _latent_rms_norm(q_compressed, self.q_layernorm.weight)
        q = _tensor(self.linear_q_up_proj(q_residual))
        if self.q_b_lora is not None:
            q = q + self.q_b_lora(self._column_lora_input(q_residual))
        kv_combined = _tensor(self.linear_kv_down_proj(hidden_states))
        if self.kv_a_lora is not None:
            kv_combined = kv_combined + self.kv_a_lora(hidden_states)
        kv_compressed, k_rope = kv_combined.split(
            (self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim), dim=-1
        )
        kv_compressed = _latent_rms_norm(kv_compressed, self.kv_layernorm.weight)
        kv_compressed = self._gather_replicated_sequence(kv_compressed)
        k_rope = self._gather_replicated_sequence(k_rope)
        seq_len, batch = kv_compressed.shape[:2]
        heads = self.num_attention_heads_per_partition
        q = q.view(
            seq_len,
            batch,
            heads,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
        )
        q_nope, q_rope = q.split(
            (self.config.qk_head_dim, self.config.qk_pos_emb_head_dim), dim=-1
        )
        if state.rope_cos.shape[:2] != (batch, seq_len):
            raise RuntimeError(
                "GLM-5.2 RoPE state does not match the attention tokens: "
                f"layer={self.layer_number}, rope={tuple(state.rope_cos.shape[:2])}, "
                f"tokens={(batch, seq_len)}, hidden={tuple(hidden_states.shape)}"
            )
        cos = state.rope_cos.permute(1, 0, 2).unsqueeze(2).to(q.dtype)
        sin = state.rope_sin.permute(1, 0, 2).unsqueeze(2).to(q.dtype)
        q_rope = _interleaved_rope(q_rope, cos, sin)
        k_rope = _interleaved_rope(k_rope.unsqueeze(2), cos, sin).squeeze(2)

        kv_weight = self.linear_kv_up_proj.weight.view(
            heads,
            self.config.qk_head_dim + self.config.v_head_dim,
            self.config.kv_lora_rank,
        )
        key_weight, value_weight = kv_weight.split(
            (self.config.qk_head_dim, self.config.v_head_dim), dim=1
        )
        q_absorbed = torch.einsum("sbhd,hdm->sbhm", q_nope, key_weight)
        q_absorbed = torch.cat((q_absorbed, q_rope), dim=-1)
        kv_absorbed = torch.cat((kv_compressed, k_rope), dim=-1)
        core = self.core_attention
        if not isinstance(core, Glm52SparseCore):
            raise TypeError(f"Expected Glm52SparseCore, got {type(core).__name__}.")
        topk = core.topk(hidden_states, q_residual, state)
        q_absorbed = q_absorbed.permute(1, 0, 2, 3).contiguous()
        kv_absorbed = kv_absorbed.permute(1, 0, 2).contiguous()
        latent_out = (
            context_parallel_sparse_mla(
                q_absorbed,
                kv_absorbed,
                topk,
                state,
                scale=self.softmax_scale,
                tp_group=self.tp_group if get_pg_size(self.tp_group) > 1 else None,
            )
            if isinstance(topk, Glm52RoutedTopk)
            else sparse_mla(
                q_absorbed,
                kv_absorbed,
                topk,
                scale=self.softmax_scale,
                tp_group=self.tp_group if get_pg_size(self.tp_group) > 1 else None,
            )
        )
        value_out = torch.einsum("bshm,hdm->bshd", latent_out, value_weight)
        value_out = value_out.permute(1, 0, 2, 3).reshape(
            seq_len, batch, heads * self.config.v_head_dim
        )
        output, _bias = self.linear_proj(value_out)
        return output, None

    def backward_dw(self) -> None:
        self.linear_q_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()
        self.linear_proj.backward_dw()
