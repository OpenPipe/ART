from __future__ import annotations

import math
from typing import Any, cast

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.attention import (
    FlexAttentionWrapper,
    SharedPrefixAttentionState,
    _configure_softmax_offset,
    _triton_num_stages_2_head_dims,
)

from .executor import run_context_parallel
from .types import ArtContextParallelState


class ArtContextParallelCoreAttention(torch.nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        del attn_mask_type, attention_type, attention_dropout, cp_comm_type
        self.layer_number = int(layer_number)
        self.config = config
        self.triton_num_stages_2_head_dims = _triton_num_stages_2_head_dims(config)
        self.dense_kernel = FlexAttentionWrapper(
            triton_num_stages_2_head_dims=self.triton_num_stages_2_head_dims
        )

        if pg_collection is None:
            tp_world_size = self.config.tensor_model_parallel_size
        else:
            tp_world_size = pg_collection.tp.size()

        kv_channels = self.config.kv_channels
        assert kv_channels is not None, "Megatron config must provide kv_channels."
        projection_size = kv_channels * self.config.num_attention_heads
        self.hidden_size_per_partition = divide(projection_size, tp_world_size)
        num_query_groups = (
            self.config.num_query_groups or self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads,
            tp_world_size,
        )
        self.num_query_groups_per_partition = divide(num_query_groups, tp_world_size)

        if softmax_scale is None:
            head_dim = divide(projection_size, self.config.num_attention_heads)
            self.softmax_scale = 1.0 / math.sqrt(head_dim)
        else:
            self.softmax_scale = softmax_scale
        _configure_softmax_offset(
            self,
            self.config,
            self.num_attention_heads_per_partition,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Any = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        del attention_mask, attn_mask_type
        assert packed_seq_params is None, (
            "PackedSeqParams is not used in the ART context parallel attention path."
        )

        if isinstance(attention_bias, ArtContextParallelState):
            assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4, (
                "ART context parallel attention expects [S, B, H, D] inputs."
            )
            assert query.size(1) == 1 and key.size(1) == 1 and value.size(1) == 1, (
                "ART context parallel attention only supports exactly one packed sequence at a time."
            )
            out = run_context_parallel(
                query=query,
                key=key,
                value=value,
                state=attention_bias,
                scale=self.softmax_scale,
                enable_gqa=self.num_attention_heads_per_partition
                != self.num_query_groups_per_partition,
                compile_enabled=True,
                sliding_window=getattr(self, "art_sliding_window", None),
                triton_num_stages_2_head_dims=self.triton_num_stages_2_head_dims,
                softmax_offset=cast(Tensor | None, getattr(self, "softmax_offset")),
            )
        else:
            if isinstance(attention_bias, SharedPrefixAttentionState):
                block_mask = attention_bias.block_mask_for_window(
                    getattr(self, "art_sliding_window", None)
                )
            else:
                assert isinstance(attention_bias, BlockMask), (
                    "Expected ArtContextParallelState, SharedPrefixAttentionState, or BlockMask in attention_bias."
                )
                block_mask = attention_bias
            q = query.permute(1, 2, 0, 3)
            k = key.permute(1, 2, 0, 3)
            v = value.permute(1, 2, 0, 3)
            out_dense = self.dense_kernel(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.softmax_scale,
                enable_gqa=self.num_attention_heads_per_partition
                != self.num_query_groups_per_partition,
                softmax_offset=cast(Tensor | None, getattr(self, "softmax_offset")),
            )
            out = out_dense.permute(2, 0, 1, 3).contiguous()

        out = out.reshape(out.size(0), out.size(1), self.hidden_size_per_partition)
        return out
