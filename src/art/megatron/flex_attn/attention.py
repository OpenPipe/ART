"""Flex attention plumbing for ART's Megatron backend."""

import math
from typing import Any, cast

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import (
    flex_backend_for_head_dims,
    get_dense_compiled_flex_attention,
)


class SharedPrefixAttentionState(BaseModel):
    """Shared-prefix sparsity metadata for one packed ART training sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    block_mask: BlockMask
    sliding_block_masks: dict[int, BlockMask] = Field(default_factory=dict)

    def block_mask_for_window(self, window: int | None) -> BlockMask:
        if window is None:
            return self.block_mask
        return self.sliding_block_masks[int(window)]


class FlexAttentionWrapper(torch.nn.Module):
    """Compiled `flex_attention` wrapper with Torchtitan-style inductor options."""

    def __init__(
        self,
        *,
        triton_num_stages_2_head_dims: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.triton_num_stages_2_head_dims = tuple(
            int(dim) for dim in triton_num_stages_2_head_dims
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        block_mask: BlockMask,
        scale: float,
        enable_gqa: bool,
    ) -> Tensor:
        # q, k, v are [B, H, S, D] tensors expected by torch.flex_attention.
        backend = flex_backend_for_head_dims(
            head_dim=int(q.shape[-1]),
            head_dim_v=int(v.shape[-1]),
        )
        return cast(
            Tensor,
            get_dense_compiled_flex_attention(
                backend=backend,
                head_dim=int(q.shape[-1]),
                head_dim_v=int(v.shape[-1]),
                triton_num_stages_2_head_dims=self.triton_num_stages_2_head_dims,
            )(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
            ),
        )


def create_shared_prefix_attention_state(
    group_ids: Tensor,
    parent_ids: Tensor,
    *,
    input_pos: Tensor | None = None,
    sliding_windows: tuple[int, ...] = (),
) -> SharedPrefixAttentionState:
    """Build a compiled block mask for ART shared-prefix packing.

    Initialized on the device of the group_ids tensor.

    Args:
        group_ids: `[B, S]` group id for each token in a packed sequence.
        parent_ids: `[B, S]` parent group id for each token in a packed sequence.
    """

    from art.megatron.shared_prefix_state import create_shared_prefix_state

    return create_shared_prefix_state(
        group_ids,
        parent_ids,
        input_pos=input_pos,
        sliding_windows=sliding_windows,
    )


class FlexDotProductAttention(torch.nn.Module):
    """Megatron core-attention module backed by compiled torch flex attention.

    The current implementation lacks support for fp8 and context parallelism (which are available in TEDotProductAttention)
    """

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
        self.flex_attention = FlexAttentionWrapper(
            triton_num_stages_2_head_dims=_triton_num_stages_2_head_dims(config)
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
            self.config.num_attention_heads, tp_world_size
        )
        self.num_query_groups_per_partition = divide(num_query_groups, tp_world_size)

        if softmax_scale is None:
            head_dim = divide(projection_size, self.config.num_attention_heads)
            self.softmax_scale = 1.0 / math.sqrt(head_dim)
        else:
            self.softmax_scale = softmax_scale

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
        """Compute self attention with compiled flex kernels.

        Args:
            query: `[S, B, Hq, D]`
            key: `[S, B, Hkv, D]`
            value: `[S, B, Hkv, D]`
            attention_mask: unused placeholder tensor kept for Megatron checkpoint API.
            attention_bias: `SharedPrefixAttentionState` or `BlockMask`.
        """

        del attention_mask, attn_mask_type
        assert packed_seq_params is None, (
            "PackedSeqParams is not used in ART Megatron flex path."
        )

        if isinstance(attention_bias, SharedPrefixAttentionState):
            block_mask = attention_bias.block_mask_for_window(
                getattr(self, "art_sliding_window", None)
            )
        else:
            assert isinstance(attention_bias, BlockMask), (
                "Expected a flex BlockMask in attention_bias."
            )
            block_mask = attention_bias

        # Megatron uses [S, B, H, D], while flex attention expects [B, H, S, D].
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        out = self.flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=self.num_attention_heads_per_partition
            != self.num_query_groups_per_partition,
        )

        # Return to Megatron's expected layout [S, B, Hq*D].
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.view(out.size(0), out.size(1), self.hidden_size_per_partition)
        return out


def _triton_num_stages_2_head_dims(config: Any) -> tuple[int, ...]:
    compile_crash_config = getattr(config, "art_flex_compile_crash_config", None)
    return tuple(
        int(dim)
        for dim in getattr(
            compile_crash_config,
            "triton_num_stages_2_head_dims",
            (),
        )
    )
