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
from torch.nn.attention.flex_attention import AuxOutput, AuxRequest, BlockMask

from art.megatron.flex_attn.compiled import (
    flex_backend_for_head_dims,
    get_dense_compiled_flex_attention,
    normalize_flex_lse,
)


class PrefixTreeAttentionState(BaseModel):
    """Prefix-tree sparsity metadata for one packed ART training sample."""

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
        softmax_offset: Tensor | None = None,
    ) -> Tensor:
        # q, k, v are [B, H, S, D] tensors expected by torch.flex_attention.
        backend = flex_backend_for_head_dims(
            head_dim=int(q.shape[-1]),
            head_dim_v=int(v.shape[-1]),
        )
        result = get_dense_compiled_flex_attention(
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
            return_aux=AuxRequest(lse=True) if softmax_offset is not None else None,
        )
        if softmax_offset is None:
            return cast(Tensor, result)
        out, aux = cast(tuple[Tensor, AuxOutput], result)
        lse = aux.lse
        if lse is None:
            raise RuntimeError("Compiled flex attention did not return lse.")
        lse = normalize_flex_lse(lse, backend=backend)
        sink = softmax_offset.to(device=lse.device, dtype=lse.dtype).view(1, -1, 1)
        scale_factor = torch.exp(lse - torch.logaddexp(lse, sink)).unsqueeze(-1)
        return out * scale_factor.to(dtype=out.dtype)


def _configure_softmax_offset(
    module: torch.nn.Module,
    config: TransformerConfig,
    num_attention_heads_per_partition: int,
) -> None:
    if config.softmax_type == "vanilla":
        setattr(module, "softmax_offset", None)
    elif config.softmax_type == "off-by-one":
        setattr(
            module,
            "softmax_offset",
            torch.zeros(
                num_attention_heads_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            ),
        )
    elif config.softmax_type == "learnable":
        module.register_parameter(
            "softmax_offset",
            torch.nn.Parameter(
                torch.empty(
                    num_attention_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            ),
        )
        if config.perform_initialization:
            assert config.init_method is not None
            softmax_offset = cast(torch.nn.Parameter, getattr(module, "softmax_offset"))
            setattr(module, "softmax_offset", config.init_method(softmax_offset))
    else:
        raise ValueError(f"Unsupported softmax_type: {config.softmax_type}")


def create_prefix_tree_attention_state(
    group_ids: Tensor,
    parent_ids: Tensor,
    *,
    input_pos: Tensor | None = None,
    sliding_windows: tuple[int, ...] = (),
) -> PrefixTreeAttentionState:
    """Build a compiled block mask for ART prefix-tree packing.

    Initialized on the device of the group_ids tensor.

    Args:
        group_ids: `[B, S]` group id for each token in a packed sequence.
        parent_ids: `[B, S]` parent group id for each token in a packed sequence.
    """

    from art.megatron.prefix_tree_state import create_prefix_tree_state

    return create_prefix_tree_state(
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
        """Compute self attention with compiled flex kernels.

        Args:
            query: `[S, B, Hq, D]`
            key: `[S, B, Hkv, D]`
            value: `[S, B, Hkv, D]`
            attention_mask: unused placeholder tensor kept for Megatron checkpoint API.
            attention_bias: `PrefixTreeAttentionState` or `BlockMask`.
        """

        del attention_mask, attn_mask_type
        assert packed_seq_params is None, (
            "PackedSeqParams is not used in ART Megatron flex path."
        )

        if isinstance(attention_bias, PrefixTreeAttentionState):
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
            softmax_offset=cast(Tensor | None, getattr(self, "softmax_offset")),
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
