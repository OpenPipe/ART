import copy
from typing import Any, cast

import einops
import torch
import torch.nn as nn

# Enable TF32 for fp32 matmul to match the precision of the TileKernels MHC
# kernels (which use TF32 tensor-core GEMM for the HC fp32 mixer).  Without
# this, PyTorch's default ``allow_tf32=False`` keeps fp32 ``F.linear`` on the
# SIMT path, which introduces a ~1e-4 mean-abs gap vs the TileKernels output;
# matching TF32 brings the gap to <=1.5e-5 mean-abs (1 ULP bf16 max-abs).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from art.megatron.dsv4.compressor import (
    DeepSeekV4Compressor,
    Dsv4CompressionLayout,
    Dsv4PrefixTreeState,
    compressed_layout_topk_idxs,
)
from art.megatron.dsv4.kernel.tilelang_sparse_mla import sparse_attn_tilelang
from art.megatron.dsv4.rope import (
    apply_rotary_emb,
    configure_rope_cache,
    get_rope_cache,
    get_rope_cache_at_positions,
)
from art.megatron.dsv4.v4_indexer import V4Indexer
from art.megatron.prefix_tree import PrefixTreeRow, PrefixTreeSegment, parse_prefix_tree


def _window_topk_idxs(
    q_positions: torch.Tensor, *, window_size: int, bsz: int
) -> torch.Tensor:
    base = q_positions.unsqueeze(1)
    offsets = torch.arange(
        min(q_positions.numel(), window_size), device=q_positions.device
    )
    k_pos = (base - window_size + 1).clamp(0) + offsets
    topk = torch.where(k_pos > base, -1, k_pos)
    return topk.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32)


def _compress_topk_idxs(
    q_positions: torch.Tensor, *, ratio: int, bsz: int
) -> torch.Tensor:
    seqlen = int(q_positions.numel())
    offset = seqlen
    k_group_idx = torch.arange(seqlen // ratio, device=q_positions.device).repeat(
        seqlen, 1
    )
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    compress = torch.where(
        k_group_idx >= q_first_invalid_group, -1, k_group_idx + offset
    )
    return compress.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32)


def _prefix_tree_tensors(
    attention_bias: Any,
    *,
    bsz: int,
    seqlen: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    group_ids = getattr(attention_bias, "group_ids", None)
    parent_ids = getattr(attention_bias, "parent_ids", None)
    if group_ids is None or parent_ids is None:
        return None
    group_ids = group_ids.to(device=device, dtype=torch.long)
    parent_ids = parent_ids.to(device=device, dtype=torch.long)
    if group_ids.shape != (bsz, seqlen) or parent_ids.shape != (bsz, seqlen):
        raise ValueError(
            "DSV4 prefix-tree metadata must match attention input shape: "
            f"group_ids={tuple(group_ids.shape)} parent_ids={tuple(parent_ids.shape)} "
            f"expected={(bsz, seqlen)}."
        )
    return group_ids, parent_ids


def _prefix_tree_window_topk_idxs(
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    bsz, seqlen = group_ids.shape
    width = min(seqlen, window_size)
    position_cpu = position_ids.detach().cpu()
    rows = parse_prefix_tree(
        group_ids=group_ids.detach().cpu(),
        parent_ids=parent_ids.detach().cpu(),
    )
    topk = torch.full((bsz, seqlen, width), -1, dtype=torch.int32)
    if width == 0:
        return topk.to(device=group_ids.device)

    def branch_indices(
        row: PrefixTreeRow,
        segment: PrefixTreeSegment,
        by_group: dict[int, PrefixTreeSegment],
        positions_by_group: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        branch_len = int(positions_by_group[segment.group_id][-1].item()) + 1
        indices = torch.full((branch_len,), -1, dtype=torch.int32)
        for group_id in (*segment.ancestors, segment.group_id):
            path_segment = by_group[group_id]
            logical_positions = positions_by_group[group_id]
            physical_indices = torch.arange(
                path_segment.start,
                path_segment.end,
                dtype=torch.int32,
            )
            indices[logical_positions] = physical_indices
        if bool((indices < 0).any().item()):
            raise ValueError(
                "DSV4 prefix-tree SWA path must cover every logical position "
                f"from 0 through {branch_len - 1}."
            )
        return indices

    for row in rows:
        by_group = {candidate.group_id: candidate for candidate in row.segments}
        positions_by_group = {
            candidate.group_id: position_cpu[
                row.row_index, candidate.start : candidate.end
            ].to(torch.long)
            for candidate in row.segments
        }
        for segment in row.segments:
            indices = branch_indices(row, segment, by_group, positions_by_group)
            padded = torch.cat(
                (
                    torch.full((width - 1,), -1, dtype=torch.int32),
                    indices,
                )
            )
            windows = padded.unfold(0, width, 1)
            q_positions = position_cpu[row.row_index, segment.start : segment.end].to(
                torch.long
            )
            topk[row.row_index, segment.start : segment.end] = windows.index_select(
                0,
                q_positions,
            )
    return topk.to(device=group_ids.device)


def _dsv4_prefix_tree_state(attention_bias: Any) -> Dsv4PrefixTreeState:
    model_state = getattr(attention_bias, "model_state", None)
    state = model_state.get("dsv4") if isinstance(model_state, dict) else None
    if not isinstance(state, Dsv4PrefixTreeState):
        raise RuntimeError(
            "DSV4 prefix-tree state is missing. Build it once per packed "
            "sequence through the model-support prefix-tree state hook."
        )
    return state


def _dsv4_topk_cache(attention_bias: Any) -> dict[Any, Any]:
    return _dsv4_prefix_tree_state(attention_bias).topk_idx_cache


def _prefix_tree_window_topk_idxs_cached(
    attention_bias: Any,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    cache = _dsv4_topk_cache(attention_bias)
    key = ("raw_swa", int(window_size))
    cached = cache.get(key)
    if cached is None:
        cached = _prefix_tree_window_topk_idxs(
            position_ids,
            group_ids,
            parent_ids,
            window_size=window_size,
        ).contiguous()
        cache[key] = cached
    return cached


def _prefix_tree_compressed_topk_idxs_cached(
    attention_bias: Any,
    layout: Dsv4CompressionLayout,
    *,
    position_ids: torch.Tensor,
    ratio: int,
    offset: int,
) -> torch.Tensor:
    cache = _dsv4_topk_cache(attention_bias)
    key = ("compressed", int(ratio), int(offset))
    cached = cache.get(key)
    if cached is None:
        cached = (
            compressed_layout_topk_idxs(
                layout,
                position_ids=position_ids,
                offset=offset,
            )
            .to(torch.int32)
            .contiguous()
        )
        cache[key] = cached
    return cached


def _prefix_tree_i32_metadata_cached(
    attention_bias: Any,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cache = _dsv4_topk_cache(attention_bias)
    key = "indexer_q_metadata_i32"
    cached = cache.get(key)
    if cached is None:
        cached = position_ids.to(torch.int32).contiguous()
        cache[key] = cached
    return cached


def _prefix_tree_layout_indexer_metadata_cached(
    attention_bias: Any,
    layout: Dsv4CompressionLayout,
    *,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = _dsv4_topk_cache(attention_bias)
    key = ("indexer_layout_i32", int(ratio))
    cached = cache.get(key)
    if cached is None:
        cached = (
            layout.query_group_pre_order.to(torch.int32).contiguous(),
            layout.entry_group_pre_order.to(torch.int32).contiguous(),
            layout.entry_group_post_order.to(torch.int32).contiguous(),
            layout.entry_end_positions.to(torch.int32).contiguous(),
        )
        cache[key] = cached
    return cached


def _prefix_tree_compression_layout(
    attention_bias: Any,
    *,
    ratio: int,
) -> Dsv4CompressionLayout:
    layouts = _dsv4_prefix_tree_state(attention_bias).compression_layouts
    if ratio not in layouts:
        raise RuntimeError(
            "DSV4 prefix-tree compression layout was not prepared on the "
            f"attention state for ratio={ratio}. Build it once per packed "
            "sequence through the model-support prefix-tree state hook."
        )
    layout = layouts[ratio]
    if not isinstance(layout, Dsv4CompressionLayout):
        raise TypeError(f"Expected Dsv4CompressionLayout for ratio={ratio}.")
    return layout


def _add_lora_if_present(
    owner: nn.Module,
    attr_name: str,
    base: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    lora = getattr(owner, attr_name, None)
    if lora is None:
        return base
    return base + lora(x)


class DeepSeekV4Attention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str | None = None,
        cp_comm_type: str | None = None,
        pg_collection=None,
    ):
        super().__init__(config=config)
        cfg = cast(Any, config)
        init_method = config.init_method
        if init_method is None:
            raise RuntimeError("DeepSeek-V4 attention requires config.init_method.")

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        else:
            assert hasattr(pg_collection, "tp")
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = pg_collection.cp if hasattr(pg_collection, "cp") else None
        self.cp_size = self.cp_group.size() if self.cp_group else 1
        if self.cp_size != 1:
            raise RuntimeError(
                "DeepSeek-V4 non-CP attention received context_parallel_size > 1."
            )

        layer_id = layer_number - 1
        del layer_number

        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // config.tensor_model_parallel_size
        self.q_lora_rank = int(cfg.q_lora_rank)
        self.o_lora_rank = int(cfg.dsv4_o_lora_rank)
        self.head_dim = int(cfg.kv_lora_rank)
        self.rope_head_dim = int(cfg.qk_pos_emb_head_dim)
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = int(cfg.dsv4_o_groups)
        self.n_local_groups = self.n_groups // config.tensor_model_parallel_size
        self.window_size = int(cfg.dsv4_window_size)
        compress_ratios = cfg.dsv4_compress_ratios
        self.compress_ratio = int(compress_ratios[layer_id]) if compress_ratios else 0
        self.eps = config.layernorm_epsilon

        assert self.o_lora_rank == 1024
        assert self.head_dim == 512
        assert self.rope_head_dim == 64
        assert self.nope_head_dim == 448
        assert self.window_size == 128

        config_no_sp = copy.copy(config)
        config_no_sp.sequence_parallel = False

        attn_sink = torch.empty(self.n_local_heads, dtype=torch.float32)
        self._keep_fp32_parameters = ("attn_sink",)
        self._keep_fp32_buffers = ("attn_sink",)
        self.attn_sink = nn.Parameter(attn_sink)
        setattr(self.attn_sink, "_keep_fp32", True)

        self.wq_a = TELinear(
            self.dim,
            self.q_lora_rank,
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_norm = TENorm(config_no_sp, self.q_lora_rank, eps=self.eps)
        self.wq_b = TEColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=config_no_sp,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.wkv = TELinear(
            self.dim,
            self.head_dim,
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.kv_norm = TENorm(config_no_sp, self.head_dim, eps=self.eps)

        for p in list(self.wq_a.parameters()) + list(self.wkv.parameters()):
            setattr(p, "sequence_parallel", False)

        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            config=config_no_sp,
            init_method=init_method,
            bias=False,
            gather_output=False,
        )
        self.wo_b = TERowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=config_no_sp,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.sequence_parallel = config.sequence_parallel

        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(
                config=config,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                rotate=False,
                cp_group=self.cp_group,
            )
            if self.compress_ratio == 4:
                self.indexer = V4Indexer(config=config, pg_collection=pg_collection)
            else:
                self.indexer = None

        rope_base = (
            cfg.dsv4_compress_rope_theta if self.compress_ratio else cfg.rotary_base
        )
        configure_rope_cache(
            self,
            config,
            rope_head_dim=self.rope_head_dim,
            base=rope_base,
        )
        self._dsv4_position_ids: torch.Tensor | None = None

    def set_position_ids(self, position_ids: torch.Tensor | None) -> None:
        self._dsv4_position_ids = position_ids

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        ans = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ans.update(
            make_sharded_tensors_for_checkpoint(
                state_dict={"attn_sink": self.attn_sink},
                prefix=prefix,
                tensor_parallel_layers_axis_map={"attn_sink": 0},
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=(metadata or {})["dp_cp_group"],
            )
        )
        return ans

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ) -> torch.Tensor:
        """Run DSV4 attention eager inside compiled transformer layers.

        Torch 2.11 currently miscompiles this TP+SP autograd graph: the
        attention module output receives a nonzero gradient, but the
        zero-initialized LoRA branches inside the DSV4 attention path get zero
        tangents. Keeping only this model-specific attention forward eager
        preserves compiled surrounding layer code and correct first-step LoRA
        gradients.
        """
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )

        x = einops.rearrange(hidden_states, "s b d -> b s d")

        bsz, seqlen_local, _ = x.size()
        position_ids = self._dsv4_position_ids
        if position_ids is not None:
            if position_ids.shape != (bsz, seqlen_local):
                raise ValueError(
                    "DSV4 position_ids must match attention input shape: "
                    f"position_ids={tuple(position_ids.shape)} expected={(bsz, seqlen_local)}."
                )
            freqs_cis = get_rope_cache_at_positions(
                self, position_ids=position_ids, device=x.device
            )
        else:
            freqs_cis = get_rope_cache(self, seqlen=seqlen_local, device=x.device)
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        prefix_tree = _prefix_tree_tensors(
            attention_bias, bsz=bsz, seqlen=seqlen_local, device=x.device
        )
        shared_layout: Dsv4CompressionLayout | None = None
        if self.compress_ratio and prefix_tree is not None and position_ids is not None:
            shared_layout = _prefix_tree_compression_layout(
                attention_bias,
                ratio=ratio,
            )

        q_after_wq_a = _add_lora_if_present(self, "wq_a_lora", self.wq_a(x)[0], x)
        qr = q = cast(Any, self.q_norm)(q_after_wq_a)
        q_after_wq_b = _add_lora_if_present(self, "wq_b_lora", self.wq_b(q)[0], q)
        q = q_after_wq_b.unflatten(-1, (self.n_local_heads, self.head_dim))
        q_fp32 = q.float()
        q = (
            q_fp32 * torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + self.eps)
        ).to(q.dtype)
        q = q.clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv_after_wkv = _add_lora_if_present(self, "wkv_lora", self.wkv(x)[0], x)
        kv_vanilla = cast(Any, self.kv_norm)(kv_after_wkv)
        kv_vanilla = kv_vanilla.clone()
        apply_rotary_emb(kv_vanilla[..., -rd:], freqs_cis)

        seqlen_global = seqlen_local
        q_positions = torch.arange(seqlen_local, device=x.device)

        if prefix_tree is not None and position_ids is not None:
            group_ids, parent_ids = prefix_tree
            topk_idxs = _prefix_tree_window_topk_idxs_cached(
                attention_bias,
                position_ids,
                group_ids,
                parent_ids,
                window_size=win,
            )
        else:
            topk_idxs = _window_topk_idxs(q_positions, window_size=win, bsz=bsz)

        if self.compress_ratio:
            kv_compress_offset = seqlen_global
            if self.indexer is not None:
                x_sbd = einops.rearrange(x, "b s d -> s b d")
                qr_sbd = einops.rearrange(qr, "b s d -> s b d")
                if self.sequence_parallel:
                    x_sbd = scatter_to_sequence_parallel_region(
                        x_sbd, group=self.tp_group
                    )
                    qr_sbd = scatter_to_sequence_parallel_region(
                        qr_sbd, group=self.tp_group
                    )
                if isinstance(self.indexer, V4Indexer):
                    shared_layout_i32 = None
                    index_position_ids = position_ids
                    if (
                        prefix_tree is not None
                        and position_ids is not None
                        and shared_layout is not None
                    ):
                        index_position_ids = _prefix_tree_i32_metadata_cached(
                            attention_bias,
                            position_ids,
                        )
                        shared_layout_i32 = _prefix_tree_layout_indexer_metadata_cached(
                            attention_bias,
                            shared_layout,
                            ratio=ratio,
                        )
                    compress_topk_idxs = self.indexer(
                        x_sbd,
                        qr_sbd,
                        position_ids=index_position_ids,
                        shared_layout=shared_layout,
                        shared_layout_i32=shared_layout_i32,
                    )
                else:
                    indexer_mask = self._compute_indexer_mask(
                        q_positions=q_positions, seqlen_global=seqlen_global
                    )
                    compress_topk_idxs = self.indexer(
                        x_sbd, qr_sbd, mask=indexer_mask, packed_seq_params=None
                    )
                if shared_layout is None:
                    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
                    topk_idx_mask = (compress_topk_idxs >= q_first_invalid_group) | (
                        compress_topk_idxs < 0
                    )
                    compress_topk_idxs = torch.where(
                        topk_idx_mask, -1, compress_topk_idxs + kv_compress_offset
                    )
                else:
                    compress_topk_idxs = torch.where(
                        compress_topk_idxs < 0,
                        -1,
                        compress_topk_idxs + kv_compress_offset,
                    )
            else:
                if shared_layout is None or prefix_tree is None or position_ids is None:
                    compress_topk_idxs = _compress_topk_idxs(
                        q_positions, ratio=ratio, bsz=bsz
                    )
                else:
                    compress_topk_idxs = _prefix_tree_compressed_topk_idxs_cached(
                        attention_bias,
                        shared_layout,
                        position_ids=position_ids,
                        ratio=ratio,
                        offset=kv_compress_offset,
                    )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.to(torch.int32)

        kv_compress = None
        if self.compress_ratio:
            x_sbd = einops.rearrange(x, "b s d -> s b d")
            kv_compress_sbd = self.compressor(
                x_sbd,
                position_ids=position_ids,
                shared_layout=shared_layout,
            )
            if kv_compress_sbd is not None:
                kv_compress = einops.rearrange(kv_compress_sbd, "s b d -> b s d")

        if self.attn_sink.dtype != torch.float32:
            raise TypeError(
                "DeepSeek-V4 attention sink must stay fp32, got "
                f"{self.attn_sink.dtype}."
            )

        if kv_compress is not None:
            kv = torch.cat([kv_vanilla, kv_compress], dim=1)
            if kv_compress_offset != kv_vanilla.size(1):
                raise RuntimeError(
                    "DeepSeek-V4 compressed KV offset must equal raw KV length, got "
                    f"{kv_compress_offset} and {kv_vanilla.size(1)}."
                )
        else:
            kv = kv_vanilla

        kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group)

        o = sparse_attn_tilelang(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        o = o.view(bsz, seqlen_local, self.n_local_groups, -1)
        wo_a_input = o
        wo_a = cast(torch.Tensor, self.wo_a.weight).view(
            self.n_local_groups, self.o_lora_rank, -1
        )
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o = _add_lora_if_present(self, "wo_a_lora", o, wo_a_input)
        wo_b_input = o.flatten(2)
        x, _ = self.wo_b(wo_b_input)
        x = _add_lora_if_present(self, "wo_b_lora", x, wo_b_input)

        output = einops.rearrange(x, "b s d -> s b d")

        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)

        return output

    def _compute_indexer_mask(
        self, *, q_positions: torch.Tensor, seqlen_global: int
    ) -> torch.Tensor:
        """Dense causal mask for legacy DSAIndexer path."""
        ratio = 4
        device = q_positions.device
        k_group_idx = torch.arange(seqlen_global // ratio, device=device).unsqueeze(0)
        q_first_invalid_group = (q_positions.unsqueeze(1) + 1) // ratio
        invalid_mask = k_group_idx >= q_first_invalid_group
        return torch.where(invalid_mask, float("-inf"), 0.0)
