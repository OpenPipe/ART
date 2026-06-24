from typing import Any, cast

import einops
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from art.megatron.dsv4.compressor import (
    DeepSeekV4Compressor,
    Dsv4CompressionLayout,
    compressed_layout_visibility,
)
from art.megatron.dsv4.kernel.tilelang_indexer_fwd import (
    indexer_topk_interface,
    shared_prefix_indexer_topk_interface,
)
from art.megatron.dsv4.rope import (
    apply_rotary_emb,
    configure_rope_cache,
    get_rope_cache,
    get_rope_cache_at_positions,
)
from art.megatron.dsv4.utils import freeze_parameters_as_buffers, rotate_activation

_INDEXER_QUERY_BLOCK = 512
_INDEXER_HEAD_BLOCK = 8


def _make_causal_cu_seqlens(seq_len_q, seq_len_kv, compress_ratio, device):
    del seq_len_kv
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


@torch.compiler.disable
def _exact_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    shared_layout: Dsv4CompressionLayout | None = None,
    position_ids: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    parent_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return frozen DSV4 indexer topk using the reference score equation.

    The indexer routing is discrete, so small BF16 fused-kernel score drift can
    choose different compressed KV rows and send CSA gradients to different
    windows. Keep the production path exact against the HF/Megatron reference:
    fp32 dot, ReLU, head weighting, compressed-causal mask, then topk. Query
    and head chunking avoid materializing the full [S, H, S / 4] score tensor.
    """

    seqlen, batch, heads, _ = q.shape
    seqlen_kv = k.shape[0]
    actual_topk = min(topk, seqlen_kv)
    out = torch.empty(batch, seqlen, actual_topk, device=q.device, dtype=torch.int32)
    if actual_topk == 0:
        return out

    kv_ids = torch.arange(seqlen_kv, device=q.device)
    q_block = _INDEXER_QUERY_BLOCK
    h_block = _INDEXER_HEAD_BLOCK
    with torch.no_grad():
        for b in range(batch):
            k_b = k[:, b].float()
            for q_start in range(0, seqlen, q_block):
                q_end = min(q_start + q_block, seqlen)
                scores = torch.zeros(
                    q_end - q_start, seqlen_kv, device=q.device, dtype=torch.float32
                )
                for h_start in range(0, heads, h_block):
                    h_end = min(h_start + h_block, heads)
                    dot = torch.einsum(
                        "qhd,kd->qhk",
                        q[q_start:q_end, b, h_start:h_end].float(),
                        k_b,
                    ).relu_()
                    scores += (
                        dot
                        * weights[q_start:q_end, b, h_start:h_end].float().unsqueeze(-1)
                    ).sum(dim=1)
                if shared_layout is None:
                    visible = (
                        kv_ids.unsqueeze(0) >= cu_seqlen_ks[q_start:q_end, None]
                    ) & (kv_ids.unsqueeze(0) < cu_seqlen_ke[q_start:q_end, None])
                else:
                    if position_ids is None or group_ids is None or parent_ids is None:
                        raise ValueError(
                            "DSV4 shared-prefix indexer requires position/group metadata."
                        )
                    visible = compressed_layout_visibility(
                        shared_layout,
                        position_ids=position_ids[b : b + 1],
                        group_ids=group_ids[b : b + 1],
                        parent_ids=parent_ids[b : b + 1],
                        q_start=q_start,
                        q_end=q_end,
                    )[0]
                scores.masked_fill_(~visible, float("-inf"))
                top_scores, top_indices = scores.topk(actual_topk, dim=-1)
                out[b, q_start:q_end] = torch.where(
                    torch.isneginf(top_scores),
                    torch.full_like(top_indices, -1),
                    top_indices,
                )
    return out


@torch.compiler.disable
def _tilelang_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    shared_layout: Dsv4CompressionLayout | None = None,
    position_ids: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    parent_ids: torch.Tensor | None = None,
    shared_layout_i32: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
) -> torch.Tensor:
    seqlen, batch, heads, _ = q.shape
    seqlen_kv = k.shape[0]
    actual_topk = min(topk, seqlen_kv)
    out = torch.empty(batch, seqlen, actual_topk, device=q.device, dtype=torch.int32)
    if actual_topk == 0:
        return out

    internal_q_block = max(1, 128 // heads)
    q_block = (_INDEXER_QUERY_BLOCK // internal_q_block) * internal_q_block
    q_block = max(internal_q_block, q_block)
    fast_seqlen = (seqlen // internal_q_block) * internal_q_block

    with torch.no_grad():
        for b in range(batch):
            if shared_layout is not None:
                if position_ids is None or group_ids is None or parent_ids is None:
                    raise ValueError(
                        "DSV4 shared-prefix indexer requires position/group metadata."
                    )
                if shared_layout_i32 is None:
                    entry_group_ids = shared_layout.entry_group_ids.to(torch.int32)
                    entry_parent_visible = shared_layout.entry_parent_visible.to(
                        torch.int32
                    )
                    entry_end_positions = shared_layout.entry_end_positions.to(
                        torch.int32
                    )
                    entry_valid = shared_layout.entry_valid.to(torch.int32)
                else:
                    (
                        entry_group_ids,
                        entry_parent_visible,
                        entry_end_positions,
                        entry_valid,
                    ) = shared_layout_i32
                position_b = position_ids[b].to(torch.int32).contiguous()
                group_b = group_ids[b].to(torch.int32).contiguous()
                parent_b = parent_ids[b].to(torch.int32).contiguous()
                entry_group_b = entry_group_ids[b].contiguous()
                entry_parent_visible_b = entry_parent_visible[b].contiguous()
                entry_end_b = entry_end_positions[b].contiguous()
                entry_valid_b = entry_valid[b].contiguous()
            for q_start in range(0, fast_seqlen, q_block):
                q_end = min(q_start + q_block, fast_seqlen)
                q_slice = q[q_start:q_end, b].contiguous()
                k_slice = k[:, b].contiguous()
                weights_slice = weights[q_start:q_end, b].contiguous()
                if shared_layout is None:
                    out[b, q_start:q_end] = indexer_topk_interface(
                        q_slice,
                        k_slice,
                        weights_slice,
                        cu_seqlen_ks[q_start:q_end].contiguous(),
                        cu_seqlen_ke[q_start:q_end].contiguous(),
                        topk,
                    )
                else:
                    out[b, q_start:q_end] = shared_prefix_indexer_topk_interface(
                        q_slice,
                        k_slice,
                        weights_slice,
                        position_b[q_start:q_end],
                        group_b[q_start:q_end],
                        parent_b[q_start:q_end],
                        entry_group_b,
                        entry_parent_visible_b,
                        entry_end_b,
                        entry_valid_b,
                        topk,
                    )
        if fast_seqlen != seqlen:
            out[:, fast_seqlen:] = _exact_indexer_topk(
                q[fast_seqlen:],
                k,
                weights[fast_seqlen:],
                cu_seqlen_ks[fast_seqlen:],
                cu_seqlen_ke[fast_seqlen:],
                topk,
                shared_layout=shared_layout,
                position_ids=None
                if position_ids is None
                else position_ids[:, fast_seqlen:],
                group_ids=None if group_ids is None else group_ids[:, fast_seqlen:],
                parent_ids=None if parent_ids is None else parent_ids[:, fast_seqlen:],
            )
    return out


class V4Indexer(MegatronModule):
    """DSA Indexer for DeepSeek-V4 C4 layers."""

    def __init__(self, config: TransformerConfig, pg_collection=None):
        super().__init__(config=config)
        cfg = cast(Any, config)
        init_method = config.init_method
        if init_method is None:
            raise RuntimeError("DeepSeek-V4 indexer requires config.init_method.")

        self.hidden_size = config.hidden_size
        self.q_lora_rank = (
            int(cfg.q_lora_rank) if cfg.q_lora_rank is not None else config.hidden_size
        )
        self.index_n_heads = int(cfg.dsa_indexer_n_heads)
        self.index_head_dim = int(cfg.dsa_indexer_head_dim)
        self.index_topk = int(cfg.dsa_indexer_topk)
        self.rope_head_dim = int(cfg.qk_pos_emb_head_dim)
        self.compress_ratio = 4

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        self.pg_collection = pg_collection

        self.linear_wq_b = TELinear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_weights_proj = TELinear(
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.compressor = DeepSeekV4Compressor(
            config=config,
            head_dim=self.index_head_dim,
            compress_ratio=self.compress_ratio,
            rotate=True,
            cp_group=None,
        )

        rope_base = (
            cfg.dsv4_compress_rope_theta if self.compress_ratio else cfg.rotary_base
        )
        configure_rope_cache(
            self, config, rope_head_dim=self.rope_head_dim, base=rope_base
        )
        freeze_parameters_as_buffers(self)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask=None,
        packed_seq_params=None,
        position_ids: torch.Tensor | None = None,
        shared_layout: Dsv4CompressionLayout | None = None,
        group_ids: torch.Tensor | None = None,
        parent_ids: torch.Tensor | None = None,
        shared_layout_i32: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ):
        """Forward pass.

        Args:
            x:  hidden states [seqlen, batch, hidden_size]
            qr: low-rank query [seqlen, batch, q_lora_rank]
            mask: unused (causal mask generated internally via cu_seqlens)
            packed_seq_params: unused

        Returns:
            topk_indices: [batch, seqlen, index_topk] int64
        """

        # =========================================
        # Gather inputs if SP is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        seqlen, bsz, _ = x.size()

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)

        rd = self.rope_head_dim
        cp_group = getattr(self.pg_collection, "cp", None)
        if cp_group is not None and cp_group.size() != 1:
            raise RuntimeError(
                "DeepSeek-V4 non-CP indexer received context_parallel_size > 1."
            )
        if position_ids is None:
            freqs_cis = get_rope_cache(self, seqlen=seqlen, device=x.device)
        else:
            freqs_cis = get_rope_cache_at_positions(
                self, position_ids=position_ids, device=x.device
            )
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)

        k = self.compressor(
            x,
            position_ids=position_ids,
            shared_layout=shared_layout,
        )
        if k.shape[0] == 0:
            return torch.empty(
                bsz,
                seqlen,
                0,
                device=x.device,
                dtype=torch.long,
            )

        weights, _ = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = weights * (self.index_n_heads**-0.5) * softmax_scale

        seqlen_global = seqlen
        seqlen_kv = k.shape[0]
        cu_ks, cu_ke = _make_causal_cu_seqlens(
            seqlen_global, seqlen_kv, self.compress_ratio, q.device
        )
        return _tilelang_indexer_topk(
            q,
            k,
            weights.float(),
            cu_ks,
            cu_ke,
            self.index_topk,
            shared_layout=shared_layout,
            position_ids=position_ids,
            group_ids=group_ids,
            parent_ids=parent_ids,
            shared_layout_i32=shared_layout_i32,
        )
