from typing import Any, cast

import einops
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from art.megatron.dsv4.compressor import DeepSeekV4Compressor
from art.megatron.dsv4.kernel.tilelang_indexer_fwd import (
    _make_causal_cu_seqlens,
    batched_indexer_fwd,
)
from art.megatron.dsv4.qat import fp8_simulate_qat
from art.megatron.dsv4.rope import (
    apply_rotary_emb,
    configure_rope_cache,
    get_rope_cache,
)
from art.megatron.dsv4.utils import rotate_activation


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
        self.use_fp8_qat = config.fp8 is not None

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
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, mask=None, packed_seq_params=None
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
        freqs_cis = get_rope_cache(self, seqlen=seqlen, device=x.device)
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)
        if self.use_fp8_qat:
            q = fp8_simulate_qat(q, 128)

        k = self.compressor(x)

        weights, _ = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = weights * (self.index_n_heads**-0.5) * softmax_scale

        seqlen_global = seqlen
        seqlen_kv = k.shape[0]
        cu_ks, cu_ke = _make_causal_cu_seqlens(
            seqlen_global, seqlen_kv, self.compress_ratio, q.device
        )
        index_scores = batched_indexer_fwd(q, k, weights.float(), cu_ks, cu_ke)
        topk = min(self.index_topk, index_scores.size(-1))
        return index_scores.topk(topk, dim=-1)[1]
