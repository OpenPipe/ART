# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_indexer_fwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - Operates on [seqlen, batch, heads, dim] (SBHD) layout, batch handled externally
#   - Uses causal mask via cu_seqlens instead of variable-length packed sequences
#   - Supports compressed KV (seq_len_kv = seq_len_q / compress_ratio)
from typing import Any, cast

import torch

from art.megatron.dsv4.kernel.tilelang_import import (
    import_tilelang,
    preserve_tilelang_env,
    sanitize_tilelang_env,
)

_tilelang, _T = import_tilelang()

tilelang = cast(Any, _tilelang)
T = cast(Any, _T)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tl_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(
                    cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv)
                )
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(
                    cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv)
                )

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(
                T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
            ):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                    )

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = (
                        logits[bn_i, bq_i]
                    )

    return tl_indexer_fwd_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


sanitize_tilelang_env()


def _make_causal_cu_seqlens(seq_len_q, seq_len_kv, compress_ratio, device):
    """Generate cu_seqlens for causal masking on compressed KV positions.

    For query at position p, valid compressed groups are [0, (p+1) // compress_ratio).
    """
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def indexer_fwd_interface(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True
):
    """Forward interface matching GLM-5's API but for a single batch element.

    Args:
        q: [seq_len, heads, index_dim] bf16
        kv: [seq_len_kv, index_dim] bf16
        weights: [seq_len, heads] fp32
        cu_seqlen_ks: [seq_len] int32 — start of valid KV range per query
        cu_seqlen_ke: [seq_len] int32 — end of valid KV range per query

    Returns:
        logits: [seq_len, seq_len_kv] fp32
    """
    if q.dtype is torch.float32:
        q = q.to(torch.bfloat16)
    if kv.dtype is torch.float32:
        kv = kv.to(torch.bfloat16)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(
            f"DSV4 indexer TileLang launch requires bf16, got {q.dtype=}, {kv.dtype=}"
        )
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    block_q = max(1, 128 // heads)
    if seq_len % block_q != 0:
        raise ValueError(
            f"DSV4 indexer TileLang query length must be divisible by {block_q}, "
            f"got {seq_len}."
        )

    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    with preserve_tilelang_env():
        clean_logits_kernel = clean_logits_()
        tl_indexer_fwd_kernel = tl_indexer_fwd_impl(heads=heads, index_dim=index_dim)
        tl_indexer_fwd_kernel(
            q.view(seq_len * heads, index_dim),
            kv,
            logits,
            weights.float(),
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
        if clean_logits:
            clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def _topk_indices_from_logits(logits, topk):
    actual_topk = min(int(topk), int(logits.shape[-1]))
    top_scores, top_indices = logits.topk(actual_topk, dim=-1)
    return torch.where(
        torch.isneginf(top_scores),
        torch.full_like(top_indices, -1),
        top_indices,
    ).to(torch.int32)


def indexer_topk_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk):
    """Compute DSV4 indexer scores and return top-k compressed ids.

    Keep topk owned by the indexer wrapper so callers do not materialize or
    inspect dense logits.
    """
    return _topk_indices_from_logits(
        indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke),
        topk,
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tl_shared_prefix_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_shared_prefix_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        QPosition: T.Tensor([seq_len], index_dtype),  # type: ignore
        QGroup: T.Tensor([seq_len], index_dtype),  # type: ignore
        QParent: T.Tensor([seq_len], index_dtype),  # type: ignore
        EntryGroup: T.Tensor([seq_len_kv], index_dtype),  # type: ignore
        EntryParentVisible: T.Tensor([seq_len_kv], index_dtype),  # type: ignore
        EntryEndPosition: T.Tensor([seq_len_kv], index_dtype),  # type: ignore
        EntryValid: T.Tensor([seq_len_kv], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(
                T.ceildiv(seq_len_kv, block_N), num_stages=num_stages
            ):
                for bn_i, d_i in T.Parallel(block_N, index_dim):
                    k_i = nbn_i * block_N + bn_i
                    index_k_shared[bn_i, d_i] = T.if_then_else(
                        k_i < seq_len_kv,
                        IndexK[k_i, d_i],
                        0,
                    )

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                    )

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    q_i = seq_len_i + bq_i
                    k_i = nbn_i * block_N + bn_i
                    if k_i < seq_len_kv:
                        entry_group = EntryGroup[k_i]
                        visible = (
                            (EntryValid[k_i] != 0)
                            and (EntryEndPosition[k_i] <= QPosition[q_i])
                            and (
                                (entry_group == QGroup[q_i])
                                or (
                                    (EntryParentVisible[k_i] != 0)
                                    and (entry_group == QParent[q_i])
                                )
                            )
                        )
                        Logits[q_i, k_i] = T.if_then_else(
                            visible,
                            logits[bn_i, bq_i],
                            -T.infinity(accum_dtype),
                        )

    return tl_shared_prefix_indexer_fwd_kernel


def _i32_contiguous(tensor):
    if tensor.dtype != torch.int32:
        tensor = tensor.to(torch.int32)
    return tensor.contiguous()


def shared_prefix_indexer_fwd_interface(
    q,
    kv,
    weights,
    position_ids,
    group_ids,
    parent_ids,
    entry_group_ids,
    entry_parent_visible,
    entry_end_positions,
    entry_valid,
):
    """Compute shared-prefix-aware indexer logits for one batch element.

    The output is intentionally block-local [query_block, compressed_kv] fp32.
    Callers run topk on this block to avoid materializing full [S, compressed_kv]
    logits for long shared-prefix packed sequences.
    """
    if q.dtype is torch.float32:
        q = q.to(torch.bfloat16)
    if kv.dtype is torch.float32:
        kv = kv.to(torch.bfloat16)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(
            f"DSV4 indexer TileLang launch requires bf16, got {q.dtype=}, {kv.dtype=}"
        )
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    block_q = max(1, 128 // heads)
    if seq_len % block_q != 0:
        raise ValueError(
            "DSV4 shared-prefix indexer TileLang query length must be divisible "
            f"by {block_q}, got {seq_len}."
        )

    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    with preserve_tilelang_env():
        tl_indexer_fwd_kernel = tl_shared_prefix_indexer_fwd_impl(
            heads=heads,
            index_dim=index_dim,
        )
        tl_indexer_fwd_kernel(
            q.view(seq_len * heads, index_dim),
            kv,
            logits,
            weights.float(),
            _i32_contiguous(position_ids),
            _i32_contiguous(group_ids),
            _i32_contiguous(parent_ids),
            _i32_contiguous(entry_group_ids),
            _i32_contiguous(entry_parent_visible),
            _i32_contiguous(entry_end_positions),
            _i32_contiguous(entry_valid),
        )
    return logits


def shared_prefix_indexer_topk_interface(
    q,
    kv,
    weights,
    position_ids,
    group_ids,
    parent_ids,
    entry_group_ids,
    entry_parent_visible,
    entry_end_positions,
    entry_valid,
    topk,
):
    """Compute shared-prefix-aware DSV4 indexer top-k compressed ids."""
    return _topk_indices_from_logits(
        shared_prefix_indexer_fwd_interface(
            q,
            kv,
            weights,
            position_ids,
            group_ids,
            parent_ids,
            entry_group_ids,
            entry_parent_visible,
            entry_end_positions,
            entry_valid,
        ),
        topk,
    )


@torch.compiler.disable
def batched_indexer_fwd(q, k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Batched forward: loops over batch dim.

    Args:
        q: [seqlen, batch, heads, dim] bf16
        k: [seqlen_kv, batch, dim] bf16
        weights: [seqlen, batch, heads] fp32
        cu_seqlen_ks: [seqlen] int32
        cu_seqlen_ke: [seqlen] int32

    Returns:
        logits: [batch, seqlen, seqlen_kv] fp32
    """
    seqlen, batch, heads, dim = q.shape
    seq_len_kv = k.shape[0]

    all_logits = torch.empty(
        [batch, seqlen, seq_len_kv], device=q.device, dtype=torch.float32
    )
    for b in range(batch):
        all_logits[b] = indexer_fwd_interface(
            q[:, b, :, :].contiguous(),
            k[:, b, :].contiguous(),
            weights[:, b, :].contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    return all_logits
