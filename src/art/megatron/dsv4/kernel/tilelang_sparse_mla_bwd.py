# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - attn_sink: gradient computation for learnable per-head scalar
#   - Single-head KV: kv shape [B, S_kv, D] (no kv_group, no D/D_tail split)
#   - Index shape: [B, S, topk] (no kv_group dim)
#   - Outputs: dQ [B, S, H, D], dKV [B, S_kv, D], dAttnSink [H]
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


@tilelang.jit(out_idx=[-1])
def preprocess(
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("batch")
    S = T.dynamic("seq_len")
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([B, S, H], accum_dtype),  # type: ignore
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o,
                )
                T.copy(
                    dO[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do,
                )
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    D,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("batch")
    S_kv = T.dynamic("seq_len_kv")
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),  # type: ignore
        dKV_out: T.Tensor(dkv_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, by):
            T.copy(
                dKV[by, bx * block_N : (bx + 1) * block_N, :],
                dKV_out[by, bx * block_N : (bx + 1) * block_N, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def bwd(
    H,
    D,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0, (
        f"topk ({topk}) must be divisible by block_size ({block_size})"
    )
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("batch")
    S = T.dynamic("seq_len")
    S_kv = T.dynamic("seq_len_kv")

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        dO: T.Tensor(o_shape, dtype),  # type: ignore
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(q_shape, dtype),  # type: ignore
        dKV: T.Tensor(kv_shape, accum_dtype),  # type: ignore
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dq_i = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            safe_indices = T.alloc_fragment([BS], indices_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1
                    safe_indices[bi_i] = T.if_then_else(
                        mask[bi_i], Indices[by, s_i, i_i * BS + bi_i], 0
                    )

                T.clear(acc_p)

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, safe_indices[bi_i], d_i]

                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], acc_p[h_i, bi_i], -T.infinity(acc_p.dtype)
                    )

                # P = exp2(scores * sm_scale_log2e - LSE)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # dP = P * (dO @ KV^T - Delta)
                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i])
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV
                T.gemm(
                    dP_shared_cast,
                    KV_shared,
                    acc_dq_i,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                for h_i, d_i in T.Parallel(block_H, D):
                    acc_dq[h_i, d_i] += acc_dq_i[h_i, d_i]

                # dKV += dP^T @ Q + P^T @ dO
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # Atomic store dKV with split to reduce register pressure
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            src_i = bi_i + s * (BS // split_store)
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[src_i, d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        src_i = bi_i + s * (BS // split_store)
                        T.atomic_addx4(
                            dKV[
                                by,
                                safe_indices[src_i],
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

            # Store dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

            # dAttnSink[h] = -sum_{b,s}( Delta[b,s,h] * p_sink[b,s,h] )
            # where p_sink = exp(attn_sink[h]) / Z = exp2(attn_sink[h]*log2e - LSE)
            # attn_sink is a pre-scaled logit, so only convert to log2 base (no sm_scale)
            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(
                        AttnSink[bz * block_H + h_i] * 1.44269504
                        - Lse[by, s_i, bz * block_H + h_i]
                    ),
                )

    return sparse_mqa_bwd_kernel


sanitize_tilelang_env()


def _tilelang_input_dtype(torch_dtype):
    if torch_dtype is torch.bfloat16:
        return T.bfloat16
    raise TypeError(f"DSV4 sparse MLA TileLang launch requires bf16, got {torch_dtype}")


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=None):
    """Backward interface for V4 sparse MQA attention.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        attn_sink: [H] fp32
        o:         [B, S, H, D] bf16 (forward output)
        do:        [B, S, H, D] bf16 (grad of output)
        topk_idxs: [B, S, topk] int32
        lse:       [B, S, H] fp32 (log-sum-exp from forward)
        sm_scale:  float or None

    Returns:
        dq:         [B, S, H, D] bf16
        dkv:        [B, S_kv, D] bf16
        d_attn_sink: [H] fp32
    """
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    dtype = _tilelang_input_dtype(q.dtype)
    assert kv.dtype == q.dtype and o.dtype == q.dtype and do.dtype == q.dtype

    # Pad topk to next multiple of block_size (kernel requires divisibility)
    block_size = 64
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full(
            (B, S, padded_topk - topk),
            -1,
            device=topk_idxs.device,
            dtype=topk_idxs.dtype,
        )
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    with preserve_tilelang_env():
        # Keep sequence lengths dynamic so changing packed workloads reuse the
        # same generated kernels.  Model/tile dimensions remain static.
        preprocess_kernel = preprocess(H, D, dtype=dtype)
        postprocess_kernel = postprocess(D, dtype=dtype)
        delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros_like(attn_sink)
    if topk <= block_size:
        with preserve_tilelang_env():
            bwd_kernel = bwd(
                H,
                D,
                topk,
                sm_scale,
                block_size=block_size,
                dtype=dtype,
            )
            dq = bwd_kernel(
                q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink
            )
    else:
        dq_accum = torch.zeros_like(q, dtype=torch.float32)
        chunk_count = topk // block_size
        with preserve_tilelang_env():
            bwd_kernel = bwd(
                H,
                D,
                block_size,
                sm_scale,
                block_size=block_size,
                dtype=dtype,
            )
            for start in range(0, topk, block_size):
                chunk = topk_idxs[:, :, start : start + block_size].contiguous()
                dq_i = bwd_kernel(
                    q, kv, do, attn_sink, chunk, lse, delta, dkv, d_attn_sink
                )
                dq_accum.add_(dq_i.float())
        dq = dq_accum.to(q.dtype)
        d_attn_sink.div_(chunk_count)
    with preserve_tilelang_env():
        dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink
