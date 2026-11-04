# ruff: noqa: E501, PLR0913, PLR0915
from __future__ import annotations

from typing import Any, cast

from fla.ops.utils.op import exp, exp2
import torch
from torch import Tensor
import triton
import triton.language as tl


def pre_process_bwd_summary_multi(
    *,
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g: Tensor,
    do: Tensor,
    dv: Tensor,
    cu_seqlens: Tensor,
    scale: float,
) -> Tensor:
    """Compute FLA CP backward summaries for all varlen chains on device."""

    _, token_count, head_count, key_dim = q.shape
    value_dim = do.shape[-1]
    sequence_count = int(cu_seqlens.numel()) - 1
    summary = q.new_zeros(
        sequence_count,
        head_count,
        key_dim,
        value_dim + key_dim,
        dtype=torch.float32,
    )
    block_size = 32 if key_dim <= 64 else 64
    grid = (
        triton.cdiv(value_dim, block_size),
        head_count,
        sequence_count,
    )
    _pre_process_bwd_kernel_merged_multi[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=None,
        do=do,
        dhm=summary,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=token_count,
        H=head_count,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK1=max(16, triton.next_power_of_2(key_dim)),
        USE_EXP2=False,
        BLOCK_SIZE=block_size,
    )
    fill_summary_transition(summary, k=k, w=w, g=g, cu_seqlens=cu_seqlens, reverse=True)
    return summary


def fill_summary_transition(
    summary: Tensor,
    *,
    k: Tensor,
    w: Tensor,
    g: Tensor,
    cu_seqlens: Tensor | None,
    reverse: bool,
) -> None:
    """Form the rank carry in FP32 without rounding decays or matrix products.

    FLA still computes the affine offset with its production local recurrence.
    Only the transition matrix uses IEEE FP32 contractions. Apply its
    low-rank factors directly to avoid materializing a dense chunk matrix.
    """
    _, tokens, heads, key_dim = k.shape
    value_dim = summary.shape[-1] - key_dim
    sequences = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    key_block = max(16, triton.next_power_of_2(key_dim))
    properties = torch.cuda.get_device_properties(k.device)
    # SM120 has less shared memory than Hopper and datacenter Blackwell.
    # Stage one token tile at a time. Bound the FP32 operand staging,
    # the carry/product tiles when splitting a
    # chunk, and BT x (BK + BC) for small keys, plus scratch.
    token_block = 32 if properties.major >= 12 and key_block >= 256 else 64
    shared_limit = properties.shared_memory_per_block_optin
    while (
        4
        * max(
            2 * token_block * key_block,
            key_block * (token_block + 2 * 32),
            token_block * (key_block + 32) + key_block * 32,
        )
        + 512
        > shared_limit
    ):
        token_block //= 2
        if token_block < 16:
            raise ValueError("Insufficient device shared memory for GDN transition")
    kernel = cast(Any, _summary_transition_kernel)
    kernel[(triton.cdiv(key_dim, 32), heads, sequences)](
        k,
        w,
        g,
        summary,
        cu_seqlens,
        tokens,
        heads,
        key_dim,
        value_dim,
        BK=key_block,
        BC=32,
        BT=token_block,
        IS_VARLEN=cu_seqlens is not None,
        REVERSE=reverse,
        num_warps=4,
        num_stages=1,
    )


@triton.jit(do_not_specialize=["T"])
def _summary_transition_kernel(
    K_ptr,
    W_ptr,
    G_ptr,
    Summary,
    Cu,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    REVERSE: tl.constexpr,
):
    column, head = tl.program_id(0), tl.program_id(1)
    sequence = tl.program_id(2).to(tl.int64)
    if IS_VARLEN:
        bos = tl.load(Cu + sequence).to(tl.int64)
        eos = tl.load(Cu + sequence + 1).to(tl.int64)
    else:
        bos, eos = tl.full((), 0, tl.int64), T.to(tl.int64)
    length = eos - bos
    row = tl.arange(0, BK)
    col = column * BC + tl.arange(0, BC)
    token = tl.arange(0, BT)
    matrix = tl.where(row[:, None] == col[None, :], 1.0, 0.0)
    for step in range(tl.cdiv(length, 64)):
        chunk = tl.cdiv(length, 64) - step - 1 if REVERSE else step
        last = tl.minimum((chunk + 1) * 64, length) - 1
        gate_last = tl.load(G_ptr + (bos + last) * H + head)
        product = tl.full((BK, BC), 0, tl.float32)
        for part in range(64 // BT):
            positions = chunk * 64 + part * BT + token
            valid = positions < length
            offsets = ((bos + positions[:, None]) * H + head) * K + row[None, :]
            key = tl.load(
                K_ptr + offsets, valid[:, None] & (row[None, :] < K), other=0
            ).to(tl.float32)
            wy = tl.load(
                W_ptr + offsets, valid[:, None] & (row[None, :] < K), other=0
            ).to(tl.float32)
            gates = tl.load(G_ptr + (bos + positions) * H + head, valid, other=0)
            key = key * tl.where(valid, exp(gate_last - gates), 0)[:, None]
            if REVERSE:
                projected = tl.dot(key, matrix, input_precision="ieee")
                product += tl.dot(tl.trans(wy), projected, input_precision="ieee")
            else:
                projected = tl.dot(wy, matrix, input_precision="ieee")
                product += tl.dot(tl.trans(key), projected, input_precision="ieee")
        matrix = exp(gate_last) * matrix - product
    offsets = ((sequence * H + head) * K + row[:, None]) * (V + K) + V + col[None, :]
    tl.store(Summary + offsets, matrix, (row[:, None] < K) & (col[None, :] < K))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _pre_process_bwd_kernel_merged_multi(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_col, i_h, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    q += ((bos * H + i_h) * K).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    dhm += ((i_n * H + i_h) * K * (V + K)).to(tl.int64)
    stride_k = H * K

    if is_dh_part:
        do += ((bos * H + i_h) * V).to(tl.int64)
        dv += ((bos * H + i_h) * V).to(tl.int64)
        stride_v = H * V
        i_v = i_col

        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
                bg_last_exp = exp(bg_last)
                p_g = tl.make_block_ptr(
                    g + bos * H + i_h,
                    (T,),
                    (H,),
                    (i_t * BT,),
                    (BT,),
                    (0,),
                )
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_g_exp = exp(b_g)

            p_dv = tl.make_block_ptr(
                dv,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BLOCK_SIZE),
                (BT, BLOCK_SIZE),
                (1, 0),
            )
            p_do = tl.make_block_ptr(
                do,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BLOCK_SIZE),
                (BT, BLOCK_SIZE),
                (1, 0),
            )
            b_do = tl.load(p_do, boundary_check=(0, 1))

            p_k = tl.make_block_ptr(
                k,
                (T, K),
                (stride_k, 1),
                (i_t * BT, 0),
                (BT, 64),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(
                    gk + last_idx * H * K + o_k1,
                    mask=o_k1 < K,
                    other=0.0,
                ).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(
                    k,
                    (T, K),
                    (stride_k, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(
                        gk + last_idx * H * K + o_k2,
                        mask=o_k2 < K,
                        other=0.0,
                    ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(
                    k,
                    (T, K),
                    (stride_k, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(
                        gk + last_idx * H * K + o_k3,
                        mask=o_k3 < K,
                        other=0.0,
                    ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(
                    k,
                    (T, K),
                    (stride_k, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(
                        gk + last_idx * H * K + o_k4,
                        mask=o_k4 < K,
                        other=0.0,
                    ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(
                w,
                (K, T),
                (1, stride_k),
                (0, i_t * BT),
                (64, BT),
                (0, 1),
            )
            p_q = tl.make_block_ptr(
                q,
                (K, T),
                (1, stride_k),
                (0, i_t * BT),
                (64, BT),
                (0, 1),
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                b_w,
                b_dv.to(b_w.dtype),
            )

            if K > 64:
                p_q = tl.make_block_ptr(
                    q,
                    (K, T),
                    (1, stride_k),
                    (64, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                p_w = tl.make_block_ptr(
                    w,
                    (K, T),
                    (1, stride_k),
                    (64, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
                b_dh2 += tl.dot(
                    b_q.to(b_q.dtype),
                    b_do.to(b_q.dtype),
                ) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(
                    q,
                    (K, T),
                    (1, stride_k),
                    (128, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                p_w = tl.make_block_ptr(
                    w,
                    (K, T),
                    (1, stride_k),
                    (128, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
                b_dh3 += tl.dot(
                    b_q.to(b_q.dtype),
                    b_do.to(b_q.dtype),
                ) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(
                    q,
                    (K, T),
                    (1, stride_k),
                    (192, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                p_w = tl.make_block_ptr(
                    w,
                    (K, T),
                    (1, stride_k),
                    (192, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
                b_dh4 += tl.dot(
                    b_q.to(b_q.dtype),
                    b_do.to(b_q.dtype),
                ) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        p_dh1 = tl.make_block_ptr(
            dhm,
            (K, V),
            (V + K, 1),
            (0, i_v * BLOCK_SIZE),
            (64, BLOCK_SIZE),
            (1, 0),
        )
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(
                dhm,
                (K, V),
                (V + K, 1),
                (64, i_v * BLOCK_SIZE),
                (64, BLOCK_SIZE),
                (1, 0),
            )
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(
                dhm,
                (K, V),
                (V + K, 1),
                (128, i_v * BLOCK_SIZE),
                (64, BLOCK_SIZE),
                (1, 0),
            )
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(
                dhm,
                (K, V),
                (V + K, 1),
                (192, i_v * BLOCK_SIZE),
                (64, BLOCK_SIZE),
                (1, 0),
            )
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            i_t = NT - 1 - _i_t
            p_k = tl.make_block_ptr(
                k,
                (T, K),
                (stride_k, 1),
                (i_t * BT, 0),
                (BT, BK1),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(
                w,
                (T, K),
                (stride_k, 1),
                (i_t * BT, 0),
                (BT, BK1),
                (1, 0),
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(
                    g + bos * H + i_h,
                    (T,),
                    (H,),
                    (i_t * BT,),
                    (BT,),
                    (0,),
                )
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_k = b_k * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + row,
                    mask=row < K,
                    other=0.0,
                ).to(tl.float32)
                if USE_EXP2:
                    b_gk_last = exp2(b_gk_last)
                else:
                    b_gk_last = exp(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            b_m_i = b_diag - b_kw
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        p_m = tl.make_block_ptr(
            dhm + V,
            (K, K),
            (V + K, 1),
            (0, i_k_col * BLOCK_SIZE),
            (BK1, BLOCK_SIZE),
            (1, 0),
        )
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))
