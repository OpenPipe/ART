"""Fixed persistent dense and single-launch expert training contractions."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def persistent_mm(
    A,
    B,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SMS: tl.constexpr,
    BM: tl.constexpr,
    BN_TILE: tl.constexpr,
    BK_TILE: tl.constexpr,
    PRECISION: tl.constexpr,
    FP32_OPERANDS: tl.constexpr,
):
    # Scheduling may depend on M; each output's instruction and K tree do not.
    tiles_n = tl.cdiv(N, BN_TILE)
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * tiles_n, SMS):
        m = (tile // tiles_n) * BM + tl.arange(0, BM)
        n = (tile % tiles_n) * BN_TILE + tl.arange(0, BN_TILE)
        k = tl.arange(0, BK_TILE)
        acc = tl.full((BM, BN_TILE), 0, tl.float32)
        for start in range(tl.cdiv(K, BK_TILE)):
            kk = start * BK_TILE + k
            aa = tl.load(
                A + m[:, None].to(tl.int64) * AM + kk[None, :] * AK,
                (m[:, None] < M) & (kk[None, :] < K),
                0,
            )
            bb = tl.load(
                B + kk[:, None].to(tl.int64) * BK + n[None, :] * BN,
                (kk[:, None] < K) & (n[None, :] < N),
                0,
            )
            if FP32_OPERANDS:
                aa, bb = aa.to(tl.float32), bb.to(tl.float32)
            acc = tl.dot(aa, bb, acc, input_precision=PRECISION)
        tl.store(
            C + m[:, None].to(tl.int64) * N + n[None, :],
            acc,
            (m[:, None] < M) & (n[None, :] < N),
        )


@triton.jit
def grouped_mm(
    A,
    B,
    C,
    OFFSETS,
    TILE_EXPERTS,
    TILE_ROWS,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BE: tl.constexpr,
    BM: tl.constexpr,
    NT: tl.constexpr,
    KT: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.load(TILE_EXPERTS + tile)
    first, stop = tl.load(OFFSETS + expert), tl.load(OFFSETS + expert + 1)
    local_row = tl.load(TILE_ROWS + tile)
    m = first + local_row + tl.arange(0, BM)
    n = tl.program_id(1) * NT + tl.arange(0, NT)
    k = tl.arange(0, KT)
    acc = tl.full((BM, NT), 0, tl.float32)
    for start in range(tl.cdiv(K, KT)):
        kk = start * KT + k
        a = tl.load(
            A + m[:, None].to(tl.int64) * K + kk[None, :],
            (m[:, None] < stop) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + expert.to(tl.int64) * BE + kk[:, None] * BK + n[None, :] * BN,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        C + m[:, None].to(tl.int64) * N + n[None, :],
        acc,
        (m[:, None] < stop) & (n[None, :] < N),
    )


@triton.jit
def expert_mm(
    A,
    WEIGHTS,
    C,
    OFFSETS,
    EXPERTS,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.load(EXPERTS + tile)
    first = tl.load(OFFSETS + expert)
    stop = tl.load(OFFSETS + expert + 1)
    weight = tl.load(WEIGHTS + expert).to(tl.pointer_type(A.dtype.element_ty))
    m = first + tl.load(ROWS + tile) + tl.arange(0, 32)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 32)
    acc = tl.full((32, 64), 0, tl.float32)
    for start in range(tl.cdiv(K, 32)):
        kk = start * 32 + k
        aa = tl.load(
            A + m[:, None].to(tl.int64) * K + kk[None, :],
            (m[:, None] < stop) & (kk[None, :] < K),
            0,
        )
        bb = tl.load(
            weight + kk[:, None].to(tl.int64) * BK + n[None, :] * BN,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        if N <= 32 or K <= 32:
            acc = tl.dot(
                aa.to(tl.float32), bb.to(tl.float32), acc, input_precision="ieee"
            )
        else:
            acc = tl.dot(aa, bb, acc, input_precision="ieee")
    tl.store(
        C + m[:, None].to(tl.int64) * N + n[None, :],
        acc,
        (m[:, None] < stop) & (n[None, :] < N),
    )
