"""Fixed geometry CUDA primitives. No autotuning, split-K, atomics or row grouping.

The public Thinking Machines batch_invariant_ops is the design reference (MIT,
revision f22b1fbe534b1fe045080f44a970a441caaf3c4f). These kernels also supply
training gradients; they do not install process-wide torch/TE overrides.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "N", "K"])
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    AM,
    AK,
    BK,
    BN,
    BM: tl.constexpr = 128,
    BN_TILE: tl.constexpr = 64,
    BK_TILE: tl.constexpr = 32,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    k = tl.arange(0, BK_TILE)
    acc = tl.full((BM, BN_TILE), 0, tl.float32)
    # One accumulator for the complete K dimension, in increasing K order.
    for start in range(tl.cdiv(K, BK_TILE)):
        kk = (start * BK_TILE + k).to(tl.int64)
        a = tl.load(
            A + m[:, None].to(tl.int64) * AM + kk[None, :] * AK,
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + kk[:, None].to(tl.int64) * BK + n[None, :] * BN,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        C + m[:, None].to(tl.int64) * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def rms_forward(
    X,
    W,
    Y,
    R,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32) + ZERO
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(Y + row.to(tl.int64) * D + col, (x * r) * w, col < D)
    tl.store(R + row, r)


@triton.jit
def rms_backward(
    X, W, R, DY, DX, D: tl.constexpr, ZERO: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32) + ZERO
    dy = tl.load(DY + row.to(tl.int64) * D + col, col < D, 0).to(tl.float32)
    r = tl.load(R + row)
    dot = tl.sum((dy * w) * x, 0) / D
    dx = r * (dy * w - x * (r * r * dot))
    tl.store(DX + row.to(tl.int64) * D + col, dx, col < D)


@triton.jit(do_not_specialize=["M"])
def rms_weight_backward(X, R, DY, DW, M, D: tl.constexpr):
    # Fixed 128-row partial trees, accumulated in increasing row order.
    cols = tl.program_id(0) * 32 + tl.arange(0, 32)
    rows = tl.arange(0, 128)
    total = tl.full((32,), 0, tl.float32)
    for tile in range(tl.cdiv(M, 128)):
        rr = tile * 128 + rows
        offsets = rr[:, None].to(tl.int64) * D + cols[None, :]
        mask = (rr[:, None] < M) & (cols[None, :] < D)
        x = tl.load(X + offsets, mask, 0).to(tl.float32)
        dy = tl.load(DY + offsets, mask, 0).to(tl.float32)
        r = tl.load(R + rr, rr < M, 0)
        total += tl.sum((x * r[:, None]) * dy, 0)
    tl.store(DW + cols, total, cols < D)


@triton.jit
def lse_forward(X, Y, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    maximum = tl.full((), -float("inf"), tl.float32)
    for start in range(tl.cdiv(D, 1024)):
        cc = start * 1024 + cols
        x = tl.load(X + row.to(tl.int64) * D + cc, cc < D, -float("inf")).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(x, 0))
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(D, 1024)):
        cc = start * 1024 + cols
        x = tl.load(X + row.to(tl.int64) * D + cc, cc < D, -float("inf")).to(tl.float32)
        total += tl.sum(tl.exp(x - maximum), 0)
    tl.store(Y + row, maximum + tl.log(total))


@triton.jit
def sum_forward(X, Y, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(D, 1024)):
        cc = start * 1024 + cols
        x = tl.load(X + row.to(tl.int64) * D + cc, cc < D, 0).to(tl.float32)
        total += tl.sum(x, 0)
    tl.store(Y + row, total)
