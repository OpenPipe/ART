# ruff: noqa
# Adapted from Miles GLM and tile-ai/tilelang DeepSeek-V3.2 sparse MLA kernels.

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import os
from typing import Any

import torch

_ENV_KEYS = (
    "PYTHONPATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TL_CUTLASS_PATH",
    "TL_TEMPLATE_PATH",
    "TL_COMPOSABLE_KERNEL_PATH",
)
_PATH_MARKERS = ("/site-packages/tilelang/", "\\site-packages\\tilelang\\")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    kept = [
        part
        for part in value.split(os.pathsep)
        if not any(marker in part for marker in _PATH_MARKERS)
    ]
    return os.pathsep.join(kept) if kept else None


def _restore(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    for key in _ENV_KEYS:
        value = _clean(os.environ.get(key))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def _preserve_env() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        yield
    finally:
        _restore(saved)


with _preserve_env():
    tilelang: Any = importlib.import_module("tilelang")
    T: Any = importlib.import_module("tilelang.language")

_LATENT = 512
_ROPE = 64
_DIM = _LATENT + _ROPE
_HEAD_BLOCK = 16
_DKV_SPLITS = 4
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _forward(heads, topk, scale, block_i=64, num_stages=2, threads=256):
    assert topk % block_i == 0
    batch = T.dynamic("batch")
    q_tokens = T.dynamic("q_tokens")
    kv_tokens = T.dynamic("kv_tokens")
    q_shape = [batch, q_tokens, heads, _DIM]
    kv_shape = [batch, kv_tokens, _DIM]
    indices_shape = [batch, q_tokens, topk]
    out_shape = [batch, q_tokens, heads, _LATENT]
    lse_shape = [batch, q_tokens, heads]
    blocks = topk // block_i
    scale_log2 = scale * _LOG2_E

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),  # type: ignore
        KV: T.Tensor(kv_shape, T.bfloat16),  # type: ignore
        Indices: T.Tensor(indices_shape, T.int32),  # type: ignore
        Output: T.Tensor(out_shape, T.bfloat16),  # type: ignore
        Lse: T.Tensor(lse_shape, T.float32),  # type: ignore
    ):
        with T.Kernel(q_tokens, batch, threads=threads) as (q_i, b_i):
            q_shared = T.alloc_shared([heads, _LATENT], T.bfloat16)
            q_rope_shared = T.alloc_shared([heads, _ROPE], T.bfloat16)
            kv_shared = T.alloc_shared([block_i, _LATENT], T.bfloat16)
            kv_rope_shared = T.alloc_shared([block_i, _ROPE], T.bfloat16)
            scores_shared = T.alloc_shared([heads, block_i], T.bfloat16)
            out_shared = T.alloc_shared([heads, _LATENT], T.bfloat16)
            valid = T.alloc_fragment([block_i], "bool")
            scores = T.alloc_fragment([heads, block_i], T.float32)
            output = T.alloc_fragment([heads, _LATENT], T.float32)
            row_sum = T.alloc_fragment([heads], T.float32)
            block_sum = T.alloc_fragment([heads], T.float32)
            row_max = T.alloc_fragment([heads], T.float32)
            previous_max = T.alloc_fragment([heads], T.float32)
            alpha = T.alloc_fragment([heads], T.float32)

            T.copy(Q[b_i, q_i, :, :_LATENT], q_shared)
            T.copy(Q[b_i, q_i, :, _LATENT:], q_rope_shared)
            T.fill(output, 0)
            T.fill(row_sum, 0)
            T.fill(row_max, -(2**30))

            for block in T.Pipelined(blocks, num_stages=num_stages):
                for i in T.Parallel(block_i):
                    index = Indices[b_i, q_i, block * block_i + i]
                    valid[i] = (index >= 0) & (index < kv_tokens - 1)
                for i, d in T.Parallel(block_i, _LATENT):
                    kv_shared[i, d] = KV[b_i, Indices[b_i, q_i, block * block_i + i], d]
                for i, d in T.Parallel(block_i, _ROPE):
                    kv_rope_shared[i, d] = KV[
                        b_i, Indices[b_i, q_i, block * block_i + i], _LATENT + d
                    ]
                for h, i in T.Parallel(heads, block_i):
                    scores[h, i] = T.if_then_else(valid[i], 0, -T.infinity(T.float32))
                T.gemm(
                    q_shared,
                    kv_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    q_rope_shared,
                    kv_rope_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(row_max, previous_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for h in T.Parallel(heads):
                    row_max[h] = T.max(row_max[h], previous_max[h])
                    alpha[h] = T.exp2((previous_max[h] - row_max[h]) * scale_log2)
                for h, i in T.Parallel(heads, block_i):
                    scores[h, i] = T.exp2((scores[h, i] - row_max[h]) * scale_log2)
                T.reduce_sum(scores, block_sum, dim=1)
                for h in T.Parallel(heads):
                    row_sum[h] = row_sum[h] * alpha[h] + block_sum[h]
                for h, d in T.Parallel(heads, _LATENT):
                    output[h, d] *= alpha[h]
                T.copy(scores, scores_shared)
                T.gemm(
                    scores_shared, kv_shared, output, policy=T.GemmWarpPolicy.FullRow
                )

            for h, d in T.Parallel(heads, _LATENT):
                output[h, d] /= T.max(row_sum[h], 1e-20)
            for h in T.Parallel(heads):
                row_sum[h] = T.if_then_else(
                    row_sum[h] > 0,
                    (T.log2(row_sum[h]) + row_max[h] * scale_log2) * _LN_2,
                    -T.infinity(T.float32),
                )
            T.copy(output, out_shared)
            T.copy(out_shared, Output[b_i, q_i, :, :])
            T.copy(row_sum, Lse[b_i, q_i, :])

    return main


@tilelang.jit(out_idx=[-1])
def _delta(heads, block=32, num_stages=5):
    batch = T.dynamic("batch")
    tokens = T.dynamic("tokens")
    shape = [batch, tokens, heads, _LATENT]

    @T.prim_func
    def main(
        Output: T.Tensor(shape, T.bfloat16),  # type: ignore
        GradOutput: T.Tensor(shape, T.bfloat16),  # type: ignore
        Delta: T.Tensor([batch, tokens, heads], T.float32),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(tokens, block), batch) as (h_i, t_i, b_i):
            output = T.alloc_fragment([block, block], T.float32)
            grad = T.alloc_fragment([block, block], T.float32)
            product = T.alloc_fragment([block, block], T.float32)
            result = T.alloc_fragment([block], T.float32)
            T.clear(product)
            for d_i in T.Pipelined(T.ceildiv(_LATENT, block), num_stages=num_stages):
                T.copy(
                    Output[
                        b_i,
                        t_i * block : (t_i + 1) * block,
                        h_i,
                        d_i * block : (d_i + 1) * block,
                    ],
                    output,
                )
                T.copy(
                    GradOutput[
                        b_i,
                        t_i * block : (t_i + 1) * block,
                        h_i,
                        d_i * block : (d_i + 1) * block,
                    ],
                    grad,
                )
                for i, d in T.Parallel(block, block):
                    product[i, d] += output[i, d] * grad[i, d]
            T.reduce_sum(product, result, dim=1)
            T.copy(result, Delta[b_i, t_i * block : (t_i + 1) * block, h_i])

    return main


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _backward(
    heads,
    topk,
    scale,
    dkv_splits=_DKV_SPLITS,
    block_i=32,
    num_stages=0,
    threads=256,
    use_tcgen_dq=False,
):
    batch = T.dynamic("batch")
    q_tokens = T.dynamic("q_tokens")
    kv_tokens = T.dynamic("kv_tokens")
    assert topk % block_i == 0
    q_shape = [batch, q_tokens, heads, _DIM]
    kv_shape = [batch, kv_tokens, 1, _DIM]
    grad_kv_shape = [batch, dkv_splits, kv_tokens, 1, _DIM]
    out_shape = [batch, q_tokens, heads, _LATENT]
    indices_shape = [batch, q_tokens, 1, topk]
    row_shape = [batch, q_tokens, heads]
    blocks = topk // block_i
    scale_log2 = scale * _LOG2_E
    split_store = 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),  # type: ignore
        KV: T.Tensor(kv_shape, T.bfloat16),  # type: ignore
        GradOutput: T.Tensor(out_shape, T.bfloat16),  # type: ignore
        Indices: T.Tensor(indices_shape, T.int32),  # type: ignore
        Lse: T.Tensor(row_shape, T.float32),  # type: ignore
        Delta: T.Tensor(row_shape, T.float32),  # type: ignore
        GradQ: T.Tensor(q_shape, T.bfloat16),  # type: ignore
        GradKV: T.Tensor(grad_kv_shape, T.float32),  # type: ignore
    ):
        with T.Kernel(q_tokens, batch, threads=threads) as (q_i, b_i):
            q_shared = T.alloc_shared([heads, _LATENT], T.bfloat16)
            q_rope_shared = T.alloc_shared([heads, _ROPE], T.bfloat16)
            kv_shared = T.alloc_shared([block_i, _LATENT], T.bfloat16)
            kv_rope_shared = T.alloc_shared([block_i, _ROPE], T.bfloat16)
            grad_out_shared = T.alloc_shared([heads, _LATENT], T.bfloat16)
            probabilities_shared = T.alloc_shared([heads, block_i], T.bfloat16)
            grad_scores_shared = T.alloc_shared([heads, block_i], T.bfloat16)
            grad_q_shared = T.alloc_shared([heads, _LATENT], T.bfloat16)
            grad_q_rope_shared = T.alloc_shared([heads, _ROPE], T.bfloat16)
            grad_kv_shared = T.alloc_shared(
                [block_i // split_store, _LATENT], T.float32
            )
            grad_kv_rope_shared = T.alloc_shared(
                [block_i // split_store, _ROPE], T.float32
            )
            if use_tcgen_dq:
                grad_q_tmem = T.alloc_tmem([heads, _LATENT], T.float32)
                grad_q_rope_tmem = T.alloc_tmem([heads, _ROPE], T.float32)
                grad_q_barrier = T.alloc_barrier(1)
                grad_q_rope_barrier = T.alloc_barrier(1)
            valid = T.alloc_fragment([block_i], "bool")
            probabilities = T.alloc_fragment([heads, block_i], T.float32)
            grad_probabilities = T.alloc_fragment([heads, block_i], T.float32)
            grad_q = T.alloc_fragment([heads, _LATENT], T.float32)
            grad_q_rope = T.alloc_fragment([heads, _ROPE], T.float32)
            grad_kv = T.alloc_fragment([block_i, _LATENT], T.float32)
            grad_kv_rope = T.alloc_fragment([block_i, _ROPE], T.float32)

            T.copy(Q[b_i, q_i, :, :_LATENT], q_shared)
            T.copy(Q[b_i, q_i, :, _LATENT:], q_rope_shared)
            T.copy(GradOutput[b_i, q_i, :, :], grad_out_shared)
            if not use_tcgen_dq:
                T.clear(grad_q)
                T.clear(grad_q_rope)

            for block in T.Pipelined(blocks, num_stages=num_stages):
                for i in T.Parallel(block_i):
                    index = Indices[b_i, q_i, 0, block * block_i + i]
                    valid[i] = (index >= 0) & (index < kv_tokens - 1)
                for h, i in T.Parallel(heads, block_i):
                    probabilities[h, i] = T.if_then_else(
                        valid[i], 0, -T.infinity(T.float32)
                    )
                for i, d in T.Parallel(block_i, _LATENT):
                    kv_shared[i, d] = KV[
                        b_i, Indices[b_i, q_i, 0, block * block_i + i], 0, d
                    ]
                for i, d in T.Parallel(block_i, _ROPE):
                    kv_rope_shared[i, d] = KV[
                        b_i,
                        Indices[b_i, q_i, 0, block * block_i + i],
                        0,
                        _LATENT + d,
                    ]
                T.gemm(
                    q_shared,
                    kv_shared,
                    probabilities,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.gemm(
                    q_rope_shared,
                    kv_rope_shared,
                    probabilities,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for h, i in T.Parallel(heads, block_i):
                    probabilities[h, i] = T.if_then_else(
                        valid[i] & (Lse[b_i, q_i, h] > -1e30),
                        T.exp2(
                            (probabilities[h, i] * scale - Lse[b_i, q_i, h]) * _LOG2_E
                        ),
                        0,
                    )
                T.copy(probabilities, probabilities_shared)
                T.gemm(
                    grad_out_shared,
                    kv_shared,
                    grad_probabilities,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                for h, i in T.Parallel(heads, block_i):
                    grad_probabilities[h, i] = (
                        probabilities[h, i]
                        * (grad_probabilities[h, i] - Delta[b_i, q_i, h])
                        * scale
                    )
                T.copy(grad_probabilities, grad_scores_shared)
                if use_tcgen_dq:
                    T.tcgen05_gemm(
                        grad_scores_shared,
                        kv_shared,
                        grad_q_tmem,
                        mbar=grad_q_barrier,
                        clear_accum=block == 0,
                    )
                    T.tcgen05_gemm(
                        grad_scores_shared,
                        kv_rope_shared,
                        grad_q_rope_tmem,
                        mbar=grad_q_rope_barrier,
                        clear_accum=block == 0,
                    )
                    T.mbarrier_wait_parity(grad_q_barrier, block % 2)
                    T.mbarrier_wait_parity(grad_q_rope_barrier, block % 2)
                else:
                    T.gemm(
                        grad_scores_shared,
                        kv_shared,
                        grad_q,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.gemm(
                        grad_scores_shared,
                        kv_rope_shared,
                        grad_q_rope,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                T.gemm(
                    grad_scores_shared,
                    q_shared,
                    grad_kv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    probabilities_shared,
                    grad_out_shared,
                    grad_kv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.clear(grad_kv_rope)
                T.gemm(
                    grad_scores_shared,
                    q_rope_shared,
                    grad_kv_rope,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for split in range(split_store):
                    for i, d in T.Parallel(block_i, _LATENT):
                        if i < block_i // split_store:
                            grad_kv_shared[i, d] = grad_kv[
                                i + split * (block_i // split_store), d
                            ]
                    for i, d in T.Parallel(block_i, _ROPE):
                        if i < block_i // split_store:
                            grad_kv_rope_shared[i, d] = grad_kv_rope[
                                i + split * (block_i // split_store), d
                            ]
                    for i, d in T.Parallel(block_i // split_store, _LATENT):
                        source = i + split * (block_i // split_store)
                        T.atomic_add(
                            GradKV[
                                b_i,
                                q_i % dkv_splits,
                                Indices[b_i, q_i, 0, block * block_i + source],
                                0,
                                d,
                            ],
                            grad_kv_shared[i, d],
                        )
                    for i, d in T.Parallel(block_i // split_store, _ROPE):
                        source = i + split * (block_i // split_store)
                        T.atomic_add(
                            GradKV[
                                b_i,
                                q_i % dkv_splits,
                                Indices[b_i, q_i, 0, block * block_i + source],
                                0,
                                _LATENT + d,
                            ],
                            grad_kv_rope_shared[i, d],
                        )

            if use_tcgen_dq:
                T.copy(grad_q_tmem, grad_q)
                T.copy(grad_q_rope_tmem, grad_q_rope)
            T.copy(grad_q, grad_q_shared)
            T.copy(grad_q_rope, grad_q_rope_shared)
            T.copy(grad_q_shared, GradQ[b_i, q_i, :, :_LATENT])
            T.copy(grad_q_rope_shared, GradQ[b_i, q_i, :, _LATENT:])

    return main


def forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = q.shape[2]
    kernel_heads = (heads + _HEAD_BLOCK - 1) // _HEAD_BLOCK * _HEAD_BLOCK
    if kernel_heads != heads:
        q = torch.cat(
            (q, q.new_zeros((*q.shape[:2], kernel_heads - heads, q.shape[3]))), dim=2
        )
    kv = torch.cat((kv, kv.new_zeros((kv.shape[0], 1, kv.shape[2]))), dim=1)
    with _preserve_env():
        output, lse = _forward(int(kernel_heads), int(indices.shape[-1]), float(scale))(
            q, kv, indices
        )
    return output[:, :, :heads], lse[:, :, :heads]


def backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = q.shape[2]
    kernel_heads = (heads + _HEAD_BLOCK - 1) // _HEAD_BLOCK * _HEAD_BLOCK
    if kernel_heads != heads:
        pad_shape = (*q.shape[:2], kernel_heads - heads)
        q = torch.cat((q, q.new_zeros((*pad_shape, q.shape[3]))), dim=2)
        output = torch.cat(
            (output, output.new_zeros((*pad_shape, output.shape[3]))), dim=2
        )
        grad_output = torch.cat(
            (grad_output, grad_output.new_zeros((*pad_shape, grad_output.shape[3]))),
            dim=2,
        )
        lse = torch.cat((lse, lse.new_zeros(pad_shape)), dim=2)
    kv = torch.cat((kv, kv.new_zeros((kv.shape[0], 1, kv.shape[2]))), dim=1)
    with _preserve_env():
        delta = _delta(int(kernel_heads))(output, grad_output)
        kv_grouped = kv.unsqueeze(2)
        indices_grouped = indices.unsqueeze(2)
        grad_kv = torch.zeros(
            (kv.shape[0], _DKV_SPLITS, kv.shape[1], 1, kv.shape[2]),
            device=kv.device,
            dtype=torch.float32,
        )
        grad_q = _backward(
            int(kernel_heads),
            int(indices.shape[-1]),
            float(scale),
            threads=min(256, int(kernel_heads) * 8),
            use_tcgen_dq=torch.cuda.get_device_capability(q.device)[0] == 10,
        )(q, kv_grouped, grad_output, indices_grouped, lse, delta, grad_kv)
    return (
        grad_q[:, :, :heads],
        grad_kv.sum(dim=1)[:, :-1].to(kv.dtype).squeeze(2),
    )
