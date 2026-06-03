"""Block-wise FP8 activation quantization for DeepSeek-V4.

Ported verbatim from deepseek-ai/DeepSeek-V4-Pro/inference/kernel.py to keep
bit-exact parity with the upstream inference kernel. Keep this file in sync
when DeepSeek updates the reference implementation.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py
"""

from typing import Any, Optional, Tuple, cast

import torch

from art.megatron.dsv4.kernel.tilelang_import import (
    import_tilelang,
    preserve_tilelang_env,
    sanitize_tilelang_env,
)

_tilelang, _T = import_tilelang()

tilelang = cast(Any, _tilelang)
T = cast(Any, _T)

should_deepgemm_weight_requant_ue8m0 = None


tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N,
    block_size=128,
    in_dtype=BF16,
    out_dtype=FP8,
    scale_dtype=FP32,
    round_scale=False,
    inplace=False,
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(
                                compute_dtype,
                                T.Cast(
                                    out_dtype,
                                    T.clamp(
                                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                                    ),
                                ),
                            )
                            * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


sanitize_tilelang_env()


def _resolve_scale_dtype(scale_fmt: Optional[str], block_size: int) -> torch.dtype:
    """Pick scale dtype consistent with the SGLang runtime:
    - UE8M0 (``torch.float8_e8m0fnu``) on Blackwell when DeepGEMM JIT is on.
    - FP32 elsewhere (Hopper, no DeepGEMM, sglang absent) — DeepSeek default.
    Only matters when scales are rounded-to-pow2 (``scale_fmt`` set);
    plain ``scale_fmt=None`` stays fp32 regardless of hardware.
    """
    if scale_fmt is None or should_deepgemm_weight_requant_ue8m0 is None:
        return torch.float32
    return (
        torch.float8_e8m0fnu
        if should_deepgemm_weight_requant_ue8m0(weight_block_size=block_size)
        else torch.float32
    )


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP).

    ``scale_dtype=None`` (default) auto-selects fp32 vs ``float8_e8m0fnu`` based
    on the SGLang runtime (Blackwell + DeepGEMM → UE8M0, otherwise fp32).
    Pass an explicit dtype to override.
    """
    N = x.size(-1)
    assert N % block_size == 0
    if scale_dtype is None:
        scale_dtype = _resolve_scale_dtype(scale_fmt, block_size)
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    z = x.contiguous()
    y = (
        torch.empty_like(z)
        if inplace
        else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    )
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    with preserve_tilelang_env():
        kernel = act_quant_kernel(
            N,
            block_size,
            scale_dtype=tl_dtype,
            round_scale=scale_fmt is not None,
            inplace=inplace,
        )
        kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s
