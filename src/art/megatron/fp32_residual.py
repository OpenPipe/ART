from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F


def _fp32_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    scale = weight.float()
    if zero_centered_gamma:
        scale = scale + 1.0
    return x_fp32 * rstd * scale, rstd


class _FP32InputRMSNormFn(torch.autograd.Function):
    """Exact FP32-input RMSNorm emitting lower-precision output via TE C++ kernels.

    Transformer Engine's ``LayerNormLinear`` and ``RMSNorm`` either reject FP32
    inputs with BF16 weights outside autocast or silently cast inputs to BF16
    before normalization. Calling TE's C++ ``rmsnorm_fwd`` / ``rmsnorm_bwd``
    directly with FP32 ``x`` and FP32 ``gamma`` preserves full FP32 residual
    accuracy while emitting ``out_dtype`` (BF16) activations and FP32 ``dx``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out_dtype: torch.dtype,
        zero_centered_gamma: bool,
        fwd_sm_margin: int,
        bwd_sm_margin: int,
    ) -> torch.Tensor:
        orig_shape = x.shape
        hidden_size = int(weight.numel())
        x_2d = x.contiguous().view(-1, hidden_size)
        rows = int(x_2d.shape[0])
        ctx.rows = rows
        ctx.orig_shape = orig_shape
        ctx.eps = float(eps)
        ctx.zero_centered_gamma = bool(zero_centered_gamma)
        ctx.bwd_sm_margin = int(bwd_sm_margin)
        ctx.is_cuda = x_2d.is_cuda

        if not x_2d.is_cuda:
            y_fp32, rstd = _fp32_rmsnorm_reference(
                x_2d, weight.view(hidden_size), ctx.eps, ctx.zero_centered_gamma
            )
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(x_2d, rstd, weight)
            return y_fp32.to(dtype=out_dtype).view(orig_shape)

        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.cpp_extensions import rmsnorm_fwd

        pad = (-rows) % 4
        x_padded = F.pad(x_2d, (0, 0, 0, pad)) if pad else x_2d
        w_fp32 = weight.float().contiguous().view(hidden_size)
        y_padded, _, rstd = rmsnorm_fwd(
            x_padded,
            w_fp32,
            ctx.eps,
            None,
            None,
            TE_DType[out_dtype],
            int(fwd_sm_margin),
            ctx.zero_centered_gamma,
        )
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x_padded, rstd, weight)
        y = y_padded[:rows] if pad else y_padded
        return y.view(orig_shape)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (grad_output,) = grad_outputs
        x_saved, rstd, weight = ctx.saved_tensors
        rows = int(ctx.rows)
        hidden_size = int(x_saved.shape[-1])
        dy_2d = grad_output.contiguous().view(rows, hidden_size).float()

        if not ctx.is_cuda:
            scale = weight.float().view(hidden_size)
            if ctx.zero_centered_gamma:
                scale = scale + 1.0
            x_fp32 = x_saved.float()
            x_norm = x_fp32 * rstd
            dx: torch.Tensor | None = None
            if ctx.needs_input_grad[0]:
                dy_scaled = dy_2d * scale
                mean_dot = (dy_scaled * x_norm).mean(dim=-1, keepdim=True)
                dx = (
                    ((dy_scaled - x_norm * mean_dot) * rstd)
                    .to(dtype=x_saved.dtype)
                    .view(ctx.orig_shape)
                )
            dw: torch.Tensor | None = None
            if ctx.needs_input_grad[1]:
                dw = (
                    (dy_2d * x_norm)
                    .sum(dim=0)
                    .to(dtype=weight.dtype)
                    .view(weight.shape)
                )
            return dx, dw, None, None, None, None, None

        from transformer_engine.pytorch.cpp_extensions import rmsnorm_bwd

        pad = int(x_saved.shape[0]) - rows
        dy_padded = F.pad(dy_2d, (0, 0, 0, pad)) if pad else dy_2d
        w_fp32 = weight.float().contiguous().view(hidden_size)
        dx_padded, dw_fp32 = rmsnorm_bwd(
            dy_padded,
            x_saved,
            rstd,
            w_fp32,
            ctx.bwd_sm_margin,
            ctx.zero_centered_gamma,
        )
        dx = (
            (dx_padded[:rows] if pad else dx_padded).view(ctx.orig_shape)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            dw_fp32.to(dtype=weight.dtype).view(weight.shape)
            if ctx.needs_input_grad[1]
            else None
        )
        return dx, dw, None, None, None, None, None


@torch.compiler.disable
def _fp32_input_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    out_dtype: torch.dtype,
    zero_centered_gamma: bool,
    fwd_sm_margin: int = 0,
    bwd_sm_margin: int = 0,
) -> torch.Tensor:
    if int(x.numel()) == 0:
        zero = (x.sum() * 0.0 + weight.float().sum() * 0.0).to(dtype=out_dtype)
        return zero.expand(x.shape).clone()
    return cast(
        torch.Tensor,
        _FP32InputRMSNormFn.apply(
            x,
            weight,
            float(eps),
            out_dtype,
            bool(zero_centered_gamma),
            int(fwd_sm_margin),
            int(bwd_sm_margin),
        ),
    )


@torch.compiler.disable
def _fp32_input_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    eps: float,
    out_dtype: torch.dtype,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    if int(x.numel()) == 0:
        zero = x.sum() * 0.0 + weight.float().sum() * 0.0
        if isinstance(bias, torch.Tensor):
            zero = zero + bias.float().sum() * 0.0
        return zero.to(dtype=out_dtype).expand(x.shape).clone()
    scale = weight.float() + 1.0 if zero_centered_gamma else weight.float()
    bias_fp32 = bias.float() if isinstance(bias, torch.Tensor) else None
    out = F.layer_norm(
        x.float(),
        tuple(weight.shape),
        scale,
        bias_fp32,
        float(eps),
    )
    return out.to(dtype=out_dtype)


@torch.compiler.disable
def _fp32_layernorm_linear_forward(
    module: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
    ln_weight: torch.Tensor,
) -> Any:
    out_dtype = weight.dtype
    normalization = str(getattr(module, "normalization", "LayerNorm"))
    eps = float(getattr(module, "eps", 1e-5))
    zero_centered_gamma = bool(getattr(module, "zero_centered_gamma", False))
    if normalization == "RMSNorm":
        fwd_sm_margin = int(
            getattr(
                module,
                "fwd_ln_sm_margin" if torch.is_grad_enabled() else "inf_ln_sm_margin",
                0,
            )
        )
        bwd_sm_margin = int(getattr(module, "bwd_ln_sm_margin", 0))
        ln_out = _fp32_input_rmsnorm(
            x,
            ln_weight,
            eps=eps,
            out_dtype=out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            fwd_sm_margin=fwd_sm_margin,
            bwd_sm_margin=bwd_sm_margin,
        )
    elif normalization == "LayerNorm":
        ln_out = _fp32_input_layernorm(
            x,
            ln_weight,
            getattr(module, "layer_norm_bias", None),
            eps=eps,
            out_dtype=out_dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
    else:
        raise ValueError(
            f"Unsupported normalization {normalization!r} for FP32 residual connection"
        )

    tp_size = int(getattr(module, "tp_size", 1) or 1)
    sequence_parallel = bool(getattr(module, "sequence_parallel", False))
    if tp_size > 1:
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            gather_from_sequence_parallel_region,
        )

        tp_group = getattr(module, "tp_group", None)
        if tp_group is None:
            tp_group = getattr(module, "_tp_group", None)
        if sequence_parallel:
            linear_inp = gather_from_sequence_parallel_region(
                ln_out,
                tensor_parallel_output_grad=True,
                group=tp_group,
            )
        else:
            linear_inp = copy_to_tensor_model_parallel_region(
                ln_out,
                group=tp_group,
            )
    else:
        linear_inp = ln_out

    apply_bias = bool(getattr(module, "apply_bias", False))
    bias_tensor = getattr(module, "bias", None)
    has_bias = isinstance(bias_tensor, torch.Tensor) and int(bias_tensor.numel()) > 0
    linear_bias = bias_tensor if apply_bias and has_bias else None
    linear_out = F.linear(linear_inp, weight, linear_bias)
    if getattr(module, "is_first_microbatch", False):
        module.is_first_microbatch = False

    return_layernorm_output = bool(getattr(module, "return_layernorm_output", False))
    return_layernorm_output_gathered = bool(
        getattr(module, "return_layernorm_output_gathered", False)
    )
    te_return_bias = bool(getattr(module, "te_return_bias", False))
    returned_bias = (
        bias_tensor.to(dtype=out_dtype)
        if te_return_bias and isinstance(bias_tensor, torch.Tensor)
        else None
    )

    if return_layernorm_output:
        ln_returned = (
            linear_inp
            if sequence_parallel and return_layernorm_output_gathered
            else ln_out
        )
        if te_return_bias:
            return linear_out, returned_bias, ln_returned
        return (linear_out, ln_returned), None
    if te_return_bias:
        return linear_out, returned_bias
    return linear_out, None


def _slice_mod4_padded_output(
    val: Any,
    orig_rows: int,
    padded_prefix: tuple[int, ...],
    expected_ndim: int,
) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim == expected_ndim and tuple(val.shape[:-1]) == padded_prefix:
            # Clone (not .contiguous()) so the sliced tensor owns its storage
            # (_base is None), matching the unpadded rows % 4 == 0 path and
            # avoiding Dynamo view-base recompiles on resumed frames.
            return val[:orig_rows].clone(memory_format=torch.contiguous_format)
        return val
    if isinstance(val, tuple):
        return tuple(
            _slice_mod4_padded_output(item, orig_rows, padded_prefix, expected_ndim)
            for item in val
        )
    if isinstance(val, list):
        return [
            _slice_mod4_padded_output(item, orig_rows, padded_prefix, expected_ndim)
            for item in val
        ]
    return val


@torch.compiler.disable
def _call_with_mod4_row_padding(
    fn: Callable[..., Any],
    module: Any,
    x: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Pad leading dim-0 rows to a multiple of 4 for batch-invariant TE RMSNorm.

    Transformer Engine 2.11 selects different CUDA RMSNorm kernels depending on
    ``rows % 4 == 0`` vs ``rows % 4 != 0``, causing up to 1 BF16 ULP divergence
    on identical prefix rows across adjacent sequence lengths. Padding dim-0 by
    at most 3 zero rows and slicing outputs back keeps every forward/backward on
    the same vectorized kernel with zero gradient contribution from padded rows.
    Sequence-parallel modules are left untouched because SP all-gathers along
    dim-0 across TP ranks.
    """
    if (
        not bool(getattr(module, "sequence_parallel", False))
        and not any(isinstance(arg, torch.Tensor) for arg in args)
        and isinstance(x, torch.Tensor)
        and x.ndim >= 2
        and int(x.shape[0]) > 0
        and int(x.shape[0]) % 4 != 0
    ):
        orig_rows = int(x.shape[0])
        pad_rows = (-orig_rows) % 4
        xp = F.pad(x, (0, 0) * (x.ndim - 1) + (0, pad_rows))
        out = fn(module, xp, *args, **kwargs)
        return _slice_mod4_padded_output(
            out,
            orig_rows,
            tuple(xp.shape[:-1]),
            x.ndim,
        )
    return fn(module, x, *args, **kwargs)


def disable_te_parameter_transpose_cache(model_chunks: Sequence[Any]) -> None:
    """Disable TE parameter transpose caching across dynamic sequence shapes."""
    for chunk in model_chunks:
        if not isinstance(chunk, torch.nn.Module):
            continue
        for module in chunk.modules():
            if hasattr(module, "disable_parameter_transpose_cache"):
                module.disable_parameter_transpose_cache = True


def _is_patched(fn: Any) -> bool:
    if getattr(fn, "__art_fp32_residual__", False):
        return True
    orig = getattr(fn, "_torchdynamo_orig_callable", None)
    return bool(orig is not None and getattr(orig, "__art_fp32_residual__", False))


def _patch_te_layernorm_column_parallel_linear() -> None:
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )

    original_forward = TELayerNormColumnParallelLinear.forward
    if _is_patched(original_forward):
        return

    def _fp32_layernorm_column_parallel_linear_forward(
        self: Any,
        x: torch.Tensor,
    ) -> Any:
        weight = getattr(self, "weight", None)
        ln_weight = getattr(self, "layer_norm_weight", None)
        if (
            isinstance(x, torch.Tensor)
            and x.dtype == torch.float32
            and isinstance(weight, torch.Tensor)
            and weight.dtype != torch.float32
            and isinstance(ln_weight, torch.Tensor)
            and str(getattr(self, "normalization", "LayerNorm"))
            in {"RMSNorm", "LayerNorm"}
        ):
            return _fp32_layernorm_linear_forward(self, x, weight, ln_weight)
        return _call_with_mod4_row_padding(original_forward, self, x)

    setattr(
        _fp32_layernorm_column_parallel_linear_forward, "__art_fp32_residual__", True
    )
    setattr(
        TELayerNormColumnParallelLinear,
        "forward",
        _fp32_layernorm_column_parallel_linear_forward,
    )


def _patch_te_layernorm_linear() -> None:
    import transformer_engine.pytorch as te

    original_forward = te.LayerNormLinear.forward
    if _is_patched(original_forward):
        return

    def _mod4_te_layernorm_linear_forward(
        self: Any,
        inp: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return _call_with_mod4_row_padding(original_forward, self, inp, *args, **kwargs)

    setattr(_mod4_te_layernorm_linear_forward, "__art_fp32_residual__", True)
    setattr(te.LayerNormLinear, "forward", _mod4_te_layernorm_linear_forward)


def _patch_te_rmsnorm() -> None:
    import transformer_engine.pytorch as te

    original_forward = te.RMSNorm.forward
    if _is_patched(original_forward):
        return

    def _fp32_rmsnorm_forward(
        self: Any,
        input: torch.Tensor,
        *extra_inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        weight = getattr(self, "weight", None)
        if (
            isinstance(input, torch.Tensor)
            and input.dtype == torch.float32
            and isinstance(weight, torch.Tensor)
            and weight.dtype != torch.float32
            and not extra_inputs
            and not kwargs
        ):
            sm_margins = getattr(self, "_sm_margins", None) or {}
            fwd_sm_margin = int(
                sm_margins.get(
                    "forward" if torch.is_grad_enabled() else "inference",
                    0,
                )
            )
            bwd_sm_margin = int(sm_margins.get("backward", 0))
            return _fp32_input_rmsnorm(
                input,
                weight,
                eps=float(getattr(self, "eps", 1e-5)),
                out_dtype=weight.dtype,
                zero_centered_gamma=bool(getattr(self, "zero_centered_gamma", False)),
                fwd_sm_margin=fwd_sm_margin,
                bwd_sm_margin=bwd_sm_margin,
            )
        return _call_with_mod4_row_padding(
            original_forward, self, input, *extra_inputs, **kwargs
        )

    setattr(_fp32_rmsnorm_forward, "__art_fp32_residual__", True)
    setattr(te.RMSNorm, "forward", _fp32_rmsnorm_forward)


def _patch_moe_layer_postprocess() -> None:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    original_postprocess = MoELayer.postprocess
    if _is_patched(original_postprocess):
        return
    was_compile_disabled = bool(
        getattr(original_postprocess, "__art_compile_disabled__", False)
    )
    inner_postprocess = getattr(
        original_postprocess, "_torchdynamo_orig_callable", original_postprocess
    )

    def _fp32_moe_postprocess(
        self: Any,
        output: torch.Tensor,
        shared_expert_output: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            shared_expert_output is None
            or not bool(
                getattr(
                    getattr(self, "config", None),
                    "fp32_residual_connection",
                    False,
                )
            )
            or output.dtype == torch.float32
        ):
            return inner_postprocess(self, output, shared_expert_output)
        output = inner_postprocess(self, output, None)
        return output.float() + shared_expert_output.float()

    setattr(_fp32_moe_postprocess, "__art_fp32_residual__", True)
    patched: Any = _fp32_moe_postprocess
    if was_compile_disabled:
        patched = torch.compiler.disable(_fp32_moe_postprocess)
        setattr(patched, "__art_compile_disabled__", True)
        setattr(patched, "__art_fp32_residual__", True)
    setattr(MoELayer, "postprocess", patched)


def install_fp32_residual_patches() -> None:
    """Install FP32 residual and batch-invariant mod-4 RMSNorm patches for TE/MoE."""
    _patch_te_layernorm_column_parallel_linear()
    _patch_te_layernorm_linear()
    _patch_te_rmsnorm()
    _patch_moe_layer_postprocess()
