from __future__ import annotations

from functools import cache
from importlib import import_module
from importlib.metadata import version
from typing import Any

import torch

CAUSAL_CONV1D_VERSION = "1.6.1"


@cache
def _extension_ops() -> tuple[Any, Any]:
    installed = version("causal-conv1d")
    if installed != CAUSAL_CONV1D_VERSION:
        raise RuntimeError(
            f"ART Mamba requires causal-conv1d {CAUSAL_CONV1D_VERSION}, got {installed}"
        )
    module = import_module("causal_conv1d.cpp_functions")
    return module.causal_conv1d_fwd_function, module.causal_conv1d_bwd_function


@torch.library.custom_op(
    "art::mamba_causal_conv1d_backward", mutates_args=(), device_types="cuda"
)
def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, backward = _extension_ops()
    dx, dweight, dbias, dinitial = backward(
        x,
        weight,
        bias,
        grad_output,
        None,
        initial,
        grad_final,
        None,
        True,
        True,
    )
    return dx, dweight, dbias, dinitial


@_backward.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output, grad_final
    return (
        torch.empty_like(x),
        torch.empty_like(weight),
        torch.empty_like(bias),
        torch.empty_like(initial),
    )


@torch.library.custom_op(
    "art::mamba_causal_conv1d", mutates_args=(), device_types="cuda"
)
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward, _ = _extension_ops()
    final = torch.empty(
        int(x.shape[0]),
        int(weight.shape[1]) - 1,
        int(x.shape[1]),
        dtype=x.dtype,
        device=x.device,
    ).transpose(1, 2)
    return forward(x, weight, bias, None, initial, final, True), final


@_forward.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del bias, initial
    final = torch.empty(
        x.shape[0],
        weight.shape[1] - 1,
        x.shape[1],
        dtype=x.dtype,
        device=x.device,
    ).transpose(1, 2)
    return torch.empty_like(x), final


def _setup_context(ctx: Any, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _autograd_backward(
    ctx: Any,
    grad_output: torch.Tensor | None,
    grad_final: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, weight, bias, initial = ctx.saved_tensors
    if grad_output is None:
        grad_output = torch.zeros_like(x)
    if grad_final is None:
        grad_final = torch.zeros_like(initial)
    return _backward(x, weight, bias, initial, grad_output, grad_final)


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward(x, weight, bias, initial)
