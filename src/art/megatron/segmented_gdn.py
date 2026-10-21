"""Opt-in native sequence geometry for FLA's scalar gate reductions.

FLA autotunes cumulative sums by batch size. Different warp counts change the
FP32 reverse sum tree and therefore alpha gradients, even with exact Q/K/V and
outputs. Keep the heavy recurrent kernels batched; only these small reductions
use each independent root's native geometry. Capture the scope on FLA's autograd
context so backward does not depend on the caller's ContextVars.
"""

from contextlib import contextmanager
import contextvars
from functools import partial
import importlib

import torch

_ACTIVE = contextvars.ContextVar("art_native_gdn_reductions", default=False)


def _cumsum(original, tensor, chunk_size, **kwargs):
    if not _ACTIVE.get():
        return original(tensor, chunk_size, **kwargs)
    cu = kwargs.get("cu_seqlens")
    if cu is None:
        if tensor.shape[0] == 1:
            return original(tensor, chunk_size, **kwargs)
        return torch.cat(
            [
                original(part.contiguous(), chunk_size, **kwargs)
                for part in tensor.split(1)
            ],
            dim=0,
        )
    if tensor.shape[0] != 1 or kwargs.get("head_first", False):
        raise ValueError("native wave GDN requires batch-one packed gate streams")
    boundaries = cu.detach().cpu().tolist()
    local = dict(kwargs, cu_seqlens=None, chunk_indices=None)
    return torch.cat(
        [
            original(tensor[:, start:stop].contiguous(), chunk_size, **local)
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ],
        dim=1,
    )


def _forward(original, ctx, *args, **kwargs):
    ctx._art_native_gdn_reductions = _ACTIVE.get()
    return original(ctx, *args, **kwargs)


def _backward(original, ctx, *args, **kwargs):
    token = _ACTIVE.set(ctx._art_native_gdn_reductions)
    try:
        return original(ctx, *args, **kwargs)
    finally:
        _ACTIVE.reset(token)


def _install(module):
    function = module.ChunkGatedDeltaRuleFunction
    if getattr(function, "_art_native_reductions", False):
        return
    module.chunk_local_cumsum = partial(_cumsum, module.chunk_local_cumsum)
    function.forward = staticmethod(partial(_forward, function.forward))
    function.backward = staticmethod(partial(_backward, function.backward))
    function._art_native_reductions = True


@contextmanager
def native_gdn_reductions():
    """Explicit independent-root MoE scope; ordinary forwards stay unchanged."""
    _install(importlib.import_module("fla.ops.gated_delta_rule.chunk"))
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
