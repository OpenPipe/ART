"""Small autograd oracle for KDA; never used as a CUDA training fallback."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def recurrent_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """HF recurrence, FP32 state (FP64 when explicitly checking derivatives)."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    query, key, value, decay, strength = (x.to(dtype) for x in (q, k, v, g, beta))
    query = query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query * (q.shape[-1] ** -0.5)
    boundaries = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = query.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        for index in range(start, end):
            state = state * decay[:, index].exp().unsqueeze(-1)
            residual = value[:, index] - (state * key[:, index, ..., None]).sum(-2)
            state = state + key[:, index, ..., None] * (
                residual * strength[:, index, :, None]
            ).unsqueeze(-2)
            outputs.append((state * query[:, index, ..., None]).sum(-2))
    if not outputs:
        return v * 0
    return torch.stack(outputs, dim=1).to(q.dtype)


def causal_conv(x: Tensor, weight: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    """Segmented causal depthwise convolution with HF's FP32 accumulation."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    boundaries = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        part = x[:, start:end].to(dtype).transpose(1, 2)
        conv = F.conv1d(
            F.pad(part, (weight.shape[-1] - 1, 0)),
            weight.to(dtype).unsqueeze(1),
            groups=x.shape[-1],
        )
        outputs.append(F.silu(conv).to(x.dtype).transpose(1, 2))
    return torch.cat(outputs, dim=1) if outputs else x * 0
