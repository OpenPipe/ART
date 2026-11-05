"""Frozen indexer equation from serving image 83f9988e (GLM-5.3 Flash).

This portable implementation follows common/attention.py:301-355 and
nvidia/ops/kpool_compress.py:26-231: FP32 pooling, normalized FWHT-128,
BF16 rotation materialization, E4M3FN with ue8m0 scales, and FP32 scores.
It is an equation port, not a DeepGEMM kernel or a performance claim.
"""

from __future__ import annotations

import torch


def hadamard128(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 128:
        raise ValueError("The serving indexer requires head dimension 128")
    result = value.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        groups = result.reshape(*value.shape[:-1], 128 // (2 * stride), 2, stride)
        left, right = groups.unbind(-2)
        result = torch.stack((left + right, left - right), -2).flatten(-3)
    return (result * 128**-0.5).bfloat16()


def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotated = hadamard128(value).float()
    scale = torch.exp2(
        torch.ceil(
            torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    return (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def compress(
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    pool_starts: torch.Tensor,
) -> torch.Tensor:
    tokens = pool_starts[:, None] + torch.arange(4, device=keys.device)
    logits = gates[tokens].float() + ape.float()[None]
    return (logits.softmax(1) * keys[tokens].float()).sum(1).bfloat16()


def scores(
    q: torch.Tensor,
    quantized_pools: torch.Tensor,
    pool_scales: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weights are the raw FP32 head projection, without H or D scaling."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        qq, qs = quantize(q)
        dots = torch.matmul(qq.float(), quantized_pools.float().T)
        dots = dots * qs * pool_scales.T
        return (dots.relu() * weights[..., None] * (q.shape[1] * 128) ** -0.5).sum(1)
