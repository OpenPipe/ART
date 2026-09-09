import torch
from torch import nn


def freeze_parameters_as_buffers(module: nn.Module) -> None:
    for child in module.modules():
        for name, param in list(child.named_parameters(recurse=False)):
            del child._parameters[name]
            child.register_buffer(name, param.detach(), persistent=True)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform over the last dimension.

    DeepSeek-V4 uses this before activation FP8 simulation in the indexer and
    compressor. The supported ART dimensions are powers of two, specifically
    128 and 512 in the DSV4 Flash config.
    """
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"rotate_activation supports bf16/fp32, got {x.dtype}")
    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}.")
    y = x.float()
    h = 1
    while h < width:
        y = y.reshape(*x.shape[:-1], width // (2 * h), 2, h)
        left, right = y.unbind(dim=-2)
        y = torch.stack((left + right, left - right), dim=-2)
        y = y.reshape(*x.shape[:-1], width)
        h *= 2
    return (y * (width**-0.5)).to(x.dtype)
