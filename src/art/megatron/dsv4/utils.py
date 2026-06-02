import torch


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform over the last dimension.

    DeepSeek-V4 uses this before activation FP8 simulation in the indexer and
    compressor. The supported ART dimensions are powers of two, specifically
    128 and 512 in the DSV4 Flash config.
    """
    assert x.dtype == torch.bfloat16
    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}.")
    y = x.float()
    h = 1
    while h < width:
        y = y.reshape(*y.shape[:-1], -1, 2, h)
        left = y[..., 0, :].clone()
        right = y[..., 1, :].clone()
        y[..., 0, :] = left + right
        y[..., 1, :] = left - right
        y = y.reshape(*x.shape)
        h *= 2
    return (y * (width**-0.5)).to(x.dtype)
