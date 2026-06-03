from typing import Any

import torch


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    Both the cast (via :func:`act_quant`) and the cast-back step are routed
    through ``deepseek-ai/TileKernels`` so we share the same FP8 kernels with
    the rest of the DeepSeek stack.
    """
    from art.megatron.dsv4.kernel.act_quant import act_quant

    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = y_flat.float() * scale_flat.float().repeat_interleave(block_size, dim=-1)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        return grad_outputs[0], None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
