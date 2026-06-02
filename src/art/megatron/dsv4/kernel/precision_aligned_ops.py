from typing import Any

import torch


class _BFloat16LinearFP32Func(torch.autograd.Function):
    # Forward matches SGLang's default DeepSeek-V4 compressor path
    # (`sglang.jit_kernel.deepseek_v4.linear_bf16_fp32`, cublas backend):
    # BF16 activation x BF16 weight -> FP32 output. This keeps Megatron's
    # compressor log-prob computation aligned with SGLang rollout. Backward is
    # only needed for training, so keep its gradient matmuls in FP32.
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        ctx.save_for_backward(x_bf16, weight_bf16)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype

        x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
        out = torch.mm(x_2d, weight_bf16.t(), out_dtype=torch.float32)
        return out.view(*x.shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        x_bf16, weight_bf16 = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.matmul(weight_bf16.float())
            grad_x = grad_x.view(ctx.input_shape).to(ctx.input_dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
            grad_weight = grad_output_2d.t().matmul(x_2d.float()).to(ctx.weight_dtype)

        return grad_x, grad_weight


@torch.compiler.disable
def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _BFloat16LinearFP32Func.apply(x, weight)
