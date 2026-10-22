"""Native FLA gate gradients must keep their per-root reduction geometry."""

import pytest
import torch

from art.megatron.segmented_gdn import native_gdn_reductions


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_batched_recurrence_matches_separate_native_gate_gradients(
    packed, checkpointed
):
    if not torch.cuda.is_available():
        pytest.skip("one CUDA device required")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(731)
    n, length, heads, width = 16, 1024, 32, 128
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(n, length, heads, width, device="cuda"), dim=-1
        )
        .bfloat16()
        .requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn_like(q, requires_grad=True)
    g = (-torch.rand(n, length, heads, device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(
        n, length, heads, device="cuda", dtype=q.dtype, requires_grad=True
    )
    dy = torch.randn_like(v)
    variables = [q, k, v, g, beta]
    expected = torch.cat(
        [
            chunk_gated_delta_rule(
                q[i : i + 1],
                k[i : i + 1],
                v[i : i + 1],
                g=g[i : i + 1],
                beta=beta[i : i + 1],
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )[0]
            for i in range(n)
        ]
    )
    expected.backward(dy)
    gradients = [x.grad.clone() for x in variables]
    for x in variables:
        x.grad = None
    args = variables
    kwargs = {}
    if packed:
        args = [x.reshape(1, n * length, *x.shape[2:]) for x in variables]
        boundaries = torch.arange(n + 1, dtype=torch.int32) * length
        kwargs = {"cu_seqlens": boundaries.cuda(), "cu_seqlens_cpu": boundaries}

    def run(*values):
        return chunk_gated_delta_rule(
            *values[:3],
            g=values[3],
            beta=values[4],
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            **kwargs,
        )[0]

    from torch.utils import checkpoint as checkpoint_module

    from art.megatron.lora import install_lora_checkpoint_context_hooks

    install_lora_checkpoint_context_hooks()
    with native_gdn_reductions():
        actual = (
            checkpoint_module.checkpoint(run, *args, use_reentrant=True)
            if checkpointed
            else run(*args)
        ).reshape_as(expected)
    # Scope has exited before backward: FLA's context must retain it.
    actual.backward(dy)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for x, gradient in zip(variables, gradients, strict=True):
        torch.testing.assert_close(x.grad, gradient, atol=0, rtol=0)
