"""Training VJPs, FP32 handoff and expert scheduling regression tests (CPU)."""

import pytest
import torch

from art.megatron.batch_invariant import ops


@pytest.mark.parametrize(
    "counts,shared",
    [(None, None), ((2, 0, 5), "a"), ((2, 0, 5), "b"), ((2, 0, 5), None)],
)
@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_lora_fp64_vjp_scaling_and_empty_experts(counts, shared, scale):
    torch.manual_seed(10)
    shape_a = (3, 7, 3) if counts else (7, 3)
    shape_b = (3, 3, 9) if counts else (3, 9)
    if shared == "a":
        shape_a = (1, 7, 3)
    if shared == "b":
        shape_b = (1, 3, 9)
    x, a, b = [
        torch.randn(s, dtype=torch.float64, requires_grad=True)
        for s in ((7, 7), shape_a, shape_b)
    ]
    dy = torch.randn(7, 9, dtype=torch.float64)
    y = ops.lora(x, a, b, counts, scale=scale)
    if counts:
        reference = (
            torch.cat(
                [
                    (part @ a[0 if len(a) == 1 else i]) @ b[0 if len(b) == 1 else i]
                    for i, part in enumerate(x.split(counts))
                ]
            )
            * scale
        )
    else:
        reference = (x @ a) @ b * scale
    actual = torch.autograd.grad(y, (x, a, b), dy)
    expected = torch.autograd.grad(reference, (x, a, b), dy)
    torch.testing.assert_close(y, reference)
    for g, r in zip(actual, expected, strict=True):
        torch.testing.assert_close(g, r)


def test_bf16_lora_fp32_handoff_and_zero_suffix():
    torch.manual_seed(11)
    x = torch.randn(65, 17, dtype=torch.bfloat16, requires_grad=True)
    a = torch.nn.Parameter(torch.randn(17, 8, dtype=x.dtype) * 0.01)
    b = torch.nn.Parameter(torch.randn(8, 19, dtype=x.dtype) * 0.01)
    for p in (a, b):
        setattr(p, "_stark_fp32_gradients", True)
    dy = torch.randn(65, 19, dtype=x.dtype)
    ops.lora(x, a, b, scale=4).backward(dy)
    assert x.grad is not None
    dx, da, db = (
        x.grad.clone(),
        getattr(a, "main_grad").clone(),
        getattr(b, "main_grad").clone(),
    )
    tmp = ops._mm(x.detach(), a.detach())
    scaled = dy * 4
    dt = torch.mm(scaled, b.detach().T)
    torch.testing.assert_close(
        da, torch.mm(x.detach().float().T, dt.float()), rtol=0, atol=0
    )
    torch.testing.assert_close(
        db, torch.mm(tmp.float().T, scaled.float()), rtol=0, atol=0
    )
    assert a.grad is None and b.grad is None and da.dtype == torch.float32
    assert not torch.equal(da, da.bfloat16().float())
    # Repeat accumulation consumes each contribution once, through saved aliases.
    ops.lora(x, a, b, scale=4).backward(dy)
    torch.testing.assert_close(getattr(a, "main_grad"), 2 * da, rtol=0, atol=0)
    setattr(a, "main_grad", None)
    setattr(b, "main_grad", None)
    long = torch.cat((x.detach(), torch.randn(1, 17, dtype=x.dtype))).requires_grad_()
    ops.lora(long, a, b, scale=4).backward(
        torch.cat((dy, torch.zeros(1, 19, dtype=x.dtype)))
    )
    assert long.grad is not None
    assert torch.equal(long.grad[:65], dx)
    torch.testing.assert_close(getattr(a, "main_grad"), da, atol=1e-5, rtol=2e-5)
    torch.testing.assert_close(getattr(b, "main_grad"), db, atol=1e-5, rtol=2e-5)
    assert not torch.cuda.is_initialized()


def test_single_launch_expert_contract_transpose_and_wgrad():
    torch.manual_seed(12)
    weights = tuple(
        torch.randn(9, 17, dtype=torch.float64, requires_grad=True).T for _ in range(3)
    )
    x = torch.randn(7, 17, dtype=torch.float64, requires_grad=True)
    y = ops.grouped_matmul(x, weights, (2, 0, 5))
    ref = torch.cat([p @ w for p, w in zip(x.split((2, 0, 5)), weights, strict=True)])
    dy = torch.randn_like(y)
    for a, b in zip(
        torch.autograd.grad(y, (x, *weights), dy),
        torch.autograd.grad(ref, (x, *weights), dy),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
