"""Deterministic backward preserves FLASH forward, including natural-log LSE."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 FLASH attention required",
)


@pytest.mark.parametrize("use_lse", [False, True])
@pytest.mark.parametrize(
    "length,dim,heads,kv_heads", [(256, 128, 8, 2), (183, 256, 16, 2), (65, 256, 24, 4)]
)
def test_masked_gqa_deterministic_backward(use_lse, length, dim, heads, kv_heads):
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask

    from art.megatron.flex_attn.compiled import (
        flash_sparse_block_size_for_head_dim,
        flex_backend_for_head_dims,
        get_dense_compiled_flex_attention,
        normalize_flex_lse,
    )

    original = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = getattr(torch.utils.deterministic, "fill_uninitialized_memory")
    generator = torch.Generator(device="cuda").manual_seed(108)
    q, k, v, grad = [
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        for shape in [
            (1, heads, length, dim),
            (1, kv_heads, length, dim),
            (1, kv_heads, length, dim),
            (1, heads, length, dim),
        ]
    ]

    def causal(batch, head, query, key):
        return query >= key

    try:
        torch.use_deterministic_algorithms(True)
        setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
        backend = flex_backend_for_head_dims(
            head_dim=dim, head_dim_v=dim, device=q.device
        )
        assert backend == "FLASH"
        attention = get_dense_compiled_flex_attention(
            backend=backend, head_dim=dim, head_dim_v=dim, device=q.device
        )
        block_size = flash_sparse_block_size_for_head_dim(
            head_dim=dim, head_dim_v=dim, device=q.device
        )
        mask = create_block_mask(
            causal, 1, None, length, length, device="cuda", BLOCK_SIZE=block_size
        )

        def forward(inputs):
            result = attention(
                *inputs,
                block_mask=mask,
                scale=dim**-0.5,
                enable_gqa=True,
                return_aux=AuxRequest(lse=True) if use_lse else None,
            )
            if use_lse:
                out, aux = result
                assert aux.lse is not None
                return out, normalize_flex_lse(aux.lse, backend=backend)
            return (result,)

        torch.use_deterministic_algorithms(False)
        baseline = forward((q, k, v))
        torch.use_deterministic_algorithms(True)
        runs = []
        for _ in range(3):
            inputs = [x.detach().requires_grad_() for x in (q, k, v)]
            outputs = forward(inputs)
            assert all(
                torch.equal(a, b) for a, b in zip(outputs, baseline, strict=True)
            )
            upstream = (grad, torch.full_like(outputs[1], 0.31)) if use_lse else (grad,)
            runs.append(
                (
                    *[x.detach() for x in outputs],
                    *torch.autograd.grad(outputs, inputs, upstream),
                )
            )
        for run in runs[1:]:
            assert all(torch.equal(a, b) for a, b in zip(runs[0], run, strict=True))

        rq, rk, rv = [x.double().requires_grad_() for x in (q, k, v)]
        scores = (
            rq @ rk.repeat_interleave(heads // kv_heads, 1).transpose(-1, -2)
        ) * dim**-0.5
        visible = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~visible, -torch.inf)
        expected = scores.softmax(-1) @ rv.repeat_interleave(heads // kv_heads, 1)
        reference_outputs = (expected, scores.logsumexp(-1)) if use_lse else (expected,)
        oracle = torch.autograd.grad(
            reference_outputs, (rq, rk, rv), tuple(x.double() for x in upstream)
        )
        for actual, reference in zip(runs[0][-3:], oracle, strict=True):
            assert torch.isfinite(actual).all()
            assert (actual.double() - reference).norm() / reference.norm() <= 0.01
    finally:
        setattr(torch.utils.deterministic, "fill_uninitialized_memory", fill)
        torch.use_deterministic_algorithms(original, warn_only=warn_only)
