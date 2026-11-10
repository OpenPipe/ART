"""CPU regression tests for context-parallel merge cotangent ownership."""

import pytest
import torch

from art.megatron.context_parallel import executor as ex


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_lse_grad", [False, True])
def test_merge_tape_preserves_upstream_gradients(dtype, with_lse_grad):
    grad_out = torch.ones(2, 2, 4, dtype=dtype)
    grad_lse = torch.ones(2, 2) if with_lse_grad else None
    before_out = grad_out.clone()
    before_lse = None if grad_lse is None else grad_lse.clone()
    records = [
        dict(
            stage_out=torch.zeros_like(grad_out),
            stage_lse=torch.zeros(2, 2),
            merge_q_is_full=True,
            merge_is_copy=True,
        ),
        dict(
            stage_out=torch.zeros(2, 1, 4, dtype=dtype),
            stage_lse=torch.zeros(2, 1),
            merge_q_is_full=False,
            merge_is_copy=False,
            merge_q_index=torch.tensor([1]),
            merge_prev_out=torch.zeros(2, 1, 4),
            merge_prev_lse=torch.zeros(2, 1),
        ),
    ]
    out_grads, lse_grads = ex._merge_stage_output_grads_from_tape(
        replay_records=records, grad_output_flat=grad_out, grad_lse_flat=grad_lse
    )
    torch.testing.assert_close(grad_out, before_out, rtol=0, atol=0)
    if grad_lse is not None:
        assert before_lse is not None
        torch.testing.assert_close(grad_lse, before_lse, rtol=0, atol=0)
    wanted = torch.ones_like(grad_out)
    wanted[:, 1] = 0.5
    torch.testing.assert_close(out_grads[0], wanted)
    torch.testing.assert_close(out_grads[1], torch.full_like(out_grads[1], 0.5))
    assert all(torch.isfinite(grad).all() for grad in (*out_grads, *lse_grads))


def test_fully_masked_stage_merge_backward_is_finite():
    output = torch.zeros(2, 1, 4)
    lse = torch.full((2, 1), float("-inf"))
    gradients = ex._stage_merge_backward_values(
        prev_out=output,
        prev_lse=lse,
        stage_out=output,
        stage_lse=lse,
        grad_merged_out=torch.ones_like(output),
        grad_merged_lse=torch.ones_like(lse),
    )
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert all(torch.count_nonzero(grad) == 0 for grad in gradients)
