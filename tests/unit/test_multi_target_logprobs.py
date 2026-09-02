from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from art.megatron.multi_target_logprobs import _row_offset
from art.megatron.selective_lm_head import selected_target_logprobs


@triton.jit
def _row_offset_probe(output, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    tl.store(output + row, _row_offset(row, STRIDE))


class _SingleTargetModel(torch.nn.Module):
    post_process = True
    mtp_process = False
    output_layer = SimpleNamespace(gather_output=False)
    config = SimpleNamespace(
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        mtp_num_layers=0,
    )

    def __init__(self) -> None:
        super().__init__()
        self.loss_calls = 0

    def compute_language_model_loss(
        self, *, labels: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        self.loss_calls += 1
        return F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)


def test_single_target_uses_model_cross_entropy() -> None:
    model = _SingleTargetModel()
    logits = torch.randn(4, 11, requires_grad=True)
    targets = torch.tensor([[1], [5], [3], [9]])

    actual = selected_target_logprobs(model, logits, targets)
    expected = torch.log_softmax(logits, dim=-1).gather(1, targets)

    assert model.loss_calls == 1
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multi_target_kernel_uses_64_bit_row_offsets() -> None:
    rows = 9000
    local_vocab_size = 248320
    offsets = torch.empty(rows, device="cuda", dtype=torch.int64)

    _row_offset_probe[(rows,)](offsets, STRIDE=local_vocab_size)

    expected = (rows - 1) * local_vocab_size
    assert expected > torch.iinfo(torch.int32).max
    assert offsets[-1].item() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multi_target_kernel_reuses_logits_and_matches_dense_gradient() -> None:
    from art.megatron.multi_target_logprobs import (
        vocab_parallel_multi_target_logprobs,
    )

    source = torch.randn(5, 257, device="cuda", requires_grad=True)
    logits = source * 1.0
    reference = source.detach().clone().requires_grad_()
    targets = torch.tensor(
        [[1, 37, -1], [5, 5, 200], [0, 256, 19], [83, -1, 7], [4, 9, 11]],
        device="cuda",
    )
    coefficients = torch.randn_like(targets, dtype=torch.float32)
    storage = logits.data_ptr()

    actual = vocab_parallel_multi_target_logprobs(logits, targets)
    valid = targets >= 0
    expected = (
        torch.log_softmax(reference, dim=-1)
        .gather(1, targets.clamp_min(0))
        .masked_fill(~valid, 0.0)
    )

    assert logits.data_ptr() == storage
    torch.testing.assert_close(logits, torch.softmax(reference.detach(), dim=-1))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(source.grad, reference.grad, rtol=2e-5, atol=2e-5)
