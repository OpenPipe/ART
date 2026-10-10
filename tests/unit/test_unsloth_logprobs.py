import math

import pytest
import torch

from art.unsloth.train import _calculate_logprobs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_logprobs_and_gradients_match_fp64(dtype: torch.dtype, chunk_size: int) -> None:
    hidden_states = torch.tensor(
        [
            [[35.0, 31.5, 30.0, 29.0], [29.0, 35.0, 28.0, 27.0], [1, 2, 3, 4]],
            [[30.0, 29.0, 35.0, 31.5], [28.0, 27.0, 29.0, 35.0], [4, 3, 2, 1]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    lm_head_t = torch.eye(4, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[0, 1, 2], [2, 3, 1]])

    logprobs, entropy = _calculate_logprobs(
        lm_head_t, hidden_states, targets, chunk_size
    )
    (-logprobs.sum()).backward()

    ref_hidden = hidden_states.detach().double().requires_grad_()
    ref_head = lm_head_t.detach().double().requires_grad_()
    ref_full = torch.log_softmax(ref_hidden @ ref_head, dim=-1)
    ref_logprobs = ref_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ref_entropy = -(ref_full.exp() * ref_full).sum(-1)
    (-ref_logprobs.sum()).backward()

    assert logprobs.dtype == entropy.dtype == torch.float32
    assert (logprobs < 0).all()
    torch.testing.assert_close(logprobs.double(), ref_logprobs, rtol=1e-4, atol=4e-6)
    torch.testing.assert_close(entropy.double(), ref_entropy, rtol=1e-4, atol=4e-6)
    assert hidden_states.grad is not None
    assert ref_hidden.grad is not None
    assert lm_head_t.grad is not None
    assert ref_head.grad is not None
    assert (hidden_states.grad.gather(-1, targets.unsqueeze(-1)) != 0).all()
    torch.testing.assert_close(
        hidden_states.grad.double(), ref_hidden.grad, rtol=6e-3, atol=4e-6
    )
    torch.testing.assert_close(
        lm_head_t.grad.double(), ref_head.grad, rtol=1e-2, atol=2e-2
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_large_finite_logits_have_normalized_logprobs_and_entropy(
    dtype: torch.dtype,
) -> None:
    hidden_states = torch.tensor([[[1000.0, 999.0, 998.0, 997.0]]], dtype=dtype)
    targets = torch.tensor([[0]])
    logprobs, entropy = _calculate_logprobs(
        torch.eye(4), hidden_states, targets, chunk_size=1
    )
    ref_full = torch.log_softmax(hidden_states.double(), -1)

    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(
        logprobs.double(), ref_full[..., 0], rtol=1e-4, atol=4e-5
    )
    torch.testing.assert_close(
        entropy.double(), -(ref_full.exp() * ref_full).sum(-1), rtol=1e-4, atol=4e-5
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_near_certain_target_keeps_logprob_and_gradient(dtype: torch.dtype) -> None:
    hidden_states = torch.tensor(
        [[[35.0, 21.0, 0.0, 0.0]]], dtype=dtype, requires_grad=True
    )
    targets = torch.tensor([[0]])
    logprobs, _ = _calculate_logprobs(torch.eye(4), hidden_states, targets, 1)
    (-logprobs.sum()).backward()

    ref_hidden = hidden_states.detach().double().requires_grad_()
    ref_logprobs = torch.log_softmax(ref_hidden, dim=-1)[..., 0]
    (-ref_logprobs.sum()).backward()

    assert (logprobs < 0).all()
    assert hidden_states.grad is not None
    assert ref_hidden.grad is not None
    assert (hidden_states.grad[..., 0] != 0).all()
    torch.testing.assert_close(logprobs.double(), ref_logprobs, rtol=1e-2, atol=1e-9)
    torch.testing.assert_close(
        hidden_states.grad[..., 0].double(),
        ref_hidden.grad[..., 0],
        rtol=1e-2,
        atol=1e-9,
    )


def test_entropy_does_not_require_grad() -> None:
    hidden_states = torch.randn(1, 3, 4, requires_grad=True)
    logprobs, entropy = _calculate_logprobs(
        torch.eye(4), hidden_states, torch.tensor([[0, 1, 2]]), chunk_size=2
    )
    assert logprobs.requires_grad
    assert not entropy.requires_grad
    (-logprobs.sum()).backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_fp16_flat_vocab_has_finite_logprobs_and_entropy() -> None:
    vocab_size = 151936
    logprobs, entropy = _calculate_logprobs(
        torch.ones(1, vocab_size, dtype=torch.float16),
        torch.zeros(1, 1, 1, dtype=torch.float16),
        torch.zeros(1, 1, dtype=torch.long),
        chunk_size=1,
    )
    expected_entropy = torch.full((1, 1), math.log(vocab_size))
    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(logprobs, -expected_entropy, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-6, atol=1e-6)
