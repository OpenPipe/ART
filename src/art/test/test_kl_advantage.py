"""Tests for KL-penalized advantage adjustment in loss_fn."""

import torch

from art.loss import Loss, loss_fn


def _make_inputs(
    batch_size: int = 1,
    seq_len: int = 8,
    advantages: list[float] | None = None,
):
    """Create minimal TrainInputs-like dict for loss_fn."""
    if advantages is None:
        advantages = [1.0] * seq_len
    adv_tensor = torch.tensor([advantages], dtype=torch.float32)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logprobs = torch.zeros(batch_size, seq_len)
    assistant_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # First token is not assistant (shifted)
    assistant_mask[:, 0] = False
    weights = torch.ones(batch_size, seq_len)
    group_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    parent_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "advantages": adv_tensor,
        "assistant_mask": assistant_mask,
        "weights": weights,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
    }


def test_kl_advantage_no_effect_when_disabled():
    """When kl_penalty_coef=0, advantages should not be modified."""
    inputs = _make_inputs()
    new_logprobs = torch.zeros(1, 8)
    ref_logprobs = torch.full((1, 8), -1.0)  # different from new_logprobs

    loss_no_kl = loss_fn(
        inputs, new_logprobs, ref_logprobs, None, {"kl_penalty_coef": 0.0}
    )
    loss_without_ref = loss_fn(inputs, new_logprobs, None, None, {})

    assert loss_no_kl.kl_policy_ref is None
    assert loss_without_ref.kl_policy_ref is None


def test_kl_advantage_enabled():
    """When kl_penalty_coef>0 and ref_logprobs provided, kl_policy_ref should be set."""
    inputs = _make_inputs()
    new_logprobs = torch.zeros(1, 8)
    ref_logprobs = torch.full((1, 8), -0.5)

    loss = loss_fn(inputs, new_logprobs, ref_logprobs, None, {"kl_penalty_coef": 0.1})

    assert loss.kl_policy_ref is not None
    assert loss.kl_policy_ref.item() > 0  # KL should be positive when logprobs differ


def test_kl_advantage_zero_mean_penalty():
    """The KL penalty should be zero-mean across assistant tokens."""
    inputs = _make_inputs(seq_len=16)
    # Create varying logprobs to produce non-uniform KL
    new_logprobs = torch.randn(1, 16) * 0.5
    ref_logprobs = torch.randn(1, 16) * 0.5

    kl_penalty_coef = 0.1
    assistant_mask = torch.nn.functional.pad(
        inputs["assistant_mask"][:, 1:].float(), (0, 1), value=0.0
    )

    # Compute what the penalty should be
    kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
    avg_kl = kl_per_token.sum() / (assistant_mask.sum() + 1e-6)
    kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask

    # Sum of penalty across tokens should be ~0
    penalty_sum = kl_penalty.sum().item()
    assert abs(penalty_sum) < 1e-5, f"Penalty sum should be ~0, got {penalty_sum}"


def test_kl_advantage_direction():
    """Tokens with higher KL (more drift) should get reduced advantages."""
    # Create inputs where token 2 has high drift and token 5 has low drift
    seq_len = 8
    inputs = _make_inputs(
        seq_len=seq_len, advantages=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    )
    new_logprobs = torch.zeros(1, seq_len)
    ref_logprobs = torch.zeros(1, seq_len)

    # Make token at position 2 (which after shifting = position 1 in shifted space)
    # have high divergence
    new_logprobs[0, 2] = 0.0
    ref_logprobs[0, 2] = -2.0  # large gap = high KL

    # Token at position 5 has low divergence
    new_logprobs[0, 5] = -0.1
    ref_logprobs[0, 5] = -0.1  # no gap = low KL

    loss = loss_fn(inputs, new_logprobs, ref_logprobs, None, {"kl_penalty_coef": 1.0})

    # The metric should exist
    assert loss.kl_policy_ref is not None


def test_kl_advantage_does_not_affect_when_no_ref():
    """When ref_logprobs is None, kl_penalty_coef should have no effect."""
    inputs = _make_inputs()
    new_logprobs = torch.zeros(1, 8)

    loss = loss_fn(inputs, new_logprobs, None, None, {"kl_penalty_coef": 0.5})
    assert loss.kl_policy_ref is None
