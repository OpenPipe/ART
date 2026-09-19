from __future__ import annotations

import math

import pytest
import torch

from art.trainer_rank import (
    ForwardOutput,
    ImportanceSamplingGradientCorrection,
    ResolvedForwardOptions,
    TopK,
)
from art.trainer_rank._corrections import capture_forward_corrections


def _fixture(policy="when_available", *, logits=False):
    target = torch.tensor([-0.5, -1.0], requires_grad=True)
    top_k = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    tokens = torch.tensor([[2, 0]])
    hidden = torch.zeros(2, 3, requires_grad=True)
    logits_tensor = torch.zeros(1, 4, requires_grad=True) if logits else None
    output = ForwardOutput(target, TopK(top_k, tokens), logits_tensor, hidden)
    # Deliberately different from dataclass order: caller defines flat indices.
    tensors = (hidden, target, tokens, top_k) + (
        () if logits_tensor is None else (logits_tensor,)
    )
    context = capture_forward_corrections(
        {"nested": [[output]]},
        tensors,
        ResolvedForwardOptions(
            stale_gradient_corrections=(
                ImportanceSamplingGradientCorrection(policy=policy),
            )
        ),
    )
    return context, tensors


def test_capture_owns_only_original_logprobs_and_ids_on_cpu() -> None:
    context, tensors = _fixture()
    assert len(context.tensors) == 3
    assert all(
        tensor.device.type == "cpu" and not tensor.requires_grad
        for tensor in context.tensors
    )
    original = tuple(tensor.clone() for tensor in context.tensors)
    with torch.no_grad():
        tensors[1].fill_(-10)
        tensors[2].fill_(3)
        tensors[3].fill_(-10)
    for actual, expected in zip(context.tensors, original, strict=True):
        torch.testing.assert_close(actual, expected)


def test_context_applies_only_logprob_gradients_and_preserves_unused_outputs() -> None:
    context, tensors = _fixture()
    gradients = (torch.ones_like(tensors[0]), torch.ones_like(tensors[1]), None, None)
    current = (tensors[0], tensors[1] - 1, tensors[2], tensors[3])
    result = context.correct(gradients, current)
    assert result[0] is gradients[0]
    assert result[2:] == (None, None)
    torch.testing.assert_close(result[1], torch.full_like(tensors[1], math.exp(-1.0)))
    torch.testing.assert_close(gradients[1], torch.ones_like(tensors[1]))
    assert context.requires_current(gradients) is False
    assert context.correct(gradients)[1] is gradients[1]


def test_always_requires_data_only_for_active_eligible_outputs() -> None:
    context, tensors = _fixture("always")
    assert not context.requires_current((torch.ones_like(tensors[0]), None, None, None))
    hidden_grad = torch.ones_like(tensors[0])
    assert context.correct((hidden_grad, None, None, None))[0] is hidden_grad
    assert context.requires_current((None, torch.ones_like(tensors[1]), None, None))
    with pytest.raises(RuntimeError, match="requires current logprobs"):
        context.correct((None, torch.ones_like(tensors[1]), None, None))
    # Merely requesting an unused unsupported output does not require correction.
    assert not context.requires_current((None, None, None, None))


def test_reordered_top_k_is_aligned_by_original_token_ids() -> None:
    context, tensors = _fixture()
    current = (tensors[0], tensors[1], tensors[2].flip(-1), (tensors[3] - 1).flip(-1))
    result = context.correct((None, None, None, torch.ones_like(tensors[3])), current)
    torch.testing.assert_close(result[3], torch.exp(torch.full_like(tensors[3], -1)))


@pytest.mark.parametrize("policy", ["when_available", "always"])
def test_changed_top_k_membership_is_unavailable_without_original_id_logits(
    policy,
) -> None:
    context, tensors = _fixture(policy)
    current = (tensors[0], tensors[1], torch.tensor([[1, 3]]), tensors[3])
    gradients = (None, None, None, torch.ones_like(tensors[3]))
    if policy == "always":
        with pytest.raises(RuntimeError, match="requires current logprobs"):
            context.correct(gradients, current)
    else:
        assert context.correct(gradients, current)[3] is gradients[3]


def test_available_full_logits_correct_changed_top_k_without_another_forward() -> None:
    context, tensors = _fixture("always", logits=True)
    current_logits = torch.tensor([[1.0, 0.0, -0.5, 0.5]])
    current = (
        tensors[0],
        tensors[1],
        torch.tensor([[1, 3]]),
        tensors[3],
        current_logits,
    )
    result = context.correct(
        (None, None, None, torch.ones_like(tensors[3]), None), current
    )
    expected = (
        current_logits.log_softmax(-1).gather(-1, tensors[2]) - tensors[3]
    ).exp()
    torch.testing.assert_close(result[3], expected)


def test_context_checks_packet_layout_and_does_not_partially_modify_gradients() -> None:
    context, tensors = _fixture("always")
    target_grad, top_grad = torch.ones_like(tensors[1]), torch.ones_like(tensors[3])
    with pytest.raises(ValueError, match="output count"):
        context.correct((target_grad,))
    with pytest.raises(RuntimeError, match="requires current logprobs"):
        context.correct(
            (None, target_grad, None, top_grad),
            (tensors[0], tensors[1] - 1, torch.tensor([[1, 3]]), tensors[3]),
        )
    torch.testing.assert_close(target_grad, torch.ones_like(target_grad))
    torch.testing.assert_close(top_grad, torch.ones_like(top_grad))


@pytest.mark.parametrize("with_logits", [False, True])
def test_top_k_ties_do_not_assume_stable_membership(with_logits: bool) -> None:
    context, tensors = _fixture("always", logits=with_logits)
    tied_logits = torch.zeros(1, 4)
    tied_logprobs = torch.full((1, 2), -math.log(4))
    current = (tensors[0], tensors[1], torch.tensor([[1, 3]]), tied_logprobs)
    gradients = (None, None, None, torch.ones_like(tensors[3]))
    if with_logits:
        result = context.correct(gradients + (None,), current + (tied_logits,))
        torch.testing.assert_close(result[3], (tied_logprobs - tensors[3]).exp())
    else:
        with pytest.raises(RuntimeError, match="requires current logprobs"):
            context.correct(gradients, current)


def test_zero_cotangent_masks_do_not_require_defined_sampling_support() -> None:
    from art.trainer_rank._corrections import correct_logprob_cotangent

    original = torch.tensor([-0.5, -float("inf"), float("nan")])
    current = torch.tensor([-1.5, -float("inf"), float("nan")])
    gradient = torch.tensor([1.0, 0.0, 0.0])
    corrected = correct_logprob_cotangent(
        gradient,
        original_logprobs=original,
        current_logprobs=current,
        correction=ImportanceSamplingGradientCorrection(policy="always"),
    )
    torch.testing.assert_close(corrected, torch.tensor([math.exp(-1), 0, 0]))
    torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0, 0.0]))


def test_zero_cotangents_do_not_require_current_data_under_always() -> None:
    context, tensors = _fixture("always")
    zero = torch.zeros_like(tensors[1])
    gradients = (None, zero, None, None)
    assert not context.requires_current(gradients)
    assert context.correct(gradients)[1] is zero


def test_only_active_top_k_ids_require_current_membership() -> None:
    context, tensors = _fixture("always")
    current = (tensors[0], tensors[1], torch.tensor([[2, 3]]), tensors[3] - 1)
    result = context.correct((None, None, None, torch.tensor([[1.0, 0.0]])), current)
    torch.testing.assert_close(result[3], torch.tensor([[math.exp(-1), 0.0]]))


def test_bad_cotangent_shape_is_rejected_before_requesting_replay() -> None:
    context, _ = _fixture("always")
    with pytest.raises(ValueError, match="shape must match"):
        context.requires_current((None, torch.ones(1), None, None))


@pytest.mark.parametrize("corrections", [(), (ImportanceSamplingGradientCorrection(),)])
def test_current_replay_rejects_changed_active_top_k_events_even_without_correction(
    corrections,
) -> None:
    values = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    tokens = torch.tensor([[2, 0]])
    output = ForwardOutput(None, TopK(values, tokens), None, None)
    context = capture_forward_corrections(
        output,
        (values, tokens),
        ResolvedForwardOptions(stale_gradient_corrections=corrections),
    )
    gradients = (torch.ones_like(values), None)
    context.validate_replay(gradients, (values - 1, tokens))
    for changed in (tokens.flip(-1), torch.tensor([[2, 3]])):
        with pytest.raises(RuntimeError, match="changed active top-k token identities"):
            context.validate_replay(gradients, (values - 1, changed))
    torch.testing.assert_close(gradients[0], torch.ones_like(values))


def test_current_replay_ignores_inactive_top_k_event_changes() -> None:
    context, tensors = _fixture("always")
    current = (tensors[0], tensors[1], torch.tensor([[2, 3]]), tensors[3] - 1)
    context.validate_replay((None, None, None, torch.tensor([[1.0, 0.0]])), current)
    context.validate_replay((None, None, None, torch.zeros_like(tensors[3])), current)
    context.validate_replay((None, None, None, None), current)


def test_current_ratio_evaluation_can_reorder_but_physical_replay_cannot() -> None:
    context, tensors = _fixture("always", logits=True)
    gradients = (None, None, None, torch.ones_like(tensors[3]), None)
    current = (
        tensors[0],
        tensors[1],
        tensors[2].flip(-1),
        (tensors[3] - 1).flip(-1),
        tensors[4],
    )
    torch.testing.assert_close(
        context.correct(gradients, current)[3],
        torch.exp(torch.full_like(tensors[3], -1)),
    )
    with pytest.raises(RuntimeError, match="changed active top-k token identities"):
        context.validate_replay(gradients, current)
