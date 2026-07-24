from __future__ import annotations

import math

import pytest
import torch

from art.distill.reference_loss import topk_plus_tail_forward_kl as reference_kl
from art.megatron.distillation_loss import topk_plus_tail_forward_kl


def _teacher_logprobs(
    probabilities: list[list[list[float]]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = torch.tensor(probabilities, dtype=torch.float64, device=device)
    tail_probs = 1.0 - teacher_probs.sum(dim=-1)
    return teacher_probs.log().to(torch.float32), tail_probs.log().to(torch.float32)


def _assert_matches_reference(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    logical_vocab_size = 7
    raw_logits = torch.tensor(
        [
            [
                [0.3, -0.7, 1.2, 0.1, -1.1, 0.8, -0.2, 99.0, -99.0],
                [-0.5, 0.4, 0.2, 1.5, -0.8, 0.7, -1.2, -99.0, 99.0],
            ],
            [
                [1.1, -0.3, 0.5, -0.9, 0.0, 0.2, 0.7, 50.0, 60.0],
                [-0.2, 0.6, -1.0, 0.9, 0.4, -0.4, 1.3, 70.0, 80.0],
            ],
        ],
        dtype=torch.float64,
        device=device,
    )
    topk_ids = torch.tensor(
        [[[2, 5], [3, 1]], [[0, 6], [6, 3]]], dtype=torch.long, device=device
    )
    teacher_logprobs, tail_logprob = _teacher_logprobs(
        [
            [[0.45, 0.25], [0.52, 0.13]],
            [[0.31, 0.29], [0.41, 0.17]],
        ],
        device=device,
    )
    mask = torch.tensor([[True, True], [False, True]], device=device)
    weights = torch.tensor([[1.0, 0.25], [math.nan, 1.5]], device=device)
    temperature = 1.7
    coefficient = 0.6

    logits = raw_logits.to(dtype).detach().requires_grad_(True)
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=topk_ids,
        teacher_topk_logprobs=teacher_logprobs,
        teacher_tail_logprob=tail_logprob,
        mask=mask,
        weights=weights,
        logical_vocab_size=logical_vocab_size,
        temperature=temperature,
        compensate_temperature_squared=True,
    )
    objective_loss = result.loss_sum * coefficient
    objective_loss.backward()
    actual_gradient = logits.grad
    assert actual_gradient is not None

    reference_logits = raw_logits.detach().clone().requires_grad_(True)
    reference = reference_kl(
        reference_logits,
        teacher_topk_ids=topk_ids,
        teacher_topk_probs=teacher_logprobs.to(torch.float64).exp(),
        teacher_tail_prob=tail_logprob.to(torch.float64).exp(),
        mask=mask,
        weights=torch.nan_to_num(weights.to(torch.float64)),
        logical_vocab_size=logical_vocab_size,
        student_temperature=temperature,
        compensate_temperature_squared=True,
    )
    expected_loss = reference.loss_sum * coefficient
    expected_loss.backward()
    expected_gradient = reference_logits.grad
    assert expected_gradient is not None

    torch.testing.assert_close(
        objective_loss.to(torch.float64), expected_loss, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        actual_gradient.to(torch.float64),
        expected_gradient,
        rtol=rtol,
        atol=atol,
    )
    assert result.token_count.dtype == torch.int64
    assert result.token_count.item() == 3
    torch.testing.assert_close(
        (result.selected_loss_sum + result.tail_loss_sum) * coefficient,
        objective_loss,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        result.teacher_selected_mass_sum + result.teacher_tail_mass_sum,
        torch.tensor(3.0, device=device),
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(
        result.student_selected_mass_sum + result.student_tail_mass_sum,
        torch.tensor(3.0, device=device),
        rtol=2e-4,
        atol=2e-5,
    )
    assert torch.count_nonzero(actual_gradient[~mask]).item() == 0
    assert torch.count_nonzero(actual_gradient[..., logical_vocab_size:]).item() == 0
    assert result.numerical_clamp_count.item() == 0


def test_cpu_value_and_gradient_match_float64_reference() -> None:
    _assert_matches_reference(
        device=torch.device("cpu"),
        dtype=torch.float32,
        rtol=3e-5,
        atol=3e-6,
    )


def test_tail_heavy_distribution_and_finite_nonzero_gradient() -> None:
    logits = torch.tensor(
        [[12.0, 11.0, -2.0, -3.0, -4.0, -5.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_topk_ids = torch.tensor([[0, 1]])
    teacher_probs = torch.tensor([[0.01, 0.01]])
    teacher_tail_prob = torch.tensor([0.98])
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_probs.log(),
        teacher_tail_logprob=teacher_tail_prob.log(),
        mask=torch.tensor([True]),
        logical_vocab_size=6,
    )
    result.loss_sum.backward()
    assert torch.isfinite(result.loss_sum)
    assert result.loss_sum.item() > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() > 0
    assert result.teacher_tail_mass_sum.item() == pytest.approx(0.98)
    assert result.student_tail_mass_sum.item() < 1e-5


def test_near_unit_selected_mass_uses_exact_tail_with_nonzero_gradient() -> None:
    logits = torch.tensor(
        [[80.0, 80.0, -80.0, -90.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_probs = torch.tensor([[0.45, 0.45]])
    teacher_tail_prob = torch.tensor([0.1])
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[0, 1]]),
        teacher_topk_logprobs=teacher_probs.log(),
        teacher_tail_logprob=teacher_tail_prob.log(),
        mask=torch.tensor([True]),
        logical_vocab_size=4,
    )
    result.loss_sum.backward()

    reference_logits = logits.detach().to(torch.float64).requires_grad_(True)
    reference = reference_kl(
        reference_logits,
        teacher_topk_ids=torch.tensor([[0, 1]]),
        teacher_topk_probs=teacher_probs.to(torch.float64),
        teacher_tail_prob=teacher_tail_prob.to(torch.float64),
        mask=torch.tensor([True]),
        logical_vocab_size=4,
    )
    reference.loss_sum.backward()

    assert torch.isfinite(result.loss_sum)
    torch.testing.assert_close(
        result.loss_sum.to(torch.float64),
        reference.loss_sum,
        rtol=2e-5,
        atol=2e-5,
    )
    assert logits.grad is not None
    assert reference_logits.grad is not None
    torch.testing.assert_close(
        logits.grad.to(torch.float64),
        reference_logits.grad,
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].item() < 0
    assert logits.grad[0, 3].item() < 0
    assert result.teacher_tail_mass_sum.item() == pytest.approx(0.1)
    assert result.numerical_clamp_count.item() == 0


def test_inactive_sentinel_rows_are_harmless() -> None:
    logits = torch.tensor(
        [[0.2, 0.5, -0.1], [math.nan, math.inf, -math.inf]],
        requires_grad=True,
    )
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[1], [-1]]),
        teacher_topk_logprobs=torch.tensor([[math.log(0.7)], [math.nan]]),
        teacher_tail_logprob=torch.tensor([math.log(0.3), math.nan]),
        mask=torch.tensor([True, False]),
        weights=torch.tensor([2.0, math.nan]),
        logical_vocab_size=3,
    )
    result.loss_sum.backward()
    assert torch.isfinite(result.loss_sum)
    assert logits.grad is not None
    assert torch.equal(logits.grad[1], torch.zeros(3))
    assert result.per_token_loss[1].item() == 0
    assert result.per_token_weighted_loss[1].item() == 0
    assert result.token_count.item() == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("logits", math.nan, "student logits"),
        ("teacher", math.nan, "top-k log-probabilities"),
        ("tail", math.inf, "tail log-probabilities"),
        ("weight", math.nan, "weights"),
    ],
)
def test_nonfinite_active_inputs_fail(field: str, value: float, match: str) -> None:
    logits = torch.tensor([[0.2, 0.5, -0.1]])
    teacher = torch.tensor([[math.log(0.7)]])
    tail = torch.tensor([math.log(0.3)])
    weight = torch.tensor([1.0])
    if field == "logits":
        logits[0, 0] = value
    elif field == "teacher":
        teacher[0, 0] = value
    elif field == "tail":
        tail[0] = value
    else:
        weight[0] = value
    with pytest.raises(ValueError, match=match):
        topk_plus_tail_forward_kl(
            logits,
            teacher_topk_ids=torch.tensor([[1]]),
            teacher_topk_logprobs=teacher,
            teacher_tail_logprob=tail,
            mask=torch.tensor([True]),
            weights=weight,
            logical_vocab_size=3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-5, 5e-6),
        (torch.bfloat16, 3e-2, 3e-3),
    ],
)
def test_cuda_value_and_gradient_match_float64_reference(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    _assert_matches_reference(
        device=torch.device("cuda"),
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )
