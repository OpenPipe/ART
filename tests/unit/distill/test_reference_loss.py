from typing import Any

import pytest
import torch

from art.distill.reference_loss import topk_plus_tail_forward_kl


def _dense_forward_kl(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_log_probs = torch.log_softmax(logits / temperature, dim=-1)
    return torch.xlogy(teacher, teacher).sum(dim=-1) - (
        teacher * student_log_probs
    ).sum(dim=-1)


def test_hand_computed_k_plus_one_result() -> None:
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64))

    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[2, 0]]),
        teacher_topk_probs=torch.tensor([[0.6, 0.1]], dtype=torch.float64),
        teacher_tail_prob=torch.tensor([0.3], dtype=torch.float64),
        mask=torch.tensor([True]),
    )

    expected = 0.6 * torch.log(torch.tensor(0.6 / 0.5))
    expected += 0.1 * torch.log(torch.tensor(0.1 / 0.2))
    expected += 0.3 * torch.log(torch.tensor(0.3 / 0.3))
    assert result.loss_sum.item() == pytest.approx(expected.item())
    assert result.token_count.item() == 1
    assert result.teacher_tail_mass_sum.item() == pytest.approx(0.3)
    assert result.student_tail_mass_sum.item() == pytest.approx(0.3)
    assert result.loss_sum == pytest.approx(
        result.selected_loss_sum + result.tail_loss_sum
    )


def test_k_equals_vocab_matches_dense_loss_and_gradient() -> None:
    logits = torch.tensor(
        [[0.7, -1.1, 2.3], [-0.2, 0.4, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=torch.float64)
    ids = torch.tensor([[2, 0, 1], [1, 2, 0]])
    reordered_teacher = torch.gather(teacher, -1, ids)

    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=ids,
        teacher_topk_probs=reordered_teacher,
        teacher_tail_prob=torch.zeros(2, dtype=torch.float64),
        mask=torch.ones(2, dtype=torch.bool),
    )
    reference = _dense_forward_kl(logits, teacher).sum()
    result_gradient = torch.autograd.grad(result.loss_sum, logits, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, logits)[0]

    torch.testing.assert_close(result.loss_sum, reference)
    torch.testing.assert_close(result_gradient, reference_gradient)
    torch.testing.assert_close(
        result.student_tail_mass_sum, torch.tensor(0.0, dtype=torch.float64)
    )


def test_aggregated_tail_matches_coarsened_dense_reference() -> None:
    logits = torch.tensor(
        [[1.2, -0.3, 0.8, 2.1], [-1.0, 0.1, 0.4, 0.7]],
        dtype=torch.float64,
    )
    teacher = torch.tensor(
        [[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.3]], dtype=torch.float64
    )
    ids = torch.tensor([[2, 0], [3, 1]])
    explicit = torch.gather(teacher, -1, ids)
    tail = 1.0 - explicit.sum(dim=-1)

    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=ids,
        teacher_topk_probs=explicit,
        teacher_tail_prob=tail,
        mask=torch.ones(2, dtype=torch.bool),
    )

    student = logits.softmax(dim=-1)
    selected_student = torch.gather(student, -1, ids)
    student_tail = 1.0 - selected_student.sum(dim=-1)
    reference = (
        torch.xlogy(explicit, explicit).sum(dim=-1)
        - (explicit * selected_student.log()).sum(dim=-1)
        + torch.xlogy(tail, tail)
        - tail * student_tail.log()
    )
    torch.testing.assert_close(result.per_token_loss, reference)
    torch.testing.assert_close(result.loss_sum, reference.sum())


def test_topk_order_does_not_change_result() -> None:
    logits = torch.tensor([[0.2, 1.0, -0.4, 0.7]], dtype=torch.float64)
    kwargs: dict[str, Any] = {
        "student_logits": logits,
        "teacher_tail_prob": torch.tensor([0.3]),
        "mask": torch.tensor([True]),
    }
    ordered = topk_plus_tail_forward_kl(
        **kwargs,
        teacher_topk_ids=torch.tensor([[0, 3]]),
        teacher_topk_probs=torch.tensor([[0.2, 0.5]]),
    )
    reversed_order = topk_plus_tail_forward_kl(
        **kwargs,
        teacher_topk_ids=torch.tensor([[3, 0]]),
        teacher_topk_probs=torch.tensor([[0.5, 0.2]]),
    )
    torch.testing.assert_close(ordered.loss_sum, reversed_order.loss_sum)
    torch.testing.assert_close(
        ordered.student_tail_mass_sum, reversed_order.student_tail_mass_sum
    )


def test_mask_weights_and_count_have_independent_semantics() -> None:
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[0], [0], [0]]),
        teacher_topk_probs=torch.tensor([[0.75], [0.75], [0.75]]),
        teacher_tail_prob=torch.tensor([0.25, 0.25, 0.25]),
        mask=torch.tensor([True, False, True]),
        weights=torch.tensor([2.0, 100.0, 0.0]),
    )
    row_loss = 0.75 * torch.log(torch.tensor(0.75 / 0.5))
    row_loss += 0.25 * torch.log(torch.tensor(0.25 / 0.5))

    assert result.token_count.item() == 2
    assert result.loss_sum.item() == pytest.approx(2 * row_loss.item())
    torch.testing.assert_close(
        result.per_token_weighted_loss,
        torch.tensor([2 * row_loss, 0.0, 0.0], dtype=torch.float64),
    )
    assert result.teacher_tail_mass_sum.item() == pytest.approx(0.5)


def test_student_temperature_and_optional_t_squared_compensation() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    teacher = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    ids = torch.tensor([[0, 1, 2]])
    common: dict[str, Any] = {
        "student_logits": logits,
        "teacher_topk_ids": ids,
        "teacher_topk_probs": teacher,
        "teacher_tail_prob": torch.tensor([0.0], dtype=torch.float64),
        "mask": torch.tensor([True]),
        "student_temperature": 2.0,
    }

    plain = topk_plus_tail_forward_kl(**common)
    compensated = topk_plus_tail_forward_kl(
        **common, compensate_temperature_squared=True
    )

    torch.testing.assert_close(
        plain.loss_sum, _dense_forward_kl(logits, teacher, temperature=2.0).sum()
    )
    torch.testing.assert_close(compensated.loss_sum, plain.loss_sum * 4)


def test_extreme_finite_logits_have_finite_loss_and_gradients() -> None:
    logits = torch.tensor(
        [[1000.0, -1000.0, 0.0], [-800.0, 800.0, 799.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[0], [2]]),
        teacher_topk_probs=torch.tensor([[0.8], [0.7]], dtype=torch.float64),
        teacher_tail_prob=torch.tensor([0.2, 0.3], dtype=torch.float64),
        mask=torch.tensor([True, True]),
    )
    result.loss_sum.backward()

    assert torch.isfinite(result.loss_sum)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"teacher_topk_ids": torch.tensor([[0, 3]])}, "out-of-range"),
        ({"teacher_topk_ids": torch.tensor([[1, 1]])}, "unique"),
        (
            {"teacher_topk_probs": torch.tensor([[0.4, 0.4]])},
            "sum to one",
        ),
        ({"teacher_tail_prob": torch.tensor([-0.1])}, "non-negative"),
        ({"student_temperature": 0.0}, "greater than zero"),
    ],
)
def test_rejects_malformed_inputs(overrides: dict[str, object], match: str) -> None:
    kwargs: dict[str, Any] = {
        "student_logits": torch.zeros((1, 3)),
        "teacher_topk_ids": torch.tensor([[0, 1]]),
        "teacher_topk_probs": torch.tensor([[0.4, 0.3]]),
        "teacher_tail_prob": torch.tensor([0.3]),
        "mask": torch.tensor([True]),
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        topk_plus_tail_forward_kl(**kwargs)


def test_rejects_nonzero_full_vocab_tail_and_zero_token_mask() -> None:
    common: dict[str, Any] = {
        "student_logits": torch.zeros((1, 2)),
        "teacher_topk_ids": torch.tensor([[0, 1]]),
        "teacher_topk_probs": torch.tensor([[0.4, 0.5]]),
        "teacher_tail_prob": torch.tensor([0.1]),
        "mask": torch.tensor([True]),
    }
    with pytest.raises(ValueError, match="tail probability must be zero"):
        topk_plus_tail_forward_kl(**common)

    common["teacher_topk_probs"] = torch.tensor([[0.4, 0.6]])
    common["teacher_tail_prob"] = torch.tensor([0.0])
    common["mask"] = torch.tensor([False])
    with pytest.raises(ValueError, match="selects zero tokens"):
        topk_plus_tail_forward_kl(**common)


def test_logical_vocab_ignores_padded_logits() -> None:
    logits = torch.tensor([[0.0, 1.0, -2.0, torch.nan]], dtype=torch.float64)
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[1, 0, 2]]),
        teacher_topk_probs=torch.tensor([[0.2, 0.3, 0.5]]),
        teacher_tail_prob=torch.tensor([0.0]),
        mask=torch.tensor([True]),
        logical_vocab_size=3,
    )
    reference = _dense_forward_kl(logits[..., :3], torch.tensor([[0.3, 0.2, 0.5]]))
    torch.testing.assert_close(result.loss_sum, reference.sum())


def test_mask_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="boolean dtype"):
        topk_plus_tail_forward_kl(
            torch.zeros((1, 2)),
            teacher_topk_ids=torch.tensor([[0]]),
            teacher_topk_probs=torch.tensor([[0.5]]),
            teacher_tail_prob=torch.tensor([0.5]),
            mask=torch.tensor([1]),
        )


def test_masked_rows_may_contain_arbitrary_sentinel_targets() -> None:
    logits = torch.tensor([[0.2, 0.5, -0.1], [3.0, -2.0, 1.0]])
    result = topk_plus_tail_forward_kl(
        logits,
        teacher_topk_ids=torch.tensor([[1, 0], [-100, 10_000]]),
        teacher_topk_probs=torch.tensor([[0.5, 0.2], [float("nan"), -4.0]]),
        teacher_tail_prob=torch.tensor([0.3, float("nan")]),
        mask=torch.tensor([True, False]),
        weights=torch.tensor([1.0, float("nan")]),
    )
    expected = topk_plus_tail_forward_kl(
        logits[:1],
        teacher_topk_ids=torch.tensor([[1, 0]]),
        teacher_topk_probs=torch.tensor([[0.5, 0.2]]),
        teacher_tail_prob=torch.tensor([0.3]),
        mask=torch.tensor([True]),
    )

    torch.testing.assert_close(result.loss_sum, expected.loss_sum)
    assert result.token_count.item() == 1
    assert result.per_token_weighted_loss[1].item() == 0
