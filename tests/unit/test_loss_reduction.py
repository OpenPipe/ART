"""Regression tests for length-aware policy-loss reductions."""

import pytest
import torch

from art._backend_training import build_rl_train_configs
from art.loss import LossInputs, loss_fn


def _inputs(
    assistant_mask: list[list[int]],
    *,
    group_ids: list[list[int]] | None = None,
) -> dict[str, torch.Tensor]:
    mask = torch.tensor(assistant_mask, dtype=torch.bool)
    batch_size, sequence_length = mask.shape
    if group_ids is None:
        group_ids = [[row + 1] * sequence_length for row in range(batch_size)]
    return {
        "tokens": torch.zeros(batch_size, sequence_length, dtype=torch.long),
        "assistant_mask": mask,
        "logprobs": torch.zeros(batch_size, sequence_length),
        "advantages": torch.ones(batch_size, sequence_length),
        "weights": torch.ones(batch_size, sequence_length),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "parent_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long),
    }


def test_dr_grpo_uses_fixed_denominator_for_unequal_completions() -> None:
    """A longer response must not receive a larger normalization weight."""
    inputs = _inputs(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
        ]
    )
    new_logprobs = torch.full((2, 6), -1.0)

    bnpo = loss_fn(
        LossInputs(inputs=inputs), new_logprobs, None, None, {}, loss_type="bnpo"
    )
    dr_grpo = loss_fn(
        LossInputs(inputs=inputs),
        new_logprobs,
        None,
        None,
        {},
        loss_type="dr_grpo",
        max_completion_length=6,
    )

    # The shifted mask contains 2 + 5 active tokens.  BNPO divides by seven,
    # while Dr. GRPO divides by B * max_completion_length = 12.
    token_loss = torch.exp(torch.tensor(-1.0))
    assert bnpo.policy_loss == pytest.approx(float(token_loss))
    assert dr_grpo.policy_loss == pytest.approx(float(token_loss * 7 / 12))
    assert dr_grpo.normalization_denominator is not None
    assert dr_grpo.normalization_denominator.item() == pytest.approx(12.0)


def test_grpo_averages_completion_means_and_handles_empty_row() -> None:
    inputs = _inputs(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    new_logprobs = torch.full((3, 5), -1.0)
    result = loss_fn(
        LossInputs(inputs=inputs),
        new_logprobs,
        None,
        None,
        {},
        loss_type="grpo",
    )

    token_loss = torch.exp(torch.tensor(-1.0))
    # The two non-empty completions each have the same per-token mean; the
    # empty row contributes zero, as in TRL's per-sequence reduction.
    assert result.policy_loss == pytest.approx(float(token_loss * 2 / 3))


def test_dr_grpo_counts_multiple_prefix_tree_tails_in_one_row() -> None:
    """Packed rows use segment IDs to recover the logical completion count."""
    inputs = _inputs(
        [[0, 1, 1, 0, 1, 1]],
        group_ids=[[10, 10, 10, 20, 20, 20]],
    )
    result = loss_fn(
        LossInputs(inputs=inputs),
        torch.full((1, 6), -1.0),
        None,
        None,
        {},
        loss_type="dr_grpo",
        max_completion_length=4,
    )

    # After causal alignment, two independent tails remain (one token each
    # in this compact fixture), so the denominator is 2 * 4 rather than 1 * 4.
    assert result.normalization_denominator is not None
    assert result.normalization_denominator.item() == pytest.approx(8.0)


@pytest.mark.parametrize("max_completion_length", [0, -1])
def test_dr_grpo_requires_positive_max_completion_length(
    max_completion_length: int,
) -> None:
    inputs = _inputs([[0, 1, 0]])
    with pytest.raises(ValueError, match="max_completion_length"):
        loss_fn(
            LossInputs(inputs=inputs),
            torch.zeros(1, 3),
            None,
            None,
            {},
            loss_type="dr_grpo",
            max_completion_length=max_completion_length,
        )


def test_sum_reduction_remains_raw_but_exposes_matching_denominator() -> None:
    inputs = _inputs([[0, 1, 1, 1], [0, 1, 0, 0]])
    result = loss_fn(
        LossInputs(inputs=inputs),
        torch.full((2, 4), -1.0),
        None,
        None,
        {},
        reduction="sum",
        loss_type="dr_grpo",
        max_completion_length=4,
    )

    assert result.policy_loss == result.policy_loss_sum
    assert result.normalization_denominator is not None
    assert result.normalization_denominator.item() == pytest.approx(8.0)


def test_zero_token_batch_is_finite_for_fixed_reduction() -> None:
    inputs = _inputs([[0, 0, 0], [0, 0, 0]])
    result = loss_fn(
        LossInputs(inputs=inputs),
        torch.zeros(2, 3),
        None,
        None,
        {},
        loss_type="dr_grpo",
        max_completion_length=5,
    )
    assert torch.isfinite(result.policy_loss)
    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.normalization_denominator is not None
    assert result.normalization_denominator.item() == pytest.approx(10.0)


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
def test_dummy_microbatch_contributes_no_normalization(loss_type: str) -> None:
    """Padding work must not change a distributed step's denominator."""
    result = loss_fn(
        LossInputs(inputs=_inputs([[0, 0, 0]]), is_dummy=True),
        torch.zeros(1, 3),
        None,
        None,
        {},
        reduction="sum",
        loss_type=loss_type,  # type: ignore[arg-type]
        max_completion_length=5,
    )
    assert result.normalization_denominator is not None
    assert result.normalization_denominator.item() == pytest.approx(0.0)


def test_dr_grpo_matches_trl_default_completion_length() -> None:
    result = loss_fn(
        LossInputs(inputs=_inputs([[0, 1, 1]])),
        torch.full((1, 3), -1.0),
        None,
        None,
        {},
        loss_type="dr_grpo",
    )
    assert result.normalization_denominator is not None
    assert result.normalization_denominator.item() == pytest.approx(256.0)


def test_context_parallel_owner_denominator_is_not_replicated() -> None:
    """CP ranks contribute a fixed completion denominator exactly once."""
    inputs = _inputs([[0, 1, 1, 1]])
    aligned = LossInputs(inputs=inputs).align_inputs()
    owner = aligned.model_copy(
        update={"normalization_denominator_override": torch.tensor(2)}
    )
    peer = aligned.model_copy(
        update={"normalization_denominator_override": torch.tensor(0)}
    )
    owner_loss = loss_fn(
        owner,
        torch.full((1, 4), -1.0),
        None,
        None,
        {},
        reduction="sum",
        loss_type="dr_grpo",
        max_completion_length=8,
    )
    peer_loss = loss_fn(
        peer,
        torch.full((1, 4), -1.0),
        None,
        None,
        {},
        reduction="sum",
        loss_type="dr_grpo",
        max_completion_length=8,
    )
    assert owner_loss.normalization_denominator is not None
    assert peer_loss.normalization_denominator is not None
    assert owner_loss.normalization_denominator.item() == pytest.approx(16.0)
    assert peer_loss.normalization_denominator.item() == pytest.approx(0.0)


def test_backend_config_threads_loss_reduction_options() -> None:
    _, config = build_rl_train_configs(
        learning_rate=1e-5,
        loss_type="dr_grpo",
        max_completion_length=128,
    )
    assert config["loss_type"] == "dr_grpo"
    assert config["max_completion_length"] == 128
