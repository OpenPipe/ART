"""The v1 method names run the restored TrainerRank methods unchanged."""

from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank


@pytest.mark.parametrize(
    "alias,method",
    [
        ("forward_batches", "forward_micro_batches"),
        ("forward", "dp_rank_forward"),
        ("reduce", "dp_reduce"),
    ],
)
def test_v1_names_are_the_restored_methods(alias, method):
    assert getattr(TrainerRank, alias) is getattr(TrainerRank, method)


@pytest.mark.parametrize("options", [None, object()])
@pytest.mark.parametrize("method", ["forward_batches", "forward", "reduce"])
def test_forward_options_are_rejected_not_ignored(method, options):
    rank = TrainerRank.__new__(TrainerRank)
    argument = (
        torch.zeros(1)
        if method == "reduce"
        else [ForwardInput(input_tokens=torch.tensor([1]))]
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'options'"):
        getattr(rank, method)(argument, options=options)


@pytest.mark.parametrize("options", [None, object()])
def test_constructor_options_are_rejected_not_ignored(options):
    with pytest.raises(TypeError, match="unexpected keyword argument 'options'"):
        TrainerRank(cast(Any, None), options=options)  # ty: ignore[unknown-argument]


def _losses(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([1.0, 2.0, 3.0])
    return (weight * inputs).sum(), weight**2 * inputs


def test_backward_matches_loss_backward():
    expected = torch.nn.Parameter(torch.tensor(0.5))
    actual = torch.nn.Parameter(torch.tensor(0.5))
    _losses(expected)[0].backward()
    TrainerRank.__new__(TrainerRank).backward(_losses(actual)[0])
    assert expected.grad is not None and actual.grad is not None
    assert torch.equal(actual.grad, expected.grad)


def test_backward_takes_sequences_gradients_and_retain_graph():
    expected = torch.nn.Parameter(torch.tensor(0.5))
    actual = torch.nn.Parameter(torch.tensor(0.5))
    gradients = (None, torch.tensor([2.0, -1.0, 0.5]))
    scalar, vector = _losses(expected)
    vector.backward(gradients[1], retain_graph=True)
    scalar.backward()
    vector.backward(gradients[1])
    rank = TrainerRank.__new__(TrainerRank)
    losses = _losses(actual)
    rank.backward(losses, gradients, retain_graph=True)
    rank.backward(losses[1], gradients[1])
    assert expected.grad is not None and actual.grad is not None
    assert torch.equal(actual.grad, expected.grad)
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        rank.backward(losses[1], gradients[1])
