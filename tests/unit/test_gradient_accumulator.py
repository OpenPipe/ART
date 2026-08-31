import pytest
import torch

from art.megatron.training.gradient_accumulator import GradientAccumulator


class _Chunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.weight.main_grad = torch.zeros_like(self.weight)

    def zero_grad_buffer(self) -> None:
        self.weight.main_grad.zero_()
        self.weight.grad = None


def test_accumulator_keeps_one_gradient_image_and_exact_order() -> None:
    chunk = _Chunk()
    accumulator = GradientAccumulator([chunk])

    chunk.weight.main_grad.copy_(torch.tensor([1.0, 2.0]))
    accumulator.record("fb-a", torch.tensor(2), expected_global_token_count=2)
    accumulator.before_forward_backward()
    chunk.zero_grad_buffer()

    chunk.weight.main_grad.copy_(torch.tensor([3.0, 5.0]))
    accumulator.record("fb-b", torch.tensor(3), expected_global_token_count=3)
    assert len(accumulator.residency_tensors()) == 2

    with pytest.raises(RuntimeError, match="contribution order"):
        accumulator.seal(("fb-b", "fb-a"))

    accumulator.seal(("fb-a", "fb-b"))
    prepared = accumulator.prepare_optimizer()
    assert prepared.expected_global_token_count == 5
    assert prepared.local_token_count.item() == 5
    torch.testing.assert_close(prepared.gradients[0], torch.tensor([4.0, 7.0]))
    assert accumulator.consume() == ("fb-a", "fb-b")
    assert accumulator.contribution_ids == ()


def test_accumulator_flushes_direct_parameter_gradients() -> None:
    chunk = _Chunk()

    def flush(chunks: list[torch.nn.Module]) -> None:
        for module in chunks:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter.main_grad.add_(parameter.grad)
                    parameter.grad = None

    accumulator = GradientAccumulator([chunk], flush_gradients=flush)
    chunk.weight.grad = torch.tensor([2.0, 4.0])
    accumulator.record("fb", torch.tensor(2), expected_global_token_count=2)
    accumulator.seal(("fb",))

    prepared = accumulator.prepare_optimizer()
    assert prepared.gradients[0] is chunk.weight.main_grad
    torch.testing.assert_close(prepared.gradients[0], torch.tensor([2.0, 4.0]))
    assert chunk.weight.grad is None


def test_accumulator_requires_bounded_provenance() -> None:
    chunk = _Chunk()
    bounded = GradientAccumulator([chunk], max_contributions=1)
    bounded.record("fb-a", torch.tensor(1))
    bounded.before_forward_backward()
    chunk.zero_grad_buffer()

    with pytest.raises(RuntimeError, match="limit"):
        bounded.record("fb-b", torch.tensor(1))

    unchecked = GradientAccumulator([chunk])
    unchecked.record("fb", torch.tensor(1))
    unchecked.seal(("fb",))
    with pytest.raises(RuntimeError, match="global token provenance"):
        unchecked.prepare_optimizer()
