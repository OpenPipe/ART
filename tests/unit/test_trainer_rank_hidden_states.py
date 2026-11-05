"""Hidden outputs preserve the final activation and its adapter gradient path."""

from types import SimpleNamespace

import pytest
from test_trainer_rank_head_recompute import _Head, _patch_local_head
import torch

from art.trainer_rank import ForwardInput, TrainerRank


@pytest.mark.parametrize("dtype", (torch.float64, torch.bfloat16))
@pytest.mark.parametrize("mode", ("hidden", "target_hidden", "both_losses"))
def test_hidden_head_matches_direct_adapter_gradients(monkeypatch, dtype, mode):
    _patch_local_head(monkeypatch)
    generator = torch.Generator().manual_seed(721)
    features = torch.randn(11, 5, generator=generator).to(dtype)
    base = torch.randn(11, 7, generator=generator).to(dtype)
    initial_a = torch.randn(5, 1, generator=generator).to(dtype) / 3
    initial_b = torch.randn(1, 7, generator=generator).to(dtype) / 3
    weight = torch.randn(19, 7, generator=generator).to(dtype) / 3
    probe = torch.randn(7, generator=generator)
    # Two logical sequences share their prefix in the packed decoder output.
    positions = (torch.arange(8), torch.tensor([0, 1, 2, 3, 8, 9, 10]))
    labels = tuple(torch.arange(len(rows)) % 19 for rows in positions)
    labels[0][2] = -100
    requests = [
        ForwardInput(
            input_tokens=torch.ones(len(rows), dtype=torch.long),
            target_tokens=targets if mode != "hidden" else None,
            hidden_states=True,
        )
        for rows, targets in zip(positions, labels, strict=True)
    ]

    def run(direct):
        adapter_a = initial_a.clone().requires_grad_()
        adapter_b = initial_b.clone().requires_grad_()
        hidden = base + (features @ adapter_a) @ adapter_b
        head = _Head(weight)
        head.weight.requires_grad_(False)
        calls = []
        head.register_forward_hook(lambda *args: calls.append(True))
        trainer = object.__new__(TrainerRank)
        trainer.runtime = SimpleNamespace(
            model=[
                SimpleNamespace(
                    output_layer=head,
                    vocab_size=19,
                    share_embeddings_and_output_weights=False,
                    _scale_logits=lambda value: value,
                )
            ]
        )
        if direct:
            outputs = [hidden.index_select(0, rows) for rows in positions]
            logprobs = []
            for output, targets in zip(outputs, labels, strict=True):
                valid = targets != -100
                logits = (output @ weight.T).float()
                logprobs.append(
                    torch.where(
                        valid,
                        logits.log_softmax(-1).gather(1, targets.clamp_min(0)[:, None])[
                            :, 0
                        ],
                        0.0,
                    )
                )
        else:
            projected = trainer._project_head(
                [trainer._forward_item(request) for request in requests],
                SimpleNamespace(
                    positions_by_item=positions,
                    source_positions_by_item=tuple(
                        torch.arange(len(p)) for p in positions
                    ),
                ),
                hidden,
            )
            outputs = [output.hidden_states for output in projected]
            logprobs = [output.target_logprobs for output in projected]
            assert bool(calls) == (mode != "hidden")
            assert all(output.dtype == dtype for output in outputs)
            assert [tuple(output.shape) for output in outputs] == [(8, 7), (7, 7)]
            if mode == "hidden":
                assert logprobs == [None, None]
        loss = torch.stack(
            [(output.float() @ probe).square().mean() for output in outputs]
        ).sum()
        if mode == "both_losses":
            loss -= torch.stack([value.mean() for value in logprobs]).sum()
        loss.backward()
        assert head.weight.grad is None
        return [output.detach() for output in outputs], adapter_a.grad, adapter_b.grad

    actual, expected = run(False), run(True)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual[1:], expected[1:], strict=True):
        # The packed head adds shared-row gradients before the BF16 cast;
        # the direct reference adds the separately rounded logical gradients.
        atol = (
            2 * torch.finfo(dtype).eps * expected_grad.abs().max().item()
            if dtype == torch.bfloat16
            else 1e-7
        )
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            atol=atol,
            rtol=0.016 if dtype == torch.bfloat16 else 1e-5,
        )
    assert actual[1].abs().sum() > 0 and actual[2].abs().sum() > 0


def test_returning_hidden_does_not_change_target_projection(monkeypatch):
    _patch_local_head(monkeypatch)
    generator = torch.Generator().manual_seed(311)
    hidden = torch.randn(37, 7, generator=generator, requires_grad=True)
    head = _Head(torch.randn(19, 7, generator=generator))
    calls = []
    head.register_forward_hook(lambda module, args, output: calls.append(args[0].shape))
    trainer = object.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        model=[
            SimpleNamespace(
                output_layer=head,
                vocab_size=19,
                share_embeddings_and_output_weights=False,
                _scale_logits=lambda value: value,
            )
        ]
    )
    targets = torch.arange(37) % 19
    targets[::5] = -100
    prepared = SimpleNamespace(
        positions_by_item=(torch.arange(37),),
        source_positions_by_item=(torch.arange(37),),
    )
    projected = []
    for return_hidden in (False, True):
        request = ForwardInput(
            input_tokens=torch.ones(37, dtype=torch.long),
            target_tokens=targets,
            hidden_states=return_hidden,
        )
        calls.clear()
        [output] = trainer._project_head(
            [trainer._forward_item(request)], prepared, hidden
        )
        projected.append((output, tuple(calls)))
    (target_only, calls_without), (target_hidden, calls_with) = projected
    assert calls_without and calls_without == calls_with
    torch.testing.assert_close(
        target_only.target_logprobs, target_hidden.target_logprobs
    )
    assert target_only.hidden_states is None
    torch.testing.assert_close(target_hidden.hidden_states, hidden, rtol=0, atol=0)
    # Requesting the final activation does not change the target-loss backward.
    target_grad = torch.autograd.grad(target_only.target_logprobs.sum(), hidden)[0]
    both_grad = torch.autograd.grad(target_hidden.target_logprobs.sum(), hidden)[0]
    torch.testing.assert_close(target_grad, both_grad)
