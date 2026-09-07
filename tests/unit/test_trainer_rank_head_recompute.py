from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank, _impl


class _Head(torch.nn.Module):
    sequence_parallel = False

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def forward(self, hidden, *, weight=None, runtime_gather_output=None):
        return hidden @ (self.weight if weight is None else weight).T, None


def _direct(function, *args, use_reentrant=False, **kwargs):
    return function(*args, **kwargs)


@pytest.mark.parametrize("top_k", (None, 3, 12))
@pytest.mark.parametrize("include_logits", (False, True))
@pytest.mark.parametrize("multi_target", (False, True))
def test_recomputed_head_preserves_outputs_and_arbitrary_loss_gradients(
    monkeypatch, top_k, include_logits, multi_target
):
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 16)
    monkeypatch.setattr(_impl, "_language_model", lambda model: model)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda value: value)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda value: value)
    monkeypatch.setattr(_impl, "_vocab_range", lambda logits: (0, logits.shape[-1]))
    monkeypatch.setattr(
        _impl,
        "_vocab_parallel_topk_from_local",
        lambda values, tokens, *, k, log_z, vocab_start: _impl.TopK(
            values[:, :k] - log_z[:, None], tokens[:, :k]
        ),
    )
    monkeypatch.setattr(
        TrainerRank, "_gather_tensor_parallel_logits", lambda self, value: value
    )
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(65, 7, generator=generator) / 3
    hidden = torch.randn(41, 7, generator=generator)
    positions = (torch.arange(25), torch.arange(10, 41))
    requests = []
    for rows in positions:
        shape = (len(rows), 2) if multi_target else (len(rows),)
        labels = torch.randint(0, 65, shape, generator=generator)
        labels[::7] = -100
        requests.append(
            ForwardInput(
                input_tokens=torch.ones(len(rows), dtype=torch.long),
                target_tokens=labels,
                top_k=top_k,
                logits=include_logits,
                hidden_states=True,
            )
        )

    def run(recompute):
        model = SimpleNamespace(
            output_layer=_Head(weight),
            vocab_size=65,
            share_embeddings_and_output_weights=False,
            _scale_logits=lambda value: value,
        )
        trainer = object.__new__(TrainerRank)
        trainer.runtime = SimpleNamespace(model=[model])
        value = hidden.clone().requires_grad_()
        saved = []

        def pack(tensor):
            saved.append(
                (
                    tuple(tensor.shape),
                    tensor.untyped_storage().data_ptr(),
                    tensor.untyped_storage().nbytes(),
                )
            )
            return tensor

        with (
            nullcontext()
            if recompute
            else patch("torch.utils.checkpoint.checkpoint", _direct)
        ):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs = trainer._project_head(
                    [trainer._forward_item(request) for request in requests],
                    SimpleNamespace(
                        positions_by_item=positions,
                        source_positions_by_item=tuple(
                            torch.arange(len(rows)) for rows in positions
                        ),
                    ),
                    value,
                )
            tensors = [output.target_logprobs for output in outputs]
            tensors += [output.hidden_states for output in outputs]
            tensors += [
                output.top_k.logprobs for output in outputs if output.top_k is not None
            ]
            tensors += [
                output.logits for output in outputs if output.logits is not None
            ]
            loss = sum(
                tensor.sin().sum() + tensor.square().mean() for tensor in tensors
            )
            loss += tensors[0].mean() * tensors[1].mean()
            loss.backward()
        return (
            [tensor.detach() for tensor in tensors],
            [output.top_k.tokens for output in outputs if output.top_k is not None],
            value.grad,
            model.output_layer.weight.grad,
            saved,
        )

    baseline = run(False)
    recomputed = run(True)
    for actual, expected in zip(recomputed[:4], baseline[:4], strict=True):
        torch.testing.assert_close(actual, expected)
    # Full-vocabulary tensors saved by normalization must disappear. Explicit
    # logits outputs remain part of the public result and are outside this claim.
    assert any(shape[-1:] == (65,) and len(shape) == 2 for shape, _, _ in baseline[4])
    assert not any(
        shape[-1:] == (65,) and len(shape) == 2 for shape, _, _ in recomputed[4]
    )
