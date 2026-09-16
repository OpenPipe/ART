from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from art.megatron.prefix_tree_packing import prefix_tree_pack
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


@pytest.fixture
def local_head(monkeypatch):
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


@pytest.mark.parametrize("top_k", (None, 3, 12))
@pytest.mark.parametrize("include_logits", (False, True))
@pytest.mark.parametrize("multi_target", (False, True))
def test_recomputed_head_preserves_outputs_and_arbitrary_loss_gradients(
    local_head, top_k, include_logits, multi_target
):
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


@pytest.mark.parametrize("cp_size", (1, 2, 4))
@pytest.mark.parametrize("fields", ("hidden", "targets", "all"))
def test_output_positions_align_sharded_fields_masks_and_gradients(
    local_head, cp_size, fields
):
    tokens = [torch.tensor(row) for row in ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 8, 9])]
    packed = prefix_tree_pack(tokens, max_depth=1)
    generator = torch.Generator().manual_seed(911)
    model = SimpleNamespace(
        output_layer=_Head(torch.randn(11, 5, generator=generator)),
        vocab_size=11,
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda value: value,
    )
    trainer = object.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(model=[model])
    hidden = torch.randn(packed.tokens.numel(), 5, generator=generator).requires_grad_()
    requests = [
        ForwardInput(
            input_tokens=row,
            target_tokens=(
                torch.stack((row, row.roll(1)), dim=1) if fields != "hidden" else None
            ),
            hidden_states=fields != "targets",
            logits=fields == "all",
            top_k=3 if fields == "all" else None,
        )
        for row in tokens
    ]
    items = [trainer._forward_item(request) for request in requests]
    full = trainer._project_head(
        items,
        SimpleNamespace(
            positions_by_item=packed.positions_by_sequence,
            source_positions_by_item=tuple(torch.arange(len(row)) for row in tokens),
        ),
        hidden,
    )
    for output, row in zip(full, tokens, strict=True):
        torch.testing.assert_close(output.positions, torch.arange(len(row)))

    # Uneven, noncontiguous ownership; the last CP4 rank owns no source rows.
    owners = torch.tensor([0, 1, 1, 2, 0, 1, 1, 2, 0]) % cp_size
    seen = [[] for _ in items]
    local_loss = hidden.sum() * 0
    full_loss = hidden.sum() * 0
    for output, row in zip(full, tokens, strict=True):
        values = output.hidden_states if fields == "hidden" else output.target_logprobs
        full_loss = full_loss + values[row % 2 == 1].square().sum()
    for cp_rank in range(cp_size):
        rows = torch.nonzero(owners == cp_rank).flatten().flip(0)
        # Include a padding row that must never appear in public positions.
        dispatched = torch.cat((rows, torch.tensor([-1])))
        pairs = [
            _impl._local_position_pairs(dispatched[None], positions)
            for positions in packed.positions_by_sequence
        ]
        local_hidden = torch.cat((hidden[rows], hidden.new_zeros((1, 5))))
        outputs = trainer._project_head(
            items,
            SimpleNamespace(
                positions_by_item=tuple(pair[0] for pair in pairs),
                source_positions_by_item=tuple(pair[1] for pair in pairs),
            ),
            local_hidden,
        )
        for index, (output, reference, row) in enumerate(
            zip(outputs, full, tokens, strict=True)
        ):
            positions = output.positions
            assert positions is not None and positions.dtype == torch.long
            assert positions.device == local_hidden.device
            assert not positions.requires_grad
            torch.testing.assert_close(
                row[positions], packed.tokens.flatten()[dispatched[pairs[index][0]]]
            )
            seen[index].extend(positions.tolist())
            for field in ("hidden_states", "target_logprobs", "logits"):
                actual, expected = getattr(output, field), getattr(reference, field)
                if expected is not None:
                    torch.testing.assert_close(actual, expected[positions])
            if output.top_k is not None:
                torch.testing.assert_close(
                    output.top_k.tokens, reference.top_k.tokens[positions]
                )
                torch.testing.assert_close(
                    output.top_k.logprobs, reference.top_k.logprobs[positions]
                )
            mask = (row % 2 == 1).index_select(0, positions)
            values = (
                output.hidden_states if fields == "hidden" else output.target_logprobs
            )
            local_loss = local_loss + values[mask].square().sum()
    assert [sorted(positions) for positions in seen] == [
        list(range(len(row))) for row in tokens
    ]
    torch.testing.assert_close(local_loss, full_loss)
    parameters = (
        (hidden, model.output_layer.weight) if fields != "hidden" else (hidden,)
    )
    torch.testing.assert_close(
        torch.autograd.grad(local_loss, parameters),
        torch.autograd.grad(full_loss, parameters),
    )
