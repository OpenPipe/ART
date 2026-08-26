from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F

from art.loss import AlignedLossInputs, loss_fn
from art.megatron.context_parallel.runtime import (
    dispatch_megatron_context_parallel_training_tensors,
    prepare_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.megatron.selective_lm_head import (
    LmHeadTokenSelection,
    forward_token_logits,
    forward_token_losses,
)
from art.preprocessing.pack import PackedTensors


def _loss_and_grads(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = F.cross_entropy(
        hidden @ weight.transpose(0, 1),
        labels,
        ignore_index=-100,
        reduction="sum",
    )
    loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


class _ToyOutput(torch.nn.Linear):
    sequence_parallel = False
    disable_grad_reduce = False
    gather_output = False
    tp_group = None


class _ToyLanguageModel(torch.nn.Module):
    post_process = True
    mtp_process = False
    config = SimpleNamespace(mtp_num_layers=0)

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.output_layer = _ToyOutput(hidden_size, vocab_size)

    def compute_language_model_loss(
        self, labels: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = self.output_layer(hidden_states)
        return logits if labels is None else self.compute_language_model_loss(labels, logits)


class _ToyGptOutput(_ToyLanguageModel):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels is not None:
            raise ValueError("test only exercises logits")
        return self.output_layer(hidden_states).transpose(0, 1).contiguous()


def test_forward_token_logits_uses_metadata_only_singleton_reshape() -> None:
    model = _ToyGptOutput(13, 19)
    hidden = torch.randn(17, 1, 13, requires_grad=True)
    selection = LmHeadTokenSelection.from_mask(torch.ones(1, 17, dtype=torch.bool))
    logits = forward_token_logits(
        model,
        selection=selection,
        forward_kwargs={"hidden_states": hidden},
    )
    assert tuple(logits.shape) == (selection.projected_row_count, 19)
    assert type(logits.grad_fn).__name__.startswith("ViewBackward")
    logits.sum().backward()
    assert hidden.grad is not None and hidden.grad.count_nonzero()


@pytest.mark.parametrize(
    ("name", "labels", "logical_rows", "padding_rows"),
    (
        ("all_trainable_nonaligned", torch.arange(17) % 11, 17, 15),
        (
            "ignored_row",
            torch.tensor([*(value % 11 for value in range(15)), -100]),
            15,
            1,
        ),
        ("empty", torch.full((9,), -100, dtype=torch.long), 0, 16),
        ("aligned", torch.arange(16) % 11, 16, 0),
    ),
)
def test_fp32_forward_gradient_parity(
    name: str,
    labels: torch.Tensor,
    logical_rows: int,
    padding_rows: int,
) -> None:
    del name
    torch.manual_seed(7)
    hidden = torch.randn(labels.numel(), 13, dtype=torch.float32)
    weight = torch.randn(19, 13, dtype=torch.float32)

    dense_hidden = hidden.clone().requires_grad_()
    dense_weight = weight.clone().requires_grad_()
    dense = _loss_and_grads(dense_hidden, dense_weight, labels)

    selection = LmHeadTokenSelection.from_labels(labels.unsqueeze(0))
    selected_hidden_source = hidden.clone().reshape(1, -1, hidden.shape[-1])
    selected_hidden_source.requires_grad_()
    selected_weight = weight.clone().requires_grad_()
    selected_hidden = selection.select_rows(selected_hidden_source)
    selected_hidden.retain_grad()
    selected_labels = selection.select_labels(labels.unsqueeze(0)).reshape(-1)
    selected = _loss_and_grads(selected_hidden, selected_weight, selected_labels)

    assert selection.logical_row_count == logical_rows
    assert selection.alignment_padding_rows == padding_rows
    assert selection.projected_row_count % 16 == 0
    torch.testing.assert_close(selected[0], dense[0], rtol=0, atol=0)
    torch.testing.assert_close(
        selected_hidden_source.grad.reshape_as(hidden), dense[1], rtol=0, atol=0
    )
    torch.testing.assert_close(selected[2], dense[2], rtol=0, atol=0)
    if logical_rows:
        assert dense[0].abs().item() > 0
        assert dense[1].abs().sum().item() > 0
        assert dense[2].abs().sum().item() > 0
    else:
        assert dense[0].item() == 0
        assert dense[1].count_nonzero().item() == 0
        assert dense[2].count_nonzero().item() == 0
    if padding_rows:
        assert torch.all(selected_labels[-padding_rows:] == -100)
        assert selected_hidden[-padding_rows:].count_nonzero().item() == 0
        assert selected_hidden.grad[-padding_rows:].count_nonzero().item() == 0
        restored = selection.restore(torch.arange(selection.projected_row_count))
        assert not torch.any(
            restored.reshape(-1).index_select(0, selection.flat_indices)
            >= logical_rows
        )

    targets = labels.reshape(1, -1, 1).expand(-1, -1, 2)
    weights = torch.ones_like(targets, dtype=torch.float32)
    selected_targets = selection.select_rows(targets, padding_value=-1)
    selected_weights = selection.select_rows(weights)
    if padding_rows:
        assert torch.all(selected_targets[-padding_rows:] == -1)
        assert selected_weights[-padding_rows:].count_nonzero().item() == 0


@pytest.mark.parametrize(
    "labels",
    (
        torch.arange(17) % 11,
        torch.tensor([*(value % 11 for value in range(15)), -100]),
        torch.full((9,), -100, dtype=torch.long),
        torch.arange(16) % 11,
    ),
)
def test_forward_token_losses_fp32_parity(labels: torch.Tensor) -> None:
    torch.manual_seed(11)
    dense_model = _ToyLanguageModel(13, 19)
    selected_model = _ToyLanguageModel(13, 19)
    selected_model.load_state_dict(dense_model.state_dict())
    dense_hidden = torch.randn(labels.numel(), 1, 13, requires_grad=True)
    selected_hidden = dense_hidden.detach().clone().requires_grad_()
    labels = labels.unsqueeze(0)

    dense_output = forward_token_losses(
        dense_model,
        labels=labels,
        selection=LmHeadTokenSelection.from_labels(labels),
        forward_kwargs={"hidden_states": dense_hidden},
        enabled=False,
    )
    selection = LmHeadTokenSelection.from_labels(labels)
    selected_output = forward_token_losses(
        selected_model,
        labels=labels,
        selection=selection,
        forward_kwargs={"hidden_states": selected_hidden},
        enabled=True,
    )
    dense_output.token_losses.sum().backward()
    selected_output.token_losses.sum().backward()

    torch.testing.assert_close(
        selected_output.restore(selected_output.token_losses),
        dense_output.token_losses,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(selected_hidden.grad, dense_hidden.grad, rtol=0, atol=0)
    for dense_parameter, selected_parameter in zip(
        dense_model.parameters(), selected_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            selected_parameter.grad, dense_parameter.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "labels",
    (
        torch.arange(17) % 11,
        torch.tensor([*(value % 11 for value in range(15)), -100]),
        torch.full((9,), -100, dtype=torch.long),
        torch.arange(16) % 11,
    ),
)
def test_aligned_rl_loss_fp32_gradient_parity(labels: torch.Tensor) -> None:
    torch.manual_seed(17)
    labels = labels.unsqueeze(0)
    active = labels != -100
    shape = labels.shape
    dense_new = torch.randn(shape, requires_grad=True)
    selected_new_source = dense_new.detach().clone().requires_grad_()
    inputs = AlignedLossInputs(
        assistant_mask=active,
        old_logprobs=torch.randn(shape),
        advantages=torch.randn(shape),
        weights=torch.ones(shape),
        group_ids=torch.ones(shape, dtype=torch.long),
    )
    config = cast(
        Any,
        {
            "importance_sampling_level": "sequence",
            "truncated_importance_sampling": 1.4,
        },
    )
    dense_loss = loss_fn(
        inputs,
        new_logprobs=dense_new,
        ref_logprobs=None,
        entropies=None,
        experimental_config=config,
        reduction="sum",
    ).policy_loss
    selection = LmHeadTokenSelection.from_labels(labels)
    selected_loss = loss_fn(
        selection.compact_loss_inputs(inputs),
        new_logprobs=selection.select(selected_new_source),
        ref_logprobs=None,
        entropies=None,
        experimental_config=config,
        reduction="sum",
    ).policy_loss
    dense_loss.backward()
    selected_loss.backward()
    torch.testing.assert_close(selected_loss, dense_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        selected_new_source.grad, dense_new.grad, rtol=0, atol=0
    )


def _cp_micro(sequence_length: int = 34) -> PackedTensors:
    shape = (1, sequence_length)
    return cast(
        PackedTensors,
        {
            "tokens": torch.arange(sequence_length).reshape(shape),
            "group_ids": torch.ones(shape, dtype=torch.long),
            "parent_ids": torch.ones(shape, dtype=torch.long),
            "input_pos": torch.arange(sequence_length).reshape(shape),
            "assistant_mask": torch.ones(shape, dtype=torch.bool),
            "logprobs": torch.zeros(shape),
            "advantages": torch.ones(shape),
            "weights": torch.ones(shape),
            "pixel_values": [None],
            "image_grid_thw": [None],
            "moe_routing_replay": None,
        },
    )


def test_cp_rank_all_trainable_nonaligned_dispatch() -> None:
    micro = _cp_micro()
    topology = ParallelTopology(cp=2)
    config = ContextParallelConfig(block_size=1)
    prepared = []
    for cp_rank in range(2):
        _state, rank_plan, spec, pad_multiple = (
            prepare_megatron_context_parallel_state(
                micro=micro,
                topology=topology,
                config=config,
                cp_group=None,
                cp_rank=cp_rank,
                target_device=torch.device("cpu"),
            )
        )
        tensors, workload = dispatch_megatron_context_parallel_training_tensors(
            micro=micro,
            rank_plan=rank_plan,
            spec=spec,
            pad_multiple=pad_multiple,
            target_device=torch.device("cpu"),
        )
        prepared.append((tensors, workload))

    tensors, workload = next(
        (tensors, workload)
        for tensors, workload in prepared
        if bool(torch.all(tensors.labels != -100))
        and tensors.labels.numel() % 16 != 0
    )
    selection = tensors.lm_head_selection
    assert selection.logical_row_count == tensors.labels.numel()
    assert selection.alignment_padding_rows == -tensors.labels.numel() % 16
    assert workload.loss_bearing_tokens == selection.logical_row_count
    assert selection.select_labels(tensors.labels)[
        :, selection.logical_row_count :
    ].eq(-100).all()
    assert selection.select(tensors.weights)[
        :, selection.logical_row_count :
    ].count_nonzero() == 0


def test_zero_length_selection_stays_zero_length() -> None:
    selection = LmHeadTokenSelection.from_labels(torch.empty((1, 0), dtype=torch.long))
    assert selection.logical_row_count == 0
    assert selection.alignment_padding_rows == 0
    assert selection.projected_row_count == 0
