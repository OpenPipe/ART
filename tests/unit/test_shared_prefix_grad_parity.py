from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(32, 8, dtype=torch.float64)
        self.position_embedding = nn.Embedding(8, 8, dtype=torch.float64)
        self.mix = nn.Linear(8, 8, bias=False, dtype=torch.float64)
        self.output = nn.Linear(8, 32, bias=False, dtype=torch.float64)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        states = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        context = causal_mask.to(states.dtype) @ states
        return self.output(torch.tanh(self.mix(context)))


@pytest.mark.parametrize("max_depth", (1, 2, 3))
@pytest.mark.parametrize("multi_target", (False, True))
def test_shared_prefix_ce_parameter_grads_match_independent_sequences(
    *,
    max_depth: int,
    multi_target: bool,
) -> None:
    input_ids = _input_ids()
    target_ids = tuple(
        _targets(tokens, multi_target=multi_target) for tokens in input_ids
    )
    pack = pack_shared_prefixes(input_ids, max_depth=max_depth)

    assert int(pack.tokens.numel()) < sum(len(row) for row in input_ids)

    torch.manual_seed(20260518)
    naive_model = _ToyCausalLM()
    packed_model = deepcopy(naive_model)

    naive_loss = torch.stack(
        [
            _sequence_ce_loss(naive_model, tokens, labels)
            for tokens, labels in zip(input_ids, target_ids, strict=True)
        ]
    ).sum()
    packed_loss = _packed_ce_loss(packed_model, pack, target_ids)

    torch.testing.assert_close(packed_loss, naive_loss, rtol=1e-12, atol=1e-12)
    naive_loss.backward()
    packed_loss.backward()

    for (name, naive_param), packed_param in zip(
        naive_model.named_parameters(),
        packed_model.parameters(),
        strict=True,
    ):
        assert naive_param.grad is not None, name
        assert packed_param.grad is not None, name
        torch.testing.assert_close(
            packed_param.grad,
            naive_param.grad,
            rtol=1e-10,
            atol=1e-10,
            msg=lambda msg, name=name: f"{name} grad mismatch:\n{msg}",
        )


@pytest.mark.parametrize("max_depth", (1, 2, 3))
def test_same_layout_mutation_preserves_forward_outputs(max_depth: int) -> None:
    pack = pack_shared_prefixes(_input_ids(), max_depth=max_depth)
    torch.manual_seed(20260518)
    model = _ToyCausalLM()
    logits = _packed_logits(model, pack)

    for positions in pack.positions_by_sequence:
        mutated_logits = _packed_logits(model, _mutated_pack(pack, keep=positions))
        torch.testing.assert_close(
            mutated_logits.index_select(0, positions),
            logits.index_select(0, positions),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("max_depth", (1, 2, 3))
@pytest.mark.parametrize("sequence_index", (0, 2, 4))
def test_same_layout_mutation_preserves_target_loss_grads(
    max_depth: int,
    sequence_index: int,
) -> None:
    input_ids = _input_ids()
    target_ids = tuple(_targets(tokens, multi_target=True) for tokens in input_ids)
    pack = pack_shared_prefixes(input_ids, max_depth=max_depth)
    mutated = _mutated_pack(pack, keep=pack.positions_by_sequence[sequence_index])

    torch.manual_seed(20260518)
    base_model = _ToyCausalLM()
    mutated_model = deepcopy(base_model)

    base_loss = _packed_sequence_ce_loss(base_model, pack, target_ids, sequence_index)
    mutated_loss = _packed_sequence_ce_loss(
        mutated_model,
        mutated,
        target_ids,
        sequence_index,
    )

    torch.testing.assert_close(mutated_loss, base_loss, rtol=0.0, atol=0.0)
    base_loss.backward()
    mutated_loss.backward()
    _assert_matching_grads(mutated_model, base_model)


def _input_ids() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3, 4, 6]),
        torch.tensor([1, 2, 3, 7]),
        torch.tensor([1, 2, 8]),
        torch.tensor([9, 10, 11]),
    )


def _targets(tokens: torch.Tensor, *, multi_target: bool) -> torch.Tensor:
    labels = (tokens * 3 + 5) % 31
    if not multi_target:
        return labels
    alternate = (tokens * 5 + 7) % 31
    stacked = torch.stack((labels, alternate), dim=1)
    if int(stacked.numel()) > 2:
        stacked[1, 1] = -100
    return stacked


def _sequence_ce_loss(
    model: _ToyCausalLM,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    seq_len = int(input_ids.numel())
    logits = model(
        input_ids,
        torch.arange(seq_len),
        torch.ones((seq_len, seq_len), dtype=torch.bool).tril(),
    )
    return _target_ce_loss(logits, target_ids)


def _packed_ce_loss(
    model: _ToyCausalLM,
    pack: SharedPrefixPack,
    target_ids: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    logits = _packed_logits(model, pack)
    losses = [
        _target_ce_loss(logits.index_select(0, positions), labels)
        for positions, labels in zip(
            pack.positions_by_sequence,
            target_ids,
            strict=True,
        )
    ]
    return torch.stack(losses).sum()


def _packed_sequence_ce_loss(
    model: _ToyCausalLM,
    pack: SharedPrefixPack,
    target_ids: tuple[torch.Tensor, ...],
    sequence_index: int,
) -> torch.Tensor:
    return _target_ce_loss(
        _packed_logits(model, pack).index_select(
            0,
            pack.positions_by_sequence[sequence_index],
        ),
        target_ids[sequence_index],
    )


def _packed_logits(model: _ToyCausalLM, pack: SharedPrefixPack) -> torch.Tensor:
    return model(
        pack.tokens.reshape(-1),
        pack.position_ids.reshape(-1),
        _shared_prefix_causal_mask(pack),
    )


def _target_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    expanded = logits.unsqueeze(1).expand(-1, int(labels.shape[1]), -1)
    return F.cross_entropy(
        expanded.reshape(-1, int(logits.shape[-1])),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _mutated_pack(pack: SharedPrefixPack, *, keep: torch.Tensor) -> SharedPrefixPack:
    tokens = pack.tokens.clone()
    mutate = torch.ones(int(tokens.shape[1]), dtype=torch.bool)
    mutate[keep] = False
    replacement = torch.arange(int(tokens.shape[1]), dtype=tokens.dtype) + 17
    tokens[0, mutate] = replacement[mutate] % 31
    return SharedPrefixPack(
        tokens=tokens,
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
        position_ids=pack.position_ids,
        positions_by_sequence=pack.positions_by_sequence,
    )


def _assert_matching_grads(actual_model: nn.Module, expected_model: nn.Module) -> None:
    for (name, expected_param), actual_param in zip(
        expected_model.named_parameters(),
        actual_model.parameters(),
        strict=True,
    ):
        assert expected_param.grad is not None, name
        assert actual_param.grad is not None, name
        torch.testing.assert_close(
            actual_param.grad,
            expected_param.grad,
            rtol=1e-10,
            atol=1e-10,
            msg=lambda msg, name=name: f"{name} grad mismatch:\n{msg}",
        )


def _shared_prefix_causal_mask(pack: SharedPrefixPack) -> torch.Tensor:
    group_ids = pack.group_ids.reshape(-1).tolist()
    parent_ids = pack.parent_ids.reshape(-1).tolist()
    position_ids = pack.position_ids.reshape(-1).tolist()
    parent_by_group: dict[int, int] = {}
    for group_id, parent_id in zip(group_ids, parent_ids, strict=True):
        previous = parent_by_group.setdefault(group_id, parent_id)
        assert previous == parent_id

    ancestors = {
        group_id: _ancestor_groups(group_id, parent_by_group)
        for group_id in parent_by_group
    }
    mask = torch.zeros((len(group_ids), len(group_ids)), dtype=torch.bool)
    for query_index, query_group in enumerate(group_ids):
        query_ancestors = ancestors[query_group]
        query_position = position_ids[query_index]
        for key_index, key_group in enumerate(group_ids):
            if (
                key_group in query_ancestors
                and position_ids[key_index] <= query_position
            ):
                mask[query_index, key_index] = True
    return mask


def _ancestor_groups(group_id: int, parent_by_group: dict[int, int]) -> set[int]:
    ancestors = {group_id}
    parent_id = parent_by_group[group_id]
    while parent_id != group_id:
        if parent_id in ancestors:
            raise AssertionError("shared-prefix group parents contain a cycle")
        ancestors.add(parent_id)
        group_id = parent_id
        parent_id = parent_by_group[group_id]
    return ancestors
