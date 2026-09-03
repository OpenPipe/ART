from __future__ import annotations

from collections.abc import Sequence
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.distributed import packing_request_from_text_datums
import art.preprocessing.pack as pack_module
from art.preprocessing.pack import packed_tensors_from_token_matrices
from art.preprocessing.token_matrix import token_matrix_batch_from_art_rollouts
from art.preprocessing.tokenize import SFTBatch, TokenizedResult
from art.training.token_matrix import (
    CapturedTokenRoutes,
    MatrixPair,
    NamedLossRequest,
    TextDatum,
    TokenMatrix,
    TokenMatrixBatch,
    dense_row,
    validate_token_matrix_batch,
)
from art.training.token_matrix_loss import (
    execute_token_matrix_loss,
    gather_logical_projection_values,
    logical_active_position_count,
)
from art.trajectories import Trajectory


def _matrix(
    matrix_id: str,
    *,
    tokens: tuple[int, ...] = (10, 11),
    targets: tuple[int, ...] = (11, 12),
    width: int = 1,
    weights: tuple[float, ...] | None = None,
    behavior: tuple[float, ...] | None = None,
    advantages: tuple[float, ...] | None = None,
    policy_versions: tuple[int, ...] | None = None,
) -> TokenMatrix:
    shape = (len(tokens), width)
    rows = [
        dense_row("token_ids", "int64", (len(tokens),), tokens),
        dense_row("target_token_ids", "int64", shape, targets),
    ]
    for name, values in (
        ("loss_weights", weights),
        ("behavior_logprobs", behavior),
        ("advantages", advantages),
    ):
        if values is not None:
            rows.append(dense_row(name, "float32", shape, values))
    if policy_versions is not None:
        rows.append(
            dense_row(
                "policy_version",
                "int64",
                (len(tokens),),
                policy_versions,
            )
        )
    return TokenMatrix(matrix_id=matrix_id, rows=tuple(rows))


def test_rollout_lowering_emits_first_class_captured_route_selector() -> None:
    result = TokenizedResult(
        advantage=1.0,
        chat="",
        token_ids=[10, 11],
        input_pos=[0, 1],
        assistant_mask=[0, 1],
        logprobs=[float("nan"), -0.25],
        pixel_values=None,
        image_grid_thw=None,
        trajectory=Trajectory(),
        choice_offsets=[1],
        extra_logprobs={},
        _tokenizer=SimpleNamespace(decode=str),
        weight=1.0,
        captured_route=("chatcmpl-captured", 2),
    )

    lowered = token_matrix_batch_from_art_rollouts([result], normalize_advantages=False)

    assert lowered.resolved_routes == {}
    assert lowered.batch.routes == (
        CapturedTokenRoutes(
            matrix_id="rollout-0",
            response_id="chatcmpl-captured",
            choice_index=2,
        ),
    )


def test_text_datum_uses_canonical_sft_lowering() -> None:
    calls: list[tuple[object, tuple[Trajectory, ...], dict[str, object]]] = []

    class _Tokenizer:
        def tokenize(
            self, model: object, trajectories: Sequence[Trajectory], **kwargs: object
        ) -> SFTBatch:
            calls.append((model, tuple(trajectories), kwargs))
            return SFTBatch(
                trajectory_tensors=[
                    {
                        "input_ids": torch.tensor([[10, 11, 12]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                        "labels": torch.tensor([[-100, 11, 12]]),
                    }
                ],
                learning_rate=0.0,
                num_trajectories=1,
                num_tokens=3,
                num_trainable_tokens=2,
            )

    model = SimpleNamespace(base_model="model")
    datum = TextDatum(
        datum_id="length-marker",
        messages=(
            {"role": "user", "content": "Return the learned marker."},
            {"role": "assistant", "content": "MARK MARK"},
        ),
        assistant_turns="last",
        packing_affinity_id="marker-objective",
    )

    request = packing_request_from_text_datums(
        (datum,),
        model=cast(Any, model),
        generation_id="length-objective-generation",
        packed_sequence_length=128,
        tokenizer=cast(Any, _Tokenizer()),
    )
    batch = request.batch

    assert batch.matrices[0].matrix_id == "length-marker"
    assert batch.matrices[0].packing_affinity_id == "marker-objective"
    assert batch.matrices[0].row("token_ids").dense_values() == (10, 11, 12)
    assert batch.matrices[0].row("loss_weights").dense_values() == (1.0, 1.0, 0.0)
    assert request.loss == NamedLossRequest(
        name="cross_entropy", normalize_advantages=False
    )
    assert request.packed_sequence_length == 128
    assert calls[0][1][0].messages_and_choices == list(datum.messages)
    assert calls[0][2] == {"assistant_turns": "last", "learning_rate": 0.0}


@pytest.mark.parametrize("loss_name", ["importance_sampling", "cispo"])
def test_rl_losses_synthesize_weights_from_advantage_activity(loss_name: str) -> None:
    request = NamedLossRequest(name=loss_name)
    batch = TokenMatrixBatch(
        matrices=(
            _matrix(
                "rollout",
                behavior=(-0.5, -0.25),
                advantages=(0.0, 2.0),
            ),
        )
    )

    packed = packed_tensors_from_token_matrices(
        batch,
        loss=request,
        seq_len=2,
        pack_results=False,
    )

    assert packed["assistant_mask"].tolist() == [[False, True]]
    assert packed["logical_loss_weights"].tolist() == [[0.0, 1.0]]
    assert packed["token_matrix_training_outcome"].accepted_trainable_tokens == 1
    learner = torch.tensor([[-0.4, -0.1]], requires_grad=True)
    output = execute_token_matrix_loss(
        request,
        learner_logprobs=learner,
        loss_weights=packed["logical_loss_weights"],
        behavior_logprobs=packed["logical_behavior_logprobs"],
        advantages=packed["logical_advantages"],
        logical_value_mask=packed["logical_value_mask"],
        logical_matrix_indices=packed["logical_matrix_indices"],
    )
    ratio = math.exp(-0.1 - -0.25)
    expected = (
        -ratio * 2.0 if loss_name == "importance_sampling" else -(-0.1) * ratio * 2.0
    )
    torch.testing.assert_close(output.reported_loss, torch.tensor(expected))


@pytest.mark.parametrize("loss_name", ["importance_sampling", "cispo"])
def test_rl_losses_preserve_optional_effective_weights(loss_name: str) -> None:
    request = NamedLossRequest(name=loss_name)
    packed = packed_tensors_from_token_matrices(
        TokenMatrixBatch(
            matrices=(
                _matrix(
                    "rollout",
                    weights=(0.0, 0.25),
                    behavior=(-0.5, -0.25),
                    advantages=(0.0, 2.0),
                ),
            )
        ),
        loss=request,
        seq_len=2,
        pack_results=False,
    )
    learner = torch.tensor([[-0.4, -0.1]], requires_grad=True)
    output = execute_token_matrix_loss(
        request,
        learner_logprobs=learner,
        loss_weights=packed["logical_loss_weights"],
        behavior_logprobs=packed["logical_behavior_logprobs"],
        advantages=packed["logical_advantages"],
        logical_value_mask=packed["logical_value_mask"],
        logical_matrix_indices=packed["logical_matrix_indices"],
    )
    ratio = math.exp(-0.1 - -0.25)
    expected = (
        -ratio * 2.0 * 0.25
        if loss_name == "importance_sampling"
        else -(-0.1) * ratio * 2.0 * 0.25
    )
    torch.testing.assert_close(output.reported_loss, torch.tensor(expected))


def test_accepted_positions_and_policy_coverage_are_logical() -> None:
    request = NamedLossRequest(name="cross_entropy")
    matrix = _matrix(
        "distillation",
        targets=(20, 21, 22, 23),
        width=2,
        weights=(1.0, 0.5, 0.0, 0.0),
        policy_versions=(7, -1),
    )
    packed = packed_tensors_from_token_matrices(
        TokenMatrixBatch(matrices=(matrix,)),
        loss=request,
        seq_len=2,
        pack_results=False,
    )

    outcome = packed["token_matrix_training_outcome"]
    assert outcome.accepted_trainable_tokens == 1
    assert (
        logical_active_position_count(
            packed["logical_value_mask"], packed["logical_advantages"]
        )
        == 1
    )
    assert outcome.policy_token_counts is not None
    assert [item.model_dump() for item in outcome.policy_token_counts] == [
        {"policy_version": 7, "accepted_trainable_tokens": 1}
    ]

    negative = matrix.model_copy(
        update={
            "rows": tuple(
                dense_row("policy_version", "int64", (2,), (-1, 7))
                if row.name == "policy_version"
                else row
                for row in matrix.rows
            )
        }
    )
    with pytest.raises(ValueError, match="negative policy_version"):
        validate_token_matrix_batch(TokenMatrixBatch(matrices=(negative,)), request)

    uncovered = _matrix(
        "uncovered",
        weights=(1.0, 0.0),
    )
    with pytest.raises(ValueError, match="cover every accepted position"):
        validate_token_matrix_batch(
            TokenMatrixBatch(matrices=(matrix, uncovered)), request
        )


def test_padded_logical_allocation_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    matrices = tuple(
        _matrix(str(index), tokens=tokens, weights=(1.0, 1.0))
        for index, tokens in enumerate(((1, 2), (1, 2), (1, 2), (3, 4)))
    )
    monkeypatch.setattr(pack_module, "MAX_MATRIX_VALUES", 10)

    with pytest.raises(ValueError, match="padded physical or logical"):
        packed_tensors_from_token_matrices(
            TokenMatrixBatch(matrices=matrices),
            loss=NamedLossRequest(name="cross_entropy"),
            seq_len=2,
            min_prefix_tree_shared_segment_length=0,
        )


def test_projection_ids_are_unique_across_packed_rows() -> None:
    batch = TokenMatrixBatch(
        matrices=(
            _matrix("first", tokens=(1, 2), weights=(1.0, 1.0)),
            _matrix(
                "second",
                tokens=(3, 4),
                targets=(31, 32),
                weights=(1.0, 1.0),
            ),
        )
    )
    packed = packed_tensors_from_token_matrices(
        batch,
        loss=NamedLossRequest(name="cross_entropy"),
        seq_len=2,
        pack_results=False,
    )

    projection_ids = packed["projection_ids"]
    selected = projection_ids[projection_ids >= 0]
    assert selected.unique().numel() == selected.numel()
    gathered = gather_logical_projection_values(
        local_values=packed["target_tokens"].to(dtype=torch.float32),
        local_projection_ids=projection_ids,
        logical_projection_ids=packed["logical_projection_ids"],
        logical_value_mask=packed["logical_value_mask"],
        projection_count=int(selected.max().item()) + 1,
        cp_group=None,
    )
    mask = packed["logical_value_mask"]
    expected = torch.tensor(
        [
            batch.matrices[matrix_index]
            .row("target_token_ids")
            .dense_values()[target_index]
            for matrix_index, target_index in zip(
                packed["logical_matrix_indices"][mask].tolist(),
                packed["logical_target_indices"][mask].tolist(),
                strict=True,
            )
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(gathered[mask], expected)


def test_empty_projection_retains_zero_autograd_dependency() -> None:
    local_values = torch.empty((0, 1), requires_grad=True)
    gathered = gather_logical_projection_values(
        local_values=local_values,
        local_projection_ids=torch.empty((0, 1), dtype=torch.long),
        logical_projection_ids=torch.tensor([[-1]]),
        logical_value_mask=torch.tensor([[False]]),
        projection_count=0,
        cp_group=None,
    )

    assert gathered.requires_grad
    gathered.sum().backward()
    assert local_values.grad is not None


def test_dpo_filters_absent_pairs_and_rejects_split_components() -> None:
    request = NamedLossRequest(
        name="dpo",
        matrix_pairs=(
            MatrixPair(
                component_id="first", chosen_matrix_id="a", rejected_matrix_id="b"
            ),
            MatrixPair(
                component_id="second", chosen_matrix_id="c", rejected_matrix_id="d"
            ),
        ),
    )
    learner = torch.tensor([[1.0, -1.0]], requires_grad=True)
    zeros = torch.zeros_like(learner)
    output = execute_token_matrix_loss(
        request,
        learner_logprobs=learner,
        loss_weights=torch.ones_like(learner),
        behavior_logprobs=zeros,
        advantages=zeros,
        logical_value_mask=torch.ones_like(learner, dtype=torch.bool),
        logical_matrix_indices=torch.tensor([[0, 1]]),
        matrix_pairs=((0, 1), (2, 3)),
    )

    expected = -torch.nn.functional.logsigmoid(torch.tensor(0.2))
    torch.testing.assert_close(output.reported_loss, expected)
    with pytest.raises(RuntimeError, match="split"):
        execute_token_matrix_loss(
            request,
            learner_logprobs=learner[:, :1],
            loss_weights=torch.ones_like(learner[:, :1]),
            behavior_logprobs=zeros[:, :1],
            advantages=zeros[:, :1],
            logical_value_mask=torch.ones_like(learner[:, :1], dtype=torch.bool),
            logical_matrix_indices=torch.tensor([[0]]),
            matrix_pairs=((0, 1), (2, 3)),
        )


def test_dpo_mean_active_token_is_packing_invariant() -> None:
    request = NamedLossRequest(
        name="dpo",
        matrix_pairs=(
            MatrixPair(component_id="ab", chosen_matrix_id="a", rejected_matrix_id="b"),
            MatrixPair(component_id="cd", chosen_matrix_id="c", rejected_matrix_id="d"),
        ),
    )
    learner = torch.tensor([[1.0, -1.0, 0.5, -0.25]], requires_grad=True)
    zeros = torch.zeros_like(learner)

    def loss(indices: torch.Tensor) -> tuple[torch.Tensor, int]:
        selected = learner[:, indices]
        logical_mask = torch.ones_like(selected, dtype=torch.bool)
        activity_markers = torch.ones_like(selected)
        output = execute_token_matrix_loss(
            request,
            learner_logprobs=selected,
            loss_weights=torch.ones_like(selected),
            behavior_logprobs=zeros[:, indices],
            advantages=activity_markers,
            logical_value_mask=logical_mask,
            logical_matrix_indices=indices.unsqueeze(0),
            matrix_pairs=((0, 1), (2, 3)),
        )
        return output.reported_loss, logical_active_position_count(
            logical_mask, activity_markers
        )

    together_sum, together_terms = loss(torch.tensor([0, 1, 2, 3]))
    first_sum, first_terms = loss(torch.tensor([0, 1]))
    second_sum, second_terms = loss(torch.tensor([2, 3]))
    together = together_sum / together_terms
    split = (first_sum + second_sum) / (first_terms + second_terms)
    expected = (
        -torch.nn.functional.logsigmoid(torch.tensor(0.2))
        - torch.nn.functional.logsigmoid(torch.tensor(0.075))
    ) / 4

    assert (together_terms, first_terms, second_terms) == (4, 2, 2)
    torch.testing.assert_close(together, expected)
    torch.testing.assert_close(together, split)


@pytest.mark.parametrize(("pack_results", "seq_len"), [(True, 2), (False, 4)])
def test_dpo_pairs_remain_in_one_packed_micro(
    pack_results: bool,
    seq_len: int,
) -> None:
    matrices = tuple(
        _matrix(
            matrix_id,
            tokens=tokens,
            weights=(1.0, 1.0),
            behavior=(0.0, 0.0),
        )
        for matrix_id, tokens in (
            ("a", (1, 2)),
            ("b", (1, 2)),
            ("c", (3, 4)),
            ("d", (3, 4)),
        )
    )
    request = NamedLossRequest(
        name="dpo",
        matrix_pairs=(
            MatrixPair(component_id="ab", chosen_matrix_id="a", rejected_matrix_id="b"),
            MatrixPair(component_id="cd", chosen_matrix_id="c", rejected_matrix_id="d"),
        ),
    )
    packed = packed_tensors_from_token_matrices(
        TokenMatrixBatch(matrices=matrices),
        loss=request,
        seq_len=seq_len,
        pack_results=pack_results,
        min_prefix_tree_shared_segment_length=0,
    )

    resident = {
        frozenset(row[mask].tolist())
        for row, mask in zip(
            packed["logical_matrix_indices"],
            packed["logical_value_mask"],
            strict=True,
        )
    }
    assert resident == {frozenset({0, 1}), frozenset({2, 3})}
    assert (
        logical_active_position_count(
            packed["logical_value_mask"], packed["logical_advantages"]
        )
        == 8
    )


def test_captured_routes_are_public_selectors_not_retained_bundles() -> None:
    from art.distributed.packing import retained_route_bundles_from_token_matrix_batch

    batch = TokenMatrixBatch(
        matrices=(_matrix("rollout", weights=(1.0, 0.0)),),
        routes=(
            CapturedTokenRoutes(
                matrix_id="rollout",
                response_id="chatcmpl-response",
                choice_index=2,
            ),
        ),
    )

    parsed = TokenMatrixBatch.model_validate(batch.model_dump(mode="json"))
    assert isinstance(parsed.routes[0], CapturedTokenRoutes)
    assert retained_route_bundles_from_token_matrix_batch(batch) == ()
    with pytest.raises(ValueError, match="must be resolved before packing"):
        packed_tensors_from_token_matrices(
            batch,
            loss=NamedLossRequest(name="cross_entropy"),
            seq_len=2,
            pack_results=False,
        )
