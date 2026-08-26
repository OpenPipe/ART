from types import SimpleNamespace

import pytest
import torch

from art.megatron.runtime.executor import _materialize_command_token_logprobs
from art.serverless.data_plane import decode_operation_result, encode_operation_result
from art.training.contracts import ForwardResult, LossFnOutput, PackingOutcome


def _public_result(
    values: tuple[dict[str, object], ...], *, physical_tokens: int
) -> ForwardResult:
    result = ForwardResult(
        operation_id="operation",
        packing=PackingOutcome(
            packed_sequence_length=physical_tokens,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=physical_tokens,
            physical_tokens=physical_tokens,
            non_padding_tokens=physical_tokens,
            loss_bearing_tokens=physical_tokens,
            trainable_assistant_tokens=physical_tokens,
            policy_token_counts=None,
            group_shapes=(),
        ),
        loss_fn_outputs=tuple(LossFnOutput(token_logprobs=value) for value in values),
    )
    ref, payload = encode_operation_result(result)
    return decode_operation_result(ref, payload, ForwardResult)


@pytest.mark.parametrize(
    ("candidate_capacity", "positions", "candidate_counts", "expected_shapes"),
    (
        (1, ((0, 2, 3),), (1,), ((3, 1),)),
        (3, ((0, 2), (1, 3)), (1, 3), ((2, 1), (2, 3))),
    ),
)
def test_tokenized_result_preserves_selected_target_matrix_shape(
    candidate_capacity: int,
    positions: tuple[tuple[int, ...], ...],
    candidate_counts: tuple[int, ...],
    expected_shapes: tuple[tuple[int, int], ...],
) -> None:
    ref = SimpleNamespace(
        training_kind="tokenized",
        tokenized_output_map=SimpleNamespace(
            packed_positions=positions,
            candidate_counts=candidate_counts,
        ),
        num_sequences=1,
        sequence_length=4,
    )
    physical = torch.arange(4 * candidate_capacity, dtype=torch.float32).reshape(
        4, candidate_capacity
    )
    result = _public_result(
        _materialize_command_token_logprobs(ref, candidate_capacity, (physical,)),
        physical_tokens=4,
    )

    assert (
        tuple(output.token_logprobs.shape for output in result.loss_fn_outputs)
        == expected_shapes
    )
    assert tuple(
        output.token_logprobs.to_list() for output in result.loss_fn_outputs
    ) == tuple(
        physical[list(selected), :candidates].flatten().tolist()
        for selected, candidates in zip(positions, candidate_counts, strict=True)
    )


def test_generic_result_remains_flat() -> None:
    ref = SimpleNamespace(training_kind="sft")
    values = _materialize_command_token_logprobs(
        ref,
        None,
        (torch.arange(4, dtype=torch.float32).reshape(2, 2),),
    )
    result = _public_result(values, physical_tokens=4)

    assert result.loss_fn_outputs[0].token_logprobs.shape == (4,)
    assert result.loss_fn_outputs[0].token_logprobs.to_list() == [0.0, 1.0, 2.0, 3.0]
