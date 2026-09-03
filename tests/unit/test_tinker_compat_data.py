import pytest

tinker = pytest.importorskip("tinker")

from art.tinker_compat.data import translate_tinker_forward_input  # noqa: E402
from art.training.token_matrix import (  # noqa: E402
    NamedLossRequest,
    TokenMatrixBatch,
)


def test_forward_input_lowers_dense_and_csr_datums_to_token_matrices() -> None:
    dense = tinker.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=[10, 11])]
        ),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[1.0, 0.25], dtype="float32", shape=[2]),
        },
    )
    csr = tinker.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=[20, 21])]
        ),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[22, 23],
                dtype="int64",
                shape=[2, 2],
                sparse_crow_indices=[0, 1, 2],
                sparse_col_indices=[1, 0],
            ),
            "weights": tinker.TensorData(
                data=[0.5, 1.5],
                dtype="float32",
                shape=[2, 2],
                sparse_crow_indices=[0, 1, 2],
                sparse_col_indices=[1, 0],
            ),
        },
    )

    translated = translate_tinker_forward_input(
        tinker.types.ForwardBackwardInput(
            data=[dense, csr],
            loss_fn="cross_entropy",
        )
    )

    assert isinstance(translated.batch, TokenMatrixBatch)
    assert translated.loss == NamedLossRequest(
        name="cross_entropy", normalize_advantages=False
    )
    assert translated.target_shapes == ((2,), (2, 2))

    dense_matrix, csr_matrix = translated.batch.matrices
    assert tuple(row.name for row in dense_matrix.rows) == (
        "token_ids",
        "target_token_ids",
        "loss_weights",
    )
    assert dense_matrix.row("target_token_ids").shape == (2, 1)
    assert dense_matrix.row("target_token_ids").dense_values() == (11, 12)
    assert dense_matrix.row("loss_weights").dense_values() == (1.0, 0.25)

    assert csr_matrix.row("target_token_ids").shape == (2, 2)
    assert csr_matrix.row("target_token_ids").dense_values() == (0, 22, 23, 0)
    assert csr_matrix.row("loss_weights").dense_values() == (0.0, 0.5, 1.5, 0.0)
