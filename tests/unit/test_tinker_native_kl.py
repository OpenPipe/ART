from typing import Any, cast
from unittest.mock import patch

import pytest
import tinker

from art import PreparedTrainingBatch, TrainableModel
from art.tinker_native.backend import TinkerNativeBackend, _apply_kl_penalty
from art.tinker_native.data import build_datum


class FakeSamplingClient(tinker.SamplingClient):
    def __init__(self, responses: dict[tuple[int, ...], list[float | None]]) -> None:
        self._responses = responses

    async def compute_logprobs_async(
        self, prompt: tinker.ModelInput
    ) -> list[float | None]:
        return self._responses[tuple(prompt.to_ints())]


@pytest.mark.asyncio
async def test_incorporate_kl_penalty_rewrites_advantages_in_place() -> None:
    datum_a = build_datum(
        prompt_tokens=[101, 102],
        completion_tokens=[201, 202],
        logprobs=[-0.4, -0.8],
        advantage=1.0,
    )
    datum_b = build_datum(
        prompt_tokens=[301, 302],
        completion_tokens=[401],
        logprobs=[-0.2],
        advantage=2.0,
    )
    assert datum_a is not None
    assert datum_b is not None

    sampling_client = FakeSamplingClient(
        {
            (101, 102, 201, 202): [None, -9.0, -0.1, -0.5],
            (301, 302, 401): [None, -7.0, -0.05],
        }
    )

    metrics = await _apply_kl_penalty(
        [datum_a, datum_b],
        sampling_client,
        kl_penalty_coef=2.0,
    )

    assert metrics == {"loss/kl_policy_ref": pytest.approx(-0.25)}
    assert datum_a.loss_fn_inputs["advantages"].tolist() == pytest.approx(
        [0.0, 1.1, 1.1]
    )
    assert datum_b.loss_fn_inputs["advantages"].tolist() == pytest.approx([0.0, 1.8])


@pytest.mark.asyncio
async def test_tinker_native_backend_rejects_current_learner_kl_source(
    tmp_path,
) -> None:
    backend = TinkerNativeBackend(tinker_api_key="test-key", path=str(tmp_path))
    model = TrainableModel(
        run_name="tinker-native-kl-source",
        name="tinker-native-kl-source",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )

    with pytest.raises(
        AssertionError,
        match="only supports kl_penalty_source='sample'",
    ):
        await backend.train(
            model,
            [],
            kl_penalty_coef=0.25,
            kl_penalty_source="current_learner",  # ty:ignore[invalid-argument-type]
        )


@pytest.mark.asyncio
async def test_tinker_native_backend_rejects_prepared_batch_before_training(
    tmp_path,
) -> None:
    backend = TinkerNativeBackend(tinker_api_key="test-key", path=str(tmp_path))
    prepared = PreparedTrainingBatch(
        schema_version=1,
        batch_id="unused",
        preparation_id="unused",
        payload=b"unused",
        payload_sha256="unused",
    )

    with (
        patch("art.tinker_native.backend.trajectory_groups_to_datums") as to_datums,
        pytest.raises(
            NotImplementedError,
            match="TinkerNativeBackend does not support PreparedTrainingBatch",
        ),
    ):
        await backend.train(cast(Any, object()), cast(Any, prepared))

    to_datums.assert_not_called()
