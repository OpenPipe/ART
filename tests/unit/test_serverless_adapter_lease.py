from unittest.mock import AsyncMock

import art
from art.serverless.backend import ServerlessBackend


async def test_serverless_adapter_lease_pins_inference_step() -> None:
    backend = ServerlessBackend(
        api_key="test-api-key",
        training_base_url="http://training.test/v1",
        inference_base_url="http://inference.test/v1",
        sampler_publisher=AsyncMock(),
    )
    model = art.TrainableModel(
        run_name="test-model",
        name="serving-model",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model._backend = backend

    assert model.get_inference_name() == "serving-model"

    async with backend.adapter_lease(model, 3):
        assert model.get_inference_name() == "serving-model"
        assert model.get_inference_name(step=4) == "serving-model@4"

    assert model.get_inference_name() == "serving-model"

    async with backend.exact_adapter_lease(model, 3):
        assert model.get_inference_name() == "serving-model@3"


async def test_serverless_adapter_lease_is_model_scoped() -> None:
    backend = ServerlessBackend(
        api_key="test-api-key",
        training_base_url="http://training.test/v1",
        inference_base_url="http://inference.test/v1",
        sampler_publisher=AsyncMock(),
    )
    model_a = art.TrainableModel(
        run_name="model-a",
        name="model-a",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model_b = art.TrainableModel(
        run_name="model-b",
        name="model-b",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model_a._backend = backend
    model_b._backend = backend

    async with backend.adapter_lease(model_a, 2):
        assert model_a.get_inference_name() == "model-a"
        assert model_b.get_inference_name() == "model-b"
