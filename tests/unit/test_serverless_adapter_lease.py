import art
from art.serverless.backend import ServerlessBackend


async def test_serverless_adapter_lease_pins_inference_step() -> None:
    backend = ServerlessBackend(api_key="test-api-key")
    model = art.TrainableModel(
        name="test-model",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model._backend = backend

    assert (
        model.get_inference_name()
        == "wandb-artifact:///test-entity/test-project/test-model"
    )

    async with backend.adapter_lease(model, 3):
        assert (
            model.get_inference_name()
            == "wandb-artifact:///test-entity/test-project/test-model:step3"
        )
        assert (
            model.get_inference_name(step=4)
            == "wandb-artifact:///test-entity/test-project/test-model:step4"
        )

    assert (
        model.get_inference_name()
        == "wandb-artifact:///test-entity/test-project/test-model"
    )


async def test_serverless_adapter_lease_is_model_scoped() -> None:
    backend = ServerlessBackend(api_key="test-api-key")
    model_a = art.TrainableModel(
        name="model-a",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model_b = art.TrainableModel(
        name="model-b",
        project="test-project",
        entity="test-entity",
        base_model="test-base-model",
    )
    model_a._backend = backend
    model_b._backend = backend

    async with backend.adapter_lease(model_a, 2):
        assert (
            model_a.get_inference_name()
            == "wandb-artifact:///test-entity/test-project/model-a:step2"
        )
        assert (
            model_b.get_inference_name()
            == "wandb-artifact:///test-entity/test-project/model-b"
        )
