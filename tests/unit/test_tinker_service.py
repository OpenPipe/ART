import pytest
import yaml


@pytest.fixture
def tinker_service_module():
    try:
        import art.tinker.service as service_module

        return service_module
    except ImportError as e:
        pytest.skip(f"Tinker dependencies not available: {e}")


@pytest.mark.asyncio
async def test_get_state_reuses_nested_user_metadata_from_training_client_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    tinker_service_module,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "0001"
    checkpoint_dir.mkdir(parents=True)
    info_path = checkpoint_dir / "info.yaml"
    info_path.write_text(
        yaml.safe_dump(
            {
                "state_with_optimizer_path": "tinker://state/0001",
                "sampler_weights_path": "tinker://sampler/0001",
            }
        )
    )

    observed: dict[str, object] = {}
    fake_training_client = object()

    class FakeServiceClient:
        def create_rest_client(self) -> object:
            return object()

        async def create_training_client_from_state_with_optimizer_async(
            self,
            *,
            path: str,
            user_metadata: dict[str, str] | None = None,
        ) -> object:
            observed["path"] = path
            observed["user_metadata"] = user_metadata
            return fake_training_client

    monkeypatch.setattr(
        tinker_service_module.tinker,
        "ServiceClient",
        FakeServiceClient,
    )

    service = tinker_service_module.TinkerService(
        model_name="test-model",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config={
            "tinker_args": {
                "renderer_name": "qwen3_5",
                "training_client_args": {
                    "user_metadata": {"tenant": "test-tenant"},
                },
            }
        },
        output_dir=str(tmp_path),
    )

    state = await service._get_state()

    assert observed["path"] == "tinker://state/0001"
    assert observed["user_metadata"] == {"tenant": "test-tenant"}
    assert state.training_client is fake_training_client
