import sys
from types import SimpleNamespace

import pytest

from art.serverless.backend import (
    ServerlessBackend,
    _wandb_checkpoint_collection_path,
)


def test_checkpoint_collection_path_prefers_explicit_source_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="source-model",
        from_project="source-project",
        from_entity="source-entity",
        model_entity="destination-entity",
        default_entity="default-entity",
    )

    assert path == "source-entity/source-project/source-model"


def test_checkpoint_collection_path_falls_back_to_destination_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="source-model",
        from_project="source-project",
        from_entity=None,
        model_entity="destination-entity",
        default_entity="default-entity",
    )

    assert path == "destination-entity/source-project/source-model"


def test_checkpoint_collection_path_falls_back_to_default_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="source-model",
        from_project="source-project",
        from_entity=None,
        model_entity=None,
        default_entity="default-entity",
    )

    assert path == "default-entity/source-project/source-model"


def test_checkpoint_collection_path_requires_an_entity():
    with pytest.raises(ValueError, match="W&B entity"):
        _wandb_checkpoint_collection_path(
            from_model="source-model",
            from_project="source-project",
            from_entity=None,
            model_entity=None,
            default_entity=None,
        )


@pytest.mark.asyncio
async def test_fork_checkpoint_uses_explicit_source_entity(monkeypatch):
    artifact_calls = []

    class FakeApi:
        default_entity = "default-entity"

        def __init__(self, api_key):
            assert api_key == "test-api-key"

        def artifacts(self, artifact_type, collection_path):
            artifact_calls.append((artifact_type, collection_path))
            return []

    fake_wandb = SimpleNamespace(Api=FakeApi)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    backend = ServerlessBackend.__new__(ServerlessBackend)
    backend._client = SimpleNamespace(api_key="test-api-key")
    model = SimpleNamespace(
        entity="destination-entity",
        project="destination-project",
        name="destination-model",
    )

    with pytest.raises(ValueError, match="No checkpoints found"):
        await backend._experimental_fork_checkpoint(
            model,
            from_model="source-model",
            from_project="source-project",
            from_entity="source-entity",
        )

    assert artifact_calls == [("lora", "source-entity/source-project/source-model")]
