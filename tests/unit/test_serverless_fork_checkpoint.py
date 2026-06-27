"""Tests for cross-entity checkpoint forking (issue #649).

``_experimental_fork_checkpoint`` previously located the source checkpoint under
the *destination* model's entity, so forking from a checkpoint in another W&B
entity was impossible. These cover the new ``from_entity`` parameter and the
entity-resolution helper it flows through.
"""

from types import SimpleNamespace

import pytest

from art.serverless.backend import (
    ServerlessBackend,
    _wandb_checkpoint_collection_path,
)


def test_collection_path_prefers_explicit_from_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="src-model",
        from_project="src-project",
        from_entity="src-entity",
        model_entity="dst-entity",
        default_entity="default-entity",
    )
    assert path == "src-entity/src-project/src-model"


def test_collection_path_falls_back_to_model_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="src-model",
        from_project="src-project",
        from_entity=None,
        model_entity="dst-entity",
        default_entity="default-entity",
    )
    assert path == "dst-entity/src-project/src-model"


def test_collection_path_falls_back_to_default_entity():
    path = _wandb_checkpoint_collection_path(
        from_model="src-model",
        from_project="src-project",
        from_entity=None,
        model_entity=None,
        default_entity="default-entity",
    )
    assert path == "default-entity/src-project/src-model"


def test_collection_path_requires_an_entity():
    with pytest.raises(ValueError, match="W&B entity"):
        _wandb_checkpoint_collection_path(
            from_model="src-model",
            from_project="src-project",
            from_entity=None,
            model_entity=None,
            default_entity=None,
        )


@pytest.mark.asyncio
async def test_fork_checkpoint_queries_explicit_source_entity(monkeypatch):
    """An explicit from_entity must be used when querying W&B artifacts, even
    when the destination model lives in a different entity."""
    artifact_calls = []

    class FakeApi:
        default_entity = "default-entity"

        def __init__(self, api_key):
            assert api_key == "test-api-key"

        def artifacts(self, artifact_type, collection_path):
            artifact_calls.append((artifact_type, collection_path))
            return []  # no versions -> method raises "No checkpoints found"

    monkeypatch.setattr("art.serverless.backend.wandb_sdk.api", FakeApi)

    backend = ServerlessBackend.__new__(ServerlessBackend)
    backend._client = SimpleNamespace(api_key="test-api-key")
    model = SimpleNamespace(
        entity="dst-entity", project="dst-project", name="dst-model"
    )

    with pytest.raises(ValueError, match="No checkpoints found"):
        await backend._experimental_fork_checkpoint(
            model,
            from_model="src-model",
            from_project="src-project",
            from_entity="src-entity",
        )

    assert artifact_calls == [("lora", "src-entity/src-project/src-model")]
