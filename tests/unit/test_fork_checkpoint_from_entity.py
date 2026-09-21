import ast
import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from art.serverless.backend import ServerlessBackend


def test_serverless_backend_signature_includes_from_entity():
    """Verify ServerlessBackend._experimental_fork_checkpoint accepts from_entity."""
    sig = inspect.signature(ServerlessBackend._experimental_fork_checkpoint)
    assert "from_entity" in sig.parameters
    param = sig.parameters["from_entity"]
    assert param.default is None


def test_local_and_tinker_native_backend_signatures_include_from_entity():
    """Verify LocalBackend and TinkerNativeBackend AST signatures include from_entity."""
    for relative_path in [
        "src/art/local/backend.py",
        "src/art/tinker_native/backend.py",
    ]:
        path = Path(relative_path)
        tree = ast.parse(path.read_text())
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_experimental_fork_checkpoint":
                arg_names = [arg.arg for arg in node.args.args]
                assert "from_entity" in arg_names
                found = True
                break
        assert found, f"_experimental_fork_checkpoint not found in {relative_path}"


@pytest.mark.asyncio
async def test_serverless_backend_from_entity_resolution():
    """Verify ServerlessBackend correctly uses from_entity when provided."""
    from unittest.mock import patch

    backend = ServerlessBackend.__new__(ServerlessBackend)
    backend._client = MagicMock()
    backend._client.api_key = "fake-api-key"

    model = MagicMock()
    model.project = "dest-project"
    model.entity = "dest-entity"

    mock_version = MagicMock()
    mock_version.name = "v1"
    mock_version.metadata = {"step": 10}

    mock_api = MagicMock()
    mock_api.default_entity = "default-entity"
    mock_api.artifacts.return_value = [mock_version]

    backend._pull_checkpoint_from_wandb = AsyncMock()
    backend._upload_checkpoint_to_wandb = AsyncMock()

    mock_run = MagicMock()
    mock_dest_artifact = MagicMock()

    with (
        patch("art.utils.wandb_sdk.api", return_value=mock_api),
        patch("art.utils.wandb_sdk.login"),
        patch("art.utils.wandb_sdk.init", return_value=mock_run),
        patch("art.utils.wandb_sdk.settings", return_value=MagicMock()),
        patch("art.utils.wandb_sdk.artifact", return_value=mock_dest_artifact),
        patch("art.serverless.backend._extract_step_from_wandb_artifact", return_value=10),
    ):
        # Test 1: Explicit from_entity overrides model.entity and api.default_entity
        await backend._experimental_fork_checkpoint(
            model=model,
            from_model="src-model",
            from_project="src-project",
            from_entity="custom-entity",
        )
        mock_api.artifacts.assert_called_with(
            "lora", "custom-entity/src-project/src-model"
        )

        # Test 2: When from_entity is None, falls back to model.entity
        mock_api.reset_mock()
        await backend._experimental_fork_checkpoint(
            model=model,
            from_model="src-model",
            from_project="src-project",
            from_entity=None,
        )
        mock_api.artifacts.assert_called_with(
            "lora", "dest-entity/src-project/src-model"
        )

        # Test 3: When from_entity and model.entity are None, falls back to api.default_entity
        model.entity = None
        mock_api.reset_mock()
        try:
            await backend._experimental_fork_checkpoint(
                model=model,
                from_model="src-model",
                from_project="src-project",
                from_entity=None,
            )
        except AssertionError:
            pass
        mock_api.artifacts.assert_called_with(
            "lora", "default-entity/src-project/src-model"
        )


