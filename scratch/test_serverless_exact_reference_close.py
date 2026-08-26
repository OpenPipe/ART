from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from art import TrainableModel
from art.serverless.backend import ServerlessBackend


@pytest.mark.asyncio
async def test_close_rejects_active_checkpoint_reference() -> None:
    sampler = AsyncMock()
    service = AsyncMock()
    client = AsyncMock()
    backend = ServerlessBackend(
        training_base_url="http://training.invalid/v1",
        inference_base_url="http://inference.invalid/v1",
        sampler_manager=sampler,
        api_key="test",
    )
    model = TrainableModel(
        name="model",
        run_name="run",
        project="scratch",
        base_model="Qwen/Qwen3.5-35B-A3B",
    )
    model_key = backend._model_key(model)
    backend._clients[model_key] = cast(Any, client)
    backend._service = cast(Any, service)
    backend._checkpoint_reference_counts[(model_key, 1)] = 1

    with pytest.raises(RuntimeError, match="active exact references"):
        await backend.delete(model)
    with pytest.raises(RuntimeError, match="active exact references"):
        await backend.close()

    client.shutdown.assert_not_awaited()
    service.close.assert_not_awaited()
    backend._checkpoint_reference_counts.clear()
    await backend.close()
