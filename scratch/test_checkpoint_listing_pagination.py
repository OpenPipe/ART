from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from pydantic import ValidationError
import pytest

from art.serverless.backend import ServerlessBackend
from art.serverless.client import RemoteTrainingServiceClient
from art.serverless.contracts import (
    MAX_CHECKPOINT_ALIASES_PER_VIEW,
    MAX_CHECKPOINT_PAGE_LIMIT,
    CheckpointPage,
    CheckpointView,
)


def _checkpoint(index: int) -> CheckpointView:
    return CheckpointView(
        checkpoint_id=f"checkpoint-{index:04d}",
        revision=1,
        learner_version=index,
        has_optimizer=True,
        state="ready",
        adapter_bytes=1,
        optimizer_bytes=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_complete_checkpoint_listing_traverses_bounded_pages() -> None:
    checkpoints = tuple(_checkpoint(index) for index in range(105))
    requests: list[tuple[str | None, str | None]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        requests.append((request.url.params.get("limit"), cursor))
        page = CheckpointPage(
            checkpoints=checkpoints[:100] if cursor is None else checkpoints[100:],
            current_checkpoint_id="checkpoint-0104",
            next_cursor="second_page" if cursor is None else None,
        )
        return httpx.Response(200, json=page.model_dump(mode="json"))

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )

    pages = [page async for page in service.iter_checkpoint_pages("run")]

    assert (
        tuple(checkpoint for page in pages for checkpoint in page.checkpoints)
        == checkpoints
    )
    assert all(page.current_checkpoint_id == "checkpoint-0104" for page in pages)
    assert all(len(page.checkpoints) <= 100 for page in pages)
    assert requests == [("100", None), ("100", "second_page")]
    await service.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_retention_snapshot_rejects_checkpoint_513() -> None:
    checkpoints = tuple(_checkpoint(index) for index in range(513))
    backend = object.__new__(ServerlessBackend)
    backend.training_client = AsyncMock(return_value=SimpleNamespace(run_id="run"))

    async def pages(_run_id: str):
        for offset in range(0, len(checkpoints), 100):
            yield CheckpointPage(
                checkpoints=checkpoints[offset : offset + 100],
                current_checkpoint_id="checkpoint-0512",
                next_cursor=("next" if offset + 100 < len(checkpoints) else None),
            )

    backend._service = SimpleNamespace(iter_checkpoint_pages=pages)

    with pytest.raises(RuntimeError, match="exceeds 512 checkpoints"):
        await backend._list_checkpoint_infos(object())


def test_checkpoint_wire_contracts_are_hard_bounded() -> None:
    checkpoint = _checkpoint(0)
    with pytest.raises(ValidationError, match="at most 512 items"):
        CheckpointPage(
            checkpoints=(checkpoint,) * (MAX_CHECKPOINT_PAGE_LIMIT + 1),
            current_checkpoint_id=checkpoint.checkpoint_id,
            next_cursor=None,
        )
    with pytest.raises(ValidationError, match="at most 100 items"):
        CheckpointView.model_validate(
            {
                **checkpoint.model_dump(),
                "aliases": ("alias",) * (MAX_CHECKPOINT_ALIASES_PER_VIEW + 1),
            }
        )
    with pytest.raises(ValidationError, match="String should match pattern"):
        CheckpointPage(
            checkpoints=(),
            current_checkpoint_id=None,
            next_cursor="not-an-opaque-cursor=",
        )
