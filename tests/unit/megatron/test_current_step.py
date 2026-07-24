from types import SimpleNamespace
from typing import Any

import pytest

from art import distill
from art.megatron.backend import MegatronBackend
from art.megatron.writer_sessions import (
    AmbiguousWriterSessionError,
    WriterLease,
)


class _CurrentStepService:
    def __init__(self, *, release_error: BaseException | None = None) -> None:
        self.capability = b"s" * 32
        self.release_error = release_error
        self.released = False

    async def acquire_current_step(self, *, revision: int, ttl_s: float) -> WriterLease:
        return WriterLease(
            model_identity="project/model",
            revision=revision,
            session_id="session-1",
            fence=3,
            expires_at=ttl_s,
            kind="current_step",
            capability=self.capability,
        )

    async def heartbeat_current_step(self, **_kwargs: Any) -> float:
        return 1_000.0

    async def release_current_step(self, **_kwargs: Any) -> None:
        self.released = True
        if self.release_error is not None:
            raise self.release_error


def _model() -> Any:
    return SimpleNamespace(
        trainable=True,
        project="project",
        _storage_name=lambda: "model",
    )


@pytest.mark.asyncio
async def test_public_current_step_shape_binds_serializable_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    backend = MegatronBackend(path=str(tmp_path))
    service = _CurrentStepService()
    model = _model()

    async def _service(_: Any) -> _CurrentStepService:
        return service

    async def _step(_: Any) -> int:
        return 7

    monkeypatch.setattr(backend, "_get_service", _service)
    monkeypatch.setattr(backend, "_get_step", _step)

    async with backend.current_step(model) as current:
        consistency = distill.CurrentStep(current)
        assert consistency == distill.CurrentStep(
            revision=7,
            session_id="session-1",
        )
        assert "capability" not in consistency.model_dump_json()
        await backend._validate_current_step(model, consistency)
        active = backend._require_active_current_step(
            model,
            consistency,
            current,
        )
        assert active.session is current
        with pytest.raises(ValueError, match="same active"):
            backend._require_active_current_step(
                model,
                consistency,
                SimpleNamespace(
                    revision=current.revision,
                    session_id=current.session_id,
                ),
            )

    assert service.released


@pytest.mark.asyncio
async def test_body_failure_and_ambiguous_release_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    backend = MegatronBackend(path=str(tmp_path))
    service = _CurrentStepService(
        release_error=AmbiguousWriterSessionError("bound outcome is ambiguous")
    )
    model = _model()

    async def _service(_: Any) -> _CurrentStepService:
        return service

    async def _step(_: Any) -> int:
        return 2

    monkeypatch.setattr(backend, "_get_service", _service)
    monkeypatch.setattr(backend, "_get_step", _step)

    with pytest.raises(BaseExceptionGroup) as raised:
        async with backend.current_step(model):
            raise ValueError("application failed after submission")

    assert any(isinstance(exc, ValueError) for exc in raised.value.exceptions)
    assert any(
        isinstance(exc, AmbiguousWriterSessionError) for exc in raised.value.exceptions
    )
