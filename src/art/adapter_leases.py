from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator

from pydantic import BaseModel, ConfigDict


class InferenceTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    model_name: str | None = None


_pinned_inference_targets: ContextVar[dict[str, InferenceTarget]] = ContextVar(
    "art_pinned_inference_targets",
    default={},
)


def in_flight_lora_name(model_name: str) -> str:
    return f"{model_name}:active"


def pinned_inference_step(model_name: str) -> int | None:
    target = _pinned_inference_targets.get().get(model_name)
    return None if target is None else target.step


def pinned_inference_name(model_name: str, step: int | None = None) -> str | None:
    target = _pinned_inference_targets.get().get(model_name)
    if target is None or (step is not None and step != target.step):
        return None
    return target.model_name


@asynccontextmanager
async def pin_inference_step(
    model_name: str,
    step: int,
) -> AsyncIterator[None]:
    async with pin_inference_target(model_name, step=step):
        yield


@asynccontextmanager
async def pin_inference_target(
    model_name: str,
    *,
    step: int,
    inference_name: str | None = None,
) -> AsyncIterator[None]:
    targets = dict(_pinned_inference_targets.get())
    targets[model_name] = InferenceTarget(step=step, model_name=inference_name)
    token = _pinned_inference_targets.set(targets)
    try:
        yield
    finally:
        _pinned_inference_targets.reset(token)
