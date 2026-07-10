from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import struct
from typing import Any

import numpy as np

MAGIC = b"ARTRTE1\0"
HEADER = struct.Struct("<8sQI")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
_CAPTURE: ContextVar[dict[int, np.ndarray] | None] = ContextVar(
    "art_binary_routed_experts", default=None
)


@contextmanager
def capture_routed_experts() -> Iterator[dict[int, np.ndarray]]:
    routes: dict[int, np.ndarray] = {}
    token = _CAPTURE.set(routes)
    try:
        yield routes
    finally:
        _CAPTURE.reset(token)


def encode_routed_experts_response(
    json_body: bytes, routes: dict[int, np.ndarray]
) -> bytes:
    chunks: list[bytes | memoryview] = [
        HEADER.pack(MAGIC, len(json_body), len(routes)),
        json_body,
    ]
    for choice_index, array in sorted(routes.items()):
        if array.ndim != 3:
            raise RuntimeError(f"Routed experts must have rank 3, got {array.shape}")
        if array.dtype == np.dtype(np.uint8):
            dtype_code = 1
        elif array.dtype == np.dtype(np.uint16):
            dtype_code = 2
            array = array.astype("<u2", copy=False)
        else:
            raise RuntimeError(
                f"vLLM routed experts must use uint8 or uint16, got {array.dtype}"
            )
        array = np.ascontiguousarray(array)
        chunks.extend(
            (
                ROUTE_HEADER.pack(choice_index, dtype_code, *array.shape),
                memoryview(array).cast("B"),
            )
        )
    return b"".join(chunks)


def patch_binary_routed_experts_response() -> None:
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    original = OpenAIServingChat.chat_completion_full_generator
    if getattr(original, "__art_binary_routes_patched__", False):
        return

    @wraps(original)
    async def patched(
        self: Any,
        request: Any,
        result_generator: AsyncIterator[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        capture = _CAPTURE.get()
        if capture is None:
            return await original(self, request, result_generator, *args, **kwargs)

        async def stripped_results() -> AsyncIterator[Any]:
            async for result in result_generator:
                for output in result.outputs:
                    if output.routed_experts is not None:
                        capture[int(output.index)] = output.routed_experts
                        output.routed_experts = None
                yield result

        return await original(self, request, stripped_results(), *args, **kwargs)

    patched.__art_binary_routes_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_full_generator = patched
