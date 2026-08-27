from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
import struct
from typing import TypeVar

from openai.types import Completion
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from art.preprocessing.moe_routing import MoeRouteArray

MAGIC = b"ARTRTE2\0"
ROUTE_OBJECT_MAGIC = b"ARTROU2\0"
HEADER = struct.Struct("<8sQII")
ROUTE_OBJECT_HEADER = struct.Struct("<8sII")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
DTYPES = {1: "u1", 2: "<u2"}
MAX_RESPONSE_BYTES = 4 << 30
MAX_JSON_BYTES = 64 << 20
MAX_ROUTE_COUNT = 4096
MAX_ROUTE_ARRAY_BYTES = 512 << 20
_Response = TypeVar("_Response", bound=BaseModel)


def is_routed_experts_response(body: bytes) -> bool:
    return body.startswith(MAGIC)


def decode_routed_experts_response(
    body: bytes,
) -> tuple[ChatCompletion, dict[int, MoeRouteArray]]:
    import numpy as np

    if len(body) > MAX_RESPONSE_BYTES:
        raise RuntimeError("ART routed-experts response exceeds the configured limit")
    if len(body) < HEADER.size:
        raise RuntimeError("Truncated ART routed-experts response header")
    magic, json_size, route_count, num_experts = HEADER.unpack_from(body)
    _validate_response_header(magic, json_size, route_count, num_experts)
    offset = HEADER.size
    json_end = offset + json_size
    if json_end > len(body):
        raise RuntimeError("Truncated ART routed-experts JSON response")
    response = ChatCompletion.model_validate_json(body[offset:json_end])
    offset = json_end
    routes: dict[int, MoeRouteArray] = {}
    for _ in range(route_count):
        if offset + ROUTE_HEADER.size > len(body):
            raise RuntimeError("Truncated ART routed-experts array header")
        choice_index, dtype_code, tokens, layers, topk = ROUTE_HEADER.unpack_from(
            body, offset
        )
        offset += ROUTE_HEADER.size
        dtype_name, element_count, size = _route_layout(
            dtype_code, tokens, layers, topk, num_experts
        )
        dtype = np.dtype(dtype_name)
        end = offset + size
        if end > len(body):
            raise RuntimeError("Truncated ART routed-experts array")
        if choice_index in routes:
            raise RuntimeError(f"Duplicate routed experts for choice {choice_index}")
        array = np.frombuffer(body, dtype=dtype, count=element_count, offset=offset)
        routes[choice_index] = MoeRouteArray(
            array.reshape((tokens, layers, topk)), num_experts=num_experts
        )
        offset = end
    if offset != len(body):
        raise RuntimeError("Unexpected trailing bytes in ART routed-experts response")
    return response, routes


async def decode_routed_experts_response_stream(
    chunks: AsyncIterable[bytes],
) -> tuple[ChatCompletion, dict[int, MoeRouteArray]]:
    return await _decode_routed_experts_response_stream(chunks, ChatCompletion)


async def decode_routed_experts_completion_response_stream(
    chunks: AsyncIterable[bytes],
) -> tuple[Completion, dict[int, MoeRouteArray]]:
    return await _decode_routed_experts_response_stream(chunks, Completion)


async def decode_routed_experts_object_stream(
    chunks: AsyncIterable[bytes],
) -> dict[int, MoeRouteArray]:
    """Decode the route-only object uploaded independently of generation JSON."""

    reader = _AsyncByteReader(chunks.__aiter__())
    header = await reader.read_exact(
        ROUTE_OBJECT_HEADER.size, "ART routed-experts object header"
    )
    magic, route_count, num_experts = ROUTE_OBJECT_HEADER.unpack_from(header)
    if magic != ROUTE_OBJECT_MAGIC:
        raise RuntimeError("Invalid ART routed-experts object magic")
    _validate_route_contract(route_count, num_experts)
    routes = await _decode_route_arrays(reader, route_count, num_experts)
    await reader.finish()
    return routes


async def _decode_routed_experts_response_stream(
    chunks: AsyncIterable[bytes], response_type: type[_Response]
) -> tuple[_Response, dict[int, MoeRouteArray]]:
    import numpy as np

    reader = _AsyncByteReader(chunks.__aiter__())
    header = await reader.read_exact(HEADER.size, "ART routed-experts response header")
    magic, json_size, route_count, num_experts = HEADER.unpack_from(header)
    _validate_response_header(magic, json_size, route_count, num_experts)
    response = response_type.model_validate_json(
        await reader.read_exact(json_size, "ART routed-experts JSON response")
    )
    routes = await _decode_route_arrays(reader, route_count, num_experts)
    await reader.finish()
    return response, routes


async def _decode_route_arrays(
    reader: "_AsyncByteReader", route_count: int, num_experts: int
) -> dict[int, MoeRouteArray]:
    import numpy as np

    routes: dict[int, MoeRouteArray] = {}
    for _ in range(route_count):
        route_header = await reader.read_exact(
            ROUTE_HEADER.size, "ART routed-experts array header"
        )
        choice_index, dtype_code, tokens, layers, topk = ROUTE_HEADER.unpack_from(
            route_header
        )
        dtype_name, element_count, size = _route_layout(
            dtype_code, tokens, layers, topk, num_experts
        )
        if choice_index in routes:
            raise RuntimeError(f"Duplicate routed experts for choice {choice_index}")
        data = await reader.read_exact(size, "ART routed-experts array")
        array = np.frombuffer(data, dtype=np.dtype(dtype_name), count=element_count)
        routes[choice_index] = MoeRouteArray(
            array.reshape((tokens, layers, topk)), num_experts=num_experts
        )
    return routes


def _validate_response_header(
    magic: bytes,
    json_size: int,
    route_count: int,
    num_experts: int,
) -> None:
    if magic != MAGIC:
        raise RuntimeError("Invalid ART routed-experts response magic")
    if not 0 < json_size <= MAX_JSON_BYTES:
        raise RuntimeError("ART routed-experts JSON size is outside configured bounds")
    _validate_route_contract(route_count, num_experts)


def _validate_route_contract(route_count: int, num_experts: int) -> None:
    if route_count < 1:
        raise RuntimeError("ART routed-experts response contains no route arrays")
    if route_count > MAX_ROUTE_COUNT:
        raise RuntimeError("ART routed-experts route count exceeds configured bounds")
    if not 1 <= num_experts <= 65_536:
        raise RuntimeError("ART routed-experts expert count is outside valid bounds")


def _route_layout(
    dtype_code: int,
    tokens: int,
    layers: int,
    topk: int,
    num_experts: int,
) -> tuple[str, int, int]:
    dtype_name = DTYPES.get(dtype_code)
    if dtype_name is None:
        raise RuntimeError(f"Unknown ART route dtype code {dtype_code}")
    expected_dtype_code = 1 if num_experts <= 256 else 2
    if dtype_code != expected_dtype_code:
        raise RuntimeError("ART route dtype disagrees with the exact expert count")
    if min(tokens, layers, topk) < 1 or topk > num_experts:
        raise RuntimeError("ART routed-experts array shape is outside valid bounds")
    element_count = int(tokens * layers * topk)
    size = element_count * dtype_code
    if size > MAX_ROUTE_ARRAY_BYTES:
        raise RuntimeError("ART routed-experts array exceeds the configured limit")
    return dtype_name, element_count, size


class _AsyncByteReader:
    def __init__(self, chunks: AsyncIterator[bytes]) -> None:
        self._chunks = chunks
        self._chunk = memoryview(b"")
        self._offset = 0
        self.byte_count = 0

    async def read_exact(self, size: int, label: str) -> bytearray:
        data = bytearray(size)
        target = memoryview(data)
        written = 0
        try:
            while written < size:
                if self._offset == len(self._chunk) and not await self._advance():
                    raise RuntimeError(f"Truncated {label}")
                count = min(size - written, len(self._chunk) - self._offset)
                target[written : written + count] = self._chunk[
                    self._offset : self._offset + count
                ]
                self._offset += count
                written += count
        finally:
            target.release()
        return data

    async def finish(self) -> None:
        if self._offset != len(self._chunk) or await self._advance():
            raise RuntimeError(
                "Unexpected trailing bytes in ART routed-experts response"
            )

    async def _advance(self) -> bool:
        while True:
            try:
                chunk = await anext(self._chunks)
            except StopAsyncIteration:
                return False
            if not isinstance(chunk, bytes):
                raise RuntimeError(
                    "ART routed-experts response stream yielded a non-bytes chunk"
                )
            if not chunk:
                continue
            self.byte_count += len(chunk)
            if self.byte_count > MAX_RESPONSE_BYTES:
                raise RuntimeError(
                    "ART routed-experts response exceeds the configured limit"
                )
            self._chunk = memoryview(chunk)
            self._offset = 0
            return True
