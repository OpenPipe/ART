from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    import numpy as np

MAGIC = b"ARTRTE1\0"
HEADER = struct.Struct("<8sQI")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
DTYPES = {1: "u1", 2: "<u2"}


def is_routed_experts_response(body: bytes) -> bool:
    return body.startswith(MAGIC)


def decode_routed_experts_response(
    body: bytes,
) -> tuple[ChatCompletion, dict[int, np.ndarray]]:
    import numpy as np

    if len(body) < HEADER.size:
        raise RuntimeError("Truncated ART routed-experts response header")
    magic, json_size, route_count = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise RuntimeError("Invalid ART routed-experts response magic")
    offset = HEADER.size
    json_end = offset + json_size
    if json_end > len(body):
        raise RuntimeError("Truncated ART routed-experts JSON response")
    response = ChatCompletion.model_validate_json(body[offset:json_end])
    offset = json_end
    routes: dict[int, np.ndarray] = {}
    for _ in range(route_count):
        if offset + ROUTE_HEADER.size > len(body):
            raise RuntimeError("Truncated ART routed-experts array header")
        choice_index, dtype_code, tokens, layers, topk = ROUTE_HEADER.unpack_from(
            body, offset
        )
        offset += ROUTE_HEADER.size
        dtype_name = DTYPES.get(dtype_code)
        if dtype_name is None:
            raise RuntimeError(f"Unknown ART route dtype code {dtype_code}")
        dtype = np.dtype(dtype_name)
        size = int(tokens * layers * topk * dtype.itemsize)
        end = offset + size
        if end > len(body):
            raise RuntimeError("Truncated ART routed-experts array")
        if choice_index in routes:
            raise RuntimeError(f"Duplicate routed experts for choice {choice_index}")
        array = np.frombuffer(
            body, dtype=dtype, count=tokens * layers * topk, offset=offset
        )
        routes[choice_index] = array.reshape((tokens, layers, topk))
        offset = end
    if offset != len(body):
        raise RuntimeError("Unexpected trailing bytes in ART routed-experts response")
    return response, routes
