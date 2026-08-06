from __future__ import annotations

import struct

import numpy as np
from openai.types.chat import ChatCompletion

from art.preprocessing.moe_routing import MoeRouteArray

MAGIC = b"ARTRTE2\0"
HEADER = struct.Struct("<8sQII")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
DTYPES = {1: np.dtype(np.uint8), 2: np.dtype("<u2")}


def is_routed_experts_response(body: bytes) -> bool:
    return body.startswith(MAGIC)


def decode_routed_experts_response(
    body: bytes,
) -> tuple[ChatCompletion, dict[int, MoeRouteArray]]:
    if len(body) < HEADER.size:
        raise RuntimeError("Truncated ART routed-experts response header")
    magic, json_size, route_count, num_experts = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise RuntimeError("Invalid ART routed-experts response magic")
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
        dtype = DTYPES.get(dtype_code)
        if dtype is None:
            raise RuntimeError(f"Unknown ART route dtype code {dtype_code}")
        size = int(tokens * layers * topk * dtype.itemsize)
        end = offset + size
        if end > len(body):
            raise RuntimeError("Truncated ART routed-experts array")
        if choice_index in routes:
            raise RuntimeError(f"Duplicate routed experts for choice {choice_index}")
        array = np.frombuffer(
            body, dtype=dtype, count=tokens * layers * topk, offset=offset
        )
        routes[choice_index] = MoeRouteArray(
            array.reshape((tokens, layers, topk)), num_experts=num_experts
        )
        offset = end
    if offset != len(body):
        raise RuntimeError("Unexpected trailing bytes in ART routed-experts response")
    return response, routes
