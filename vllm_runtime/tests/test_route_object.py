import asyncio

from art_vllm_runtime.binary_routes import routed_experts_object_chunks
import numpy as np
import pytest

from art.vllm_route_transport import decode_routed_experts_object_stream


def test_route_only_object_roundtrip_preserves_choices_and_exact_dtype() -> None:
    routes = {
        0: np.asarray([[[0, 255]]], dtype=np.uint8),
        3: np.asarray([[[12, 7]], [[6, 4]]], dtype=np.uint8),
    }
    encoded = routed_experts_object_chunks(routes, num_experts=256)

    async def chunks():
        for value in encoded:
            yield bytes(value)

    decoded = asyncio.run(decode_routed_experts_object_stream(chunks()))
    assert set(decoded) == {0, 3}
    assert decoded[0].num_experts == 256
    assert decoded[0].dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(decoded[0], routes[0])
    np.testing.assert_array_equal(decoded[3], routes[3])


def test_route_only_object_rejects_truncation_and_trailing_bytes() -> None:
    encoded = routed_experts_object_chunks(
        {0: np.zeros((2, 1, 1), dtype=np.uint16)}, num_experts=257
    )
    body = b"".join(encoded)

    async def truncated():
        yield body[:-1]

    with pytest.raises(RuntimeError, match="Truncated ART routed-experts array"):
        asyncio.run(decode_routed_experts_object_stream(truncated()))

    async def trailing():
        yield body + b"x"

    with pytest.raises(RuntimeError, match="Unexpected trailing bytes"):
        asyncio.run(decode_routed_experts_object_stream(trailing()))
