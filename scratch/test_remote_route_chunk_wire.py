from __future__ import annotations

import asyncio
import hashlib
import struct

import numpy as np
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
import pytest

from art.distributed.moe_route_store import hydrate_trajectory_group_routes
from art.distributed.packing import _ChoiceRoutingPayload
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.openai import ART_MOE_ROUTING_METADATA_KEY
from art.preprocessing.moe_routing import (
    MoeRouteArray,
    MoeRouteSegments,
    choice_moe_routing_metadata,
)
from art.serverless.contracts import RemoteForwardRequest
from art.serverless.data_plane import (
    FORWARD_SUBMISSION_CHUNK_BYTES,
    FORWARD_SUBMISSION_PREFIX_BYTES,
    EncodedRouteObject,
    decode_forward_submission_manifest,
    decode_forward_submission_prefix,
    decode_trajectory_group,
    encode_forward_submission,
    encode_trajectory_group,
    prepare_training_batch,
)
from art.training.contracts import LossConfig, RlTrajectoryBatch
from art.trajectories import Trajectory, TrajectoryGroup
from art.vllm_route_transport import (
    HEADER,
    MAGIC,
    MAX_JSON_BYTES,
    MAX_ROUTE_ARRAY_BYTES,
    ROUTE_HEADER,
    decode_routed_experts_response,
    decode_routed_experts_response_stream,
)

ROUTES = (
    (1, 2, 3, 4, 5),
    (1, 2, 3, 6, 7),
    (1, 2, 8, 9, 10),
    (1, 2, 8, 11, 12),
)
EXPECTED_SLICES = (
    (0, 0, 0, 2),
    (0, 1, 2, 1),
    (0, 2, 3, 2),
    (1, 0, 0, 2),
    (1, 1, 2, 1),
    (1, 2, 5, 2),
    (2, 0, 0, 2),
    (2, 1, 7, 1),
    (2, 2, 8, 2),
    (3, 0, 0, 2),
    (3, 1, 7, 1),
    (3, 2, 10, 2),
)


def _route(values: tuple[int, ...]) -> _ChoiceRoutingPayload:
    routes = MoeRouteArray(
        np.asarray(values, dtype=np.uint8).reshape(len(values), 1, 1),
        num_experts=16,
    )
    return _ChoiceRoutingPayload.from_metadata(
        {
            "prompt_token_ids": [10, 11, 12],
            "completion_token_ids": [20, 21],
            "num_experts": 16,
            "routed_experts": routes,
        }
    )


def _group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=[
                    Choice.model_validate(
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "x"},
                            ART_MOE_ROUTING_METADATA_KEY: _route(values).build(),
                        }
                    )
                ],
                reward=float(index),
            )
            for index, values in enumerate(ROUTES)
        ]
    )


def _route_values(group: TrajectoryGroup) -> tuple[tuple[int, ...], ...]:
    values = []
    for trajectory in group.trajectories:
        choice = trajectory.messages_and_choices[0]
        assert isinstance(choice, Choice)
        metadata = (choice.model_extra or {})[ART_MOE_ROUTING_METADATA_KEY]
        routes = metadata["routed_experts"]
        array = (
            np.concatenate(routes.segments)
            if isinstance(routes, MoeRouteSegments)
            else routes
        )
        values.append(tuple(map(int, array[:, 0, 0])))
    return tuple(values)


def _binary_route_body() -> bytes:
    response = ChatCompletion.model_validate(
        {
            "id": "response",
            "object": "chat.completion",
            "created": 1,
            "model": "model",
            "prompt_token_ids": [10, 11],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                    "token_ids": [12],
                }
            ],
        }
    )
    response_bytes = response.model_dump_json().encode()
    route_bytes = np.arange(12, dtype=np.uint8).tobytes()
    return (
        HEADER.pack(MAGIC, len(response_bytes), 1, 16)
        + response_bytes
        + ROUTE_HEADER.pack(0, 1, 3, 2, 2)
        + route_bytes
    )


async def _chunks(payload: bytes, width: int):
    for offset in range(0, len(payload), width):
        await asyncio.sleep(0)
        yield payload[offset : offset + width]


def test_trajectory_record_routes_decode_as_readonly_views() -> None:
    bundle = TrajectoryGroupBundle.from_group(_group())
    payload = bundle.payload()
    for record, trajectory in zip(bundle.records, payload.trajectories, strict=True):
        chunk = trajectory.choice_routing_metadata[0].data[0]
        assert isinstance(chunk, memoryview)
        assert chunk.readonly
        assert chunk.obj is record


def test_route_encoding_preserves_exact_bytes_slices_and_round_trip() -> None:
    encoded = encode_trajectory_group(
        TrajectoryGroupBundle.from_group(_group()), object_id="1" * 64
    )
    route = encoded.routes[0]
    route_bytes = b"".join(route.chunks)
    assert route_bytes == bytes(range(1, 13))
    assert route.ref.byte_count == len(route_bytes)
    assert route.ref.sha256 == hashlib.sha256(route_bytes).hexdigest()
    assert all(
        chunk.readonly and 0 < len(chunk) <= FORWARD_SUBMISSION_CHUNK_BYTES
        for chunk in route.chunks
    )
    assert (
        tuple(
            (item.trajectory_index, item.segment_index, item.offset, item.byte_count)
            for item in encoded.remote.routes[0].slices
        )
        == EXPECTED_SLICES
    )

    decoded = decode_trajectory_group(
        encoded.remote,
        encoded.data.payload,
        route_payloads={route.ref.object_id: route_bytes},
    )
    rebuilt = hydrate_trajectory_group_routes(
        decoded.bundle.payload(), decoded.routes
    ).build()
    assert _route_values(rebuilt) == ROUTES


def test_route_chunks_fail_closed_outside_wire_bounds() -> None:
    encoded = encode_trajectory_group(
        TrajectoryGroupBundle.from_group(_group()), object_id="1" * 64
    )
    route = encoded.routes[0]
    with pytest.raises(ValueError, match="bounded readonly"):
        EncodedRouteObject(
            ref=route.ref,
            chunks=(memoryview(bytearray(route.ref.byte_count)),),
        )
    with pytest.raises(ValueError, match="bounded readonly"):
        EncodedRouteObject(
            ref=route.ref.model_copy(
                update={"byte_count": FORWARD_SUBMISSION_CHUNK_BYTES + 1}
            ),
            chunks=(memoryview(bytes(FORWARD_SUBMISSION_CHUNK_BYTES + 1)),),
        )


@pytest.mark.asyncio
async def test_forward_submission_stream_is_byte_identical_and_bounded() -> None:
    batch = RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle.from_group(_group()),),
        min_source_version=0,
        max_source_version=0,
    )
    encoded = prepare_training_batch(batch, identity="batch")
    request = RemoteForwardRequest(
        run_id="run",
        request_id="request",
        sequence_id=0,
        batch=encoded.remote,
        loss=LossConfig(name="ppo"),
    )
    submission = encode_forward_submission(request, encoded)
    streamed_chunks = tuple([chunk async for chunk in submission.stream()])
    wire = b"".join(streamed_chunks)
    expected = (
        submission.preamble
        + b"".join(value.payload for value in encoded.objects)
        + b"".join(chunk for value in encoded.route_objects for chunk in value.chunks)
    )
    assert wire == expected
    assert len(wire) == submission.byte_count
    assert all(
        isinstance(chunk, memoryview)
        and chunk.readonly
        and len(chunk) <= FORWARD_SUBMISSION_CHUNK_BYTES
        for chunk in streamed_chunks
    )

    manifest_bytes = decode_forward_submission_prefix(
        wire[:FORWARD_SUBMISSION_PREFIX_BYTES]
    )
    manifest_end = FORWARD_SUBMISSION_PREFIX_BYTES + manifest_bytes
    manifest = decode_forward_submission_manifest(
        wire[FORWARD_SUBMISSION_PREFIX_BYTES:manifest_end]
    )
    assert manifest.request == request
    assert manifest.objects == tuple(value.ref for value in encoded.objects)
    assert manifest.route_objects == tuple(value.ref for value in encoded.route_objects)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [1, 7, 64])
async def test_streaming_vllm_decode_is_byte_identical(width: int) -> None:
    body = _binary_route_body()
    expected_response, expected_routes = decode_routed_experts_response(body)
    response, routes = await decode_routed_experts_response_stream(_chunks(body, width))
    assert response.model_dump(mode="python") == expected_response.model_dump(
        mode="python"
    )
    np.testing.assert_array_equal(routes[0], expected_routes[0])
    assert not routes[0].flags.writeable


@pytest.mark.asyncio
async def test_openai_proxy_uses_streaming_binary_response() -> None:
    from art.model import _OpenAIChatCompletionsProxy

    class _RawResponse:
        def iter_bytes(self):
            return _chunks(_binary_route_body(), 5)

    class _ResponseContext:
        exited = False

        async def __aenter__(self):
            return _RawResponse()

        async def __aexit__(self, *_args):
            self.exited = True

    context = _ResponseContext()

    class _StreamingResponse:
        def create(self, *_args, **_kwargs):
            return context

    class _BinaryCompletions:
        with_streaming_response = _StreamingResponse()

        @property
        def with_raw_response(self):
            raise AssertionError("raw-response buffering must not be used")

    proxy = _OpenAIChatCompletionsProxy(
        object(),
        lambda _response: None,
        binary_completions=_BinaryCompletions(),
    )
    response = await proxy.create(model="model", messages=[])
    metadata = choice_moe_routing_metadata(response.choices[0])
    assert metadata is not None
    np.testing.assert_array_equal(
        metadata["routed_experts"], np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    )
    assert context.exited


@pytest.mark.asyncio
async def test_streaming_vllm_decode_rejects_bounds_and_trailing_bytes() -> None:
    oversized_json = HEADER.pack(MAGIC, MAX_JSON_BYTES + 1, 0, 16)
    with pytest.raises(RuntimeError, match="JSON size"):
        await decode_routed_experts_response_stream(_chunks(oversized_json, 3))
    with pytest.raises(RuntimeError, match="trailing bytes"):
        await decode_routed_experts_response_stream(
            _chunks(_binary_route_body() + b"x", 5)
        )
    body = _binary_route_body()
    _, json_size, _, _ = HEADER.unpack_from(body)
    response_end = HEADER.size + json_size
    oversized_route = (
        HEADER.pack(MAGIC, json_size, 1, 16)
        + body[HEADER.size : response_end]
        + ROUTE_HEADER.pack(0, 1, MAX_ROUTE_ARRAY_BYTES + 1, 1, 1)
    )
    with pytest.raises(RuntimeError, match="array exceeds"):
        await decode_routed_experts_response_stream(_chunks(oversized_route, 11))
