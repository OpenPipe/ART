from __future__ import annotations

import gc
import mmap
import tracemalloc

from msgspec import msgpack
import numpy as np
import pytest
import torch

from art.distributed.data_plane import ByteStreamTransfer
from art.distributed.packing import TokenizedBatchTransfer, encode_tokenized_batch
from art.preprocessing.moe_routing import MoeRouteSegments
from art.preprocessing.pack import packed_tensors_from_tokenized_datums
import art.serverless.data_plane as tokenized_data_plane
from art.serverless.data_plane import (
    decode_tokenized_batch,
    decode_trusted_tokenized_batch_wire,
    prepare_training_batch,
)
from art.training.contracts import (
    ForwardBackwardRequest,
    LossConfig,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum, TokenizedMoeRoutes


def _routed_batch(
    *,
    datum_count: int,
    token_count: int,
    shared_tokens: int,
    layers: int,
    topk: int,
    token_stride: int = 10_000,
) -> TokenizedTrainingBatch:
    bytes_per_token = layers * topk
    prefix = bytes([1]) * (shared_tokens * bytes_per_token)
    datums = []
    for index in range(datum_count):
        input_tokens = tuple(range(shared_tokens)) + tuple(
            token_stride * (index + 1) + offset
            for offset in range(token_count - shared_tokens)
        )
        route_bytes = prefix + bytes([index + 2]) * (
            (token_count - shared_tokens) * bytes_per_token
        )
        datums.append(
            TokenizedDatum(
                input_tokens=input_tokens,
                target_tokens=tuple((token + 1,) for token in input_tokens),
                weights=tuple(
                    (0.0 if position < shared_tokens else 1.0,)
                    for position in range(token_count)
                ),
                packing_group_id=0,
                moe_routes=TokenizedMoeRoutes(
                    num_experts=256,
                    dtype="uint8",
                    shape=(token_count, layers, topk),
                    data=(route_bytes,),
                ),
            )
        )
    return TokenizedTrainingBatch(datums=tuple(datums))


def _route_array(routes: TokenizedMoeRoutes) -> np.ndarray:
    built = routes.build()
    return (
        np.concatenate(built.segments) if isinstance(built, MoeRouteSegments) else built
    )


def _route_segments(routes: TokenizedMoeRoutes) -> tuple[np.ndarray, ...]:
    built = routes.build()
    return built.segments if isinstance(built, MoeRouteSegments) else (built,)


def test_route_pool_round_trip_shares_views_and_matches_canonical_packing() -> None:
    batch = _routed_batch(
        datum_count=8,
        token_count=64,
        shared_tokens=48,
        layers=4,
        topk=2,
    )
    encoded = prepare_training_batch(batch)
    restored = decode_tokenized_batch(encoded.remote.data, encoded.objects[0].payload)
    packing_batch = decode_trusted_tokenized_batch_wire(
        bytearray(encode_tokenized_batch(restored))
    )

    segments = tuple(
        segment
        for datum in restored.datums
        for segment in datum.moe_routes.data  # type: ignore[union-attr]
    )
    assert all(
        isinstance(segment, memoryview)
        and segment.readonly
        and segment.c_contiguous
        and segment.format == "B"
        for segment in segments
    )
    assert all(hash(segment) == hash(bytes(segment)) for segment in segments)
    assert len({id(segment.obj) for segment in segments}) == 1
    restored_routes = tuple(datum.moe_routes for datum in restored.datums)
    assert all(routes is not None for routes in restored_routes)
    assert len({id(routes.data[0]) for routes in restored_routes if routes}) == len(
        restored_routes
    )
    for actual, expected in zip(restored.datums, batch.datums, strict=True):
        assert actual.moe_routes is not None and expected.moe_routes is not None
        np.testing.assert_array_equal(
            _route_array(actual.moe_routes), _route_array(expected.moe_routes)
        )
        assert all(
            not segment.flags.writeable
            for segment in _route_segments(actual.moe_routes)
        )

    assert (
        prepare_training_batch(restored).objects[0].payload
        == encoded.objects[0].payload
    )
    bytes_backed = restored.model_copy(
        update={
            "datums": tuple(
                datum.model_copy(
                    update={
                        "moe_routes": datum.moe_routes.model_copy(
                            update={"data": tuple(map(bytes, datum.moe_routes.data))}
                        )
                    }
                )
                for datum in restored.datums
                if datum.moe_routes is not None
            )
        }
    )
    assert (
        prepare_training_batch(restored, route_encoding="inline").objects[0].payload
        == prepare_training_batch(bytes_backed, route_encoding="inline")
        .objects[0]
        .payload
    )

    expected_packed = packed_tensors_from_tokenized_datums(
        batch.datums,
        loss="cross_entropy",
        seq_len=1024,
        min_prefix_tree_shared_segment_length=1,
    )
    actual_packed = packed_tensors_from_tokenized_datums(
        packing_batch.datums,
        loss="cross_entropy",
        seq_len=1024,
        min_prefix_tree_shared_segment_length=1,
    )
    for name in (
        "tokens",
        "group_ids",
        "parent_ids",
        "input_pos",
        "target_tokens",
        "loss_weights",
    ):
        torch.testing.assert_close(actual_packed[name], expected_packed[name])
    torch.testing.assert_close(
        actual_packed["moe_routing_replay"].expert_indices,
        expected_packed["moe_routing_replay"].expert_indices,
    )


def test_route_pool_decode_allocation_is_bounded_by_packed_pool() -> None:
    datum_count = 24
    bytes_per_token = 256 << 10
    batch = _routed_batch(
        datum_count=datum_count,
        token_count=2,
        shared_tokens=1,
        layers=bytes_per_token // 8,
        topk=8,
    )
    encoded = prepare_training_batch(batch)
    packed_pool_bytes = (datum_count + 1) * bytes_per_token
    logical_route_bytes = datum_count * 2 * bytes_per_token

    gc.collect()
    tracemalloc.start()
    restored = decode_tokenized_batch(encoded.remote.data, encoded.objects[0].payload)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pool_objects = {
        id(segment.obj)
        for datum in restored.datums
        for segment in datum.moe_routes.data  # type: ignore[union-attr]
    }
    assert len(pool_objects) == 1
    assert len(restored.datums[0].moe_routes.data[0].obj) == packed_pool_bytes  # type: ignore[union-attr]
    allocation_bound = packed_pool_bytes + (2 << 20)
    assert current_bytes < allocation_bound
    assert peak_bytes < allocation_bound
    payload = encoded.objects[0].payload
    assert payload is not None
    del restored
    received = bytearray(payload)
    gc.collect()
    tracemalloc.start()
    packing_batch = decode_trusted_tokenized_batch_wire(received)
    packing_current_bytes, packing_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert packing_current_bytes < allocation_bound
    assert packing_peak_bytes < allocation_bound
    assert (
        len(
            {
                id(segment.obj)
                for datum in packing_batch.datums
                for segment in datum.moe_routes.data  # type: ignore[union-attr]
            }
        )
        == 1
    )
    print(
        {
            "packed_pool_bytes": packed_pool_bytes,
            "logical_route_bytes": logical_route_bytes,
            "decode_current_bytes": current_bytes,
            "decode_peak_bytes": peak_bytes,
            "packing_decode_current_bytes": packing_current_bytes,
            "packing_decode_peak_bytes": packing_peak_bytes,
            "allocation_bound": allocation_bound,
        }
    )


def test_remote_tokenized_wire_is_validated_once_and_never_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _routed_batch(
        datum_count=8,
        token_count=8192,
        shared_tokens=4940,
        layers=61,
        topk=3,
        token_stride=100_000,
    )
    encoded = prepare_training_batch(batch)
    compact = encoded.objects[0].payload
    assert compact is not None
    validation_payloads = []
    validate = tokenized_data_plane._validated_tokenized_wire

    def tracked_validate(payload: bytes):
        validation_payloads.append(payload)
        return validate(payload)

    monkeypatch.setattr(
        tokenized_data_plane, "_validated_tokenized_wire", tracked_validate
    )
    restored = decode_tokenized_batch(encoded.remote.data, compact)
    request = ForwardBackwardRequest(
        run_id="run",
        request_id="request",
        sequence_id=0,
        batch=restored,
        loss=LossConfig(name="cross_entropy"),
    )
    assert isinstance(request.batch, TokenizedTrainingBatch)
    runtime_wire = encode_tokenized_batch(request.batch)
    packing_batch = decode_trusted_tokenized_batch_wire(bytearray(runtime_wire))
    expanded = msgpack.encode(restored.model_dump(mode="python"))

    assert validation_payloads == [compact]
    assert runtime_wire is compact
    assert len(runtime_wire) / (1 << 20) == pytest.approx(6.5604, abs=0.001)
    assert len(expanded) / (1 << 20) == pytest.approx(12.5947, abs=0.001)
    assert packing_batch.datums == restored.datums
    assert (
        len(
            {
                id(segment.obj)
                for datum in packing_batch.datums
                for segment in datum.moe_routes.data  # type: ignore[union-attr]
            }
        )
        == 1
    )


@pytest.mark.parametrize("route_encoding", ["inline", "prefix_tree"])
def test_trusted_materializer_accepts_both_v2_route_encodings(
    route_encoding: str,
) -> None:
    batch = _routed_batch(
        datum_count=2,
        token_count=8,
        shared_tokens=4,
        layers=4,
        topk=2,
    )
    encoded = prepare_training_batch(batch, route_encoding=route_encoding)
    payload = encoded.objects[0].payload
    assert payload is not None
    validated = decode_tokenized_batch(encoded.remote.data, payload)
    trusted = decode_trusted_tokenized_batch_wire(bytearray(payload))

    assert trusted.datums == validated.datums


@pytest.mark.asyncio
async def test_tokenized_transfer_does_not_copy_received_bytearray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = bytearray(b"validated compact wire")
    expected = _routed_batch(
        datum_count=1,
        token_count=2,
        shared_tokens=1,
        layers=4,
        topk=2,
    )
    observed = []

    async def receive(*_args, **_kwargs) -> bytearray:
        return source

    def decode(payload):
        observed.append(payload)
        return expected

    monkeypatch.setattr("art.distributed.packing.receive_byte_stream", receive)
    monkeypatch.setattr(
        tokenized_data_plane, "decode_trusted_tokenized_batch_wire", decode
    )
    transfer = TokenizedBatchTransfer(
        stream=ByteStreamTransfer(
            stream_id="batch",
            host="127.0.0.1",
            port=1,
            token="0" * 64,
            byte_count=len(source),
        )
    )

    assert await transfer.receive(timeout_s=1.0) is expected
    assert observed == [source]
    assert observed[0] is source


@pytest.mark.parametrize(
    "segment",
    [
        memoryview(bytearray(8)),
        memoryview(bytes(16))[::2],
        memoryview(bytes(8)).cast("H").toreadonly(),
    ],
)
def test_tokenized_routes_reject_non_readonly_contiguous_byte_views(
    segment: memoryview,
) -> None:
    with pytest.raises(ValueError, match="readonly contiguous bytes"):
        TokenizedMoeRoutes(
            num_experts=16,
            dtype="uint8",
            shape=(1, 4, 2),
            data=(segment,),
        )


def test_tokenized_routes_reject_readonly_view_of_mutable_owner() -> None:
    owner = bytearray(8)
    segment = memoryview(owner).toreadonly()
    assert segment.readonly and segment.c_contiguous and segment.format == "B"

    with pytest.raises(ValueError, match="bytes backing"):
        TokenizedMoeRoutes(
            num_experts=16,
            dtype="uint8",
            shape=(1, 4, 2),
            data=(segment,),
        )


def test_tokenized_routes_reject_hashable_file_backing(tmp_path) -> None:
    path = tmp_path / "routes.bin"
    path.write_bytes(bytes(8))
    with path.open("rb") as stream:
        owner = mmap.mmap(stream.fileno(), 8, access=mmap.ACCESS_READ)
        segment = memoryview(owner)
        try:
            assert hash(segment) == hash(bytes(segment))
            with pytest.raises(ValueError, match="bytes backing"):
                TokenizedMoeRoutes(
                    num_experts=16,
                    dtype="uint8",
                    shape=(1, 4, 2),
                    data=(segment,),
                )
        finally:
            segment.release()
            owner.close()


def test_tokenized_routes_own_memoryview_lifetime() -> None:
    owner = bytes(8)
    caller_view = memoryview(owner)
    routes = TokenizedMoeRoutes(
        num_experts=16,
        dtype="uint8",
        shape=(1, 4, 2),
        data=(caller_view,),
    )

    assert routes.data[0] is not caller_view
    assert isinstance(routes.data[0], memoryview)
    assert routes.data[0].obj is owner
    caller_view.release()

    np.testing.assert_array_equal(routes.build(), np.zeros((1, 4, 2), dtype=np.uint8))
