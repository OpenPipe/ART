from __future__ import annotations

import gc
import mmap
import tracemalloc

import numpy as np
import pytest
import torch

from art.preprocessing.moe_routing import MoeRouteSegments
from art.preprocessing.pack import packed_tensors_from_tokenized_datums
from art.serverless.data_plane import decode_tokenized_batch, prepare_training_batch
from art.training.contracts import TokenizedTrainingBatch
from art.training.tokenized import TokenizedDatum, TokenizedMoeRoutes


def _routed_batch(
    *,
    datum_count: int,
    token_count: int,
    shared_tokens: int,
    layers: int,
    topk: int,
) -> TokenizedTrainingBatch:
    bytes_per_token = layers * topk
    prefix = bytes([1]) * (shared_tokens * bytes_per_token)
    datums = []
    for index in range(datum_count):
        input_tokens = tuple(range(shared_tokens)) + tuple(
            10_000 * (index + 1) + offset
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
        restored.datums,
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
    print(
        {
            "packed_pool_bytes": packed_pool_bytes,
            "logical_route_bytes": logical_route_bytes,
            "decode_current_bytes": current_bytes,
            "decode_peak_bytes": peak_bytes,
            "allocation_bound": allocation_bound,
        }
    )


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
