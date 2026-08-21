from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
from openai.types.chat.chat_completion import Choice
import pytest

from art.distributed import trajectory_store
from art.distributed.data_plane import ByteStreamTransfer
from art.distributed.moe_route_store import (
    MoeRouteGroupPayload,
    MoeRouteObjectBatchTransfer,
    MoeRouteObjectPayload,
    MoeRouteStoredObject,
)
from art.distributed.object_store import S3ObjectStoreConfig
from art.distributed.packing import TrajectoryGroupPayload
from art.distributed.trajectory_store import (
    TrajectoryBatchTransfer,
    TrajectoryGroupBundle,
    TrajectoryGroupLayout,
    TrajectoryRouteSequenceLayout,
)
from art.openai import ART_MOE_ROUTING_METADATA_KEY
from art.preprocessing.moe_routing import MoeRouteArray, MoeRouteSegments
from art.serverless import data_plane
from art.serverless.data_plane import (
    decode_trajectory_group,
    encode_trajectory_group,
    prepare_training_batch,
)
from art.training.contracts import ForwardBackwardRequest, LossConfig, RlTrajectoryBatch
from art.training.sequencing import _request_fingerprint
from art.trajectories import Trajectory, TrajectoryGroup


def _route(values: tuple[int, ...]) -> dict[str, object]:
    return {
        "prompt_token_ids": list(range(100, 100 + len(values))),
        "completion_token_ids": [],
        "num_experts": 128,
        "routed_experts": MoeRouteArray(
            np.asarray(values, dtype=np.uint8).reshape(len(values), 1, 1),
            num_experts=128,
        ),
    }


def _group(depth: int) -> TrajectoryGroup:
    common = tuple(range(1, depth + 1))
    routes = (
        common + tuple(range(32, 38 - depth)),
        common + tuple(range(64, 70 - depth)),
    )
    return TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=[
                    Choice.model_validate(
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": str(index)},
                            ART_MOE_ROUTING_METADATA_KEY: _route(values),
                        }
                    )
                ],
                reward=float(index),
            )
            for index, values in enumerate(routes)
        ]
    )


def _route_values(group: TrajectoryGroup) -> tuple[tuple[int, ...], ...]:
    values = []
    for trajectory in group.trajectories:
        choice = trajectory.messages_and_choices[0]
        assert isinstance(choice, Choice)
        routes = (choice.model_extra or {})[ART_MOE_ROUTING_METADATA_KEY][
            "routed_experts"
        ]
        array = (
            np.concatenate(routes.segments)
            if isinstance(routes, MoeRouteSegments)
            else routes
        )
        values.append(tuple(map(int, array[:, 0, 0])))
    return tuple(values)


def _stored_transfer(bundle: TrajectoryGroupBundle) -> MoeRouteObjectBatchTransfer:
    objects = tuple(
        MoeRouteStoredObject(
            object_id=hashlib.sha256(bytes(value.data)).hexdigest(),
            byte_count=len(value.data),
            slices=value.slices,
        )
        for value in bundle.route_group_payload().objects
    )
    return MoeRouteObjectBatchTransfer(
        tenant_id="tenant",
        run_id="run",
        store=S3ObjectStoreConfig(
            endpoint_url="https://objects.example.test",
            region="test",
            bucket="routes",
            prefix="training/routes",
        ),
        groups=(objects,),
    )


def test_local_bundle_creation_never_invokes_prefix_tree_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local trajectory storage planned a transport prefix tree")

    monkeypatch.setattr(data_plane, "prefix_tree_pack_segments", forbidden)
    bundle = TrajectoryGroupBundle.from_group(_group(5))
    assert len(bundle.route_sequences) == 2
    assert _route_values(bundle.build()) == _route_values(_group(5))


@pytest.mark.parametrize("depth", range(1, 6))
def test_remote_prefix_tree_routes_round_trip_at_depth(depth: int) -> None:
    group = _group(depth)
    bundle = TrajectoryGroupBundle.from_group(group)
    route_free = bundle.route_free_payload()
    assert all(
        not trajectory.choice_routing_metadata[0].data
        for trajectory in route_free.trajectories
    )
    assert sum(len(segment) for route in bundle.route_sequences for segment in route.data) == 12
    assert _route_values(bundle.build()) == _route_values(group)

    encoded = encode_trajectory_group(bundle, object_id="1" * 64)
    assert len(encoded.routes) == 1
    route = encoded.routes[0]
    assert route.ref.byte_count == 12 - depth
    data_bytes = b"".join(encoded.data.wire_chunks())
    decoded = decode_trajectory_group(
        encoded.remote,
        data_bytes,
        route_payloads={route.ref.object_id: b"".join(route.chunks)},
    )
    assert isinstance(decoded.bundle.header, memoryview)
    assert decoded.bundle.header.obj is data_bytes
    assert all(
        isinstance(record, memoryview) and record.obj is data_bytes
        for record in decoded.bundle.records
    )
    rebuilt = trajectory_store.hydrate_trajectory_group_routes(
        decoded.bundle.route_free_payload(), decoded.routes
    ).build()
    assert _route_values(rebuilt) == _route_values(group)


def test_remote_handoff_reuses_records_and_route_segments_without_model_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(5))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("remote handoff decoded or rebuilt trajectory payloads")

    monkeypatch.setattr(TrajectoryGroupBundle, "payload", forbidden)
    monkeypatch.setattr(TrajectoryGroupBundle, "route_free_payload", forbidden)
    monkeypatch.setattr(TrajectoryGroupBundle, "from_payload", forbidden)
    monkeypatch.setattr(TrajectoryGroupPayload, "model_validate", forbidden)
    encoded = encode_trajectory_group(bundle, object_id="2" * 64)

    record_sources = (bundle.header, *bundle.records)
    route_sources = {
        segment for sequence in bundle.route_sequences for segment in sequence.data
    }
    assert tuple(chunk.obj for chunk in encoded.data.chunks) == record_sources
    assert all(chunk.obj in route_sources for chunk in encoded.routes[0].chunks)


def test_third_party_inline_routes_are_planned_once_at_remote_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    planner = data_plane.prefix_tree_pack_segments

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return planner(*args, **kwargs)

    monkeypatch.setattr(data_plane, "prefix_tree_pack_segments", counted)
    bundle = TrajectoryGroupBundle.from_group(_group(3))
    assert calls == 0
    encode_trajectory_group(bundle, object_id="3" * 64)
    assert calls == 1


def test_plain_inline_route_input_is_accepted_and_deduplicated_on_wire() -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(5))
    encoded = encode_trajectory_group(
        bundle, object_id="4" * 64, route_encoding="inline"
    )
    assert len(encoded.routes) == 1
    assert encoded.routes[0].ref.byte_count == 7


def test_first_party_route_refs_bypass_planning_and_command_route_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(5))
    transfer = _stored_transfer(bundle)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("first-party route references planned transport bytes")

    monkeypatch.setattr(data_plane, "prefix_tree_pack_segments", forbidden)
    batch = RlTrajectoryBatch.from_group_bundles(
        (bundle,),
        min_source_version=0,
        max_source_version=0,
        moe_route_object_transfer=transfer,
    )
    encoded = prepare_training_batch(batch)
    assert not encoded.route_objects
    assert all(
        route.ref.transport == "object_store"
        for route in encoded.remote.groups[0].routes
    )


def test_prepacked_prefix_tree_wire_routes_are_accepted_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(5))
    packed = encode_trajectory_group(bundle, object_id="5" * 64)
    route = packed.routes[0]
    route_group = MoeRouteGroupPayload(
        objects=(
            MoeRouteObjectPayload(
                data=b"".join(route.chunks), slices=packed.remote.routes[0].slices
            ),
        )
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an accepted prefix-tree route object was replanned")

    monkeypatch.setattr(data_plane, "prefix_tree_pack_segments", forbidden)
    batch = RlTrajectoryBatch.from_group_bundles(
        (bundle,),
        min_source_version=0,
        max_source_version=0,
        moe_route_groups=(route_group,),
    )
    encoded = prepare_training_batch(batch)
    assert len(encoded.route_objects) == 1
    assert b"".join(encoded.route_objects[0].chunks) == b"".join(route.chunks)


def test_training_object_identity_is_content_addressed_across_commands() -> None:
    def prepare(group: TrajectoryGroup):
        return prepare_training_batch(
            RlTrajectoryBatch.from_groups([group], default_source_version=0)
        )

    first = prepare(_group(5))
    repeated = prepare(_group(5))
    assert tuple(value.ref for value in first.objects) == tuple(
        value.ref for value in repeated.objects
    )
    assert tuple(value.ref for value in first.route_objects) == tuple(
        value.ref for value in repeated.route_objects
    )

    changed = _group(5)
    changed.trajectories[0].reward += 1.0
    changed_encoded = prepare(changed)
    assert changed_encoded.objects[0].ref.object_id != first.objects[0].ref.object_id


def test_route_free_identity_matches_original_immutable_bytes() -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(5))
    payload = b"".join((bundle.header, *bundle.records))
    encoded = encode_trajectory_group(bundle, object_id="6" * 64)

    assert bundle.route_free_identity.sha256 == hashlib.sha256(payload).hexdigest()
    assert bundle.route_free_identity.byte_count == len(payload)
    assert encoded.data.ref.sha256 == bundle.route_free_identity.sha256
    assert b"".join(encoded.data.wire_chunks()) == payload


def test_distributed_transfer_preserves_uncompressed_route_sequences() -> None:
    bundle = TrajectoryGroupBundle.from_group(_group(4))
    layout = TrajectoryGroupLayout(
        header_byte_count=len(bundle.header),
        record_byte_counts=tuple(map(len, bundle.records)),
        route_free_identity=bundle.route_free_identity,
        route_sequences=tuple(
            TrajectoryRouteSequenceLayout(
                trajectory_index=value.trajectory_index,
                scope=value.scope,
                scope_index=value.scope_index,
                choice_index=value.choice_index,
                dtype=value.dtype,
                shape=value.shape,
                token_ids=value.token_ids,
                segment_byte_counts=tuple(map(len, value.data)),
            )
            for value in bundle.route_sequences
        ),
    )
    chunks = (
        bundle.header,
        *bundle.records,
        *(
            segment
            for sequence in bundle.route_sequences
            for segment in sequence.data
        ),
    )
    payload = bytearray(b"".join(chunks))
    transfer = TrajectoryBatchTransfer(
        stream=ByteStreamTransfer(
            stream_id="stream",
            host="127.0.0.1",
            port=1,
            token="1" * 64,
            byte_count=len(payload),
        ),
        groups=(layout,),
    )
    restored = transfer._build_bundles(payload)[0]
    assert isinstance(restored.header, memoryview) and restored.header.obj is payload
    assert all(
        isinstance(record, memoryview) and record.obj is payload
        for record in restored.records
    )
    assert all(
        isinstance(segment, memoryview) and segment.obj is payload
        for sequence in restored.route_sequences
        for segment in sequence.data
    )
    assert restored.route_free_identity == bundle.route_free_identity
    assert _route_values(restored.build()) == _route_values(bundle.build())
    assert tuple(map(bytes, restored.route_sequences[0].data)) == bundle.route_sequences[0].data


def test_received_bundle_encoding_does_not_rescan_route_free_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = TrajectoryGroupBundle.from_group(TrajectoryGroup([Trajectory()]))
    payload = bytearray(b"".join((bundle.header, *bundle.records)))
    transfer = TrajectoryBatchTransfer(
        stream=ByteStreamTransfer(
            stream_id="stream",
            host="127.0.0.1",
            port=1,
            token="1" * 64,
            byte_count=len(payload),
        ),
        groups=(
            TrajectoryGroupLayout(
                header_byte_count=len(bundle.header),
                record_byte_counts=tuple(map(len, bundle.records)),
                route_free_identity=bundle.route_free_identity,
            ),
        ),
    )
    restored = transfer._build_bundles(payload)[0]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("received trajectory bytes were hashed again")

    monkeypatch.setattr(data_plane.hashlib, "sha256", forbidden)
    encoded = encode_trajectory_group(restored, object_id="7" * 64)

    assert encoded.data.ref.sha256 == bundle.route_free_identity.sha256
    assert all(chunk.obj is payload for chunk in encoded.data.chunks)


def test_command_identity_includes_separate_route_content() -> None:
    def request(group: TrajectoryGroup) -> ForwardBackwardRequest:
        return ForwardBackwardRequest(
            run_id="run",
            request_id="request",
            sequence_id=0,
            batch=RlTrajectoryBatch.from_groups([group], default_source_version=0),
            loss=LossConfig(name="ppo"),
        )

    first = _group(5)
    second = deepcopy(first)
    route = second.trajectories[1].messages_and_choices[0]
    assert isinstance(route, Choice)
    metadata = (route.model_extra or {})[ART_MOE_ROUTING_METADATA_KEY]
    metadata["routed_experts"][-1] = 7
    first_bundle = TrajectoryGroupBundle.from_group(first)
    second_bundle = TrajectoryGroupBundle.from_group(second)
    assert (first_bundle.header, first_bundle.records) == (
        second_bundle.header,
        second_bundle.records,
    )
    assert first_bundle.route_sequences != second_bundle.route_sequences
    assert _request_fingerprint(request(first)) != _request_fingerprint(request(second))
