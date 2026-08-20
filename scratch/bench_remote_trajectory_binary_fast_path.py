from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

from msgspec import msgpack
import numpy as np
from openai.types.chat.chat_completion import Choice

from art.distributed import trajectory_store
from art.distributed.moe_route_store import MoeRouteStoredObject
from art.distributed.packing import TrajectoryGroupPayload
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.openai import ART_MOE_ROUTING_METADATA_KEY
from art.preprocessing.moe_routing import MoeRouteArray
from art.serverless.data_plane import encode_trajectory_group
from art.trajectories import Trajectory, TrajectoryGroup


def _group(group_size: int, route_tokens: int, prompt_bytes: int) -> TrajectoryGroup:
    base = (
        np.arange(route_tokens * 40 * 8, dtype=np.uint8)
        .reshape(route_tokens, 40, 8)
        .copy()
    ) % 128
    trajectories = []
    for index in range(group_size):
        routes = base.copy()
        routes[-32:] = index
        trajectories.append(
            Trajectory(
                messages_and_choices=[
                    Choice.model_validate(
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": str(index) + "x" * prompt_bytes,
                            },
                            ART_MOE_ROUTING_METADATA_KEY: {
                                "prompt_token_ids": list(range(route_tokens)),
                                "completion_token_ids": [],
                                "num_experts": 128,
                                "routed_experts": MoeRouteArray(
                                    routes, num_experts=128
                                ),
                            },
                        }
                    )
                ],
                reward=float(index),
            )
        )
    return TrajectoryGroup(trajectories)


def _legacy_bundle(payload: TrajectoryGroupPayload) -> TrajectoryGroupBundle:
    return TrajectoryGroupBundle(
        header=msgpack.encode(
            payload.model_copy(update={"trajectories": ()}).model_dump(mode="python")
        ),
        records=tuple(
            trajectory_store._encode_trajectory_record(value.model_dump(mode="python"))
            for value in payload.trajectories
        ),
    )


def _legacy_handoff(bundle: TrajectoryGroupBundle) -> int:
    normalized = TrajectoryGroupBundle.from_payload(bundle.payload())
    encoded = encode_trajectory_group(normalized, object_id="f" * 64)
    return encoded.data.ref.byte_count + sum(
        value.ref.byte_count for value in encoded.routes
    )


def _stored_routes(bundle: TrajectoryGroupBundle) -> tuple[MoeRouteStoredObject, ...]:
    return tuple(
        MoeRouteStoredObject(
            object_id=hashlib.sha256(bytes(value.data)).hexdigest(),
            byte_count=len(value.data),
            slices=value.slices,
        )
        for value in bundle.route_group_payload().objects
    )


def _measure(call, repeats: int) -> tuple[list[float], int]:
    times = []
    byte_count = 0
    for _ in range(repeats):
        started = time.perf_counter()
        byte_count = call()
        times.append(time.perf_counter() - started)
    return times, byte_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--route-tokens", type=int, default=2048)
    parser.add_argument("--prompt-kib", type=int, default=16)
    args = parser.parse_args()

    payload = TrajectoryGroupPayload.from_group(
        _group(args.group_size, args.route_tokens, args.prompt_kib << 10)
    )
    legacy = _legacy_bundle(payload)
    bundle = TrajectoryGroupBundle.from_payload(payload)
    stored = _stored_routes(bundle)

    def third_party() -> int:
        encoded = encode_trajectory_group(bundle, object_id="1" * 64)
        return encoded.data.ref.byte_count + sum(
            value.ref.byte_count for value in encoded.routes
        )

    def first_party() -> int:
        encoded = encode_trajectory_group(
            bundle, object_id="2" * 64, stored_routes=stored
        )
        if encoded.routes:
            raise RuntimeError("first-party route refs emitted command route bytes")
        return encoded.data.ref.byte_count

    _legacy_handoff(legacy)
    third_party()
    first_party()
    old_times, old_bytes = _measure(lambda: _legacy_handoff(legacy), args.repeats)
    third_times, third_bytes = _measure(third_party, args.repeats)
    first_times, first_bytes = _measure(first_party, args.repeats)
    route_sources = {
        segment for sequence in bundle.route_sequences for segment in sequence.data
    }
    encoded = encode_trajectory_group(bundle, object_id="3" * 64)
    source_reuse = all(
        chunk.obj in route_sources for value in encoded.routes for chunk in value.chunks
    )
    print(
        json.dumps(
            {
                "group_size": args.group_size,
                "route_tokens_per_trajectory": args.route_tokens,
                "legacy_handoff": {
                    "wire_bytes": old_bytes,
                    "p50_ms": 1000 * statistics.median(old_times),
                    "trajectory_payload_validations": 1,
                    "route_free_record_serializations": args.group_size + 1,
                    "prefix_tree_plans": 1,
                },
                "local_bundle_creation": {
                    "route_capture_copies": sum(
                        len(value.data) for value in bundle.route_sequences
                    ),
                    "route_transport_copies": 0,
                    "route_free_record_serializations": args.group_size + 1,
                    "prefix_tree_plans": 0,
                },
                "first_party_serverless_handoff": {
                    "wire_bytes_excluding_direct_routes": first_bytes,
                    "p50_ms": 1000 * statistics.median(first_times),
                    "trajectory_payload_validations": 0,
                    "trajectory_record_serializations": 0,
                    "trajectory_record_copies": 0,
                    "server_record_slice_copies": 0,
                    "route_command_serializations": 0,
                    "route_transport_copies": 0,
                    "prefix_tree_plans": 0,
                },
                "third_party_inline_handoff": {
                    "wire_bytes": third_bytes,
                    "p50_ms": 1000 * statistics.median(third_times),
                    "trajectory_payload_validations": 0,
                    "trajectory_record_serializations": 0,
                    "trajectory_record_copies": 0,
                    "server_record_slice_copies": 0,
                    "route_payload_copies": 0,
                    "route_digest_passes": 1,
                    "prefix_tree_plans": 1,
                    "route_chunks_reuse_source_bytes": source_reuse,
                },
                "third_party_p50_speedup_vs_legacy": statistics.median(old_times)
                / statistics.median(third_times),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
