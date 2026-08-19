from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time
from typing import Any


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * 4096


class _PeakRss:
    def __init__(self) -> None:
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "_PeakRss":
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_bytes())

    def _sample(self) -> None:
        while not self._stop.wait(0.001):
            self.peak = max(self.peak, _rss_bytes())


def _build_record_backed_route(byte_count: int):
    from art.distributed.packing import (
        TrajectoryGroupPayload,
        TrajectoryPayload,
        _ChoiceRoutingPayload,
    )
    from art.distributed.trajectory_store import TrajectoryGroupBundle

    source = b"\x5a" * byte_count
    route = _ChoiceRoutingPayload.model_construct(
        metadata={
            "prompt_token_ids": [1],
            "completion_token_ids": [],
            "num_experts": 256,
        },
        dtype="uint8",
        shape=(1, byte_count, 1),
        data=(source,),
    )
    trajectory = TrajectoryPayload.model_construct(
        payload={},
        choice_positions=(),
        additional_history_choice_positions=(),
        choice_routing_metadata={0: route},
        additional_history_choice_routing_metadata=(),
        exchange_choice_routing_metadata=(),
    )
    payload = TrajectoryGroupPayload.model_construct(
        trajectories=(trajectory,),
        exceptions=(),
        metadata={},
        metrics={},
        logs=(),
        collect_packing_shape=False,
    )
    bundle = TrajectoryGroupBundle.from_payload(payload)
    del source, route, trajectory, payload
    gc.collect()
    decoded = bundle.payload()
    route = decoded.trajectories[0].choice_routing_metadata[0]
    chunk = route.data[0]
    if not isinstance(chunk, memoryview) or chunk.obj is not bundle.records[0]:
        raise RuntimeError("trajectory route decode did not retain its record view")
    return bundle, decoded, route


def _run(mode: str, mib: int) -> dict[str, Any]:
    from art.serverless.data_plane import _pack_route_sequences

    byte_count = mib << 20
    bundle, decoded, route = _build_record_backed_route(byte_count)
    del decoded
    gc.collect()
    started = time.perf_counter()
    with _PeakRss() as rss:
        if mode == "legacy":
            decoded_copy = bytes(route.data[0])
            route = route.model_copy(update={"data": (decoded_copy,)})
        packed = _pack_route_sequences(
            ((route, (1,)),), compute_sha256=mode == "chunked"
        )
        if mode == "legacy":
            accumulator = bytearray()
            for chunk in packed.chunks:
                accumulator.extend(chunk)
            wire = bytes(accumulator)
            digest = hashlib.sha256(wire).hexdigest()
            wire_chunks = 1
            whole_route_copies = 3
        else:
            wire = None
            digest = packed.sha256
            wire_chunks = len(packed.chunks)
            whole_route_copies = 0
    elapsed = time.perf_counter() - started
    if packed.byte_count != byte_count or digest is None:
        raise RuntimeError("route benchmark produced the wrong byte count")
    del bundle
    return {
        "mode": mode,
        "route_mib": mib,
        "route_bytes": byte_count,
        "wire_chunks": wire_chunks,
        "whole_route_copies_after_record": whole_route_copies,
        "sha256": digest,
        "slices": packed.slices,
        "elapsed_s": round(elapsed, 6),
        "baseline_rss_mib": round(rss.baseline / (1 << 20), 3),
        "peak_rss_mib": round(rss.peak / (1 << 20), 3),
        "incremental_peak_rss_mib": round((rss.peak - rss.baseline) / (1 << 20), 3),
    }


def _compare(mib: int) -> None:
    results = []
    for mode in ("legacy", "chunked"):
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                mode,
                "--mib",
                str(mib),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout))
    legacy, chunked = results
    if legacy["sha256"] != chunked["sha256"] or legacy["slices"] != chunked["slices"]:
        raise RuntimeError("legacy and chunked route bytes differ")
    print(
        json.dumps(
            {
                "route_mib": mib,
                "byte_identical": True,
                "legacy": legacy,
                "chunked": chunked,
                "incremental_peak_reduction_mib": round(
                    legacy["incremental_peak_rss_mib"]
                    - chunked["incremental_peak_rss_mib"],
                    3,
                ),
                "eliminated_whole_route_copies": (
                    legacy["whole_route_copies_after_record"]
                    - chunked["whole_route_copies_after_record"]
                ),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=276)
    parser.add_argument(
        "--mode", choices=("compare", "legacy", "chunked"), default="compare"
    )
    args = parser.parse_args()
    if args.mib < 1:
        raise ValueError("--mib must be positive")
    if args.mode == "compare":
        _compare(args.mib)
    else:
        print(json.dumps(_run(args.mode, args.mib)))


if __name__ == "__main__":
    main()
