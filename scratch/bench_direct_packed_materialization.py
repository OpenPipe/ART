from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from art.distributed.data_plane import (
    SharedMemoryPackedBatchStore,
    packed_plan_storage_byte_count,
)
from art.megatron.prefix_tree_packing import PrefixTreePackSegment
from art.preprocessing.pack import (
    PrefixTreePackingPlan,
    _PrefixTreePackItem,
    _PrefixTreeRowPlan,
    materialize_packed_tensors,
    materialize_packed_tensors_into,
)


def _plan(rows: int, sequence_length: int) -> PrefixTreePackingPlan:
    token_ids = tuple(range(sequence_length))
    item = _PrefixTreePackItem(
        token_ids=token_ids,
        sharing_ids=token_ids,
        input_pos=np.arange(sequence_length, dtype=np.int64),
        assistant_mask=np.ones(sequence_length, dtype=np.bool_),
        logprobs=np.full(sequence_length, -0.1, dtype=np.float32),
        advantage=1.0,
        weight=1.0,
        prompt_id=0,
        shareable_length=0,
        pixel_values=None,
        image_grid_thw=None,
        moe_routes=None,
        policy_versions=np.full(sequence_length, -1, dtype=np.int64),
    )
    row_plan = _PrefixTreeRowPlan(
        segments=(
            PrefixTreePackSegment(
                sequence_indices=(0,),
                start=0,
                end=sequence_length,
                packed_start=0,
                group_id=1,
                parent_id=1,
            ),
        ),
        length=sequence_length,
    )
    return PrefixTreePackingPlan(
        items=[item] * rows,
        planned_rows=[([item], row_plan) for _ in range(rows)],
        sequence_length=sequence_length,
        include_moe_routing=False,
    )


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is absent from /proc/self/status")


def _run(mode: str, rows: int, sequence_length: int) -> None:
    plan = _plan(rows, sequence_length)
    payload_bytes = packed_plan_storage_byte_count(plan)
    store = SharedMemoryPackedBatchStore(
        owner_actor_id="benchmark", capacity_bytes=payload_bytes
    )
    gc.collect()
    rss_before = _rss_bytes()
    started = time.perf_counter()
    if mode == "legacy":
        packed = materialize_packed_tensors(plan)
        ref = store.create(packed, batch_id=mode)
    else:
        writer = store.reserve_plan(plan, batch_id=mode)
        tensors = writer.tensors
        assert tensors is not None
        writer.begin()
        materialized = False
        try:
            materialize_packed_tensors_into(plan, tensors)
            materialized = True
        finally:
            writer.finish(success=materialized)
        ref = store.commit_plan(writer)
    elapsed_s = time.perf_counter() - started
    stats = store.stats()
    result = {
        "mode": mode,
        "payload_bytes": ref.byte_count,
        "storage_bytes": ref.storage_byte_count,
        "elapsed_s": elapsed_s,
        "hot_rss_delta_bytes": _rss_bytes() - rss_before,
        "peak_rss_delta_bytes": max(
            0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss_before
        ),
        "data_plane_copied_bytes": stats.copied_bytes,
        "data_plane_copy_count": stats.copy_count,
    }
    print(json.dumps(result))
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "direct"))
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if args.mode is not None:
        _run(args.mode, args.rows, args.sequence_length)
        return
    records = []
    for mode in ("legacy", "direct"):
        completed = subprocess.run(
            [
                sys.executable,
                os.fspath(Path(__file__).resolve()),
                "--mode",
                mode,
                "--rows",
                str(args.rows),
                "--sequence-length",
                str(args.sequence_length),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        records.append(json.loads(completed.stdout))
    print(
        json.dumps(
            {
                "model": {
                    "legacy_live_packed_payloads": 2,
                    "direct_live_packed_payloads": 1,
                    "legacy_data_plane_copy_bytes": "P",
                    "direct_data_plane_copy_bytes": 0,
                },
                "measurements": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
