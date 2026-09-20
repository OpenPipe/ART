"""Two real Gloo DP peers, native checkpoint gather, CPU admission facades."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HEAD = Path(__file__).resolve().parents[2] / "src"


@pytest.mark.parametrize("mode", ("fit", "both", "asymmetric"))
def test_native_checkpoint_gather_after_recovery(tmp_path, mode):
    selected = HEAD
    children = []
    logs = []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            env = os.environ | {
                "PYTHONPATH": str(selected),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            child = subprocess.Popen(
                [sys.executable, __file__, "worker", str(rank), mode, str(tmp_path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(child)
        for index, child in enumerate(children):
            assert child.wait(timeout=25) == 0, (
                tmp_path / f"rank-{index}.log"
            ).read_text()
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        for log in logs:
            log.close()
    rows = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    assert all(row["error"] is None for row in rows), rows
    assert all(row["barrier_error"] is None for row in rows), rows
    assert all(row["ensures"] == 1 for row in rows), rows


def worker(index, mode, directory):
    from datetime import timedelta
    import threading
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from trainer_rank_test_support import gloo_group

    from art.trainer_rank import TrainerRank, _impl

    torch.set_num_threads(1)
    with gloo_group(index, f"file://{directory}/rendezvous", timeout=3):
        rank = TrainerRank.__new__(TrainerRank)
        rank.device = torch.device("cpu")
        rank._checkpoint_mutation_lock = threading.RLock()
        rank._checkpoint_prefetch_lock = threading.Lock()
        rank._checkpoint_slots = {}
        rank._checkpoint_group_lock = threading.Lock()
        # Native _ensure_checkpoint_slots uses these all-rank groups unchanged.
        rank._checkpoint_process_group = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=3)
        )
        rank._checkpoint_finalize_process_group = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=3)
        )
        groups = [
            dist.new_group([r], backend="gloo", timeout=timedelta(seconds=3))
            for r in range(2)
        ]
        rank._forward_memory_group = lambda: groups[index]
        plan = SimpleNamespace(
            packed_tokens=1,
            logical_tokens=1,
            active_logical_tokens=1,
            grad_segment_count=0,
            output_bytes=0,
            signature=_impl._MemorySignature(
                topology=(2, 1, 1, 1),
                planner_coefficients=(0, None),
                slot_group_count=1,
                request_mix=("hidden_states",),
                grad_enabled=False,
                grad_modes=(False,),
            ),
        )
        rank._plan_flat_forward = lambda *args, **kwargs: plan
        rank._estimate_required_memory_bytes_from_values = lambda **kwargs: 80
        rank._snapshot_planning_telemetry = lambda *args: None
        reads = []
        ensures = []

        def available():
            reads.append(1)
            deficient = mode == "both" or (mode == "asymmetric" and index == 0)
            return 10 if deficient and len(reads) <= 2 else 200

        rank._available_memory_bytes = available
        native = rank._ensure_checkpoint_slots

        def ensure(values):
            ensures.append(1)
            return native(values)

        rank._ensure_checkpoint_slots = ensure
        request = SimpleNamespace(
            target_tokens=None,
            logits=False,
            top_k=None,
            hidden_states=True,
            checkpoint=None,
        )
        error = barrier_error = None
        try:
            rank._plan_admissible_forward([request], checkpoint=None, context="forward")
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            dist.barrier()
        except BaseException as exc:
            barrier_error = {"type": type(exc).__name__, "message": str(exc)}
        (directory / f"rank-{index}.json").write_text(
            json.dumps(
                {
                    "source": _impl.__file__,
                    "rank": index,
                    "mode": mode,
                    "ensures": len(ensures),
                    "samples": len(reads),
                    "error": error,
                    "barrier_error": barrier_error,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    worker(int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
