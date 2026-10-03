"""A local body failure must unwind before any new collective on reader close."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

import pytest
import safetensors
from test_checkpoint_selective_read import _dependencies, _Meta, _prepared, _trainer
import torch
import torch.distributed as dist

from art.trainer_rank import _checkpoint


@pytest.mark.parametrize("scope", ["context", "finalize"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_asymmetric_body_error_reaches_local_cleanup(tmp_path, scope, close_fails):
    if not dist.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    children, logs = [], []
    completed = tmp_path / "rank1-cleaned.json"
    ready = tmp_path / "rank1-ready"
    peer_ready = tmp_path / "rank0-recv"
    try:
        for rank in range(2):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        str(rank),
                        str(tmp_path),
                        scope,
                        str(int(close_fails)),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ
                    | {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
            )
        deadline = time.monotonic() + 35
        while not ready.exists() and time.monotonic() < deadline:
            if any(child.poll() is not None for child in children):
                break
            time.sleep(0.02)
        assert ready.exists(), [p.read_text() for p in tmp_path.glob("*.log")]
        deadline = time.monotonic() + 8
        while (
            not (completed.exists() and peer_ready.exists())
            and time.monotonic() < deadline
        ):
            if children[1].poll() is not None and not completed.exists():
                break
            time.sleep(0.02)
        assert completed.exists(), (
            "rank1 could not reach local cleanup before collective coordination",
            [p.read_text() for p in tmp_path.glob("*.log")],
        )
        result = json.loads(completed.read_text())
        assert result["primary_identity"] and result["readers_closed"]
        assert result["close_collectives"] == 0
        assert peer_ready.exists()
        assert result["local_close_note"] == close_fails
        if scope == "finalize":
            assert result["outcome"] == "abort"
            assert result["queue_next"] == 1
            assert result["snapshot_removed"]
            assert result["boundary"] == "before inherited finalization gather"
            # The peer is still in recv. We observe cleanup, not recovery of the
            # pre-existing asymmetric finalization protocol; parent owns shutdown.
        else:
            for child in children:
                assert child.wait(timeout=10) == 0
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
        (tmp_path / "process-closure.json").write_text(
            json.dumps(
                [
                    {"pid": child.pid, "returncode": child.returncode}
                    for child in children
                ]
            )
        )
    assert all(child.poll() is not None for child in children)


def _worker(rank, root, scope, close_fails):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{root / 'rendezvous'}",
        timeout=timedelta(seconds=6),
    )
    primary = RuntimeError("rank1 local send failure")
    close = OSError("rank1 reader close failure")
    handles = []
    real_open = safetensors.safe_open
    real_gather = _checkpoint._gather
    real_finish = _checkpoint._finish
    real_send = dist.send
    real_recv = dist.recv
    real_raise = _checkpoint.raise_distributed
    identity = False
    close_collectives = 0
    failed_at = None

    class Reader:
        def __init__(self, *args, **kwargs):
            self.inner = real_open(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            handles.append(self)
            return self

        def __exit__(self, *args):
            try:
                return self.inner.__exit__(*args)
            finally:
                handles.remove(self)
                if rank == 1 and close_fails:
                    raise close

        def offset_keys(self):
            return self.inner.offset_keys()

        def get_tensor(self, key):
            return self.inner.get_tensor(key)

    def receipt(**extra):
        path = root / "rank1-cleaned.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "primary_identity": identity,
                    "close_collectives": close_collectives,
                    "elapsed_since_body_failure": time.monotonic() - failed_at,
                    "source_path": _checkpoint.__file__,
                    "source_sha256": hashlib.sha256(
                        Path(_checkpoint.__file__).read_bytes()
                    ).hexdigest(),
                    "primary_notes": getattr(primary, "__notes__", []),
                    "readers_closed": not handles,
                    "local_close_note": any(
                        "rank1 reader close failure" in note
                        for note in getattr(primary, "__notes__", [])
                    ),
                    **extra,
                }
            )
        )
        temporary.replace(path)

    def recv(*args, **kwargs):
        if rank == 0:
            # Exercise the valid ordering where rank1 unwinds and exits before
            # its peer enters recv; the parent must await both observations.
            time.sleep(0.1)
            (root / "rank0-recv").touch()
        return real_recv(*args, **kwargs)

    def report(error, phase, group=None):
        nonlocal close_collectives
        if phase == "close checkpoint snapshot block":
            close_collectives += 1
        return real_raise(error, phase, group)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(safetensors, "safe_open", Reader)
            patch.setattr(dist, "recv", recv)
            patch.setattr(_checkpoint, "raise_distributed", report)
            prepared = _prepared(root / f"rank{rank}")
            if scope == "context":
                if rank == 1:
                    (root / "rank1-ready").touch()
                dist.barrier()
                try:
                    with _checkpoint._snapshot_block(None) as block:
                        _checkpoint._read_snapshot(
                            prepared, "block0.safetensors", "lora", snapshot=block
                        )
                        if rank == 1:
                            failed_at = time.monotonic()
                            raise primary
                        dist.recv(torch.empty(1), src=1)
                except BaseException as error:
                    if rank == 1:
                        identity = error is primary
                        receipt()
                    else:
                        assert isinstance(error, RuntimeError)
                return

            _dependencies(patch)
            prepared = replace(
                prepared,
                destination=root / "result",
                shards=tuple(
                    _checkpoint._LocalShard(
                        cast(
                            Any,
                            _Meta(
                                s.metadata.key,
                                s.metadata.block,
                                s.metadata.shape,
                                s.metadata.dtype_name,
                                owner_rank=1,
                            ),
                        ),
                        s.file,
                    )
                    for s in prepared.shards
                )
                if rank == 1
                else (),
            )
            trainer = _trainer(prepared)

            def send(*args, **kwargs):
                nonlocal failed_at
                if rank == 1:
                    (root / "rank1-ready").touch()
                    failed_at = time.monotonic()
                    raise primary
                return real_send(*args, **kwargs)

            def finish(*args, **kwargs):
                nonlocal identity
                try:
                    return real_finish(*args, **kwargs)
                except BaseException as error:
                    identity = error is primary
                    raise

            def gather(value, group=None):
                if (
                    rank == 1
                    and isinstance(value, tuple)
                    and len(value) == 2
                    and value[0] == repr(primary)
                ):
                    receipt(
                        outcome=trainer._checkpoint_save_outcomes[
                            str(prepared.destination)
                        ],
                        queue_next=trainer._checkpoint_save_next,
                        snapshot_removed=not prepared.snapshot.exists(),
                        boundary="before inherited finalization gather",
                    )
                return real_gather(value, group)

            patch.setattr(dist, "send", send)
            patch.setattr(_checkpoint, "_finish", finish)
            patch.setattr(_checkpoint, "_gather", gather)
            _checkpoint.finish_checkpoint_save(trainer, str(prepared.destination))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])))
