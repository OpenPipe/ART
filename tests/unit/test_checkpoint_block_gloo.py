"""Real CPU finalizer collectives; only public metadata and shard merge are fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

import pytest
import safetensors
from safetensors.torch import load_file, save_file
from test_checkpoint_selective_read import (
    _dependencies,
    _eager,
    _files,
    _Meta,
    _prepared,
    _trainer,
)
import torch
import torch.distributed as dist

from art.trainer_rank import _checkpoint


@dataclass(frozen=True)
class _ShardMeta(_Meta):
    shard_rank: int = 0
    world: int = 2

    @property
    def manifest(self):
        return {
            "sharded": True,
            "shard_world_size": self.world,
            "shard_rank": self.shard_rank,
            "export_shard_strategy": "uniform",
            "export_shard_dim": 1,
        }


def test_block_close_real_gloo_fanout_and_cleanup(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    children, logs = [], []
    deadline = time.monotonic() + 90
    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [sys.executable, __file__, str(rank), str(tmp_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ
                    | {
                        "PYTHONPATH": os.pathsep.join(
                            (
                                str(Path(__file__).resolve().parents[2] / "src"),
                                os.environ.get("PYTHONPATH", ""),
                            )
                        ),
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTHONHASHSEED": "0",
                    },
                )
            )
        for index, child in enumerate(children):
            assert child.wait(timeout=max(0.1, deadline - time.monotonic())) == 0, (
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
    assert all(child.poll() == 0 for child in children)


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{root / 'rendezvous'}",
        timeout=timedelta(seconds=10),
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Reuse public merge/config fixtures, leaving group helpers untouched.
            _dependencies(patch)
            publish = sys.modules["art.megatron.weights.lora_publish"]

            def merge(entries):
                out = {}
                for key, parts in entries.items():
                    if parts[0][0]["sharded"]:
                        assert [m["shard_rank"] for m, _ in parts] == [0, 1]
                        out[key] = torch.cat([value for _, value in parts], dim=1)
                    else:
                        assert len(parts) == 1
                        out[key] = parts[0][1]
                return out

            patch.setattr(publish, "merge_sharded_adapter_entries", merge)
            outputs = []
            for case in (
                "eager",
                "reuse",
                "eager-cp",
                "reuse-cp",
                "read",
                "close",
                "read-close",
                "cancel-close",
            ):
                prepared = _prepared(root / case / f"rank{rank}")
                prepared.reservation.rmdir()
                if rank == 0:
                    (root / case / "reservation").mkdir()
                prepared = replace(
                    prepared,
                    destination=root / case / "result",
                    reservation=root / case / "reservation",
                    shards=tuple(
                        _checkpoint._LocalShard(
                            cast(
                                Any,
                                _ShardMeta(
                                    shard.metadata.key,
                                    shard.metadata.block,
                                    shard.metadata.shape,
                                    shard.metadata.dtype_name,
                                    owner_rank=rank,
                                    shard_rank=rank,
                                ),
                            ),
                            shard.file,
                        )
                        for shard in prepared.shards
                    ),
                )
                if case.endswith("cp"):
                    # CP replicas share a shard identity. Only owner 0 may be read.
                    prepared = replace(
                        prepared,
                        shards=tuple(
                            _checkpoint._LocalShard(
                                cast(
                                    Any,
                                    _Meta(
                                        s.metadata.key,
                                        s.metadata.block,
                                        s.metadata.shape,
                                        s.metadata.dtype_name,
                                        owner_rank=rank,
                                    ),
                                ),
                                "never-open" if rank else s.file,
                            )
                            for s in prepared.shards
                        ),
                    )
                if rank:
                    for path in prepared.snapshot.iterdir():
                        save_file(
                            {
                                k: v if k.startswith("step/") else v + 1
                                for k, v in load_file(path).items()
                            },
                            path,
                        )
                dist.barrier()
                trainer = _trainer(prepared)
                primary = (
                    asyncio.CancelledError("rank1 cancelled read")
                    if case == "cancel-close"
                    else KeyError("rank1 read")
                )
                close = OSError("rank1 close")
                real = safetensors.safe_open
                handles = []

                class Reader:
                    def __init__(self, *args, **kwargs):
                        self.inner = real(*args, **kwargs)
                        self.name = Path(args[0]).name

                    def __enter__(self):
                        self.inner.__enter__()
                        handles.append(self)
                        return self

                    def __exit__(self, *args):
                        try:
                            self.inner.__exit__(*args)
                        finally:
                            handles.remove(self)
                        if (
                            rank == 1
                            and "close" in case
                            and self.name == "block0.safetensors"
                        ):
                            raise close

                    def offset_keys(self):
                        return self.inner.offset_keys()

                    def keys(self):
                        return self.inner.keys()

                    def get_tensor(self, key):
                        if (
                            rank == 1
                            and ("read" in case or "cancel" in case)
                            and key.startswith("master/")
                        ):
                            raise primary
                        return self.inner.get_tensor(key)

                error = None
                with pytest.MonkeyPatch.context() as reading:
                    reading.setattr(safetensors, "safe_open", Reader)
                    if case.startswith("eager"):
                        reading.setattr(_checkpoint, "_read_snapshot", _eager)
                    try:
                        _checkpoint.finish_checkpoint_save(
                            trainer, str(prepared.destination)
                        )
                    except BaseException as exc:
                        error = exc
                success = case.startswith(("eager", "reuse"))
                if success:
                    assert error is None
                    if rank == 0:
                        outputs.append(_files(prepared.destination))
                else:
                    assert error is not None
                    if rank == 1:
                        assert error is (close if case == "close" else primary)
                    else:
                        assert isinstance(
                            error, RuntimeError
                        ) and "Another rank failed" in str(error)
                    if case in ("read-close", "cancel-close"):
                        assert any(
                            "snapshot close also failed" in note.lower()
                            for note in error.__notes__
                        )
                    assert not prepared.destination.exists()
                assert not handles
                assert (
                    not prepared.snapshot.exists() and not prepared.reservation.exists()
                )
                assert (
                    not trainer._prepared_checkpoint_saves
                    and not trainer._checkpoint_finalizing_saves
                )
                assert trainer._finalized_checkpoint_saves[
                    str(prepared.destination)
                ].outcome == ("finish" if success else "abort")
                assert not list((root / case).glob(".result.tmp-*"))
                assert (
                    dist.get_backend(trainer._checkpoint_finalize_process_group)
                    == "gloo"
                )
                dist.destroy_process_group(trainer._checkpoint_finalize_process_group)
                dist.destroy_process_group(trainer._checkpoint_process_group)
                dist.barrier()
            if rank == 0:
                assert len(outputs) == 4
                assert outputs[0] == outputs[1] and outputs[2] == outputs[3]
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), Path(sys.argv[2]))
