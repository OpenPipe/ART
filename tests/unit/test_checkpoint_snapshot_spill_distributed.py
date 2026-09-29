"""Real Gloo ownership checks for asymmetric CPU snapshot persistence failures."""

from datetime import timedelta
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import safetensors.torch
from test_trainer_rank_validation import _save_state_trainer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank._impl import _CheckpointSlot, _CustomObject


def _worker(rank, directory, failure, action, prepared, released, finalized):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{directory}/gloo",
        timeout=timedelta(seconds=15),
    )
    trainer = _save_state_trainer()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    trainer._checkpoint_slots["a"] = _CheckpointSlot(
        params=(parameter,),
        config={
            "base_model_name_or_path": "test/model",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
        },
        custom={"p": _CustomObject("parameter", parameter, object())},
    )
    output = str(Path(directory) / "failed")
    original_write, original_start = safetensors.torch.save_file, threading.Thread.start
    original_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        if rank == 0 and failure == "mkdir" and ".snapshot-r" in path.name:
            raise OSError("rank zero writer directory failed")
        return original_mkdir(path, *args, **kwargs)

    def write(tensors, path):
        if rank == 0 and failure == "write":
            raise OSError("rank zero writer failed")
        assert released.wait(15), "parent did not release the sibling writer"
        # The queued training mutation after capture must not reach the writer.
        torch.testing.assert_close(tensors["p"], torch.tensor([1.0]))
        original_write(tensors, path)

    def start(thread):
        if rank == 0 and failure == "start" and thread.name == "checkpoint-snapshot":
            raise RuntimeError("rank zero writer could not start")
        return original_start(thread)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(trainer, "_slot_ref", lambda _: None)
            patch.setitem(
                sys.modules,
                "art.megatron.lora",
                SimpleNamespace(LoRA=type("UnusedLoRA", (), {})),
            )
            patch.setitem(
                sys.modules,
                "art.megatron.weights.lora_publish",
                SimpleNamespace(collect_local_lora_entries=lambda *a, **kw: ({}, [])),
            )
            patch.setattr(safetensors.torch, "save_file", write)
            patch.setattr(threading.Thread, "start", start)
            patch.setattr(Path, "mkdir", mkdir)
            trainer.prepare_checkpoint_save(output, "a")
            owned = trainer._prepared_checkpoint_saves[output]
            assert owned.writer is not None
            with torch.no_grad():
                parameter.add_(100)
            prepared[rank].set()
            if action == "finish":
                with pytest.raises((OSError, RuntimeError), match="writer"):
                    trainer.finish_checkpoint_save(output)
            else:
                trainer.abort_checkpoint_save(output)
            assert not trainer._prepared_checkpoint_saves
            assert not owned.snapshot.exists()
            assert not owned.reservation.exists()
            assert trainer._checkpoint_save_next == 1
            finalized[rank].set()

            # Restore successful persistence, then reuse both collective groups.
            patch.setattr(safetensors.torch, "save_file", original_write)
            patch.setattr(threading.Thread, "start", original_start)
            patch.setattr(Path, "mkdir", original_mkdir)
            following = str(Path(directory) / "following")
            trainer.prepare_checkpoint_save(following, "a")
            trainer.abort_checkpoint_save(following)
            assert not trainer._prepared_checkpoint_saves
            assert trainer._checkpoint_save_next == 2
            worker = trainer._checkpoint_snapshot_spill.thread
            if worker is not None:
                worker.join(3)
                assert not worker.is_alive()
            complete = torch.tensor(1)
            dist.all_reduce(complete)
            assert complete.item() == 2
    finally:
        released.set()
        dist.destroy_process_group()


@pytest.mark.parametrize("action", ["finish", "abort"])
@pytest.mark.parametrize("failure", ["start", "write", "mkdir"])
def test_asymmetric_snapshot_failure_does_not_block_capture(tmp_path, action, failure):
    context = mp.get_context("spawn")
    prepared = [context.Event() for _ in range(2)]
    finalized = [context.Event() for _ in range(2)]
    released = context.Event()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, str(tmp_path), failure, action, prepared, released, finalized),
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 40
    try:
        for process in processes:
            process.start()
        for event in prepared:
            assert event.wait(max(0, deadline - time.monotonic())), (
                "capture waited for sibling disk after an asymmetric writer failure"
            )
        assert not any(event.is_set() for event in finalized)
        released.set()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
            assert process.exitcode == 0
        assert all(event.is_set() for event in finalized)
    finally:
        released.set()
        for process in processes:
            if process.pid is not None:
                process.join(1)
                if process.is_alive():
                    process.terminate()
                    process.join(3)
                if process.is_alive():
                    process.kill()
                    process.join(3)
                assert not process.is_alive()
