"""Real Gloo ownership checks for asymmetric CPU snapshot persistence failures."""

from datetime import timedelta
from pathlib import Path
import threading
import time

import pytest
import safetensors.torch
from test_checkpoint_snapshot_spill import _snapshot_trainer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _checkpoint as cp


def _worker(rank, directory, failure, action, prepared, released, finalized):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{directory}/gloo",
        timeout=timedelta(seconds=15),
    )
    output = str(Path(directory) / "failed")
    original_write, original_start = safetensors.torch.save_file, threading.Thread.start
    original_capture = cp._local_state

    def capture(rank, name, files):
        captured, optimizer, custom = original_capture(rank, name, files)
        owned = files["custom_tensors.safetensors"]["p"]
        return (
            (
                cp._LoraSnapshot(
                    "p", "lora_A.weight", None, 0, {}, {"lora": owned}, None
                ),
            ),
            optimizer,
            custom,
        )

    def expand(captured, files):
        if rank == 0:
            raise RuntimeError("rank zero writer expansion failed")
        assert released.wait(15), "parent did not release sibling expansion"
        torch.testing.assert_close(captured[0].tensors["lora"], torch.tensor([1.0]))
        return ()

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
            trainer = _snapshot_trainer(patch)
            parameter = trainer._checkpoint_slots["a"].params[0]
            if failure == "admission":
                with pytest.MonkeyPatch.context() as admission:
                    admission.setattr(
                        trainer,
                        "_available_cpu_memory_bytes",
                        lambda: 0 if rank == 0 else 12,
                    )
                    admission.setattr(
                        cp,
                        "_local_state",
                        lambda *_: pytest.fail(
                            "capture ran before collective admission"
                        ),
                    )
                    admission.setattr(
                        "art.trainer_rank._heads.synchronize_head_buffers",
                        lambda *_: pytest.fail("buffer copies ran before admission"),
                    )
                    with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
                        trainer.prepare_checkpoint_save(output, "a")
                assert not trainer._checkpoint_preparing_saves
                assert not trainer._prepared_checkpoint_saves
                assert trainer._checkpoint_snapshot_spill is None
                assert not list(Path(directory).glob(".failed.*"))
                assert trainer._checkpoint_save_sequence == 0
                failure = "write"
            patch.setattr(safetensors.torch, "save_file", write)
            patch.setattr(threading.Thread, "start", start)
            if failure == "expand":
                patch.setattr(cp, "_local_state", capture)
                patch.setattr(cp, "_expand_local_state", expand)
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
            patch.setattr(cp, "_local_state", original_capture)
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


@pytest.mark.parametrize(
    "failure,action",
    [
        ("start", "finish"),
        ("start", "abort"),
        ("write", "finish"),
        ("write", "abort"),
        ("expand", "finish"),
        ("expand", "abort"),
        ("admission", "finish"),
    ],
)
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
    deadline = time.monotonic() + 60
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
