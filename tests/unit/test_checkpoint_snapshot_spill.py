from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import weakref

import pytest
import safetensors.torch
from test_trainer_rank_validation import _prepared_save, _save_state_trainer
import torch

from art.trainer_rank import _checkpoint as cp
from art.trainer_rank._impl import _CheckpointSlot, _CustomObject, _DynamicOptimizer


def test_captured_cpu_custom_optimizer_is_independent() -> None:
    trainer = _save_state_trainer()
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    buffer = torch.tensor([3.0])
    master = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.Adam((master,), lr=0.125)
    optimizer.state[master] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.ones(2),
        "exp_avg_sq": torch.full((2,), 2.0),
    }
    trainer._checkpoint_slots["a"] = _CheckpointSlot(
        params=(parameter,),
        config={
            "base_model_name_or_path": "test/model",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
        },
        optimizer=_DynamicOptimizer(optimizer, (master,)),
        custom={
            "p": _CustomObject("parameter", parameter, object()),
            "b": _CustomObject("buffer", buffer, object()),
        },
    )
    payloads = {}
    records = cp._custom_snapshot(trainer, "a", payloads)
    before = {
        file: {key: value.clone() for key, value in tensors.items()}
        for file, tensors in payloads.items()
    }
    with torch.no_grad():
        parameter.add_(100)
        buffer.add_(100)
        master.add_(100)
        for value in optimizer.state[master].values():
            value.add_(100)
    for file, tensors in payloads.items():
        for key, value in tensors.items():
            torch.testing.assert_close(value, before[file][key])
            assert value.device.type == "cpu" and not value.requires_grad
    assert records["p"]["trainable_keys"] == ["p"]


def test_one_spill_writer_drains_payloads_without_finalization(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = safetensors.torch.save_file

    def write(tensors, path):
        calls.append(threading.current_thread())
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        original(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", write)
    spill = cp._SnapshotSpill()
    tensors = [torch.ones(4) for _ in range(8)]
    refs = [weakref.ref(value) for value in tensors]
    payloads = [{"v.safetensors": {"v": value}} for value in tensors]
    del tensors
    results = [
        spill.submit(tmp_path / str(i), values) for i, values in enumerate(payloads)
    ]
    worker = spill.thread
    try:
        assert entered.wait(3)
        assert all(not result.done() for result in results)
        assert len(spill.pending) == 7
    finally:
        release.set()
        assert worker is not None
        worker.join(3)
    assert not worker.is_alive()
    for result in results:
        result.result()
    assert len(set(calls)) == 1
    assert spill.thread is None and not spill.pending
    assert all(not values for values in payloads)
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("action", ("finish", "abort"))
@pytest.mark.parametrize("fails", (False, True))
def test_finish_abort_wait_for_owned_write_before_cleanup(
    tmp_path, monkeypatch, action, fails
):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    trainer = _save_state_trainer()
    prepared = _prepared_save(tmp_path, 0)
    original = safetensors.torch.save_file
    error = OSError("owned disk failure")

    def write(tensors, path):
        entered.set()
        assert release.wait(3)
        assert prepared.snapshot.is_dir()
        if fails:
            raise error
        original(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", write)
    spill = cp._SnapshotSpill()
    writer = spill.submit(prepared.snapshot, {"v.safetensors": {"v": torch.ones(1)}})
    prepared = replace(prepared, writer=writer)
    trainer._prepared_checkpoint_saves["save"] = prepared
    monkeypatch.setattr(cp, "_finish", lambda *_: None)
    errors = []

    def finish():
        try:
            getattr(cp, f"{action}_checkpoint_save")(trainer, "save")
        except BaseException as exc:
            errors.append(exc)
        finally:
            closed.set()

    thread = threading.Thread(target=finish)
    thread.start()
    try:
        assert entered.wait(3)
        assert not closed.wait(0.05)
        assert prepared.snapshot.exists()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert errors == ([error] if fails else [])
    assert not prepared.snapshot.exists() and not prepared.reservation.exists()
    assert not trainer._prepared_checkpoint_saves
    assert trainer._checkpoint_save_next == 1


def test_failed_spill_does_not_strand_following_save(tmp_path, monkeypatch):
    original = safetensors.torch.save_file
    error = OSError("first write failed")

    def write(tensors, path):
        if "first" in str(path):
            raise error
        original(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", write)
    spill = cp._SnapshotSpill()
    first = spill.submit(tmp_path / "first", {"v.safetensors": {"v": torch.ones(1)}})
    second = spill.submit(tmp_path / "second", {"v.safetensors": {"v": torch.ones(1)}})
    with pytest.raises(OSError) as caught:
        first.result(3)
    assert caught.value is error
    second.result(3)
    assert (tmp_path / "second/v.safetensors").is_file()


def test_rank_prepare_returns_before_disk_and_owns_capture(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

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
    monkeypatch.setattr(trainer, "_slot_ref", lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.lora",
        SimpleNamespace(LoRA=type("UnusedLoRA", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.weights.lora_publish",
        SimpleNamespace(collect_local_lora_entries=lambda *a, **kw: ({}, [])),
    )
    entered, release = threading.Event(), threading.Event()
    saved = {}
    original = safetensors.torch.save_file

    def write(tensors, path):
        entered.set()
        assert release.wait(3)
        saved.update({key: value.clone() for key, value in tensors.items()})
        original(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", write)
    output = str(tmp_path / "checkpoint")
    try:
        trainer.prepare_checkpoint_save(output, "a")
        assert entered.wait(3)
        prepared = trainer._prepared_checkpoint_saves[output]
        assert prepared.writer is not None and not prepared.writer.done()
        with torch.no_grad():
            parameter.add_(100)
        config = trainer._checkpoint_slots["a"].config
        assert config is not None
        config["r"] = 99
        assert prepared.config["r"] == 1
    finally:
        release.set()
        trainer.abort_checkpoint_save(output)
    torch.testing.assert_close(saved["p"], torch.tensor([1.0]))
    assert not trainer._prepared_checkpoint_saves


def test_writer_start_failure_has_no_orphaned_backlog(tmp_path, monkeypatch):
    spill = cp._SnapshotSpill()
    error = RuntimeError("cannot start writer")

    def fail(_self):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError) as caught:
            spill.submit(tmp_path / "failed", {"v.safetensors": {"v": torch.ones(1)}})
        assert caught.value is error
    assert spill.thread is None and not spill.pending
    spill.submit(tmp_path / "next", {"v.safetensors": {"v": torch.ones(1)}}).result(3)
    assert (tmp_path / "next/v.safetensors").is_file()


def test_custom_digest_failure_is_reported_before_next_collective(
    tmp_path, monkeypatch
):
    trainer = _save_state_trainer()
    prepared = replace(
        _prepared_save(tmp_path, 0),
        custom_tensors={
            "p": {
                "kind": "parameter",
                "tensor_keys": ["p"],
                "trainable_keys": ["p"],
                "parameter_aliases": [],
                "buffer_aliases": [],
                "persistent_buffer_keys": [],
            }
        },
    )
    error = OSError("cannot read custom snapshot")
    seen = []

    def digest(_path):
        raise error

    def gather(value, _group):
        seen.append(value)
        return (value, None)

    monkeypatch.setattr(cp, "_file_digest", digest)
    monkeypatch.setattr(cp, "_gather", gather)
    with pytest.raises(OSError) as caught:
        cp._finish(trainer, prepared)
    assert caught.value is error
    # The local failure is the collective payload; no rank advances to metadata.
    assert seen == [repr(error)]
