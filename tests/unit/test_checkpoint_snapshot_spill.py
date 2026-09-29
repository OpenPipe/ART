from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import pickle
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest
import safetensors.torch
from test_trainer_rank_validation import _prepared_save, _save_state_trainer
import torch

from art.trainer_rank import _checkpoint as cp
from art.trainer_rank._impl import _CheckpointSlot, _CustomObject, _DynamicOptimizer


def _snapshot_trainer(monkeypatch, parameter=None):
    trainer = _save_state_trainer()
    if parameter is None:
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
    return trainer


def test_captured_cpu_custom_optimizer_is_independent(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    buffer = torch.tensor([3.0])
    master = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.Adam((master,), lr=0.125)
    optimizer.state[master] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.ones(2),
        "exp_avg_sq": torch.full((2,), 2.0),
    }
    trainer = _snapshot_trainer(monkeypatch, parameter)
    slot = trainer._checkpoint_slots["a"]
    slot.optimizer = _DynamicOptimizer(optimizer, (master,))
    slot.custom["b"] = _CustomObject("buffer", buffer, object())
    # Parameter/buffer copies alone fit; the optimizer capture does not.
    monkeypatch.setattr(trainer, "_available_cpu_memory_bytes", lambda: 36)
    with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
        cp._admit_snapshot(trainer, "a")
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
        assert list(spill.workspace.values()) == [16] * 8
    finally:
        release.set()
        assert worker is not None
        worker.join(3)
    assert not worker.is_alive()
    for result in results:
        result.result()
    assert len(set(calls)) == 1
    assert spill.thread is None and not spill.pending and not spill.workspace
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
    assert errors == ([error] if fails and action == "finish" else [])
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
    assert not spill.workspace


def test_rank_prepare_returns_before_disk_and_owns_capture(tmp_path, monkeypatch):
    trainer = _snapshot_trainer(monkeypatch)
    parameter = trainer._checkpoint_slots["a"].params[0]
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


@pytest.mark.parametrize("stage", ("__init__", "start"))
def test_writer_start_failure_has_no_orphaned_backlog(tmp_path, monkeypatch, stage):
    spill = cp._SnapshotSpill()
    error = RuntimeError("cannot start writer")

    def fail(_self, *args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, stage, fail)
        with pytest.raises(RuntimeError) as caught:
            spill.submit(tmp_path / "failed", {"v.safetensors": {"v": torch.ones(1)}})
        assert caught.value is error
    assert spill.thread is None and not spill.pending and not spill.workspace
    spill.submit(tmp_path / "next", {"v.safetensors": {"v": torch.ones(1)}}).result(3)
    assert (tmp_path / "next/v.safetensors").is_file()


@pytest.mark.parametrize("action", ("finish", "abort"))
def test_start_failure_is_owned_until_collective_finalization(
    tmp_path, monkeypatch, action
):
    trainer = _save_state_trainer()
    trainer._checkpoint_slots["a"] = _CheckpointSlot(
        config={
            "base_model_name_or_path": "test/model",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
        }
    )
    monkeypatch.setattr(cp, "_validate_save_state", lambda *_: {})
    refs = []

    def capture(_trainer, _name, files):
        value = torch.ones(1)
        refs.append(weakref.ref(value))
        files["v.safetensors"] = {"v": value}
        return (), None, {}

    monkeypatch.setattr(cp, "_local_state", capture)
    error = RuntimeError("cannot start writer")

    def fail(_self):
        raise error

    output = str(tmp_path / "failed")
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail)
        trainer.prepare_checkpoint_save(output, "a")
    prepared = trainer._prepared_checkpoint_saves[output]
    assert prepared.writer is not None
    assert prepared.writer.exception() is error
    assert all(ref() is None for ref in refs)
    assert prepared.snapshot.exists() and prepared.reservation.exists()
    if action == "finish":
        with pytest.raises(RuntimeError) as caught:
            trainer.finish_checkpoint_save(output)
        assert caught.value is error
    else:
        trainer.abort_checkpoint_save(output)
    assert not prepared.snapshot.exists() and not prepared.reservation.exists()
    assert not trainer._prepared_checkpoint_saves
    assert trainer._checkpoint_save_next == 1
    following = str(tmp_path / "next")
    trainer.prepare_checkpoint_save(following, "a")
    trainer.abort_checkpoint_save(following)
    assert trainer._checkpoint_save_next == 2


def test_abort_writer_failure_does_not_hide_failed_cleanup(tmp_path, monkeypatch):
    trainer = _save_state_trainer()
    writer = Future()
    writer.set_exception(OSError("discarded snapshot failed"))
    prepared = replace(_prepared_save(tmp_path, 0), writer=writer)
    trainer._prepared_checkpoint_saves["save"] = prepared
    cleanup = OSError("cannot remove owned snapshot")
    with monkeypatch.context() as patch:
        patch.setattr(cp, "_cleanup_paths", lambda _paths: cleanup)
        with pytest.raises(OSError) as caught:
            trainer.abort_checkpoint_save("save")
    assert caught.value is cleanup
    assert trainer._prepared_checkpoint_saves["save"] is prepared
    assert prepared.snapshot.exists()
    trainer.abort_checkpoint_save("save")
    assert not trainer._prepared_checkpoint_saves
    assert not prepared.snapshot.exists() and not prepared.reservation.exists()


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


@pytest.mark.parametrize("wrapped", ("plain", "cause", "context", "group"))
def test_failed_future_releases_payload_while_preserving_error(
    tmp_path, monkeypatch, wrapped
):
    import gc

    spill = cp._SnapshotSpill()
    value = torch.ones(8)
    ref = weakref.ref(value)
    original = OSError("snapshot disk failed")
    expected = original if wrapped == "plain" else RuntimeError("wrapper")
    if wrapped == "group":
        expected = ExceptionGroup("snapshot failures", [original])

    def inner(tensors):
        assert tensors["v"] is ref()
        raise original

    def fail(tensors, path):
        try:
            inner(tensors)
        except OSError:
            if wrapped == "plain":
                raise
            if wrapped == "cause":
                raise expected from original
            raise expected

    monkeypatch.setattr(safetensors.torch, "save_file", fail)
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = spill.submit(tmp_path / "failed", {"v.safetensors": {"v": value}})
        del value
        worker = spill.thread
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive()
        assert result.exception(3) is expected
        assert expected.__traceback__ is not None
        assert ref() is None
        if wrapped == "cause":
            assert expected.__cause__ is original
        if wrapped == "context":
            assert expected.__context__ is original
        if wrapped == "group":
            assert isinstance(expected, ExceptionGroup)
            assert expected.exceptions == (original,)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("fails", (False, True))
def test_noncontiguous_cpu_packing_is_owned_by_writer(tmp_path, monkeypatch, fails):
    value = torch.arange(12.0).reshape(3, 4).T.to("cpu", copy=True)
    assert not value.is_contiguous()
    raw = weakref.ref(value)
    packed = []
    threads = []
    entered, release = threading.Event(), threading.Event()

    def save(tensors, path):
        threads.append(threading.current_thread())
        selected = tensors["v"]
        packed.append(weakref.ref(selected))
        assert selected.device.type == "cpu" and selected.is_contiguous()
        source = raw()
        assert source is not None
        torch.testing.assert_close(selected, source)
        assert selected.data_ptr() != source.data_ptr()
        entered.set()
        assert release.wait(3)
        if fails:
            raise OSError("packed snapshot write failed")

    monkeypatch.setattr(safetensors.torch, "save_file", save)
    spill = cp._SnapshotSpill()
    result = spill.submit(tmp_path / "packed", {"v.safetensors": {"v": value}})
    worker = spill.thread
    try:
        assert entered.wait(3)
        del value
    finally:
        release.set()
        assert worker is not None
        worker.join(3)
    assert not worker.is_alive()
    assert threads == [worker]
    assert raw() is None and all(ref() is None for ref in packed)
    assert isinstance(result.exception(), OSError) if fails else result.result() is None


@pytest.mark.parametrize("backlog", ("empty", "active", "queued"))
def test_snapshot_capture_admission_precedes_allocation(tmp_path, monkeypatch, backlog):
    parameter = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4).T)
    if backlog == "queued":
        parameter.data = parameter.data.contiguous()
    trainer = _snapshot_trainer(monkeypatch, parameter)
    entered, release = threading.Event(), threading.Event()
    available = 384
    calls = []
    original_state, original_write = cp._local_state, safetensors.torch.save_file

    def capture(*args):
        calls.append(args[1])
        return original_state(*args)

    def write(tensors, path):
        entered.set()
        assert release.wait(3)
        original_write(tensors, path)

    monkeypatch.setattr(trainer, "_available_cpu_memory_bytes", lambda: available)
    monkeypatch.setattr(cp, "_local_state", capture)
    monkeypatch.setattr(safetensors.torch, "save_file", write)
    output = str(tmp_path / "refused")
    try:
        if backlog != "empty":
            for index in range(2):
                trainer.prepare_checkpoint_save(str(tmp_path / str(index)), "a")
                parameter.data = parameter.data.T.contiguous().T
            assert entered.wait(3)
            assert len(trainer._checkpoint_snapshot_spill.pending) == 1
        # Headroom is additional allocation, already excluding resident captures.
        available = 144 if backlog != "empty" else 0
        with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
            trainer.prepare_checkpoint_save(output, "a")
        assert len(calls) == (2 if backlog != "empty" else 0)
        assert len(trainer._prepared_checkpoint_saves) == (
            2 if backlog != "empty" else 0
        )
        assert not trainer._checkpoint_preparing_saves
        assert not list(tmp_path.glob(".refused.*"))
        if backlog != "empty":
            # Existing captures are already resident: do not charge them twice.
            available = 192
            trainer.prepare_checkpoint_save(output, "a")
            assert len(calls) == 3
    finally:
        release.set()
        for pending in list(trainer._prepared_checkpoint_saves):
            trainer.abort_checkpoint_save(pending)
    # Reservations must disappear on drain, and a refused destination is reusable.
    available = 144
    trainer.prepare_checkpoint_save(output, "a")
    trainer.abort_checkpoint_save(output)
    assert not trainer._prepared_checkpoint_saves
    torch.testing.assert_close(parameter, torch.arange(12.0).reshape(3, 4).T)


@pytest.mark.parametrize("has_optimizer", (False, True))
def test_lazy_custom_snapshot_admission_counts_cached_state(monkeypatch, has_optimizer):
    trainer = _snapshot_trainer(monkeypatch)
    slot = trainer._checkpoint_slots["a"]
    payloads = {}
    records = cp._custom_snapshot(trainer, "a", payloads)
    slot.custom.clear()
    slot.params = ()
    cached_optimizer: dict[str, torch.Tensor] = (
        {
            f"{key}/p": torch.ones(1)
            for key in ("master", "exp_avg", "exp_avg_sq", "step")
        }
        if has_optimizer
        else {}
    )
    slot.custom_payload = cp.PreparedCustomPayload(
        records, payloads["custom_tensors.safetensors"], cached_optimizer
    )
    slot.optimizer = _DynamicOptimizer(
        torch.optim.Adam((torch.nn.Parameter(torch.ones(1)),)), ()
    )
    # Both loaded optimizer data and synthesized missing state need admission.
    monkeypatch.setattr(trainer, "_available_cpu_memory_bytes", lambda: 12)
    with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
        cp._admit_snapshot(trainer, "a")


def test_module_snapshot_admission_reads_metadata_without_hooks(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2))
    trainer = _snapshot_trainer(monkeypatch, parameter)
    module = torch.nn.Module()
    module.register_parameter("left", parameter)
    module.register_parameter("right", parameter)
    module.register_buffer("scratch", torch.ones(2), persistent=False)
    hooks = []
    module.register_state_dict_pre_hook(lambda *_: hooks.append(True))
    trainer._checkpoint_slots["a"].custom["p"] = _CustomObject(
        "module", module, object()
    )
    available = 47
    monkeypatch.setattr(trainer, "_available_cpu_memory_bytes", lambda: available)
    with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
        cp._admit_snapshot(trainer, "a")
    assert not hooks
    # Two eight-byte saved keys, capture/packing/serialization; no scratch buffer.
    available = 48
    cp._admit_snapshot(trainer, "a")
    assert not hooks
    cp._custom_snapshot(trainer, "a", {})
    assert hooks == [True]


@pytest.mark.parametrize("world", (2, 8))
@pytest.mark.parametrize("kind", ("buffer", "module"))
def test_snapshot_admission_counts_buffer_gather(monkeypatch, world, kind):
    from art.trainer_rank._heads import _plain

    trainer = _snapshot_trainer(monkeypatch)
    buffer = torch.ones(64)[1:3]
    value: torch.Tensor | torch.nn.Module = buffer
    if kind == "module":
        value = torch.nn.Module()
        value.register_buffer("saved", buffer)
        value.register_buffer("scratch", torch.ones(64), persistent=False)
    trainer._checkpoint_slots["a"].custom["b"] = _CustomObject(kind, value, object())
    payload = _plain(buffer).cpu()
    assert payload.untyped_storage().nbytes() == 8  # Not the 256-byte backing view.
    assert 8 < len(pickle.dumps({("a", "b"): (0, {"saved": payload})})) <= 4104
    trainer._checkpoint_snapshot_spill = SimpleNamespace(
        lock=threading.Lock(), workspace={Future(): 512}
    )
    monkeypatch.setattr(cp, "_distributed", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: world)
    # Padded gather plus cloning/serialization/deserialization and an old writer.
    available = (world + 6) * 4104 + 512 - 1
    monkeypatch.setattr(trainer, "_available_cpu_memory_bytes", lambda: available)
    with pytest.raises(RuntimeError, match="checkpoint.*host memory"):
        cp._admit_snapshot(trainer, "a")
    available += 1
    cp._admit_snapshot(trainer, "a")
