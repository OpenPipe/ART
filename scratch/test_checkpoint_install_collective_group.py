from datetime import timedelta
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _checkpoint


def test_checkpoint_groups_are_created_once_in_fixed_order(monkeypatch) -> None:
    groups = [object(), object(), object()]
    calls: list[str] = []

    def new_group(*, backend: str):
        calls.append(backend)
        return groups[len(calls) - 1]

    trainer = SimpleNamespace(_checkpoint_group_lock=threading.Lock())
    monkeypatch.setattr(_checkpoint, "_distributed", lambda: True)
    monkeypatch.setattr(_checkpoint.dist, "new_group", new_group)

    assert _checkpoint._ensure_group(trainer) is groups[0]
    assert _checkpoint._ensure_finalize_group(trainer) is groups[1]
    assert _checkpoint._ensure_install_group(trainer) is groups[2]
    assert _checkpoint._ensure_groups(trainer) == tuple(groups)
    assert calls == ["gloo", "gloo", "gloo"]


def test_staged_install_uses_install_group(monkeypatch) -> None:
    install_group = object()
    observed: list[object] = []
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(model=()),
        _checkpoint_slots={},
        _guard_slot_can_load=lambda _ref: None,
        _slot_ref=lambda name: name,
        _validate_loaded_checkpoint_config=lambda _name, _config: None,
    )
    prepared = _checkpoint.PreparedCheckpointSlotInstall(
        name="run-b",
        source=_checkpoint.PreparedCheckpoint(
            path=Path("run-b"),
            config={},
            keys=(),
            manifest=None,
            digest="digest",
        ),
        config={},
        sites=(),
        expected_keys=frozenset(),
    )

    monkeypatch.setattr(
        _checkpoint, "_ensure_install_group", lambda _trainer: install_group
    )
    lora = ModuleType("art.megatron.lora")
    lora.LoRA = type("LoRA", (), {})
    lora.LoRASlotRef = type("LoRASlotRef", (), {})
    implementation = ModuleType("art.trainer_rank._impl")
    implementation._CheckpointSlot = type("_CheckpointSlot", (), {})
    monkeypatch.setitem(sys.modules, "art.megatron.lora", lora)
    monkeypatch.setitem(sys.modules, "art.trainer_rank._impl", implementation)

    def phase(_action, _label: str, group):
        observed.append(group)

    monkeypatch.setattr(_checkpoint, "_phase", phase)

    _checkpoint.install_staged_checkpoint_slot(trainer, prepared)

    assert observed
    assert set(observed) == {install_group}


def _overlapped_phase_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    result_root: str,
    isolated: bool,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    trainer = SimpleNamespace(_checkpoint_group_lock=threading.Lock())
    prepare_group = _checkpoint._ensure_group(trainer)
    install_group = (
        _checkpoint._ensure_install_group(trainer) if isolated else prepare_group
    )
    primary = "prepare" if rank == 0 else "install"
    secondary = "install" if rank == 0 else "prepare"
    primary_entered = threading.Event()
    results: dict[str, tuple[tuple[str, int], ...]] = {}
    errors: list[str] = []
    original_gather = _checkpoint.dist.all_gather_object

    def observed_gather(*args, **kwargs):
        if threading.current_thread().name == "primary-phase":
            primary_entered.set()
        return original_gather(*args, **kwargs)

    def run_phase(name: str) -> None:
        group = prepare_group if name == "prepare" else install_group
        try:
            _checkpoint._phase(lambda: None, f"run {name} phase", group)
            results[name] = _checkpoint._gather((name, rank), group)
        except BaseException as error:  # pragma: no cover - reported to parent
            errors.append(repr(error))

    _checkpoint.dist.all_gather_object = observed_gather
    try:
        first = threading.Thread(
            target=run_phase, args=(primary,), name="primary-phase"
        )
        first.start()
        if not primary_entered.wait(timeout=5):
            raise TimeoutError("primary checkpoint phase did not enter its collective")
        dist.barrier()
        primary_blocked = first.is_alive()
        second = threading.Thread(
            target=run_phase, args=(secondary,), name="secondary-phase"
        )
        second.start()
        first.join(timeout=10)
        second.join(timeout=10)
        if first.is_alive() or second.is_alive():
            raise TimeoutError("overlapped checkpoint phases did not complete")
        Path(result_root, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "primary_blocked": primary_blocked,
                    "results": results,
                    "errors": errors,
                }
            ),
            encoding="utf-8",
        )
    finally:
        _checkpoint.dist.all_gather_object = original_gather
        dist.destroy_process_group()


def _run_overlapped_phases(tmp_path: Path, *, isolated: bool) -> tuple[dict, ...]:
    root = tmp_path / ("isolated" if isolated else "shared")
    root.mkdir()
    mp.spawn(
        _overlapped_phase_worker,
        args=(2, str(root / "rendezvous"), str(root), isolated),
        nprocs=2,
        join=True,
    )
    return tuple(
        json.loads((root / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    )


def test_overlapped_prepare_and_install_collectives_are_phase_isolated(
    tmp_path: Path,
) -> None:
    shared = _run_overlapped_phases(tmp_path, isolated=False)
    assert not any(result["errors"] for result in shared)
    assert all(
        {item[0] for item in gathered} == {"prepare", "install"}
        for result in shared
        for gathered in result["results"].values()
    )

    isolated = _run_overlapped_phases(tmp_path, isolated=True)
    assert not any(result["errors"] for result in isolated)
    assert all(result["primary_blocked"] for result in isolated)
    assert all(
        {item[0] for item in gathered} == {phase}
        for result in isolated
        for phase, gathered in result["results"].items()
    )
