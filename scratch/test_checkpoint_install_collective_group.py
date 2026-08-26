from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

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
