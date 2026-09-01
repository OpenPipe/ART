from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

from art.megatron import checkpoint
from art.megatron.runtime.portable_snapshot import (
    PortableSnapshotCommittedFile,
    PortableSnapshotGeneration,
    PortableSnapshotRankReceipt,
    build_portable_snapshot_archive,
    export_portable_checkpoint,
    prepare_portable_checkpoint,
)
from art.megatron.runtime.run_slots import MegatronRunSlots, OptimizerConfig


def _slots(monkeypatch: pytest.MonkeyPatch) -> MegatronRunSlots:
    from art.megatron import lora as lora_module
    from art.megatron.lora import LoRA

    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps, "get_data_parallel_rank", lambda **_kwargs: 0
    )
    lora = LoRA("layer.q_proj", 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    runtime = SimpleNamespace(
        model=(lora,),
        model_identifier="test/model",
        model_support_spec=SimpleNamespace(model_names=("test/model",)),
        model_support_handler=SimpleNamespace(
            canonicalize_loaded_lora_state=lambda state, _model: state
        ),
        provider=SimpleNamespace(),
    )
    slots = MegatronRunSlots(runtime)
    adapter = {
        "layer.q_proj.lora_A.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
        "layer.q_proj.lora_B.weight": torch.arange(8, dtype=torch.float32).view(4, 2),
    }
    loaded = slots._load_checkpoint_slot("run", adapter, alpha=2, _prepared=True)
    params = slots._validate_checkpoint_consistency("run", loaded, set(adapter))
    slots._checkpoint_slots["run"] = checkpoint.CheckpointSlot(
        params=params,
        config={
            "base_model_name_or_path": "test/model",
            "r": 2,
            "lora_alpha": 2,
            "target_modules": ["q_proj"],
        },
    )
    optimizer = slots.prepare_fresh_checkpoint_slot_optimizer_for_residency(
        "run", OptimizerConfig(learning_rate=3e-4)
    )
    with torch.no_grad():
        for index, tensor in enumerate(optimizer):
            tensor.add_(index + 1)
    return slots


def test_portable_transport_bounds_component_and_file_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_slots = _slots(monkeypatch)
    weights = source_slots.checkpoint_slot_parameters("run")
    optimizer = source_slots.checkpoint_slot_optimizer_tensors("run")
    active: set[str] = set()
    max_active = 0

    @contextmanager
    def components(component: str) -> Iterator[tuple[torch.Tensor, ...]]:
        nonlocal max_active
        assert component not in active
        active.add(component)
        max_active = max(max_active, len(active))
        try:
            yield weights if component == "weights" else optimizer
        finally:
            active.remove(component)

    payloads: dict[str, bytes] = {}

    class Sink:
        def commit_prepared(self, *, directory: Path, files: Any, **_kwargs: Any) -> Any:
            assert len(files) == 1
            assert len([path for path in directory.rglob("*") if path.is_file()]) == 1
            item = files[0]
            payloads[item.relative_path] = (directory / item.relative_path).read_bytes()
            return (
                PortableSnapshotCommittedFile(
                    relative_path=item.relative_path,
                    object_id=f"object/{item.relative_path}",
                    source_ref=f"memory://{item.relative_path}",
                ),
            )

        def close(self, *, deadline: float | None = None) -> None:
            del deadline

    generation = PortableSnapshotGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id=f"step-{1:08d}-{'a' * 32}",
    )
    receipt = export_portable_checkpoint(
        source_slots,
        Sink(),  # type: ignore[arg-type]
        generation,
        export_id="export",
        name="run",
        rank=0,
        components=components,  # type: ignore[arg-type]
    )
    assert isinstance(receipt, PortableSnapshotRankReceipt)
    assert max_active == 1
    assert active == set()
    assert set(payloads) == {item.relative_path for item in receipt.files}

    class Source:
        def __init__(self) -> None:
            self.read_widths: list[int] = []

        def read_prepared(self, _receipt: Any, files: Any) -> None:
            self.read_widths.append(len(files))
            assert len(files) == 1
            for relative, target in files.items():
                target[:] = payloads[relative]

        def close(self, *, deadline: float | None = None) -> None:
            del deadline

    archive = build_portable_snapshot_archive(
        generation=generation,
        checkpoint_digest=receipt.checkpoint_digest,
        ranks=(receipt,),
    )
    target_slots = _slots(monkeypatch)
    target_slots.release_checkpoint_slot("run")
    source = Source()
    with prepare_portable_checkpoint(
        target_slots,
        source,  # type: ignore[arg-type]
        archive,
        destination_rank=0,
        expected_lora_rank=2,
        expected_lora_target_modules=("q_proj",),
        restore_optimizer=True,
    ) as prepared:
        restored_weights, restored_optimizer = (
            target_slots.install_prepared_checkpoint_for_residency(
                "restored",
                prepared.checkpoint,
                restore_optimizer=True,
                require_optimizer=True,
                materialize=prepared.materialize,
            )
        )
        assert all(source.read_widths)
        assert set(source.read_widths) == {1}
        retained = [
            path.name
            for path in Path(prepared._temporary.name).rglob("*")
            if path.is_file()
        ]
        assert sorted(retained) == ["adapter_config.json", "checkpoint.json"]
        assert {item.relative_path for item in prepared.receipt.files} == set(payloads)

    for actual, expected in zip(restored_weights, weights, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(restored_optimizer, optimizer, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
