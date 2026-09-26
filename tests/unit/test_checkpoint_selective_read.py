from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import safetensors
import safetensors.torch
from safetensors.torch import load_file, save_file
import torch

from art.trainer_rank import _checkpoint


def _eager(prepared, relative, prefix, keys=None, *, snapshot=None):
    payload = load_file(prepared.snapshot / relative)
    if keys is None:
        return {
            k.removeprefix(prefix + "/"): v
            for k, v in payload.items()
            if k.startswith(prefix + "/")
        }
    return {key: payload[f"{prefix}/{key}"] for key in keys}


@dataclass(frozen=True)
class _Meta:
    key: str
    block: str
    shape: tuple[int, ...]
    dtype_name: str
    owner_rank: int = 0

    @property
    def manifest(self):
        return {"sharded": False, "shard_world_size": 1, "shard_rank": 0}


@pytest.fixture
def dependencies(monkeypatch):
    _dependencies(monkeypatch)
    monkeypatch.setattr(_checkpoint, "_ensure_finalize_group", lambda trainer: None)


def _dependencies(monkeypatch):
    # Only the unchanged single-rank replicated merge and config writer are stand-ins.
    publish = ModuleType("art.megatron.weights.lora_publish")

    def merge(entries):
        assert all(
            len(parts) == 1 and not parts[0][0]["sharded"] for parts in entries.values()
        )
        return {key: parts[0][1] for key, parts in entries.items()}

    monkeypatch.setattr(publish, "merge_sharded_adapter_entries", merge, raising=False)
    disk = ModuleType("art.megatron.model_support.lora_disk")
    monkeypatch.setattr(
        disk,
        "save_adapter_config",
        lambda path, config: (path / "adapter_config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2) + "\n"
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, publish.__name__, publish)
    monkeypatch.setitem(sys.modules, disk.__name__, disk)


def _prepared(root, *, dtype=torch.bfloat16, rank=1, optimizer=True, damage=None):
    snapshot = root / "snapshot"
    reservation = root / "reservation"
    snapshot.mkdir(parents=True)
    reservation.mkdir()
    shards = []
    for block in range(2):
        payload = {}
        for index in range(2):
            key = f"layer{block}.expert{index}.lora_A.weight"
            tensor = (
                torch.arange(6 * rank).reshape(6, rank).T + block * 32 + index
            ).to(dtype)
            payload[f"lora/{key}"] = tensor.contiguous()
            if optimizer:
                for offset, name in enumerate(("master", "exp_avg", "exp_avg_sq")):
                    payload[f"{name}/{key}"] = tensor.float().contiguous() + offset
                # Mixed scalar dtypes ensure the old offset ordering is preserved.
                payload[f"step/{key}"] = torch.tensor(
                    350.0, dtype=torch.float64 if index else torch.float32
                )
            shards.append(
                _checkpoint._LocalShard(
                    cast(
                        Any,
                        _Meta(
                            key,
                            f"layer{block}",
                            tuple(tensor.shape),
                            str(dtype).removeprefix("torch."),
                        ),
                    ),
                    f"block{block}.safetensors",
                )
            )
        if block == 0 and damage:
            key = "layer0.expert0.lora_A.weight"
            if damage == "missing_master":
                del payload[f"master/{key}"]
            elif damage == "missing_step":
                del payload[f"step/{key}"]
            elif damage == "nonscalar_step":
                payload[f"step/{key}"] = torch.ones(2)
        path = snapshot / f"block{block}.safetensors"
        save_file(payload, path)
        if block == 0 and damage == "truncated":
            path.write_bytes(path.read_bytes()[:-1])
    opt: _checkpoint.OptimizerConfig | None = (
        dict(learning_rate=0.001, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01)
        if optimizer
        else None
    )
    return _checkpoint._PreparedSave(
        0,
        snapshot,
        reservation,
        root / "result",
        {
            "base_model_name_or_path": "public/model",
            "r": rank,
            "lora_alpha": 32,
            "target_modules": ["q_proj"],
        },
        tuple(shards),
        opt,
    )


def _trainer(prepared):
    return SimpleNamespace(
        _slot_state_error=ValueError,
        _checkpoint_finalize_lock=threading.Lock(),
        _checkpoint_save_condition=threading.Condition(),
        _prepared_checkpoint_saves={str(prepared.destination): prepared},
        _finalized_checkpoint_saves={},
        _checkpoint_save_outcomes={},
        _checkpoint_finalizing_saves={},
        _checkpoint_save_next=0,
        _checkpoint_save_skipped=set(),
    )


def _files(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("optimizer", [False, True])
def test_final_checkpoint_bytes_match_eager(
    dependencies, monkeypatch, tmp_path, dtype, rank, optimizer
):
    actual = _prepared(tmp_path / "actual", dtype=dtype, rank=rank, optimizer=optimizer)
    expected = _prepared(
        tmp_path / "expected", dtype=dtype, rank=rank, optimizer=optimizer
    )
    _checkpoint.finish_checkpoint_save(_trainer(actual), str(actual.destination))
    with monkeypatch.context() as patch:
        patch.setattr(_checkpoint, "_read_snapshot", _eager)
        _checkpoint.finish_checkpoint_save(
            _trainer(expected), str(expected.destination)
        )
    assert _files(actual.destination) == _files(expected.destination)
    assert not actual.snapshot.exists() and not actual.reservation.exists()
    manifest = json.loads((actual.destination / "checkpoint.json").read_text())
    assert set(manifest["steps"].values()) == ({350.0} if optimizer else set())
    for path in actual.destination.rglob("*.safetensors"):
        for key, value in load_file(path).items():
            assert value.dtype == (
                torch.float32 if "optimizer" in path.parts else dtype
            )


@pytest.mark.parametrize(
    "damage", ["missing_master", "missing_step", "nonscalar_step", "truncated"]
)
def test_failure_matches_eager_and_closes_finalization(
    dependencies, monkeypatch, tmp_path, damage
):
    errors = []
    for name, reader in (("eager", _eager), ("selective", _checkpoint._read_snapshot)):
        prepared = _prepared(tmp_path / name, damage=damage)
        trainer = _trainer(prepared)
        with monkeypatch.context() as patch:
            patch.setattr(_checkpoint, "_read_snapshot", reader)
            with pytest.raises(Exception) as caught:
                _checkpoint.finish_checkpoint_save(trainer, str(prepared.destination))
        errors.append((type(caught.value), str(caught.value)))
        assert not list((tmp_path / name).iterdir())
        assert (
            not trainer._prepared_checkpoint_saves
            and not trainer._checkpoint_finalizing_saves
        )
        assert (
            trainer._finalized_checkpoint_saves[str(prepared.destination)].outcome
            == "abort"
        )
    assert errors[0] == errors[1]


def test_reader_exact_keys_order_and_closed_handle(tmp_path):
    payload = {
        "lora/a/b": torch.tensor([-0.0, 3.0]),
        "lora/a": torch.tensor([5.0]),
        "step/a": torch.tensor(350.0),
        "steps/not_step": torch.ones(2),
    }
    save_file(payload, tmp_path / "block")
    prepared = cast(_checkpoint._PreparedSave, SimpleNamespace(snapshot=tmp_path))
    out = _checkpoint._read_snapshot(
        prepared, "block", "lora", iter(["a/b", "a", "a/b"])
    )
    assert list(out) == ["a/b", "a"]
    steps = _checkpoint._read_snapshot(prepared, "block", "step")
    assert list(steps) == ["a"] and steps["a"].item() == 350.0
    (tmp_path / "block").unlink()
    assert (
        out["a/b"].view(torch.int32).tolist()
        == payload["lora/a/b"].view(torch.int32).tolist()
    )
    with pytest.raises(FileNotFoundError):
        _checkpoint._read_snapshot(prepared, "block", "lora", [])


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_get_tensor_error_identity_and_cleanup(
    dependencies, monkeypatch, tmp_path, error_type
):
    prepared = _prepared(tmp_path)
    real = safetensors.safe_open
    error = error_type("public injected tensor-read failure")
    closed = []

    class Failing:
        def __init__(self, *args, **kwargs):
            self.reader = real(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            closed.append(True)
            return self.reader.__exit__(*args)

        def offset_keys(self):
            return self.reader.offset_keys()

        def get_tensor(self, key):
            raise error

    monkeypatch.setattr(safetensors, "safe_open", Failing)
    with pytest.raises(error_type) as caught:
        _checkpoint.finish_checkpoint_save(
            _trainer(prepared), str(prepared.destination)
        )
    assert caught.value is error and closed == [True]
    assert not list(tmp_path.iterdir())


def test_only_requested_component_bytes_materialized(
    dependencies, monkeypatch, tmp_path
):
    real = safetensors.safe_open
    reads = []

    class Tracking:
        def __init__(self, filename, **kwargs):
            self.reader = real(filename, **kwargs)
            self.snapshot = Path(filename).parent.name == "snapshot"

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def offset_keys(self):
            return self.reader.offset_keys()

        def get_tensor(self, key):
            value = self.reader.get_tensor(key)
            if self.snapshot:
                reads.append((key, value.numel() * value.element_size()))
            return value

        def get_tensors(self):
            values = cast(Any, self.reader).get_tensors()
            if self.snapshot:
                reads.extend(
                    (key, value.numel() * value.element_size())
                    for key, value in values.items()
                )
            return values

    monkeypatch.setattr(safetensors, "safe_open", Tracking)
    monkeypatch.setattr(safetensors.torch, "safe_open", Tracking)
    results = {}
    for name, reader in (("eager", _eager), ("selective", _checkpoint._read_snapshot)):
        prepared = _prepared(tmp_path / name, rank=3)
        reads.clear()
        with monkeypatch.context() as patch:
            patch.setattr(_checkpoint, "_read_snapshot", reader)
            _checkpoint._finish(
                cast(Any, SimpleNamespace(_slot_state_error=ValueError)), prepared
            )
        results[name] = (len(reads), sum(size for _, size in reads))
    assert results["eager"] == (100, 5160)
    assert results["selective"] == (20, 1032)


def test_finalizer_opens_and_indexes_each_snapshot_once(
    dependencies, monkeypatch, tmp_path
):
    real = safetensors.safe_open
    events = []

    class Tracking:
        def __init__(self, filename, **kwargs):
            self.reader = real(filename, **kwargs)
            self.path = Path(filename)

        def __enter__(self):
            self.reader.__enter__()
            if self.path.parent.name == "snapshot":
                events.append(("open", self.path.name))
            return self

        def __exit__(self, *args):
            try:
                return self.reader.__exit__(*args)
            finally:
                if self.path.parent.name == "snapshot":
                    events.append(("close", self.path.name))

        def offset_keys(self):
            events.append(("index", self.path.name))
            return self.reader.offset_keys()

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, key):
            return self.reader.get_tensor(key)

    monkeypatch.setattr(safetensors, "safe_open", Tracking)
    prepared = _prepared(tmp_path)
    _checkpoint.finish_checkpoint_save(_trainer(prepared), str(prepared.destination))
    assert events == [
        (action, f"block{block}.safetensors")
        for block in range(2)
        for action in ("open", "index", "close")
    ]


def test_block_reader_multiple_files_selected_keys_and_tensor_lifetime(tmp_path):
    payload = {"lora/a": torch.tensor([-0.0, 3.0]), "step/a": torch.tensor(350.0)}
    for name in ("one", "two"):
        save_file(payload, tmp_path / name)
    prepared = cast(_checkpoint._PreparedSave, SimpleNamespace(snapshot=tmp_path))
    with _checkpoint._snapshot_block(None) as snapshot:
        first = _checkpoint._read_snapshot(
            prepared, "one", "lora", iter(["a", "a"]), snapshot=snapshot
        )
        second = _checkpoint._read_snapshot(prepared, "two", "step", snapshot=snapshot)
        with pytest.raises(KeyError):
            _checkpoint._read_snapshot(
                prepared, "one", "lora", ["missing"], snapshot=snapshot
            )
    for name in ("one", "two"):
        (tmp_path / name).unlink()
    assert list(first) == ["a"]
    assert (
        first["a"].view(torch.int32).tolist()
        == payload["lora/a"].view(torch.int32).tolist()
    )
    assert second["a"].item() == 350.0
