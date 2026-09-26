from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from safetensors.torch import load_file, save_file
import torch

from art.trainer_rank import _checkpoint


@dataclass
class _Meta:
    key: str
    owner_rank: int = 0
    block: str = "block"


class _Exports:
    def __init__(self, parameters, order):
        self.entries = [(f"p{p}.expert{e}", parameters[p], e) for p, e in order]

    def _export_items(self, ref):
        return self.entries

    def modules(self):
        return (self,)


def _trainer(monkeypatch, *, dtype=torch.float32, rank=1, mode="full", missing=()):
    parameters = [torch.nn.Parameter(torch.zeros(4, rank, 3)) for _ in range(2)]
    # Noncontiguous source storage, distinct values for parameters/components.
    masters = [
        torch.nn.Parameter(
            (torch.arange(12 * rank).reshape(4, 3, rank) + 100 * i)
            .to(dtype)
            .transpose(1, 2)
        )
        for i in range(2)
    ]
    state = {
        master: {
            **{
                component: master.detach().clone() + offset
                for component, offset in (("exp_avg", 10), ("exp_avg_sq", 20))
                if component not in missing
            },
            "step": torch.tensor(650.0),
        }
        for master in masters
    }
    order = [(p, e) for p in range(2) for e in range(4)]
    if mode == "interleaved":
        order.sort(key=lambda pair: pair[1])
    elif mode == "reversed":
        order.reverse()
    elif mode == "sparse":
        order = [(p, e) for p, e in order if e % 2 == 0]
    model = _Exports(parameters, order)
    metadata = [_Meta(key) for key, _, _ in model.entries]
    if mode == "metadata_subset":
        metadata = metadata[::2]
    lora = ModuleType("art.megatron.lora")
    setattr(lora, "LoRA", _Exports)
    publish = ModuleType("art.megatron.weights.lora_publish")
    setattr(
        publish,
        "collect_local_lora_entries",
        lambda *args, **kwargs: (
            {item.key: torch.zeros(3, rank) for item in metadata},
            metadata,
        ),
    )
    monkeypatch.setitem(sys.modules, lora.__name__, lora)
    monkeypatch.setitem(sys.modules, publish.__name__, publish)
    optimizer = SimpleNamespace(
        state=state,
        param_groups=[dict(lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)],
    )
    slot = SimpleNamespace(
        config={"r": rank},
        snapshot=False,
        params=parameters,
        optimizer=SimpleNamespace(master_params=masters, optimizer=optimizer),
        custom={},
        custom_payload=None,
    )
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(model=[model]),
        _checkpoint_slots={"slot": slot},
        _slot_ref=lambda _: None,
        _checkpoint_grad_flags=lambda _: (False,),
        _slot_state_error=ValueError,
        _checkpoint_prepare_lock=threading.Lock(),
        _checkpoint_finalize_lock=threading.Lock(),
        _checkpoint_save_condition=threading.Condition(),
        _checkpoint_preparing_saves=set(),
        _prepared_checkpoint_saves={},
        _finalized_checkpoint_saves={},
        _checkpoint_save_sequence=0,
        _checkpoint_save_next=0,
        _checkpoint_save_skipped=set(),
        _checkpoint_finalizing_saves={},
        _checkpoint_save_outcomes={},
    )
    expected = {}
    for item in metadata:
        p, e = map(int, item.key.removeprefix("p").split(".expert"))
        expected[f"lora/{item.key}"] = torch.zeros(3, rank)
        for component in ("master", "exp_avg", "exp_avg_sq"):
            value = (
                masters[p]
                if component == "master"
                else state[masters[p]].get(component, torch.zeros_like(masters[p]))
            )
            expected[f"{component}/{item.key}"] = value[e].T.float().detach().clone()
        expected[f"step/{item.key}"] = torch.tensor(650.0)
    return trainer, expected


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("missing", [(), ("exp_avg",), ("exp_avg", "exp_avg_sq")])
def test_checkpoint_optimizer_tensor_contents(
    monkeypatch, tmp_path, dtype, rank, missing
):
    trainer, expected = _trainer(monkeypatch, dtype=dtype, rank=rank, missing=missing)
    _checkpoint._local_state(trainer, "slot", tmp_path)
    actual = load_file(tmp_path / "block-000000.safetensors")
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode", ["full", "sparse", "metadata_subset", "interleaved", "reversed"]
)
def test_coalescing_requires_complete_consecutive_exports(monkeypatch, tmp_path, mode):
    trainer, _ = _trainer(monkeypatch, mode=mode)
    original_to = torch.Tensor.to
    original_cpu = torch.Tensor.cpu
    calls = []

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy"):
            calls.append(("owned", tensor.numel()))
        return original_to(tensor, *args, **kwargs)

    def cpu(tensor, *args, **kwargs):
        calls.append(("slice", tensor.numel()))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    _checkpoint._local_state(trainer, "slot", tmp_path)
    exported = 4 if mode in ("sparse", "metadata_subset") else 8
    assert sum(n for _, n in calls) == exported * 3 * 4
    assert sum(kind == "owned" for kind, _ in calls) == (6 if mode == "full" else 0)
    assert len(calls) == (exported + 6 if mode == "full" else exported * 4)


def test_cpu_snapshot_owns_storage_before_file_write(monkeypatch, tmp_path):
    trainer, expected = _trainer(monkeypatch)

    def mutate_then_save(tensors, filename):
        dynamic = trainer._checkpoint_slots["slot"].optimizer
        with torch.no_grad():
            for master in dynamic.master_params:
                master.add_(1000)
                for component in ("exp_avg", "exp_avg_sq"):
                    dynamic.optimizer.state[master][component].add_(1000)
        save_file(tensors, filename)

    monkeypatch.setattr("safetensors.torch.save_file", mutate_then_save)
    _checkpoint._local_state(trainer, "slot", tmp_path)
    actual = load_file(tmp_path / "block-000000.safetensors")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_failed_owned_copy_preserves_error_and_cleans_snapshot(
    monkeypatch, tmp_path, error_type
):
    trainer, _ = _trainer(monkeypatch)
    original_to = torch.Tensor.to
    error = error_type("owned copy failed")

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy"):
            raise error
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as failing:
        failing.setattr(torch.Tensor, "to", to)
        with pytest.raises(error_type) as caught:
            _checkpoint.prepare_checkpoint_save(trainer, str(tmp_path / "save"), "slot")
    assert caught.value is error
    assert not list(tmp_path.iterdir())
    assert not trainer._checkpoint_preparing_saves
    assert not trainer._prepared_checkpoint_saves
    assert trainer._checkpoint_save_next == 1
    _checkpoint.prepare_checkpoint_save(trainer, str(tmp_path / "save"), "slot")
    _checkpoint.abort_checkpoint_save(trainer, str(tmp_path / "save"))
    assert not list(tmp_path.iterdir())


def test_concurrent_prepares_keep_snapshot_order(monkeypatch, tmp_path):
    trainer, _ = _trainer(monkeypatch)
    entered, release, second_started = (threading.Event() for _ in range(3))
    original = _checkpoint._local_state
    calls, errors = [], []

    def local_state(*args):
        calls.append(args[2])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(*args)

    def prepare(path, second=False):
        try:
            if second:
                second_started.set()
            _checkpoint.prepare_checkpoint_save(trainer, str(path), "slot")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(_checkpoint, "_local_state", local_state)
    first, second = tmp_path / "first", tmp_path / "second"
    a = threading.Thread(target=prepare, args=(first,))
    b = threading.Thread(target=prepare, args=(second, True))
    try:
        a.start()
        assert entered.wait(5)
        b.start()
        assert second_started.wait(5)
        assert len(calls) == 1 and not trainer._prepared_checkpoint_saves
    finally:
        release.set()
        a.join(5)
        if b.ident is not None:
            b.join(5)
    assert not a.is_alive() and not b.is_alive() and not errors
    assert [
        trainer._prepared_checkpoint_saves[str(p)].sequence for p in (first, second)
    ] == [0, 1]
    with pytest.raises(RuntimeError, match="preparation order"):
        _checkpoint.abort_checkpoint_save(trainer, str(second))
    _checkpoint.abort_checkpoint_save(trainer, str(first))
    _checkpoint.abort_checkpoint_save(trainer, str(second))
    assert not list(tmp_path.iterdir())
