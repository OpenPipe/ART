"""Tiny CPU captures through the real collector; no CUDA performance claim."""

import gc
from importlib.util import find_spec
import traceback
from typing import Any, cast

import pytest
import safetensors.torch
from test_trainer_rank_validation import _adapter_config, _save_state_trainer
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from art.trainer_rank import _checkpoint as cp
from art.trainer_rank._impl import _CheckpointSlot, _CustomObject, _DynamicOptimizer


@pytest.fixture(params=("custom", "dense", "expert"))
def captured_state(request, monkeypatch):
    if find_spec("megatron") is None:
        pytest.skip("requires Megatron")
    from art.megatron import lora as lm
    from art.megatron.weights.lora_publish import collect_local_lora_entries

    for name in (
        "get_expert_model_parallel_rank",
        "get_expert_data_parallel_rank",
        "get_data_parallel_rank",
    ):
        monkeypatch.setattr(lm.ps, name, lambda **_: 0)
    trainer = _save_state_trainer()
    monkeypatch.setattr(trainer, "_slot_ref", lambda _: None)
    custom = {}
    expected = {}
    if request.param == "custom":
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3).T)
        custom = {"p": _CustomObject("parameter", parameter, object())}
        params = (parameter,)
        entries = [("p", parameter, None)]
        expected["custom_tensors.safetensors"] = {"p": parameter.detach().clone()}
        optimizer_file = "optimizer/custom.safetensors"
    else:
        model = lm.LoRA(
            "layer.experts.{expert}.q_proj"
            if request.param == "expert"
            else "layer.q_proj",
            3,
            4,
            2,
            2,
            torch.bfloat16,
            torch.device("cpu"),
            num_local_experts=3 if request.param == "expert" else 1,
        )
        if request.param == "expert":
            model.bind_expert_layout((8, None, 10), ())
        params = tuple(model.parameters())
        entries = model._export_items()
        tensors, _ = collect_local_lora_entries(cast(Any, [model]), {}, owner_rank=0)
        optimizer_file = "block-000000.safetensors"
        expected[optimizer_file] = {
            f"lora/{key}": value.clone() for key, value in tensors.items()
        }
    masters = tuple(
        torch.nn.Parameter(
            torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) + 1
        )
        for param in params
    )
    optimizer = torch.optim.Adam(masters, lr=0.125)
    trainer.runtime.model = cast(Any, [model])
    trainer._checkpoint_slots["a"] = _CheckpointSlot(
        params=params,
        optimizer=_DynamicOptimizer(optimizer, masters),
        custom=custom,
    )
    return trainer, params, masters, optimizer, entries, expected, optimizer_file


@pytest.mark.parametrize("present", ((), ("exp_avg",), ("exp_avg", "exp_avg_sq")))
@pytest.mark.parametrize(
    "allocation_guard", (False, True), ids=("payload", "allocations")
)
def test_moment_capture(captured_state, monkeypatch, present, allocation_guard):
    trainer, params, masters, optimizer, entries, expected, optimizer_file = (
        captured_state
    )
    by_param = dict(zip(map(id, params), masters, strict=True))
    for master in masters:
        optimizer.state[master] = {
            "step": torch.tensor(7.0),
            **{
                component: torch.full_like(master, index + 0.25)
                for index, component in enumerate(present)
            },
        }
    is_custom = bool(trainer._checkpoint_slots["a"].custom)
    expected_optimizer = expected.setdefault(optimizer_file, {})
    for key, param, expert in entries:
        master = by_param[id(param)]
        for component in ("master", "exp_avg", "exp_avg_sq"):
            value = (
                master
                if component == "master"
                else optimizer.state[master].get(component)
            )
            if value is None:
                value = torch.zeros_like(master)
            local = value if expert is None else value[expert]
            expected_optimizer[f"{component}/{key}"] = (
                (local if is_custom else local.T).detach().clone()
            )
        expected_optimizer[f"step/{key}"] = torch.tensor(7.0)

    live_storage = {value.untyped_storage().data_ptr() for value in (*params, *masters)}
    zeros = []
    original = torch.zeros_like

    def owned_cpu_zero(value, *args, **kwargs):
        assert value.device.type == "cpu"
        assert value.untyped_storage().data_ptr() not in live_storage, (
            "zero fallback allocated from live optimizer storage"
        )
        zeros.append(value)
        return original(value, *args, **kwargs)

    contiguous = torch.Tensor.contiguous

    def pack_owned(value, *args, **kwargs):
        assert value.untyped_storage().data_ptr() not in live_storage, (
            "capture packed live LoRA storage before its immutable CPU copy"
        )
        return contiguous(value, *args, **kwargs)

    payloads = {}
    if allocation_guard:
        monkeypatch.setattr(torch, "zeros_like", owned_cpu_zero)
        monkeypatch.setattr(torch.Tensor, "contiguous", pack_owned)
    shards, config, records = cp._local_state(trainer, "a", payloads)
    if allocation_guard:
        assert len(zeros) == len(entries) * (2 - len(present))
        captured_masters = {
            id(value)
            for key, value in payloads[optimizer_file].items()
            if key.startswith("master/")
        }
        assert all(id(value) in captured_masters for value in zeros)
    assert bool(records) is is_custom
    assert len(shards) == (0 if is_custom else len(entries))
    assert config == cp._optimizer_config(trainer._checkpoint_slots["a"].optimizer)
    assert payloads.keys() == expected.keys()
    for filename, tensors in payloads.items():
        assert tensors.keys() == expected[filename].keys()
        for key, value in tensors.items():
            reference = expected[filename][key]
            assert value.device.type == "cpu" and not value.requires_grad
            assert value.dtype == reference.dtype and value.shape == reference.shape
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert safetensors.torch.save(
            {key: value.contiguous() for key, value in tensors.items()}
        ) == safetensors.torch.save(
            {key: value.contiguous() for key, value in expected[filename].items()}
        )
    with torch.no_grad():
        for value in (*params, *masters):
            value.add_(100)
        for state in optimizer.state.values():
            for value in state.values():
                value.add_(100)
    for filename, tensors in payloads.items():
        for key, value in tensors.items():
            torch.testing.assert_close(value, expected[filename][key], rtol=0, atol=0)
    saved = payloads[optimizer_file]
    for key, _, _ in entries:
        before = saved[f"exp_avg_sq/{key}"].clone()
        saved[f"exp_avg/{key}"].add_(100)
        torch.testing.assert_close(saved[f"exp_avg_sq/{key}"], before, rtol=0, atol=0)


@pytest.mark.parametrize("captured_state", ("custom", "dense"), indirect=True)
def test_failed_capture_releases_partial_copies(captured_state, tmp_path, monkeypatch):
    trainer, params, masters, _, _, expected, _ = captured_state
    trainer._checkpoint_slots["a"].config = _adapter_config(
        rank=2, alpha=2, target_modules=("q_proj",)
    )
    borrowed = [
        StorageWeakRef(value.untyped_storage()) for value in (*params, *masters)
    ]
    sentinel = torch.tensor([13.0])
    copies = []
    calls = 0
    error, cause = OSError("CPU capture copy failed"), RuntimeError("copy cause")
    original = torch.Tensor.to

    def copy(value, *args, **kwargs):
        nonlocal calls
        foreign_marker = sentinel
        if kwargs.get("copy"):
            calls += 1
            if calls == 2:
                try:
                    raise cause
                except RuntimeError:
                    raise error from cause
        result = original(value, *args, **kwargs)
        if kwargs.get("copy"):
            copies.append(StorageWeakRef(result.untyped_storage()))
        assert foreign_marker is sentinel
        return result

    output = str(tmp_path / "reusable")
    enabled = gc.isenabled()
    gc.disable()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", copy)
            with pytest.raises(OSError) as caught:
                trainer.prepare_checkpoint_save(output, "a")
        assert caught.value is error and error.__cause__ is cause
        for failure in (error, cause):
            frame = next(
                frame
                for frame, _ in traceback.walk_tb(failure.__traceback__)
                if frame.f_code is copy.__code__
            )
            assert frame.f_locals["foreign_marker"] is sentinel
        assert calls == 2 and len(copies) == 1 and copies[0].expired()
        assert all(not storage.expired() for storage in borrowed)
        assert not trainer._checkpoint_preparing_saves
        assert not trainer._prepared_checkpoint_saves
        assert not list(tmp_path.glob(".reusable.*"))
        trainer.prepare_checkpoint_save(output, "a")
        prepared = trainer._prepared_checkpoint_saves[output]
        assert prepared.writer is not None
        prepared.writer.result(3)
        for filename, tensors in expected.items():
            actual = safetensors.torch.load_file(prepared.snapshot / filename)
            for key, reference in tensors.items():
                torch.testing.assert_close(actual[key], reference, rtol=0, atol=0)
    finally:
        if enabled:
            gc.enable()
        for pending in list(trainer._prepared_checkpoint_saves):
            trainer.abort_checkpoint_save(pending)
