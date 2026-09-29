"""Tiny CPU captures through the real collector; no CUDA performance claim."""

from importlib.util import find_spec
import threading
from typing import Any, cast

import pytest
import safetensors.torch
from test_trainer_rank_validation import _save_state_trainer
import torch

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
    captured, config, records = cp._local_state(trainer, "a", payloads)
    # Expansion must not consult live model, optimizer, or expert-layout state.
    with torch.no_grad():
        for value in (*params, *masters):
            value.add_(100)
        for state in optimizer.state.values():
            for value in state.values():
                value.add_(100)
    for model in trainer.runtime.model:
        if hasattr(model, "expert_ids"):
            model.expert_ids = (99, 98, 97)
            model.adapter_model_prefix = "changed"
    shards = cp._expand_local_state(captured, payloads)
    if allocation_guard:
        assert len(zeros) == len(params) * (2 - len(present))
        captured_storage = {
            value.untyped_storage().data_ptr()
            for key, value in payloads[optimizer_file].items()
            if key.startswith("master/")
        }
        assert all(
            value.untyped_storage().data_ptr() in captured_storage for value in zeros
        )
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


def test_capture_copies_physical_parameters_once(captured_state, monkeypatch):
    trainer, params, masters, optimizer, *_ = captured_state
    for master in masters:
        optimizer.state[master] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.ones_like(master),
            "exp_avg_sq": torch.ones_like(master),
        }
    copies = []
    original = torch.Tensor.to

    def copy(value, *args, **kwargs):
        if kwargs.get("copy"):
            copies.append(value.numel())
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy)
    cp._local_state(trainer, "a", {})
    assert len(copies) == 4 * len(params)
    assert sum(copies) == 4 * sum(param.numel() for param in params)


def test_expert_expansion_releases_capture_and_preserves_snapshot(
    captured_state, tmp_path, monkeypatch
):
    trainer, params, masters, optimizer, *_ = captured_state
    if trainer._checkpoint_slots["a"].custom:
        pytest.skip("custom state does not expand expert keys")
    from art.megatron.weights.lora_publish import collect_local_lora_entries

    tensors, metadata = collect_local_lora_entries(
        trainer.runtime.model, {}, owner_rank=0
    )
    expected = {key: value.clone() for key, value in tensors.items()}
    config = {"base_model_name_or_path": "test/model", "r": 2}
    trainer._checkpoint_slots["a"].config = config
    monkeypatch.setattr(cp, "_validate_save_state", lambda *_: config)
    entered, release = threading.Event(), threading.Event()
    expand = cp._expand_local_state

    def blocked(captured, files):
        assert threading.current_thread().name == "checkpoint-snapshot"
        entered.set()
        assert release.wait(5), "capture did not release expert expansion"
        return expand(captured, files)

    monkeypatch.setattr(cp, "_expand_local_state", blocked)
    output = str(tmp_path / "checkpoint")
    try:
        trainer.prepare_checkpoint_save(output, "a")
        assert entered.wait(5)
        prepared = trainer._prepared_checkpoint_saves[output]
        assert prepared.writer is not None and not prepared.writer.done()
        with torch.no_grad():
            for value in (*params, *masters):
                value.add_(100)
        for model in trainer.runtime.model:
            model.expert_ids = (90, 91, 92)
            model.adapter_model_prefix = "changed"
        config["r"] = 99
        assert prepared.config["r"] == 2
    finally:
        release.set()

    def finish(_trainer, owned):
        assert [shard.metadata for shard in owned.shards] == metadata
        saved = safetensors.torch.load_file(owned.snapshot / owned.shards[0].file)
        for key, value in expected.items():
            torch.testing.assert_close(saved[f"lora/{key}"], value, rtol=0, atol=0)

    monkeypatch.setattr(cp, "_finish", finish)
    trainer.finish_checkpoint_save(output)
    assert not trainer._prepared_checkpoint_saves
    assert not prepared.snapshot.exists()
