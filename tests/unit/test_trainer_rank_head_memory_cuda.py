"""Allocator assertions; requires a validation-owned GPU reservation."""

import json
import os

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from art.trainer_rank import TrainerRankMemoryError
from art.trainer_rank._commands import _Executor
from art.trainer_rank._heads import LiveHead, export_head
from art.trainer_rank._tensors import CotangentCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="reserved GPU required"
)


@pytest.mark.parametrize("existing_gradient", [False, True])
def test_repeated_remote_head_cotangents_fit_original_gradient_reserve(
    existing_gradient,
):
    torch.set_num_threads(2)
    trainer, api = _trainer("student")
    trainer.device = torch.device("cuda")
    parameter = api.parameter(
        "head", lambda: torch.ones(4 * 1024**2), checkpoint="student"
    )
    if existing_gradient:
        parameter.grad = torch.ones_like(parameter)
    collector = CotangentCollector()
    live = LiveHead(
        export_head(trainer, "student", "head"), torch.ones(parameter.shape), collector
    )
    assert isinstance(live.value, torch.Tensor)
    packets = collector.backward(
        torch.stack([(live.value * factor).sum() for factor in range(1, 9)]).sum()
    )
    assert all(
        gradient.device.type == "cpu"
        for packet in packets
        for gradient in packet.gradients
        if gradient is not None
    )
    reserve = trainer._lora_gradient_staging_bytes(trainer._slot_ref("student"))
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    executor = _Executor(trainer, "zero")
    for _ in range(2):
        executor._backward(packets, retain_graph=False)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    assert peak <= reserve
    torch.testing.assert_close(
        parameter.grad, torch.full_like(parameter, 73 if existing_gradient else 72)
    )
    print(
        "REMOTE_HEAD_RESERVATION="
        + json.dumps(
            dict(
                existing_gradient=existing_gradient,
                peak=peak,
                reserve=reserve,
                repeated_captures=8,
            )
        )
    )


@pytest.mark.parametrize("kind", ["parameter", "buffer", "frozen"])
def test_rejected_late_registration_releases_gpu_storage_with_live_traceback(
    monkeypatch, kind
):
    trainer, api = _trainer("student")
    trainer.device = torch.device("cuda")
    cache = trainer._forward_graph_cache()
    handle, _ = cache.run(
        lambda value: (value.sin(),),
        torch.ones(1024, device="cuda", requires_grad=True),
        retention="replay",
        execution_peak_bytes=1024**2,
        checkpoint_versions=(trainer._capture_checkpoint_version("student"),),
    )
    monkeypatch.setattr(trainer, "_available_memory_bytes", lambda: 1024**2 - 1)
    storages = []
    initialize = trainer._initialize_custom_object

    def watch(checkpoint, name, custom):
        initialize(checkpoint, name, custom)
        tensors = (
            custom.value.parameters()
            if isinstance(custom.value, torch.nn.Module)
            else (custom.value,)
        )
        storages.extend(StorageWeakRef(tensor.untyped_storage()) for tensor in tensors)

    monkeypatch.setattr(trainer, "_initialize_custom_object", watch)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    with pytest.raises(TrainerRankMemoryError) as failure:
        if kind == "parameter":
            api.parameter("late", lambda: torch.ones(4 * 1024**2), checkpoint="student")
        elif kind == "buffer":
            trainer.buffer(
                "late", lambda: torch.ones(4 * 1024**2), checkpoint="student"
            )
        else:
            api.module(
                "late",
                lambda: torch.nn.Linear(1024, 4096, bias=False).requires_grad_(False),
                checkpoint="student",
            )
    assert failure.value is not None
    torch.cuda.synchronize()
    assert all(storage.expired() for storage in storages)
    assert torch.cuda.memory_allocated() <= baseline
    assert not trainer._checkpoint_slots["student"].custom
    cache.release(handle)
    print(
        "LATE_HEAD_RELEASE="
        + json.dumps(
            dict(
                kind=kind, retained_extra_bytes=torch.cuda.memory_allocated() - baseline
            )
        )
    )
