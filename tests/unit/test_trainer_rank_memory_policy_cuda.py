"""Real allocator/host measurements; run only on a reserved validation GPU."""

from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._graphs import GraphCache
from art.trainer_rank._impl import _CheckpointSlot
from art.trainer_rank._memory_policy import (
    ForwardMemoryCost,
    host_memory_budget,
    placement_cost,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="requires a reserved GPU"
)


@pytest.mark.parametrize("existing_gradient", [False, True])
def test_sequential_gradient_publications_fit_original_admission_reserve(
    monkeypatch, existing_gradient
):
    trainer = TrainerRank.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.ones(4 * 1024**2, device="cuda"))
    source = torch.ones_like(parameter)
    if existing_gradient:
        parameter.grad = torch.zeros_like(parameter)
    trainer.device = parameter.device
    trainer.runtime = SimpleNamespace(model=[], optimizer=None)
    trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(parameter,))}
    monkeypatch.setattr(
        trainer, "_iter_slot_parameters", lambda _ref: iter((parameter,))
    )
    ref = trainer._slot_ref("student")
    # Two pending forwards observe the same pre-backward gradient state.
    reserved = min(trainer._lora_gradient_staging_bytes(ref) for _ in range(2))
    state = trainer._version_state()
    version = state.capture("student")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with trainer._gradient_transaction():
            state.accumulate(((version, 2, parameter, source),))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    assert peak <= reserved
    assert peak == parameter.numel() * parameter.element_size() * (
        2 if existing_gradient else 3
    )
    print(
        "GRADIENT_RESERVATION="
        + json.dumps(
            {
                "existing_gradient": existing_gradient,
                "reserved_bytes": reserved,
                "peak_bytes": peak,
            }
        )
    )
    torch.testing.assert_close(parameter.grad, source * 2)


def _rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


@pytest.fixture(scope="module")
def workload():
    torch.manual_seed(1729)
    inputs = torch.randn(4096, 1024)
    weight = torch.nn.Parameter(torch.randn(1024, device="cuda"))

    def execute(value):
        return (value.to("cuda").sin() * weight,)

    # Warm kernels/gradient allocation before measuring the same real workload.
    execute(inputs)[0].sum().backward()
    weight.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (output,) = execute(inputs)
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - baseline
    output_bytes = output.numel() * output.element_size()
    output.backward(torch.ones_like(output))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    assert weight.grad is not None
    expected = weight.grad.detach().clone()
    weight.grad = None
    del output
    cost = ForwardMemoryCost(
        peak_bytes=int(peak * 1.1),
        retained_bytes=int(retained * 1.1),
        output_bytes=output_bytes,
        replay_bytes=inputs.numel() * inputs.element_size() + 65536,
    )
    return inputs, weight, execute, expected, cost


def _run_root(workload, state, device):
    inputs, weight, execute, expected, cost = workload
    cache = GraphCache()
    weight.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline, rss_before = torch.cuda.memory_allocated(), _rss()
    torch.cuda.reset_peak_memory_stats()
    storage = weight.untyped_storage().data_ptr()
    handles, outputs = [], []
    for _ in range(4):
        handle, (output,) = cache.run(
            execute,
            inputs,
            retention=state,
            output_device="cpu" if device == "cpu" else "cuda",
            cuda_devices=[torch.cuda.current_device()],
            keep_on_device=lambda tensor: (
                tensor.untyped_storage().data_ptr() == storage
            ),
        )
        handles.append(handle)
        outputs.append(output)
    torch.cuda.synchronize()
    forward_retained = torch.cuda.memory_allocated() - baseline
    records = [cache.state(handle) for handle in handles]
    rss_retained = _rss() - rss_before
    caller_bytes = sum(output.numel() * output.element_size() for output in outputs)
    # Stream cotangents from CPU so this measures runtime restoration workspace,
    # independently of arbitrary caller loss graphs/temporary CUDA tensors.
    cache.backward_many(
        tuple(
            (handle, (torch.ones(output.shape, dtype=output.dtype),))
            for handle, output in zip(handles, outputs, strict=True)
        )
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    torch.testing.assert_close(weight.grad, expected * 4, rtol=2e-5, atol=2e-4)
    assert not cache.handles()
    planned = placement_cost((cost,) * 4, backward_state=state, output_device=device)
    measurement = {
        "state": state,
        "output_device": device,
        "gpu_forward_retained_bytes": forward_retained,
        "gpu_peak_bytes": peak,
        "rss_forward_delta_bytes": rss_retained,
        "cache_gpu_bytes": sum(record.gpu_bytes for record in records),
        "cache_cpu_bytes": sum(record.cpu_bytes for record in records),
        "caller_output_bytes": caller_bytes,
        "planned": asdict(planned),
        "host_budget_two_ranks": asdict(host_memory_budget(local_world_size=2)),
    }
    print("MEMORY_MEASUREMENT=" + json.dumps(measurement, sort_keys=True))
    assert peak <= planned.gpu_required_bytes + 8 * 1024**2
    if device == "cpu" and state == "replay":
        assert forward_retained < 1024**2
    if state in ("cpu", "replay"):
        assert (
            sum(record.cpu_bytes for record in records)
            >= cost.replay_bytes * 4 - 4 * 65536
        )
    del outputs
    return measurement


@pytest.mark.parametrize("state", ["gpu", "cpu", "replay"])
@pytest.mark.parametrize("device", ["model", "cpu"])
def test_real_root_memory_and_gradient_policy(workload, state, device):
    _run_root(workload, state, device)


def test_real_root_replay_with_cpu_outputs_fits_budget(workload):
    *_, cost = workload
    budget = cost.peak_bytes + 8 * 1024**2
    cpu_budget = host_memory_budget(local_world_size=2).available_bytes
    retained = placement_cost((cost,) * 4, backward_state="gpu", output_device="model")
    replay = placement_cost((cost,) * 4, backward_state="replay", output_device="cpu")
    assert replay.gpu_required_bytes <= budget < retained.gpu_required_bytes
    assert replay.cpu_required_bytes <= cpu_budget
    measured = _run_root(workload, "replay", "cpu")
    assert measured["gpu_peak_bytes"] <= budget
