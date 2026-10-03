"""Logical placement must preserve native pending-backward reservations."""

from types import SimpleNamespace

import pytest
from test_trainer_rank_memory_admission import _requests, rank  # noqa: F401
import torch

from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput, _impl
from art.trainer_rank._commands import _Executor, _OutputPacket, _view
from art.trainer_rank._tensors import detach_tree


def _output(size=80, *, policy="auto", handle="new"):
    return (
        _OutputPacket(
            detach_tree(handle, ForwardOutput(None, None, None, torch.ones(size // 4))),
            (False,),
            False,
        ),
        ForwardInput(
            input_tokens=torch.tensor([1]), options=ForwardOptions(output_device=policy)
        ),
    )


def _pending(rank, monkeypatch, *states):
    monkeypatch.setattr(
        rank,
        "_graph_cache",
        SimpleNamespace(
            handles=lambda: tuple(range(len(states))), state=states.__getitem__
        ),
        raising=False,
    )


@pytest.mark.parametrize("policy", ["auto", "model", "cpu"])
def test_logical_copy_preserves_pending_restore(rank, monkeypatch, policy):
    _pending(rank, monkeypatch, SimpleNamespace(restore_workspace_bytes=100))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda **_: 120)
    view = _view(_Executor(rank, "zero"))
    if policy == "model":
        with pytest.raises(MemoryError, match="only 20 bytes"):
            view._place_outputs([_output(policy=policy)])
    else:
        assert view._place_outputs([_output(policy=policy)])[0].cpu == (True,)


def test_required_copy_reuses_cache_but_optional_keeps_physical_budget(
    rank, monkeypatch
):
    _pending(rank, monkeypatch, SimpleNamespace(restore_workspace_bytes=100))
    monkeypatch.setattr(
        rank,
        "_available_memory_bytes",
        lambda reusable_cache=False: 300 if reusable_cache else 120,
    )
    view = _view(_Executor(rank, "zero"))
    planned = view._place_outputs(
        [_output(policy="model", handle="a"), _output(policy="auto", handle="b")]
    )
    assert [output.cpu for output in planned] == [(False,), (True,)]
    with pytest.raises(MemoryError, match="only 200 bytes"):
        view._place_outputs([_output(240, policy="model")])


def test_logical_copy_reserves_distinct_checkpoints_and_standalone_heads(
    rank, monkeypatch
):
    parameters = [torch.nn.Parameter(torch.ones(size)) for size in (16, 8, 4, 64)]
    parameters[1].grad = torch.ones_like(parameters[1])
    rank._tag_custom_parameters(parameters[2:3])
    for name, parameter in zip(("first", "second", "head_only", "unused"), parameters):
        rank._checkpoint_slots[name] = _impl._CheckpointSlot(params=(parameter,))
    _pending(
        rank,
        monkeypatch,
        *(
            SimpleNamespace(
                restore_workspace_bytes=workspace,
                checkpoint_versions=(rank._capture_checkpoint_version(name),),
            )
            for name, workspace in (("first", 80), ("first", 100), ("second", 60))
        ),
    )
    assert rank._pending_backward_memory() == (100, 192 + 64 + 48)
    assert rank._pending_backward_memory(exclude_staging=("first",)) == (100, 112)
    free = 484
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda **_: free)
    view = _view(_Executor(rank, "zero"))
    planned = view._place_outputs([_output(handle="a"), _output(handle="b")])
    assert [output.cpu for output in planned] == [(False,), (True,)]
    free -= 80  # First wave's GPU output is now live.
    assert view._place_outputs([_output()])[0].cpu == (True,)
    parameters[1].grad = None  # Staging is recomputed, not cached with prior placement.
    assert rank._pending_backward_memory() == (100, 192 + 96 + 48)


def test_client_cpu_transport_does_not_consume_worker_gpu_reserve(rank, monkeypatch):
    view = _view(_Executor(rank, "zero"))
    view._transport_handles = []
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda **_: pytest.fail("GPU query")
    )
    assert view._place_outputs([_output(policy="model")])[0].cpu == (True,)
    assert view._transport_handles == ["new"]


def test_native_admission_keeps_standalone_head_staging(rank, monkeypatch):
    rank._checkpoint_slots["head_only"] = _impl._CheckpointSlot()
    rank.parameter("head", lambda: torch.ones(1), checkpoint="head_only")
    plan = rank._plan_flat_forward(
        _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[:1]
    )
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda **_: 111)
    _, check = rank._admit_graph_memory(plan)
    assert not check.fits and check.estimated_required_bytes == 112


@pytest.mark.parametrize("available,pending", [(0, 0), (100, 200)])
def test_oversized_model_outputs_keep_caller_device_after_forward(
    rank, monkeypatch, available, pending
):
    # Captured 060 second wave: 159440 FP32 selected-token logprobs.
    _pending(rank, monkeypatch, SimpleNamespace(restore_workspace_bytes=pending))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda **_: available)
    view = _view(_Executor(rank, "zero"))
    output = _output(637760, policy="model")
    with pytest.raises(MemoryError, match="require 637760 GPU bytes; only 0"):
        view._place_outputs([output])
    rank._allow_oversized_batches = True
    assert view._place_outputs([output])[0].cpu == (False,)
    # The opt-in neither changes requested output devices nor invents auto credit.
    assert view._place_outputs([_output(policy="auto")])[0].cpu == (True,)
    assert view._place_outputs([_output(policy="cpu")])[0].cpu == (True,)


def test_oversized_default_model_output_uses_retained_060_wave_geometry(
    rank, monkeypatch
):
    # Sanitized lengths from the retained first 060 batch, after its first
    # 11097-position microbatch. No token values or model execution are needed.
    lengths = (
        10129,
        9839,
        9315,
        10376,
        10299,
        11220,
        10617,
        10571,
        12260,
        9945,
        9970,
        9859,
        10404,
        12087,
        12549,
    )
    requests = [
        ForwardInput(input_tokens=torch.arange(n), target_tokens=torch.arange(n))
        for n in lengths
    ]
    output = _OutputPacket(
        detach_tree(
            "retained-060",
            [ForwardOutput(torch.ones(n), None, None, None) for n in lengths],
        ),
        (False,) * len(lengths),
        False,
    )
    assert sum(t.numel() * t.element_size() for t in output.packet.tensors) == 637760
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda **_: 0)
    view = _view(_Executor(rank, "zero"))
    with pytest.raises(MemoryError, match="637760"):
        view._place_outputs([(output, requests)])
    rank._allow_oversized_batches = True
    assert view._place_outputs([(output, requests)])[0].cpu == (False,) * 15
