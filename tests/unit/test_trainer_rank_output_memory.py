"""Logical placement must preserve native pending-backward reservations."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
from test_trainer_rank_memory_admission import _requests, rank  # noqa: F401
import torch

from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput, _impl, _memory
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
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 120)
    view = _view(_Executor(rank, "zero"))
    if policy == "model":
        with pytest.raises(MemoryError, match="only 20 bytes"):
            view._place_outputs([_output(policy=policy)])
    else:
        assert view._place_outputs([_output(policy=policy)])[0].cpu == (True,)


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
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: free)
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
        rank, "_available_memory_bytes", lambda: pytest.fail("GPU query")
    )
    assert view._place_outputs([_output(policy="model")])[0].cpu == (True,)
    assert view._transport_handles == ["new"]


def test_native_admission_keeps_standalone_head_staging(rank, monkeypatch):
    rank._checkpoint_slots["head_only"] = _impl._CheckpointSlot()
    rank.parameter("head", lambda: torch.ones(1), checkpoint="head_only")
    plan = rank._plan_flat_forward(
        _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[:1]
    )
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 111)
    _, check = rank._admit_graph_memory(plan)
    assert not check.fits and check.estimated_required_bytes == 112


@pytest.mark.parametrize("available,pending", [(0, 0), (100, 200)])
def test_oversized_model_outputs_keep_caller_device_after_forward(
    rank, monkeypatch, available, pending
):
    # Captured 060 second wave: 159440 FP32 selected-token logprobs.
    _pending(rank, monkeypatch, SimpleNamespace(restore_workspace_bytes=pending))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: available)
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
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 0)
    view = _view(_Executor(rank, "zero"))
    with pytest.raises(MemoryError, match="637760"):
        view._place_outputs([(output, requests)])
    rank._allow_oversized_batches = True
    assert view._place_outputs([(output, requests)])[0].cpu == (False,) * 15


_GIB = 1024**3
# torch.cuda.mem_get_info total on an H200.
_H200_BYTES = 150_754_820_096


class _NativeAllocator:
    """Native caching-allocator counters: physical free excludes cached blocks."""

    def __init__(self, monkeypatch, *, allocated, reserved):
        self.allocated, self.reserved, self.releases = allocated, reserved, 0
        cuda = torch.cuda
        monkeypatch.setattr(cuda, "is_available", lambda: True)
        monkeypatch.setattr(cuda, "get_allocator_backend", lambda: "native")
        monkeypatch.setattr(
            cuda,
            "mem_get_info",
            lambda device=None: (_H200_BYTES - self.reserved, _H200_BYTES),
        )
        monkeypatch.setattr(
            cuda, "memory_allocated", lambda device=None: self.allocated
        )
        monkeypatch.setattr(cuda, "memory_reserved", lambda device=None: self.reserved)
        monkeypatch.setattr(cuda, "empty_cache", self.empty_cache)
        monkeypatch.setattr(cuda, "device", lambda device: nullcontext())

    def empty_cache(self):
        self.releases += 1
        self.reserved = self.allocated


def _native_rank(rank, monkeypatch, *, allocated, reserved, restore):
    monkeypatch.setattr(rank, "device", torch.device("cuda", 0))
    monkeypatch.setattr(
        rank,
        "_available_memory_bytes",
        MethodType(_memory._available_memory_bytes, rank),
    )
    _pending(rank, monkeypatch, SimpleNamespace(restore_workspace_bytes=restore))
    return _NativeAllocator(monkeypatch, allocated=allocated, reserved=reserved)


def test_model_outputs_reclaim_forward_cache_before_refusing(rank, monkeypatch):
    # Shaped like 062's first large wave on one H200 (220420 B of FP32
    # logprobs): a wave larger than its predecessors left 15 GiB of forward
    # transients cached beside its graph. Physical free (5.4 GiB) minus the 3%
    # reserve and the full 6 GiB backward reserve is negative, though the
    # backward reuses those cached blocks.
    allocator = _native_rank(
        rank, monkeypatch, allocated=120 * _GIB, reserved=135 * _GIB, restore=6 * _GIB
    )
    view = _view(_Executor(rank, "rank"))
    output = _output(220420, policy="model")
    assert view._place_outputs([output])[0].cpu == (False,)
    assert allocator.releases == 1 and allocator.reserved == allocator.allocated


def test_output_cache_release_only_replaces_a_refusal(rank, monkeypatch):
    allocator = _native_rank(
        rank, monkeypatch, allocated=120 * _GIB, reserved=128 * _GIB, restore=6 * _GIB
    )
    view = _view(_Executor(rank, "rank"))
    # Fitting model copies and over-budget auto copies keep their placement
    # without touching the allocator.
    assert view._place_outputs([_output(220420, policy="model")])[0].cpu == (False,)
    assert view._place_outputs([_output(4 * _GIB)])[0].cpu == (True,)
    assert allocator.releases == 0
    # Nothing cached: the refusal stands, without a release.
    allocator.reserved = allocator.allocated = 136 * _GIB
    with pytest.raises(MemoryError, match="require 220420 GPU bytes; only 0"):
        view._place_outputs([_output(220420, policy="model")])
    assert allocator.releases == 0
    # A release that still leaves too little refuses with the fresh sample.
    allocator.allocated = 128 * _GIB
    with pytest.raises(MemoryError, match=r"require 4294967296 GPU bytes; only \d+"):
        view._place_outputs([_output(4 * _GIB, policy="model")])
    assert allocator.releases == 1
    # The oversized opt-in already permits the copy: no release.
    allocator.reserved = 136 * _GIB
    rank._allow_oversized_batches = True
    assert view._place_outputs([_output(220420, policy="model")])[0].cpu == (False,)
    assert allocator.releases == 1
