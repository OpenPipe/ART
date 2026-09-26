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
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 120)
    view = _view(_Executor(rank, "zero"))
    released = []
    monkeypatch.setattr(view, "_invoke", lambda *args: released.append(args))
    if policy == "model":
        with pytest.raises(MemoryError, match="only 20 bytes"):
            view._place_outputs([_output(policy=policy)])
        assert released == [("release", ("new",))]
    else:
        assert view._place_outputs([_output(policy=policy)])[0].cpu == (True,)
        assert released == []


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
