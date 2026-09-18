"""Registered head admission and the real native/remote gradient transaction."""

from dataclasses import replace
import weakref

import pytest
from test_trainer_rank_memory_admission import _requests, rank  # noqa: F401
import torch

from art.trainer_rank import ForwardOptions, TrainerRankMemoryError, _impl
from art.trainer_rank._commands import _Executor
from art.trainer_rank._heads import LiveHead, export_head
from art.trainer_rank._tensors import CotangentCollector


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))
        self.tied = self.weight
        self.frozen = torch.nn.Parameter(torch.ones(16), requires_grad=False)
        self.register_buffer("buffer", torch.ones(16))

    def forward(self, inputs):
        return inputs @ self.weight


def _register(rank, kind="module"):
    rank._checkpoint_slots["student"] = _impl._CheckpointSlot()
    if kind == "module":
        return rank.module("head", _Head, checkpoint="student")
    return rank.parameter("head", lambda: torch.ones(4, 4), checkpoint="student")


def _plan(rank):
    requests = _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[
        :1
    ]
    plan = rank._plan_flat_forward(requests)
    return replace(
        plan, groups=(replace(plan.groups[0], slot_ref=rank._slot_ref("student")),)
    )


@pytest.mark.parametrize("kind", ["module", "parameter"])
@pytest.mark.parametrize("existing_gradient", [False, True])
def test_registered_head_forward_admission_and_native_backward(
    rank, monkeypatch, kind, existing_gradient
):
    head = _register(rank, kind)
    parameter = rank._checkpoint_slots["student"].params[0]
    if existing_gradient:
        parameter.grad = torch.zeros_like(parameter)
    expected = 64 * (2 if existing_gradient else 3)
    assert rank._lora_gradient_staging_bytes(rank._slot_ref("student")) == expected
    assert rank._lora_version_capture_bytes(rank._slot_ref("student")) == 0
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 150)
    _, check = rank._admit_graph_memory(_plan(rank))
    assert not check.fits and check.estimated_required_bytes == 100 + expected
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 100 + expected)
    assert all(rank._admit_graph_memory(_plan(rank))[1].fits for _ in range(2))
    cache = rank._forward_graph_cache()
    losses = []
    for factor in (2, 3):
        handle, tensors = cache.run(
            lambda inputs: (inputs.sin(),),
            torch.ones(4, requires_grad=True),
            retention="replay",
        )
        from art.trainer_rank._tensors import detach_tree

        value = rank._forward_cotangent_collector().attach(
            detach_tree(handle, tensors)
        )[0]
        result = head(value) if kind == "module" else value @ head
        losses.append(result.sum() * factor)
    for loss in losses:
        rank.backward(loss)
    torch.testing.assert_close(
        parameter.grad,
        torch.full_like(parameter, 5 * torch.sin(torch.tensor(1.0)).item()),
    )
    assert not cache.handles()


def test_late_registration_admits_known_staging_and_preserves_old_graph(
    rank, monkeypatch
):
    rank._checkpoint_slots["student"] = _impl._CheckpointSlot()
    cache = rank._forward_graph_cache()
    handle, outputs = cache.run(
        lambda value: (value.sin(),),
        torch.ones(4, requires_grad=True),
        retention="replay",
        execution_peak_bytes=100,
        checkpoint_versions=(rank._capture_checkpoint_version("student"),),
    )
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 250)
    references = []
    tag = rank._tag_custom_parameters

    def watch(parameters):
        references.extend(weakref.ref(parameter) for parameter in parameters)
        tag(parameters)

    monkeypatch.setattr(rank, "_tag_custom_parameters", watch)
    with pytest.raises(TrainerRankMemoryError, match="292 GPU bytes") as failure:
        rank.module("head", _Head, checkpoint="student")
    assert failure.value is not None
    assert all(reference() is None for reference in references)
    assert rank._checkpoint_slots["student"].params == ()
    assert not rank._checkpoint_slots["student"].custom
    assert cache.handles() == (handle,)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 292)
    head = rank.module("head", _Head, checkpoint="student")
    from art.trainer_rank._tensors import detach_tree

    value = rank._forward_cotangent_collector().attach(detach_tree(handle, outputs))[0]
    rank.backward(head(value).sum())
    torch.testing.assert_close(
        head.weight.grad,
        torch.full_like(head.weight, torch.sin(torch.tensor(1.0)).item()),
    )


def test_remote_repeated_head_targets_stream_and_preserve_source_packets(
    rank, monkeypatch
):
    _register(rank, "parameter")
    parameter = rank._checkpoint_slots["student"].params[0]
    collector = CotangentCollector()
    live = LiveHead(export_head(rank, "student", "head"), torch.ones(4, 4), collector)
    assert isinstance(live.value, torch.Tensor)
    packets = collector.backward(
        torch.stack([(live.value * factor).sum() for factor in (2, 3, 4)]).sum()
    )
    sources = tuple(
        g.clone() for packet in packets if (g := packet.gradients[0]) is not None
    )
    assert len(sources) == len(packets)
    commit = rank._commit_versioned_gradients
    sizes = []

    def record(gradients):
        sizes.append(len(gradients))
        commit(gradients)

    monkeypatch.setattr(rank, "_commit_versioned_gradients", record)
    _Executor(rank, "zero")._backward(packets, retain_graph=False)
    assert sizes == [1, 1, 1]
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 9))
    for packet, source in zip(packets, sources, strict=True):
        torch.testing.assert_close(packet.gradients[0], source)


def test_frozen_and_buffer_only_registration_needs_no_gradient_reserve(
    rank, monkeypatch
):
    rank._checkpoint_slots["student"] = _impl._CheckpointSlot()
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 0)
    rank.buffer("buffer", lambda: torch.ones(16), checkpoint="student")
    rank.module("frozen", lambda: _Head().requires_grad_(False), checkpoint="student")
    assert rank._checkpoint_slots["student"].params == ()


@pytest.mark.parametrize("kind", ["buffer", "frozen"])
def test_nontrainable_registration_preserves_pending_graph_workspace(
    rank, monkeypatch, kind
):
    rank._checkpoint_slots["student"] = _impl._CheckpointSlot()
    cache = rank._forward_graph_cache()
    handle, outputs = cache.run(
        lambda value: (value.sin(),),
        torch.ones(4, requires_grad=True),
        retention="replay",
        execution_peak_bytes=100,
        checkpoint_versions=(rank._capture_checkpoint_version("student"),),
    )

    def register():
        if kind == "buffer":
            return rank.buffer("head", lambda: torch.ones(16), checkpoint="student")
        return rank.module(
            "head", lambda: _Head().requires_grad_(False), checkpoint="student"
        )

    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 99)
    with pytest.raises(TrainerRankMemoryError, match="100 GPU bytes"):
        register()
    assert not rank._checkpoint_slots["student"].custom
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 100)
    register()
    assert rank._checkpoint_slots["student"].params == ()
    from art.trainer_rank._tensors import detach_tree

    value = rank._forward_cotangent_collector().attach(detach_tree(handle, outputs))[0]
    rank.backward(value.sum())
    assert not cache.handles()
