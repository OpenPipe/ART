"""Real CPU graph/bridge engines with simulated CUDA observer devices and tails.

Only physical model tensors expose a CUDA device to the observer. Caller CPU
proxies retain their actual device, so attaching to them disables accounting.
This exercises engine boundaries, not native CUDA timing or memory behavior.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
from test_trainer_rank_backward_work import CUDA, Clock
from test_trainer_rank_validation import _runtime
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    ForwardOutput,
    TrainerRank,
    _backward_work,
    run_rank_callback,
)
from art.trainer_rank._tensors import CotangentCollector

MODEL_NS = 1_000_000
CALLER_NS = 1_000_000_000


class _PhysicalTensor:
    device = torch.device("cuda:0")

    def __init__(self, tensor, released):
        self.ref = weakref.ref(tensor, released)

    @property
    def requires_grad(self):
        tensor = self.ref()
        assert tensor is not None
        return tensor.requires_grad

    def register_hook(self, callback):
        tensor = self.ref()
        assert tensor is not None
        return tensor.register_hook(callback)


@pytest.fixture
def rig(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    rank = TrainerRank(_runtime(model))
    weight = model.weight
    with torch.no_grad():
        weight.fill_(2)
    clock, cuda = Clock(), CUDA()
    # Preserve the real engine task IDs and completion callbacks. Only device
    # labels, event readiness, and elapsed time are controlled by this fixture.
    monkeypatch.setattr(_backward_work, "time", clock)
    monkeypatch.setattr(
        _backward_work,
        "torch",
        SimpleNamespace(
            cuda=cuda, _C=torch._C, compiler=torch.compiler, autograd=torch.autograd
        ),
    )
    work = _backward_work.BackwardWork(
        rank._recovery_state().lock, _PhysicalTensor.device
    )
    monkeypatch.setattr(rank, "_backward_work", lambda: work)
    physical, references = {}, []
    state = SimpleNamespace(before_backward=lambda: None, forwards=0)
    attach = work.attach

    def observe(outputs):
        attach(
            [
                replace(
                    output,
                    hidden_states=physical.get(
                        id(output.hidden_states), output.hidden_states
                    ),
                )
                for output in outputs
            ]
        )

    monkeypatch.setattr(work, "attach", observe)

    class Model(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter, tokens):
            ctx.save_for_backward(tokens)
            state.forwards += 1
            clock.value += CALLER_NS
            return parameter.sum() * tokens

        @staticmethod
        def backward(ctx, *gradients):
            (gradient,) = gradients
            state.before_backward()
            clock.value += MODEL_NS
            (tokens,) = ctx.saved_tensors
            return (gradient * tokens).sum().reshape_as(weight), None

    def forward(items, prepared):
        outputs = []
        for item in items:
            value = Model.apply(weight, item.input_ids.float())
            key = id(value)
            physical[key] = _PhysicalTensor(
                value, lambda _, key=key: physical.pop(key, None)
            )
            references.append(weakref.ref(value))
            outputs.append(ForwardOutput(None, None, None, value))
        return outputs

    monkeypatch.setattr(rank, "_topology", lambda: ParallelTopology())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_configure_hybridep", lambda *args, **kwargs: None)
    monkeypatch.setattr(rank, "_prepare_packed_forward", lambda packed: None)
    monkeypatch.setattr(rank, "_forward_packed", forward)
    yield SimpleNamespace(
        rank=rank,
        weight=weight,
        clock=clock,
        cuda=cuda,
        work=work,
        state=state,
        references=references,
    )
    work.close()


def _input(retention="gpu", output_device="cpu"):
    return ForwardInput(
        input_tokens=torch.tensor([1, 2]),
        hidden_states=True,
        options=ForwardOptions(backward_state=retention, output_device=output_device),
    )


@pytest.mark.parametrize("api", ["forward", "forward_batches"])
@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay", "evict"])
def test_physical_backward_excludes_proxy_idle_and_replay_time(rig, api, retention):
    rank, work = rig.rank, rig.work
    request = _input("gpu" if retention == "evict" else retention)
    batches = rank.forward_batches([request]) if api == "forward_batches" else None
    output = next(batches).outputs[0] if batches else rank.forward(request)
    value = output.hidden_states
    assert value is not None and value.device.type == "cpu"
    cache = rank._forward_graph_cache()
    if retention == "evict":
        cache.evict(cache.handles()[0])
    if retention in {"replay", "evict"}:
        assert rig.references[0]() is None
    assert work.work_ns == 0 and not work.rows and not work.disabled

    def caller(_):
        rig.clock.value += CALLER_NS

    value.register_hook(caller)
    rig.clock.value += CALLER_NS  # Caller idle time is outside either engine.
    rank.backward(value.sum(), retain_graph=True)
    assert rig.state.forwards == (2 if retention in {"replay", "evict"} else 1)
    assert len(work.rows) == len(rig.cuda.events) == 1
    (row,) = work.rows.values()
    assert row.ended is not None and not row.blocked
    elapsed = row.ended - row.started
    assert MODEL_NS <= elapsed < MODEL_NS + 100
    assert rig.cuda.events[0].stream == ("caller", 0)
    work.harvest()
    assert work.work_ns == 0  # Engine completion alone cannot credit GPU work.
    rig.clock.value += CALLER_NS
    rig.cuda.events[0].ready = True
    work.harvest()
    assert work.work_ns == elapsed and not work.rows
    rank.backward(value.sum())
    rig.cuda.events[-1].ready = True
    work.harvest()
    assert 2 * MODEL_NS <= work.work_ns < 2 * MODEL_NS + 200
    torch.testing.assert_close(rig.weight.grad, torch.tensor([[6.0]]))
    assert not cache.handles()
    assert all(reference() is None for reference in rig.references)
    assert not work.disabled
    if batches is not None:
        assert next(batches, None) is None


@pytest.mark.parametrize("mode", ["rank", "zero"])
@pytest.mark.parametrize("remote", [False, True])
def test_logical_and_remote_backwards_credit_one_physical_engine(rig, mode, remote):
    def run(callback):
        return asyncio.run(run_rank_callback(rig.rank, callback, mode=mode)).value

    if remote:
        packet = run(lambda view: view.export_forward(view.forward(_input("replay"))))
        collector = CotangentCollector()
        output = collector.attach(packet)
        packets = collector.backward(output.hidden_states.sum())
        assert not rig.work.rows and not rig.cuda.events
        run(lambda view: view.backward_packets(packets))
    else:

        def train(view):
            output = view.forward(_input("replay"))
            assert not rig.work.rows and not rig.cuda.events
            view.backward(output.hidden_states.sum())

        run(train)
    assert len(rig.work.rows) == len(rig.cuda.events) == 1
    rig.cuda.events[0].ready = True
    rig.work.harvest()
    assert MODEL_NS <= rig.work.work_ns < MODEL_NS + 100
    torch.testing.assert_close(rig.weight.grad, torch.tensor([[3.0]]))
    assert not rig.rank._forward_graph_cache().handles()


def test_failed_physical_engine_prevents_later_credit(rig):
    primary = RuntimeError("physical backward failed")

    def fail():
        raise primary

    output = rig.rank.forward(_input())
    rig.state.before_backward = fail
    with pytest.raises(RuntimeError) as caught:
        rig.rank.backward(output.hidden_states.sum())
    assert caught.value is primary
    assert len(rig.work.rows) == 1 and not rig.cuda.events
    assert next(iter(rig.work.rows.values())).ended is None
    assert rig.weight.grad is None
    assert not rig.rank._forward_graph_cache().handles()
    rig.state.before_backward = lambda: None
    output = rig.rank.forward(_input("replay"))
    rig.rank.backward(output.hidden_states.sum())
    assert len(rig.cuda.events) == 1
    rig.cuda.events[0].ready = True
    rig.work.harvest()
    assert rig.work.work_ns == 0 and not rig.work.disabled


@pytest.mark.parametrize("nested", ["forward", "backward"])
def test_nested_physical_work_is_excluded(rig, nested):
    output = rig.rank.forward(_input())
    other = rig.rank.forward(_input()) if nested == "backward" else None

    def reenter():
        rig.state.before_backward = lambda: None
        if other is None:
            rig.rank.forward(_input(), no_grad=True)
        else:
            with torch.enable_grad():
                rig.rank.backward(other.hidden_states.sum())

    rig.state.before_backward = reenter
    rig.rank.backward(output.hidden_states.sum())
    assert len(rig.work.rows) == (2 if nested == "backward" else 1)
    assert all(row.blocked for row in rig.work.rows.values())
    for event in rig.cuda.events:
        event.ready = True
    rig.work.harvest()
    assert rig.work.work_ns == 0 and not rig.work.rows and not rig.work.disabled
