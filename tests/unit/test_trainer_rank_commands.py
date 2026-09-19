"""Command participation tests; native model numerics are a separate GPU gate."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import gc
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    MicroBatch,
    MicroBatchStats,
    TrainerRank,
    TrainerRankZero,
    run_rank_callback,
    run_rank_callback_stream,
)


class _Rank:
    device = torch.device("cpu")
    hidden_size = 1

    def __init__(self, dp: int = 0, size: int = 1) -> None:
        self.dp, self.size = dp, size
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.closed = 0
        self.steps = 0

    def _dp_rank_and_size(self):
        return self.dp, self.size

    def forward(self, tree, **kwargs):
        if isinstance(tree, ForwardInput):
            enabled = (
                torch.is_grad_enabled()
                if kwargs.get("no_grad") is None
                else not kwargs["no_grad"]
            )
            with torch.set_grad_enabled(enabled):
                value = tree.input_tokens.float() * self.weight
            return ForwardOutput(None, None, None, value)
        from art.trainer_rank._impl import _rebuild_forward_tree

        return _rebuild_forward_tree(
            tree, [self.forward(child, **kwargs) for child in tree]
        )

    def forward_batches(self, inputs, **kwargs):
        try:
            # One global root per wave guarantees empty DP partitions.
            for index, item in enumerate(inputs):
                owned = index % self.size == self.dp
                yield MicroBatch(
                    [item] if owned else [],
                    [self.forward(item, **kwargs)] if owned else [],
                    [index] if owned else [],
                    MicroBatchStats(
                        index, index + 1, 1, int(owned), 0, 0, 0, 0, 0, False
                    ),
                )
        finally:
            self.closed += 1

    def zero_grad(self):
        self.weight.grad = None

    def optim_step(self, **kwargs):
        self.steps += 1
        return {"steps": self.steps}


def _input(value):
    return ForwardInput(input_tokens=torch.tensor([value]))


def _suspended_abort_worker(physical, rendezvous):
    from test_trainer_rank_custom_tensors import _trainer

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=physical,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        native, _ = _trainer("student")
        native._checkpoint_process_group = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=8)
        )
        native._checkpoint_finalize_process_group = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=8)
        )
        rank: Any = _Rank()
        actor_thread = threading.get_ident()

        async def run(mode):
            entered, release = asyncio.Event(), asyncio.Event()

            def zero_grad():
                assert threading.get_ident() == actor_thread
                entered.set()

            rank.zero_grad = zero_grad

            async def callback(view):
                view.zero_grad()
                await release.wait()
                view.zero_grad()

            async def generator(view):
                view.zero_grad()
                try:
                    yield "suspended"
                finally:
                    view.zero_grad()

            async def consume():
                if mode != "stream":
                    return await run_rank_callback(rank, callback, mode="zero")
                stream = run_rank_callback_stream(rank, generator, mode="zero")
                result = await anext(stream)
                if physical == 0:
                    assert result.value == "suspended"
                    await release.wait()
                await stream.aclose()

            pending = asyncio.create_task(consume())

            if mode == "shutdown":
                await entered.wait()
                # asyncio.run cancels every pending Task, not just the callback.
                # The control receive must remain joinable during that drain.
                return

            async def abort():
                await entered.wait()
                await asyncio.sleep(0.05)
                if mode == "cancel" and physical == 1:
                    pending.cancel()
                    await asyncio.sleep(0)
                # The same synchronous physical abort dispatched by Caladan,
                # on the actor loop while the logical callback is suspended.
                native.abort_checkpoint_save("unprepared-save")
                assert not pending.done()
                release.set()

            await abort()
            result = (await asyncio.gather(pending, return_exceptions=True))[0]
            if mode == "cancel" and physical == 1:
                assert isinstance(result, asyncio.CancelledError)
            elif isinstance(result, BaseException):
                raise result

        for mode in ("async", "stream", "cancel", "shutdown"):
            asyncio.run(run(mode))
    finally:
        dist.destroy_process_group()


def test_suspended_callbacks_leave_physical_checkpoint_abort_responsive(tmp_path):
    mp.spawn(_suspended_abort_worker, args=(str(tmp_path / "abort-init"),), nprocs=2)


def _loss_tree(tree):
    if isinstance(tree, ForwardOutput):
        return tree.hidden_states.sum()
    return sum(_loss_tree(item) for item in tree)


def test_native_facade_owns_backward_and_hides_zero_reduce():
    rank: Any = _Rank()

    def callback(view):
        assert isinstance(view, TrainerRankZero)
        assert not hasattr(view, "reduce")
        outputs = view.forward([[_input(2), [_input(3)]]])
        extra = view.forward(_input(5))
        unused = view.forward(_input(100))
        view.backward((_loss_tree(outputs) + _loss_tree(extra)).square())
        assert rank.weight.grad.item() == 400
        assert unused.hidden_states.item() == 200
        assert view.optim_step() == {"steps": 1}
        return outputs[0][1][0].hidden_states.item()

    result = asyncio.run(run_rank_callback(rank, callback, mode="zero"))
    assert result.logical_rank == 0
    assert result.value == 6
    assert rank.closed == 3


def test_client_packet_registry_survives_callbacks():
    rank: Any = _Rank()
    from art.trainer_rank._tensors import CotangentCollector

    packet = asyncio.run(
        run_rank_callback(
            rank,
            lambda view: view.export_forward(view.forward([_input(3)])),
            mode="zero",
        )
    ).value
    collector = CotangentCollector()
    output = collector.attach(packet)
    gradients = collector.backward(output[0].hidden_states.square().sum())
    asyncio.run(
        run_rank_callback(
            rank, lambda view: view.backward_packets(gradients), mode="zero"
        )
    )
    assert rank.weight.grad.item() == 36


def test_rank_facade_is_trainer_rank_and_dispatches_inherited_methods():
    rank: Any = _Rank()

    def callback(view):
        assert isinstance(view, TrainerRank)
        view.zero_grad()
        return view.optim_step()

    assert asyncio.run(run_rank_callback(rank, callback)).value == {"steps": 1}


def test_stream_forwards_sends_and_closes():
    rank: Any = _Rank()
    closed = []

    def callback(view):
        try:
            sent = yield view.forward(_input(3)).hidden_states.item()
            yield sent * 2
        finally:
            closed.append(True)

    async def run():
        stream = run_rank_callback_stream(rank, callback, mode="zero")
        assert (await anext(stream)).value == 6
        assert (await stream.asend(9)).value == 18
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]


class _Unserializable:
    def __reduce_ex__(self, protocol):
        raise TypeError("intentional serialization failure")


def _rank_local_restore_failure():
    if dist.get_rank() == 1:
        raise ValueError("intentional peer deserialization failure")
    return None


class _BadRestore:
    def __reduce_ex__(self, protocol):
        return _rank_local_restore_failure, ()


class _FailPhysicalBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        if dist.get_rank() == 1:
            raise RuntimeError("intentional physical backward failure")
        return gradient


def _distributed_worker(physical, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=physical,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        dp, tp = divmod(physical, 2)
        ps = SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: tp,
            get_context_parallel_rank=lambda: 0,
            get_data_parallel_rank=lambda: dp,
            get_data_parallel_world_size=lambda: 2,
            get_tensor_and_context_parallel_group=lambda **kwargs: groups[dp],
        )
        megatron = ModuleType("megatron")
        core = ModuleType("megatron.core")
        setattr(core, "parallel_state", ps)
        setattr(megatron, "core", core)
        sys.modules.update({"megatron": megatron, "megatron.core": core})
        rank: Any = _Rank(dp, 2)
        counts = [0, 0]

        def zero(view):
            counts[0] += 1
            result = view.forward([[_input(2), [_input(3)]], _input(7)])
            second = view.forward(_input(11))
            view.backward((_loss_tree(result) + _loss_tree(second)).square())
            # Fail one physical rank's packet preflight before any backward.
            from art.trainer_rank._tensors import CotangentPacket

            with pytest.raises((RuntimeError, ValueError)):
                view._invoke(
                    "backward",
                    (CotangentPacket("zero:missing:dp:1", (None,)),),
                    retain_graph=False,
                )
            with pytest.raises(RuntimeError, match="serialization failed"):
                view._invoke("zero_grad", _Unserializable())
            with pytest.raises(RuntimeError, match="deserialization failed"):
                view._invoke("zero_grad", _BadRestore())
            iterator = view.forward_batches([_input(1), _input(2)])
            next(iterator)
            iterator.close()
            return [_loss_tree(root).item() for root in result]

        result = asyncio.run(run_rank_callback(rank, zero, mode="zero"))
        # Global scalar loss is (2 * (2+3+7+11)) ** 2.
        expected = 2 * 46 * (16 if dp == 0 else 7)
        torch.testing.assert_close(rank.weight.grad, torch.tensor(float(expected)))
        assert rank.closed == 3

        def logical(view):
            counts[1] += 1
            view.zero_grad()
            roots = [] if dp == 1 else [[_input(5)]]
            out = view.forward(roots)
            if out:
                view.backward(_loss_tree(out))
            view.optim_step()
            return dp

        per_dp = asyncio.run(run_rank_callback(rank, logical))

        # Persistent streams release the command scope between waves. Every
        # physical participant must retain its iterator while serving other jobs.
        def run(callback):
            return asyncio.run(run_rank_callback(rank, callback, mode="zero")).value

        handle = run(lambda view: view.open_forward_batches([_input(3), _input(5)]))
        assert rank._rank_command_state.iterators
        run(lambda view: view.zero_grad())
        first = run(lambda view: view.next_forward_batch(handle))
        run(lambda view: view.backward(_loss_tree(first.outputs)))
        run(lambda view: view.optim_step())
        second = run(lambda view: view.next_forward_batch(handle))
        run(lambda view: view.backward(_loss_tree(second.outputs)))
        assert run(lambda view: view.next_forward_batch(handle)) is None
        run(lambda view: view.close_forward_batches(handle))
        assert not rank._rank_command_state.iterators
        assert rank.weight.grad.item() == (3 if dp == 0 else 5)

        from art.trainer_rank import _commands

        decoded_outputs = []
        loads = _commands.cloudpickle.loads

        def track_loads(payload):
            value = loads(payload)
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], _commands._OutputPacket)
            ):
                decoded_outputs.append(value[1].packet.tensors)
            return value

        setattr(_commands.cloudpickle, "loads", track_loads)
        try:
            large = run(
                lambda view: view.forward(
                    [
                        [
                            ForwardInput(
                                input_tokens=torch.ones(131072, dtype=torch.long)
                            )
                        ],
                        [
                            ForwardInput(
                                input_tokens=torch.ones(131072, dtype=torch.long)
                            )
                        ],
                    ],
                    no_grad=True,
                )
            )
        finally:
            setattr(_commands.cloudpickle, "loads", loads)
        assert bool(decoded_outputs) is (physical == 0)
        if physical == 0:
            assert (
                sum(
                    t.numel() * t.element_size()
                    for tensors in decoded_outputs
                    for t in tensors
                )
                == 2 * 131072 * 4
            )
            assert large[1][0].hidden_states.shape == (131072,)
        rank._available_cpu_memory_bytes = lambda: 0 if physical == 0 else 1 << 60

        def host_refusal(view):
            with pytest.raises(MemoryError, match="CPU bytes"):
                view.forward([_input(3)], no_grad=True)
            view.zero_grad()

        run(host_refusal)
        del rank._available_cpu_memory_bytes

        retained_before_failure = set(rank._rank_command_state.graphs)

        def fail_result_decode(payload):
            value = loads(payload)
            if (
                physical == 0
                and isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], _commands._OutputPacket)
            ):
                raise ValueError("intentional result decode failure")
            return value

        def decode_refusal(view):
            with pytest.raises(ValueError, match="result decode failure"):
                view.forward([_input(11)])

        setattr(_commands.cloudpickle, "loads", fail_result_decode)
        try:
            run(decode_refusal)
        finally:
            setattr(_commands.cloudpickle, "loads", loads)
        assert set(rank._rank_command_state.graphs) <= retained_before_failure
        assert not rank._rank_command_state.iterators
        run(lambda view: view.backward(_loss_tree(view.forward([_input(11)]))))
        if dp == 0:
            assert rank.weight.grad.item() == 11

        # Snapshot fits, but serialization's transient buffers do not. Refuse
        # before pickle allocates those buffers on any participant.
        dumps, serialized = _commands.cloudpickle.dumps, []

        def track_dumps(value):
            if isinstance(value, tuple) and len(value) == 2:
                serialized.append(True)
            return dumps(value)

        rank._available_cpu_memory_bytes = lambda: 1024 if physical == 0 else 1 << 60
        setattr(_commands.cloudpickle, "dumps", track_dumps)
        try:
            run(host_refusal)
        finally:
            setattr(_commands.cloudpickle, "dumps", dumps)
            del rank._available_cpu_memory_bytes
        assert not serialized

        forward = rank.forward
        retained_before_failure = set(rank._rank_command_state.graphs)

        def failing_forward(tree, **kwargs):
            from dataclasses import replace

            result = forward(tree, **kwargs)
            if isinstance(result, ForwardOutput):
                result = replace(
                    result,
                    hidden_states=_FailPhysicalBackward.apply(result.hidden_states),
                )
            return result

        def backward_refusal(view):
            with pytest.raises(RuntimeError, match="physical backward failure"):
                view.backward(_loss_tree(view.forward([_input(7)])))

        rank.forward = failing_forward
        try:
            run(backward_refusal)
        finally:
            rank.forward = forward
        assert set(rank._rank_command_state.graphs) <= retained_before_failure
        run(lambda view: view.zero_grad())
        del sys.modules["megatron"], sys.modules["megatron.core"]
        import megatron.core as real_core
        from test_trainer_rank_custom_tensors import _trainer

        setattr(real_core, "parallel_state", ps)
        native, _ = _trainer("student")
        factories = []

        def head_callback(view):
            class LocalHead(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    factories.append(True)
                    self.weight = torch.nn.Parameter(torch.tensor(2.0))

                def forward(self, value):
                    return self.weight.square() * value

            head = view.module("head", LocalHead, checkpoint="student")
            view.backward(head(torch.tensor(3.0)))

        asyncio.run(run_rank_callback(native, head_callback, mode="zero"))
        weight = native._checkpoint_slots["student"].custom["head"].value.weight
        gradient = (
            torch.zeros_like(weight) if weight.grad is None else weight.grad.clone()
        )
        torch.testing.assert_close(gradient, torch.tensor(12.0 if dp == 0 else 0.0))
        dist.all_reduce(gradient, group=dp_groups[tp])
        torch.testing.assert_close(gradient, torch.tensor(12.0))
        assert len(factories) == int(physical == 0)
        # Only DP0 has a logical handle after Zero. Refresh must stay inside
        # its TP group when the next callback is independently per-DP.
        asyncio.run(run_rank_callback(native, lambda view: view.zero_grad()))

        gathered = [None] * 4
        dist.all_gather_object(gathered, (counts, result, per_dp, rank.steps))
        if physical == 0:
            torch.save(gathered, output)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_gloo_dp2_tp2_participation_and_gradients(tmp_path):
    output = tmp_path / "result.pt"
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "init"), str(output)),
        nprocs=4,
        join=True,
    )
    rows = torch.load(output, weights_only=False)
    assert [row[0] for row in rows] == [[1, 1], [0, 0], [0, 1], [0, 0]]
    assert [row[1].logical_rank for row in rows] == [0, None, None, None]
    assert rows[0][1].value == [10, 14]
    assert [row[2].logical_rank for row in rows] == [0, None, 1, None]
    assert [row[3] for row in rows] == [2, 2, 2, 2]


def test_released_client_graph_releases_physical_bridge():
    rank: Any = _Rank()
    packet = asyncio.run(
        run_rank_callback(
            rank, lambda view: view.export_forward(view.forward(_input(3))), mode="zero"
        )
    ).value
    assert rank._rank_command_state.graphs
    asyncio.run(
        run_rank_callback(
            rank, lambda view: view.release_forward([packet.handle]), mode="zero"
        )
    )
    assert not rank._rank_command_state.exports
    gc.collect()
    asyncio.run(
        run_rank_callback(rank, lambda view: view.release_forward([]), mode="zero")
    )
    assert not rank._rank_command_state.graphs


def test_forward_batches_captures_policy_before_iteration():
    from art.trainer_rank import ForwardOptions

    rank = object.__new__(TrainerRank)
    rank._forward_options = ForwardOptions(max_gradient_staleness=1, allow_replay=False)
    rank._skipped_forward_waves = {}
    request = _input(3)
    request.options = ForwardOptions(max_gradient_staleness=0)

    def batches(inputs, **kwargs):
        yield MicroBatch(
            inputs, [], [0], MicroBatchStats(0, 1, 1, 1, 0, 0, 0, 0, 0, False)
        )

    rank._forward_batches = batches
    iterator = rank.forward_batches(
        [request], options=ForwardOptions(allow_cpu_offload=False), yield_empty=True
    )
    request.options = ForwardOptions(max_gradient_staleness=4)
    rank._forward_options = ForwardOptions(max_gradient_staleness=7)
    captured = next(iterator).inputs[0].options
    assert captured.max_gradient_staleness == 0
    assert captured.allow_cpu_offload is False
    assert captured.allow_replay is False
    iterator.close()


@pytest.mark.parametrize("mode", ["rank", "zero"])
def test_tuple_root_and_nested_tuple_shape(mode):
    rank: Any = _Rank()
    inputs = ([_input(2), (_input(3),)],)
    result = asyncio.run(
        run_rank_callback(rank, lambda view: view.forward(inputs), mode=mode)
    ).value
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert isinstance(result[0][1], tuple)
    assert result[0][1][0].hidden_states.item() == 6


def test_logical_head_factory_runs_once_and_head_only_client_backward():
    from test_trainer_rank_custom_tensors import _trainer

    from art.trainer_rank._heads import LiveHead
    from art.trainer_rank._tensors import CotangentCollector

    trainer, _ = _trainer("student")
    calls = []

    def factory():
        calls.append(True)
        return torch.tensor(2.0)

    def register(view):
        parameter = view.parameter("gain", factory, checkpoint="student")
        assert view.parameter("gain", factory, checkpoint="student") is parameter
        return view._invoke("head", "head_export", (("student", "gain"),))[0]

    state = asyncio.run(run_rank_callback(trainer, register, mode="zero")).value
    assert calls == [True]
    collector = CotangentCollector()
    client = LiveHead(state, torch.tensor(2.0), collector)
    packets = collector.backward(client.value.square() * 3)
    asyncio.run(
        run_rank_callback(
            trainer, lambda view: view.backward_packets(packets), mode="zero"
        )
    )
    parameter = trainer._checkpoint_slots["student"].custom["gain"].value
    torch.testing.assert_close(parameter.grad, torch.tensor(12.0))


def test_logical_native_head_backward_commits_after_local_autograd():
    from test_trainer_rank_custom_tensors import _trainer

    trainer, _ = _trainer("student")

    class LocalHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.register_buffer("count", torch.tensor(0))

        def forward(self, value):
            self.count.add_(1)
            return value * self.weight.square()

    def callback(view):
        head = view.module("head", LocalHead, checkpoint="student")
        view.backward(head(torch.tensor(3.0)))

    asyncio.run(run_rank_callback(trainer, callback, mode="zero"))
    native = trainer._checkpoint_slots["student"].custom["head"].value
    torch.testing.assert_close(native.weight.grad, torch.tensor(12.0))
    assert native.count.item() == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_logical_iterator_captures_ambient_grad_mode(enabled):
    rank: Any = _Rank()

    def callback(view):
        with torch.set_grad_enabled(enabled):
            iterator = view.forward_batches([_input(3)])
        with torch.set_grad_enabled(not enabled):
            result = next(iterator).outputs[0].hidden_states
        assert result.requires_grad is enabled
        iterator.close()

    asyncio.run(run_rank_callback(rank, callback, mode="zero"))


def test_persistent_iterator_binds_policy_and_checkpoint_and_pulls_one_wave():
    from art.trainer_rank import ForwardOptions
    from art.trainer_rank._tensors import CotangentCollector

    class Rank(_Rank):
        _capture_forward_options = TrainerRank._capture_forward_options

        def __init__(self):
            super().__init__()
            self._forward_options = ForwardOptions(allow_replay=False)
            self._default_slot_ref = SimpleNamespace(name="original")
            self.seen = []

        def forward(self, tree, **kwargs):
            if isinstance(tree, ForwardInput):
                self.seen.append((tree.options, kwargs["checkpoint"]))
            return super().forward(tree, **kwargs)

        def optim_step(self, **kwargs):
            with torch.no_grad():
                self.weight.add_(1)
            self.zero_grad()

    rank: Any = Rank()

    def run(callback):
        return asyncio.run(run_rank_callback(rank, callback, mode="zero")).value

    first = _input(3)
    first.options = ForwardOptions(max_gradient_staleness=0)
    handle = run(
        lambda view: view.open_forward_batches(
            [first, _input(5), _input(7)],
            options=ForwardOptions(allow_cpu_offload=False),
        )
    )
    assert rank.seen == []
    first.options = ForwardOptions(max_gradient_staleness=4)
    rank._forward_options = ForwardOptions(allow_replay=True)
    rank._default_slot_ref = SimpleNamespace(name="later")
    collector = CotangentCollector()
    with torch.no_grad():
        packet = run(lambda view: view.export_forward(view.next_forward_batch(handle)))
    batch = collector.attach(packet)
    assert batch.indices == [0]
    assert len(rank.seen) == 1
    policy, checkpoint = rank.seen[0]
    assert checkpoint == "original"
    assert policy.max_gradient_staleness == 0
    assert policy.allow_cpu_offload is False
    assert policy.allow_replay is False
    cotangents = collector.backward(_loss_tree(batch.outputs).square())
    run(lambda view: view.backward_packets(cotangents))
    assert rank.weight.grad.item() == 36
    run(lambda view: view.optim_step())
    packet = run(lambda view: view.export_forward(view.next_forward_batch(handle)))
    batch = collector.attach(packet)
    assert batch.indices == [1]
    assert _loss_tree(batch.outputs).item() == 15
    cotangents = collector.backward(_loss_tree(batch.outputs))
    run(lambda view: view.backward_packets(cotangents))
    assert rank.weight.grad.item() == 5
    run(lambda view: view.close_forward_batches(handle))
    run(lambda view: view.close_forward_batches(handle))
    assert run(lambda view: view.next_forward_batch(handle)) is None
    assert len(rank.seen) == 2
    assert rank.closed == 1
    assert not rank._rank_command_state.iterators
    assert not rank._rank_command_state.batch_inputs


def test_persistent_iterator_captures_no_grad_before_next_callback():
    rank: Any = _Rank()

    def run(callback):
        return asyncio.run(run_rank_callback(rank, callback, mode="zero")).value

    with torch.no_grad():
        handle = run(lambda view: view.open_forward_batches([_input(3)]))
    batch = run(lambda view: view.next_forward_batch(handle))
    assert not batch.outputs[0].hidden_states.requires_grad
    assert run(lambda view: view.next_forward_batch(handle)) is None
    assert rank.closed == 1


def test_nested_aggregate_outputs_admit_before_any_model_copy():
    from dataclasses import replace

    from art.trainer_rank import ForwardOptions
    from art.trainer_rank._commands import _Executor, _OutputPacket, _view
    from art.trainer_rank._tensors import detach_tree

    rank: Any = _Rank()
    rank._available_memory_bytes = lambda: 1024 * 1024
    view = _view(_Executor(rank, "zero"))
    request = ForwardInput(
        input_tokens=torch.tensor([1]), options=ForwardOptions(output_device="auto")
    )
    outputs = [
        _OutputPacket(
            detach_tree(
                f"zero:{index}:dp:{index}",
                [[ForwardOutput(None, None, None, torch.ones(262144))]],
            ),
            (False,),
            False,
        )
        for index in range(2)
    ]
    planned = view._place_outputs([(output, [[request]]) for output in outputs])
    assert [output.cpu for output in planned] == [(False,), (True,)]
    assert planned[1].managed
    request = replace(request, options=ForwardOptions(output_device="model"))
    with pytest.raises(MemoryError):
        view._place_outputs([(output, [[request]]) for output in outputs])
