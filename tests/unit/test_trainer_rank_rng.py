from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from art.trainer_rank import AdamParams, ForwardInput, ForwardOutput, TrainerRank
from art.trainer_rank._impl import _CheckpointSlot, _GatherContextParallelRows
from art.trainer_rank._rng import RNGState, TrainerRNG, caller_group


def test_model_seed_does_not_depend_on_logical_leader_draws():
    torch.manual_seed(517)
    leader = TrainerRNG(torch.device("cpu"))
    peer = TrainerRNG(torch.device("cpu"))
    # Only the logical leader runs the caller's head factory and random masks.
    torch.nn.Linear(7, 11)
    torch.rand(37)
    with leader.model():
        expected = torch.rand(23)
    torch.manual_seed(919)
    with peer.model():
        actual = torch.rand(23)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("retention", ("gpu", "cpu", "replay"))
def test_graph_replay_preserves_caller_and_private_model_progress(
    monkeypatch, retention
):
    from art.trainer_rank._graphs import GraphCache

    trainer = _trainer()
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.tensor(2.0))
    states, handles = [], []

    def execute(_):
        output = torch.nn.functional.dropout(torch.ones(29), 0.3) * weight
        states.append(torch.get_rng_state())
        return (output,)

    def forward():
        handle, (output,) = cache.run(execute, None, retention=retention)
        handles.append(handle)
        return [ForwardOutput(output, None, None, None)]

    _stub_forward(monkeypatch, trainer, forward)
    torch.manual_seed(791)
    caller = torch.get_rng_state()
    output = trainer.forward([ForwardInput(input_tokens=torch.arange(29))])[0]
    assert output.target_logprobs is not None
    assert torch.equal(torch.get_rng_state(), caller)
    torch.rand(17)
    caller = torch.get_rng_state()
    cache.backward(handles[0], (torch.ones_like(output.target_logprobs),))
    assert torch.equal(torch.get_rng_state(), caller)
    assert weight.grad is not None
    torch.testing.assert_close(weight.grad, (output.target_logprobs / 2).sum())
    generator = torch.Generator().set_state(states[0])
    with trainer._rng.model():
        torch.testing.assert_close(torch.rand(19), torch.rand(19, generator=generator))


def _trainer(device="cpu"):
    model = torch.nn.Linear(3, 4, bias=False, device=device)
    runtime = SimpleNamespace(
        model=[model],
        optimizer=None,
        provider=SimpleNamespace(hidden_size=4, num_layers=1),
        model_support_handler=SimpleNamespace(
            build_gdn_execution_spec=False,
            zero_internal_padding_grads=lambda _: None,
        ),
    )
    return TrainerRank(cast(Any, runtime))


def _stub_forward(monkeypatch, trainer, execute):
    monkeypatch.setattr(
        trainer, "_plan_admissible_forward", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(trainer, "_execute_admitted_plan", lambda *a, **k: execute())


def _state(device):
    return RNGState.capture(
        (device.index,) if device.type == "cuda" else (), torch_only=True
    )


def _assert_state_equal(left, right):
    assert torch.equal(left.cpu, right.cpu)
    assert left.cuda.keys() == right.cuda.keys()
    for device in left.cuda:
        assert torch.equal(left.cuda[device], right.cuda[device])


def test_model_stream_advances_without_advancing_caller(monkeypatch):
    trainer = _trainer()
    torch.manual_seed(811)
    caller = torch.get_rng_state()
    expected = torch.rand(3, 12)
    torch.set_rng_state(caller)
    observed = []
    internal_states = []

    def execute():
        # Nested internal work shares the model stream instead of restarting it.
        with trainer._rng.model():
            observed.append(torch.rand(12))
            internal_states.append(torch.get_rng_state())
        return []

    _stub_forward(monkeypatch, trainer, execute)
    trainer.forward([], no_grad=True)
    assert torch.equal(torch.get_rng_state(), caller)
    # Caller randomness advances independently between model forwards.
    assert torch.equal(torch.rand(12), expected[0])
    trainer.forward([])
    assert not torch.equal(observed[0], expected[0])
    generator = torch.Generator().set_state(internal_states[0])
    torch.testing.assert_close(observed[1], torch.rand(12, generator=generator))
    assert torch.equal(torch.rand(12), expected[1])


def test_forward_failure_restores_caller_and_advances_model(monkeypatch):
    trainer = _trainer()
    torch.manual_seed(981)
    caller = torch.get_rng_state()
    internal_states = []

    def fail():
        torch.rand(7)
        internal_states.append(torch.get_rng_state())
        raise ValueError("model failed")

    _stub_forward(monkeypatch, trainer, fail)
    with pytest.raises(ValueError, match="model failed"):
        trainer.forward([])
    assert torch.equal(torch.get_rng_state(), caller)
    generator = torch.Generator().set_state(internal_states[0])
    with trainer._rng.model():
        assert torch.equal(torch.rand(7), torch.rand(7, generator=generator))


@pytest.mark.parametrize("yield_empty", (False, True))
def test_microbatch_yields_and_close_do_not_hold_rng_context(monkeypatch, yield_empty):
    trainer = _trainer()
    torch.manual_seed(177)
    caller = torch.get_rng_state()
    draws = torch.rand(5, 8)
    torch.set_rng_state(caller)
    observed = []
    internal_states = []

    def batches(*args, **kwargs):
        for index in range(3):
            observed.append(torch.rand(8))
            internal_states.append(torch.get_rng_state())
            yield SimpleNamespace(
                outputs=[] if index == 0 else [index],
                stats=SimpleNamespace(global_count=1),
            )

    monkeypatch.setattr(trainer, "_forward_batches", batches)
    iterator = trainer.forward_batches([], yield_empty=yield_empty)
    next(iterator)
    assert trainer._rng._depth == 0
    assert torch.equal(torch.get_rng_state(), caller)
    assert torch.equal(torch.rand(8), draws[0])
    next(iterator)
    assert trainer._rng._depth == 0
    iterator.close()
    assert torch.equal(torch.rand(8), draws[1])
    generator = torch.Generator().set_state(internal_states[0])
    for draw in observed[1:]:
        torch.testing.assert_close(draw, torch.rand(8, generator=generator))


def test_uninitialized_model_parallel_group_does_not_mean_world(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert caller_group() is None
    monkeypatch.setattr(
        dist, "broadcast", lambda *a, **k: pytest.fail("unexpected WORLD broadcast")
    )
    TrainerRNG(torch.device("cpu")).synchronize(None)


def test_caller_reseed_and_restore_between_forwards(monkeypatch):
    trainer = _trainer()
    _stub_forward(monkeypatch, trainer, lambda: (torch.rand(9), [])[1])
    trainer.forward([])
    torch.manual_seed(191)
    saved = torch.get_rng_state()
    expected = torch.rand(9)
    torch.set_rng_state(saved)
    trainer.forward([])
    assert torch.equal(torch.rand(9), expected)
    torch.set_rng_state(saved)
    trainer.forward([])
    assert torch.equal(torch.rand(9), expected)


@pytest.mark.parametrize("microbatches", (False, True))
def test_input_iterators_use_caller_rng(monkeypatch, microbatches):
    trainer = _trainer()
    torch.manual_seed(617)
    state = torch.get_rng_state()
    expected = torch.rand(2, 5)
    torch.set_rng_state(state)
    request = ForwardInput(input_tokens=torch.arange(8), hidden_states=True)

    def inputs():
        assert torch.equal(torch.rand(5), expected[0])
        yield request

    def execute():
        torch.rand(11)
        return [ForwardOutput(None, None, None, torch.ones(8, 4))]

    _stub_forward(monkeypatch, trainer, execute)
    if microbatches:

        def batches(items, **kwargs):
            assert len(items) == 1
            torch.testing.assert_close(items[0].input_tokens, request.input_tokens)
            yield SimpleNamespace(
                outputs=execute(), stats=SimpleNamespace(global_count=1)
            )

        monkeypatch.setattr(trainer, "_forward_batches", batches)
        list(trainer.forward_batches(inputs()))
    else:
        trainer.forward(inputs())
    assert torch.equal(torch.rand(5), expected[1])


@pytest.mark.parametrize("dp_size", (1, 2))
def test_replicated_caller_randomness_cpu(dp_size, tmp_path):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _distributed_worker,
        args=(dp_size, "cp", "gloo", f"file://{tmp_path / 'rng'}"),
        nprocs=2 * dp_size,
        join=True,
    )


@pytest.mark.parametrize("parallelism", ("tp", "cp"))
def test_replicated_caller_randomness_cuda(parallelism, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    pytest.importorskip("megatron.core")
    mp.spawn(
        _distributed_worker,
        args=(1, parallelism, "nccl", f"file://{tmp_path / 'rng'}"),
        nprocs=2,
        join=True,
    )


def _distributed_worker(rank, dp_size, parallelism, backend, init_method):
    from megatron.core import parallel_state as ps

    device = torch.device("cpu" if backend == "gloo" else f"cuda:{rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=2 * dp_size,
        timeout=timedelta(seconds=90),
    )
    try:
        replica_groups = [dist.new_group([2 * dp, 2 * dp + 1]) for dp in range(dp_size)]
        dp_groups = [
            dist.new_group(list(range(replica, 2 * dp_size, 2))) for replica in range(2)
        ]
        dp_rank, replica_rank = divmod(rank, 2)
        replica_group, dp_group = replica_groups[dp_rank], dp_groups[replica_rank]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                ps, "get_tensor_and_context_parallel_group", lambda **_: replica_group
            )
            patch.setattr(
                ps,
                "get_tensor_model_parallel_world_size",
                lambda: 2 if parallelism == "tp" else 1,
            )
            patch.setattr(
                ps,
                "get_context_parallel_world_size",
                lambda: 2 if parallelism == "cp" else 1,
            )
            patch.setattr(
                ps,
                "get_tensor_model_parallel_group",
                lambda **_: replica_group if parallelism == "tp" else None,
            )
            patch.setattr(
                ps,
                "get_data_parallel_group",
                lambda **_: dp_group if parallelism == "tp" else dist.group.WORLD,
            )
            patch.setattr(ps, "get_data_parallel_rank", lambda: dp_rank)
            patch.setattr(ps, "get_data_parallel_world_size", lambda: dp_size)
            _gradient_oracle(
                patch,
                device,
                rank,
                dp_rank,
                dp_size,
                replica_rank,
                replica_group,
                dp_group,
                parallelism,
            )
    finally:
        dist.destroy_process_group()


def _gradient_oracle(
    patch,
    device,
    rank,
    dp_rank,
    dp_size,
    replica_rank,
    replica_group,
    dp_group,
    parallelism,
):
    trainer = _trainer(device)
    decoder = trainer.runtime.model[0].weight
    with torch.no_grad():
        decoder.copy_(torch.arange(12, device=device).reshape(4, 3) / 30)
    if parallelism == "tp":
        decoder.grad_sync_op = "sum"
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        config={
            "base_model_name_or_path": "test",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": [],
        },
        params=(decoder,),
    )
    # Registration repairs initially different CPU/CUDA states within each DP
    # worker, while the registered weights remain common to all DP workers.
    torch.manual_seed(711 + 100 * dp_rank + replica_rank)
    head = trainer.module(
        "head",
        lambda: torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(4, 2)),
        checkpoint="student",
    )
    params = trainer._checkpoint_slots["student"].params
    reference = tuple(torch.nn.Parameter(param.detach().clone()) for param in params)
    optimizer = torch.optim.AdamW(reference, lr=0.01, weight_decay=0.0)
    features = (
        torch.arange(24, device=device, dtype=torch.float32).reshape(8, 3) / 20
        + dp_rank / 10
    )
    rows = torch.arange(replica_rank * 4, (replica_rank + 1) * 4, device=device)
    request = ForwardInput(input_tokens=torch.arange(8), hidden_states=True)
    masks = []
    recompute_masks = []
    cpu_draws = []
    previous_caller_mask = None
    tracker = None
    if device.type == "cuda" and parallelism == "cp":
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        tracker = get_cuda_rng_tracker()
        tracker.add("art-test-model", 3199 + rank)

    def execute():
        # Internal consumption deliberately differs across physical ranks. It
        # must not leak into caller masks or custom-head dropout.
        torch.rand(rank + 1)
        torch.rand(rank + 3, device=device)
        records = []
        local_rows = rows
        if parallelism == "cp" and not masks:
            # One CP peer owns no tokens but still consumes the caller's full
            # output and participates in backward and RNG synchronization.
            local_rows = torch.arange(8 if replica_rank == 0 else 0, device=device)

        def model(weight):
            with (
                tracker.fork("art-test-model") if tracker is not None else nullcontext()
            ):
                mask = torch.nn.functional.dropout(
                    torch.ones_like(features[local_rows]), 0.2
                )
            records.append(mask.detach().clone())
            return (features[local_rows] * mask) @ weight.T

        if tracker is not None:
            from megatron.core.tensor_parallel import checkpoint as megatron_checkpoint

            hidden = megatron_checkpoint(model, False, decoder)
        else:
            hidden = checkpoint(model, decoder, use_reentrant=False)
        recompute_masks.append(records)
        full_mask = torch.zeros_like(features)
        full_mask[local_rows] = records[0]
        dist.all_reduce(full_mask, group=replica_group)
        masks.append(full_mask)
        if parallelism == "tp":
            hidden = trainer._gather_sequence_parallel_hidden(hidden[:, None])
        else:
            hidden = _GatherContextParallelRows.apply(
                hidden, local_rows, len(features), replica_group
            )
        return [ForwardOutput(None, None, None, hidden)]

    _stub_forward(patch, trainer, execute)

    def batches(*args, **kwargs):
        yield SimpleNamespace(
            outputs=execute(), stats=SimpleNamespace(global_count=dp_size)
        )

    patch.setattr(trainer, "_forward_batches", batches)
    for step in range(2):
        losses = []
        expected_losses = []
        for micro in range(2):
            if step == micro == 0:
                # Disagree again after registration: forward return must repair
                # the caller even though the model uses a separate stream.
                torch.manual_seed(1231 + 100 * dp_rank + replica_rank)
            before = _state(device)
            if step == 0:
                output = trainer.forward([request])[0]
            else:
                iterator = trainer.forward_batches([request])
                output = next(iterator).outputs[0]
                assert trainer._rng._depth == 0
                iterator.close()
            if replica_rank == 0:
                _assert_state_equal(_state(device), before)
            caller = _state(device)
            cpu_draw = torch.rand(16)
            cpu_draws.append(cpu_draw)
            mask = torch.rand(len(features), device=device) > 0.35
            if previous_caller_mask is not None:
                assert not torch.equal(cpu_draw, previous_caller_mask)
            previous_caller_mask = cpu_draw
            loss = head(output.hidden_states[mask]).square().sum()
            losses.append(loss)
            # The unsplit reference replays the exact caller stream, including
            # the CPU mask draw and dropout; model dropout is taken from the
            # actual shards so this checks caller/model gradient consistency.
            with torch.random.fork_rng(
                devices=[device.index] if device.type == "cuda" else []
            ):
                caller.restore()
                torch.testing.assert_close(torch.rand(16), cpu_draw)
                expected_mask = torch.rand(len(features), device=device) > 0.35
                assert torch.equal(expected_mask, mask)
                hidden = (features * masks[-1]) @ reference[0].T
                dropped = torch.nn.functional.dropout(hidden[expected_mask], 0.3)
                expected_loss = (
                    torch.nn.functional.linear(dropped, reference[1], reference[2])
                    .square()
                    .sum()
                )
                torch.testing.assert_close(loss, expected_loss)
                expected_losses.append(expected_loss)
            # The collective comparison is independent of the reference replay.
            copies = [torch.empty_like(cpu_draw, device=device) for _ in range(2)]
            dist.all_gather(copies, cpu_draw.to(device), group=replica_group)
            assert torch.equal(copies[0], copies[1])
        before_backward = _state(device)
        tracker_before = tracker.get_states() if tracker is not None else {}
        trainer.backward(torch.stack(losses).sum())
        _assert_state_equal(_state(device), before_backward)
        if tracker is not None:
            for key, state in tracker_before.items():
                assert torch.equal(tracker.get_states()[key], state)
        torch.stack(expected_losses).sum().backward()
        reduced = trainer._reduce_dynamic_grads(params, scale_grads=1 / dp_size)
        for actual, expected in zip(reduced, reference, strict=True):
            assert expected.grad is not None
            dist.all_reduce(expected.grad, group=dp_group)
            expected.grad.div_(dp_size)
            torch.testing.assert_close(actual, expected.grad, rtol=2e-5, atol=2e-5)
        optimizer.step()
        optimizer.zero_grad()
        metrics = trainer.optim_step(
            params=AdamParams(learning_rate=0.01, weight_decay=0.0, grad_clip_norm=0),
            scale_grads=1 / dp_size,
            checkpoints=["student"],
        )
        assert metrics["update_successful"] == 1
        for actual, expected in zip(params, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for records in recompute_masks:
        assert len(records) == 2
        assert torch.equal(records[0], records[1])
    assert not torch.equal(masks[0], masks[1])
    if dp_size > 1:
        copies = [torch.empty_like(cpu_draws[0]) for _ in range(dp_size)]
        dist.all_gather(copies, cpu_draws[0], group=dp_group)
        assert not torch.equal(copies[0], copies[1])
