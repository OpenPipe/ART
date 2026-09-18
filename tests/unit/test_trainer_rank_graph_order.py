"""Physical peers must enter cached/replayed backward collectives identically."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import TrainerRank, _graphs
from art.trainer_rank._impl import _CheckpointSlot


class _Collective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, tag):
        ctx.tag = tag
        return value.clone()

    @staticmethod
    def backward(ctx, *gradients):
        tag = torch.tensor([ctx.tag])
        tags = [torch.empty_like(tag) for _ in range(dist.get_world_size())]
        dist.all_gather(tags, tag)
        assert all(item.item() == ctx.tag for item in tags), (
            "different physical graph order"
        )
        gradient = gradients[0].clone()
        dist.all_reduce(gradient)
        return gradient, None


def _worker(rank, rendezvous, fail_replay):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        names = iter(("z", "a") if rank == 0 else ("a", "z"))
        setattr(_graphs, "uuid4", lambda: SimpleNamespace(hex=next(names)))
        cache = _graphs.GraphCache()
        parameters, packets = [], []
        trainer = TrainerRank.__new__(TrainerRank)
        trainer._checkpoint_slots = {"student": _CheckpointSlot(params=())}
        trainer.runtime = SimpleNamespace(model=[], optimizer=None)
        for tag in range(2):
            parameter = torch.nn.Parameter(torch.tensor(2.0))
            parameters.append(parameter)
            trainer._checkpoint_slots["student"].params = tuple(parameters)
            snapshot = trainer._snapshot_parameter(
                parameter, trainer._capture_checkpoint_version("student")
            )
            calls = [0]

            def execute(x, snapshot=snapshot, tag=tag, calls=calls):
                calls[0] += 1
                output = _Collective.apply(snapshot * x, tag)
                if fail_replay and rank == 0 and tag == 1 and calls[0] > 1:
                    output = output.expand(2)
                return (output,)

            handle, _ = cache.run(
                execute,
                torch.tensor(3.0),
                retention="replay",
            )
            packets.append((handle, (torch.tensor(1.0),)))

        def coordinate(function):
            result, error = None, None
            try:
                result = function()
            except Exception as exc:
                error = str(exc)
            errors = [None, None]
            dist.all_gather_object(errors, error)
            if any(errors):
                raise RuntimeError(str(errors))
            return result

        for parameter in parameters:
            parameter.grad = torch.tensor(7.0)

        def backward():
            with trainer._gradient_transaction(before_commit=coordinate):
                cache.backward_many(sorted(packets), coordinate=coordinate)

        if fail_replay:
            with pytest.raises(RuntimeError, match="metadata differs"):
                backward()
            assert not trainer._version_state()._origins
        else:
            backward()
        for parameter in parameters:
            torch.testing.assert_close(
                parameter.grad, torch.tensor(7.0 if fail_replay else 13.0)
            )
        assert not cache.handles()
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_replay", [False, True])
def test_backward_collectives_follow_creation_order_despite_different_handles(
    tmp_path, fail_replay
):
    mp.spawn(
        _worker, args=(f"file://{tmp_path / 'order'}", fail_replay), nprocs=2, join=True
    )
