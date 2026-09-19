"""Real collectives around injected transfer failure; no CUDA memory claims."""

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _impl
from art.trainer_rank._memory_policy import ForwardMemoryCost
from art.trainer_rank._options import resolve_forward_options


def _reclaim_worker(index: int, directory: str, sync_across_dp: bool) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        rank = object.__new__(_impl.TrainerRank)
        rank.device = torch.device("cpu")
        rank._graph_memory_policy_enabled = lambda: True
        rank._available_cpu_memory_bytes = lambda: 1000
        rank._forward_memory_group = lambda: dist.group.WORLD
        setattr(torch.cuda, "empty_cache", lambda: None)

        def offload(_handle):
            if index == 0:
                raise RuntimeError("injected CPU allocation failure")

        rank._graph_cache = SimpleNamespace(
            handles=lambda: (f"rank-local-{index}",),
            state=lambda _: SimpleNamespace(
                retention="gpu", offloadable=True, replayable=True, offload_bytes=100
            ),
            offload=offload,
            evict=lambda _: None,
        )
        try:
            rank._reclaim_graph_memory(
                _impl._MemoryCheck(2000, 1000, False), sync_across_dp=sync_across_dp
            )
        except RuntimeError as error:
            result = str(error)
        else:
            raise AssertionError("Failed graph transfer was incorrectly accepted")
        gathered = [None, None]
        # Proves both participants left reclamation at a matching boundary.
        dist.all_gather_object(gathered, result)
        if index == 0:
            Path(directory, "results.json").write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sync_across_dp", [False, True])
def test_failed_offload_does_not_strand_a_physical_peer(tmp_path, sync_across_dp):
    mp.spawn(_reclaim_worker, args=(str(tmp_path), sync_across_dp), nprocs=2, join=True)
    assert json.loads((tmp_path / "results.json").read_text()) == [
        "injected CPU allocation failure",
        "Graph reclamation failed on another physical rank",
    ]


def _fallback_worker(index: int, directory: str, sync_across_dp: bool) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        rank = object.__new__(_impl.TrainerRank)
        rank.device = torch.device("cpu")
        rank._forward_memory_group = lambda: dist.group.WORLD
        # Locally rank 0 prefers CPU (.2 versus .5 seconds), while rank 1
        # prefers replay (2 versus .1). Both must use the same global order.
        rank._graph_cache = SimpleNamespace(
            transfer_stats=SimpleNamespace(
                offload_bytes=100,
                restore_bytes=100,
                offload_seconds=0.1 if index == 0 else 1.0,
                restore_seconds=0.1 if index == 0 else 1.0,
            )
        )
        cost = ForwardMemoryCost(
            110, 110, 10, replay_seconds=0.5 if index == 0 else 0.1
        )
        units = [(0, (0,), cost, resolve_forward_options())]
        candidates = list(
            rank._graph_memory_candidates(units, sync_across_dp=sync_across_dp)
        )
        assert candidates[2][0] == "replay"
        assert candidates[2][2] == {
            "source": "measured_forward_and_transfers",
            "cpu_extra_seconds": 2.0,
            "replay_extra_seconds": 0.5,
            "preferred": "replay",
        }
        # One cold peer keeps every participant conservative.
        if index == 1:
            rank._graph_cache.transfer_stats.restore_bytes = 0
        candidates = list(
            rank._graph_memory_candidates(units, sync_across_dp=sync_across_dp)
        )
        assert candidates[2][0] == "cpu"
        assert candidates[2][2]["source"] == "insufficient_samples"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sync_across_dp", [False, True])
def test_measured_fallback_order_agrees_across_physical_peers(tmp_path, sync_across_dp):
    mp.spawn(
        _fallback_worker, args=(str(tmp_path), sync_across_dp), nprocs=2, join=True
    )


def _head_failure_worker(index: int, directory: str, failure: str) -> None:
    from test_trainer_rank_custom_tensors import _trainer

    from art.trainer_rank._commands import _Executor
    from art.trainer_rank._heads import LiveHead, export_head
    from art.trainer_rank._tensors import CotangentCollector

    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        trainer, api = _trainer("student")
        parameter = api.parameter("head", lambda: torch.ones(4), checkpoint="student")
        parameter.grad = torch.ones_like(parameter)
        collector = CotangentCollector()
        live = LiveHead(
            export_head(trainer, "student", "head"), torch.ones(4), collector
        )
        assert isinstance(live.value, torch.Tensor)
        packets = collector.backward(
            torch.stack([(live.value * scale).sum() for scale in (2, 3)]).sum()
        )
        cache = trainer._forward_graph_cache()
        setattr(
            cache,
            "backward_many",
            lambda *_a, **_k: pytest.fail("peer entered model backward"),
        )
        commit, convert = trainer._commit_versioned_gradients, torch.Tensor.to
        calls = 0

        def stage(gradients):
            nonlocal calls
            calls += 1
            if index == 0 and failure == "stage" and calls == 2:
                raise MemoryError("injected head stage allocation")
            return commit(gradients)

        source_ids = {id(packet.gradients[0]) for packet in packets}

        def copy(tensor, *args, **kwargs):
            if index == 0 and failure == "copy" and id(tensor) in source_ids:
                raise MemoryError("injected head copy allocation")
            return convert(tensor, *args, **kwargs)

        setattr(trainer, "_commit_versioned_gradients", stage)
        setattr(torch.Tensor, "to", copy)
        try:
            with pytest.raises((MemoryError, RuntimeError), match="head .* allocation"):
                _Executor(trainer, "zero")._backward(packets, retain_graph=False)
        finally:
            setattr(torch.Tensor, "to", convert)
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        assert trainer._version_state()._transaction is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["copy", "stage"])
def test_remote_head_allocation_failure_is_coordinated_before_model_backward(
    tmp_path, failure
):
    mp.spawn(_head_failure_worker, args=(str(tmp_path), failure), nprocs=2, join=True)
