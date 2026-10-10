"""Shared fake runtimes and process groups for trainer-rank tests."""

from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from art.trainer_rank import TrainerRank


def fake_rank(rank_type: type["TrainerRank"], model, **provider) -> "TrainerRank":
    runtime: Any = SimpleNamespace(
        model=model,
        optimizer=None,
        provider=SimpleNamespace(**provider),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )
    return rank_type(runtime)


def recompute_model(
    block_type, hidden_size, num_layers, sequence_parallel, /, *, layers=(), **config
):
    block = block_type.__new__(block_type)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        hidden_size=hidden_size,
        num_layers=num_layers,
        padded_vocab_size=32,
        params_dtype=torch.bfloat16,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=sequence_parallel,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        **config,
    )
    block.layers = torch.nn.ModuleList(
        list(layers)
        + [torch.nn.Linear(1, 1).bfloat16() for _ in range(num_layers - len(layers))]
    )
    block.num_layers_per_pipeline_rank = num_layers
    model: Any = torch.nn.Module()
    model.config, model.decoder = block.config, block
    model._preprocess = lambda: None
    return model


@contextmanager
def process_group(rank, rendezvous, *, world_size=2, timeout=30, backend="gloo"):
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=None if timeout is None else timedelta(seconds=timeout),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


class AllReduceSum(torch.autograd.Function):
    """Megatron's reduce-from-tensor-parallel region: sum, identity backward."""

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.clone()
        dist.all_reduce(output)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0]


def all_reduce_max(tensor):
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.MAX)
    return output


def _traced_worker(worker, errors, rank, *args):
    try:
        worker(rank, *args)
    except BaseException:
        (Path(errors) / f"rank-{rank}.txt").write_text(traceback.format_exc())
        raise


def spawn_and_join(worker, args, *, timeout, failure, nprocs=2):
    """Bound a collective test while preserving every worker's traceback.

    A worker that fails tears down its process group, so its peers fail next
    with only a closed connection: each rank's own traceback is attached.
    """
    with tempfile.TemporaryDirectory() as errors:
        try:
            _spawn_and_join(
                partial(_traced_worker, worker, errors),
                args,
                timeout=timeout,
                failure=failure,
                nprocs=nprocs,
            )
        except BaseException as error:
            for path in sorted(Path(errors).glob("rank-*.txt")):
                error.add_note(f"{path.stem} traceback:\n{path.read_text()}")
            raise


def _spawn_and_join(worker, args, *, timeout, failure, nprocs):
    processes = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    error = None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Peers of a failed rank get time to record their own failure.
            if processes.join(timeout=1, grace_period=10):
                return
        pytest.fail(failure)
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            survivors = [p.pid for p in processes.processes if p.is_alive()]
            if survivors:
                pytest.fail(f"Spawned workers survived SIGKILL: {survivors}")
        except BaseException as cleanup_error:
            if error is None:
                raise
            try:
                BaseException.add_note(
                    error, f"Worker cleanup failed: {cleanup_error!r}"
                )
            except BaseException:
                pass
