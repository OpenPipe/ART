"""Shared runtime construction and process groups for trainer-rank contract tests."""

from contextlib import contextmanager
from datetime import timedelta
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from art.megatron.train import TrainingRuntime


def checkpoint_runtime(
    model: torch.nn.Module | None = None,
    *,
    optimizer: object | None = None,
) -> "TrainingRuntime":
    # Deliberately lightweight structural fake; importing/constructing the real
    # Megatron runtime would make these CPU-only unit tests require Megatron.
    return SimpleNamespace(
        model=[model or torch.nn.Linear(1, 1)],
        optimizer=optimizer,
        provider=SimpleNamespace(
            hidden_size=4,
            num_layers=1,
            kv_channels=2,
            art_flex_sliding_windows=(16,),
        ),
        model_support_handler=SimpleNamespace(
            build_gdn_execution_spec=True,
            canonicalize_loaded_lora_state=lambda state, _model: state,
            from_vllm_lora_tensors=lambda state, **_kwargs: state,
            to_vllm_lora_tensors=lambda state, **kwargs: (
                state,
                kwargs["adapter_config"],
            ),
            zero_internal_padding_grads=lambda _model: None,
            zero_internal_padding_params=lambda _model: None,
        ),
        rank=0,
        world_size=1,
    )  # type: ignore


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


gloo_group = process_group


@contextmanager
def megatron_topology(physical, *, dp_size, tp_size):
    """Install just the callback topology, with each real TP group created in order."""
    assert dist.get_world_size() == dp_size * tp_size
    groups = (
        [dist.group.WORLD]
        if dp_size == 1
        else [
            dist.new_group(list(range(dp * tp_size, (dp + 1) * tp_size)))
            for dp in range(dp_size)
        ]
    )
    dp, tp = divmod(physical, tp_size)
    megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
    setattr(
        core,
        "parallel_state",
        SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: tp,
            get_context_parallel_rank=lambda: 0,
            get_data_parallel_rank=lambda: dp,
            get_data_parallel_world_size=lambda: dp_size,
            get_tensor_and_context_parallel_group=lambda **kwargs: groups[dp],
        ),
    )
    setattr(megatron, "core", core)
    with pytest.MonkeyPatch.context() as modules:
        modules.setitem(sys.modules, "megatron", megatron)
        modules.setitem(sys.modules, "megatron.core", core)
        yield getattr(core, "parallel_state")


def spawn_and_join(worker, args, *, timeout, failure, nprocs=2):
    """Bound a collective test while preserving spawned-worker tracebacks."""
    processes = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    error = None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if processes.join(timeout=1):
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
