"""Shared setup for the native trainer-v1 validation programs."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoints

from art.trainer_rank import TrainerRank


@contextmanager
def nccl_group(local_rank=None):
    torch.cuda.set_device(
        int(os.environ["LOCAL_RANK"]) if local_rank is None else local_rank
    )
    dist.init_process_group("nccl")
    try:
        yield
    finally:
        dist.destroy_process_group()


def build_rank(model, *, layers=None, seed=90217, eval_mode=True, **runtime_options):
    from art.megatron.train import build_training_runtime

    torch.manual_seed(seed)
    runtime_options.setdefault("print_env", False)
    runtime_options.setdefault(
        "provider_configure",
        (lambda p: setattr(p, "num_layers", layers)) if layers is not None else None,
    )
    runtime = build_training_runtime(model_identifier=model, **runtime_options)
    if eval_mode:
        for chunk in runtime.model:
            chunk.eval()
    return TrainerRank(runtime)


def load_checkpoint(rank, model):
    (checkpoint,) = load_random_checkpoints(
        rank.runtime, rank, 1, base_model=model, lora_rank=2
    )
    return checkpoint


def deterministic_kernels():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    import art.megatron.flex_attn.compiled as flex

    setattr(flex, "_FORCED_FLEX_BACKEND", "TRITON")
    flex._FORCED_FLEX_KERNEL_OPTIONS = {"BACKEND": "TRITON"}
    flex.dense_compiled_flex_attention = flex.triton_dense_compiled_flex_attention
    flex.sparse_compiled_flex_attention = flex.triton_sparse_compiled_flex_attention


def source_commits():
    source_file = sys.modules["art.trainer_rank"].__file__
    assert source_file is not None
    source = Path(source_file).resolve().parents[3]
    harness = Path(__file__).resolve().parent.parent
    return {
        f"{name}_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True
        ).strip()
        for name, directory in (("source", source), ("harness", harness))
    }


def _leaves(tree):
    if isinstance(tree, (list, tuple)):
        return [leaf for child in tree for leaf in _leaves(child)]
    return [tree]


def _cached_backward(rank, loss):
    with rank._gradient_transaction():
        packets = rank._forward_cotangent_collector().backward(loss)
        rank._forward_graph_cache().backward_many(
            [(packet.handle, packet.gradients) for packet in packets]
        )
