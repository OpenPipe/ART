from __future__ import annotations

from contextlib import ExitStack
import json
import os
from threading import Event
import time

import torch
import torch.distributed as dist
import typer

from art.distributed.data_plane import SharedMemoryPackedBatchStore
from art.megatron.runtime.data_plane import InMemoryPackedBatch
from art.megatron.runtime.executor import MegatronTrainJobExecutor
from art.megatron.runtime.specs import (
    CurrentTrainConfig,
    ForwardBackwardJobSpec,
    OptimizerJobSpec,
    TrainerGeneration,
)
from art.preprocessing.pack import PackedTensors
from art.training import AdamConfig, OperationRef


def main(
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    sequence_length: int = 128,
    contributions: int = 2,
) -> None:
    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
    if not 1 <= contributions <= 64:
        raise ValueError("contributions must be between 1 and 64")
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("megatron_split_command_check requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    started = time.perf_counter()
    executor = None
    store = None
    try:
        from art.megatron.train import build_training_runtime

        torch.manual_seed(1234)
        runtime = build_training_runtime(
            model_identifier=model,
            model_initialization="random",
            provider_configure=lambda provider: setattr(provider, "num_layers", layers),
            print_env=dist.get_rank() == 0,
        )
        source = _generation(0)
        runtime.resident_run_id = "split-command-check"
        runtime.resident_training_session_id = source.training_session_id
        runtime.resident_policy_step = source.policy_step
        runtime.resident_generation_id = source.generation_id
        runtime.optimizer_state_loaded = True
        runtime.adapter_export_dtypes = {}
        runtime.adapter_export_config = {}

        from megatron.core import parallel_state as ps

        global_batch_sequences = int(ps.get_data_parallel_world_size())

        executor = MegatronTrainJobExecutor(runtime)
        store = SharedMemoryPackedBatchStore(
            owner_actor_id=f"rank-{dist.get_rank()}", capacity_bytes=64 << 20
        )
        contribution_ids = []
        forward_backward_s = []
        with ExitStack() as batches:
            for index in range(contributions):
                tensors = _packed_tensors(
                    sequence_length,
                    sequences=global_batch_sequences,
                    offset=index * global_batch_sequences * sequence_length,
                )
                ref = store.create(tensors, batch_id=f"batch-{index}")
                batch = InMemoryPackedBatch.open(ref, ref)
                batches.callback(batch.close)
                operation_id = f"fb-{index}"
                operation = OperationRef(
                    run_id="split-command-check",
                    operation_id=operation_id,
                    sequence_id=index,
                    learner_parent_version=0,
                    kind="forward_backward",
                )
                job = ForwardBackwardJobSpec(
                    operation=operation,
                    training_session_id=source.training_session_id,
                    source=source,
                    optimizer_state_path="unused-resident-optimizer",
                    batch=ref,
                    expected_global_loss_bearing_tokens=int(
                        tensors["assistant_mask"][:, 1:].sum().item()
                    ),
                    config=CurrentTrainConfig(
                        grad_accumulation_sequences=global_batch_sequences
                    ),
                )
                result = executor.execute_forward_backward(
                    job, batch, cancelled=Event()
                )
                if result["operation_id"] != operation_id:
                    raise RuntimeError("F/B result operation identity changed")
                contribution_ids.append(operation_id)
                forward_backward_s.append(
                    float(result["metrics"]["time/forward_backward_s"])
                )

        optimizer = OptimizerJobSpec(
            operation=OperationRef(
                run_id="split-command-check",
                operation_id="optimizer",
                sequence_id=contributions,
                learner_parent_version=0,
                reserved_output_learner_version=1,
                kind="optim_step",
            ),
            training_session_id=source.training_session_id,
            source=source,
            optimizer_state_path="unused-resident-optimizer",
            generation=_generation(1),
            contributing_forward_backward_operation_ids=tuple(contribution_ids),
            optimizer=AdamConfig(learning_rate=1e-5),
        )
        optimizer_result = executor.execute_optimizer(optimizer)
        if optimizer_result["contributing_forward_backward_operation_ids"] != tuple(
            contribution_ids
        ):
            raise RuntimeError("optimizer consumed the wrong F/B contributions")
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "contributions": contribution_ids,
                        "forward_backward_s": forward_backward_s,
                        "optimizer_metrics": optimizer_result["metrics"],
                        "resident_policy_step": runtime.resident_policy_step,
                        "elapsed_s": time.perf_counter() - started,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                        "topology": {
                            "world": dist.get_world_size(),
                            "dp": int(ps.get_data_parallel_world_size()),
                            "tp": int(ps.get_tensor_model_parallel_world_size()),
                            "cp": int(ps.get_context_parallel_world_size()),
                            "ep": int(ps.get_expert_model_parallel_world_size()),
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.close()
        elif dist.is_initialized():
            dist.destroy_process_group()
        if store is not None:
            store.close()


def _generation(step: int) -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="split-command-session",
        policy_step=step,
        generation_id=f"step-{step:08d}-{'a' * 32}",
        adapter_path=f"unused-resident-adapter-{step}",
    )


def _packed_tensors(
    sequence_length: int, *, sequences: int, offset: int
) -> PackedTensors:
    shape = (sequences, sequence_length)
    tokens = (
        torch.arange(sequences * sequence_length, dtype=torch.long).reshape(shape)
        + offset
    ) % 32_000 + 100
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    assistant_mask[:, sequence_length // 2 :] = True
    return {
        "tokens": tokens,
        "group_ids": torch.zeros(shape, dtype=torch.long),
        "parent_ids": torch.zeros(shape, dtype=torch.long),
        "input_pos": torch.arange(sequence_length, dtype=torch.long)
        .expand(shape)
        .clone(),
        "assistant_mask": assistant_mask,
        "logprobs": torch.where(
            assistant_mask,
            torch.full(shape, -1.0, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        ),
        "advantages": assistant_mask.to(dtype=torch.float32),
        "weights": assistant_mask.to(dtype=torch.float32),
        "pixel_values": [None] * sequences,
        "image_grid_thw": [None] * sequences,
        "moe_routing_replay": None,
    }


if __name__ == "__main__":
    typer.run(main)
