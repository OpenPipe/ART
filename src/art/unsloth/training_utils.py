"""Training utilities that don't depend on vLLM.

These functions are extracted from unsloth/service.py to allow use
by backends that don't use vLLM (e.g., SGLang backend).
"""

import asyncio
import os
from typing import TYPE_CHECKING, AsyncIterator

import torch

from .. import dev, types
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import PackedTensors
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from .train import gc_and_empty_cuda_cache

if TYPE_CHECKING:
    from peft.peft_model import PeftModelForCausalLM
    from trl import GRPOTrainer


def precalculate_new_logprobs(
    trainer: "GRPOTrainer",
    peft_model: "PeftModelForCausalLM",
    packed_tensors: PackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
) -> torch.Tensor:
    """Precalculate logprobs for all offsets and return as a tensor."""
    return torch.cat(
        [
            trainer.compute_loss(
                peft_model,
                TrainInputs(  # ty:ignore[missing-typed-dict-key]
                    **{
                        k: v[_offset : _offset + 1]
                        for k, v in packed_tensors.items()
                        if isinstance(v, torch.Tensor)
                    },
                    pixel_values=packed_tensors["pixel_values"][_offset : _offset + 1],
                    image_grid_thw=packed_tensors["image_grid_thw"][
                        _offset : _offset + 1
                    ],
                    config=config,
                    _config=_config,
                    return_new_logprobs=True,
                ),
            )
            for _offset in range(0, packed_tensors["tokens"].shape[0])
        ]
    ).to("cpu")


async def process_train_batch(
    packed_tensors: PackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
    inputs_queue: asyncio.Queue[TrainInputs],
    results_queue: asyncio.Queue[dict[str, float]],
    train_task: asyncio.Task[None],
    trainer: "GRPOTrainer",
    peft_model: "PeftModelForCausalLM",
    warmup: bool,
    verbose: bool = False,
) -> AsyncIterator[dict[str, float]]:
    """
    Process training batches and yield results.

    Yields tuples of (result, warmup_done) where warmup_done indicates if warmup just finished.
    """
    precalculate_logprobs = _config.get("precalculate_logprobs", False)

    for offset in range(0, packed_tensors["tokens"].shape[0]):
        for _ in range(2 if warmup else 1):
            if precalculate_logprobs and not warmup:
                # Preserve original logprobs before overwriting
                packed_tensors["original_logprobs"] = packed_tensors["logprobs"]  # type: ignore
                packed_tensors["logprobs"] = precalculate_new_logprobs(
                    trainer, peft_model, packed_tensors, config, _config
                )
                precalculate_logprobs = False

            inputs_queue.put_nowait(
                create_train_inputs(packed_tensors, offset, config, _config, warmup)
            )

            # Wait for a result from the queue or for the training task to,
            # presumably, raise an exception
            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(results_queue.get()),
                    train_task,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if verbose:
                print(
                    "Done waiting for a result from the queue or for the training task to, presumably, raise an exception"
                )
            for task in done:
                result = task.result()
                # If `result` is `None`, the training task finished somehow.
                assert result is not None, "The training task should never finish."
                results_queue.task_done()
                if warmup:
                    gc_and_empty_cuda_cache()
                    await asyncio.sleep(0.1)
                    warmup = False
                else:
                    yield result


def save_checkpoint(
    trainer: "GRPOTrainer",
    output_dir: str,
    verbose: bool = False,
) -> str:
    """Save a checkpoint and return the checkpoint directory path."""
    if verbose:
        print("Saving new LoRA adapter...")
    next_step = get_step_from_dir(output_dir) + 1
    checkpoint_dir = get_step_checkpoint_dir(output_dir, next_step)
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save_model(checkpoint_dir)
    return checkpoint_dir
