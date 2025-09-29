import asyncio
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Any, AsyncIterator, cast

import peft
import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import GenerationMixin, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer
from vllm import AsyncEngineArgs
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import logger

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from .train import gc_and_empty_cuda_cache, train


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


class TrainInputs(PackedTensors):
    config: types.TrainConfig
    _config: dev.TrainConfig
    return_new_logprobs: bool


@dataclass
class UnslothState:
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]


@dataclass
class DecoupledUnslothService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            # Create initial LoRA checkpoint if none exists
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self._state.trainer.save_model(lora_path)
        await openai_server_task(
            engine=await self.llm,
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                base_model=self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                lora_path=lora_path,
                config=config,
            ),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        llm = await self.llm
        pids_path = f"{self.output_dir}/pids.txt"
        # reset the pids file
        with open(pids_path, "w") as f:
            f.write("")
        # start putting the workers to sleep
        sleep_task = asyncio.create_task(
            run_on_workers(
                llm,
                sleep,
                level=1,
                pids_path=pids_path,
                profile=verbose,
            )
        )
        # wait for the workers to write their pids twice, indicating that they are asleep
        while True:
            pids = Counter(open(pids_path).read().splitlines())
            if set(pids.values()) == {2}:
                break
            await asyncio.sleep(0.25)

        self._is_sleeping = True

        # Free memory after vLLM workers are asleep
        gc_and_empty_cuda_cache()

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for existing batches to finish
        await self._state.results_queue.join()

        # If we haven't already, start the training task
        if not hasattr(self, "_train_task") or self._train_task is None:
            self._train_task = asyncio.create_task(
                train(
                    trainer=self._state.trainer,
                    results_queue=self._state.results_queue,
                )
            )
            warmup = True
        else:
            warmup = False
        precalculate_logprobs = _config.get("precalculate_logprobs", False)
        # Train on the batch
        for offset in range(0, packed_tensors["tokens"].shape[0]):
            for _ in range(2 if warmup else 1):
                if precalculate_logprobs and not warmup:
                    packed_tensors["logprobs"] = torch.cat(
                        [
                            self._state.trainer.compute_loss(
                                self._state.peft_model,
                                TrainInputs(
                                    **{
                                        k: v[_offset : _offset + 1]
                                        for k, v in packed_tensors.items()
                                        if isinstance(v, torch.Tensor)
                                    },
                                    config=config,
                                    _config=_config,
                                    return_new_logprobs=True,
                                ),  # type: ignore
                            )
                            for _offset in range(0, packed_tensors["tokens"].shape[0])
                        ]
                    ).to("cpu")
                    precalculate_logprobs = False
                self._state.inputs_queue.put_nowait(
                    TrainInputs(
                        **{
                            k: (
                                v[offset : offset + 1, :1024]
                                if warmup and v.dim() > 1
                                else v[offset : offset + 1]
                            )
                            for k, v in packed_tensors.items()
                            if isinstance(v, torch.Tensor)
                        },
                        config=(
                            config.model_copy(
                                update={"lr": 1e-9, "beta": 0.0, "kl_coef": 0.0}
                            )
                            if warmup
                            else config
                        ),
                        _config=_config,
                        return_new_logprobs=False,
                    )
                )
                # Wait for a result from the queue or for the training task to,
                # presumably, raise an exception
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(self._state.results_queue.get()),
                        self._train_task,
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
                    self._state.results_queue.task_done()
                    if warmup:
                        gc_and_empty_cuda_cache()
                        await asyncio.sleep(0.1)
                        warmup = False
                    else:
                        yield result

        if verbose:
            print("Saving new LoRA adapter...")
        # Save checkpoint after training
        next_step = get_step_from_dir(self.output_dir) + 1
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._state.trainer.save_model(checkpoint_dir)

        # Free memory before waking up vLLM
        gc_and_empty_cuda_cache()

        # Remove pids.txt to signal workers to wake up
        if os.path.exists(pids_path):
            os.remove(pids_path)
            if verbose:
                print("Removed pids.txt to signal workers to wake up")

        # wait for the workers to wake up
        await sleep_task

        self._is_sleeping = False

        # swap out the LoRA adapter
        await llm.remove_lora(1)
        await llm.add_lora(
            LoRARequest(
                lora_name=self.model_name,
                lora_int_id=1,
                lora_path=checkpoint_dir,
            )
        )

        if verbose:
            print("DecoupledUnslothService.train complete")

    @cached_property
    def _state(self) -> UnslothState:
        import unsloth

        # Initialize Unsloth model
        init_args = self.config.get("init_args", {})
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            init_args["model_name"] = checkpoint_dir
        else:
            init_args["model_name"] = self.base_model

        model, tokenizer = cast(
            tuple[CausalLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )

        # Initialize PEFT model
        peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                model, **self.config.get("peft_args", {})
            ),
        )

        # Initialize trainer with dummy dataset
        data = {"prompt": ""}
        trainer = GRPOTrainer(
            model=peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),  # type: ignore
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=tokenizer,
        )

        # Initialize queues
        inputs_queue: asyncio.Queue[TrainInputs] = asyncio.Queue()
        results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

        # Patch trainer _prepare_inputs() to pull from queue
        def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
            async def get_inputs() -> TrainInputs:
                return await inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        trainer._prepare_inputs = _async_prepare_inputs

        return UnslothState(
            model=model,
            tokenizer=tokenizer,
            peft_model=peft_model,
            trainer=trainer,
            inputs_queue=inputs_queue,
            results_queue=results_queue,
        )

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        return asyncio.create_task(
            get_llm(
                AsyncEngineArgs(
                    **{**self.config.get("engine_args", {}), "enable_lora": True}
                )
            )
        )


def sleep(*, level: int, pids_path: str, profile: bool) -> None:
    """
    Put the worker to sleep until signaled to wake up.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        profile: Whether to profile
    """
    with open(pids_path, "a") as f:
        f.write(f"{os.getpid()}\n")
    worker = get_worker()
    allocator = CuMemAllocator.get_instance()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        setattr(allocator, "_override_tags", {"weights", "kv_cache"})
        with worker.time("sleep"):
            worker.sleep(level)
        with open(pids_path, "a") as f:
            f.write(f"{os.getpid()}\n")

        # Wait for the signal to wake up
        while os.path.exists(pids_path):
            time.sleep(1)

        with worker.time("wake_up"):
            worker.wake_up()
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")
