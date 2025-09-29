import asyncio
import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

import torch

from art.utils.get_model_step import get_step_from_dir

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from .train import train

if TYPE_CHECKING:
    from unsloth_zoo.vllm_lora_request import LoRARequest  # type: ignore

    from .state import ModelState


class TrainInputs(PackedTensors):
    config: types.TrainConfig
    _config: dev.TrainConfig
    return_new_logprobs: bool


@dataclass
class UnslothService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _openai_server_task: asyncio.Task[None] | None = None
    _train_task: asyncio.Task[None] | None = None

    @functools.cached_property
    def state(self) -> "ModelState":
        from .state import ModelState

        return ModelState(self.config)

    @functools.cached_property
    def results_queue(self) -> asyncio.Queue[dict[str, float]]:
        return asyncio.Queue()

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        from ..vllm import openai_server_task

        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            from ..utils.output_dirs import get_step_checkpoint_dir

            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self.state.trainer.save_model(lora_path)
        await self.stop_openai_server()
        self._openai_server_task = await openai_server_task(
            engine=self.state.vllm.async_engine,
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                base_model=self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                lora_path=lora_path,
                config=config,
            ),
        )
        self._set_lora(lora_path)

    async def vllm_engine_is_sleeping(self) -> bool:
        return await self.state.vllm.async_engine.is_sleeping()

    async def stop_openai_server(self) -> None:
        if self._openai_server_task:
            self._openai_server_task.cancel()
            self._openai_server_task = None

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        # Get the packed tensors from disk
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        # Wait for existing batches to finish
        await self.results_queue.join()
        # If we haven't already, start the training task
        if self._train_task is None:
            self._train_task = asyncio.create_task(
                train(
                    trainer=self.state.trainer,
                    results_queue=self.results_queue,
                )
            )
            warmup = True
        else:
            warmup = False
        precalculate_logprobs = _config.get("precalculate_logprobs", False)
        # Enter training mode
        async with self.state.vllm.train_mode():
            for offset in range(0, packed_tensors["tokens"].shape[0]):
                for _ in range(2 if warmup else 1):
                    if precalculate_logprobs and not warmup:
                        packed_tensors["original_logprobs"] = packed_tensors["logprobs"]  # type: ignore
                        packed_tensors["logprobs"] = torch.cat(
                            [
                                self.state.trainer.compute_loss(
                                    self.state.peft_model,
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
                                for _offset in range(
                                    0, packed_tensors["tokens"].shape[0]
                                )
                            ]
                        ).to("cpu")
                        precalculate_logprobs = False
                    self.state.inputs_queue.put_nowait(
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
                            asyncio.create_task(self.results_queue.get()),
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
                        assert result is not None, (
                            "The training task should never finish."
                        )
                        self.results_queue.task_done()
                        if warmup:
                            from .state import gc_and_empty_cuda_cache

                            gc_and_empty_cuda_cache()
                            await asyncio.sleep(0.1)
                            warmup = False
                        else:
                            yield result
            if verbose:
                print("Saving new LoRA adapter...")
            # Save the new LoRA adapter
            from ..utils.output_dirs import get_step_checkpoint_dir

            next_step = get_step_from_dir(self.output_dir) + 1
            checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            self.state.trainer.save_model(checkpoint_dir)
            if verbose:
                print("Setting new LoRA adapter...")
            # Set the new LoRA adapter
            self._set_lora(checkpoint_dir)
            if verbose:
                print("New LoRA adapter set")

        if verbose:
            print("ModelService.train complete")

    def _set_lora(self, lora_path: str) -> None:
        """Sets the LoRA adapter with ID 1 in the vLLM engine."""
        lora_request: "LoRARequest" = self.state.peft_model.load_lora(
            lora_path,
            load_tensors=True,
        )  # type: ignore
        lora_request.lora_int_id = 1
        lora_request.lora_name = self.model_name
        lora_request.lora_path = lora_path
        self.state.vllm.async_engine.engine.remove_lora(1)
        self.state.vllm.async_engine.engine.add_lora(lora_request)  # type: ignore
