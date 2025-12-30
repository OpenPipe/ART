import asyncio
import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import (
    DiskPackedTensors,
    packed_tensors_from_dir,
)
from .shared import (
    process_train_batch,
    save_checkpoint,
)
from .train import train

if TYPE_CHECKING:
    from .state import ModelState


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
            # Train on the batch using shared logic
            async for result in process_train_batch(
                packed_tensors=packed_tensors,
                config=config,
                _config=_config,
                inputs_queue=self.state.inputs_queue,
                results_queue=self.results_queue,
                train_task=self._train_task,
                trainer=self.state.trainer,
                peft_model=self.state.peft_model,
                warmup=warmup,
                verbose=verbose,
            ):
                yield result
            # Save the new LoRA adapter
            checkpoint_dir = save_checkpoint(
                trainer=self.state.trainer,
                output_dir=self.output_dir,
                verbose=verbose,
            )
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
        from vllm.lora.request import LoRARequest

        if hasattr(self.state.peft_model, "load_lora"):
            lora_request: LoRARequest = self.state.peft_model.load_lora(
                lora_path,
                load_tensors=True,
            )  # type: ignore
            lora_request.lora_int_id = 1
            lora_request.lora_name = self.model_name
            lora_request.lora_path = lora_path
        else:
            lora_request = LoRARequest(
                lora_name=self.model_name,
                lora_int_id=1,
                lora_path=lora_path,
            )
        self.state.vllm.async_engine.engine.remove_lora(1)
        self.state.vllm.async_engine.engine.add_lora(lora_request)  # type: ignore
