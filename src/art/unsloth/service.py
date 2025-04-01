import asyncio
from dataclasses import dataclass
import functools
import torch
from typing import TYPE_CHECKING

from .. import types
from .checkpoints import get_iteration, get_last_iteration_dir
from ..config.model import ModelConfig
from ..config.openai_server import get_openai_server_config, OpenAIServerConfig
from .pack import DiskPackedTensors, packed_tensors_from_dir, PackedTensors
from .train import train
from .vllm import openai_server_task

if TYPE_CHECKING:
    from unsloth_zoo.vllm_lora_request import LoRARequest

    from .state import ModelState


class TuneInputs(PackedTensors):
    config: types.TuneConfig


@dataclass
class ModelService:
    host: str
    port: int
    model_name: str
    base_model: types.BaseModel
    config: ModelConfig
    output_dir: str
    _openai_server_task: asyncio.Task[None] | None = None
    _train_task: asyncio.Task[None] | None = None

    async def start_openai_server(
        self, tool_use: bool, config: OpenAIServerConfig | None
    ) -> None:
        lora_path = get_last_iteration_dir(self.output_dir)
        if lora_path is None:
            lora_path = f"{self.output_dir}/0000"
            self.state.trainer.save_model(lora_path)
        await self.stop_openai_server()
        self._openai_server_task = openai_server_task(
            state=self.state.vllm,
            config=get_openai_server_config(
                model_name=self.model_name,
                base_model=self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                lora_path=lora_path,
                tool_use=tool_use,
                config=config,
            ),
        )
        done, _ = await asyncio.wait([self._openai_server_task], timeout=1.0)
        for task in done:
            task.result()

    async def stop_openai_server(self) -> None:
        if self._openai_server_task:
            self._openai_server_task.cancel()
            self._openai_server_task = None

    async def tune(
        self, disk_packed_tensors: DiskPackedTensors, config: types.TuneConfig
    ) -> None:
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        inputs_queue = self.state.inputs_queue
        await inputs_queue.join()
        peft_model = self.state.peft_model
        trainer = self.state.trainer
        for i in range(packed_tensors["tokens"].shape[0]):
            inputs_queue.put_nowait(
                TuneInputs(
                    **{
                        k: v[i : i + 1]
                        for k, v in packed_tensors.items()
                        if isinstance(v, torch.Tensor)
                    },
                    config=config,
                )
            )
        if self._train_task is None:
            self._train_task = asyncio.create_task(train(trainer, inputs_queue))
        (done,), _ = await asyncio.wait(
            [self._train_task, asyncio.create_task(inputs_queue.join())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if exception := done.exception():
            raise exception
        # Save the new lora
        iteration_dir = f"{self.output_dir}/{get_iteration(self.output_dir) + 1:04d}"
        trainer.save_model(iteration_dir)
        # Swap in the new lora
        lora_request: "LoRARequest" = peft_model.load_lora(
            iteration_dir,
            load_tensors=True,
        )
        lora_request.lora_int_id = 1
        lora_request.lora_name = self.model_name
        self.state.vllm.async_engine.engine.remove_lora(1)
        self.state.vllm.async_engine.engine.add_lora(lora_request)  # type: ignore

    @functools.cached_property
    def state(self) -> "ModelState":
        from .state import ModelState

        return ModelState(self.config)
