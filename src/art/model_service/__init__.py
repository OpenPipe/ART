if __name__ == "__main__":
    import unsloth  # type: ignore


import asyncio
import atexit
import httpx
from fastapi import FastAPI
import functools
import os
from peft.peft_model import PeftModel
from pydantic import BaseModel
import signal
import sys
import torch
import traceback
from transformers import PreTrainedTokenizerBase
import uvicorn
from trl import GRPOConfig, GRPOTrainer
from typing import Awaitable, Callable, cast, ParamSpec, TypeVar

from art import types
from art.unsloth.checkpoints import get_iteration, get_last_iteration_dir
from art.unsloth.model import get_model_and_tokenizer
from art.unsloth.pack import DiskPackedTensors, packed_tensors_from_dir, PackedTensors
from art.unsloth.train import get_trainer, train
from art.unsloth.vllm import openai_server_task, set_vllm_log_file

T = TypeVar("T")
P = ParamSpec("P")


def catch_and_print_errors(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T]]:
    """
    Decorator that catches, prints, and reraises any errors that occur in the wrapped function.
    """

    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await cast(Awaitable[T], func(*args, **kwargs))
        except Exception as e:
            if __name__ == "__main__":
                print(f"Error in {func.__name__}: {e}")
                traceback.print_exc()
            raise

    return async_wrapper


class StartOpenaiServer(BaseModel):
    tool_use: bool


class TuneInputs(PackedTensors):
    config: types.TuneConfig


class ModelService(BaseModel):
    host: str
    port: int
    model_name: str
    base_model: "types.BaseModel"
    output_dir: str
    process: asyncio.subprocess.Process | None = None
    _openai_server_task: asyncio.Task | None = None
    _train_task: asyncio.Task | None = None

    class Config:
        arbitrary_types_allowed = True
        exclude = {
            "process",  # asyncio.subprocess.Process can't be serialized
            "_openai_server_task",  # asyncio.Task can't be serialized
            "_train_task",  # asyncio.Task can't be serialized
            "model_and_tokenizer",  # cached property
            "packed_tensors_queue",  # cached property
            "trainer",  # cached property
        }
        extra = "allow"

    @catch_and_print_errors
    async def start_openai_server(self, request: StartOpenaiServer) -> None:
        peft_model, _ = self.model_and_tokenizer
        lora_path = get_last_iteration_dir(self.output_dir)
        if lora_path is None:
            lora_path = f"{self.output_dir}/0000"
            self.trainer.save_model(lora_path)
        await self.stop_openai_server()
        self._openai_server_task = openai_server_task(
            model=peft_model,
            model_name=self.model_name,
            base_model=self.base_model,
            tool_use=request.tool_use,
            lora_path=lora_path,
        )
        done, _ = await asyncio.wait([self._openai_server_task], timeout=1.0)
        for task in done:
            task.result()

    @catch_and_print_errors
    async def stop_openai_server(self) -> None:
        if self._openai_server_task:
            self._openai_server_task.cancel()
            self._openai_server_task = None

    @catch_and_print_errors
    async def tune(
        self, disk_packed_tensors: DiskPackedTensors, config: types.TuneConfig
    ) -> None:
        from unsloth_zoo.training_utils import set_training, unset_training  # type: ignore

        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        await self.inputs_queue.join()
        model, _ = self.model_and_tokenizer
        set_training(model)
        trainer = self.trainer
        for i in range(packed_tensors["tokens"].shape[0]):
            self.inputs_queue.put_nowait(
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
            self._train_task = asyncio.create_task(train(trainer, self.inputs_queue))
        (done,), _ = await asyncio.wait(
            [self._train_task, asyncio.create_task(self.inputs_queue.join())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # unset_training(peft_model)
        if exception := done.exception():
            raise exception
        # Save the new lora
        iteration_dir = f"{self.output_dir}/{get_iteration(self.output_dir) + 1:04d}"
        trainer.save_model(iteration_dir)
        # Swap in the new lora
        lora_request = model.load_lora(
            iteration_dir,
            load_tensors=True,
        )
        lora_request.lora_int_id = 1
        lora_request.lora_name = self.model_name
        model.vllm_engine.engine.remove_lora(1)
        model.vllm_engine.engine.add_lora(lora_request)

    async def serve(self) -> "ModelService":
        # Ensure logs directory exists
        os.makedirs("./logs", exist_ok=True)

        # Serialize the service object to JSON
        service_json = self.model_dump_json()

        # Open log file
        log_file = open("./logs/unsloth.log", "w")

        # Start the service as an asyncio subprocess
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            service_json,
            stdout=log_file,
            stderr=log_file,
        )

        # Register cleanup handlers for process termination
        def cleanup_process():
            if self.process is not None and self.process.returncode is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass  # Process already gone

        atexit.register(cleanup_process)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda signum, frame: cleanup_process())

        # Create a client for the service
        client = httpx.AsyncClient(base_url=f"http://{self.host}:{self.port}")

        async def start_openai_server(request: StartOpenaiServer) -> None:
            response = await client.post(
                "/start_openai_server",
                json=request.model_dump(),
                timeout=httpx.Timeout(600.0),
            )
            response.raise_for_status()

        async def stop_openai_server() -> None:
            response = await client.post("/stop_openai_server")
            response.raise_for_status()

        async def tune(
            disk_packed_tensors: "DiskPackedTensors", config: types.TuneConfig
        ) -> None:
            response = await client.post(
                "/tune",
                json={
                    "disk_packed_tensors": disk_packed_tensors,
                    "config": config.model_dump(),
                },
                timeout=httpx.Timeout(None),
            )
            response.raise_for_status()

        # Set the endpoints
        self.start_openai_server = start_openai_server
        self.stop_openai_server = stop_openai_server
        self.tune = tune

        while True:
            try:
                await client.get("/health")
                break
            except httpx.ConnectError:
                await asyncio.sleep(1)

        print(f"Started service on {self.host}:{self.port} (PID: {self.process.pid})")
        return self

    @functools.cached_property
    def model_and_tokenizer(self) -> tuple["PeftModel", "PreTrainedTokenizerBase"]:
        return get_model_and_tokenizer(self.base_model)

    @functools.cached_property
    def inputs_queue(self) -> asyncio.Queue[TuneInputs]:
        return asyncio.Queue()

    @functools.cached_property
    def trainer(self) -> GRPOTrainer:
        peft_model, tokenizer = self.model_and_tokenizer
        self._trainer = get_trainer(
            model=peft_model,
            tokenizer=tokenizer,
            args=GRPOConfig(
                learning_rate=5e-6,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.1,
                lr_scheduler_type="constant",
                optim="paged_adamw_8bit",
                beta=0.0,
                logging_steps=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,  # Increase to 4 for smoother training
                num_generations=4,  # Decrease if out of memory
                save_strategy="no",
                # max_steps=1_000_000,
                # save_steps=1_000_000,
                # max_grad_norm=10.0,
                # report_to="none",  # Can use Weights & Biases
                output_dir=self.output_dir,
            ),
            inputs_queue=self.inputs_queue,
        )
        return self._trainer


if __name__ == "__main__":
    service = ModelService.model_validate_json(sys.argv[1])
    app = FastAPI()
    app.get("/health")(lambda: "OK")
    app.post("/start_openai_server")(service.start_openai_server)
    app.post("/stop_openai_server")(service.stop_openai_server)
    app.post("/tune")(service.tune)
    set_vllm_log_file(f"{service.output_dir}/logs/vllm.log")
    uvicorn.run(app, host=service.host, port=service.port, loop="asyncio")
