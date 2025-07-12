import asyncio
from collections import Counter
from datasets import Dataset
from dataclasses import dataclass
from functools import cached_property
import gc
import os
import peft
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import PreTrainedModel, GenerationMixin
import torch
from trl import GRPOConfig, GRPOTrainer
from typing import AsyncIterator, TYPE_CHECKING, cast, Any
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev
from ..local.pack import DiskPackedTensors, packed_tensors_from_dir, PackedTensors
from .. import types
from ..vllm import get_llm, openai_server_task, run_on_workers
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..local.checkpoints import get_last_checkpoint_dir
from .train import train
from .sleep_worker import sleep

if TYPE_CHECKING:
    pass


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


class TrainInputs(PackedTensors):
    config: types.TrainConfig
    _config: dev.TrainConfig


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

        # Free memory after vLLM workers are asleep
        self._free_memory()

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

        # Train on the batch
        for offset in range(0, packed_tensors["tokens"].shape[0]):
            for _ in range(2 if warmup else 1):
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
                        self._free_memory()
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
        # self._save_lora_without_prefix(checkpoint_dir)

        # Free memory before waking up vLLM
        self._free_memory()

        # Remove pids.txt to signal workers to wake up
        if os.path.exists(pids_path):
            os.remove(pids_path)
            if verbose:
                print("Removed pids.txt to signal workers to wake up")

        # wait for the workers to wake up
        await sleep_task

        # TODO: Set the new LoRA adapter in vLLM
        # For now, we'll rely on vLLM to reload the LoRA from disk
        # self._set_lora(checkpoint_dir)

        if verbose:
            print("DecoupledUnslothService.train complete")
        print(
            "DEBUG: DecoupledUnslothService.train() method returning control to caller"
        )

    # def _save_lora_without_prefix(self, lora_path: str) -> None:
    #     """Save LoRA checkpoint with weights renamed to remove base_model prefix."""
    #     # First save normally
    #     self._state.trainer.save_model(lora_path)

    #     # Now load and fix the weights
    #     # Read the saved weights
    #     weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    #     state_dict = safetensors.torch.load_file(weights_path)

    #     # Remove base_model prefix from all keys
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith("base_model.model."):
    #             # Remove "base_model.model." prefix
    #             new_key = key[len("base_model.model.") :]
    #         elif key.startswith("base_model."):
    #             # Remove "base_model." prefix
    #             new_key = key[len("base_model.") :]
    #         else:
    #             new_key = key
    #         new_state_dict[new_key] = value

    #     # Save the fixed weights
    #     safetensors.torch.save_file(new_state_dict, weights_path)

    def _free_memory(self) -> None:
        """Free GPU memory."""
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    @cached_property
    def _state(self) -> UnslothState:
        # Import unsloth here to avoid loading it in vLLM worker process
        import unsloth  # type: ignore

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
        # Ensure we use vLLM V1 engine
        os.environ["VLLM_USE_V1"] = "1"
        # Get engine args and ensure LoRA is enabled
        engine_args = self.config.get("engine_args", {}).copy()
        engine_args["enable_lora"] = True
        engine_args["max_lora_rank"] = engine_args.get("max_lora_rank", 64)
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**engine_args))  # type: ignore
        )
