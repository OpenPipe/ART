import asyncio
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import gc
import os
from pathlib import Path
import torch
from typing import AsyncIterator, TYPE_CHECKING, cast, Any
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
# Moved unsloth import to _init_unsloth to avoid import in worker process
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import PreTrainedModel, GenerationMixin
import peft
from trl import GRPOConfig, GRPOTrainer
import safetensors.torch

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
            self._save_initial_lora(lora_path)

        # Store the engine for later use
        self._engine = await self.llm
        
        # Skip setting initial LoRA for now - vLLM will handle it through lora_path
        
        await openai_server_task(
            engine=self._engine,
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

        # Initialize Unsloth trainer if not already done
        if not hasattr(self, "_trainer"):
            self._init_unsloth()

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for existing batches to finish
        await self._results_queue.join()

        # If we haven't already, start the training task
        if not hasattr(self, "_train_task") or self._train_task is None:
            self._train_task = asyncio.create_task(
                train(
                    trainer=self._trainer,
                    results_queue=self._results_queue,
                )
            )
            warmup = True
        else:
            warmup = False

        # Train on the batch
        for offset in range(0, packed_tensors["tokens"].shape[0]):
            for _ in range(2 if warmup else 1):
                self._inputs_queue.put_nowait(
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
                        asyncio.create_task(self._results_queue.get()),
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
                    self._results_queue.task_done()
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
        self._save_lora_without_prefix(checkpoint_dir)

        # Free memory before waking up vLLM
        self._free_memory()

        # Remove pids.txt to signal workers to wake up
        pids_path = f"{self.output_dir}/pids.txt"
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
        print("DEBUG: DecoupledUnslothService.train() method returning control to caller")

    def _init_unsloth(self) -> None:
        """Initialize Unsloth model and trainer."""
        # Import unsloth here to avoid loading it in vLLM worker process
        import unsloth  # type: ignore
        
        # Initialize Unsloth model
        init_args = self.config.get("init_args", {})
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            init_args["model_name"] = checkpoint_dir
        else:
            init_args["model_name"] = self.base_model

        self._model, self._tokenizer = cast(
            tuple[CausalLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )

        # Initialize PEFT model
        self._peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self._model, **self.config.get("peft_args", {})
            ),
        )

        # Initialize trainer with dummy dataset
        data = {"prompt": ""}
        self._trainer = GRPOTrainer(
            model=self._peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),  # type: ignore
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=self._tokenizer,
        )

        # Initialize queues
        self._inputs_queue: asyncio.Queue[TrainInputs] = asyncio.Queue()
        self._results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

        # Patch trainer _prepare_inputs() to pull from queue
        def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
            async def get_inputs() -> TrainInputs:
                return await self._inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        self._trainer._prepare_inputs = _async_prepare_inputs

    def _set_lora(self, lora_path: str) -> None:
        """Sets the LoRA adapter with ID 1 in the vLLM engine."""
        from vllm.lora.request import LoRARequest
        
        # Create LoRA request
        lora_request = LoRARequest(
            lora_name=self.model_name,
            lora_int_id=1,
            lora_path=lora_path,
        )
        
        # Remove old LoRA and add new one
        # AsyncLLM has engine_core attribute which contains the actual engine
        engine_core = getattr(self._engine, 'engine_core', None)
        if engine_core and hasattr(engine_core, 'engine'):
            engine = engine_core.engine
        else:
            # Try to find the engine through model_executor
            engine = self._engine
        
        # Remove and add LoRA
        if hasattr(engine, 'remove_lora'):
            engine.remove_lora(1)
            engine.add_lora(lora_request)
        elif hasattr(engine, 'engine_core'):
            engine.engine_core.engine.remove_lora(1)
            engine.engine_core.engine.add_lora(lora_request)

    def _save_initial_lora(self, lora_path: str) -> None:
        """Save initial LoRA checkpoint when none exists."""
        # We need to temporarily initialize Unsloth to save the initial LoRA
        if not hasattr(self, "_trainer"):
            self._init_unsloth()
        self._save_lora_without_prefix(lora_path)
        # Clean up to free memory
        del self._trainer
        del self._peft_model
        del self._model
        del self._tokenizer
        self._free_memory()

    def _save_lora_without_prefix(self, lora_path: str) -> None:
        """Save LoRA checkpoint with weights renamed to remove base_model prefix."""
        # First save normally
        self._trainer.save_model(lora_path)
        
        # Now load and fix the weights
        # Read the saved weights
        weights_path = os.path.join(lora_path, "adapter_model.safetensors")
        state_dict = safetensors.torch.load_file(weights_path)
        
        # Remove base_model prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("base_model.model."):
                # Remove "base_model.model." prefix
                new_key = key[len("base_model.model."):]
            elif key.startswith("base_model."):
                # Remove "base_model." prefix  
                new_key = key[len("base_model."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Save the fixed weights
        safetensors.torch.save_file(new_state_dict, weights_path)

    def _free_memory(self) -> None:
        """Free GPU memory."""
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

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


