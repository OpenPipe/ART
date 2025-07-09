import asyncio
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import gc
import logging
import os
from pathlib import Path
from safetensors.torch import save_file, load_file
import time
import torch
from typing import AsyncIterator, TYPE_CHECKING, cast
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
import unsloth  # type: ignore
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import PreTrainedModel, GenerationMixin
import peft
from trl import GRPOConfig, GRPOTrainer

from .. import dev
from ..local.pack import DiskPackedTensors, packed_tensors_from_dir, PackedTensors
from .. import types
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..local.checkpoints import get_last_checkpoint_dir

if TYPE_CHECKING:
    from unsloth_zoo.vllm_lora_request import LoRARequest  # type: ignore


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
        weights_path = "/dev/shm/weights.safetensors"
        # remove the weights file if it exists
        Path(weights_path).unlink(missing_ok=True)
        
        # start putting the workers to sleep
        sleep_task = asyncio.create_task(
            run_on_workers(
                llm,
                sleep,
                level=1,
                pids_path=pids_path,
                weights_path=weights_path,
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
        if not hasattr(self, '_trainer'):
            self._init_unsloth()
        
        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        
        # Train on the batch
        results = []
        num_sequences = packed_tensors["tokens"].shape[0]
        
        for offset in range(num_sequences):
            # Prepare inputs for this sequence
            inputs = TrainInputs(
                **{
                    k: v[offset : offset + 1]
                    for k, v in packed_tensors.items()
                    if isinstance(v, torch.Tensor)
                },
                config=config,
                _config=_config,
            )
            
            # Run one training step
            metrics = self._train_step(inputs)
            results.append(metrics)
            metrics["num_gradient_steps"] = num_sequences
            yield metrics
        
        # Save checkpoint after training
        next_step = get_step_from_dir(self.output_dir) + 1
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._trainer.save_model(checkpoint_dir)
        
        # Export weights for vLLM to load
        self._export_lora_weights(weights_path)
        
        # Free memory before waking up vLLM
        self._free_memory()
        
        # wait for the workers to wake up
        await sleep_task
        # remove the weights file
        Path(weights_path).unlink(missing_ok=True)

    def _init_unsloth(self) -> None:
        """Initialize Unsloth model and trainer."""
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
            args=GRPOConfig(**self.config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=self._tokenizer,
        )
        
        # Initialize results queue
        self._results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

    def _train_step(self, inputs: TrainInputs) -> dict[str, float]:
        """Run a single training step."""
        # Override _prepare_inputs to return our inputs
        self._trainer._prepare_inputs = lambda *_, **__: cast(dict[str, torch.Tensor], inputs)
        
        # Zero gradients
        self._trainer.optimizer.zero_grad()
        
        # Get model outputs
        outputs = self._trainer.model(**inputs)
        
        # Calculate loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        if inputs.config.learning_rate > 0:
            # Update learning rate
            for param_group in self._trainer.optimizer.param_groups:
                param_group['lr'] = inputs.config.learning_rate
            
            self._trainer.optimizer.step()
        
        # Prepare metrics
        metrics = {
            "loss": loss.item(),
        }
        
        # Add reward if available
        if "rewards" in inputs:
            metrics["reward"] = inputs["rewards"].mean().item()
            
        return metrics

    def _export_lora_weights(self, weights_path: str) -> None:
        """Export LoRA weights for vLLM to load."""
        # Get LoRA weights
        lora_state_dict = {}
        for name, param in self._peft_model.named_parameters():
            if param.requires_grad:  # Only LoRA parameters
                lora_state_dict[name] = param.detach().cpu()
        
        # Save weights
        save_file(lora_state_dict, weights_path)

    def _save_initial_lora(self, lora_path: str) -> None:
        """Save initial LoRA checkpoint when none exists."""
        # We need to temporarily initialize Unsloth to save the initial LoRA
        if not hasattr(self, '_trainer'):
            self._init_unsloth()
        self._trainer.save_model(lora_path)
        # Clean up to free memory
        del self._trainer
        del self._peft_model
        del self._model
        del self._tokenizer
        self._free_memory()

    def _free_memory(self) -> None:
        """Free GPU memory."""
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        # Ensure we use vLLM V1 engine
        os.environ["VLLM_USE_V1"] = "1"
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**self.config.get("engine_args", {})))  # type: ignore
        )


def sleep(
    *, level: int, pids_path: str, weights_path: str, profile: bool
) -> None:
    """
    Put the worker to sleep until the new model weights are loaded.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        weights_path: The path to the weights file.
        profile: Whether to profile
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.worker.gpu_worker import logger

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
        
        # Wait for weights to be saved by the training process
        while True:
            if os.path.exists(weights_path):
                # Load the new weights
                with worker.time("load_file"):
                    weights = load_file(weights_path)
                break
            elif not os.path.exists(pids_path):
                # no pids file indicates we can wake up without new weights
                weights = None
                break
            else:
                time.sleep(1)
                continue
                
        with worker.time("wake_up"):
            worker.wake_up()
        
        if weights is not None:
            with worker.time("load_weights"):
                worker.model_runner.model.load_weights(weights.items())  # type: ignore
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")