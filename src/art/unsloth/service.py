"""Unsloth training service with decoupled vLLM inference."""

import asyncio
from dataclasses import dataclass
from functools import cached_property
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, cast

from datasets import Dataset
import peft
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import GenerationMixin, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from .train import gc_and_empty_cuda_cache, train

if TYPE_CHECKING:
    from peft.peft_model import PeftModelForCausalLM
    from trl import GRPOTrainer


# ============================================================================
# Shared Utilities
# ============================================================================


class SupportsLoadLora(Protocol):
    """Protocol for models that support the optimized load_lora method."""

    def load_lora(self, lora_path: str, load_tensors: bool = True) -> LoRARequest: ...


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
):
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


# ============================================================================
# Model Classes
# ============================================================================


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


@dataclass
class UnslothState:
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]
    _is_offloaded: bool = False
    _pinned_buffers: dict[str, torch.Tensor] | None = None

    def offload_to_cpu(self) -> None:
        """Offload training model and optimizer to CPU using pinned memory for faster transfers."""
        if self._is_offloaded:
            return

        # Initialize pinned buffer storage
        if self._pinned_buffers is None:
            self._pinned_buffers = {}

        # Offload model parameters to pinned memory for faster reload
        for name, param in self.peft_model.named_parameters():
            if param.device.type == "cuda":
                # Create pinned buffer if not exists or wrong size
                if (
                    name not in self._pinned_buffers
                    or self._pinned_buffers[name].shape != param.shape
                ):
                    self._pinned_buffers[name] = torch.empty(
                        param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                    )
                # Async copy to pinned memory
                self._pinned_buffers[name].copy_(param.data, non_blocking=True)
                param.data = self._pinned_buffers[name]

        # Offload optimizer state to pinned memory
        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for param_id, state in optimizer.state.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                        key = f"opt_{id(param_id)}_{k}"
                        if (
                            key not in self._pinned_buffers
                            or self._pinned_buffers[key].shape != v.shape
                        ):
                            self._pinned_buffers[key] = torch.empty(
                                v.shape, dtype=v.dtype, device="cpu", pin_memory=True
                            )
                        self._pinned_buffers[key].copy_(v, non_blocking=True)
                        state[k] = self._pinned_buffers[key]

        # Sync to ensure all copies are complete before freeing GPU memory
        torch.cuda.synchronize()

        self._is_offloaded = True
        gc_and_empty_cuda_cache()

    def reload_to_gpu(self, device: str = "cuda:0") -> None:
        """Reload training model and optimizer back to GPU using async transfers."""
        if not self._is_offloaded:
            return

        # Reload model parameters from pinned memory (fast async transfer)
        for name, param in self.peft_model.named_parameters():
            if param.device.type == "cpu":
                # Allocate on GPU and async copy from pinned memory
                gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                gpu_tensor.copy_(param.data, non_blocking=True)
                param.data = gpu_tensor

        # Reload optimizer state
        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                        gpu_tensor = torch.empty(v.shape, dtype=v.dtype, device=device)
                        gpu_tensor.copy_(v, non_blocking=True)
                        state[k] = gpu_tensor

        # Sync to ensure all copies are complete before training
        torch.cuda.synchronize()

        self._is_offloaded = False


# ============================================================================
# Service
# ============================================================================


@dataclass
class UnslothService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1  # Start from 1 since 0 is reserved

    def _next_lora_id(self) -> int:
        """Return a new unique LoRA ID to avoid collisions in vLLM."""
        self._lora_id_counter += 1
        return self._lora_id_counter

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            # Create initial LoRA checkpoint if none exists
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self._state.trainer.save_model(lora_path)
            self._latest_step = 0
        else:
            # Extract step from checkpoint path
            self._latest_step = get_step_from_dir(self.output_dir)

        # Offload training model to CPU before vLLM starts to free GPU memory
        self._state.offload_to_cpu()

        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path,
            config=config,
        )
        await openai_server_task(
            engine=await self.llm,
            config=server_config,
        )
        return server_config.get("server_args", {}).get(
            "host"
        ) or "0.0.0.0", server_config.get("server_args", {}).get("port", 8000)

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        """Register a LoRA adapter for a specific checkpoint step.

        This is called when training is skipped but the checkpoint is renamed.
        """
        llm = await self.llm
        await llm.pause_generation()
        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{step}",
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(
                f"Failed to add LoRA adapter for step {step} at {checkpoint_dir}"
            )
        self._latest_step = step
        await llm.resume_generation()

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        llm = await self.llm

        # Pause generation to prevent new requests during training
        await llm.pause_generation()

        # Determine sleep level based on outstanding requests:
        # - level 1: offload KV cache to CPU (can resume with existing KV state)
        # - level 2: discard KV cache (fresh start after wake)
        has_unfinished = llm.output_processor.has_unfinished_requests()
        if has_unfinished:
            sleep_level = 1
        else:
            # Reset prefix cache before discarding KV cache
            await llm.reset_prefix_cache()
            sleep_level = 2

        # Put workers to sleep
        await run_on_workers(llm, do_sleep, level=sleep_level)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        # Reload training model to GPU (after vLLM is asleep)
        self._state.reload_to_gpu()

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

        # Train on the batch using shared logic
        async for result in process_train_batch(
            packed_tensors=packed_tensors,
            config=config,
            _config=_config,
            inputs_queue=self._state.inputs_queue,
            results_queue=self._state.results_queue,
            train_task=self._train_task,
            trainer=self._state.trainer,
            peft_model=self._state.peft_model,
            warmup=warmup,
            verbose=verbose,
        ):
            yield result

        # Save checkpoint after training
        checkpoint_dir = save_checkpoint(
            trainer=self._state.trainer,
            output_dir=self.output_dir,
            verbose=verbose,
        )

        # Offload training model to CPU before waking vLLM
        self._state.offload_to_cpu()

        # Free memory before waking up vLLM
        gc_and_empty_cuda_cache()
        await asyncio.sleep(
            0.5
        )  # Longer delay to allow memory cleanup and pending ops to complete

        # Wake up workers
        await run_on_workers(llm, do_wake_up)
        self._is_sleeping = False

        # Determine the new step from the checkpoint directory
        # checkpoint_dir format is: {output_dir}/checkpoints/{step:04d}
        new_step = int(os.path.basename(checkpoint_dir))

        # Add the new LoRA adapter
        # We keep old LoRAs loaded - vLLM will page them out as needed
        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{new_step}",
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(
                f"Failed to add LoRA adapter for step {new_step} at {checkpoint_dir}"
            )
        self._latest_step = new_step

        # Resume generation after LoRA add is complete
        await llm.resume_generation()

        if verbose:
            print("UnslothService.train complete")

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

        # Initialize PEFT model - skip if already a PeftModel (e.g. loaded from checkpoint)
        if (
            hasattr(model, "peft_config")
            and getattr(model, "peft_config", None) is not None
        ):
            # Model already has LoRA adapters (loaded from checkpoint)
            peft_model = cast(peft.peft_model.PeftModelForCausalLM, model)
        else:
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
            args=GRPOConfig(**self.config.get("trainer_args", {})),
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
        # Filter engine args to remove incompatible boolean flags
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
        }
        # Remove boolean flags that vLLM's argparse doesn't accept as =False
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))  # ty:ignore[invalid-argument-type]


# ============================================================================
# Worker Sleep/Wake Functions
# ============================================================================


def do_sleep(*, level: int) -> None:
    """
    Put the worker to sleep, offloading both weights and KV cache.

    Args:
        level: The sleep level:
            - 1: offload KV cache to CPU (can resume with existing KV state)
            - 2: discard KV cache (fresh start after wake)
    """
    import ctypes
    import gc

    import torch
    from vllm.device_allocator.cumem import (
        CuMemAllocator,
        libcudart,
        unmap_and_release,
    )

    try:
        from vllm.utils.platform_utils import is_pin_memory_available
    except ImportError:
        from vllm.utils import is_pin_memory_available

    worker = get_worker()
    allocator = CuMemAllocator.get_instance()

    # Determine what to offload based on level:
    # level=1: offload both weights and kv_cache to CPU
    # level=2: offload weights, discard kv_cache
    offload_to = "cpu" if level == 1 else "none"
    tags_to_process = {"weights", "kv_cache"}

    # Save buffers before level 2 sleep (like vLLM does)
    if level == 2:
        model = worker.model_runner.model
        worker._sleep_saved_buffers = {
            name: buffer.cpu().clone() for name, buffer in model.named_buffers()
        }

    for ptr, data in allocator.pointer_to_data.items():
        if data.tag not in tags_to_process:
            continue
        handle = data.handle
        size_in_bytes = handle[1]

        # Always backup weights; backup kv_cache only at level 1
        if offload_to != "none" or data.tag == "weights":
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(  # ty:ignore[possibly-missing-attribute]
                ctypes.c_void_p(cpu_ptr), ctypes.c_void_p(ptr), size_in_bytes
            )
            data.cpu_backup_tensor = cpu_backup_tensor

        unmap_and_release(handle)

    gc.collect()
    torch.cuda.empty_cache()


def do_wake_up() -> None:
    """
    Wake up the worker from sleep, restoring offloaded weights and KV cache.
    """
    import ctypes

    from vllm.device_allocator.cumem import (
        CuMemAllocator,
        create_and_map,
        libcudart,
    )

    worker = get_worker()
    allocator = CuMemAllocator.get_instance()

    tags_to_process = {"weights", "kv_cache"}

    for ptr, data in allocator.pointer_to_data.items():
        if data.tag not in tags_to_process:
            continue
        create_and_map(data.handle)
        if data.cpu_backup_tensor is not None:
            cpu_backup_tensor = data.cpu_backup_tensor
            size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(  # ty:ignore[possibly-missing-attribute]
                ctypes.c_void_p(ptr),
                ctypes.c_void_p(cpu_ptr),
                size_in_bytes,
            )
            data.cpu_backup_tensor = None

    # Restore buffers after level 2 sleep (like vLLM does)
    if hasattr(worker, "_sleep_saved_buffers") and worker._sleep_saved_buffers:
        model = worker.model_runner.model
        for name, buffer in model.named_buffers():
            if name in worker._sleep_saved_buffers:
                buffer.copy_(worker._sleep_saved_buffers[name].to(buffer.device))
        worker._sleep_saved_buffers = {}
