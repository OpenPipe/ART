"""SGLang service for inference with Unsloth training.

This service manages the SGLang inference server and training lifecycle.
In multi-GPU mode, the server stays running and weights are hot-reloaded.
In single-GPU mode, the server is restarted for each training step.

Key features:
- Persistent SGLang server preserves RadixAttention cache
- Hot-reload LoRA weights via SGLang API (no restart needed)
- Automatic fallback to restart mode on single GPU
- Health monitoring and graceful shutdown
"""

import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, cast

import aiohttp
import torch
from datasets import Dataset
import peft
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.inputs import TrainInputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..unsloth.train import gc_and_empty_cuda_cache, train

from .config import DeviceConfig, SGLangConfig

if TYPE_CHECKING:
    from peft.peft_model import PeftModelForCausalLM


# Type alias for Unsloth model
CausalLM = Any


@dataclass
class TrainingState:
    """Container for training model state."""
    
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: "PeftModelForCausalLM"
    trainer: "GRPOTrainer"
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]
    _pinned_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    _is_offloaded: bool = False

    def offload_to_cpu(self) -> None:
        """Offload training model to CPU to free GPU memory."""
        if self._is_offloaded:
            return

        for name, param in self.peft_model.named_parameters():
            if param.device.type == "cuda":
                if (
                    name not in self._pinned_buffers
                    or self._pinned_buffers[name].shape != param.shape
                ):
                    self._pinned_buffers[name] = torch.empty(
                        param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                    )
                self._pinned_buffers[name].copy_(param.data, non_blocking=True)
                param.data = self._pinned_buffers[name]

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

        torch.cuda.synchronize()
        self._is_offloaded = True
        gc_and_empty_cuda_cache()

    def reload_to_gpu(self, device: str = "cuda:0") -> None:
        """Reload training model and optimizer back to GPU."""
        if not self._is_offloaded:
            return

        for name, param in self.peft_model.named_parameters():
            if param.device.type == "cpu":
                gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                gpu_tensor.copy_(param.data, non_blocking=True)
                param.data = gpu_tensor

        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                        gpu_tensor = torch.empty(v.shape, dtype=v.dtype, device=device)
                        gpu_tensor.copy_(v, non_blocking=True)
                        state[k] = gpu_tensor

        torch.cuda.synchronize()
        self._is_offloaded = False


@dataclass
class SGLangService:
    """Service using SGLang for inference and Unsloth for training.
    
    This implements the ModelService protocol while using SGLang
    instead of vLLM for the inference server.
    
    Multi-GPU Mode (recommended):
        - SGLang server runs persistently on inference_device
        - Training runs on training_devices
        - Weights hot-reloaded via API after each training step
        - RadixAttention cache preserved across training
    
    Single-GPU Mode (fallback):
        - SGLang server killed before training
        - Server restarted after training with new LoRA
        - Cache lost on each restart
    """
    
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    device_config: DeviceConfig
    sglang_config: SGLangConfig
    
    _is_sleeping: bool = False
    _latest_step: int = 0
    _server_process: subprocess.Popen | None = None
    _server_port: int = 8000
    _server_host: str = "127.0.0.1"
    _train_task: asyncio.Task | None = None
    _lora_counter: int = 1

    def _next_lora_id(self) -> int:
        """Generate unique LoRA ID."""
        self._lora_counter += 1
        return self._lora_counter

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        """Start SGLang OpenAI-compatible server.
        
        In multi-GPU mode, training model stays on training GPUs.
        In single-GPU mode, training model is offloaded to CPU first.
        """
        # Get or create initial LoRA checkpoint
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self._training_state.trainer.save_model(lora_path)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)

        # In single-GPU mode, offload training model before starting SGLang
        if not self.device_config.is_split_mode:
            self._training_state.offload_to_cpu()
            gc_and_empty_cuda_cache()  # Ensure GPU memory is freed for SGLang

        # Get server configuration
        server_config = config or {}
        server_args = server_config.get("server_args", {})
        
        self._server_host = server_args.get("host", "127.0.0.1")
        self._server_port = server_args.get("port", 8000)
        
        # Create logs directory
        log_dir = f"{self.output_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Start SGLang server subprocess
        await self._start_server_process(lora_path)
        
        return self._server_host, self._server_port

    async def _start_server_process(self, lora_path: str | None = None) -> None:
        """Start SGLang server as subprocess with proper device isolation.
        
        Uses a separate Python environment if sglang_python_path is configured.
        This allows SGLang (torchao==0.9.0) and unsloth (torchao>=0.13.0) to coexist.
        """
        # Build environment with device isolation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device_config.inference_cuda_devices
        env.update(self.sglang_config.to_env_vars())
        
        # Get Python executable for SGLang server (may be different venv)
        server_python = self.sglang_config.get_server_python()
        
        # Build server command
        cmd = [
            server_python, "-m", "sglang.launch_server",
            "--model-path", self.base_model,
            "--host", self._server_host,
            "--port", str(self._server_port),
            "--mem-fraction-static", str(self.sglang_config.mem_fraction_static),
            "--log-level", self.sglang_config.log_level,
            "--enable-lora",  # Enable LoRA hot-reload endpoint
        ]
        
        # Add tensor parallelism if configured
        if self.sglang_config.tensor_parallel_size > 1:
            cmd.extend(["--tp-size", str(self.sglang_config.tensor_parallel_size)])
        
        # Add context length if specified
        if self.sglang_config.context_length:
            cmd.extend(["--context-length", str(self.sglang_config.context_length)])
        
        # Add LoRA configuration
        if lora_path and os.path.exists(lora_path):
            cmd.extend(["--lora-paths", lora_path])
            cmd.extend(["--max-loras-per-batch", str(self.sglang_config.max_loras_per_batch)])
        
        # Disable radix cache only if explicitly requested (not recommended)
        if self.sglang_config.disable_radix_cache:
            cmd.append("--disable-radix-cache")
        
        # Start server
        log_file = open(f"{self.output_dir}/logs/sglang.log", "a")
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
        )
        
        # Wait for server to be ready
        await self._wait_for_server()

    async def _wait_for_server(self) -> None:
        """Wait for SGLang server to be ready."""
        timeout = self.sglang_config.server_timeout
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server process died with code {self._server_process.returncode}. "
                    f"Check logs at {self.output_dir}/logs/sglang.log"
                )
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self._server_host}:{self._server_port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        
        raise TimeoutError(
            f"SGLang server did not start within {timeout} seconds. "
            f"Check logs at {self.output_dir}/logs/sglang.log"
        )

    async def _stop_server_process(self) -> None:
        """Stop SGLang server subprocess gracefully."""
        if self._server_process is None:
            return
        
        try:
            # Force kill immediately for fast cleanup
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self._server_process.kill()
            
            # Non-blocking wait with short timeout
            for _ in range(10):  # Max 1 second
                if self._server_process.poll() is not None:
                    break
                await asyncio.sleep(0.1)
        except Exception:
            pass  # Best effort cleanup
        finally:
            self._server_process = None
        
        self._server_process = None
        gc_and_empty_cuda_cache()

    async def _hot_reload_lora(self, checkpoint_dir: str, step: int) -> None:
        """Hot-reload LoRA weights without restarting server.
        
        Uses SGLang's update_weights_from_lora API.
        This preserves the RadixAttention cache.
        """
        lora_name = f"{self.model_name}@{step}"
        
        # Call SGLang's LoRA update endpoint
        async with aiohttp.ClientSession() as session:
            payload = {
                "lora_path": checkpoint_dir,
                "lora_name": lora_name,
            }
            
            if self.sglang_config.flush_cache_on_sync:
                payload["flush_cache"] = True
            
            try:
                async with session.post(
                    f"http://{self._server_host}:{self._server_port}/load_lora_adapter",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Failed to hot-reload LoRA: {error_text}")
            except aiohttp.ClientError as e:
                # Fallback: try add_lora endpoint (older SGLang versions)
                try:
                    async with session.post(
                        f"http://{self._server_host}:{self._server_port}/add_lora",
                        json={
                            "lora_path": checkpoint_dir,
                            "lora_name": lora_name,
                            "lora_int_id": self._next_lora_id(),
                        },
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"Failed to add LoRA: {await resp.text()}")
                except Exception:
                    raise RuntimeError(f"Failed to hot-reload LoRA: {e}") from e

    async def vllm_engine_is_sleeping(self) -> bool:
        """Check if engine is sleeping (for LocalBackend compatibility).
        
        In multi-GPU mode, server never sleeps.
        In single-GPU mode, returns True during training.
        """
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Run training step.
        
        Multi-GPU mode:
            1. Training runs on training_devices (server keeps running)
            2. Save LoRA checkpoint
            3. Hot-reload weights via API
        
        Single-GPU mode:
            1. Stop SGLang server
            2. Reload training model to GPU
            3. Train
            4. Save checkpoint
            5. Restart server with new LoRA
        """
        if self.device_config.is_split_mode:
            # Multi-GPU: server stays running
            async for metrics in self._train_split_mode(
                disk_packed_tensors, config, _config, verbose
            ):
                yield metrics
        else:
            # Single-GPU: need to swap
            async for metrics in self._train_shared_mode(
                disk_packed_tensors, config, _config, verbose
            ):
                yield metrics

    async def _train_split_mode(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Training in multi-GPU split mode.
        
        Server keeps running. Weights hot-reloaded after training.
        """
        # Training device is cuda:0 after CUDA_VISIBLE_DEVICES is set in _training_state
        # (e.g., if training GPUs are [1,2,3], GPU 1 becomes cuda:0 after setting CUDA_VISIBLE_DEVICES="1,2,3")
        training_device = "cuda:0"
        
        # Ensure training model is on GPU
        self._training_state.reload_to_gpu(training_device)

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for any pending batches
        await self._training_state.results_queue.join()

        # Start training task if needed
        if self._train_task is None:
            self._train_task = asyncio.create_task(
                train(
                    trainer=self._training_state.trainer,
                    results_queue=self._training_state.results_queue,
                )
            )
            warmup = True
        else:
            warmup = False

        # Process training batch
        from ..unsloth.training_utils import process_train_batch
        
        async for result in process_train_batch(
            packed_tensors=packed_tensors,
            config=config,
            _config=_config,
            inputs_queue=self._training_state.inputs_queue,
            results_queue=self._training_state.results_queue,
            train_task=self._train_task,
            trainer=self._training_state.trainer,
            peft_model=self._training_state.peft_model,
            warmup=warmup,
            verbose=verbose,
        ):
            yield result

        # Save checkpoint
        from ..unsloth.training_utils import save_checkpoint
        
        checkpoint_dir = save_checkpoint(
            trainer=self._training_state.trainer,
            output_dir=self.output_dir,
            verbose=verbose,
        )

        # Determine new step
        new_step = int(os.path.basename(checkpoint_dir))
        
        # Hot-reload LoRA weights (no server restart!)
        if self.sglang_config.weight_sync_method == "lora":
            await self._hot_reload_lora(checkpoint_dir, new_step)
        elif self.sglang_config.weight_sync_method == "disk":
            await self._reload_from_disk(checkpoint_dir)
        else:
            # Fallback: restart server
            await self._stop_server_process()
            await self._start_server_process(checkpoint_dir)
        
        self._latest_step = new_step

        if verbose:
            print(f"SGLangService.train complete (split mode, step {new_step})")

    async def _train_shared_mode(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Training in single-GPU shared mode.
        
        Server is stopped during training, restarted after.
        """
        # Stop SGLang server to free GPU memory
        await self._stop_server_process()
        self._is_sleeping = True
        gc_and_empty_cuda_cache()
        
        # Reload training model to GPU
        self._training_state.reload_to_gpu("cuda:0")

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for pending batches
        await self._training_state.results_queue.join()

        # Start training task if needed
        if self._train_task is None:
            self._train_task = asyncio.create_task(
                train(
                    trainer=self._training_state.trainer,
                    results_queue=self._training_state.results_queue,
                )
            )
            warmup = True
        else:
            warmup = False

        # Process training batch
        from ..unsloth.training_utils import process_train_batch
        
        async for result in process_train_batch(
            packed_tensors=packed_tensors,
            config=config,
            _config=_config,
            inputs_queue=self._training_state.inputs_queue,
            results_queue=self._training_state.results_queue,
            train_task=self._train_task,
            trainer=self._training_state.trainer,
            peft_model=self._training_state.peft_model,
            warmup=warmup,
            verbose=verbose,
        ):
            yield result

        # Save checkpoint
        from ..unsloth.training_utils import save_checkpoint
        
        checkpoint_dir = save_checkpoint(
            trainer=self._training_state.trainer,
            output_dir=self.output_dir,
            verbose=verbose,
        )

        # Offload training model
        self._training_state.offload_to_cpu()
        gc_and_empty_cuda_cache()

        # Restart SGLang server with new LoRA
        new_step = int(os.path.basename(checkpoint_dir))
        await self._start_server_process(checkpoint_dir)
        
        self._latest_step = new_step
        self._is_sleeping = False

        if verbose:
            print(f"SGLangService.train complete (shared mode, step {new_step})")

    async def _reload_from_disk(self, checkpoint_dir: str) -> None:
        """Reload weights from disk (alternative to LoRA hot-reload)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{self._server_host}:{self._server_port}/update_weights_from_disk",
                json={
                    "model_path": checkpoint_dir,
                    "load_format": "auto",
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to reload weights: {await resp.text()}")

    async def shutdown(self) -> None:
        """Clean shutdown of service."""
        await self._stop_server_process()
        
        if self._train_task:
            self._train_task.cancel()
            try:
                await self._train_task
            except asyncio.CancelledError:
                pass
            self._train_task = None

    @cached_property
    def _training_state(self) -> TrainingState:
        """Initialize Unsloth model and trainer on training device."""
        import unsloth

        # Set training device with proper GPU isolation
        if self.device_config.is_split_mode:
            # CRITICAL: Set CUDA_VISIBLE_DEVICES to training GPUs only
            # This ensures training doesn't accidentally use the inference GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_config.training_cuda_devices
            device = "cuda:0"  # After CUDA_VISIBLE_DEVICES, GPU 0 is the first training GPU
            torch.cuda.set_device(0)
        else:
            device = "cuda:0"

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

        if (
            hasattr(model, "peft_config")
            and getattr(model, "peft_config", None) is not None
        ):
            peft_model = cast(peft.peft_model.PeftModelForCausalLM, model)
        else:
            peft_model = cast(
                peft.peft_model.PeftModelForCausalLM,
                unsloth.FastLanguageModel.get_peft_model(
                    model, **self.config.get("peft_args", {})
                ),
            )

        data = {"prompt": ""}
        trainer = GRPOTrainer(
            model=peft_model,
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=tokenizer,
        )

        inputs_queue: asyncio.Queue[TrainInputs] = asyncio.Queue()
        results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

        def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
            async def get_inputs() -> TrainInputs:
                return await inputs_queue.get()
            inputs = asyncio.run(get_inputs())
            return cast(dict[str, torch.Tensor], inputs)

        trainer._prepare_inputs = _async_prepare_inputs

        return TrainingState(
            model=model,
            tokenizer=tokenizer,
            peft_model=peft_model,
            trainer=trainer,
            inputs_queue=inputs_queue,
            results_queue=results_queue,
        )
