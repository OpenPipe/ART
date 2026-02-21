"""SGLang-based backend for ART.

This module provides SGLangBackend, an alternative to LocalBackend that uses
SGLang for inference instead of vLLM. Training remains the same (Unsloth/GRPO).

Architecture:
    Multi-GPU (recommended):
        GPU 0: SGLang server (persistent, RadixAttention cache preserved)
        GPU 1+: Training (Unsloth/GRPO)
        Weight sync: Hot-reload via API (no restart)
    
    Single-GPU (fallback):
        GPU 0: Shared between SGLang and training
        Weight sync: Server restart (cache lost)

Benefits over vLLM:
    - RadixAttention: Better prefix caching for multi-turn agent trajectories
    - Zero-overhead scheduler: Lower latency for RL rollouts
    - Faster structured outputs: Better tool call parsing

Limitations:
    - No Tinker support yet
    - Requires separate environment from vLLM (dependency conflicts)
    - Multi-GPU recommended for best performance
"""

import asyncio
import os
import subprocess

from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..utils.output_dirs import get_model_dir

from .config import DeviceConfig, SGLangConfig
from .service import SGLangService


class SGLangBackend(LocalBackend):
    """Backend using SGLang for inference instead of vLLM.
    
    This is a drop-in replacement for LocalBackend with SGLang-specific
    optimizations for RL training workloads.
    
    Args:
        inference_device: GPU index for SGLang server (default: 0)
        training_devices: GPU indices for training (default: auto-detect)
        in_process: Run service in-process (default: False)
        path: Path for checkpoints/logs (default: ".art")
        sglang_config: SGLang-specific configuration
    
    Example:
        # Multi-GPU setup (recommended)
        backend = SGLangBackend(
            inference_device=0,
            training_devices=[1, 2],
        )
        
        # Single-GPU (auto-fallback)
        backend = SGLangBackend()
        
        # With custom config
        backend = SGLangBackend(
            sglang_config=SGLangConfig(
                mem_fraction_static=0.85,
                weight_sync_method="lora",
            )
        )
        
        await backend.register(model)
        result = await backend.train(model, trajectory_groups)
    """
    
    def __init__(
        self,
        *,
        inference_device: int | None = None,
        training_devices: list[int] | None = None,
        in_process: bool = False,
        path: str | None = None,
        sglang_config: SGLangConfig | None = None,
    ) -> None:
        """Initialize SGLangBackend.
        
        Args:
            inference_device: GPU for SGLang (None = auto-detect)
            training_devices: GPUs for training (None = auto-detect)
            in_process: Run in-process (mainly for debugging)
            path: Checkpoint/log directory
            sglang_config: SGLang server configuration
        """
        # Validate SGLang is available
        self._validate_sglang_installation()
        
        # Initialize device configuration
        if inference_device is not None or training_devices is not None:
            self._device_config = DeviceConfig(
                inference_device=inference_device or 0,
                training_devices=training_devices or [1],
                auto_detect=False,
            )
        else:
            self._device_config = DeviceConfig(auto_detect=True)
        
        # SGLang configuration
        self._sglang_config = sglang_config or SGLangConfig()
        
        # In single-GPU mode, always use restart for weight sync
        if not self._device_config.is_split_mode:
            if self._sglang_config.weight_sync_method != "restart":
                print(
                    f"Note: Single-GPU mode detected. Using 'restart' weight sync "
                    f"instead of '{self._sglang_config.weight_sync_method}'. "
                    f"For better performance, use 2+ GPUs."
                )
                self._sglang_config.weight_sync_method = "restart"
        
        # Initialize parent
        super().__init__(in_process=in_process, path=path)
        
        # Log configuration
        self._log_config()
    
    def _validate_sglang_installation(self) -> None:
        """Check that SGLang server environment is available.
        
        SGLang can run in a separate venv to avoid torchao conflicts with unsloth.
        This checks if the configured server Python has sglang installed.
        """
        pass  # Validation happens when server starts (in the server's Python)
    
    def _log_config(self) -> None:
        """Log configuration for debugging."""
        mode = "split" if self._device_config.is_split_mode else "shared"
        print(f"SGLangBackend initialized:")
        print(f"  Mode: {mode}-GPU")
        print(f"  Inference device: cuda:{self._device_config.inference_device}")
        print(f"  Training devices: cuda:{self._device_config.training_devices}")
        print(f"  Weight sync: {self._sglang_config.weight_sync_method}")
        if self._device_config.is_split_mode:
            print(f"  RadixAttention cache: preserved across training")
        else:
            print(f"  RadixAttention cache: cleared on each training step")

    async def _get_service(self, model: TrainableModel) -> ModelService:
        """Get or create the SGLang-based model service.
        
        Overrides LocalBackend._get_service to use SGLangService.
        """
        from ..dev.get_model_config import get_model_config

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            
            # Check for tinker config
            if config.get("tinker_args") is not None:
                raise NotImplementedError(
                    "SGLangBackend does not support tinker models yet. "
                    "Use LocalBackend for tinker models."
                )
            
            # Create SGLang service
            service = SGLangService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
                device_config=self._device_config,
                sglang_config=self._sglang_config,
            )
            
            self._services[model.name] = service
            
            if not self._in_process:
                # Kill any existing SGLang processes
                subprocess.run(
                    ["pkill", "-9", "-f", "sglang.launch_server"],
                    capture_output=True,
                )
        
        return self._services[model.name]

    async def _monitor_openai_server(
        self, model_name: str, base_url: str, api_key: str
    ) -> None:
        """Monitor the SGLang OpenAI-compatible server.
        
        SGLang uses different metrics, so we use simpler health checks.
        """
        import aiohttp
        from openai import AsyncOpenAI
        
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        try:
            async with aiohttp.ClientSession() as session:
                while not getattr(self, '_monitor_should_stop', False):
                    # Sleep in small increments to allow fast shutdown
                    for _ in range(int(self._sglang_config.health_check_interval)):
                        if getattr(self, '_monitor_should_stop', False):
                            return
                        await asyncio.sleep(1)
                    
                    # Check stop flag after sleep
                    if getattr(self, '_monitor_should_stop', False):
                        return
                    
                    try:
                        # Check if service is sleeping (single-GPU mode during training)
                        service = self._services.get(model_name)
                        if service and await service.vllm_engine_is_sleeping():
                            consecutive_failures = 0
                            continue
                        
                        # Health check via models endpoint
                        async with session.get(
                            f"{base_url.replace('/v1', '')}/v1/models",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status == 200:
                                consecutive_failures = 0
                                continue
                        
                        # Fallback: try completion
                        await openai_client.completions.create(
                            model=model_name,
                            prompt="Hi",
                            max_tokens=1,
                            timeout=5.0,
                        )
                        consecutive_failures = 0
                        
                    except Exception:
                        # Check stop flag - don't error during shutdown
                        if getattr(self, '_monitor_should_stop', False):
                            return
                        
                        # Check sleep status during exception
                        try:
                            service = self._services.get(model_name)
                            if service and await service.vllm_engine_is_sleeping():
                                consecutive_failures = 0
                                continue
                        except Exception:
                            pass
                        
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            raise
        except asyncio.CancelledError:
            # Graceful shutdown
            return
        except aiohttp.ClientError:
            # Connection errors during shutdown are expected
            if getattr(self, '_monitor_should_stop', False):
                return
            raise

    async def close(self) -> None:
        """Clean up resources and shutdown SGLang servers."""
        # Signal monitor to stop
        self._monitor_should_stop = True
        
        # Brief pause for monitor to notice stop flag
        await asyncio.sleep(0.1)
        
        # Shutdown all SGLang services
        for name, service in list(self._services.items()):
            if isinstance(service, SGLangService):
                await service.shutdown()
        
        # Call parent close
        await super().close()

    @property
    def device_config(self) -> DeviceConfig:
        """Get device configuration."""
        return self._device_config
    
    @property
    def sglang_config(self) -> SGLangConfig:
        """Get SGLang configuration."""
        return self._sglang_config
