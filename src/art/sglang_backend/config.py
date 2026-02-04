"""Configuration classes for SGLang backend.

These configurations control device placement, memory allocation,
and weight synchronization behavior.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DeviceConfig:
    """GPU device assignment configuration.
    
    For optimal performance, SGLang inference and training should run on
    separate GPUs. This eliminates memory release/reclaim overhead and
    keeps the RadixAttention cache warm.
    
    Attributes:
        inference_device: GPU index for SGLang server (default: 0)
        training_devices: GPU indices for training (default: [1] or [0] if single GPU)
        auto_detect: If True, automatically detect available GPUs
    
    Example:
        # 2-GPU setup
        config = DeviceConfig(inference_device=0, training_devices=[1])
        
        # 4-GPU setup with multi-GPU training
        config = DeviceConfig(inference_device=0, training_devices=[1, 2, 3])
        
        # Single GPU (fallback mode with server restart)
        config = DeviceConfig(inference_device=0, training_devices=[0])
    """
    inference_device: int = 0
    training_devices: list[int] = field(default_factory=lambda: [1])
    auto_detect: bool = True
    
    def __post_init__(self):
        if self.auto_detect:
            self._auto_configure()
    
    def _auto_configure(self):
        """Auto-detect GPU count and configure devices."""
        try:
            import torch
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 1
        
        if gpu_count == 0:
            raise RuntimeError("No CUDA GPUs available. SGLang requires GPU.")
        elif gpu_count == 1:
            # Single GPU: shared mode (will use restart)
            self.inference_device = 0
            self.training_devices = [0]
        else:
            # Multi-GPU: split mode
            self.inference_device = 0
            if not self.training_devices or self.training_devices == [1]:
                self.training_devices = list(range(1, gpu_count))
    
    @property
    def is_split_mode(self) -> bool:
        """True if inference and training use separate GPUs."""
        return self.inference_device not in self.training_devices
    
    @property
    def inference_cuda_devices(self) -> str:
        """CUDA_VISIBLE_DEVICES string for inference subprocess."""
        return str(self.inference_device)
    
    @property
    def training_cuda_devices(self) -> str:
        """CUDA_VISIBLE_DEVICES string for training."""
        return ",".join(str(d) for d in self.training_devices)


@dataclass
class SGLangConfig:
    """SGLang server and weight sync configuration.
    
    Attributes:
        sglang_python_path: Path to Python executable in SGLang server venv.
            SGLang requires torchao==0.9.0 which conflicts with unsloth's torchao>=0.13.0.
            Solution: Run SGLang server in a separate venv with its own dependencies.
            Set this to the path of that venv's Python (e.g., ".venv-sglang-server/bin/python").
            If None, uses sys.executable (same Python, may have dependency conflicts).
        
        mem_fraction_static: GPU memory fraction for SGLang (0.0-1.0)
        disable_radix_cache: If True, disable RadixAttention (NOT recommended)
        max_loras_per_batch: Maximum LoRA adapters to batch
        context_length: Maximum context length (None = model default)
        
        weight_sync_method: How to sync weights after training
            - "lora": Use update_weights_from_lora (recommended)
            - "disk": Use update_weights_from_disk
            - "restart": Restart server (fallback, slow)
        
        flush_cache_on_sync: Clear KV cache when syncing weights
        server_timeout: Seconds to wait for server startup
        health_check_interval: Seconds between health checks
    
    References:
        - verl config: https://verl.readthedocs.io/en/latest/examples/config.html
        - SGLang issues on weight sync: #3726, #4283, #8076
    
    Two-Environment Setup:
        # 1. Create main training env (with unsloth)
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e ".[sglang]"
        
        # 2. Create SGLang server env (separate, with sglang[srt])
        python3 -m venv .venv-sglang-server
        .venv-sglang-server/bin/pip install -e ".[sglang-server]"
        
        # 3. Configure to use server env
        config = SGLangConfig(sglang_python_path=".venv-sglang-server/bin/python")
    """
    # Two-environment architecture: path to SGLang server's Python
    # This allows sglang (torchao==0.9.0) and unsloth (torchao>=0.13.0) to coexist
    sglang_python_path: str | None = None
    
    # Memory configuration
    # NOTE: Set to 0.5 to leave enough GPU memory for training when CUDA_VISIBLE_DEVICES
    # can't be set early enough (before PyTorch initialization)
    mem_fraction_static: float = 0.5
    disable_radix_cache: bool = False  # Keep False for RL training!
    max_loras_per_batch: int = 4
    context_length: int | None = None
    
    # Weight synchronization
    weight_sync_method: Literal["lora", "disk", "restart"] = "lora"
    flush_cache_on_sync: bool = False  # Keep cache warm
    
    # Server configuration
    server_timeout: float = 120.0
    health_check_interval: float = 30.0
    
    # Environment variables (from verl docs)
    disable_tp_memory_check: bool = True  # SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK
    
    # Tensor parallelism (for large models)
    tensor_parallel_size: int = 1
    
    # Logging
    log_level: str = "warning"
    
    def get_server_python(self) -> str:
        """Get Python executable path for SGLang server subprocess.
        
        Auto-detection order:
        1. Explicit sglang_python_path if set
        2. .venv-sglang-server/bin/python if exists
        3. sys.executable (same Python, may have conflicts)
        """
        import os
        import sys
        
        if self.sglang_python_path:
            # Resolve relative paths from current working directory
            path = os.path.abspath(self.sglang_python_path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"SGLang server Python not found at {path}. "
                    f"Create the server venv: python3 -m venv .venv-sglang-server && "
                    f".venv-sglang-server/bin/pip install -e '.[sglang-server]'"
                )
            return path
        
        # Auto-detect: check for .venv-sglang-server in common locations
        search_paths = [
            ".venv-sglang-server/bin/python",  # Same directory
            "../.venv-sglang-server/bin/python",  # Parent directory
        ]
        
        for rel_path in search_paths:
            abs_path = os.path.abspath(rel_path)
            if os.path.exists(abs_path):
                print(f"Auto-detected SGLang server venv: {abs_path}")
                return abs_path
        
        # Fallback to same Python (may have dependency conflicts)
        return sys.executable
    
    def to_server_args(self) -> dict:
        """Convert to SGLang server launch arguments."""
        args = {
            "mem_fraction_static": self.mem_fraction_static,
            "disable_radix_cache": self.disable_radix_cache,
            "tp_size": self.tensor_parallel_size,
            "log_level": self.log_level,
        }
        if self.context_length:
            args["context_length"] = self.context_length
        return args
    
    def to_env_vars(self) -> dict[str, str]:
        """Environment variables to set for SGLang subprocess."""
        env = {}
        if self.disable_tp_memory_check:
            env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "True"
        return env
