"""SGLang-based backend for ART with Multi-GPU Split architecture.

This module provides an alternative backend that uses SGLang for inference
instead of vLLM. The key advantage is RadixAttention prefix caching which
significantly improves performance for multi-turn agent trajectories.

Architecture (Multi-GPU Split):
    GPU 0: SGLang inference server (persistent, preserves RadixAttention cache)
    GPU 1+: Training with Unsloth/GRPO

    This separation means:
    - No memory release/reclaim overhead between train/inference
    - RadixAttention cache stays warm across training steps
    - Weight sync via hot-reload API (no server restart)

IMPORTANT: SGLang and vLLM have conflicting dependencies (different PyTorch
versions). Use SEPARATE virtual environments:

    # For vLLM (default)
    pip install openpipe-art[backend]

    # For SGLang (separate environment)
    pip install openpipe-art[sglang]

Usage:
    from art.sglang_backend import SGLangBackend

    # Multi-GPU (recommended, requires 2+ GPUs)
    backend = SGLangBackend(
        inference_device=0,      # SGLang on GPU 0
        training_devices=[1],    # Training on GPU 1
    )

    # Single-GPU fallback (uses restart mode, slower)
    backend = SGLangBackend()  # Auto-detects single GPU

    await backend.register(model)
    result = await backend.train(model, trajectory_groups)

References:
    - verl SGLang integration: https://verl.readthedocs.io/en/latest/workers/sglang_worker.html
    - SGLang weight sync: https://hebiao064.github.io/rl-weight-sync
    - slime framework: https://github.com/Tsinghua-MARS-Lab/Slime
"""

from .backend import SGLangBackend
from .config import SGLangConfig, DeviceConfig

__all__ = [
    "SGLangBackend",
    "SGLangConfig",
    "DeviceConfig",
]
