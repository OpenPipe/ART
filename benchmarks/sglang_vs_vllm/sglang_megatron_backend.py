"""
SGLang + Megatron backend — verl-style hybrid engine.

Uses SGLang for inference (persistent, never restarts) and Megatron for training.
Inherits all training/checkpoint logic from LocalBackend.

Architecture matches verl-project/verl:
  - SGLang server starts once, stays alive across all RL steps
  - Weight sync via CUDA IPC (update_weights_from_tensor), not restart
  - Memory managed via sleep/wake (release/resume_memory_occupation)
  - Native generation returns actual token IDs, not SSE chunks
"""

from __future__ import annotations

import os

from art.local.backend import LocalBackend
from art.local.service import ModelService
from art.model import Model, TrainableModel
from art.utils.output_dirs import get_model_dir

from .sglang_megatron_service import SGLangMegatronService


class SGLangMegatronBackend(LocalBackend):
    """Backend: SGLang inference + Megatron training (verl-style).

    Key difference from old implementation:
      - SGLang server NEVER restarts between training steps
      - Weights are synced in-place via CUDA IPC
      - KV cache is managed via sleep/wake, not stop/start
    """

    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        sglang_python: str = "python",
        port: int = 8200,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.7,
        max_running_requests: int = 256,
    ) -> None:
        super().__init__(in_process=in_process, path=path)
        self._sglang_python = sglang_python
        self._port = port
        self._tp = tensor_parallel_size
        self._gpu_mem = gpu_memory_utilization
        self._max_reqs = max_running_requests

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        """Return LoRA adapter name after hot-reload, base model before.

        Before any training step, SGLang serves under the base model name
        (e.g. "Qwen/Qwen3-30B-A3B-Instruct-2507"). After hot-reload, it
        serves the LoRA adapter under the name set by _hot_reload_lora
        (e.g. "bench-sglang@step1"). We must return the correct name so
        inference requests hit the LoRA-augmented model, not naked base.
        """
        service = self._services.get(model.name)
        if service is not None and getattr(service, "_active_lora_name", None):
            return service._active_lora_name
        return model.base_model if hasattr(model, "base_model") else model.name

    async def _get_service(self, model: TrainableModel) -> ModelService:
        if model.name not in self._services:
            output_dir = get_model_dir(model=model, art_path=self._path)

            service = SGLangMegatronService(
                model_name=model.name,
                base_model=model.base_model,
                output_dir=output_dir,
                sglang_python=self._sglang_python,
                port=self._port,
                tensor_parallel_size=self._tp,
                gpu_memory_utilization=self._gpu_mem,
                max_running_requests=self._max_reqs,
            )
            self._services[model.name] = service  # type: ignore[assignment]

        return self._services[model.name]

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config=None,
    ) -> tuple[str, str]:
        service = await self._get_service(model)
        assert isinstance(service, SGLangMegatronService)

        host, port = await service.start_openai_server(config)
        base_url = f"http://{host}:{port}/v1"
        api_key = "sglang-benchmark"
        return base_url, api_key
