"""
SGLang + Megatron service — verl-style hybrid engine.

Architecture:
  - SGLang server starts ONCE and NEVER restarts
  - Before training: sleep(kv_cache+weights) releases GPU memory
  - Training: Megatron runs as SEPARATE subprocess (produces LoRA adapter)
  - After training: wake_up(kv_cache+weights) restores base weights
  - Weight sync: hot-reload LoRA adapter via /load_lora_adapter (<2s)
  - Result: SGLang serves base + LoRA on-the-fly, CUDA graphs intact

Weight sync (ART's recommended weight_sync_method="lora"):
  1. _merge_lora_shards() — combine TP-sharded adapters into one (~2MB)
  2. POST /load_lora_adapter — SGLang loads adapter, applies during inference
  3. Generate with model=lora_name — SGLang uses base + adapter

  vs old approach (464s): build 60GB merged model dir + SGLang reload

Key difference from verl: our Megatron subprocess is separate (not shared
process), so we must release BOTH kv_cache and weights during sleep to give
Megatron enough GPU memory. verl's colocated design uses CUDA IPC (zero-copy).

Reference:
  - src/art/sglang_backend/service.py :: _hot_reload_lora()
  - src/art/sglang_backend/config.py :: weight_sync_method="lora" (recommended)
  - verl/workers/rollout/sglang_rollout/sglang_rollout.py (ServerAdapter)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel
import torch

from .sglang_server import SGLangServer, SGLangServerConfig

logger = logging.getLogger(__name__)


class SGLangMegatronTrainingJob(BaseModel):
    """Job format for Megatron train.py — MUST stay in sync."""
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: dict
    config: dict
    experimental_config: dict


@dataclass
class SGLangMegatronService:
    """verl-style SGLang inference + Megatron training lifecycle.

    Key difference from old implementation:
      OLD: stop SGLang → train → restart SGLang (60-90s overhead)
      NEW: sleep → train → wake → load_lora (ART recommended)

    Key difference from verl:
      verl: training + inference share same process, same GPU memory (CUDA IPC)
      ours: Megatron is a SEPARATE subprocess, needs its own GPU memory
      → we must release weights too, not just KV cache

    Loop:
      1. generate()      — SGLang active, KV cache + weights on GPU
      2. sleep()         — release KV cache AND weights (for Megatron)
      3. Megatron train  — uses freed GPU memory, saves LoRA adapter
      4. wake_up()       — restore base weights + KV cache from CPU
      5. load_lora()     — hot-reload ~2MB adapter (ART recommended method)
    """

    model_name: str
    base_model: str
    output_dir: str
    sglang_python: str = "python"
    port: int = 8200
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.7
    max_running_requests: int = 256
    log_dir: str = ""

    _server: SGLangServer | None = None
    _latest_step: int = 0
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None
    _is_sleeping: bool = False
    _active_lora_name: str | None = None

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Server management — start ONCE, never restart
    # ------------------------------------------------------------------

    def _create_server(self) -> SGLangServer:
        """Create SGLang server with LoRA support for dynamic adapter loading.

        Starts base model with --enable-lora so /load_lora_adapter works.
        No adapters loaded initially — they're hot-reloaded after training.
        Mirrors ART's weight_sync_method="lora" (recommended).
        """
        return SGLangServer(SGLangServerConfig(
            model_path=self.base_model,
            served_model_name=self.base_model,
            port=self.port,
            host="0.0.0.0",
            tensor_parallel_size=self.tensor_parallel_size,
            mem_fraction_static=self.gpu_memory_utilization,
            max_running_requests=self.max_running_requests,
            python_executable=self.sglang_python,
            log_file=os.path.join(self.log_dir, "sglang.log"),
            trust_remote_code=True,
            enable_p2p_check=True,
            chunked_prefill_size=32768,
            enable_memory_saver=True,  # Required for sleep/wake
            # LoRA: required for dynamic /load_lora_adapter after training
            # Modules match src/art/megatron/service.py LoraConfig.target_modules
            enable_lora=True,
            max_lora_rank=8,  # Megatron trains rank=1, headroom for future
        ))

    async def start_openai_server(
        self, config: Any = None
    ) -> tuple[str, int]:
        """Start SGLang server ONCE. It stays alive for the entire benchmark.

        Mirrors verl's SGLangHttpServer.launch_server() which launches
        subprocesses once and keeps them alive across all RL steps.
        """
        self._latest_step = 0

        self._server = self._create_server()
        await self._server.start()

        logger.info(
            f"SGLang ready (verl-style, persistent) — "
            f"serving {self.base_model} on port {self.port}"
        )
        return "0.0.0.0", self.port

    async def vllm_engine_is_sleeping(self) -> bool:
        """Compat: check if inference engine is sleeping."""
        return self._is_sleeping

    # ------------------------------------------------------------------
    # verl-style sleep / wake_up
    # Mirrors: verl/workers/rollout/sglang_rollout/async_sglang_server.py
    # ------------------------------------------------------------------

    async def sleep(self) -> float:
        """Release GPU memory for training — verl's sleep().

        Releases BOTH KV cache AND model weights from GPU. This is critical
        because Megatron runs as a SEPARATE subprocess and needs GPU memory
        for its own copy of model weights, optimizer states, and activations.

        Unlike verl (where training and inference share the same process and
        same GPU memory), our architecture runs them in separate processes on
        the same GPUs. Only releasing KV cache (~35GB) is not enough — SGLang's
        model weights (~7.5GB/GPU for Qwen3-30B-A3B) must also be freed.

        SGLang process stays alive: NCCL communicators, tokenizer survive.
        Weights will be restored via wake_up(), then LoRA loaded via /load_lora_adapter.

        verl equivalent:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await tokenizer_manager.release_memory_occupation(obj, None)
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        # Release both KV cache and weights — Megatron needs the GPU memory
        elapsed = await self._server.sleep(tags=["kv_cache", "weights"])
        self._is_sleeping = True
        logger.info(f"SGLang sleeping (kv_cache + weights released) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    async def wake_up(self) -> float:
        """Resume GPU memory after training — verl's wake_up().

        Restores BOTH KV cache and model weights from CPU backup.
        After wake, _hot_reload_lora() will load the LoRA adapter
        on top of these restored base weights.

        verl equivalent:
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await tokenizer_manager.resume_memory_occupation(obj, None)
            await tokenizer_manager.flush_cache()
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        # Resume both KV cache and weights — LoRA adapter loaded separately after
        elapsed = await self._server.wake_up(tags=["kv_cache", "weights"])
        self._is_sleeping = False
        logger.info(f"SGLang awake (kv_cache + weights resumed) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # verl-style weight sync
    # Mirrors: verl/workers/rollout/sglang_rollout/sglang_rollout.py
    # ------------------------------------------------------------------

    async def _hot_reload_lora(self, lora_path: str, step: int) -> float:
        """Hot-reload LoRA adapter — ART's recommended weight_sync_method.

        Mirrors: src/art/sglang_backend/service.py :: _hot_reload_lora()
        Config:  src/art/sglang_backend/config.py :: weight_sync_method="lora"

        Instead of building a 60GB merged model dir and reloading all weights,
        SGLang loads the ~2MB adapter and applies it on-the-fly during inference.
        Base weights stay UNTOUCHED on GPU after wake_up().

        Old path (464s):
          read 60GB base → compute B@A deltas → write 60GB merged → SGLang reloads 60GB
        New path (<2s):
          POST /load_lora_adapter with 2MB adapter → done
        """
        if self._server is None:
            logger.warning("No server — skipping LoRA hot-reload")
            return 0.0

        adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            logger.warning(f"No adapter at {adapter_file}, skipping")
            return 0.0

        # SGLang's LoRAConfig constructor requires adapter_config.json.
        # Megatron doesn't create it — generate from known LoRA params.
        # Mirrors: src/art/megatron/service.py :: _ensure_lora_adapter_config()
        self._ensure_adapter_config(lora_path)

        lora_name = f"{self.model_name}@step{step}"
        elapsed = await self._server.load_lora_adapter(
            lora_path=lora_path,
            lora_name=lora_name,
            flush_cache=True,
        )

        if elapsed < 0:
            # Fallback: if load_lora_adapter not supported, log and continue
            # Base weights are still correct (just without LoRA update)
            logger.error(
                "load_lora_adapter failed — SGLang may not support dynamic "
                "LoRA loading. Base weights are intact but NOT updated."
            )
            return 0.0

        self._active_lora_name = lora_name
        logger.info(
            f"LoRA hot-reload: '{lora_name}' loaded in {elapsed:.2f}s "
            f"(was 464s with disk merge)"
        )
        return elapsed

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{step:04d}")

    def _get_last_checkpoint_dir(self) -> str | None:
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None
        steps = sorted(
            int(d) for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
        )
        return os.path.join(ckpt_dir, f"{steps[-1]:04d}") if steps else None

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is None:
            self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
            os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    # ------------------------------------------------------------------
    # Megatron process management
    # ------------------------------------------------------------------

    async def _ensure_megatron_running(self) -> None:
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge
            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent.parent.parent / "src" / "art" / "megatron" / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        subprocess.run(["pkill", "-9", "megatron-service"], check=False)

        train_script = Path(__file__).parent.parent.parent / "src" / "art" / "megatron" / "train.py"
        num_gpus = torch.cuda.device_count()
        os.environ["MODEL_IDENTIFIER"] = self.base_model

        command = f"{setup_cmd}uv run torchrun --nproc_per_node {num_gpus} {train_script}"
        self._megatron_process = await asyncio.create_subprocess_shell(command)

    # ------------------------------------------------------------------
    # Training step — verl-style: sleep → train → wake → load_lora
    # ------------------------------------------------------------------

    async def train(
        self,
        disk_packed_tensors: dict,
        config: dict,
        experimental_config: dict,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """verl-style training step: sleep → train → wake → load_lora.

        OLD architecture (464s weight sync):
          sleep → train → wake → build 60GB merged dir → SGLang reloads 60GB
          wake_up restores base weights... immediately overwritten by merged

        NEW architecture (<2s weight sync, ART recommended):
          sleep → train → wake → load_lora_adapter (2MB)
          wake_up restores base weights → they STAY as base
          SGLang applies LoRA on-the-fly during inference
        """

        # Phase 1: Sleep — release KV cache + model weights from GPU
        # Megatron subprocess needs the GPU memory for training
        t0 = time.perf_counter()
        sleep_time = await self.sleep()
        logger.info(f"Phase 1 — sleep(kv_cache+weights): {sleep_time:.2f}s")

        # Phase 2: Megatron training
        # verl: actor_output = self._update_actor(batch)
        await self._ensure_megatron_running()

        lora_path = self._get_last_checkpoint_dir()
        if lora_path is None:
            lora_path = self._get_checkpoint_dir(0)
            os.makedirs(lora_path, exist_ok=True)

        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        for f in os.listdir(jobs_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(jobs_dir, f))

        job = SGLangMegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._get_optimizer_state_path(),
            disk_packed_tensors=disk_packed_tensors,
            config=config if isinstance(config, dict) else config.model_dump(),
            experimental_config=experimental_config,
        )
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())

        # Monitor training log
        num_lines = 0
        while True:
            await asyncio.sleep(0.1)
            try:
                with open("/tmp/megatron_training_log.jsonl", "a+") as lf:
                    lf.seek(0)
                    lines = lf.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                self._merge_lora_shards(lora_path)
                                os.remove("/tmp/megatron_training_log.jsonl")
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        # Phase 3: New checkpoint
        next_step = self._latest_step + 1
        new_ckpt = self._get_checkpoint_dir(next_step)
        os.makedirs(new_ckpt, exist_ok=True)
        adapter_src = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(adapter_src):
            shutil.copy(adapter_src, os.path.join(new_ckpt, "adapter_model.safetensors"))
        for cfg_name in ["adapter_config.json"]:
            src = os.path.join(lora_path, cfg_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(new_ckpt, cfg_name))
        self._latest_step = next_step

        # Phase 4: Wake up — restore base weights + KV cache to GPU
        # Base weights come back UNTOUCHED — no more "restore then overwrite"
        t_wake = time.perf_counter()
        wake_time = await self.wake_up()
        logger.info(f"Phase 4 — wake_up(kv_cache+weights): {wake_time:.2f}s")

        # Phase 5: Hot-reload LoRA adapter (~2MB, <2s)
        # ART's recommended weight_sync_method="lora"
        # vs old: build 60GB merged dir + SGLang reload = 464s
        t_ws = time.perf_counter()
        weight_sync_time = await self._hot_reload_lora(new_ckpt, next_step)
        logger.info(
            f"Phase 5 — _hot_reload_lora: {weight_sync_time:.2f}s"
        )

        total_overhead = time.perf_counter() - t0
        logger.info(
            f"Total transition overhead: {total_overhead:.2f}s "
            f"(was 464s+ with disk merge)"
        )

    @staticmethod
    def _ensure_adapter_config(lora_path: str) -> None:
        """Write adapter_config.json if missing — SGLang requires it.

        Megatron saves only adapter_model.safetensors (no PEFT metadata).
        SGLang's LoRAConfig constructor reads adapter_config.json to get
        rank, alpha, and target_modules.

        Mirrors: src/art/megatron/service.py :: _ensure_lora_adapter_config()
        Config:  src/art/megatron/service.py :: _default_lora_adapter_config()
        """
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            return

        # Must match Megatron's LoraConfig exactly (service.py line 65-78)
        adapter_config = {
            "r": 1,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

    def _merge_lora_shards(self, lora_path: str) -> None:
        """Merge sharded LoRA adapters from distributed training."""
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        base_dir = Path(lora_path)
        shards = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
        if not shards:
            return

        adapter_path = base_dir / "adapter_model.safetensors"
        sharded: dict[str, list[torch.Tensor]] = {}
        for fn in shards:
            with safe_open(fn, framework="pt") as f:
                for k in f.keys():
                    sharded.setdefault(k, []).append(f.get_tensor(k))

        merged: dict[str, torch.Tensor] = {}
        if adapter_path.exists():
            merged = load_file(adapter_path)
        for k, tensors in sharded.items():
            merged[k] = torch.cat(tensors, dim=self._shard_cat_dim(k))

        save_file(merged, adapter_path)
        for fn in shards:
            fn.unlink()

    @staticmethod
    def _shard_cat_dim(key: str) -> int:
        """Determine the correct concat dimension for TP-sharded LoRA weights.

        In Megatron/Transformer TP sharding:
          - Column-parallel layers (gate_proj, up_proj, q_proj, k_proj, v_proj):
              base weight sharded on dim=0 → lora_A on dim=0, lora_B on dim=0
          - Row-parallel layers (down_proj, o_proj):
              base weight sharded on dim=1 → lora_A on dim=1, lora_B on dim=0

        For LoRA:
          - lora_A has shape (r, in_features) or shard thereof
          - lora_B has shape (out_features, r) or shard thereof

        The naive "lora_A → dim=1, lora_B → dim=0" heuristic fails for
        row-parallel layers where lora_A is sharded on dim=1 (input dim)
        but lora_B is NOT sharded (it's the small r-dimension output).
        """
        # Row-parallel layers: down_proj, o_proj (and MoE shared_expert variants)
        is_row_parallel = any(rp in key for rp in ["down_proj", "o_proj"])

        if "lora_A" in key:
            # lora_A shape: (r, in_features)
            # Column-parallel: in_features is NOT sharded → no concat needed (dim=0 is r)
            # Row-parallel: in_features IS sharded on dim=1 → concat on dim=1
            return 1 if is_row_parallel else 0
        else:
            # lora_B shape: (out_features, r)
            # Column-parallel: out_features IS sharded → concat on dim=0
            # Row-parallel: out_features is NOT sharded → no concat (dim=0 safe)
            return 0

    # ------------------------------------------------------------------
    # Native generation (verl-style, non-streaming)
    # ------------------------------------------------------------------

    async def generate_native(
        self,
        prompt: str | list[dict[str, str]],
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generation with actual token counts.

        Mirrors verl's SGLangHttpServer.generate() which uses
        tokenizer_manager.generate_request() directly, returning
        actual output_ids instead of SSE chunks.
        """
        if self._server is None:
            return {"error": "Server not started"}
        return await self._server.generate_native(prompt, sampling_params)
