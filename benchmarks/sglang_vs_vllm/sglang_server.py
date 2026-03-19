"""
SGLang server lifecycle management — verl-style.

The server is started ONCE and NEVER restarted. Between training steps,
memory is managed via release_memory_occupation / resume_memory_occupation
(matching verl's sleep/wake pattern). Weights are synced via
/update_weights (disk-based reload) since CUDA IPC requires in-process
SGLang Python API (not available over HTTP).

Architecture (mirrors verl/workers/rollout/sglang_rollout/):
  - SGLang process stays alive across all RL steps
  - KV cache is freed before training, reallocated after
  - Model weights are reloaded from merged safetensors on disk
  - CUDA graphs, NCCL communicators, tokenizer all survive
  - Native /generate endpoint returns actual token IDs (no SSE parsing)

Key SGLang HTTP endpoints used:
  - POST /flush_cache           — flush RadixAttention KV cache
  - POST /release_memory_occupation — free GPU memory (kv_cache, weights)
  - POST /resume_memory_occupation  — reallocate GPU memory
  - POST /update_weights        — reload weights from disk path
  - POST /generate              — native generation (returns token IDs)
  - GET  /health                — health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SGLangServerConfig:
    """Configuration for launching an SGLang server."""

    model_path: str
    served_model_name: str = ""  # defaults to model_path if empty
    port: int = 8200
    host: str = "0.0.0.0"
    tensor_parallel_size: int = 2
    mem_fraction_static: float = 0.85
    max_running_requests: int = 256
    dtype: str = "auto"
    trust_remote_code: bool = True
    python_executable: str = "python"
    log_file: str | None = None

    # LoRA — format must be "name=path"
    lora_paths: list[str] = field(default_factory=list)
    # Dynamic LoRA: required for /load_lora_adapter at runtime
    enable_lora: bool = False
    max_lora_rank: int = 8  # must >= max rank of any adapter loaded dynamically
    # Space-separated modules — SGLang uses nargs='+'. Use ["all"] for every module.
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Performance
    chunked_prefill_size: int = 32768
    disable_cuda_graph: bool = False  # set True if cuda_fp8.h missing
    enable_p2p_check: bool = True  # prevents multi-GPU hangs

    # verl-style: enable memory saver for sleep/wake support
    enable_memory_saver: bool = True

    # Additional raw args
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.served_model_name:
            self.served_model_name = self.model_path


class SGLangServerError(Exception):
    """Raised when the SGLang server encounters an error."""


class SGLangServer:
    """
    Manages the lifecycle of an SGLang inference server process.

    verl-style lifecycle:
      - start() launches the server ONCE
      - sleep() releases GPU memory (KV cache + optionally weights)
      - wake_up() resumes GPU memory (+ flushes stale radix cache)
      - update_weights_from_disk() reloads merged weights from safetensors
      - flush_cache() clears KV cache
      - generate_native() returns actual token IDs (not SSE streaming)
      - stop() is only called at the very end of the benchmark
    """

    def __init__(self, config: SGLangServerConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._startup_time: float = 0.0
        self._shutdown_time: float = 0.0
        self._log_fh: Any = None
        self._is_sleeping: bool = False
        # LoRA hot-reload state — mirrors ART's _hot_reload_lora pattern
        self._active_lora_name: str | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def last_startup_time(self) -> float:
        return self._startup_time

    @property
    def last_shutdown_time(self) -> float:
        return self._shutdown_time

    # ------------------------------------------------------------------
    # Build launch command
    # ------------------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        c = self.config
        cmd = [
            c.python_executable, "-m", "sglang.launch_server",
            "--model-path", c.model_path,
            "--served-model-name", c.served_model_name,
            "--port", str(c.port),
            "--host", c.host,
            "--tp", str(c.tensor_parallel_size),
            "--mem-fraction-static", str(c.mem_fraction_static),
            "--max-running-requests", str(c.max_running_requests),
            "--dtype", c.dtype,
            "--chunked-prefill-size", str(c.chunked_prefill_size),
        ]
        if c.trust_remote_code:
            cmd.append("--trust-remote-code")
        if c.disable_cuda_graph:
            cmd.append("--disable-cuda-graph")
        if c.enable_p2p_check:
            cmd.append("--enable-p2p-check")
        # verl-style: enable memory saver for sleep/wake support
        if c.enable_memory_saver:
            cmd.append("--enable-memory-saver")
        # LoRA: enable dynamic adapter loading at runtime
        if c.enable_lora:
            cmd.append("--enable-lora")
            cmd.extend(["--max-lora-rank", str(c.max_lora_rank)])
            cmd.append("--lora-target-modules")
            cmd.extend(c.lora_target_modules)  # nargs='+': each module is a separate arg
        # LoRA paths: each must be "name=path"
        for lp in c.lora_paths:
            cmd.extend(["--lora-paths", lp])
        cmd.extend(c.extra_args)
        return cmd

    # ------------------------------------------------------------------
    # Start / stop / health  (start once, stop only at end)
    # ------------------------------------------------------------------

    async def start(self, timeout: int = 600) -> float:
        """Start server ONCE. This server stays alive for the entire benchmark."""
        if self.is_running:
            logger.warning("Server already running — stopping first")
            await self.stop()

        await self._kill_port(self.config.port)

        cmd = self._build_cmd()
        logger.info("Starting SGLang (verl-style, will NOT restart): %s", " ".join(cmd))

        # Log file
        out_target: Any = subprocess.DEVNULL
        if self.config.log_file:
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
            self._log_fh = open(self.config.log_file, "a")
            out_target = self._log_fh

        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # Required for TP>1 — prevents false memory imbalance errors
        # Ref: https://verl.readthedocs.io/en/v0.5.x/workers/sglang_worker.html
        env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "True"

        t0 = time.perf_counter()
        self._process = subprocess.Popen(
            cmd,
            stdout=out_target,
            stderr=subprocess.STDOUT if self.config.log_file else subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )

        try:
            await self._wait_healthy(timeout)
        except Exception:
            await self.stop(timeout=10)
            raise

        self._startup_time = time.perf_counter() - t0
        self._is_sleeping = False
        logger.info("SGLang ready in %.2fs (pid=%s) — will stay alive for all steps",
                     self._startup_time, self._process.pid)
        return self._startup_time

    async def stop(self, timeout: int = 60) -> float:
        """Stop server — ONLY called at the very end of the benchmark."""
        if not self.is_running:
            self._shutdown_time = 0.0
            return 0.0

        pid = self._process.pid  # type: ignore[union-attr]
        t0 = time.perf_counter()
        logger.info("Stopping SGLang (final shutdown, pid=%s)", pid)

        # SIGTERM → wait → SIGKILL
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                break
            await asyncio.sleep(0.5)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if self._process is not None:
                self._process.wait(timeout=10)

        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self._process = None

        await self._kill_port(self.config.port)
        await asyncio.sleep(1.0)

        self._shutdown_time = time.perf_counter() - t0
        logger.info("SGLang stopped in %.2fs (final)", self._shutdown_time)
        return self._shutdown_time

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # verl-style memory management: sleep / wake_up
    # Mirrors: verl/workers/rollout/sglang_rollout/async_sglang_server.py
    # ------------------------------------------------------------------

    async def sleep(self, tags: list[str] | None = None) -> float:
        """Release GPU memory for training — verl's ReleaseMemoryOccupationReqInput.

        Frees KV cache (and optionally weights) so Megatron can use the GPU.
        The SGLang process stays alive — CUDA graphs, NCCL, tokenizer survive.
        """
        if tags is None:
            tags = ["kv_cache"]

        t0 = time.perf_counter()
        # Flush KV cache first (verl does this too)
        await self.flush_cache()

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/release_memory_occupation",
                    json={"tags": tags},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(f"release_memory_occupation failed: {r.status} {body[:200]}")
                    else:
                        self._is_sleeping = True
                        # LoRA adapter won't survive sleep/wake — base weights
                        # get restored on wake, adapter must be re-loaded
                        self._active_lora_name = None
                        elapsed = time.perf_counter() - t0
                        logger.info(f"SGLang sleep (release memory) in {elapsed:.2f}s — tags={tags}")
                        return elapsed
        except Exception as e:
            logger.warning(f"sleep() failed: {e}")

        return time.perf_counter() - t0

    async def wake_up(self, tags: list[str] | None = None) -> float:
        """Resume GPU memory after training — verl's ResumeMemoryOccupationReqInput.

        Reallocates KV cache (and restores weights if offloaded).
        """
        if tags is None:
            tags = ["kv_cache"]

        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/resume_memory_occupation",
                    json={"tags": tags},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(f"resume_memory_occupation failed: {r.status} {body[:200]}")
                    else:
                        self._is_sleeping = False
                        # Flush cache AFTER successful wake to clear stale radix tree
                        # entries that may point to deallocated KV blocks from before sleep.
                        # verl does this: await tokenizer_manager.flush_cache()
                        await self.flush_cache()
                        elapsed = time.perf_counter() - t0
                        logger.info(f"SGLang wake_up (resume memory) in {elapsed:.2f}s — tags={tags}")
                        return elapsed
        except Exception as e:
            logger.warning(f"wake_up() failed: {e}")

        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Weight sync via disk reload
    # ------------------------------------------------------------------

    async def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str = "auto",
    ) -> float:
        """Fallback: reload weights from disk path.

        Slower than CUDA IPC but works when IPC is not available.
        Still avoids full server restart.
        """
        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/update_weights_from_disk",
                    json={
                        "model_path": model_path,
                        "load_format": load_format,
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(f"update_weights (disk) failed: {r.status} {body[:200]}")
                    else:
                        elapsed = time.perf_counter() - t0
                        logger.info(f"Weight sync from disk in {elapsed:.2f}s")
                        return elapsed
        except Exception as e:
            logger.warning(f"update_weights (disk) failed: {e}")

        return time.perf_counter() - t0

    async def load_lora_adapter(
        self,
        lora_path: str,
        lora_name: str,
        flush_cache: bool = True,
    ) -> float:
        """Hot-reload LoRA adapter — ART's recommended weight_sync_method.

        Mirrors: src/art/sglang_backend/service.py :: _hot_reload_lora()

        SGLang loads the tiny adapter (~2MB for rank-1) and applies it
        on-the-fly during inference. Base weights stay UNTOUCHED.
        Generate requests must use lora_name as the 'model' parameter.

        vs update_weights_from_disk (464s):
          - No 60GB merged model dir build
          - No 60GB SGLang reload
          - Just ~2MB adapter load → <2s
        """
        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as s:
                payload: dict[str, Any] = {
                    "lora_path": lora_path,
                    "lora_name": lora_name,
                }
                # Primary endpoint (SGLang v0.4+)
                async with s.post(
                    f"{self.base_url}/load_lora_adapter",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(
                            f"load_lora_adapter failed: {r.status} {body[:200]}"
                        )
                        return -1.0

                    elapsed = time.perf_counter() - t0
                    self._active_lora_name = lora_name
                    logger.info(
                        f"LoRA adapter '{lora_name}' loaded in {elapsed:.2f}s"
                    )

                    if flush_cache:
                        await self.flush_cache()

                    return elapsed

        except Exception as e:
            logger.warning(f"load_lora_adapter failed: {e}")
            return -1.0

    @property
    def active_model_name(self) -> str:
        """Model name for generate requests — lora_name if loaded, else base."""
        return self._active_lora_name or self.config.served_model_name
    # Mirrors: verl/workers/rollout/sglang_rollout/async_sglang_server.py
    # ------------------------------------------------------------------

    async def generate_native(
        self,
        prompt: str | list[dict[str, str]],
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Native generation — returns actual token IDs and counts.

        Unlike HTTP streaming (which requires SSE parsing and undercounts tokens),
        this uses SGLang's /generate endpoint directly, matching verl's approach
        of using tokenizer_manager.generate_request().

        Returns:
            dict with: text, completion_tokens, prompt_tokens, finish_reason
        """
        if sampling_params is None:
            sampling_params = {}

        # Build request matching SGLang's /v1/chat/completions (non-streaming)
        if isinstance(prompt, list):
            # Chat format — OpenAI API uses model=lora_name
            body: dict[str, Any] = {
                "model": self.active_model_name,
                "messages": prompt,
                "stream": False,  # NON-streaming — get actual token counts
                **sampling_params,
            }
            endpoint = f"{self.openai_base_url}/chat/completions"
        else:
            # Raw text — SGLang native API uses lora_path field
            body: dict[str, Any] = {
                "model": self.config.served_model_name,
                "text": prompt,
                "sampling_params": sampling_params,
            }
            if self._active_lora_name:
                body["lora_path"] = self._active_lora_name
            endpoint = f"{self.base_url}/generate"

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    endpoint,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as r:
                    if r.status != 200:
                        err = await r.text()
                        return {"error": f"HTTP {r.status}: {err[:200]}"}

                    data = await r.json()

                    # Parse response — OpenAI chat format
                    if "choices" in data:
                        choice = data["choices"][0]
                        usage = data.get("usage", {})
                        return {
                            "text": choice.get("message", {}).get("content", ""),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "finish_reason": choice.get("finish_reason", ""),
                        }
                    # Raw /generate format
                    return {
                        "text": data.get("text", ""),
                        "completion_tokens": data.get("meta_info", {}).get(
                            "completion_tokens", 0
                        ),
                        "prompt_tokens": data.get("meta_info", {}).get(
                            "prompt_tokens", 0
                        ),
                    }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    async def flush_cache(self) -> bool:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/flush_cache",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _wait_healthy(self, timeout: int) -> None:
        deadline = time.perf_counter() + timeout
        interval = 2.0
        last_err: Exception | None = None

        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise SGLangServerError(
                    f"SGLang exited with code {self._process.returncode} during startup. "
                    f"Check: {self.config.log_file}"
                )
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{self.base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        if r.status == 200:
                            # Quick smoke test
                            await self._smoke_test()
                            return
            except Exception as e:
                last_err = e
            await asyncio.sleep(interval)
            interval = min(interval * 1.2, 10.0)

        raise SGLangServerError(
            f"SGLang not ready after {timeout}s. Last error: {last_err}"
        )

    async def _smoke_test(self) -> None:
        """One tiny request to confirm model is loaded."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.openai_base_url}/chat/completions",
                    json={
                        "model": self.config.served_model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        raise SGLangServerError(f"Smoke test: {r.status} {body[:200]}")
        except aiohttp.ClientError as e:
            raise SGLangServerError(f"Smoke test failed: {e}")

    @staticmethod
    async def _kill_port(port: int) -> None:
        try:
            p = await asyncio.create_subprocess_shell(
                f"lsof -ti:{port} | xargs -r kill -9 2>/dev/null || true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
        except Exception:
            pass
