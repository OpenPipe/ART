"""Unsloth training service with decoupled vLLM inference."""

import asyncio
from dataclasses import dataclass, field
from functools import cached_property
import json
import logging
import os
import subprocess
import sys
from typing import Any, AsyncIterator, Literal, cast

import torch
from trl import GRPOTrainer
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..dev.validate import is_dedicated_mode
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.inputs import TrainInputs
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..utils.convert_moe_lora import convert_checkpoint_if_needed
from ..utils.get_model_step import get_step_from_dir
from ..utils.network import find_free_tcp_port
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from .train import (
    UnslothTrainContext,
    create_unsloth_train_context,
    gc_and_empty_cuda_cache,
    run_unsloth_rl_training,
    run_unsloth_sft_training,
)

logger = logging.getLogger(__name__)


def save_checkpoint(
    trainer: "GRPOTrainer",
    output_dir: str,
    verbose: bool = False,
) -> str:
    """Save a checkpoint and return the checkpoint directory path."""
    # _use_adapter() may load reference adapters for KL/logprob computation and
    # keep them attached to the PEFT model. Before saving, keep only active
    # adapter(s) and drop the rest to release GPU/CPU memory.
    try:
        peft_model = trainer.accelerator.unwrap_model(  # type: ignore[attr-defined]
            trainer.model, keep_fp32_wrapper=False
        )
        active_adapters = peft_model.active_adapter
        if isinstance(active_adapters, str):
            keep_adapters = {active_adapters}
        else:
            keep_adapters = set(active_adapters)

        before_adapters = list(peft_model.peft_config.keys())
        print(f"Adapters before cleanup: {before_adapters}")
        print(f"Keeping active adapter(s): {sorted(keep_adapters)}")

        for adapter_name in before_adapters:
            if adapter_name not in keep_adapters:
                peft_model.delete_adapter(adapter_name)
                print(f"Deleted unused adapter: {adapter_name}")

        after_adapters = list(peft_model.peft_config.keys())
        print(f"Adapters after cleanup: {after_adapters}")
    except Exception as e:
        print(f"Warning: failed to cleanup unused adapters: {e}")

    if verbose:
        print("Saving new LoRA adapter...")
    next_step = get_step_from_dir(output_dir) + 1
    checkpoint_dir = get_step_checkpoint_dir(output_dir, next_step)
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save_model(checkpoint_dir)
    convert_checkpoint_if_needed(checkpoint_dir)

    gc_and_empty_cuda_cache()
    return checkpoint_dir


def _normalize_merged_checkpoint_name(name: str) -> str:
    # PEFT wraps adapted modules under `.base_layer`, but vLLM expects the
    # original checkpoint parameter names during update_weights().
    normalized = name.removeprefix("base_model.model.")
    while ".base_layer." in normalized:
        normalized = normalized.replace(".base_layer.", ".")
    return normalized


_find_free_tcp_port = find_free_tcp_port


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
    # Dedicated mode subprocess state
    _vllm_process: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]
    _vllm_log_file: Any = field(default=None, repr=False)
    _vllm_host: str = "127.0.0.1"
    _vllm_port: int = 0
    _weight_transfer_group: Any = field(default=None, init=False, repr=False)

    @property
    def is_dedicated(self) -> bool:
        return is_dedicated_mode(self.config)

    @property
    def rollout_weights_mode(self) -> Literal["lora", "merged"]:
        mode = self.config["rollout_weights_mode"]
        assert mode in {"lora", "merged"}
        return mode

    @property
    def _vllm_base_url(self) -> str:
        return f"http://{self._vllm_host}:{self._vllm_port}"

    def _next_lora_id(self) -> int:
        """Return a new unique LoRA ID to avoid collisions in vLLM."""
        self._lora_id_counter += 1
        return self._lora_id_counter

    async def aclose(self) -> None:
        state = self.__dict__.get("_state")
        if isinstance(state, UnslothTrainContext):
            await state.stop_background_training()
        self.close()

    # =========================================================================
    # Dedicated mode: vLLM subprocess lifecycle
    # =========================================================================

    async def _start_vllm_subprocess(
        self,
        lora_path: str,
        port: int,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, int]:
        """Launch vLLM as a subprocess on inference GPUs. Returns (host, port)."""
        import atexit

        inference_gpu_ids = self.config["inference_gpu_ids"]
        cuda_devices = ",".join(str(g) for g in inference_gpu_ids)

        # Build server_args: ART defaults, then user overrides, strip CLI-handled keys
        server_args: dict[str, object] = {
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }
        if config and "server_args" in config:
            server_args.update(dict(config["server_args"]))
        for key in ("port", "host", "lora_modules", "api_key"):
            server_args.pop(key, None)

        # Build engine_args: model-level config, then user server overrides,
        # add dedicated-mode defaults, strip CLI-handled keys
        engine_args = dict(self.config.get("engine_args", {}))
        if config and "engine_args" in config:
            engine_args.update(dict(config["engine_args"]))
        engine_args.setdefault("generation_config", "vllm")
        if self.rollout_weights_mode == "merged":
            engine_args["weight_transfer_config"] = {"backend": "nccl"}
            engine_args.pop("enable_lora", None)
            engine_args.pop("max_loras", None)
        else:
            engine_args["enable_lora"] = True
            engine_args.setdefault("max_loras", 2)
        for key in ("model", "served_model_name", "enable_sleep_mode"):
            engine_args.pop(key, None)

        cmd = [
            sys.executable,
            "-m",
            "art.vllm.dedicated_server",
            f"--model={self.base_model}",
            f"--port={port}",
            f"--host={self._vllm_host}",
            f"--cuda-visible-devices={cuda_devices}",
            f"--lora-path={lora_path}",
            f"--served-model-name={self.model_name}@{self._latest_step}",
            f"--rollout-weights-mode={self.rollout_weights_mode}",
            f"--engine-args-json={json.dumps(engine_args)}",
            f"--server-args-json={json.dumps(server_args)}",
        ]

        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._vllm_log_file = open(
            os.path.join(log_dir, "vllm-dedicated.log"), "w", buffering=1
        )

        self._vllm_process = subprocess.Popen(
            cmd, stdout=self._vllm_log_file, stderr=subprocess.STDOUT, bufsize=1
        )
        self._vllm_port = port

        import httpx

        timeout = float(os.environ.get("ART_DEDICATED_VLLM_TIMEOUT", 600))
        poll_interval = 1.0
        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                if self._vllm_process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM subprocess exited with code {self._vllm_process.returncode}. "
                        f"Check logs at {log_dir}/vllm-dedicated.log"
                    )
                try:
                    resp = await client.get(
                        f"http://{self._vllm_host}:{self._vllm_port}/v1/models",
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        break
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            else:
                self.close()
                raise TimeoutError(
                    f"vLLM subprocess did not become ready within {timeout}s. "
                    f"Check logs at {log_dir}/vllm-dedicated.log"
                )

        atexit.register(self.close)
        logger.info("vLLM subprocess ready on port %d (GPUs: %s)", port, cuda_devices)
        return self._vllm_host, self._vllm_port

    async def _set_served_model_name(self, step: int) -> None:
        import httpx

        served_model_name = f"{self.model_name}@{step}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/art/set_served_model_name",
                json={"name": served_model_name},
                timeout=30.0,
            )
            response.raise_for_status()
        logger.info(
            "[DEDICATED] Updated merged rollout alias to %s",
            served_model_name,
        )

    async def _init_merged_weight_transfer(self) -> None:
        import httpx
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        if self._weight_transfer_group is not None:
            return

        async with httpx.AsyncClient() as client:
            world_size_response = await client.get(
                f"{self._vllm_base_url}/get_world_size",
                timeout=30.0,
            )
            try:
                world_size_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    "Merged rollout weights require a vLLM build with the "
                    "/get_world_size endpoint"
                ) from exc
            inference_world_size = int(world_size_response.json()["world_size"])

            master_port = find_free_tcp_port()
            init_info = {
                "master_address": "127.0.0.1",
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": inference_world_size + 1,
            }

            remote_init_task = asyncio.create_task(
                client.post(
                    f"{self._vllm_base_url}/init_weight_transfer_engine",
                    json={"init_info": init_info},
                    timeout=300.0,
                )
            )
            # TODO: replace this with a real readiness handshake if this ever flakes.
            await asyncio.sleep(1.0)
            self._weight_transfer_group = await asyncio.to_thread(
                NCCLWeightTransferEngine.trainer_init,
                {
                    "master_address": init_info["master_address"],
                    "master_port": init_info["master_port"],
                    "world_size": init_info["world_size"],
                },
            )
            remote_init_response = await remote_init_task
            try:
                remote_init_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    "Merged rollout weights require a vLLM build with the "
                    "/init_weight_transfer_engine endpoint"
                ) from exc

        logger.info(
            "[DEDICATED] Initialized merged weight transfer: inference_world_size=%d",
            inference_world_size,
        )

    def _merged_checkpoint_weights_for_vllm(self) -> list[tuple[str, torch.Tensor]]:
        model = self._state.peft_model.base_model.model
        device = next(model.parameters()).device
        assert device.type == "cuda"

        weights: list[tuple[str, torch.Tensor]] = []
        normalized_names: set[str] = set()
        for name, tensor in model.state_dict().items():
            if "lora_" in name:
                continue
            normalized_name = _normalize_merged_checkpoint_name(name)
            assert normalized_name not in normalized_names
            normalized_names.add(normalized_name)
            detached = tensor.detach()
            if detached.device != device:
                detached = detached.to(device=device, non_blocking=True)
            weights.append((normalized_name, detached))

        assert weights
        return weights

    async def _sync_merged_weights(
        self,
        step: int,
        pause_generation: bool,
    ) -> None:
        import httpx
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        assert self._weight_transfer_group is not None

        peft_model = self._state.peft_model
        merged = False
        error: Exception | None = None
        logger.info("[DEDICATED] Syncing merged rollout weights for step %d", step)

        async with httpx.AsyncClient() as client:
            try:
                if pause_generation:
                    response = await client.post(
                        f"{self._vllm_base_url}/pause",
                        params={"mode": "wait"},
                        timeout=300.0,
                    )
                    response.raise_for_status()

                peft_model.merge_adapter()
                merged = True
                torch.cuda.synchronize()

                weights = self._merged_checkpoint_weights_for_vllm()
                update_info = {
                    "names": [name for name, _ in weights],
                    "dtype_names": [
                        str(tensor.dtype).removeprefix("torch.")
                        for _, tensor in weights
                    ],
                    "shapes": [list(tensor.shape) for _, tensor in weights],
                    "is_checkpoint_format": True,
                }

                _, update_response = await asyncio.gather(
                    asyncio.to_thread(
                        NCCLWeightTransferEngine.trainer_send_weights,
                        iter(weights),
                        {"group": self._weight_transfer_group},
                    ),
                    client.post(
                        f"{self._vllm_base_url}/update_weights",
                        json={"update_info": update_info},
                        timeout=600.0,
                    ),
                )
                try:
                    update_response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(
                        "Merged rollout weights require a vLLM build with the "
                        "/update_weights endpoint"
                    ) from exc
                self._latest_step = step
                await self._set_served_model_name(step)
            except Exception as exc:
                error = exc
                raise
            finally:
                if merged:
                    peft_model.unmerge_adapter()
                    torch.cuda.synchronize()
                if pause_generation:
                    try:
                        response = await client.post(
                            f"{self._vllm_base_url}/resume",
                            timeout=30.0,
                        )
                        response.raise_for_status()
                    except Exception:
                        if error is None:
                            raise
                        logger.exception(
                            "Failed to resume generation after merged weight sync error"
                        )

        logger.info(
            "[DEDICATED] Merged rollout sync complete for step %d",
            step,
        )

    async def _reload_adapter(self, checkpoint_path: str, step: int) -> None:
        """Reload LoRA adapter in vLLM subprocess via HTTP."""
        import httpx

        lora_name = f"{self.model_name}@{step}"
        logger.info(
            f"[DEDICATED] _reload_adapter START: lora_name={lora_name} "
            f"path={checkpoint_path}"
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self._vllm_host}:{self._vllm_port}/v1/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": checkpoint_path,
                    "load_inplace": True,
                },
                timeout=60.0,
            )
            response.raise_for_status()
        logger.info(
            f"[DEDICATED] _reload_adapter DONE: lora_name={lora_name} "
            f"status={response.status_code}"
        )

    def close(self) -> None:
        """Terminate vLLM subprocess if running."""
        self._weight_transfer_group = None
        if self._vllm_process is None:
            return
        self._vllm_process.terminate()
        try:
            self._vllm_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._vllm_process.kill()
            self._vllm_process.wait()
        self._vllm_process = None
        if self._vllm_log_file is not None:
            self._vllm_log_file.close()
            self._vllm_log_file = None

    # =========================================================================
    # start_openai_server
    # =========================================================================

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self._state.trainer.save_model(lora_path)
            convert_checkpoint_if_needed(lora_path)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)

        if self.is_dedicated:
            port = (config or {}).get("server_args", {}).get("port", 8000)
            vllm_location = await self._start_vllm_subprocess(
                lora_path,
                port,
                config=config,
            )
            if self.rollout_weights_mode == "merged":
                _ = self._state
                await self._init_merged_weight_transfer()
                await self._sync_merged_weights(self._latest_step, False)
            return vllm_location

        # Shared mode: in-process vLLM
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
        if self.is_dedicated:
            return False
        return self._is_sleeping

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        """Register a LoRA adapter for a specific checkpoint step.
        This is called when training is skipped but the checkpoint is renamed.
        """
        logger.info(
            f"[DEDICATED] register_lora_for_step called: step={step} "
            f"checkpoint_dir={checkpoint_dir} is_dedicated={self.is_dedicated}"
        )
        if self.is_dedicated:
            if self.rollout_weights_mode == "merged":
                await self._set_served_model_name(step)
            else:
                await self._reload_adapter(checkpoint_dir, step)
            self._latest_step = step
            return

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
        if self.is_dedicated:
            async for result in self._train_dedicated(
                disk_packed_tensors, config, _config, verbose
            ):
                yield result
            return

        async for result in self._train_shared(
            disk_packed_tensors, config, _config, verbose
        ):
            yield result

    async def _train_dedicated(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train in dedicated mode — no sleep/wake, vLLM keeps running on separate GPU."""
        async for result in run_unsloth_rl_training(
            self._state,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            _config=_config,
            verbose=verbose,
        ):
            yield result

        checkpoint_dir = save_checkpoint(
            trainer=self._state.trainer,
            output_dir=self.output_dir,
            verbose=verbose,
        )

        new_step = int(os.path.basename(checkpoint_dir))
        if self.rollout_weights_mode == "merged":
            logger.info(
                "[DEDICATED] _train_dedicated: saved checkpoint step=%s, syncing merged weights...",
                new_step,
            )
            await self._sync_merged_weights(new_step, True)
        else:
            logger.info(
                "[DEDICATED] _train_dedicated: saved checkpoint step=%s, reloading adapter...",
                new_step,
            )
            await self._reload_adapter(checkpoint_dir, new_step)
        self._latest_step = new_step
        logger.info(
            f"[DEDICATED] _train_dedicated: inference weights updated for step {new_step}"
        )

    async def _train_shared(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train in shared mode — sleep/wake cycle with in-process vLLM."""
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

        async for result in run_unsloth_rl_training(
            self._state,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            _config=_config,
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

    # =========================================================================
    # SFT training
    # =========================================================================

    async def train_sft(
        self,
        batches: list[SFTBatch],
        config: types.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train using SFT on pre-computed batches.

        Args:
            batches: List of SFTBatch objects to train on.
            config: SFT batch/grad-accumulation configuration.
            verbose: Whether to print detailed logs.

        Yields:
            Dictionary containing training metrics for each batch.
        """
        if self.is_dedicated:
            raise NotImplementedError(
                "train_sft is not yet supported in dedicated mode"
            )
        import time

        llm = await self.llm

        # === Setup ===
        # Pause generation to prevent new requests during training
        await llm.pause_generation()

        # Determine sleep level based on outstanding requests
        has_unfinished = llm.output_processor.has_unfinished_requests()
        if has_unfinished:
            sleep_level = 1
        else:
            await llm.reset_prefix_cache()
            sleep_level = 2

        # Put workers to sleep
        await run_on_workers(llm, do_sleep, level=sleep_level)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        # Reload training model to GPU (after vLLM is asleep)
        self._state.reload_to_gpu()
        if verbose:
            print("SFT training started")

        async for result in run_unsloth_sft_training(
            self._state,
            batches,
            verbose=verbose,
            max_grad_norm=1.0,
        ):
            yield {
                "loss/train": result["loss"],
                "loss/learning_rate": result["learning_rate"],
                "loss/grad_norm": result["grad_norm"],
            }

        # === Cleanup ===
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
        await asyncio.sleep(0.5)

        # Wake up workers
        await run_on_workers(llm, do_wake_up)
        self._is_sleeping = False

        # Add the new LoRA adapter
        new_step = int(os.path.basename(checkpoint_dir))
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

        # Resume generation after LoRA swap is complete
        await llm.resume_generation()

        if verbose:
            print("SFT training finished")

    @cached_property
    def _state(self) -> UnslothTrainContext:
        init_args = dict(self.config.get("init_args", {}))
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            init_args["model_name"] = checkpoint_dir
        else:
            init_args["model_name"] = self.base_model
        return create_unsloth_train_context(
            init_args=init_args,
            peft_args=cast(dict[str, Any], self.config.get("peft_args", {})),
            trainer_args=cast(dict[str, Any], self.config.get("trainer_args", {})),
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
