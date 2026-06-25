"""Unsloth training service with decoupled vLLM inference."""

import asyncio
from dataclasses import dataclass, field
from functools import cached_property
import logging
import os
import socket
from typing import Any, AsyncIterator, Literal, TypedDict, cast

import torch
from trl import GRPOTrainer

from .. import dev, types
from ..dev.validate import is_dedicated_mode
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.inputs import TrainInputs
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..utils.convert_moe_lora import convert_checkpoint_if_needed
from ..utils.get_model_step import get_step_from_dir
from ..utils.lifecycle import (
    ChildProcessSupervisor,
    ServiceLifecycle,
)
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm_runtime import (
    ManagedVllmRuntime,
    VllmRuntimeLaunchConfig,
)
from ..weight_transfer import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    trainer_init,
    trainer_send_weights,
)
from .train import (
    UnslothTrainContext,
    create_unsloth_train_context,
    gc_and_empty_cuda_cache,
    run_unsloth_rl_training,
    run_unsloth_sft_training,
)

logger = logging.getLogger(__name__)


def _peft_args_from_lora_config(lora_config: dev.LoRAConfig) -> dict[str, Any]:
    aliases = {
        "rank": "r",
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
        "init_weights": "init_lora_weights",
    }
    return {
        "r": 8,
        "lora_alpha": 16,
        **{aliases.get(k, k): v for k, v in lora_config.items()},
    }


class _RuntimeRequestKwargs(TypedDict, total=False):
    headers: dict[str, str]


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


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _normalize_merged_checkpoint_name(name: str) -> str:
    # PEFT wraps adapted modules under `.base_layer`, but vLLM expects the
    # original checkpoint parameter names during update_weights().
    normalized = name.removeprefix("base_model.model.")
    while ".base_layer." in normalized:
        normalized = normalized.replace(".base_layer.", ".")
    return normalized


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
    _vllm_runtime: ManagedVllmRuntime = field(
        default_factory=ManagedVllmRuntime,
        init=False,
        repr=False,
    )
    _weight_transfer_group: Any = field(default=None, init=False, repr=False)
    _lifecycle: ServiceLifecycle = field(
        default_factory=ServiceLifecycle,
        init=False,
        repr=False,
    )
    _child_processes: ChildProcessSupervisor = field(init=False, repr=False)
    _loaded_adapter_steps: set[int] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._child_processes = ChildProcessSupervisor(self._on_child_process_exit)

    def _on_child_process_exit(self, error: RuntimeError) -> None:
        logger.error("%s", error)
        self.close()

    def _raise_if_child_failed(self) -> None:
        self._child_processes.raise_if_failed()

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
        return self._vllm_runtime.base_url

    @property
    def _vllm_host(self) -> str:
        return self._vllm_runtime.host

    @property
    def _vllm_port(self) -> int:
        return self._vllm_runtime.port

    @_vllm_port.setter
    def _vllm_port(self, port: int) -> None:
        self._vllm_runtime.port = port

    @property
    def _vllm_api_key(self) -> str | None:
        return self._vllm_runtime.api_key

    @property
    def _vllm_nccl_so_path(self) -> str | None:
        return self._vllm_runtime.nccl_so_path

    def _runtime_cuda_visible_devices(self) -> str:
        if self.is_dedicated:
            return ",".join(str(gpu_id) for gpu_id in self.config["inference_gpu_ids"])
        if visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
            return visible
        return ",".join(str(index) for index in range(torch.cuda.device_count()))

    def _runtime_engine_args(
        self, config: dev.OpenAIServerConfig | None
    ) -> dict[str, object]:
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
        for key in ("model", "served_model_name"):
            engine_args.pop(key, None)
        return engine_args

    def _runtime_server_args(
        self, config: dev.OpenAIServerConfig | None
    ) -> dict[str, object]:
        server_args: dict[str, object] = {
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }
        if config and "server_args" in config:
            server_args.update(dict(config["server_args"]))
        for key in ("port", "host", "lora_modules"):
            server_args.pop(key, None)
        return server_args

    def _runtime_headers(self) -> dict[str, str]:
        if self._vllm_api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._vllm_api_key}"}

    def _runtime_request_kwargs(self) -> _RuntimeRequestKwargs:
        headers = self._runtime_headers()
        return {"headers": headers} if headers else {}

    def _sleep_mode_enabled(self) -> bool:
        return bool(self.config.get("engine_args", {}).get("enable_sleep_mode", True))

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
        self._raise_if_child_failed()
        server_args = self._runtime_server_args(config)
        location = await self._vllm_runtime.start(
            launch_config=VllmRuntimeLaunchConfig(
                base_model=self.base_model,
                port=port,
                host=self._vllm_runtime.host,
                cuda_visible_devices=self._runtime_cuda_visible_devices(),
                lora_path=lora_path,
                served_model_name=f"{self.model_name}@{self._latest_step}",
                rollout_weights_mode=self.rollout_weights_mode,
                engine_args=self._runtime_engine_args(config),
                server_args=server_args,
            ),
            output_dir=self.output_dir,
            child_processes=self._child_processes,
            install_parent_cleanup=lambda: self._lifecycle.install_parent_cleanup(
                self.close
            ),
            cleanup_on_error=self.close,
        )
        logger.info(
            "vLLM runtime ready on port %d (GPUs: %s)",
            port,
            self._runtime_cuda_visible_devices(),
        )
        return location

    async def _set_served_model_name(self, step: int) -> None:
        import httpx

        self._raise_if_child_failed()
        served_model_name = f"{self.model_name}@{step}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/art/set_served_model_name",
                json={"name": served_model_name},
                **self._runtime_request_kwargs(),
                timeout=30.0,
            )
            response.raise_for_status()
        logger.info(
            "[DEDICATED] Updated merged rollout alias to %s",
            served_model_name,
        )

    async def _init_merged_weight_transfer(self) -> None:
        import httpx

        self._raise_if_child_failed()
        if self._weight_transfer_group is not None:
            return

        async with httpx.AsyncClient() as client:
            world_size_response = await client.get(
                f"{self._vllm_base_url}/get_world_size",
                **self._runtime_request_kwargs(),
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
            if self._vllm_nccl_so_path is None:
                raise RuntimeError("vLLM runtime NCCL path is not initialized")

            master_port = _find_free_tcp_port()
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
                    **self._runtime_request_kwargs(),
                    timeout=300.0,
                )
            )
            self._weight_transfer_group = await asyncio.to_thread(
                trainer_init,
                {
                    "master_address": init_info["master_address"],
                    "master_port": init_info["master_port"],
                    "world_size": init_info["world_size"],
                    "nccl_so_path": self._vllm_nccl_so_path,
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

        self._raise_if_child_failed()
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
                        **self._runtime_request_kwargs(),
                        timeout=300.0,
                    )
                    response.raise_for_status()

                peft_model.merge_adapter()
                merged = True
                torch.cuda.synchronize()

                weights = self._merged_checkpoint_weights_for_vllm()
                response = await client.post(
                    f"{self._vllm_base_url}/start_weight_update",
                    json={"is_checkpoint_format": True},
                    **self._runtime_request_kwargs(),
                    timeout=300.0,
                )
                response.raise_for_status()
                update_info = {
                    "names": [name for name, _ in weights],
                    "dtype_names": [
                        str(tensor.dtype).removeprefix("torch.")
                        for _, tensor in weights
                    ],
                    "shapes": [list(tensor.shape) for _, tensor in weights],
                    "packed": True,
                    "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                    "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
                }

                _, update_response = await asyncio.gather(
                    asyncio.to_thread(
                        trainer_send_weights,
                        iter(weights),
                        {
                            "group": self._weight_transfer_group,
                            "packed": True,
                            "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                            "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
                        },
                    ),
                    client.post(
                        f"{self._vllm_base_url}/update_weights",
                        json={"update_info": update_info},
                        **self._runtime_request_kwargs(),
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
                response = await client.post(
                    f"{self._vllm_base_url}/finish_weight_update",
                    **self._runtime_request_kwargs(),
                    timeout=600.0,
                )
                response.raise_for_status()
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
                            **self._runtime_request_kwargs(),
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

        self._raise_if_child_failed()
        lora_name = f"{self.model_name}@{step}"
        logger.info(
            f"[DEDICATED] _reload_adapter START: lora_name={lora_name} "
            f"path={checkpoint_path}"
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": checkpoint_path,
                    "load_inplace": True,
                },
                **self._runtime_request_kwargs(),
                timeout=60.0,
            )
            response.raise_for_status()
        logger.info(
            f"[DEDICATED] _reload_adapter DONE: lora_name={lora_name} "
            f"status={response.status_code}"
        )
        self._latest_step = step
        self._loaded_adapter_steps.add(step)

    async def _unload_adapter(self, step: int) -> None:
        import httpx

        self._raise_if_child_failed()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/v1/unload_lora_adapter",
                json={"lora_name": f"{self.model_name}@{step}"},
                **self._runtime_request_kwargs(),
                timeout=30.0,
            )
            if response.status_code == 404:
                self._loaded_adapter_steps.discard(step)
                return
            response.raise_for_status()
        self._loaded_adapter_steps.discard(step)

    async def prune_loaded_adapters(self, *, retain_steps: set[int]) -> None:
        if self.rollout_weights_mode != "lora" or self._vllm_port == 0:
            return
        for step in sorted(self._loaded_adapter_steps - retain_steps):
            if step == self._latest_step:
                continue
            await self._unload_adapter(step)

    def close(self) -> None:
        """Terminate vLLM subprocess if running."""
        if not self._lifecycle.begin_close():
            return
        weight_transfer_group = self._weight_transfer_group
        self._weight_transfer_group = None
        try:
            try:
                if weight_transfer_group is not None:
                    close = getattr(weight_transfer_group, "close", None)
                    if close is not None:
                        close()
            finally:
                self._child_processes.close()
                self._vllm_runtime.close()
                self._loaded_adapter_steps.clear()
        finally:
            self._lifecycle.restore_parent_cleanup()

    # =========================================================================
    # start_openai_server
    # =========================================================================

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        self._raise_if_child_failed()
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self._state.trainer.save_model(lora_path)
            convert_checkpoint_if_needed(lora_path)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)

        if not self.is_dedicated:
            if not self._sleep_mode_enabled():
                raise ValueError(
                    "Shared-GPU mode requires engine_args.enable_sleep_mode=True "
                    "for the external vLLM runtime"
                )
            self._state.offload_to_cpu()

        port = (config or {}).get("server_args", {}).get("port", 8000)
        vllm_location = await self._start_vllm_subprocess(
            lora_path,
            port,
            config=config,
        )
        if self.rollout_weights_mode == "lora":
            self._loaded_adapter_steps.add(self._latest_step)
        try:
            if self.rollout_weights_mode == "merged":
                _ = self._state
                await self._init_merged_weight_transfer()
                await self._sync_merged_weights(self._latest_step, False)
        except BaseException:
            await self.aclose()
            raise
        return vllm_location

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def _sleep_runtime(self) -> None:
        import httpx

        self._raise_if_child_failed()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/sleep",
                params={"level": 1, "mode": "wait"},
                **self._runtime_request_kwargs(),
                timeout=300.0,
            )
            response.raise_for_status()
        self._is_sleeping = True

    async def _wake_runtime(self) -> None:
        import httpx

        self._raise_if_child_failed()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/wake_up",
                **self._runtime_request_kwargs(),
                timeout=300.0,
            )
            response.raise_for_status()
        self._is_sleeping = False

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        if self.rollout_weights_mode == "merged":
            await self._set_served_model_name(step)
        else:
            await self._reload_adapter(checkpoint_dir, step)
        self._latest_step = step

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        try:
            self._raise_if_child_failed()
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
        except BaseException:
            await self.aclose()
            raise

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
        await self._sleep_runtime()
        gc_and_empty_cuda_cache()
        self._state.reload_to_gpu()

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

        self._state.offload_to_cpu()
        gc_and_empty_cuda_cache()
        await asyncio.sleep(0.5)
        await self._wake_runtime()

        new_step = int(os.path.basename(checkpoint_dir))
        await self._reload_adapter(checkpoint_dir, new_step)
        self._latest_step = new_step

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
        try:
            self._raise_if_child_failed()
            if self.is_dedicated:
                raise NotImplementedError(
                    "train_sft is not yet supported in dedicated mode"
                )

            await self._sleep_runtime()
            gc_and_empty_cuda_cache()
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

            checkpoint_dir = save_checkpoint(
                trainer=self._state.trainer,
                output_dir=self.output_dir,
                verbose=verbose,
            )

            self._state.offload_to_cpu()
            gc_and_empty_cuda_cache()
            await asyncio.sleep(0.5)
            await self._wake_runtime()
            new_step = int(os.path.basename(checkpoint_dir))
            await self._reload_adapter(checkpoint_dir, new_step)
            self._latest_step = new_step

            if verbose:
                print("SFT training finished")
        except BaseException:
            await self.aclose()
            raise

    @cached_property
    def _state(self) -> UnslothTrainContext:
        init_args = dict(self.config.get("init_args", {}))
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        init_args["model_name"] = checkpoint_dir or self.base_model
        return create_unsloth_train_context(
            init_args=init_args,
            peft_args=_peft_args_from_lora_config(
                cast(dev.BackendModelConfig, self.config).get("lora_config", {})
            ),
            trainer_args=cast(dict[str, Any], self.config.get("trainer_args", {})),
        )
