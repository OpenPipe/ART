import asyncio
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any, AsyncIterator, Literal, TypedDict, cast

from peft.tuners.lora.config import LoraConfig
import torch

from .. import dev, types
from ..dev.get_model_config import default_target_modules
from ..dev.validate import is_dedicated_mode
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..unsloth.train import gc_and_empty_cuda_cache
from ..utils.convert_moe_lora import convert_checkpoint_if_needed
from ..utils.get_model_step import get_step_from_dir
from ..utils.lifecycle import (
    ChildProcessSupervisor,
    ServiceLifecycle,
    managed_process_cmd,
    terminate_asyncio_process_group,
    terminate_popen_process_group,
)
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm_runtime import (
    VllmRuntimeLaunchConfig,
    build_vllm_runtime_server_cmd,
    get_vllm_runtime_nccl_so_path,
    get_vllm_runtime_working_dir,
    wait_for_vllm_runtime,
)
from .lora import LORA_ALPHA, LORA_RANK
from .model_support.lora_disk import normalize_lora_checkpoint_to_vllm
from .runtime.client import (
    create_megatron_job_paths,
    stream_megatron_job,
    write_megatron_job,
)
from .runtime.jobs import (
    MegatronMergedTrainingJob,
    MegatronSFTTrainingJob,
    MegatronSyncJob,
    MegatronTrainingJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)
from .training.sft_batches import materialize_sft_batches

safetensors = importlib.import_module("safetensors")
safe_open = safetensors.safe_open


class _RuntimeRequestKwargs(TypedDict, total=False):
    headers: dict[str, str]


def create_identity_lora(
    base_model: str,
    lora_path: str,
    rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    random_state: int | None = None,
    allow_unvalidated_arch: bool = False,
) -> None:
    """Create an identity LoRA adapter for a Megatron model.

    For MoE models, this targets fused expert parameters and converts them to
    per-expert format. The conversion swaps lora_A/lora_B, producing A=zeros and
    B=Kaiming — which is critical for stable training when alpha/rank is large.

    Args:
        base_model: HuggingFace model identifier.
        lora_path: Directory to save the adapter files.
        rank: LoRA rank (default 1 for Megatron models).
        lora_alpha: LoRA alpha scaling factor.
    """
    from unittest.mock import patch

    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from .model_support import get_model_support_handler

    if random_state is not None:
        torch.manual_seed(random_state)
    target_modules = default_target_modules(base_model)
    handler = get_model_support_handler(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model_config = handler.identity_lora_model_config(base_config)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            model_config, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    model.name_or_path = base_model

    lora_config = LoraConfig(
        base_model_name_or_path=base_model,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=[],
        target_parameters=handler.identity_lora_target_parameters(
            model,
            target_modules=target_modules,
        ),
        bias="none",
    )

    meta = torch.device("meta")
    orig_to = torch.nn.Module.to

    def _skip_meta_to(
        module: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        device = kwargs.get("device") or (args[0] if args else None)
        if device == meta or str(device) == "meta":
            return module
        return orig_to(module, *args, **kwargs)

    with patch.object(torch.nn.Module, "to", _skip_meta_to):
        peft_model = get_peft_model(model, lora_config)

    os.makedirs(lora_path, exist_ok=True)
    peft_model.save_pretrained(lora_path)
    convert_checkpoint_if_needed(lora_path)

    final_config = LoraConfig(
        base_model_name_or_path=base_model,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
    ).to_dict()
    normalize_lora_checkpoint_to_vllm(
        lora_path,
        handler=handler,
        adapter_config=final_config,
    )
    del peft_model, model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@dataclass
class MegatronService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _latest_step: int = 0
    _megatron_process: asyncio.subprocess.Process | None = None
    _megatron_log_file: Any = None
    _megatron_log_path: str | None = None
    _vllm_process: subprocess.Popen[Any] | None = None
    _vllm_log_file: Any = None
    _vllm_log_path: str | None = None
    _vllm_host: str = "127.0.0.1"
    _vllm_port: int = 0
    _vllm_api_key: str | None = None
    _vllm_nccl_so_path: str | None = None
    _merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None = None
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
        self._validate_megatron_dependencies()

    def _on_child_process_exit(self, _error: RuntimeError) -> None:
        self.close()

    def _raise_if_child_failed(self) -> None:
        self._child_processes.raise_if_failed()

    @property
    def is_dedicated(self) -> bool:
        return is_dedicated_mode(self.config)

    @property
    def rollout_weights_mode(self) -> Literal["lora", "merged"]:
        mode = self.config.get("rollout_weights_mode", "lora")
        assert mode in {"lora", "merged"}
        return mode

    @property
    def _vllm_base_url(self) -> str:
        return f"http://{self._vllm_host}:{self._vllm_port}"

    def _megatron_random_state(self) -> int | None:
        for config_key in ("peft_args", "init_args"):
            random_state = self.config.get(config_key, {}).get("random_state")
            if random_state is not None:
                return int(random_state)
        return None

    @property
    def _allow_unvalidated_arch(self) -> bool:
        return bool(self.config.get("allow_unvalidated_arch", False))

    def _megatron_runtime_paths(self) -> tuple[str, str, str]:
        runtime_dir = Path(self.output_dir) / "megatron_runtime"
        jobs_dir = runtime_dir / "jobs"
        training_log_dir = runtime_dir / "training_logs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        training_log_dir.mkdir(parents=True, exist_ok=True)
        return (
            str(jobs_dir),
            str(training_log_dir),
            str(runtime_dir / "vllm_waking.lock"),
        )

    def _staging_lora_dir(self, step: int) -> str:
        return str(
            Path(self.output_dir) / "megatron_runtime" / "staging" / f"{step:04d}"
        )

    def _prepare_training_lora_dir(self, source_path: str, step: int) -> str:
        staging_dir = self._staging_lora_dir(step)
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        shutil.copytree(source_path, staging_dir)
        return staging_dir

    def _clear_wake_lock(self) -> None:
        _, _, wake_lock_path = self._megatron_runtime_paths()
        if os.path.exists(wake_lock_path):
            os.remove(wake_lock_path)

    def _allocate_master_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def _install_parent_signal_cleanup(self) -> None:
        self._lifecycle.install_parent_cleanup(self.close)

    def _restore_parent_signal_cleanup(self) -> None:
        self._lifecycle.restore_parent_cleanup()

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

    def _get_optimizer_state_path(self, job_type: Literal["rl", "sft"]) -> str:
        optimizer_state_path = os.path.join(
            self.output_dir, f"optimizer_states_{job_type}"
        )
        os.makedirs(optimizer_state_path, exist_ok=True)
        return optimizer_state_path

    def _default_lora_adapter_config(self) -> LoraConfig:
        return LoraConfig(
            base_model_name_or_path=self.base_model,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=default_target_modules(self.base_model),
            bias="none",
        )

    def _adapter_exists_and_loads(self, lora_path: str) -> bool:
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return False
        with safe_open(adapter_path, framework="pt") as adapter_file:
            keys = list(adapter_file.keys())
            if not keys:
                raise RuntimeError(f"LoRA adapter contains no tensors: {adapter_path}")
            for key in keys:
                adapter_file.get_tensor(key)
        return True

    def _create_identity_lora(self, lora_path: str) -> None:
        create_identity_lora(
            self.base_model,
            lora_path,
            random_state=self._megatron_random_state(),
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )

    def _ensure_identity_lora(self, lora_path: str) -> None:
        if self._adapter_exists_and_loads(lora_path):
            return
        self._create_identity_lora(lora_path)

    def _ensure_lora_adapter_config(
        self, lora_path: str, *, source_path: str | None = None
    ) -> None:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            return
        os.makedirs(lora_path, exist_ok=True)
        if source_path is not None:
            source_config = os.path.join(source_path, "adapter_config.json")
            if os.path.exists(source_config):
                shutil.copy(source_config, config_path)
                return
        self._default_lora_adapter_config().save_pretrained(lora_path)

    def _build_merged_weight_transfer_spec(self, step: int) -> MergedWeightTransferSpec:
        init_info = self._merged_weight_transfer_init_info
        assert init_info is not None
        if self._vllm_nccl_so_path is None:
            raise RuntimeError("vLLM runtime NCCL path is not initialized")
        return MergedWeightTransferSpec(
            init_info=init_info,
            vllm_base_url=self._vllm_base_url,
            served_model_name=f"{self.model_name}@{step}",
            api_key=self._vllm_api_key,
            nccl_so_path=self._vllm_nccl_so_path,
        )

    def _resolve_active_lora_path(self) -> str:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)
        return lora_path

    async def _set_served_model_name(self, step: int) -> None:
        import httpx

        self._raise_if_child_failed()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/art/set_served_model_name",
                json={"name": f"{self.model_name}@{step}"},
                **self._runtime_request_kwargs(),
                timeout=30.0,
            )
            response.raise_for_status()
        self._latest_step = step

    async def _init_merged_weight_transfer(self) -> None:
        import httpx

        self._raise_if_child_failed()
        if self._merged_weight_transfer_init_info is not None:
            return
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._vllm_base_url}/get_world_size",
                **self._runtime_request_kwargs(),
                timeout=30.0,
            )
            response.raise_for_status()
            inference_world_size = int(response.json()["world_size"])
        self._merged_weight_transfer_init_info = MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=self._allocate_master_port(),
            rank_offset=1,
            world_size=inference_world_size + 1,
        )

    async def _start_vllm_subprocess(
        self,
        lora_path: str,
        port: int,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, int]:
        import httpx

        self._raise_if_child_failed()
        server_args = self._runtime_server_args(config)
        api_key = server_args.get("api_key")
        self._vllm_api_key = api_key if isinstance(api_key, str) else None
        self._vllm_nccl_so_path = (
            str(get_vllm_runtime_nccl_so_path())
            if self.rollout_weights_mode == "merged"
            else None
        )
        cmd = build_vllm_runtime_server_cmd(
            VllmRuntimeLaunchConfig(
                base_model=self.base_model,
                port=port,
                host=self._vllm_host,
                cuda_visible_devices=self._runtime_cuda_visible_devices(),
                lora_path=lora_path,
                served_model_name=f"{self.model_name}@{self._latest_step}",
                rollout_weights_mode=self.rollout_weights_mode,
                engine_args=self._runtime_engine_args(config),
                server_args=server_args,
            )
        )

        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._vllm_log_path = os.path.join(log_dir, "vllm-runtime.log")
        self._vllm_log_file = open(self._vllm_log_path, "w", buffering=1)
        self._vllm_process = subprocess.Popen(
            managed_process_cmd(cmd),
            cwd=str(get_vllm_runtime_working_dir()),
            env=os.environ.copy(),
            stdout=self._vllm_log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        self._install_parent_signal_cleanup()
        self._vllm_port = port

        timeout = float(os.environ.get("ART_DEDICATED_VLLM_TIMEOUT", 1200))
        async with httpx.AsyncClient() as client:
            try:
                await wait_for_vllm_runtime(
                    process=self._vllm_process,
                    host=self._vllm_host,
                    port=self._vllm_port,
                    timeout=timeout,
                )
            except TimeoutError as exc:
                self._stop_vllm_subprocess()
                raise TimeoutError(
                    f"vLLM subprocess did not become ready within {timeout}s. "
                    f"Check logs at {log_dir}/vllm-runtime.log"
                ) from exc
            except RuntimeError as exc:
                returncode = self._vllm_process.returncode
                self._stop_vllm_subprocess()
                raise RuntimeError(
                    f"vLLM subprocess exited with code {returncode}. "
                    f"Check logs at {log_dir}/vllm-runtime.log"
                ) from exc

            try:
                response = await client.get(
                    f"{self._vllm_base_url}/v1/models",
                    **self._runtime_request_kwargs(),
                    timeout=5.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                self._stop_vllm_subprocess()
                raise RuntimeError(
                    "vLLM passed /health but /v1/models was not reachable. "
                    f"Check logs at {log_dir}/vllm-runtime.log"
                ) from exc
        assert self._vllm_process is not None
        assert self._vllm_log_path is not None
        self._child_processes.watch_popen(
            "vLLM runtime",
            self._vllm_process,
            log_path=self._vllm_log_path,
        )
        return self._vllm_host, self._vllm_port

    async def _reload_adapter(self, checkpoint_path: str, step: int) -> None:
        import httpx

        self._raise_if_child_failed()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": f"{self.model_name}@{step}",
                    "lora_path": checkpoint_path,
                    "load_inplace": True,
                },
                **self._runtime_request_kwargs(),
                timeout=60.0,
            )
            response.raise_for_status()
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

    async def _sync_dedicated_merged_weights(
        self,
        *,
        lora_path: str,
        step: int,
    ) -> None:
        self._raise_if_child_failed()
        await self._ensure_megatron_running()
        await self._init_merged_weight_transfer()
        self._clear_pending_jobs()
        job_path, log_path = self._create_megatron_job_paths()
        job = MegatronSyncJob(
            lora_path=lora_path,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
            merged_weight_transfer=self._build_merged_weight_transfer_spec(step),
            log_path=log_path,
        )
        write_megatron_job(job, job_path=job_path)
        async for _ in stream_megatron_job(
            job,
            job_path=job_path,
            process=self._megatron_process,
            process_log_path=self._megatron_log_path,
        ):
            pass
        self._latest_step = step

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
        self._raise_if_child_failed()
        if self.rollout_weights_mode == "merged":
            await self._set_served_model_name(step)
        else:
            await self._reload_adapter(checkpoint_dir, step)
        self._latest_step = step

    def _validate_megatron_dependencies(self) -> None:
        try:
            import megatron.bridge  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Megatron dependencies are not available in the active ART environment. "
                "Run `setup.sh` for this worktree and build the project venv with "
                "`uv sync --extra backend --extra megatron` before starting Megatron "
                "training."
            ) from exc

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        self._raise_if_child_failed()
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        self._validate_megatron_dependencies()

        train_script = Path(__file__).parent / "train.py"
        project_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        if self.is_dedicated:
            trainer_gpu_ids = self.config["trainer_gpu_ids"]
            num_gpus = len(trainer_gpu_ids)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in trainer_gpu_ids
            )
        else:
            num_gpus = torch.cuda.device_count()
        jobs_dir, _training_log_dir, wake_lock_path = self._megatron_runtime_paths()
        env["MODEL_IDENTIFIER"] = self.base_model
        if self._allow_unvalidated_arch:
            env["ART_MEGATRON_ALLOW_UNVALIDATED_ARCH"] = "1"
        env["ART_MEGATRON_JOBS_DIR"] = jobs_dir
        env["ART_MEGATRON_WAKE_LOCK_PATH"] = wake_lock_path
        master_addr = env.get("MASTER_ADDR", "127.0.0.1")
        master_port = str(self._allocate_master_port())
        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = master_port
        random_state = self._megatron_random_state()
        if random_state is not None:
            env["ART_MEGATRON_RANDOM_STATE"] = str(random_state)

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr",
            master_addr,
            "--master-port",
            master_port,
            "--nproc_per_node",
            str(num_gpus),
            str(train_script),
        ]
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        megatron_log_path = str(log_dir / "megatron-runtime.log")
        self._megatron_log_path = megatron_log_path
        self._megatron_log_file = open(
            megatron_log_path,
            "w",
            buffering=1,
        )
        self._megatron_process = await asyncio.create_subprocess_exec(
            *managed_process_cmd(command),
            cwd=str(project_root),
            env=env,
            stdout=self._megatron_log_file,
            stderr=self._megatron_log_file,
            start_new_session=True,
        )
        self._install_parent_signal_cleanup()
        self._child_processes.watch_asyncio_process(
            "Megatron worker",
            self._megatron_process,
            log_path=megatron_log_path,
        )

    def _clear_pending_jobs(self) -> None:
        jobs_dir, _training_log_dir, _wake_lock_path = self._megatron_runtime_paths()
        os.makedirs(jobs_dir, exist_ok=True)
        for job_name in os.listdir(jobs_dir):
            if job_name.endswith(".json"):
                os.remove(os.path.join(jobs_dir, job_name))

    def _create_megatron_job_paths(self) -> tuple[str, str]:
        jobs_dir, training_log_dir, _wake_lock_path = self._megatron_runtime_paths()
        return create_megatron_job_paths(
            jobs_dir=jobs_dir,
            training_log_dir=training_log_dir,
        )

    def _resolve_training_lora_path(self) -> str:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)
        return lora_path

    async def _prepare_for_training(self) -> str:
        self._raise_if_child_failed()
        self._validate_megatron_dependencies()
        await self._ensure_megatron_running()
        await self._sleep_runtime()
        gc_and_empty_cuda_cache()

        lora_path = self._resolve_training_lora_path()
        self._clear_pending_jobs()
        return lora_path

    async def _publish_training_checkpoint(
        self,
        *,
        lora_path: str,
    ) -> None:
        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)

        _jobs_dir, _training_log_dir, wake_lock_path = self._megatron_runtime_paths()
        try:
            with open(wake_lock_path, "w") as lock_file:
                lock_file.write("waking vllm\n")
            await self._wake_runtime()
        finally:
            if os.path.exists(wake_lock_path):
                os.remove(wake_lock_path)

        await self._reload_adapter(new_checkpoint_dir, next_step)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        self._raise_if_child_failed()
        lora_path = self._resolve_active_lora_path()

        if not self.is_dedicated and not self._sleep_mode_enabled():
            raise ValueError(
                "Shared-GPU mode requires engine_args.enable_sleep_mode=True "
                "for the external vLLM runtime"
            )

        port = (config or {}).get("server_args", {}).get("port", 8000)
        location = await self._start_vllm_subprocess(lora_path, port, config)
        if self.rollout_weights_mode == "lora":
            self._loaded_adapter_steps.add(self._latest_step)
        try:
            if self.rollout_weights_mode == "merged":
                await self._sync_dedicated_merged_weights(
                    lora_path=lora_path,
                    step=self._latest_step,
                )
        except BaseException:
            await self.aclose()
            raise
        return location

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        try:
            self._raise_if_child_failed()
            if _config.get("moe_routing_replay_bundle") is not None:
                raise RuntimeError(
                    "moe_routing_replay_bundle is only supported for in-process/runtime APIs; "
                    "MegatronService subprocess jobs must use moe_routing_replay_path."
                )
            if self.is_dedicated:
                await self._ensure_megatron_running()
                lora_path = self._resolve_active_lora_path()
                self._clear_pending_jobs()
                next_step = self._latest_step + 1
                new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
                staging_lora_path = self._prepare_training_lora_dir(
                    lora_path,
                    next_step,
                )
                job_path, log_path = self._create_megatron_job_paths()
                if self.rollout_weights_mode == "merged":
                    await self._init_merged_weight_transfer()
                    job: MegatronTrainingJob | MegatronMergedTrainingJob = (
                        MegatronMergedTrainingJob(
                            lora_path=staging_lora_path,
                            allow_unvalidated_arch=self._allow_unvalidated_arch,
                            optimizer_state_path=self._get_optimizer_state_path("rl"),
                            disk_packed_tensors=disk_packed_tensors,
                            config=config,
                            experimental_config=cast(dict[str, Any], _config),
                            moe_routing_replay_path=_config.get(
                                "moe_routing_replay_path"
                            ),
                            moe_routing_replay_strict=_config.get(
                                "moe_routing_replay_strict",
                                True,
                            ),
                            merged_weight_transfer=self._build_merged_weight_transfer_spec(
                                next_step
                            ),
                            log_path=log_path,
                        )
                    )
                else:
                    job = MegatronTrainingJob(
                        lora_path=staging_lora_path,
                        allow_unvalidated_arch=self._allow_unvalidated_arch,
                        optimizer_state_path=self._get_optimizer_state_path("rl"),
                        disk_packed_tensors=disk_packed_tensors,
                        config=config,
                        experimental_config=cast(dict[str, Any], _config),
                        moe_routing_replay_path=_config.get("moe_routing_replay_path"),
                        moe_routing_replay_strict=_config.get(
                            "moe_routing_replay_strict",
                            True,
                        ),
                        log_path=log_path,
                    )
                write_megatron_job(job, job_path=job_path)
                async for result in stream_megatron_job(
                    job,
                    job_path=job_path,
                    merge_output_path=new_checkpoint_dir,
                    process=self._megatron_process,
                    process_log_path=self._megatron_log_path,
                ):
                    yield {key: float(value) for key, value in result.items()}

                self._ensure_lora_adapter_config(
                    new_checkpoint_dir, source_path=staging_lora_path
                )
                if not self._adapter_exists_and_loads(new_checkpoint_dir):
                    raise RuntimeError(
                        f"Megatron training did not publish LoRA adapter: "
                        f"{new_checkpoint_dir}"
                    )
                if self.rollout_weights_mode == "merged":
                    self._latest_step = next_step
                else:
                    await self._reload_adapter(new_checkpoint_dir, next_step)
                shutil.rmtree(staging_lora_path, ignore_errors=True)
                return

            lora_path = await self._prepare_for_training()
            job_path, log_path = self._create_megatron_job_paths()
            job = MegatronTrainingJob(
                lora_path=lora_path,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
                optimizer_state_path=self._get_optimizer_state_path("rl"),
                disk_packed_tensors=disk_packed_tensors,
                config=config,
                experimental_config=cast(dict[str, Any], _config),
                moe_routing_replay_path=_config.get("moe_routing_replay_path"),
                moe_routing_replay_strict=_config.get(
                    "moe_routing_replay_strict", True
                ),
                log_path=log_path,
            )
            write_megatron_job(job, job_path=job_path)

            async for result in stream_megatron_job(
                job,
                job_path=job_path,
                process=self._megatron_process,
                process_log_path=self._megatron_log_path,
            ):
                yield {key: float(value) for key, value in result.items()}

            await self._publish_training_checkpoint(lora_path=lora_path)
        except BaseException:
            await self.aclose()
            raise

    async def train_sft(
        self,
        batches: list[SFTBatch],
        config: types.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        try:
            self._raise_if_child_failed()
            if self.is_dedicated:
                raise NotImplementedError(
                    "train_sft is not yet supported in dedicated mode"
                )
            lora_path = await self._prepare_for_training()
            serialized_batches = materialize_sft_batches(batches)
            job_path, log_path = self._create_megatron_job_paths()
            grad_accumulation_sequences = (
                config.batch_size if isinstance(config.batch_size, int) else None
            )
            job = MegatronSFTTrainingJob(
                lora_path=lora_path,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
                optimizer_state_path=self._get_optimizer_state_path("sft"),
                sft_data_dir=serialized_batches.sft_data_dir,
                num_batches=serialized_batches.num_batches,
                learning_rates=serialized_batches.learning_rates,
                grad_accumulation_sequences=grad_accumulation_sequences,
                log_path=log_path,
            )
            write_megatron_job(job, job_path=job_path)

            async for result in stream_megatron_job(
                job,
                job_path=job_path,
                process=self._megatron_process,
                process_log_path=self._megatron_log_path,
            ):
                yield {
                    "loss/train": float(result["loss"]),
                    "loss/learning_rate": float(result["learning_rate"]),
                    "loss/grad_norm": float(result["grad_norm"]),
                }

            await self._publish_training_checkpoint(lora_path=lora_path)
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        self.close()

    def _stop_vllm_subprocess(self) -> None:
        if self._vllm_process is not None:
            terminate_popen_process_group(self._vllm_process)
            self._vllm_process = None
        if self._vllm_log_file is not None:
            self._vllm_log_file.close()
            self._vllm_log_file = None
        self._vllm_log_path = None
        self._vllm_nccl_so_path = None
        self._merged_weight_transfer_init_info = None
        self._loaded_adapter_steps.clear()

    def _stop_megatron_process(self) -> None:
        if self._megatron_process is None:
            if self._megatron_log_file is not None:
                self._megatron_log_file.close()
                self._megatron_log_file = None
            self._megatron_log_path = None
            return
        terminate_asyncio_process_group(self._megatron_process)
        self._megatron_process = None
        if self._megatron_log_file is not None:
            self._megatron_log_file.close()
            self._megatron_log_file = None
        self._megatron_log_path = None

    def close(self) -> None:
        if not self._lifecycle.begin_close():
            return
        try:
            self._child_processes.close()
            self._stop_vllm_subprocess()
            self._stop_megatron_process()
            self._clear_wake_lock()
        finally:
            self._restore_parent_signal_cleanup()
