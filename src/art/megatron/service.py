import asyncio
from dataclasses import dataclass, field
from functools import cached_property
import importlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from typing import Any, AsyncIterator, Literal, cast

from peft.tuners.lora.config import LoraConfig
import torch
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..dev.get_model_config import default_target_modules
from ..dev.validate import is_dedicated_mode
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..unsloth.service import do_sleep, do_wake_up, gc_and_empty_cuda_cache
from ..utils.convert_moe_lora import convert_checkpoint_if_needed
from ..utils.get_model_step import get_step_from_dir
from ..utils.network import find_free_tcp_port
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, openai_server_task, run_on_workers
from .client import create_megatron_job_paths, stream_megatron_job, write_megatron_job
from .jobs import (
    MegatronMergedTrainJob,
    MegatronSFTTrainingJob,
    MegatronSyncJob,
    MegatronTrainingJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)
from .lora import LORA_ALPHA, LORA_RANK
from .sft_batches import materialize_sft_batches

safetensors = importlib.import_module("safetensors")
safe_open = safetensors.safe_open


def create_identity_lora(
    base_model: str,
    lora_path: str,
    rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    random_state: int | None = None,
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

    if random_state is not None:
        torch.manual_seed(random_state)
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model_config = base_config
    nested_text_config = getattr(base_config, "text_config", None)
    if not hasattr(base_config, "vocab_size") and hasattr(
        nested_text_config, "vocab_size"
    ):
        model_config = nested_text_config
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
        target_parameters=[
            name
            for name, _ in model.named_parameters()
            if name.endswith(
                (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.out_proj.weight",
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                    "mlp.shared_expert.gate_proj.weight",
                    "mlp.shared_expert.up_proj.weight",
                    "mlp.shared_expert.down_proj.weight",
                )
            )
        ],
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

    # Write final adapter_config with per-expert target_modules
    LoraConfig(
        base_model_name_or_path=base_model,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=default_target_modules(base_model),
        bias="none",
    ).save_pretrained(lora_path)

    del peft_model, model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


logger = logging.getLogger(__name__)


@dataclass
class MegatronService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1
    _megatron_process: asyncio.subprocess.Process | None = None
    _vllm_process: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]
    _vllm_log_file: Any = field(default=None, repr=False)
    _vllm_host: str = "127.0.0.1"
    _vllm_port: int = 0
    _merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None = field(
        default=None,
        repr=False,
    )

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

    def _allocate_master_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

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

    def _adapter_has_weights(self, lora_path: str) -> bool:
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return False
        try:
            with safe_open(adapter_path, framework="pt") as adapter_file:
                for key in adapter_file.keys():
                    tensor = adapter_file.get_tensor(key)
                    if torch.any(tensor != 0):
                        return True
        except Exception:
            return False
        return False

    def _create_identity_lora(self, lora_path: str) -> None:
        create_identity_lora(
            self.base_model,
            lora_path,
            random_state=self._megatron_random_state(),
        )

    def _ensure_identity_lora(self, lora_path: str) -> None:
        if self._adapter_has_weights(lora_path):
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
        return MergedWeightTransferSpec(
            init_info=init_info,
            vllm_base_url=self._vllm_base_url,
            served_model_name=f"{self.model_name}@{step}",
        )

    def _resolve_active_lora_path(self) -> str:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        if self.is_dedicated or self.rollout_weights_mode == "lora":
            self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)
        return lora_path

    async def _set_served_model_name(self, step: int) -> None:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/art/set_served_model_name",
                json={"name": f"{self.model_name}@{step}"},
                timeout=30.0,
            )
            response.raise_for_status()
        self._latest_step = step

    async def _init_merged_weight_transfer(self) -> None:
        import httpx

        if self._merged_weight_transfer_init_info is not None:
            return
        assert len(self.config["trainer_gpu_ids"]) == 1
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._vllm_base_url}/get_world_size",
                timeout=30.0,
            )
            response.raise_for_status()
            inference_world_size = int(response.json()["world_size"])
        self._merged_weight_transfer_init_info = MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=find_free_tcp_port(),
            rank_offset=1,
            world_size=inference_world_size + 1,
        )

    async def _start_vllm_subprocess(
        self,
        lora_path: str,
        port: int,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, int]:
        import atexit

        import httpx

        inference_gpu_ids = self.config["inference_gpu_ids"]
        cuda_devices = ",".join(str(gpu_id) for gpu_id in inference_gpu_ids)

        server_args: dict[str, object] = {
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }
        if config and "server_args" in config:
            server_args.update(dict(config["server_args"]))
        for key in ("port", "host", "lora_modules", "api_key"):
            server_args.pop(key, None)

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
            cmd,
            stdout=self._vllm_log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self._vllm_port = port

        timeout = float(os.environ.get("ART_DEDICATED_VLLM_TIMEOUT", 600))
        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                if self._vllm_process.poll() is not None:
                    raise RuntimeError(
                        "vLLM subprocess exited with code "
                        f"{self._vllm_process.returncode}. "
                        f"Check logs at {log_dir}/vllm-dedicated.log"
                    )
                try:
                    response = await client.get(
                        f"{self._vllm_base_url}/v1/models",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        break
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(1.0)
                elapsed += 1.0
            else:
                self._stop_vllm_subprocess()
                raise TimeoutError(
                    f"vLLM subprocess did not become ready within {timeout}s. "
                    f"Check logs at {log_dir}/vllm-dedicated.log"
                )

        atexit.register(self.close)
        logger.info("vLLM subprocess ready on port %d (GPUs: %s)", port, cuda_devices)
        return self._vllm_host, self._vllm_port

    async def _reload_adapter(self, checkpoint_path: str, step: int) -> None:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": f"{self.model_name}@{step}",
                    "lora_path": checkpoint_path,
                    "load_inplace": True,
                },
                timeout=60.0,
            )
            response.raise_for_status()
        self._latest_step = step

    async def _sync_dedicated_merged_weights(
        self,
        *,
        lora_path: str,
        step: int,
    ) -> None:
        await self._ensure_megatron_running()
        await self._init_merged_weight_transfer()
        job_path, log_path = self._create_megatron_job_paths()
        job = MegatronSyncJob(
            lora_path=lora_path,
            merged_weight_transfer=self._build_merged_weight_transfer_spec(step),
            log_path=log_path,
        )
        write_megatron_job(job, job_path=job_path)
        async for _ in stream_megatron_job(job, job_path=job_path):
            pass
        self._latest_step = step

    def _stop_vllm_subprocess(self) -> None:
        if self._vllm_process is not None:
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
        self._merged_weight_transfer_init_info = None

    def _stop_megatron_process(self) -> None:
        if self._megatron_process is None:
            return
        if self._megatron_process.returncode is None:
            os.killpg(os.getpgid(self._megatron_process.pid), signal.SIGTERM)
        self._megatron_process = None

    async def _add_lora_aliases(
        self, llm: AsyncLLM, step: int, checkpoint_dir: str
    ) -> None:
        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{step}",
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(f"Failed to add LoRA adapter for step {step}")
        self._latest_step = step

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        if self.is_dedicated:
            if self.rollout_weights_mode == "merged":
                await self._set_served_model_name(step)
            else:
                await self._reload_adapter(checkpoint_dir, step)
            return
        llm = await self.llm
        await llm.pause_generation()
        await self._add_lora_aliases(llm, step, checkpoint_dir)
        await llm.resume_generation()

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge  # type: ignore

            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        train_script = Path(__file__).parent / "train.py"
        project_root = Path(__file__).resolve().parents[3]
        jobs_dir, _training_log_dir, wake_lock_path = self._megatron_runtime_paths()
        launch_env = os.environ.copy()
        if self.is_dedicated:
            trainer_gpu_ids = self.config["trainer_gpu_ids"]
            num_gpus = len(trainer_gpu_ids)
            launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in trainer_gpu_ids
            )
        else:
            num_gpus = torch.cuda.device_count()
        launch_env["MODEL_IDENTIFIER"] = self.base_model
        launch_env["ART_MEGATRON_JOBS_DIR"] = jobs_dir
        launch_env["ART_MEGATRON_WAKE_LOCK_PATH"] = wake_lock_path
        master_addr = launch_env.get("MASTER_ADDR", "127.0.0.1")
        master_port = str(self._allocate_master_port())
        launch_env["MASTER_ADDR"] = master_addr
        launch_env["MASTER_PORT"] = master_port
        random_state = self._megatron_random_state()
        if random_state is not None:
            launch_env["ART_MEGATRON_RANDOM_STATE"] = str(random_state)

        command = (
            f"{setup_cmd}uv run --project {shlex.quote(str(project_root))} "
            f"torchrun --master-addr {shlex.quote(master_addr)} "
            f"--master-port {shlex.quote(master_port)} "
            f"--nproc_per_node {num_gpus} {shlex.quote(str(train_script))}"
        )
        self._megatron_process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(project_root),
            env=launch_env,
            start_new_session=True,
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
        return self._resolve_active_lora_path()

    async def _prepare_for_training(self) -> tuple[AsyncLLM, str]:
        llm = await self.llm
        await llm.pause_generation()
        await llm.reset_prefix_cache()
        await run_on_workers(llm, do_sleep, level=2)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        await self._ensure_megatron_running()
        lora_path = self._resolve_training_lora_path()
        self._clear_pending_jobs()
        return llm, lora_path

    async def _publish_training_checkpoint(
        self,
        *,
        llm: AsyncLLM,
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
            await run_on_workers(llm, do_wake_up)
            self._is_sleeping = False
        finally:
            if os.path.exists(wake_lock_path):
                os.remove(wake_lock_path)

        await self._add_lora_aliases(llm, next_step, new_checkpoint_dir)
        await llm.resume_generation()

    async def _publish_dedicated_training_checkpoint(self, *, lora_path: str) -> None:
        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)
        await self._reload_adapter(new_checkpoint_dir, next_step)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = self._resolve_active_lora_path()

        if self.is_dedicated:
            port = (config or {}).get("server_args", {}).get("port", 8000)
            location = await self._start_vllm_subprocess(lora_path, port, config)
            if self.rollout_weights_mode == "merged":
                self._clear_pending_jobs()
                await self._sync_dedicated_merged_weights(
                    lora_path=lora_path,
                    step=self._latest_step,
                )
            return location

        lora_path_for_server = (
            lora_path if self._adapter_has_weights(lora_path) else None
        )
        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path_for_server,
            config=config,
        )
        await openai_server_task(engine=await self.llm, config=server_config)
        return (
            server_config.get("server_args", {}).get("host") or "0.0.0.0",
            server_config.get("server_args", {}).get("port", 8000),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        if self.is_dedicated:
            return False
        return self._is_sleeping

    async def aclose(self) -> None:
        self.close()

    def close(self) -> None:
        self._stop_vllm_subprocess()
        self._stop_megatron_process()

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if self.is_dedicated:
            await self._ensure_megatron_running()

            lora_path = self._resolve_active_lora_path()
            self._clear_pending_jobs()
            if _config.get("moe_routing_replay_bundle") is not None:
                raise RuntimeError(
                    "moe_routing_replay_bundle is only supported for in-process/runtime APIs; "
                    "MegatronService subprocess jobs must use moe_routing_replay_path."
                )
            job_path, log_path = self._create_megatron_job_paths()
            next_step = self._latest_step + 1
            if self.rollout_weights_mode == "merged":
                await self._init_merged_weight_transfer()
                job = MegatronMergedTrainJob(
                    lora_path=lora_path,
                    optimizer_state_path=self._get_optimizer_state_path("rl"),
                    disk_packed_tensors=disk_packed_tensors,
                    config=config,
                    experimental_config=cast(dict[str, Any], _config),
                    moe_routing_replay_path=_config.get("moe_routing_replay_path"),
                    moe_routing_replay_strict=_config.get(
                        "moe_routing_replay_strict", True
                    ),
                    merged_weight_transfer=self._build_merged_weight_transfer_spec(
                        next_step
                    ),
                    log_path=log_path,
                )
            else:
                job = MegatronTrainingJob(
                    lora_path=lora_path,
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

            async for result in stream_megatron_job(job, job_path=job_path):
                yield {key: float(value) for key, value in result.items()}

            if self.rollout_weights_mode == "merged":
                new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
                os.makedirs(new_checkpoint_dir, exist_ok=True)
                shutil.copy(
                    f"{lora_path}/adapter_model.safetensors",
                    f"{new_checkpoint_dir}/adapter_model.safetensors",
                )
                self._ensure_lora_adapter_config(
                    new_checkpoint_dir,
                    source_path=lora_path,
                )
                self._latest_step = next_step
            else:
                await self._publish_dedicated_training_checkpoint(lora_path=lora_path)
            return
        llm, lora_path = await self._prepare_for_training()
        if _config.get("moe_routing_replay_bundle") is not None:
            raise RuntimeError(
                "moe_routing_replay_bundle is only supported for in-process/runtime APIs; "
                "MegatronService subprocess jobs must use moe_routing_replay_path."
            )
        job_path, log_path = self._create_megatron_job_paths()
        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._get_optimizer_state_path("rl"),
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=cast(dict[str, Any], _config),
            moe_routing_replay_path=_config.get("moe_routing_replay_path"),
            moe_routing_replay_strict=_config.get("moe_routing_replay_strict", True),
            log_path=log_path,
        )
        write_megatron_job(job, job_path=job_path)

        async for result in stream_megatron_job(job, job_path=job_path):
            yield {key: float(value) for key, value in result.items()}

        await self._publish_training_checkpoint(llm=llm, lora_path=lora_path)

    async def train_sft(
        self,
        batches: list[SFTBatch],
        config: types.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if self.is_dedicated:
            raise NotImplementedError(
                "SFT training is not supported for dedicated MegatronService"
            )
        llm, lora_path = await self._prepare_for_training()
        serialized_batches = materialize_sft_batches(batches)
        job_path, log_path = self._create_megatron_job_paths()
        grad_accumulation_sequences = (
            config.batch_size if isinstance(config.batch_size, int) else None
        )
        job = MegatronSFTTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._get_optimizer_state_path("sft"),
            sft_data_dir=serialized_batches.sft_data_dir,
            num_batches=serialized_batches.num_batches,
            learning_rates=serialized_batches.learning_rates,
            grad_accumulation_sequences=grad_accumulation_sequences,
            log_path=log_path,
        )
        write_megatron_job(job, job_path=job_path)

        async for result in stream_megatron_job(job, job_path=job_path):
            yield {
                "loss/train": float(result["loss"]),
                "loss/learning_rate": float(result["learning_rate"]),
                "loss/grad_norm": float(result["grad_norm"]),
            }

        await self._publish_training_checkpoint(llm=llm, lora_path=lora_path)

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
        }
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))  # type: ignore
