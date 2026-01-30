import asyncio
import datetime
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import AsyncIterator

import torch
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..unsloth.service import do_sleep, do_wake_up, gc_and_empty_cuda_cache
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, openai_server_task, run_on_workers


class MegatronTrainingJob(BaseModel):
    """Job format for communication with train.py"""

    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig


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
    _optimizer_state_path: str | None = None

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge  # noqa: F401

            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        train_script = Path(__file__).parent / "train.py"
        num_gpus = torch.cuda.device_count()
        os.environ["MODEL_IDENTIFIER"] = self.base_model

        command = f"{setup_cmd}uv run torchrun --nproc_per_node {num_gpus} {train_script}"
        self._megatron_process = await asyncio.create_subprocess_shell(command)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(lora_path, exist_ok=True)
            save_file({}, f"{lora_path}/adapter_model.safetensors")
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)

        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path,
            config=config,
        )
        await openai_server_task(engine=await self.llm, config=server_config)
        return (
            server_config.get("server_args", {}).get("host") or "0.0.0.0",
            server_config.get("server_args", {}).get("port", 8000),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        await self._ensure_megatron_running()

        llm = await self.llm
        await llm.pause_generation()
        await llm.reset_prefix_cache()
        await run_on_workers(llm, do_sleep, level=2)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)

        if self._optimizer_state_path is None:
            self._optimizer_state_path = f"/tmp/megatron_optimizer_states/{uuid.uuid4().hex}"

        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=_config,
        )
        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())

        num_lines = 0
        while True:
            await asyncio.sleep(0.1)
            try:
                with open("/tmp/megatron_training_log.jsonl", "a+") as log_file:
                    log_file.seek(0)
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                self._merge_lora_adapter(lora_path)
                                os.remove("/tmp/megatron_training_log.jsonl")
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )

        await run_on_workers(llm, do_wake_up)
        self._is_sleeping = False

        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{next_step}",
                lora_int_id=self._next_lora_id(),
                lora_path=new_checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(f"Failed to add LoRA adapter for step {next_step}")
        self._latest_step = next_step
        await llm.resume_generation()

    def _merge_lora_adapter(self, lora_path: str) -> None:
        """Merge sharded LoRA adapters from distributed training."""
        base_dir = Path(lora_path)
        shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
        if not shard_filenames:
            return

        adapter_model_path = base_dir / "adapter_model.safetensors"
        sharded_tensors: dict[str, list[torch.Tensor]] = {}

        for filename in shard_filenames:
            with safe_open(filename, framework="pt") as file:
                for key in file.keys():
                    tensor = file.get_tensor(key)
                    sharded_tensors.setdefault(key, []).append(tensor)

        adapter_model: dict[str, torch.Tensor] = {}
        if adapter_model_path.exists():
            adapter_model = load_file(adapter_model_path)

        for key, tensors in sharded_tensors.items():
            tensor = torch.cat(tensors, dim=1 if "lora_A" in key else 0)
            adapter_model[key] = tensor

        save_file(adapter_model, adapter_model_path)
        for filename in shard_filenames:
            filename.unlink()

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
        }
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))
