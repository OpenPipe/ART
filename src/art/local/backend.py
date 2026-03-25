import asyncio
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import time
from types import TracebackType
from typing import AsyncIterator, Iterable, Literal, cast
import warnings

logger = logging.getLogger(__name__)

_AUTO_GPU_HOURLY_PRICING_USD = {
    "H200": 3.0,
}

import aiohttp
import numpy as np
from openai import AsyncOpenAI
import polars as pl
import torch
from tqdm import auto as tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import Self

from art.utils.output_dirs import (
    get_default_art_path,
    get_model_dir,
    get_output_dir_from_model_properties,
    get_step_checkpoint_dir,
)
from art.utils.record_provenance import record_provenance
from art.utils.s3 import (
    ExcludableOption,
    pull_model_from_s3,
    push_model_to_s3,
)
from mp_actors import close_proxy, move_to_child_process

from .. import dev
from ..backend import AnyTrainableModel, Backend
from ..costs import build_cost_calculator, get_model_pricing
from ..metrics_taxonomy import (
    TRAIN_GRADIENT_STEPS_KEY,
    average_metric_samples,
    build_training_summary_metrics,
    summarize_trajectory_groups,
)
from ..model import Model, TrainableModel
from ..preprocessing.pack import (
    PackedTensors,
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
    plot_packed_tensors,
)
from ..preprocessing.tokenize import (
    tokenize_sft_batch,
    tokenize_trajectory_groups,
)
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import LocalTrainResult, Message, TrainConfig, TrainSFTConfig
from ..utils import format_message, get_model_step
from .checkpoints import (
    delete_checkpoints,
)
from .service import ModelService


class LocalBackend(Backend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        gpu_cost_per_hour_usd: float | None = None,
    ) -> None:
        """
        Initializes a local, directory-based Backend interface at the given path.

        Note:
            The local Backend uses Weights & Biases for training monitoring.
            If you don't have a W&B account, you can create one at https://wandb.ai.

        Args:
            in_process: Whether to run the local service in-process.
            path: The path to the local directory. Defaults to "{repo_root}/.art".
            gpu_cost_per_hour_usd: Optional per-GPU hourly price override used for
                automatic `costs/gpu` accounting on train steps. When unset,
                ART auto-detects supported GPU types (H200 at $3/hr today) and
                skips GPU cost logging for unknown devices instead of guessing.
        """
        self._in_process = in_process
        self._path = path or get_default_art_path()
        self._gpu_cost_per_hour_usd = (
            float(gpu_cost_per_hour_usd) if gpu_cost_per_hour_usd is not None else None
        )
        os.makedirs(self._path, exist_ok=True)

        # Other initialization
        self._services: dict[str, ModelService] = {}
        self._tokenizers: dict[str, PreTrainedTokenizerBase] = {}
        self._image_processors: dict[str, BaseImageProcessor | None] = {}

    def supports_automatic_train_step_metrics(self) -> bool:
        return True

    def automatic_gpu_cost_per_hour_usd(self, model: Model) -> float | None:
        per_gpu_cost = self._resolve_gpu_cost_per_hour_usd()
        if per_gpu_cost is None:
            return None

        gpu_count = self._allocated_gpu_count(model)
        if gpu_count <= 0:
            return None
        return per_gpu_cost * gpu_count

    def _resolve_gpu_cost_per_hour_usd(self) -> float | None:
        if self._gpu_cost_per_hour_usd is not None:
            return self._gpu_cost_per_hour_usd
        if not torch.cuda.is_available():
            return None

        num_visible_gpus = torch.cuda.device_count()
        if num_visible_gpus <= 0:
            return None

        resolved_costs: list[float] = []
        for index in range(num_visible_gpus):
            device_name = torch.cuda.get_device_name(index).upper()
            for gpu_name, hourly_cost in _AUTO_GPU_HOURLY_PRICING_USD.items():
                if gpu_name in device_name:
                    resolved_costs.append(hourly_cost)
                    break
            else:
                return None

        if not resolved_costs:
            return None
        if len(set(resolved_costs)) != 1:
            return None
        return resolved_costs[0]

    def _allocated_gpu_count(self, model: Model) -> int:
        if isinstance(model, TrainableModel) and model._internal_config is not None:
            trainer_gpu_ids = set(model._internal_config.get("trainer_gpu_ids", []))
            inference_gpu_ids = set(model._internal_config.get("inference_gpu_ids", []))
            allocated_gpu_ids = trainer_gpu_ids | inference_gpu_ids
            if allocated_gpu_ids:
                return len(allocated_gpu_ids)

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    def __enter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._close()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """
        If running vLLM in a separate process, this will kill that process and close the communication threads.
        """
        for service in self._services.values():
            aclose = getattr(service, "aclose", None)
            if aclose is None:
                close = getattr(service, "close", None)
                if close is not None:
                    close()
            else:
                await aclose()
            close_proxy(service)

    def _close(self) -> None:
        for service in self._services.values():
            close = getattr(service, "close", None)
            if close is not None:
                close()
            close_proxy(service)

    async def register(
        self,
        model: Model,
    ) -> None:
        """
        Registers a model with the local Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        # Ensure model state/logging uses the backend path
        model.base_path = self._path
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            json.dump(model.model_dump(), f)

        # Auto-migrate any old JSONL trajectory files to Parquet
        from art.utils.trajectory_migration import auto_migrate_on_register

        auto_migrate_on_register(output_dir)

        # Initialize wandb early if this is a trainable model
        # (wandb initialization is now handled by the model's _get_wandb_run method)
        if model.trainable and "WANDB_API_KEY" in os.environ:
            _ = model._get_wandb_run()
        if model.trainable:
            trainable_model = cast(TrainableModel, model)
            pricing = get_model_pricing(trainable_model.base_model)
            if pricing is not None:
                trainable_model.set_cost_calculator(build_cost_calculator(pricing))

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        """Return the inference name for a model checkpoint.

        For LocalBackend with vLLM, the base model is served under its HF name,
        and LoRA adapters are served as `model.name@step`.

        Args:
            model: The model.
            step: If provided, returns name for specific checkpoint.
                  If None, returns name for latest checkpoint (step 0 initially).
        """

        requested_step = step

        if step is None and isinstance(model, TrainableModel):
            from ..dev.validate import is_dedicated_mode

            service = self._services.get(model.name)
            if service is not None and is_dedicated_mode(
                model._internal_config or dev.InternalModelConfig()
            ):
                loaded_step = getattr(service, "_latest_step", None)
                if isinstance(loaded_step, int):
                    step = loaded_step

        if step is None:
            # The checkpoint directory is written before dedicated-mode
            # vLLM finishes reloading the new adapter.
            step = self.__get_step(model)
        name = f"{model.name}@{step}"
        logger.debug(
            f"[BACKEND] _model_inference_name: step_arg={requested_step} "
            f"actual_step={step} -> {name}"
        )
        return name

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from ..dev.validate import is_dedicated_mode, validate_dedicated_config

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            validate_dedicated_config(config)
            dedicated = is_dedicated_mode(config)

            is_tinker = config.get("tinker_args") is not None
            if is_tinker:
                from ..tinker.service import TinkerService

                service_class = TinkerService
            else:
                from ..unsloth.service import UnslothService

                service_class = UnslothService
                # When moving the service to a child process, import unsloth
                # early to maximize optimizations
                os.environ["IMPORT_UNSLOTH"] = "1"

            if dedicated:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(g) for g in config["trainer_gpu_ids"]
                )

            self._services[model.name] = service_class(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not dedicated and not self._in_process:
                # Kill all "model-service" processes to free up GPU memory
                subprocess.run(["pkill", "-9", "model-service"])
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="tinker-service" if is_tinker else "model-service",
                )
        return self._services[model.name]

    def _get_packed_tensors(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        advantage_balance: float,
        allow_training_without_logprobs: bool,
        scale_rewards: bool,
        plot_tensors: bool,
    ) -> PackedTensors | None:
        if model.base_model not in self._tokenizers:
            self._tokenizers[model.base_model] = AutoTokenizer.from_pretrained(
                model.base_model
            )
        if model.base_model not in self._image_processors:
            try:
                self._image_processors[model.base_model] = (
                    AutoImageProcessor.from_pretrained(model.base_model, use_fast=True)
                )
            except Exception:
                self._image_processors[model.base_model] = None
        tokenizer = self._tokenizers[model.base_model]
        tokenized_results = list(
            tokenize_trajectory_groups(
                tokenizer,
                trajectory_groups,
                allow_training_without_logprobs,
                scale_rewards,
                image_processor=self._image_processors[model.base_model],
            )
        )
        if not tokenized_results:
            return None
        max_tokens = max(len(result.token_ids) for result in tokenized_results)
        # Round up max_tokens to the nearest multiple of 2048
        sequence_length = math.ceil(max_tokens / 2048) * 2048
        # Cap sequence length at the model's max sequence length
        sequence_length = min(
            sequence_length,
            (model._internal_config or dev.InternalModelConfig())
            .get("init_args", {})
            .get("max_seq_length", 32_768),
        )
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            sequence_length,
            pad_token_id=tokenizer.eos_token_id,
            advantage_balance=advantage_balance,
        )
        if (
            not allow_training_without_logprobs
            and np.isnan(packed_tensors["logprobs"]).all()
        ):
            print(
                "There are no assistant logprobs to train on. Did you forget to include at least one Choice in Trajectory.messages_and_choices?"
            )
            return None
        if plot_tensors:
            plot_packed_tensors(
                packed_tensors, get_model_dir(model=model, art_path=self._path)
            )
        else:
            print(
                f"Packed {len(tokenized_results)} trajectories into {packed_tensors['tokens'].shape[0]} sequences of length {packed_tensors['tokens'].shape[1]}"
            )
        return packed_tensors

    async def _get_step(self, model: AnyTrainableModel) -> int:
        return self.__get_step(model)

    def __get_step(self, model: Model) -> int:
        if model.trainable:
            model = cast(TrainableModel, model)
            return get_model_step(model, self._path)
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        """Delete checkpoint files, keeping only the specified steps."""

        output_dir = get_model_dir(model=model, art_path=self._path)
        service = await self._get_service(model)
        try:
            from ..tinker.service import TinkerService

            if isinstance(service, TinkerService):
                await service.delete_checkpoints(steps_to_keep)
                return
        except ImportError:
            pass
        delete_checkpoints(output_dir, steps_to_keep)

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        config_dict: dict = dict(config or {})
        server_args = dict(config_dict.get("server_args", {}))

        # Avoid binding collisions on busy hosts when no explicit port is provided.
        if "port" not in server_args:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                server_args["port"] = s.getsockname()[1]
        config_dict["server_args"] = server_args
        resolved_config = cast(dev.OpenAIServerConfig, config_dict)

        service = await self._get_service(model)
        host, port = await service.start_openai_server(config=resolved_config)

        base_url = f"http://{host}:{port}/v1"
        api_key = server_args.get("api_key") or "default"

        def done_callback(_: asyncio.Task[None]) -> None:
            close_proxy(self._services.pop(model.name))

        asyncio.create_task(
            self._monitor_openai_server(model, base_url, api_key)
        ).add_done_callback(done_callback)

        return base_url, api_key

    async def _monitor_openai_server(
        self, model: AnyTrainableModel, base_url: str, api_key: str
    ) -> None:
        model_name = model.name
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        consecutive_failures = 0
        max_consecutive_failures = 3
        async with aiohttp.ClientSession() as session:
            while True:
                # Wait 30 seconds before checking again
                await asyncio.sleep(30)
                try:
                    # If the server is sleeping, skip the check
                    if await self._services[model_name].vllm_engine_is_sleeping():
                        consecutive_failures = 0
                        continue
                    # Check the metrics with a timeout
                    async with session.get(
                        f"{base_url.split('/v1')[0]}/metrics",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        metrics = await response.text()
                    # Parse Prometheus metrics for running requests
                    running_requests = 0
                    pending_requests = 0
                    for line in metrics.split("\n"):
                        if line.startswith("vllm:num_requests_running"):
                            running_requests = int(float(line.split()[1]))
                        elif line.startswith("vllm:num_requests_waiting"):
                            pending_requests = int(float(line.split()[1]))
                    # If there are no running or pending requests, send a health check
                    if running_requests == 0 and pending_requests == 0:
                        try:
                            # Send a health check with a short timeout
                            await openai_client.completions.create(
                                model=self._model_inference_name(model),
                                prompt="Hi",
                                max_tokens=1,
                                timeout=float(
                                    os.environ.get("ART_SERVER_MONITOR_TIMEOUT", 5.0)
                                ),
                            )
                        except Exception as e:
                            # If the server is sleeping, a failed health check is okay
                            if await self._services[
                                model_name
                            ].vllm_engine_is_sleeping():
                                consecutive_failures = 0
                                continue
                            raise e
                    # Reset failure counter on success
                    consecutive_failures = 0
                except Exception:
                    # If the server is sleeping during an exception, it's okay
                    try:
                        if await self._services[model_name].vllm_engine_is_sleeping():
                            consecutive_failures = 0
                            continue
                    except Exception:
                        pass  # If we can't check sleeping status, count it as a failure
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise
                    # Otherwise, continue and try again

    # Note: _log() method has been moved to the Model class (frontend)

    def _trajectory_log(self, trajectory: Trajectory) -> str:
        """Format a trajectory into a readable log string."""
        header = f"reward: {trajectory.reward} {' '.join(f'{k}: {v}' for k, v in trajectory.metrics.items())}\n\n"
        formatted_messages = []
        for message_or_choice in trajectory.messages_and_choices:
            if isinstance(message_or_choice, dict):
                message = message_or_choice
            else:
                message = cast(Message, message_or_choice.message.model_dump())  # ty:ignore[possibly-missing-attribute]
            formatted_messages.append(format_message(message))
        return header + "\n".join(formatted_messages)

    async def train(  # type: ignore[override]
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training parameters
        learning_rate: float = 5e-6,
        loss_fn: Literal["cispo", "ppo"] = "cispo",
        loss_fn_config: dict | None = None,
        normalize_advantages: bool = True,
        adam_params: object | None = None,
        # KL-penalized advantage adjustment
        kl_penalty_coef: float = 0.0,
        kl_penalty_reference_step: int | None = None,
        kl_ref_adapter_path: str | None = None,
        kl_penalty_source: Literal["current_learner", "sample"] = "current_learner",
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        # Advantage computation
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        # Importance sampling
        importance_sampling_level: Literal[
            "token", "sequence", "average", "geometric_average"
        ] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        # Experimental parameters
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
        # LocalBackend-specific parameters
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        truncated_importance_sampling: float | None = None,
        scale_learning_rate_by_reward_std_dev: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        num_trajectories_learning_rate_multiplier_power: float = 0.0,
        # Checkpoint behavior
        save_checkpoint: bool = True,
        # Verbosity
        verbose: bool = False,
    ) -> LocalTrainResult:
        """Train the model on the given trajectory groups.

        This is the recommended way to train models. Unlike model.train(), this
        method does NOT automatically log trajectories or metrics. Call model.log()
        explicitly before and/or after training if you want to log data.

        Args:
            model: The trainable model to train.
            trajectory_groups: Batches of trajectories to train on.
            learning_rate: Learning rate for training. Defaults to 5e-6.
            loss_fn: RL loss function. LocalBackend currently supports
                "cispo" and "ppo".
            loss_fn_config: Additional loss-function config. Not supported by
                LocalBackend.
            normalize_advantages: Whether to normalize advantages. LocalBackend
                currently requires True.
            adam_params: Custom optimizer params. Not supported by
                LocalBackend.
            kl_penalty_coef: Coefficient for KL-penalized advantage adjustment.
                Tokens diverging more from the reference get reduced advantages.
                Defaults to 0.0 (disabled).
            kl_penalty_reference_step: Checkpoint step of the training model to
                use as the KL reference. If None, uses the base model (LoRA
                disabled) as reference.
            kl_ref_adapter_path: Direct filesystem path to a LoRA adapter
                checkpoint to use as the KL reference. Alternative to
                kl_penalty_reference_step.
            kl_penalty_source: Which policy's logprobs to compare against the
                reference when building the centered KL penalty. Use
                "current_learner" to match the original ART implementation, or
                "sample" to shape from the rollout policy logprobs, which is
                usually better for async/off-policy workloads.
            epsilon: Clip epsilon for importance sampling. Defaults based on loss_fn.
            epsilon_high: Asymmetric upper clip bound. Defaults to epsilon.
            advantage_balance: Balance between negative and positive advantages
                in range [-1.0, 1.0]. Defaults to 0.0 (balanced).
            scale_rewards: Whether to scale rewards by standard deviation.
                Defaults to True.
            importance_sampling_level: Level at which to compute importance
                sampling weights. Defaults to "token".
            max_negative_advantage_importance_sampling_weight: Maximum weight
                for negative advantage samples.
            mask_prob_ratio: Whether to mask probability ratios. Defaults to False.
            kimi_k2_tau: Tau parameter for Kimi K2 algorithm.
            precalculate_logprobs: Whether to precalculate logprobs.
            allow_training_without_logprobs: Allow training even when no logprobs
                are available. Defaults to False.
            plot_tensors: Whether to plot training tensors for debugging.
                Defaults to False.
            truncated_importance_sampling: Truncation threshold for importance
                sampling weights.
            scale_learning_rate_by_reward_std_dev: Whether to scale learning rate
                by reward standard deviation. Defaults to False.
            logprob_calculation_chunk_size: Chunk size for logprob calculation.
                Defaults to 1024.
            num_trajectories_learning_rate_multiplier_power: Power for learning
                rate multiplier based on number of trajectories.
            save_checkpoint: Whether to save a checkpoint after training.
                Defaults to True.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            LocalTrainResult with step number, training metrics, and checkpoint path.

        Example:
            # Before (deprecated):
            await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))

            # After (recommended):
            await model.log(trajectory_groups, split="train")
            result = await backend.train(model, trajectory_groups, learning_rate=5e-6)
            # Optionally log training metrics:
            # await model.log(metrics=result.metrics, step=result.step)
        """
        groups_list = list(trajectory_groups)
        if loss_fn not in {"cispo", "ppo"}:
            raise ValueError("LocalBackend only supports loss_fn='cispo' or 'ppo'.")
        if loss_fn_config is not None:
            raise ValueError("LocalBackend requires loss_fn_config=None.")
        if not normalize_advantages:
            raise ValueError("LocalBackend requires normalize_advantages=True.")
        if adam_params is not None:
            raise ValueError("LocalBackend requires adam_params=None.")
        assert kl_penalty_source in {"current_learner", "sample"}

        # Build config objects from explicit kwargs
        config = TrainConfig(
            learning_rate=learning_rate,
            kl_penalty_coef=kl_penalty_coef,
            kl_penalty_source=kl_penalty_source,
        )
        dev_config: dev.TrainConfig = {
            "advantage_balance": advantage_balance,
            "allow_training_without_logprobs": allow_training_without_logprobs,
            "importance_sampling_level": importance_sampling_level,
            "kl_penalty_coef": kl_penalty_coef,
            "kl_penalty_source": kl_penalty_source,
            "mask_prob_ratio": mask_prob_ratio,
            "plot_tensors": plot_tensors,
            "ppo": loss_fn == "ppo",
            "precalculate_logprobs": precalculate_logprobs,
            "scale_learning_rate_by_reward_std_dev": scale_learning_rate_by_reward_std_dev,
            "scale_rewards": scale_rewards,
            "logprob_calculation_chunk_size": logprob_calculation_chunk_size,
            "num_trajectories_learning_rate_multiplier_power": num_trajectories_learning_rate_multiplier_power,
        }
        # Only include optional fields if they're set
        if epsilon is not None:
            dev_config["epsilon"] = epsilon
        if epsilon_high is not None:
            dev_config["epsilon_high"] = epsilon_high
        if max_negative_advantage_importance_sampling_weight is not None:
            dev_config["max_negative_advantage_importance_sampling_weight"] = (
                max_negative_advantage_importance_sampling_weight
            )
        if kimi_k2_tau is not None:
            dev_config["kimi_k2_tau"] = kimi_k2_tau
        if truncated_importance_sampling is not None:
            dev_config["truncated_importance_sampling"] = truncated_importance_sampling
        if kl_ref_adapter_path is not None:
            dev_config["kl_ref_adapter_path"] = kl_ref_adapter_path
        elif kl_penalty_reference_step is not None:
            ref_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path),
                kl_penalty_reference_step,
            )
            dev_config["kl_ref_adapter_path"] = ref_checkpoint_dir

        # Collect metrics from training
        training_metrics: list[dict[str, float]] = []
        trainer_started = time.monotonic()
        async for metrics in self._train_model(
            model, groups_list, config, dev_config, verbose
        ):
            training_metrics.append(metrics)

        # Aggregate metrics
        avg_metrics = average_metric_samples(training_metrics)
        summary = summarize_trajectory_groups(groups_list)
        avg_metrics.setdefault(
            "time/step_trainer_s", time.monotonic() - trainer_started
        )
        avg_metrics.update(
            {
                key: value
                for key, value in build_training_summary_metrics(
                    summary,
                    include_trainable_groups=True,
                ).items()
                if key not in avg_metrics
            }
        )

        # Get step and checkpoint path
        step = await self._get_step(model)
        checkpoint_path: str | None = None
        if save_checkpoint:
            checkpoint_path = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), step
            )
            if not os.path.exists(checkpoint_path):
                checkpoint_path = None

        # Record provenance on the latest W&B artifact
        wandb_run = model._get_wandb_run()
        if wandb_run is not None:
            record_provenance(wandb_run, "local-rl")

        return LocalTrainResult(
            step=step,
            metrics=avg_metrics,
            checkpoint_path=checkpoint_path,
        )

    async def _train_model(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if verbose:
            print("Starting _train_model")
        service = await self._get_service(model)
        # Note: Logging is now handled by the frontend (Model.train() calls Model.log())
        if verbose:
            print("Packing tensors...")

        summary = summarize_trajectory_groups(trajectory_groups)
        base_metrics = build_training_summary_metrics(
            summary,
            include_trainable_groups=True,
        )

        packed_tensors = self._get_packed_tensors(
            model,
            trajectory_groups,
            advantage_balance=dev_config.get("advantage_balance", 0.0),
            allow_training_without_logprobs=dev_config.get(
                "allow_training_without_logprobs", False
            ),
            scale_rewards=dev_config.get("scale_rewards", True),
            plot_tensors=dev_config.get("plot_tensors", False),
        )
        if packed_tensors is None:
            print(
                "Skipping tuning as there is no suitable data. "
                "This can happen when all the trajectories in the same group "
                "have the same reward and thus no advantage to train on."
            )

            # Still advance the step by renaming the checkpoint directory
            current_step = self.__get_step(model)
            next_step = current_step + 1
            logger.info(
                f"[BACKEND] _train_model SKIP: current_step={current_step} "
                f"next_step={next_step} (all rewards equal)"
            )
            current_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), current_step
            )
            next_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), next_step
            )

            # If the current checkpoint exists, copy it to the next step
            if os.path.exists(current_checkpoint_dir):
                shutil.copytree(
                    current_checkpoint_dir,
                    next_checkpoint_dir,
                    dirs_exist_ok=True,
                )
                logger.info(
                    f"[BACKEND] _train_model SKIP: copied checkpoint "
                    f"{current_step} -> {next_step}, calling register_lora_for_step..."
                )

                try:
                    # Register the copied checkpoint as a new LoRA adapter
                    # so it's available for inference at the new step
                    if hasattr(service, "register_lora_for_step"):
                        await service.register_lora_for_step(  # type: ignore[attr-defined]
                            next_step, next_checkpoint_dir
                        )
                    logger.info(
                        f"[BACKEND] _train_model SKIP: register_lora_for_step "
                        f"completed for step {next_step}"
                    )
                except ModuleNotFoundError:
                    pass  # Unsloth is not installed

            # Yield metrics showing no groups were trainable
            # (the frontend will handle logging)
            yield {
                **base_metrics,
                "data/step_num_groups_trainable": 0.0,
                "data/step_trainer_tokens": 0.0,
                TRAIN_GRADIENT_STEPS_KEY: 0.0,
            }
            return
        base_metrics["data/step_trainer_tokens"] = float(
            packed_tensors["assistant_mask"].sum().item()
        )
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors, f"{get_model_dir(model=model, art_path=self._path)}/tensors"
        )
        # Note: scale_learning_rate_by_reward_std_dev is now handled by the frontend (Model.train())
        estimated_gradient_steps = disk_packed_tensors["num_sequences"]
        pbar = tqdm.tqdm(total=estimated_gradient_steps, desc="train")
        async for result in service.train(
            disk_packed_tensors, config, dev_config, verbose
        ):
            num_gradient_steps = int(
                result.pop(TRAIN_GRADIENT_STEPS_KEY, estimated_gradient_steps)
            )
            assert num_gradient_steps == estimated_gradient_steps, (
                f"num_gradient_steps {num_gradient_steps} != estimated_gradient_steps {estimated_gradient_steps}"
            )
            yield {
                **base_metrics,
                **result,
                TRAIN_GRADIENT_STEPS_KEY: float(num_gradient_steps),
            }
            pbar.update(1)
            pbar.set_postfix(result)
        pbar.close()
        # Note: Metrics logging is now handled by the frontend (Model.train())
        if verbose:
            print("_train_model complete")

    # Note: _get_reward_std_dev_learning_rate_multiplier and _log_metrics
    # have been moved to the Model class (frontend)

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train the model using supervised fine-tuning.

        Args:
            model: The trainable model to fine-tune
            trajectories: Iterable of Trajectory objects
            config: SFT configuration with batch_size and learning rates.
                    If learning_rate is a list, streaming mode is used automatically.
            dev_config: Developer configuration
            verbose: Whether to print detailed logs

        Yields:
            Dictionary containing training metrics for each batch
        """
        if verbose:
            print("Starting _train_sft")

        # Get tokenizer
        if model.base_model not in self._tokenizers:
            self._tokenizers[model.base_model] = AutoTokenizer.from_pretrained(
                model.base_model
            )
        tokenizer = self._tokenizers[model.base_model]

        # Determine batch_size
        batch_size = config.batch_size
        if batch_size == "auto":
            batch_size = 2  # Default to 2 for SFT

        # Auto-detect instruction/response parts from model
        from ..utils.model_config import get_instruction_response_parts

        instruction_part, response_part = get_instruction_response_parts(
            model.base_model, tokenizer
        )

        if verbose:
            print(f"Using instruction_part: {instruction_part!r}")
            print(f"Using response_part: {response_part!r}")

        import itertools
        from typing import Iterator

        from ..preprocessing.tokenize import SFTBatch

        if isinstance(config.learning_rate, list):
            learning_rates_iter: Iterator[float] = iter(config.learning_rate)
        else:
            learning_rates_iter = itertools.repeat(config.learning_rate)

        # Build all batches in memory
        trajectory_list = list(trajectories)
        batches: list[SFTBatch] = []
        for i in range(0, len(trajectory_list), batch_size):
            batch_trajectories = trajectory_list[i : i + batch_size]
            batches.append(
                tokenize_sft_batch(
                    trajectory_batch=batch_trajectories,
                    learning_rate=next(learning_rates_iter),
                    tokenizer=tokenizer,
                    instruction_part=instruction_part,
                    response_part=response_part,
                )
            )

        # Get the service and train
        service = await self._get_service(model)

        pbar = tqdm.tqdm(total=len(batches), desc="sft train")
        total_trainable_tokens = sum(batch.num_trainable_tokens for batch in batches)
        total_trajectories = len(trajectory_list)
        batch_count = 0

        async for result in service.train_sft(batches, verbose):
            pbar.update(1)
            pbar.set_postfix({"loss": f"{result.get('loss/train', 0):.4f}"})
            batch_count += 1
            yield {
                **result,
                "data/step_num_trajectories": float(total_trajectories),
                "data/step_trainer_tokens": float(total_trainable_tokens),
                TRAIN_GRADIENT_STEPS_KEY: float(len(batches)),
            }

        pbar.close()

        if batch_count > 0 and total_trainable_tokens == 0:
            print(
                "WARNING: No trainable tokens found! "
                "Check instruction_part and response_part settings."
            )

        if verbose:
            print("_train_sft complete")

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    async def _experimental_pull_model_checkpoint(
        self,
        model: "TrainableModel",
        *,
        step: int | Literal["latest"] | None = None,
        local_path: str | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Pull a model checkpoint to a local path.

        For LocalBackend, this:
        1. When step is "latest" or None, checks both local storage and S3 (if provided)
           to find the latest checkpoint, preferring local if steps are equal
        2. If checkpoint exists locally, uses it (optionally copying to local_path)
        3. If checkpoint doesn't exist locally but s3_bucket is provided, pulls from S3
        4. Returns the final checkpoint path

        Args:
            model: The model to pull checkpoint for.
            step: The step to pull. Can be an int for a specific step,
                 or "latest" to pull the latest checkpoint. If None, pulls latest.
            local_path: Custom directory to save/copy the checkpoint to.
                       If None, returns checkpoint from backend's default art path.
            s3_bucket: S3 bucket to check/pull from. When step is "latest", both
                       local storage and S3 are checked to find the true latest.
            prefix: S3 prefix.
            verbose: Whether to print verbose output.

        Returns:
            Path to the local checkpoint directory.
        """
        # Determine which step to use
        resolved_step: int
        if step is None or step == "latest":
            # Check both local storage and S3 (if provided) for the latest checkpoint
            local_latest_step: int | None = None
            s3_latest_step: int | None = None

            # Get latest from local storage
            try:
                local_latest_step = get_model_step(model, self._path)
                if local_latest_step == 0:
                    # get_model_step returns 0 if no checkpoints exist
                    local_latest_step = None
            except Exception:
                local_latest_step = None

            # Get latest from S3 if bucket provided
            if s3_bucket is not None:
                from art.utils.s3_checkpoint_utils import (
                    get_latest_checkpoint_step_from_s3,
                )

                s3_latest_step = await get_latest_checkpoint_step_from_s3(
                    model_name=model.name,
                    project=model.project,
                    s3_bucket=s3_bucket,
                    prefix=prefix,
                )

            # Determine which source has the latest checkpoint
            if local_latest_step is None and s3_latest_step is None:
                raise ValueError(
                    f"No checkpoints found for {model.project}/{model.name} in local storage or S3"
                )
            elif local_latest_step is None:
                resolved_step = s3_latest_step  # type: ignore[assignment]
                if verbose:
                    print(f"Using latest checkpoint from S3: step {resolved_step}")
            elif s3_latest_step is None:
                resolved_step = local_latest_step
                if verbose:
                    print(
                        f"Using latest checkpoint from local storage: step {resolved_step}"
                    )
            elif local_latest_step >= s3_latest_step:
                # Prefer local if equal or greater
                resolved_step = local_latest_step
                if verbose:
                    print(
                        f"Using latest checkpoint from local storage: step {resolved_step} "
                    )
            else:
                resolved_step = s3_latest_step
                if verbose:
                    print(f"Using latest checkpoint from S3: step {resolved_step} ")
        else:
            resolved_step = step

        # Check if checkpoint exists in the original training location
        original_checkpoint_dir = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self._path), resolved_step
        )

        # Step 1: Ensure checkpoint exists at original_checkpoint_dir
        if not os.path.exists(original_checkpoint_dir):
            if s3_bucket is None:
                raise FileNotFoundError(
                    f"Checkpoint not found at {original_checkpoint_dir} and no S3 bucket specified"
                )
            if verbose:
                print(f"Pulling checkpoint step {resolved_step} from S3...")
            await pull_model_from_s3(
                model_name=model.name,
                project=model.project,
                step=resolved_step,
                s3_bucket=s3_bucket,
                prefix=prefix,
                verbose=verbose,
                art_path=self._path,
                exclude=["logs", "trajectories"],
            )
            # Validate that the checkpoint was actually downloaded
            if not os.path.exists(original_checkpoint_dir) or not os.listdir(
                original_checkpoint_dir
            ):
                raise FileNotFoundError(f"Checkpoint step {resolved_step} not found")

        # Step 2: Handle local_path if provided
        if local_path is not None:
            if verbose:
                print(
                    f"Copying checkpoint from {original_checkpoint_dir} to {local_path}..."
                )
            import shutil

            os.makedirs(local_path, exist_ok=True)
            shutil.copytree(original_checkpoint_dir, local_path, dirs_exist_ok=True)
            if verbose:
                print(f"✓ Checkpoint copied successfully")
            return local_path

        if verbose:
            print(
                f"Checkpoint step {resolved_step} exists at {original_checkpoint_dir}"
            )
        return original_checkpoint_dir

    async def _experimental_pull_from_s3(
        self,
        model: Model,
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
        # LocalBackend extensions (not part of the base interface)
        step: int | None = None,
        exclude: list[ExcludableOption] | None = None,
        latest_only: bool = False,
    ) -> None:
        """Download the model directory from S3 into local Backend storage. Right now this can be used to pull trajectory logs for processing or model checkpoints.

        .. deprecated::
            This method is deprecated. Use `_experimental_pull_model_checkpoint` instead.

        Args:
            model: The model to pull from S3.
            step: DEPRECATED. Use only_step instead.
            s3_bucket: The S3 bucket to pull from. If None, the default bucket will be used.
            prefix: The prefix to pull from S3. If None, the model name will be used.
            verbose: Whether to print verbose output.
            delete: Whether to delete the local model directory.
            exclude: List of directories to exclude from sync. Valid options: "checkpoints", "logs", "trajectories".
            latest_only: DEPRECATED. Use only_step="latest" instead.
            only_step: If specified, only pull this specific step. Can be an int for a specific step,
                      or "latest" to pull only the latest checkpoint. If None, pulls all steps.
        """
        warnings.warn(
            "_experimental_pull_from_s3 is deprecated. Use _experimental_pull_model_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Handle backward compatibility and new only_step parameter
        if only_step is None and latest_only:
            only_step = "latest"

        # Handle the only_step parameter
        if only_step is not None and step is None:
            if only_step == "latest":
                from art.utils.s3_checkpoint_utils import (
                    get_latest_checkpoint_step_from_s3,
                )

                latest_step = await get_latest_checkpoint_step_from_s3(
                    model_name=model.name,
                    project=model.project,
                    s3_bucket=s3_bucket,
                    prefix=prefix,
                )

                if latest_step is not None:
                    step = latest_step
                    if verbose:
                        print(f"Found latest checkpoint at step {step}")
                else:
                    if verbose:
                        print("No checkpoints found in S3")
                    return
            else:
                # only_step is an int
                step = only_step
                if verbose:
                    print(f"Pulling specific checkpoint at step {step}")

        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
            exclude=exclude,
        )

    async def _experimental_push_to_s3(
        self,
        model: Model,
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Upload the model directory from local storage to S3."""
        await push_model_to_s3(
            model_name=model.name,
            project=model.project,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
        )

    async def _experimental_fork_checkpoint(
        self,
        model: Model,
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        """Fork a checkpoint from another model to initialize this model.

        Args:
            model: The model to fork to.
            from_model: The name of the model to fork from.
            from_project: The project of the model to fork from. Defaults to model.project.
            from_s3_bucket: Optional S3 bucket to pull the checkpoint from. If provided,
                will pull from S3 first. Otherwise, will fork from local disk.
            not_after_step: Optional step number. If provided, will copy the last saved
                checkpoint that is <= this step. Otherwise, copies the latest checkpoint.
            verbose: Whether to print verbose output.
            prefix: Optional S3 prefix for the bucket.
        """
        # Default from_project to model.project if not provided
        from_project = from_project or model.project

        # Get source and destination directories
        source_model_dir = get_output_dir_from_model_properties(
            project=from_project,
            name=from_model,
            art_path=self._path,
        )
        dest_model_dir = get_output_dir_from_model_properties(
            project=model.project,
            name=model.name,
            art_path=self._path,
        )

        # If S3 bucket is provided, pull from S3 first
        if from_s3_bucket is not None:
            if verbose:
                print(
                    f"DEBUG: Fork checkpoint - from_s3_bucket={from_s3_bucket}, not_after_step={not_after_step}"
                )

            # Determine which checkpoint to pull
            if not_after_step is None:
                # Pull only the latest checkpoint
                if verbose:
                    print(
                        f"Pulling latest checkpoint for model {from_model} from S3 bucket {from_s3_bucket}..."
                    )
                await self._experimental_pull_from_s3(
                    Model(name=from_model, project=from_project),
                    s3_bucket=from_s3_bucket,
                    verbose=verbose,
                    exclude=["logs", "trajectories"],
                    only_step="latest",
                )
            else:
                # Find the right checkpoint not after the specified step
                from art.utils.s3_checkpoint_utils import (
                    get_checkpoint_step_not_after_from_s3,
                )

                if verbose:
                    print(
                        f"Finding checkpoint not after step {not_after_step} for model {from_model} in S3..."
                    )

                # Find which step to pull
                target_step = await get_checkpoint_step_not_after_from_s3(
                    model_name=from_model,
                    project=from_project,
                    not_after_step=not_after_step,
                    s3_bucket=from_s3_bucket,
                    prefix=prefix,
                )

                if target_step is None:
                    raise ValueError(
                        f"No checkpoints found not after step {not_after_step} for model {from_model} in S3"
                    )

                if verbose:
                    print(
                        f"Found checkpoint at step {target_step}, pulling only this checkpoint..."
                    )

                # Pull only the specific checkpoint we need
                await pull_model_from_s3(
                    model_name=from_model,
                    project=from_project,
                    step=target_step,
                    s3_bucket=from_s3_bucket,
                    verbose=verbose,
                    art_path=self._path,
                    exclude=["logs", "trajectories"],  # Only need checkpoints
                )

        # Find the checkpoint to fork
        checkpoint_base_dir = os.path.join(source_model_dir, "checkpoints")
        if not os.path.exists(checkpoint_base_dir):
            raise FileNotFoundError(
                f"No checkpoints found for model {from_model} in project {from_project}"
            )

        if verbose:
            print(f"DEBUG: Checkpoint base dir: {checkpoint_base_dir}")
            print(
                f"DEBUG: Contents: {os.listdir(checkpoint_base_dir) if os.path.exists(checkpoint_base_dir) else 'Does not exist'}"
            )

        # Get all available checkpoint steps
        available_steps = sorted(
            int(d)
            for d in os.listdir(checkpoint_base_dir)
            if os.path.isdir(os.path.join(checkpoint_base_dir, d)) and d.isdigit()
        )

        if not available_steps:
            raise FileNotFoundError(
                f"No checkpoint directories found for model {from_model}"
            )

        # Determine which step to use
        if not_after_step is None:
            # Use the latest checkpoint
            selected_step = available_steps[-1]
        else:
            # Find the last checkpoint not after the specified step
            valid_steps = [s for s in available_steps if s <= not_after_step]
            if not valid_steps:
                raise ValueError(
                    f"No checkpoints found not after step {not_after_step}. "
                    f"Available steps: {available_steps}"
                )
            selected_step = valid_steps[-1]

        # Create destination checkpoint directory
        dest_checkpoint_dir = get_step_checkpoint_dir(dest_model_dir, selected_step)
        os.makedirs(os.path.dirname(dest_checkpoint_dir), exist_ok=True)

        # Copy the checkpoint
        source_checkpoint_dir = os.path.join(
            checkpoint_base_dir, f"{selected_step:04d}"
        )
        if verbose:
            print(
                f"Copying checkpoint from {source_checkpoint_dir} to {dest_checkpoint_dir}"
            )
            print(f"DEBUG: Source dir exists: {os.path.exists(source_checkpoint_dir)}")
            if os.path.exists(source_checkpoint_dir):
                print(
                    f"DEBUG: Source dir contents: {os.listdir(source_checkpoint_dir)}"
                )
                print(
                    f"DEBUG: Source dir is empty: {len(os.listdir(source_checkpoint_dir)) == 0}"
                )

        import shutil

        # Remove destination if it already exists (empty directory from previous attempts)
        if os.path.exists(dest_checkpoint_dir):
            if verbose:
                print("DEBUG: Destination already exists, removing it first")
            shutil.rmtree(dest_checkpoint_dir)

        shutil.copytree(source_checkpoint_dir, dest_checkpoint_dir)

        if verbose:
            print(
                f"Successfully forked checkpoint from {from_model} (step {selected_step}) to {model.name}"
            )
