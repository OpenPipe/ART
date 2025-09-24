import asyncio
import json
import math
import os
import subprocess
from datetime import datetime
from types import TracebackType
from typing import AsyncIterator, Literal, cast

import aiohttp
import numpy as np
import polars as pl
import torch
import wandb
import weave
from openai import AsyncOpenAI
from tqdm import auto as tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import Self
from wandb.sdk.wandb_run import Run
from weave.trace.weave_client import WeaveClient

from art.utils.deploy_model import (
    LoRADeploymentJob,
    LoRADeploymentProvider,
    deploy_model,
)
from art.utils.old_benchmarking.calculate_step_metrics import calculate_step_std_dev
from art.utils.output_dirs import (
    get_default_art_path,
    get_model_dir,
    get_output_dir_from_model_properties,
    get_step_checkpoint_dir,
    get_trajectories_split_dir,
)
from art.utils.s3 import (
    ExcludableOption,
    pull_model_from_s3,
    push_model_to_s3,
)
from art.utils.trajectory_logging import serialize_trajectory_groups
from mp_actors import close_proxy, move_to_child_process

from .. import dev
from ..backend import Backend
from ..model import Model, TrainableModel
from ..preprocessing.pack import (
    PackedTensors,
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
    plot_packed_tensors,
)
from ..preprocessing.tokenize import tokenize_trajectory_groups
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import Message, TrainConfig
from ..utils import format_message, get_model_step
from .checkpoints import (
    delete_checkpoints,
)
from .service import ModelService


class LocalBackend(Backend):
    def __init__(self, *, in_process: bool = False, path: str | None = None) -> None:
        """
        Initializes a local, directory-based Backend interface at the given path.

        Note:
            The local Backend uses Weights & Biases for training monitoring.
            If you don't have a W&B account, you can create one at https://wandb.ai.

        Args:
            in_process: Whether to run the local service in-process.
            path: The path to the local directory. Defaults to "{repo_root}/.art".
        """
        self._in_process = in_process
        self._path = path or get_default_art_path()
        os.makedirs(self._path, exist_ok=True)

        # Other initialization
        self._services: dict[str, ModelService] = {}
        self._tokenizers: dict[str, "PreTrainedTokenizerBase"] = {}
        self._wandb_runs: dict[str, Run] = {}
        self._weave_clients: dict[str, WeaveClient] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._close()

    async def close(self) -> None:
        """
        If running vLLM in a separate process, this will kill that process and close the communication threads.
        """
        self._close()

    def _close(self) -> None:
        for _, service in self._services.items():
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
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            json.dump(model.model_dump(), f)

        # Initialize wandb and weave early if this is a trainable model
        if model.trainable and "WANDB_API_KEY" in os.environ:
            _ = self._get_wandb_run(model)

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from ..torchtune.service import TorchtuneService
        from ..unsloth.decoupled_service import DecoupledUnslothService
        from ..unsloth.service import UnslothService

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            if config.get("torchtune_args") is not None:
                service_class = TorchtuneService
            elif config.get("_decouple_vllm_and_unsloth", False):
                service_class = DecoupledUnslothService
            else:
                service_class = UnslothService
            self._services[model.name] = service_class(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                # Kill all "model-service" processes to free up GPU memory
                subprocess.run(["pkill", "-9", "model-service"])
                if isinstance(
                    self._services[model.name],
                    (UnslothService, DecoupledUnslothService),
                ):
                    # To enable sleep mode, import peft before unsloth
                    # Unsloth will issue warnings, but everything appears to be okay
                    if config.get("engine_args", {}).get("enable_sleep_mode", False):
                        os.environ["IMPORT_PEFT"] = "1"
                    # When moving the service to a child process, import unsloth
                    # early to maximize optimizations
                    os.environ["IMPORT_UNSLOTH"] = "1"
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="model-service",
                )
        return self._services[model.name]

    def _get_packed_tensors(
        self,
        model: TrainableModel,
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
        tokenizer = self._tokenizers[model.base_model]
        tokenized_results = list(
            tokenize_trajectory_groups(
                tokenizer,
                trajectory_groups,
                allow_training_without_logprobs,
                scale_rewards,
            )
        )
        if not tokenized_results:
            return None
        max_tokens = max(len(result.tokens) for result in tokenized_results)
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
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
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

    async def _get_step(self, model: TrainableModel) -> int:
        return self.__get_step(model)

    def __get_step(self, model: Model) -> int:
        if model.trainable:
            model = cast(TrainableModel, model)
            return get_model_step(model, self._path)
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoints(
        self,
        model: TrainableModel,
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        output_dir = get_model_dir(model=model, art_path=self._path)
        # Keep the latest step
        steps_to_keep = [get_model_step(model, self._path)]
        try:
            best_step = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=[benchmark])
                .group_by("step")
                .mean()
                .with_columns(pl.col(benchmark).ewm_mean(alpha=benchmark_smoothing))
                .sort(benchmark)
                .select(pl.col("step").last())
                .item()
            )
            steps_to_keep.append(best_step)
        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "{benchmark}" metric found in history')
        delete_checkpoints(output_dir, steps_to_keep)

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        service = await self._get_service(model)
        await service.start_openai_server(config=config)
        server_args = (config or {}).get("server_args", {})

        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        api_key = server_args.get("api_key", None) or "default"

        def done_callback(_: asyncio.Task[None]) -> None:
            close_proxy(self._services.pop(model.name))

        asyncio.create_task(
            self._monitor_openai_server(model.name, base_url, api_key)
        ).add_done_callback(done_callback)

        return base_url, api_key

    async def _monitor_openai_server(
        self, model_name: str, base_url: str, api_key: str
    ) -> None:
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        async with aiohttp.ClientSession() as session:
            while True:
                # Wait 30 seconds before checking again
                await asyncio.sleep(30)
                # If the server is sleeping, skip the check
                if await self._services[model_name].vllm_engine_is_sleeping():
                    continue
                # Check the metrics
                async with session.get(
                    f"{base_url.split('/v1')[0]}/metrics"
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
                        # Send a health check with a 5 second timeout
                        await openai_client.completions.create(
                            model=model_name,
                            prompt="Hi",
                            max_tokens=1,
                            timeout=5,
                        )
                        # get the completion response, exit the loop
                        break
                    except Exception as e:
                        # If the server is sleeping, a failed health check is okay
                        if await self._services[model_name].vllm_engine_is_sleeping():
                            continue
                        raise e

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        # Save logs for trajectory groups
        parent_dir = get_trajectories_split_dir(
            get_model_dir(model=model, art_path=self._path), split
        )
        os.makedirs(parent_dir, exist_ok=True)

        # Get the file name for the current iteration, or default to 0 for non-trainable models
        iteration = self.__get_step(model)
        file_name = f"{iteration:04d}.jsonl"

        # Write the logs to the file
        with open(f"{parent_dir}/{file_name}", "w") as f:
            f.write(serialize_trajectory_groups(trajectory_groups))

        # Collect all metrics (including reward) across all trajectories
        all_metrics: dict[str, list[float]] = {"reward": [], "exception_rate": []}

        for group in trajectory_groups:
            for trajectory in group:
                if isinstance(trajectory, BaseException):
                    all_metrics["exception_rate"].append(1)
                    continue
                else:
                    all_metrics["exception_rate"].append(0)
                # Add reward metric
                all_metrics["reward"].append(trajectory.reward)

                # Collect other custom metrics
                for metric, value in trajectory.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))

        # Calculate averages for all metrics
        averages = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        # Calculate average standard deviation of rewards within groups
        averages["reward_std_dev"] = calculate_step_std_dev(trajectory_groups)

        self._log_metrics(model, averages, split)

    def _trajectory_log(self, trajectory: Trajectory) -> str:
        """Format a trajectory into a readable log string."""
        header = f"reward: {trajectory.reward} {' '.join(f'{k}: {v}' for k, v in trajectory.metrics.items())}\n\n"
        formatted_messages = []
        for message_or_choice in trajectory.messages_and_choices:
            if isinstance(message_or_choice, dict):
                message = message_or_choice
            else:
                message = cast(Message, message_or_choice.message.model_dump())
            formatted_messages.append(format_message(message))
        return header + "\n".join(formatted_messages)

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
        if verbose:
            print("Logging training data to disk...")
        await self._log(model, trajectory_groups, "train")
        if verbose:
            print("Packing tensors...")

        # Count submitted groups and trainable groups
        num_groups_submitted = len(trajectory_groups)
        num_groups_trainable = sum(
            1
            for group in trajectory_groups
            if group and len(set(trajectory.reward for trajectory in group)) > 1
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
            current_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), current_step
            )
            next_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), next_step
            )

            # If the current checkpoint exists, rename it to the next step
            if os.path.exists(current_checkpoint_dir):
                os.rename(current_checkpoint_dir, next_checkpoint_dir)
                print(
                    f"Advanced step from {current_step} to {next_step} (no training occurred)"
                )

            # Log metrics showing no groups were trainable
            self._log_metrics(
                model,
                {
                    "num_groups_submitted": num_groups_submitted,
                    "num_groups_trainable": 0,
                },
                "train",
                step=next_step,
            )
            return
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors, f"{get_model_dir(model=model, art_path=self._path)}/tensors"
        )
        if dev_config.get("scale_learning_rate_by_reward_std_dev", False):
            config = config.model_copy(
                update={
                    "learning_rate": config.learning_rate
                    * self._get_reward_std_dev_learning_rate_multiplier(model)
                }
            )
        results: list[dict[str, float]] = []
        estimated_gradient_steps = disk_packed_tensors["num_sequences"]
        if torchtune_args := (model._internal_config or dev.InternalModelConfig()).get(
            "torchtune_args"
        ):
            tp = torchtune_args.get("tensor_parallel_dim", 1)
            cp = torchtune_args.get("context_parallel_dim", 1)
            world_size = torch.cuda.device_count()
            dp = world_size // (tp * cp)
            estimated_gradient_steps = math.ceil(estimated_gradient_steps / dp)
        pbar = tqdm.tqdm(total=estimated_gradient_steps, desc="train")
        async for result in service.train(
            disk_packed_tensors, config, dev_config, verbose
        ):
            num_gradient_steps = int(
                result.pop("num_gradient_steps", estimated_gradient_steps)
            )
            assert num_gradient_steps == estimated_gradient_steps, (
                f"num_gradient_steps {num_gradient_steps} != estimated_gradient_steps {estimated_gradient_steps}"
            )
            results.append(result)
            yield {**result, "num_gradient_steps": num_gradient_steps}
            pbar.update(1)
            pbar.set_postfix(result)
        pbar.close()
        if verbose:
            print("Logging metrics...")
        data = {
            k: sum(d.get(k, 0) for d in results) / sum(1 for d in results if k in d)
            for k in {k for d in results for k in d}
        }
        # Add group counting metrics
        data["num_groups_submitted"] = num_groups_submitted
        data["num_groups_trainable"] = num_groups_trainable
        # Get the current step after training
        current_step = self.__get_step(model)
        self._log_metrics(model, data, "train", step=current_step)
        if verbose:
            print("_train_model complete")

    def _get_reward_std_dev_learning_rate_multiplier(
        self, model: TrainableModel
    ) -> float:
        output_dir = get_model_dir(model=model, art_path=self._path)
        learning_rate_multiplier = 1.0  # Default prior
        try:
            std_dev_history = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=["train/reward_std_dev"])
                .group_by("step")
                .mean()
                .sort("step")
            )

            # Fit linear regression to std_dev_history
            if len(std_dev_history) > 1:
                steps = std_dev_history["step"].to_numpy()
                std_devs = std_dev_history["train/reward_std_dev"].to_numpy()

                # Fit linear regression: y = mx + b
                # polyfit returns [coefficient, intercept] for degree 1
                coefficient, intercept = np.polyfit(steps, std_devs, deg=1)

                # Get prediction for the last step
                last_step = steps[-1]
                last_step_prediction = coefficient * last_step + intercept
                last_step_actual = std_devs[-1]

                # Calculate R-squared and adjusted R-squared
                predictions = coefficient * steps + intercept
                ss_residual = np.sum((std_devs - predictions) ** 2)
                ss_total = np.sum((std_devs - np.mean(std_devs)) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

                # Adjusted R-squared accounts for sample size
                # For simple linear regression: adj_R² = 1 - (1 - R²) * (n - 1) / (n - 2)
                n_samples = len(steps)
                if n_samples > 2:
                    adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (
                        n_samples - 2
                    )
                else:
                    adjusted_r_squared = (
                        0  # Not enough samples for meaningful adjustment
                    )

                # Calculate learning rate multiplier
                # raw_multiplier = last_step_prediction / intercept (if intercept > 0)
                # adjusted by goodness of fit: multiplier = 1 + adj_R² * (raw_multiplier - 1)
                if intercept > 0:
                    raw_multiplier = last_step_prediction / intercept
                    # learning_rate_multiplier = 1 + adjusted_r_squared * (
                    #     raw_multiplier - 1
                    # )
                    learning_rate_multiplier = raw_multiplier
                else:
                    # If intercept <= 0, can't calculate meaningful ratio, stick with prior
                    raw_multiplier = 1.0
                    learning_rate_multiplier = 1.0

                print(f"Regression fitted: y = {coefficient:.6f}x + {intercept:.6f}")
                print(f"  Coefficient (slope): {coefficient:.6f}")
                print(f"  Intercept: {intercept:.6f}")
                print(f"  R-squared: {r_squared:.4f}")
                print(
                    f"  Adjusted R-squared: {adjusted_r_squared:.4f} (n={n_samples} samples)"
                )
                print(
                    f"  Last step ({last_step}) prediction: {last_step_prediction:.6f}"
                )
                print(f"  Last step actual value: {last_step_actual:.6f}")
                print(
                    f"  Prediction error: {abs(last_step_actual - last_step_prediction):.6f}"
                )
                print(f"  Raw LR multiplier (pred/intercept): {raw_multiplier:.4f}")
                print(f"  Adjusted LR multiplier: {learning_rate_multiplier:.4f}")
            else:
                print(
                    f"Not enough data points to fit regression (need at least 2, got {len(std_dev_history)})"
                )

        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "train/reward_std_dev" metric found in history')

        return learning_rate_multiplier

    def _log_metrics(
        self,
        model: Model,
        metrics: dict[str, float],
        split: str,
        step: int | None = None,
    ) -> None:
        metrics = {f"{split}/{metric}": value for metric, value in metrics.items()}
        step = step if step is not None else self.__get_step(model)

        with open(
            f"{get_model_dir(model=model, art_path=self._path)}/history.jsonl", "a"
        ) as f:
            f.write(
                json.dumps(
                    {
                        k: v for k, v in metrics.items() if v == v
                    }  # Filter out NaN values
                    | {"step": step, "recorded_at": datetime.now().isoformat()}
                )
                + "\n"
            )

        # If we have a W&B run, log the data there
        if run := self._get_wandb_run(model):
            # Mark the step metric itself as hidden so W&B doesn't create an automatic chart for it
            wandb.define_metric("training_step", hidden=True)

            # Enabling the following line will cause W&B to use the training_step metric as the x-axis for all metrics
            # wandb.define_metric(f"{split}/*", step_metric="training_step")
            run.log({"training_step": step, **metrics}, step=step)

    def _get_wandb_run(self, model: Model) -> Run | None:
        if "WANDB_API_KEY" not in os.environ:
            return None
        if (
            model.name not in self._wandb_runs
            or self._wandb_runs[model.name]._is_finished
        ):
            run = wandb.init(
                project=model.project,
                name=model.name,
                id=model.name,
                resume="allow",
                settings=wandb.Settings(
                    x_stats_open_metrics_endpoints={
                        "vllm": "http://localhost:8000/metrics",
                    },
                    x_stats_open_metrics_filters=(
                        "vllm.vllm:num_requests_waiting",
                        "vllm.vllm:num_requests_running",
                    ),
                ),
            )
            self._wandb_runs[model.name] = run
            os.environ["WEAVE_PRINT_CALL_LINK"] = os.getenv(
                "WEAVE_PRINT_CALL_LINK", "False"
            )
            os.environ["WEAVE_LOG_LEVEL"] = os.getenv("WEAVE_LOG_LEVEL", "CRITICAL")
            self._weave_clients[model.name] = weave.init(model.project)
        return self._wandb_runs[model.name]

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

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

    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: "TrainableModel",
        step: int | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        pull_s3: bool = True,
        wait_for_completion: bool = True,
    ) -> LoRADeploymentJob:
        """
        Deploy the model's latest checkpoint to a hosted inference endpoint.

        Together is currently the only supported provider. See link for supported base models:
        https://docs.together.ai/docs/lora-inference#supported-base-models
        """
        return await deploy_model(
            deploy_to=deploy_to,
            model=model,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            pull_s3=pull_s3,
            wait_for_completion=wait_for_completion,
            art_path=self._path,
        )
