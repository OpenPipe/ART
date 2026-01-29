from datetime import datetime
import json
import os
from typing import TYPE_CHECKING, Any, Generic, Iterable, Optional, cast, overload
import warnings

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
import polars as pl
from pydantic import BaseModel
from typing_extensions import Never, TypeVar

from . import dev
from .trajectories import Trajectory, TrajectoryGroup
from .types import TrainConfig
from .utils.old_benchmarking.calculate_step_metrics import calculate_step_std_dev
from .utils.trajectory_logging import write_trajectory_groups_parquet

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from art.backend import Backend


ModelConfig = TypeVar("ModelConfig", bound=BaseModel | None)
StateType = TypeVar("StateType", bound=dict[str, Any], default=dict[str, Any])


class Model(
    BaseModel,
    Generic[ModelConfig, StateType],
):
    """
    A model is an object that can be passed to your `rollout` function, and used
    to log completions. Additionally, a `TrainableModel`, which is a subclass of
    `Model`, can be used to train a model.

    The `Model` abstraction is useful for comparing prompted model performance
    to the performance of your trained models.

    You can instantiate a prompted model like so:

    ``python model = art.Model(
        name="gpt-4.1", project="my-project",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )
    ``

    Or, if you're pointing at OpenRouter:

    ``python model = art.Model(
        name="gemini-2.5-pro", project="my-project",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
        inference_model_name="google/gemini-2.5-pro-preview-03-25",
    )
    ``

    For trainable (`art.TrainableModel`) models the inference values will be
    populated automatically by `model.register(api)` so you generally don't need
    to think about them.
    """

    name: str
    project: str
    entity: str | None = None
    id: str | None = None
    config: ModelConfig
    # Discriminator field for FastAPI serialization
    trainable: bool = False

    # --- Inference connection information (populated automatically for
    #     TrainableModel or set manually for prompted / comparison models) ---
    inference_api_key: str | None = None
    inference_base_url: str | None = None
    # If set, this will be used instead of `self.name` when calling the
    # inference endpoint.
    inference_model_name: str | None = None

    # --- Frontend logging configuration ---
    base_path: str = ".art"  # Same default as LocalBackend for backward compat
    report_metrics: list[str] | None = None  # None = default (wandb if key present)

    _backend: Optional["Backend"] = None
    _s3_bucket: str | None = None
    _s3_prefix: str | None = None
    _openai_client: AsyncOpenAI | None = None
    _wandb_run: Optional["Run"] = None  # Private, for lazy wandb initialization

    def __init__(
        self,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig | None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            entity=entity,
            config=config,
            inference_api_key=inference_api_key,
            inference_base_url=inference_base_url,
            inference_model_name=inference_model_name,
            base_path=base_path,
            report_metrics=report_metrics,
            **kwargs,
        )

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
    ) -> "Model[None, dict[str, Any]]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
    ) -> "Model[ModelConfig, dict[str, Any]]": ...

    def __new__(  # pyright: ignore[reportInconsistentOverload]
        cls,
        *args,
        **kwargs,
    ) -> "Model[ModelConfig, StateType]":
        return super().__new__(cls)

    def safe_model_dump(self, *args, **kwargs) -> dict:
        """
        Dump the model, but remove the config field to prevent serialization errors in the backend.
        """
        data = super().model_dump(*args, **kwargs)
        # remove config from dumped_model to prevent serialization errors
        data["config"] = None
        return data

    def backend(self) -> "Backend":
        if self._backend is None:
            raise ValueError(
                "Model is not registered with the Backend. You must call `model.register()` first."
            )
        return self._backend

    async def register(self, backend: "Backend") -> None:
        if self.config is not None:
            try:
                self.config.model_dump_json()  # ty:ignore[invalid-argument-type, possibly-missing-attribute]
            except Exception as e:
                raise ValueError(
                    "The model config cannot be serialized to JSON. Please ensure that all fields are JSON serializable and try again."
                ) from e

        self._backend = backend
        await self._backend.register(self)

    def openai_client(
        self,
    ) -> AsyncOpenAI:
        if self._openai_client is not None:
            return self._openai_client

        if self.inference_api_key is None or self.inference_base_url is None:
            if self.trainable:
                raise ValueError(
                    "OpenAI client not yet available on this trainable model. You must call `model.register()` first."
                )
            else:
                raise ValueError(
                    "In order to create an OpenAI client you must provide an `inference_api_key` and `inference_base_url`."
                )
        self._openai_client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=1200, connect=5.0),
                limits=httpx.Limits(
                    max_connections=100_000, max_keepalive_connections=100_000
                ),
            ),
        )
        return self._openai_client

    def litellm_completion_params(self, step: int | None = None) -> dict:
        """Return the parameters that should be sent to litellm.completion.

        Args:
            step: If provided, returns params for specific checkpoint using
                  the `name@step` convention. If None, returns params for
                  latest checkpoint (default, backwards compatible).
        """
        model_name = self.get_inference_name(step)
        if self.trainable:
            model_name = f"hosted_vllm/{model_name}"
        return {
            "model": model_name,
            "base_url": self.inference_base_url,
            "api_key": self.inference_api_key,
            "temperature": 1,  # Important for trainable models
        }

    # ------------------------------------------------------------------
    # Inference name helpers
    # ------------------------------------------------------------------

    def get_inference_name(self, step: int | None = None) -> str:
        """Return the name that should be sent to the inference endpoint.

        Args:
            step: If provided, returns name for specific checkpoint.
                  If None, returns name for latest/default checkpoint.

        Note:
            For TrainableModel with LocalBackend, vLLM serves LoRA adapters
            as `model.name@step`, so this always includes the step suffix.
            For ServerlessBackend, it uses W&B artifact naming conventions.
        """
        # If we have a registered backend with _model_inference_name, use it
        # This ensures proper step handling for each backend type
        if self._backend is not None and hasattr(
            self._backend, "_model_inference_name"
        ):
            return self._backend._model_inference_name(self, step=step)

        # Fallback for non-registered models or backends without the method
        base_name = self.inference_model_name or self.name
        if step is not None:
            return f"{base_name}@{step}"
        return base_name

    def _get_output_dir(self) -> str:
        """Get the output directory for this model."""
        return f"{self.base_path}/{self.project}/models/{self.name}"

    def overwrite_state(self, state: StateType) -> None:
        """Overwrite persistent state in the model directory as JSON.

        This state is stored in `state.json` within the model's output directory
        and can be used to track training progress, dataset position, or any
        other information that should persist across runs.

        Warning:
            This overwrites the entire state file. Prefer `merge_state()` unless
            you intentionally want to replace all existing keys.

        Args:
            state: A dictionary of JSON-serializable values to persist.

        Example:
            model.overwrite_state({
                "step": 5,
                "dataset_offset": 100,
                "last_checkpoint_time": "2024-01-15T10:30:00",
            })
        """
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/state.json", "w") as f:
            json.dump(state, f, indent=2)

    def write_state(self, state: StateType) -> None:
        """Deprecated: use `overwrite_state()` or `merge_state()` instead."""
        warnings.warn(
            "write_state() is deprecated. Use overwrite_state() or merge_state() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.overwrite_state(state)

    def merge_state(self, state: StateType) -> StateType:
        """Deep-merge state into the existing state and persist it.

        Args:
            state: A dictionary of JSON-serializable values to merge.

        Returns:
            The merged state dictionary that was persisted.
        """
        existing = self.read_state() or {}
        merged = self._deep_merge_dicts(existing, state)
        self.overwrite_state(merged)
        return cast(StateType, merged)

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any], updates: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Model._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def read_state(self) -> StateType | None:
        """Read persistent state from the model directory.

        Returns:
            The state dictionary if it exists, or None if no state has been saved.

        Example:
            state = model.read_state()
            if state:
                start_step = state["step"]
                dataset_offset = state["dataset_offset"]
        """
        output_dir = self._get_output_dir()
        state_path = f"{output_dir}/state.json"
        if not os.path.exists(state_path):
            return None
        with open(state_path, "r") as f:
            return json.load(f)

    def _get_wandb_run(self) -> Optional["Run"]:
        """Get or create the wandb run for this model."""
        import wandb

        if "WANDB_API_KEY" not in os.environ:
            return None
        if self._wandb_run is None or self._wandb_run._is_finished:
            run = wandb.init(
                project=self.project,
                name=self.name,
                id=self.name,
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
            self._wandb_run = run

            # Define training_step as the x-axis for all metrics.
            # This allows out-of-order logging (e.g., async validation for previous steps).
            wandb.define_metric("training_step")
            wandb.define_metric("train/*", step_metric="training_step")
            wandb.define_metric("val/*", step_metric="training_step")
        return self._wandb_run

    def _log_metrics(
        self,
        metrics: dict[str, float],
        split: str,
        step: int,
    ) -> None:
        """Log metrics to history.jsonl and optionally wandb."""
        prefixed = {f"{split}/{k}": v for k, v in metrics.items()}
        output_dir = self._get_output_dir()

        # Write to history.jsonl
        with open(f"{output_dir}/history.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        k: v for k, v in prefixed.items() if v == v
                    }  # Filter out NaN values
                    | {"step": step, "recorded_at": datetime.now().isoformat()}
                )
                + "\n"
            )

        # Log to wandb if enabled
        should_log_wandb = (
            self.report_metrics is None and "WANDB_API_KEY" in os.environ
        ) or (self.report_metrics is not None and "wandb" in self.report_metrics)
        if should_log_wandb:
            if run := self._get_wandb_run():
                run.log({"training_step": step, **prefixed})

    async def log(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[TrajectoryGroup] | None
        ) = None,
        split: str = "val",
        *,
        metrics: dict[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log trajectories and/or metrics.

        Can be used in two ways:
        1. Log trajectories: `await model.log(trajectory_groups, split="train")`
        2. Log raw metrics: `await model.log(metrics={"loss": 0.5}, step=1)`
        3. Both: `await model.log(trajectory_groups, metrics=extra_metrics)`

        Args:
            trajectories: A batch of trajectories or trajectory groups. Optional if
                logging only metrics.
            split: The evaluation's split. Defaults to "val".
            metrics: Optional dict of metrics to log directly (e.g., training metrics
                from backend.train()).
            step: Optional step number for metrics. If not provided, uses current step.
        """
        # Determine the step to use
        if step is None:
            step = await self.get_step() if self.trainable else 0

        # If only metrics provided (no trajectories), just log them and return
        if trajectories is None:
            if metrics is not None:
                self._log_metrics(metrics, split, step)
            return

        # Convert to list[TrajectoryGroup]
        if any(isinstance(t, Trajectory) for t in trajectories) or any(
            isinstance(t, BaseException) for t in trajectories
        ):
            trajectory_groups = [
                TrajectoryGroup(
                    cast(Iterable[Trajectory | BaseException], trajectories)
                )
            ]
        else:
            trajectory_groups = cast(list[TrajectoryGroup], list(trajectories))

        # Ensure output directories exist
        output_dir = self._get_output_dir()
        trajectories_dir = f"{output_dir}/trajectories/{split}"
        os.makedirs(trajectories_dir, exist_ok=True)

        # 1. Write parquet
        file_name = f"{step:04d}.parquet"
        write_trajectory_groups_parquet(
            trajectory_groups, f"{trajectories_dir}/{file_name}"
        )

        # 2. Calculate aggregate metrics
        all_metrics: dict[str, list[float]] = {"reward": [], "exception_rate": []}
        group_metrics: dict[str, list[float]] = {}

        for group in trajectory_groups:
            if group.trajectories:
                for metric, value in group.metrics.items():
                    if metric not in group_metrics:
                        group_metrics[metric] = []
                    group_metrics[metric].append(float(value))
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
        averages: dict[str, float] = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        # Aggregate group-level metrics once per group
        for metric, values in group_metrics.items():
            if len(values) > 0:
                averages[f"group_metric_{metric}"] = sum(values) / len(values)

        # Calculate average standard deviation of rewards within groups
        averages["reward_std_dev"] = calculate_step_std_dev(trajectory_groups)

        # Merge in any additional metrics passed directly
        if metrics is not None:
            averages.update(metrics)

        # 3. Log metrics (writes to history.jsonl and wandb)
        self._log_metrics(averages, split, step)

    async def get_step(self) -> int:
        """
        Get the model's current training step. For non-trainable models, returns 0.
        """
        if self.trainable:
            return await self.backend()._get_step(self)  # type: ignore
        return 0


# ---------------------------------------------------------------------------
# Trainable models
# ---------------------------------------------------------------------------


class TrainableModel(Model[ModelConfig, StateType], Generic[ModelConfig, StateType]):
    base_model: str
    # Override discriminator field for FastAPI serialization
    trainable: bool = True

    # The fields within `_internal_config` are unstable and subject to change.
    # Use at your own risk.
    _internal_config: dev.InternalModelConfig | None = None

    def __init__(
        self,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig | None = None,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            entity=entity,
            id=id,
            config=config,
            base_model=base_model,
            base_path=base_path,
            report_metrics=report_metrics,
            **kwargs,
        )
        if _internal_config is not None:
            # Bypass BaseModel __setattr__ to allow setting private attr
            object.__setattr__(self, "_internal_config", _internal_config)

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: None = None,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[None, dict[str, Any]]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[ModelConfig, dict[str, Any]]": ...

    def __new__(  # pyright: ignore[reportInconsistentOverload]
        cls,
        *args,
        **kwargs,
    ) -> "TrainableModel[ModelConfig, StateType]":
        return super().__new__(cls)

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["_internal_config"] = self._internal_config
        return data

    def safe_model_dump(self, *args, **kwargs) -> dict:
        """
        Dump the model, but remove the config field to prevent serialization errors in the backend.
        """
        data = self.model_dump(*args, **kwargs)
        # remove config from dumped_model to prevent serialization errors
        data["config"] = None
        return data

    async def register(
        self,
        backend: "Backend",
        _openai_client_config: dev.OpenAIServerConfig | None = None,
    ) -> None:
        await super().register(backend)
        base_url, api_key = await backend._prepare_backend_for_training(
            self, _openai_client_config
        )

        # Populate the top-level inference fields so that the rest of the
        # code (and any user code) can create an OpenAI client immediately.
        self.inference_base_url = base_url
        self.inference_api_key = api_key
        self.inference_model_name = (
            hasattr(backend, "_model_inference_name")
            and getattr(backend, "_model_inference_name")(self)
            or self.name
        )

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        output_dir = self._get_output_dir()
        steps_to_keep = [await self.get_step()]  # Keep latest

        # Read history.jsonl to find best step
        try:
            best_step = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=[best_checkpoint_metric])
                .group_by("step")
                .mean()
                .sort(best_checkpoint_metric)
                .select(pl.col("step").last())
                .item()
            )
            steps_to_keep.append(best_step)
        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "{best_checkpoint_metric}" metric found in history')

        # Backend only does file deletion
        await self.backend()._delete_checkpoint_files(self, steps_to_keep)

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        .. deprecated::
            Use ``backend.train(model, trajectory_groups, ...)`` instead.
            This method will be removed in a future version.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        warnings.warn(
            "model.train() is deprecated. Use backend.train(model, ...) instead.\n\n"
            "Migration guide:\n"
            "  # Before (deprecated):\n"
            "  await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))\n\n"
            "  # After (recommended):\n"
            "  result = await backend.train(model, trajectory_groups, learning_rate=5e-6)\n"
            "  await model.log(trajectory_groups, metrics=result.metrics, step=result.step, split='train')\n\n"
            "Key differences:\n"
            "  - backend.train() does NOT automatically log trajectories or metrics\n"
            "  - backend.train() returns a TrainResult with step, metrics, and checkpoint info\n"
            "  - Each backend has its own type-checked parameters (no more generic config objects)",
            DeprecationWarning,
            stacklevel=2,
        )
        groups_list = list(trajectory_groups)
        _config = _config or {}  # ty:ignore[invalid-assignment]

        # 1. Log trajectories first (frontend handles this now)
        await self.log(groups_list, split="train")

        # 2. Train (backend no longer logs internally)
        training_metrics: list[dict[str, float]] = []
        async for metrics in self.backend()._train_model(
            self,
            groups_list,
            config,
            _config,  # ty:ignore[invalid-argument-type]
            verbose,
        ):
            training_metrics.append(metrics)

        # 3. Log training metrics (loss, gradient norms, etc.)
        if training_metrics:
            avg_metrics = {
                k: sum(d.get(k, 0) for d in training_metrics)
                / sum(1 for d in training_metrics if k in d)
                for k in {k for d in training_metrics for k in d}
                if k != "num_gradient_steps"
            }
            # Get the current step after training
            step = await self.get_step()
            self._log_metrics(avg_metrics, "train", step)
