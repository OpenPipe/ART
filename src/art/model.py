from openai import AsyncOpenAI
from typing import cast, Iterable, TYPE_CHECKING, Optional

from . import dev
from .openai import patch_openai
from .trajectories import Trajectory, TrajectoryGroup
from .types import TrainConfig
from pydantic import BaseModel
from openai import (
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
import httpx

if TYPE_CHECKING:
    from .local.api import LocalAPI


class Model(BaseModel):
    name: str
    project: str

    config: BaseModel | None = None

    # These are generally only used on trainable models, but are included on the
    # base because they're sometimes useful for prompted models that you'd like
    # to call via an OpenAI-compatible API as well.
    base_url: str | None = None
    api_key: str | None = None

    trainable: bool = False
    _api: Optional["LocalAPI"] = None
    _s3_bucket: str | None = None
    _s3_prefix: str | None = None

    def api(self) -> "LocalAPI":
        if self._api is None:
            raise ValueError(
                "Model is not registered with the API. You must call `model.register()` first."
            )
        return self._api

    async def register(self, api: "LocalAPI") -> None:
        if self.config is not None:
            try:
                self.config.model_dump_json()
            except Exception as e:
                raise ValueError(
                    "The model config cannot be serialized to JSON. Please ensure that all fields are JSON serializable and try again."
                ) from e

        self._api = api
        await self._api.register(self)

    async def log(
        self,
        trajectories: Iterable[Trajectory] | Iterable[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectories or trajectory groups.

        Args:
            trajectories: A batch of trajectories or trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        if any(isinstance(t, Trajectory) for t in trajectories):
            trajectory_groups = [
                TrajectoryGroup(cast(Iterable[Trajectory], trajectories))
            ]
        else:
            trajectory_groups = cast(list[TrajectoryGroup], list(trajectories))
        await self.api()._log(
            self,
            trajectory_groups,
            split,
        )

    async def pull_from_s3(
        self,
        s3_bucket: str,
        *,
        prefix: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Sync the model directory from S3 into the local output directory.

        Args:
            s3_bucket: The name of the S3 bucket (without the ``s3://`` scheme).
            prefix: Optional key prefix inside the bucket. Do **not** include leading or trailing slashes.
            verbose: When *True*, stream `aws` output; otherwise, suppress it.
        """
        await self.api()._pull_from_s3(
            self, s3_bucket=s3_bucket, prefix=prefix, verbose=verbose
        )
        # Cache for later pushes.
        self._s3_bucket, self._s3_prefix = s3_bucket, prefix

    async def push_to_s3(
        self,
        s3_bucket: str | None = None,
        *,
        prefix: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Sync the local model directory **to** S3.

        Args:
            s3_bucket: The bucket to upload to. If omitted, the bucket from the most recent pull/push is reused.
            prefix: Optional prefix; if omitted, reuse the stored prefix from the last pull/push.
            verbose: When *True*, stream `aws` output; otherwise, suppress it.
        """
        s3_bucket = s3_bucket or self._s3_bucket
        if s3_bucket is None:
            raise ValueError(
                "No S3 bucket specified — pass `s3_bucket=` explicitly or call `pull_from_s3` first."
            )
        prefix = prefix if prefix is not None else self._s3_prefix

        await self.api()._push_to_s3(
            self, s3_bucket=s3_bucket, prefix=prefix, verbose=verbose
        )
        # Remember for next time.
        self._s3_bucket, self._s3_prefix = s3_bucket, prefix


class TrainableModel(Model):
    base_model: str
    trainable: bool = True

    # The fields within `_internal_config` are unstable and subject to change.
    # Use at your own risk.
    _internal_config: dev.InternalModelConfig | None = None

    async def register(
        self,
        api: "LocalAPI",
        _openai_client_config: dev.OpenAIServerConfig | None = None,
    ) -> None:
        await super().register(api)
        base_url, api_key = await api._prepare_backend_for_training(
            self, _openai_client_config
        )
        self.base_url = base_url
        self.api_key = api_key

    def openai_client(
        self,
    ) -> AsyncOpenAI:
        if self.base_url is None or self.api_key is None:
            raise ValueError(
                "OpenAI client not yet available. You must call `model.register()` first."
            )
        openai_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=1200, connect=5.0),
                limits=httpx.Limits(
                    max_connections=100_000, max_keepalive_connections=100_000
                ),
            ),
        )
        patch_openai(openai_client)
        return openai_client

    async def get_step(self) -> int:
        """
        Get the model's current training step.
        """
        return await self.api()._get_step(self)

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        await self.api()._delete_checkpoints(self, best_checkpoint_metric)

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        await self.api()._train_model(
            self, list(trajectory_groups), config, _config or {}
        )
