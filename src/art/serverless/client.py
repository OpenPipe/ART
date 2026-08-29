from datetime import datetime
import os
from typing import Any, Iterable, Literal, Type, TypedDict, TypeVar, cast

import httpx
from openai import AsyncOpenAI, BaseModel, _exceptions
from openai._base_client import (
    AsyncAPIClient,
    AsyncPaginator,
    make_request_options,
)
from openai._compat import cached_property
from openai._models import FinalRequestOptions
from openai._qs import Querystring
from openai._resource import AsyncAPIResource
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN, NotGiven, Omit
from openai._utils import is_mapping, maybe_transform
from openai._version import __version__
from openai.pagination import AsyncCursorPage
from pydantic import Field, model_validator
from typing_extensions import override

from art.training import LoadStateRequest, RunCommand, TrainingRunSpec

from ..trajectories import TrajectoryGroup
from ..types import SFTMetricLoggingConfig

ResponseT = TypeVar("ResponseT")


class Model(BaseModel):
    id: str
    entity: str
    project: str
    name: str
    base_model: str
    run_id: str | None


class Checkpoint(BaseModel):
    id: str
    step: int
    metrics: dict[str, float]


class CheckpointListParams(TypedDict, total=False):
    after: str
    limit: int
    order: Literal["asc", "desc"]


class DeleteCheckpointsResponse(BaseModel):
    deleted_count: int
    not_found_steps: list[int]


class ExperimentalTrainingConfig(TypedDict, total=False):
    advantage_balance: float | None
    allow_training_without_logprobs: bool | None
    epsilon: float | None
    epsilon_high: float | None
    importance_sampling_level: (
        Literal["token", "sequence", "average", "geometric_average"] | None
    )
    kimi_k2_tau: float | None
    kl_penalty_coef: float | None
    kl_penalty_reference_step: int | None
    kl_penalty_source: Literal["current_learner", "sample"] | None
    kl_penalty_step_lag: int | None
    kl_ref_adapter_path: str | None
    learning_rate: float | None
    logprob_calculation_chunk_size: int | None
    loss_fn: Literal["cispo", "ppo"] | None
    mask_prob_ratio: bool | None
    max_negative_advantage_importance_sampling_weight: float | None
    normalize_advantages: bool | None
    num_trajectories_learning_rate_multiplier_power: float | None
    packed_sequence_length: int | None
    plot_tensors: bool | None
    ppo: bool | None
    precalculate_logprobs: bool | None
    scale_learning_rate_by_reward_std_dev: bool | None
    scale_rewards: bool | None
    truncated_importance_sampling: float | None


class SFTTrainingConfig(TypedDict, total=False):
    batch_size: int | None
    learning_rate: float | list[float] | None
    assistant_turns: Literal["all", "last"]
    metric_logging: SFTMetricLoggingConfig | None


class TrainingJob(BaseModel):
    id: str


class SFTTrainingJob(BaseModel):
    id: str


class TrainingJobEventListParams(TypedDict, total=False):
    after: str
    limit: int


class TrainingJobEvent(BaseModel):
    id: str
    type: Literal[
        "training_started", "gradient_step", "training_ended", "training_failed"
    ]
    data: dict[str, Any]


NativeRunStatus = Literal["open", "closing", "closed", "failed"]
NativeOperationKind = Literal[
    "forward",
    "forward_backward",
    "optim_step",
    "save_state",
    "load_state",
    "save_weights",
]
NativeOperationStatus = Literal[
    "admitted",
    "packing",
    "ready",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


class NativeTrainingRun(BaseModel):
    run_id: str
    run_name: str
    spec: dict[str, Any]
    status: NativeRunStatus
    next_sequence_id: int
    projected_learner_version: int
    committed_learner_version: int


class NativeTrainingOperation(BaseModel):
    operation_id: str
    run_id: str
    request_id: str
    sequence_id: int
    kind: NativeOperationKind
    status: NativeOperationStatus
    learner_parent_version: int
    reserved_output_learner_version: int | None
    contributing_operation_ids: tuple[str, ...]
    command: dict[str, Any]
    command_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    admitted_at: datetime
    execution_started_at: datetime | None
    execution_ended_at: datetime | None
    cancel_requested: bool
    event_cursor: int
    result: dict[str, Any] | None
    error: dict[str, Any] | None

    @model_validator(mode="after")
    def _validate_execution_timeline(self) -> "NativeTrainingOperation":
        if (
            self.execution_started_at is not None
            and self.execution_started_at < self.admitted_at
        ):
            raise ValueError("operation execution starts before admission")
        lower = self.execution_started_at or self.admitted_at
        if self.execution_ended_at is not None and self.execution_ended_at < lower:
            raise ValueError("operation execution ends before it starts")
        return self


class Models(AsyncAPIResource):
    async def create(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
        name: str | None = None,
        base_model: str,
        return_existing: bool = False,
    ) -> Model:
        return await self._post(
            "/preview/models",
            cast_to=Model,
            body={
                "entity": entity,
                "project": project,
                "name": name,
                "base_model": base_model,
                "return_existing": return_existing,
            },
        )

    async def log(
        self,
        *,
        model_id: str,
        trajectory_groups: list[TrajectoryGroup],
        split: str,
    ) -> None:
        return await self._post(
            f"/preview/models/{model_id}/log",
            body={
                "model_id": model_id,
                "trajectory_groups": [
                    trajectory_group.model_dump(mode="json")
                    for trajectory_group in trajectory_groups
                ],
                "split": split,
            },
            cast_to=type(None),
        )

    async def delete(self, *, model_id: str) -> None:
        return await self._delete(
            f"/preview/models/{model_id}",
            cast_to=type(None),
        )

    @cached_property
    def checkpoints(self) -> "Checkpoints":
        return Checkpoints(cast(AsyncOpenAI, self._client))  # ty:ignore[redundant-cast]


class Checkpoints(AsyncAPIResource):
    def list(
        self,
        *,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        model_id: str,
        order: Literal["asc", "desc"] | NotGiven = NOT_GIVEN,
    ) -> AsyncPaginator[Checkpoint, AsyncCursorPage[Checkpoint]]:
        return self._get_api_list(
            f"/preview/models/{model_id}/checkpoints",
            page=AsyncCursorPage[Checkpoint],
            options=make_request_options(
                query=maybe_transform(
                    {
                        "after": after,
                        "limit": limit,
                        "order": order,
                    },
                    CheckpointListParams,
                ),
            ),
            model=Checkpoint,
        )

    async def delete(
        self, *, model_id: str, steps: Iterable[int]
    ) -> DeleteCheckpointsResponse:
        return await self._delete(
            f"/preview/models/{model_id}/checkpoints",
            body={"steps": steps},
            cast_to=DeleteCheckpointsResponse,
        )


class TrainingJobs(AsyncAPIResource):
    async def create(
        self,
        *,
        model_id: str,
        trajectory_groups: list[TrajectoryGroup],
        experimental_config: ExperimentalTrainingConfig | None = None,
    ) -> TrainingJob:
        return await self._post(
            "/preview/training-jobs",
            cast_to=TrainingJob,
            body={
                "model_id": model_id,
                "trajectory_groups": [
                    trajectory_group.model_dump(mode="json")
                    for trajectory_group in trajectory_groups
                ],
                "experimental_config": experimental_config,
            },
        )

    @cached_property
    def events(self) -> "TrainingJobEvents":
        return TrainingJobEvents(cast(AsyncOpenAI, self._client))  # ty:ignore[redundant-cast]


class TrainingJobEvents(AsyncAPIResource):
    def list(
        self,
        *,
        training_job_id: str,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
    ) -> AsyncPaginator[TrainingJobEvent, AsyncCursorPage[TrainingJobEvent]]:
        return self._get_api_list(
            f"/preview/training-jobs/{training_job_id}/events",
            page=AsyncCursorPage[TrainingJobEvent],
            options=make_request_options(
                query=maybe_transform(
                    {
                        "after": after,
                        "limit": limit,
                    },
                    TrainingJobEventListParams,
                ),
            ),
            model=TrainingJobEvent,
        )


class TrainingRuns(AsyncAPIResource):
    """Typed transport for the native sequenced training API."""

    async def resolve(
        self,
        *,
        request_id: str,
        run_name: str,
        spec: TrainingRunSpec,
    ) -> NativeTrainingRun:
        wire_spec = {
            "base_model": spec.base_model,
            "dtype": spec.dtype,
            "lora_rank": spec.adapter.rank,
            "lora_target_modules": list(spec.adapter.target_modules),
            "optimizer": "adamw",
        }
        run = await self._post(
            "/training/runs:resolve",
            cast_to=NativeTrainingRun,
            body={
                "request_id": request_id,
                "run_name": run_name,
                "spec": wire_spec,
            },
        )
        if run.run_name != run_name or run.spec != wire_spec:
            raise RuntimeError("native training run identity changed during resolve")
        return run

    async def get(self, run_id: str) -> NativeTrainingRun:
        return await self._get(f"/training/runs/{run_id}", cast_to=NativeTrainingRun)

    async def submit(self, request: RunCommand) -> NativeTrainingOperation:
        return await self._post(
            f"/training/runs/{request.run_id}/{request_kind_endpoint(request)}",
            cast_to=NativeTrainingOperation,
            body={
                "request_id": request.request_id,
                "sequence_id": request.sequence_id,
                "command": request.model_dump(mode="json"),
            },
        )

    async def operation(
        self, run_id: str, operation_id: str
    ) -> NativeTrainingOperation:
        return await self._get(
            f"/training/runs/{run_id}/operations/{operation_id}",
            cast_to=NativeTrainingOperation,
        )

    async def cancel(self, run_id: str, operation_id: str) -> NativeTrainingOperation:
        return await self._post(
            f"/training/runs/{run_id}/operations/{operation_id}:cancel",
            cast_to=NativeTrainingOperation,
        )

    async def close(self, run_id: str) -> NativeTrainingRun:
        return await self._post(
            f"/training/runs/{run_id}:close", cast_to=NativeTrainingRun
        )


def request_kind_endpoint(request: RunCommand) -> str:
    kind = request_kind(request)
    if kind == "save_sampler":
        return "save_weights_for_sampler"
    if kind == "load_state":
        assert isinstance(request, LoadStateRequest)
        return (
            "load_state_with_optimizer" if request.restore_optimizer else "load_state"
        )
    return kind


def request_kind(request: RunCommand) -> str:
    from art.training import (
        ForwardBackwardRequest,
        ForwardRequest,
        OptimStepRequest,
        SaveStateRequest,
        SaveWeightsForSamplerRequest,
    )

    if isinstance(request, ForwardBackwardRequest):
        return "forward_backward"
    if isinstance(request, ForwardRequest):
        return "forward"
    if isinstance(request, OptimStepRequest):
        return "optim_step"
    if isinstance(request, SaveWeightsForSamplerRequest):
        return "save_sampler"
    if isinstance(request, SaveStateRequest):
        return "save_state"
    if isinstance(request, LoadStateRequest):
        return "load_state"
    raise TypeError(f"unsupported native training request {type(request).__name__}")


class SFTTrainingJobs(AsyncAPIResource):
    async def create(
        self,
        *,
        model_id: str,
        training_data_url: str,
        config: SFTTrainingConfig | None = None,
    ) -> SFTTrainingJob:
        return await self._post(
            "/preview/sft-training-jobs",
            cast_to=SFTTrainingJob,
            body={
                "model_id": model_id,
                "training_data_url": training_data_url,
                "config": config,
            },
        )

    @cached_property
    def events(self) -> "TrainingJobEvents":
        return TrainingJobEvents(cast(AsyncOpenAI, self._client))  # ty:ignore[redundant-cast]


class Client(AsyncAPIClient):
    api_key: str

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the WANDB_API_KEY environment variable"
            )
        self.api_key = api_key
        super().__init__(
            version=__version__,
            base_url=base_url or "https://api.training.wandb.ai/v1",
            _strict_response_validation=False,
            max_retries=3,
        )

    @override
    async def request(
        self,
        cast_to: Type[ResponseT],
        options: FinalRequestOptions,
        *,
        stream: bool = False,
        stream_cls: type[AsyncStream[Any]] | None = None,
    ) -> ResponseT | AsyncStream[Any]:
        # Preview POSTs lack idempotency keys; native run commands do not.
        if options.method.upper() == "POST" and str(options.url).startswith(
            "/preview/"
        ):
            options.max_retries = 0
        return await super().request(
            cast_to=cast_to, options=options, stream=stream, stream_cls=stream_cls
        )

    @cached_property
    def models(self) -> Models:
        return Models(cast(AsyncOpenAI, self))

    @cached_property
    def training_jobs(self) -> TrainingJobs:
        return TrainingJobs(cast(AsyncOpenAI, self))

    @cached_property
    def training_runs(self) -> TrainingRuns:
        return TrainingRuns(cast(AsyncOpenAI, self))

    @cached_property
    def sft_training_jobs(self) -> SFTTrainingJobs:
        return SFTTrainingJobs(cast(AsyncOpenAI, self))

    ############################
    # AsyncOpenAI overrides #
    ############################

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="brackets")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        return {"Authorization": f"Bearer {api_key}"}

    def _auth_headers(self, security: Any | None = None) -> dict[str, str]:  # noqa: ARG002
        return self.auth_headers

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": "false",
            # "OpenAI-Organization": self.organization
            # if self.organization is not None
            # else Omit(),
            # "OpenAI-Project": self.project if self.project is not None else Omit(),
            **self._custom_headers,
        }

    @override
    def _make_status_error(
        self, err_msg: str, *, body: object, response: httpx.Response
    ) -> _exceptions.APIStatusError:
        data = body.get("error", body) if is_mapping(body) else body
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=data)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(
                err_msg, response=response, body=data
            )

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(
                err_msg, response=response, body=data
            )

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=data)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=data)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(
                err_msg, response=response, body=data
            )

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=data)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(
                err_msg, response=response, body=data
            )
        return _exceptions.APIStatusError(err_msg, response=response, body=data)
