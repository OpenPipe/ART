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
from typing_extensions import override

from ..trajectories import TrajectoryGroup

ResponseT = TypeVar("ResponseT")


class Model(BaseModel):
    id: str
    entity: str
    project: str
    name: str
    base_model: str


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
    epsilon: float | None
    epsilon_high: float | None
    importance_sampling_level: (
        Literal["token", "sequence", "average", "geometric_average"] | None
    )
    kimi_k2_tau: float | None
    learning_rate: float | None
    mask_prob_ratio: bool | None
    max_negative_advantage_importance_sampling_weight: float | None
    ppo: bool | None
    precalculate_logprobs: bool | None
    scale_rewards: bool | None


class TrainingJob(BaseModel):
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
                    trajectory_group.model_dump()
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
        # Disable retries for POST requests
        if options.method.upper() == "POST":
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
