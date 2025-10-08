import inspect
import os
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

import httpx
import tenacity
from openai import AsyncOpenAI, BaseModel, _exceptions
from openai._base_client import AsyncAPIClient, AsyncPaginator, make_request_options
from openai._compat import cached_property
from openai._qs import Querystring
from openai._resource import AsyncAPIResource
from openai._types import NOT_GIVEN, NotGiven, Omit
from openai._utils import is_mapping, maybe_transform
from openai._version import __version__
from openai.pagination import AsyncCursorPage
from typing_extensions import override

from .trajectories import TrajectoryGroup

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


@overload
def retry_status_codes(
    fn: Callable[P, AsyncPaginator[R, AsyncCursorPage[R]]],
) -> Callable[P, AsyncIterable[R]]: ...


@overload
def retry_status_codes(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...


@overload
def retry_status_codes(fn: Callable[P, R]) -> Callable[P, R]: ...


def retry_status_codes(
    fn: (
        Callable[P, R]
        | Callable[P, Awaitable[R]]
        | Callable[P, AsyncPaginator[R, AsyncCursorPage[R]]]
    ),
) -> Callable[P, R | AsyncIterable[R]] | Callable[P, Awaitable[R]]:
    def is_retryable_status(exc: BaseException) -> bool:
        if isinstance(exc, _exceptions.APIStatusError):
            response = exc.response
            if response is not None:
                status = response.status_code
                return status in {429, *range(500, 600)}
        return False

    stop = tenacity.stop_after_attempt(3)
    wait = tenacity.wait_random_exponential(multiplier=0.5, max=2.0)
    retry = tenacity.retry_if_exception(is_retryable_status)
    reraise = True

    async def retrying_awaitable(awaitable_fn: Callable[[], Awaitable[T]]) -> T:
        async for attempt in tenacity.AsyncRetrying(
            stop=stop,
            wait=wait,
            retry=retry,
            reraise=reraise,
        ):
            with attempt:
                return await awaitable_fn()

        # Unreachable if tenacity produces at least one attempt
        raise RuntimeError("retry attempt sequence unexpectedly exhausted")

    if inspect.iscoroutinefunction(fn):
        async_fn = cast(Callable[P, Awaitable[R]], fn)

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await retrying_awaitable(lambda: async_fn(*args, **kwargs))

        return async_wrapper

    async def retrying_async_iterable(
        async_paginator: AsyncPaginator[R, AsyncCursorPage[R]],
    ) -> AsyncIterable[R]:
        page = await retrying_awaitable(lambda: async_paginator)
        for item in page._get_page_items():
            yield item
        while page.has_next_page():
            page = await retrying_awaitable(lambda: page.get_next_page())
            for item in page._get_page_items():
                yield item

    sync_fn = cast(Callable[P, R], fn)

    def sync_or_async_paginator_wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> R | AsyncIterable[R]:
        for attempt in tenacity.Retrying(
            stop=stop,
            wait=wait,
            retry=retry,
            reraise=reraise,
        ):
            with attempt:
                result = sync_fn(*args, **kwargs)
                if isinstance(result, AsyncPaginator):
                    return retrying_async_iterable(result)
                return result

        # Unreachable if tenacity produces at least one attempt
        raise RuntimeError("retry attempt sequence unexpectedly exhausted")

    return sync_or_async_paginator_wrapper


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
    learning_rate: float | None
    precalculate_logprobs: bool | None


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

    @cached_property
    def checkpoints(self) -> "Checkpoints":
        return Checkpoints(cast(AsyncOpenAI, self._client))


class Checkpoints(AsyncAPIResource):
    @retry_status_codes
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

    @retry_status_codes
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
        return TrainingJobEvents(cast(AsyncOpenAI, self._client))


class TrainingJobEvents(AsyncAPIResource):
    @retry_status_codes
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
            max_retries=0,
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
