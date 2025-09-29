import asyncio
import os
from typing import AsyncIterator, Literal, TypedDict, cast

import httpx
from openai._base_client import AsyncAPIClient, AsyncPaginator, make_request_options
from openai._compat import cached_property
from openai._qs import Querystring
from openai._resource import AsyncAPIResource
from openai._types import NOT_GIVEN, NotGiven, Omit
from openai._utils import is_mapping, maybe_transform
from openai._version import __version__
from openai.pagination import AsyncCursorPage
from openai.resources.files import AsyncFiles  # noqa: F401
from openai.resources.models import AsyncModels  # noqa: F401
from typing_extensions import override

from openai import AsyncOpenAI, BaseModel, _exceptions

from .trajectories import TrajectoryGroup


class Checkpoint(BaseModel):
    id: str
    model_id: str
    step: int


class CheckpointListParams(TypedDict, total=False):
    model_id: str


class Checkpoints(AsyncAPIResource):
    async def retrieve(
        self, *, model_id: str, step: int | Literal["latest"]
    ) -> Checkpoint:
        return await self._get(
            f"/preview/models/{model_id}/checkpoints/{step}",
            cast_to=Checkpoint,
        )

    def list(
        self,
        *,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        model_id: str,
    ) -> AsyncPaginator[Checkpoint, AsyncCursorPage[Checkpoint]]:
        return self._get_api_list(
            f"/preview/models/{model_id}/checkpoints",
            page=AsyncCursorPage[Checkpoint],
            options=make_request_options(
                # extra_headers=extra_headers,
                # extra_query=extra_query,
                # extra_body=extra_body,
                # timeout=timeout,
                query=maybe_transform(
                    {
                        "after": after,
                        "limit": limit,
                    },
                    CheckpointListParams,
                ),
            ),
            model=Checkpoint,
        )


class Model(BaseModel):
    id: str
    entity: str
    project: str
    name: str
    base_model: str

    async def get_step(self) -> int:
        raise NotImplementedError

    async def train(self, trajectory_groups: list[TrajectoryGroup]) -> None:
        raise NotImplementedError


class ModelListParams(TypedDict, total=False):
    after: str
    """A cursor for use in pagination.

    `after` is an object ID that defines your place in the list. For instance, if
    you make a list request and receive 100 objects, ending with obj_foo, your
    subsequent call can include after=obj_foo in order to fetch the next page of the
    list.
    """

    limit: int
    """A limit on the number of objects to be returned.

    Limit can range between 1 and 100, and the default is 20.
    """

    # order: Literal["asc", "desc"]
    # """Sort order by the `created_at` timestamp of the objects.

    # `asc` for ascending order and `desc` for descending order.
    # """

    entity: str
    project: str
    name: str
    base_model: str


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
        return self._patch_model(
            await self._post(
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
        )

    async def list(
        self,
        *,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        # order: Literal["asc", "desc"] | NotGiven = NOT_GIVEN,
        entity: str | NotGiven = NOT_GIVEN,
        project: str | NotGiven = NOT_GIVEN,
        name: str | NotGiven = NOT_GIVEN,
        base_model: str | NotGiven = NOT_GIVEN,
        # # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # # The extra values given here take precedence over values defined on the client or passed to this method.
        # extra_headers: Headers | None = None,
        # extra_query: Query | None = None,
        # extra_body: Body | None = None,
        # timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[Model]:
        """
        Lists the currently available models, and provides basic information about each
        one such as the owner and availability.
        """
        async for model in self._get_api_list(
            "/preview/models",
            page=AsyncCursorPage[Model],
            options=make_request_options(
                # extra_headers=extra_headers,
                # extra_query=extra_query,
                # extra_body=extra_body,
                # timeout=timeout,
                query=maybe_transform(
                    {
                        "after": after,
                        "limit": limit,
                        # "order": order,
                        "entity": entity,
                        "project": project,
                        "name": name,
                        "base_model": base_model,
                    },
                    ModelListParams,
                ),
            ),
            model=Model,
        ):
            yield self._patch_model(model)

    def _patch_model(self, model: Model) -> Model:
        """Patch model instance with async method implementations."""

        async def get_step() -> int:
            return 0

        model.get_step = get_step

        async def train(trajectory_groups: list[TrajectoryGroup]) -> None:
            training_job = await cast("Client", self._client).training_jobs.create(
                model_id=model.id,
                trajectory_groups=trajectory_groups,
            )
            while training_job.status != "COMPLETED":
                await asyncio.sleep(1)
                training_job = await cast(
                    "Client", self._client
                ).training_jobs.retrieve(training_job.id)

        model.train = train
        return model


class TrainingJob(BaseModel):
    id: int
    status: str


class TrainingJobs(AsyncAPIResource):
    async def create(
        self,
        *,
        model_id: str,
        trajectory_groups: list[TrajectoryGroup],
    ) -> TrainingJob:
        return await self._post(
            "/preview/training-jobs",
            cast_to=TrainingJob,
            body={
                "model_id": model_id,
                "trajectory_groups": [
                    trajectory_group.model_dump()
                    for trajectory_group in trajectory_groups
                ],
            },
        )

    async def retrieve(self, training_job_id: int) -> TrainingJob:
        return await self._get(
            f"/preview/training-jobs/{training_job_id}",
            cast_to=TrainingJob,
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
            base_url=base_url or "http://0.0.0.0:8000/v1",
            _strict_response_validation=False,
        )

    @cached_property
    def models(self) -> Models:
        return Models(cast(AsyncOpenAI, self))

    @cached_property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(cast(AsyncOpenAI, self))

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
