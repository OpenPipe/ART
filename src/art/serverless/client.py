from datetime import datetime
import math
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
from pydantic import ConfigDict, Field, FiniteFloat, model_validator
from typing_extensions import override

from art.training import (
    CheckpointArchiveRef,
    CheckpointRef,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    NamedLossOutcome,
    OperationResult,
    OptimStepResult,
    PackedInputCaptureRef,
    RunCommand,
    RunInitialState,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    ServiceCheckpointSource,
    TokenLogprobs,
    TrainingInputObjectRef,
    TrainingOutcome,
    TrainingRunSpec,
    WandbArtifactCheckpointSource,
)

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
    "save_weights_for_sampler",
]
NativeOperationStatus = Literal[
    "admitted",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


class NativeTrainingRun(BaseModel):
    run_id: str
    run_name: str | None = None
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
    admitted_at: datetime
    execution_started_at: datetime | None
    execution_ended_at: datetime | None
    cancel_requested: bool
    latest_event_cursor: int
    result_available: bool
    result: dict[str, Any] | None = None
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


class NativeTrainingResult(BaseModel):
    operation_id: str
    kind: NativeOperationKind
    result: dict[str, Any]


class NativeTrainingResultRelease(BaseModel):
    operation_id: str
    request_id: str
    released: Literal[True]


class RemoteSamplerPublicationResult(OperationResult):
    """Logical sampler identity returned by the public serverless boundary."""

    kind: Literal["save_sampler"] = "save_sampler"
    checkpoint: CheckpointRef
    target: Literal["active", "saved_generation"]
    model_alias: str = Field(min_length=1, max_length=255)
    generation_id: str = Field(min_length=1, max_length=255)


class _PublicWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PublicCheckpointRef(_PublicWireModel):
    checkpoint_id: str
    learner_version: int = Field(ge=0)


class _PublicPackingOutcome(_PublicWireModel):
    packed_sequence_length: int = Field(ge=1)
    packed_sequences: int = Field(ge=0)
    target_packed_sequences: int = Field(ge=1)
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)
    packed_capacity_tokens: int = Field(ge=0)
    padding_tokens: int = Field(ge=0)


class _PublicPackingLeafShape(_PublicWireModel):
    matrix_id: str = Field(min_length=1, max_length=255)
    token_ids: tuple[int, ...] = Field(min_length=1)
    shareable_length: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_shareable_length(self) -> "_PublicPackingLeafShape":
        if self.shareable_length > len(self.token_ids):
            raise ValueError("shareable_length exceeds packing token count")
        return self


class _PublicPackedGroupShape(_PublicWireModel):
    leaves: tuple[_PublicPackingLeafShape, ...] = Field(min_length=1)


class _PublicPackedInputRef(_PublicWireModel):
    capture_id: str
    manifest_sha256: str
    content_sha256: str | None
    input_object: TrainingInputObjectRef | None = None


class _PublicTrainingResult(_PublicWireModel):
    operation_id: str
    kind: NativeOperationKind
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    packing: _PublicPackingOutcome | None = None
    training: TrainingOutcome | None = None
    loss: NamedLossOutcome | None = None
    produced_gradient: bool | None = None
    token_logprobs: tuple[TokenLogprobs, ...] = ()
    group_shapes: tuple[_PublicPackedGroupShape, ...] = ()
    packed_input: _PublicPackedInputRef | None = None
    contributing_operation_ids: tuple[str, ...] | None = None
    checkpoint: _PublicCheckpointRef | None = None
    learner_version: int | None = Field(default=None, ge=0)
    target: Literal["active", "saved_generation"] | None = None
    model_alias: str | None = None
    generation_id: str | None = None
    archive: CheckpointArchiveRef | None = None
    optimizer_restored: bool | None = None


class _PublicResultRef(_PublicWireModel):
    result_id: str
    media_type: Literal[
        "application/json",
        "application/vnd.coreweave.training-result+msgpack; version=1",
    ]
    size_bytes: int = Field(ge=1)
    expires_at: datetime


class _PublicOperationError(_PublicWireModel):
    code: str


class _PublicTrainingOperation(_PublicWireModel):
    operation_id: str
    run_id: str
    request_id: str
    sequence_id: int
    kind: NativeOperationKind
    status: NativeOperationStatus
    learner_parent_version: int
    reserved_output_learner_version: int | None
    admitted_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    cancel_requested: bool
    latest_event_cursor: int
    result_summary: _PublicTrainingResult | None
    result_ref: _PublicResultRef | None
    error: _PublicOperationError | None


class _NativePublicWireAdapter:
    """Translate between ART contracts and the frozen public training wire."""

    @staticmethod
    def command(request: RunCommand) -> dict[str, Any]:
        body = request.model_dump(mode="json", exclude={"run_id"})
        if isinstance(request, (ForwardRequest, ForwardBackwardRequest)):
            if isinstance(
                request.batch, (PackedInputCaptureRef, TrainingInputObjectRef)
            ):
                raise ValueError(
                    "native serverless commands require a public raw training batch"
                )
            if request.loss.name == "ppo":
                raise ValueError("native serverless commands do not support PPO")
            values = dict(request.loss.values)
            allowed = {"clip_low_threshold", "clip_high_threshold"}
            unsupported = sorted(set(values) - allowed)
            if unsupported:
                raise ValueError(
                    "native serverless loss has unsupported values: "
                    + ", ".join(unsupported)
                )
            if request.loss.name != "cispo" and values:
                raise ValueError(
                    "native serverless clip thresholds are supported only for CISPO"
                )
            loss: dict[str, Any] = {
                "name": request.loss.name,
                "normalize_advantages": request.loss.normalize_advantages,
            }
            for key, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"native serverless loss value {key!r} must be finite"
                    )
                loss[key] = float(value)
            body["loss"] = loss
        elif isinstance(request, SaveWeightsForSamplerRequest):
            target = {
                "in_flight_lora": "active",
                "versioned_lora": "saved_generation",
            }.get(request.publication.mode)
            if target is None or request.publication.model_alias is None:
                raise ValueError(
                    "native serverless publication requires active or saved-generation LoRA"
                )
            body["publication"] = {
                "target": target,
                "model_alias": request.publication.model_alias,
            }
        elif isinstance(request, LoadStateRequest):
            checkpoint = request.checkpoint
            if checkpoint.startswith("service-checkpoint:"):
                source = ServiceCheckpointSource(
                    checkpoint_id=checkpoint.removeprefix("service-checkpoint:")
                )
            elif checkpoint.startswith("wandb-artifact:"):
                source = WandbArtifactCheckpointSource(
                    artifact=checkpoint.removeprefix("wandb-artifact:")
                )
            else:
                source = ServiceCheckpointSource(checkpoint_id=checkpoint)
            body["source"] = source.model_dump(mode="json")
            del body["checkpoint"]
        return body

    @classmethod
    def operation(cls, public: _PublicTrainingOperation) -> NativeTrainingOperation:
        result = None
        if public.result_summary is not None:
            if public.result_summary.kind != public.kind:
                raise ValueError("public operation result kind changed")
            result = cls._result_payload(public.run_id, public.result_summary)
        return NativeTrainingOperation(
            operation_id=public.operation_id,
            run_id=public.run_id,
            request_id=public.request_id,
            sequence_id=public.sequence_id,
            kind=public.kind,
            status=public.status,
            learner_parent_version=public.learner_parent_version,
            reserved_output_learner_version=public.reserved_output_learner_version,
            admitted_at=public.admitted_at,
            execution_started_at=public.started_at,
            execution_ended_at=public.ended_at,
            cancel_requested=public.cancel_requested,
            latest_event_cursor=public.latest_event_cursor,
            result_available=(
                public.result_summary is not None or public.result_ref is not None
            ),
            result=result,
            error=(
                None if public.error is None else public.error.model_dump(mode="python")
            ),
        )

    @classmethod
    def result(cls, run_id: str, public: _PublicTrainingResult) -> NativeTrainingResult:
        return NativeTrainingResult(
            operation_id=public.operation_id,
            kind=public.kind,
            result=cls._result_payload(run_id, public),
        )

    @staticmethod
    def _checkpoint(run_id: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("public checkpoint result must be an object")
        return {"run_id": run_id, **value}

    @classmethod
    def _result_payload(
        cls, run_id: str, public: _PublicTrainingResult
    ) -> dict[str, Any]:
        payload = public.model_dump(mode="json", exclude_unset=True)
        kind = public.kind
        if kind in {"forward", "forward_backward"}:
            packing = payload.get("packing")
            if not isinstance(packing, dict):
                raise ValueError("public forward result has no packing object")
            packing["group_shapes"] = payload.pop("group_shapes", [])
            packed_input = payload.pop("packed_input", None)
            if packed_input is not None:
                if not isinstance(packed_input, dict):
                    raise ValueError("public packed-input result must be an object")
                payload["packed_input_capture"] = {
                    "kind": "captured",
                    "run_id": run_id,
                    **packed_input,
                }
            result_type = (
                ForwardBackwardResult if kind == "forward_backward" else ForwardResult
            )
        elif kind == "optim_step":
            payload["contributing_forward_backward_operation_ids"] = payload.pop(
                "contributing_operation_ids"
            )
            learner_version = payload.pop("learner_version")
            payload["checkpoint"] = cls._checkpoint(run_id, payload["checkpoint"])
            if learner_version != payload["checkpoint"]["learner_version"]:
                raise ValueError("public optimizer learner version changed")
            result_type = OptimStepResult
        elif kind == "save_weights_for_sampler":
            learner_version = payload.pop("learner_version")
            payload["kind"] = "save_sampler"
            payload["checkpoint"] = cls._checkpoint(run_id, payload["checkpoint"])
            if learner_version != payload["checkpoint"]["learner_version"]:
                raise ValueError("public sampler learner version changed")
            result_type = RemoteSamplerPublicationResult
        elif kind == "save_state":
            if "archive" not in payload:
                raise ValueError("public save-state result has no archive")
            payload["checkpoint"] = cls._checkpoint(run_id, payload["checkpoint"])
            result_type = SaveStateResult
        elif kind == "load_state":
            payload["checkpoint"] = cls._checkpoint(run_id, payload["checkpoint"])
            result_type = LoadStateResult
        else:
            raise ValueError(f"unsupported public training result {kind!r}")
        return result_type.model_validate(payload).model_dump(mode="python")


_NATIVE_PUBLIC_WIRE = _NativePublicWireAdapter()


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
        run_name: str | None = None,
        spec: TrainingRunSpec,
        initial_state: RunInitialState | None = None,
    ) -> NativeTrainingRun:
        wire_spec = {
            "base_model": spec.base_model,
            "dtype": spec.dtype,
            "lora_rank": spec.adapter.rank,
            "lora_target_modules": sorted(spec.adapter.target_modules),
            "seed": spec.seed,
        }
        body: dict[str, Any] = {
            "request_id": request_id,
            "spec": wire_spec,
            "initial_state": (
                None if initial_state is None else initial_state.model_dump(mode="json")
            ),
        }
        if run_name is not None:
            body["run_name"] = run_name
        run = await self._post(
            "/training/runs:resolve",
            cast_to=NativeTrainingRun,
            body=body,
        )
        if run.run_name != run_name or run.spec != wire_spec:
            raise RuntimeError("native training run identity changed during resolve")
        return run

    async def get(self, run_id: str) -> NativeTrainingRun:
        return await self._get(f"/training/runs/{run_id}", cast_to=NativeTrainingRun)

    async def submit(self, request: RunCommand) -> NativeTrainingOperation:
        public = await self._post(
            f"/training/runs/{request.run_id}/{request_kind_endpoint(request)}",
            cast_to=_PublicTrainingOperation,
            body=_NATIVE_PUBLIC_WIRE.command(request),
        )
        return _NATIVE_PUBLIC_WIRE.operation(public)

    async def operation(
        self, run_id: str, operation_id: str
    ) -> NativeTrainingOperation:
        public = await self._get(
            f"/training/runs/{run_id}/operations/{operation_id}",
            cast_to=_PublicTrainingOperation,
        )
        return _NATIVE_PUBLIC_WIRE.operation(public)

    async def cancel(self, run_id: str, operation_id: str) -> NativeTrainingOperation:
        public = await self._post(
            f"/training/runs/{run_id}/operations/{operation_id}:cancel",
            cast_to=_PublicTrainingOperation,
        )
        return _NATIVE_PUBLIC_WIRE.operation(public)

    async def result(self, run_id: str, operation_id: str) -> NativeTrainingResult:
        public = await self._get(
            f"/training/runs/{run_id}/operations/{operation_id}/result",
            cast_to=_PublicTrainingResult,
        )
        return _NATIVE_PUBLIC_WIRE.result(run_id, public)

    async def release_result(
        self, run_id: str, operation_id: str, *, request_id: str
    ) -> NativeTrainingResultRelease:
        return await self._post(
            f"/training/runs/{run_id}/operations/{operation_id}/result:release",
            cast_to=NativeTrainingResultRelease,
            body={"request_id": request_id},
        )

    async def close(self, run_id: str, *, request_id: str) -> NativeTrainingRun:
        return await self._post(
            f"/training/runs/{run_id}:close",
            cast_to=NativeTrainingRun,
            body={"request_id": request_id},
        )


def request_kind_endpoint(request: RunCommand) -> str:
    kind = request_kind(request)
    if kind == "save_sampler":
        return "save_weights_for_sampler"
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
