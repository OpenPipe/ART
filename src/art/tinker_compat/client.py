from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
import os
import secrets
import threading
import time
from typing import Any, Literal, Protocol, TypeVar
import uuid

import tinker
from tinker import APIFuture

from art._source_revision import art_source_revision
from art.serverless.client import RemoteTrainingClient, RemoteTrainingServiceClient
from art.serverless.contracts import (
    CreateTrainingRunRequest,
    TrainingCapabilities,
    TrainingRunSpec,
)
from art.training.client import TrainingClient as CanonicalTrainingClient
from art.training.contracts import (
    COMMAND_CONTRACT_VERSION,
    PACKING_CONTRACT_VERSION,
    AdamConfig,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    LoadStateResult,
    LossConfig,
    OptimStepRequest,
    SamplerPublication,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.utils.lifecycle import process_shutdown_timeout

from ._runtime import AsyncRuntime
from .data import to_tinker_forward_output, to_tokenized_batch, validate_loss
from .errors import UnsupportedCapabilityError
from .sampling import SamplingClient, SamplingProvider, SamplingTarget

T = TypeVar("T")

_ATTENTION_TARGETS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_proj",
        "kv_a_proj_with_mqa",
        "o_a_proj",
        "o_b_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "compressor.kv_proj",
        "compressor.gate_proj",
    }
)
_MLP_TARGETS = frozenset({"gate_proj", "up_proj", "down_proj", "experts"})


class ManagedTrainingClient(CanonicalTrainingClient, Protocol):
    async def shutdown(self) -> None: ...


class TrainingClientFactory(Protocol):
    def __call__(
        self, service: Any, request: CreateTrainingRunRequest
    ) -> Awaitable[ManagedTrainingClient]: ...


def _default_target_modules(base_model: str) -> Sequence[str]:
    from art.megatron.model_support import default_target_modules_for_model

    return default_target_modules_for_model(base_model)


class ServiceClient:
    """Tinker-compatible client facade for Remote Training."""

    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        remote_service = kwargs.pop("remote_service", None)
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        request_timeout_s = kwargs.pop("request_timeout_s", 30.0)
        max_retries = kwargs.pop("max_retries", 3)
        self._training_client_factory: TrainingClientFactory = kwargs.pop(
            "training_client_factory", RemoteTrainingClient.create
        )
        self._sampling_provider: SamplingProvider | None = kwargs.pop(
            "sampling_provider", None
        )
        self._tokenizer_factory: Callable[[str], Any] | None = kwargs.pop(
            "tokenizer_factory", None
        )
        self._target_modules_resolver: Callable[[str], Sequence[str]] = kwargs.pop(
            "target_modules_resolver", _default_target_modules
        )
        self._run_name_factory: Callable[[], str] = kwargs.pop(
            "run_name_factory", lambda: f"tinker-{uuid.uuid4().hex}"
        )
        if kwargs:
            raise TypeError(f"unsupported ServiceClient options: {sorted(kwargs)}")
        if remote_service is None:
            api_key = api_key or os.environ.get("REMOTE_TRAINING_API_KEY")
            base_url = base_url or os.environ.get("REMOTE_TRAINING_BASE_URL")
            if not api_key or not base_url:
                raise ValueError(
                    "Remote Training requires base_url and api_key; "
                    "REMOTE_TRAINING_BASE_URL and REMOTE_TRAINING_API_KEY are accepted"
                )
            remote_service = RemoteTrainingServiceClient(
                api_key=api_key,
                base_url=base_url,
                request_timeout_s=request_timeout_s,
                max_retries=max_retries,
            )
        self._service = remote_service
        self._runtime = AsyncRuntime()
        self._user_metadata = dict(user_metadata or {})
        self._project_id = project_id
        self._clients: list[TrainingClient] = []
        self._sampling_targets: dict[str, SamplingTarget] = {}
        self._closed = False

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        return self._runtime.submit(
            self._create_lora_training_client(
                base_model,
                rank,
                seed,
                train_mlp,
                train_attn,
                train_unembed,
                user_metadata,
            )
        ).result()

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        return await self._runtime.submit(
            self._create_lora_training_client(
                base_model,
                rank,
                seed,
                train_mlp,
                train_attn,
                train_unembed,
                user_metadata,
            )
        )

    async def _create_lora_training_client(
        self,
        base_model: str,
        rank: int,
        seed: int | None,
        train_mlp: bool,
        train_attn: bool,
        train_unembed: bool,
        user_metadata: dict[str, str] | None,
    ) -> TrainingClient:
        self._ensure_open()
        targets = self._resolve_targets(
            base_model,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
        await self._validate_capabilities(rank)
        spec = TrainingRunSpec(
            run_name=self._run_name_factory(),
            base_model=base_model,
            adapter={"rank": rank, "target_modules": targets},
            seed=seed if seed is not None else secrets.randbelow(2**31),
            packing_contract_version=PACKING_CONTRACT_VERSION,
            art_version=art_source_revision(),
            metadata=self._metadata(user_metadata),
        )
        return await self._create_training_client(
            spec, CreateTrainingRunRequest(spec=spec)
        )

    def create_training_client_from_state(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return self._runtime.submit(
            self._create_training_client_from_state(
                path,
                user_metadata,
                weights_access_token,
                restore_optimizer=False,
            )
        ).result()

    async def create_training_client_from_state_async(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return await self._runtime.submit(
            self._create_training_client_from_state(
                path,
                user_metadata,
                weights_access_token,
                restore_optimizer=False,
            )
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return self._runtime.submit(
            self._create_training_client_from_state(
                path,
                user_metadata,
                weights_access_token,
                restore_optimizer=True,
            )
        ).result()

    async def create_training_client_from_state_with_optimizer_async(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return await self._runtime.submit(
            self._create_training_client_from_state(
                path,
                user_metadata,
                weights_access_token,
                restore_optimizer=True,
            )
        )

    async def _create_training_client_from_state(
        self,
        path: str,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
        *,
        restore_optimizer: bool,
    ) -> TrainingClient:
        self._ensure_open()
        if weights_access_token is not None:
            raise UnsupportedCapabilityError(
                "weights_access_token is Tinker auth; use native Remote Training "
                "credentials"
            )
        source_run_id, kind, checkpoint = _parse_checkpoint_path(path)
        if kind != "training":
            raise ValueError("training state must use a /weights/ checkpoint path")
        source = await self._service.get_run(source_run_id)
        await self._validate_capabilities(source.spec.adapter.rank)
        spec = source.spec.model_copy(
            update={
                "run_name": self._run_name_factory(),
                "metadata": self._metadata(user_metadata),
            }
        )
        return await self._create_training_client(
            spec,
            CreateTrainingRunRequest(
                spec=spec,
                checkpoint=checkpoint,
                restore_optimizer=restore_optimizer,
            ),
        )

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: object | None = None,
    ) -> SamplingClient:
        return self._runtime.submit(
            self._create_sampling_client(model_path, base_model, retry_config)
        ).result()

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: object | None = None,
    ) -> SamplingClient:
        return await self._runtime.submit(
            self._create_sampling_client(model_path, base_model, retry_config)
        )

    async def _create_sampling_client(
        self,
        model_path: str | None,
        base_model: str | None,
        retry_config: object | None,
    ) -> SamplingClient:
        self._ensure_open()
        if retry_config is not None:
            raise UnsupportedCapabilityError(
                "Tinker RetryConfig cannot be applied to an external inference provider"
            )
        if (model_path is None) == (base_model is None):
            raise ValueError("provide exactly one of model_path or base_model")
        if self._sampling_provider is None:
            raise UnsupportedCapabilityError(
                "RemoteTrainingClient has no sampling API; configure sampling_provider"
            )
        if model_path is not None:
            target = self._sampling_targets.get(model_path)
            if target is None:
                run_id, _kind, _checkpoint = _parse_checkpoint_path(model_path)
                run = await self._service.get_run(run_id)
                target = SamplingTarget(
                    base_model=run.spec.base_model,
                    model_path=model_path,
                )
        else:
            assert base_model is not None
            target = SamplingTarget(base_model=base_model)
        return SamplingClient(
            self._runtime,
            self._sampling_provider,
            target,
            self._tokenizer_factory,
        )

    def get_server_capabilities(self) -> tinker.types.GetServerCapabilitiesResponse:
        raise UnsupportedCapabilityError(
            "Remote Training capabilities do not expose supported model names or "
            "context lengths required by Tinker's capability response"
        )

    async def get_server_capabilities_async(
        self,
    ) -> tinker.types.GetServerCapabilitiesResponse:
        return self.get_server_capabilities()

    def create_rest_client(self) -> None:
        raise UnsupportedCapabilityError(
            "Tinker project, audit, and REST control-plane APIs are outside this profile"
        )

    def get_telemetry(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            self._runtime.stop()
            return
        deadline = time.monotonic() + process_shutdown_timeout(1)
        future = self._runtime.submit_future(self._close())
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            future.cancel()
            self._closed = True
            self._runtime.stop(max(0.0, deadline - time.monotonic()))

    async def close_async(self) -> None:
        if self._closed:
            await asyncio.to_thread(self._runtime.stop)
            return
        deadline = time.monotonic() + process_shutdown_timeout(1)
        future = self._runtime.submit_future(self._close())
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                await asyncio.wrap_future(future)
        finally:
            future.cancel()
            self._closed = True
            await asyncio.to_thread(
                self._runtime.stop, max(0.0, deadline - time.monotonic())
            )

    async def _close(self) -> None:
        failures = await asyncio.gather(
            *(client._close_remote() for client in self._clients),
            return_exceptions=True,
        )
        try:
            await self._service.close()
        except BaseException as error:
            failures.append(error)
        self._closed = True
        errors = [error for error in failures if isinstance(error, BaseException)]
        if errors:
            raise BaseExceptionGroup("Tinker compatibility close failed", errors)

    def _remember_sampling_target(
        self, path: str, result: SamplerWeightsResult, base_model: str
    ) -> None:
        self._sampling_targets[path] = SamplingTarget(
            base_model=base_model,
            model_path=path,
            lora=result.lora,
        )

    def _metadata(self, user_metadata: dict[str, str] | None) -> dict[str, str]:
        metadata = {**self._user_metadata, **(user_metadata or {})}
        if self._project_id is not None:
            metadata.setdefault("project_id", self._project_id)
        return metadata

    async def _create_training_client(
        self, spec: TrainingRunSpec, request: CreateTrainingRunRequest
    ) -> TrainingClient:
        remote = await self._training_client_factory(self._service, request)
        client = TrainingClient(self, remote, spec)
        self._clients.append(client)
        return client

    async def _validate_capabilities(self, rank: int) -> None:
        capabilities: TrainingCapabilities = await self._service.capabilities()
        if capabilities.command_contract_version != COMMAND_CONTRACT_VERSION:
            raise RuntimeError("Remote Training command contract is incompatible")
        if PACKING_CONTRACT_VERSION not in capabilities.packing_contract_versions:
            raise RuntimeError("Remote Training packing contract is incompatible")
        missing = {"cross_entropy", "importance_sampling", "ppo", "cispo"} - set(
            capabilities.supported_losses
        )
        if missing:
            raise RuntimeError(f"Remote Training is missing losses: {sorted(missing)}")
        if "bfloat16" not in capabilities.supported_dtypes:
            raise RuntimeError("Remote Training does not support bfloat16")
        if rank > capabilities.max_lora_rank:
            raise ValueError(
                f"LoRA rank {rank} exceeds Remote Training maximum "
                f"{capabilities.max_lora_rank}"
            )

    def _resolve_targets(
        self,
        base_model: str,
        *,
        train_mlp: bool,
        train_attn: bool,
        train_unembed: bool,
    ) -> tuple[str, ...]:
        if train_unembed:
            raise UnsupportedCapabilityError(
                "current Megatron handler profiles expose no unembedding LoRA target; "
                "pass train_unembed=False"
            )
        if not train_mlp and not train_attn:
            raise ValueError("at least one LoRA target group must be enabled")
        defaults = tuple(self._target_modules_resolver(base_model))
        if train_mlp and train_attn:
            return defaults
        known = _ATTENTION_TARGETS | _MLP_TARGETS
        unknown = sorted(set(defaults) - known)
        if unknown:
            raise UnsupportedCapabilityError(
                "cannot map Tinker's broad target switches for handler targets "
                f"{unknown}"
            )
        enabled = (_MLP_TARGETS if train_mlp else frozenset()) | (
            _ATTENTION_TARGETS if train_attn else frozenset()
        )
        selected = tuple(target for target in defaults if target in enabled)
        if not selected:
            raise UnsupportedCapabilityError(
                "the selected Tinker target group has no targets in this model handler"
            )
        return selected

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ServiceClient is closed")

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class TrainingClient:
    def __init__(
        self,
        service: ServiceClient,
        remote: ManagedTrainingClient,
        spec: TrainingRunSpec,
    ) -> None:
        self._service = service
        self._runtime = service._runtime
        self._remote = remote
        self._spec = spec
        self.model_id = remote.run_id
        self._submission_lock = threading.Lock()
        self._admission_tail: Future[None] = Future()
        self._admission_tail.set_result(None)
        self._closed = False

    def forward(
        self,
        data: list[tinker.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        return self._forward(data, loss_fn, loss_fn_config, backward=False)

    async def forward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        return self.forward(data, loss_fn, loss_fn_config)

    def forward_backward(
        self,
        data: list[tinker.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        return self._forward(data, loss_fn, loss_fn_config, backward=True)

    async def forward_backward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        return self.forward_backward(data, loss_fn, loss_fn_config)

    def _forward(
        self,
        data: list[tinker.Datum],
        loss_fn: object,
        loss_fn_config: dict[str, float] | None,
        *,
        backward: bool,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        loss_name = validate_loss(loss_fn)
        batch, target_shapes = to_tokenized_batch(data, loss_name)
        loss = LossConfig(
            name=loss_name,
            normalize_advantages=False,
            values=dict(loss_fn_config or {}),
        )
        request_id = uuid.uuid4().hex
        request_type = ForwardBackwardRequest if backward else ForwardRequest
        submit = self._remote.forward_backward if backward else self._remote.forward
        return self._schedule(
            lambda sequence_id: request_type(
                run_id=self._remote.run_id,
                request_id=request_id,
                sequence_id=sequence_id,
                batch=batch,
                loss=loss,
            ),
            submit,
            lambda result: to_tinker_forward_output(result, target_shapes),
        )

    def forward_backward_custom(
        self,
        data: list[tinker.Datum],
        loss_fn: Callable[
            [list[tinker.Datum], list[Any]], tuple[Any, dict[str, float]]
        ],
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        del data, loss_fn, loss_type_input
        raise UnsupportedCapabilityError(
            "forward_backward_custom is outside the Remote Training profile"
        )

    async def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: Callable[
            [list[tinker.Datum], list[Any]], tuple[Any, dict[str, float]]
        ],
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        return self.forward_backward_custom(
            data, loss_fn, loss_type_input=loss_type_input
        )

    def optim_step(
        self, adam_params: tinker.AdamParams
    ) -> APIFuture[tinker.OptimStepResponse]:
        optimizer = AdamConfig(
            learning_rate=adam_params.learning_rate,
            beta1=adam_params.beta1,
            beta2=adam_params.beta2,
            eps=adam_params.eps,
            weight_decay=adam_params.weight_decay,
            grad_clip_norm=adam_params.grad_clip_norm,
        )
        request_id = uuid.uuid4().hex
        return self._schedule(
            lambda sequence_id: OptimStepRequest(
                run_id=self._remote.run_id,
                request_id=request_id,
                sequence_id=sequence_id,
                optimizer=optimizer,
            ),
            self._remote.optim_step,
            lambda result: tinker.OptimStepResponse(metrics=dict(result.metrics)),
        )

    async def optim_step_async(
        self, adam_params: tinker.AdamParams
    ) -> APIFuture[tinker.OptimStepResponse]:
        return self.optim_step(adam_params)

    def save_state(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> APIFuture[tinker.types.SaveWeightsResponse]:
        _validate_checkpoint_name(name)
        request_id = uuid.uuid4().hex
        return self._schedule(
            lambda sequence_id: SaveStateRequest(
                run_id=self._remote.run_id,
                request_id=request_id,
                sequence_id=sequence_id,
                checkpoint_name=name,
                ttl_seconds=ttl_seconds,
                overwrite=overwrite,
            ),
            self._remote.save_state,
            lambda result: tinker.types.SaveWeightsResponse(
                path=_checkpoint_path(result, "weights")
            ),
        )

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> APIFuture[tinker.types.SaveWeightsResponse]:
        return self.save_state(name, ttl_seconds, overwrite)

    def load_state(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[tinker.types.LoadWeightsResponse]:
        return self._load_state(path, weights_access_token, restore_optimizer=False)

    async def load_state_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[tinker.types.LoadWeightsResponse]:
        return self.load_state(path, weights_access_token)

    def load_state_with_optimizer(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[tinker.types.LoadWeightsResponse]:
        return self._load_state(path, weights_access_token, restore_optimizer=True)

    async def load_state_with_optimizer_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[tinker.types.LoadWeightsResponse]:
        return self.load_state_with_optimizer(path, weights_access_token)

    def _load_state(
        self,
        path: str,
        weights_access_token: str | None,
        *,
        restore_optimizer: bool,
    ) -> APIFuture[tinker.types.LoadWeightsResponse]:
        if weights_access_token is not None:
            raise UnsupportedCapabilityError(
                "weights_access_token is Tinker auth; use native Remote Training "
                "credentials"
            )
        run_id, kind, checkpoint = _parse_checkpoint_path(path)
        if kind != "training":
            raise ValueError("training state must use a /weights/ checkpoint path")
        if run_id != self._remote.run_id:
            raise UnsupportedCapabilityError(
                "load_state can only address checkpoints in the current canonical run; "
                "cross-run checkpoint identity is absent from LoadStateRequest"
            )
        request_id = uuid.uuid4().hex
        submit = (
            self._remote.load_state_with_optimizer
            if restore_optimizer
            else self._remote.load_state
        )
        return self._schedule(
            lambda sequence_id: LoadStateRequest(
                run_id=self._remote.run_id,
                request_id=request_id,
                sequence_id=sequence_id,
                checkpoint=checkpoint,
                restore_optimizer=restore_optimizer,
            ),
            submit,
            lambda result: _to_load_response(
                result, path=path, restore_optimizer=restore_optimizer
            ),
        )

    def save_weights_for_sampler(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[tinker.types.SaveWeightsForSamplerResponse]:
        _validate_checkpoint_name(name)
        request_id = uuid.uuid4().hex

        def convert(
            result: SamplerWeightsResult,
        ) -> tinker.types.SaveWeightsForSamplerResponse:
            path = _checkpoint_path(result, "sampler_weights")
            self._service._remember_sampling_target(path, result, self._spec.base_model)
            return tinker.types.SaveWeightsForSamplerResponse(path=path)

        return self._schedule(
            lambda sequence_id: SaveWeightsForSamplerRequest(
                run_id=self._remote.run_id,
                request_id=request_id,
                sequence_id=sequence_id,
                checkpoint_name=name,
                ttl_seconds=ttl_seconds,
                publication=SamplerPublication(mode="none"),
            ),
            self._remote.save_weights_for_sampler,
            convert,
        )

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[tinker.types.SaveWeightsForSamplerResponse]:
        return self.save_weights_for_sampler(name, ttl_seconds)

    def create_sampling_client(
        self, model_path: str, retry_config: object | None = None
    ) -> SamplingClient:
        return self._service.create_sampling_client(
            model_path=model_path, retry_config=retry_config
        )

    async def create_sampling_client_async(
        self, model_path: str, retry_config: object | None = None
    ) -> SamplingClient:
        return await self._service.create_sampling_client_async(
            model_path=model_path, retry_config=retry_config
        )

    def save_weights_and_get_sampling_client(
        self, name: str | None = None, retry_config: object | None = None
    ) -> SamplingClient:
        checkpoint_name = name or f"ephemeral-{uuid.uuid4().hex}"
        path = self.save_weights_for_sampler(checkpoint_name).result().path
        return self.create_sampling_client(path, retry_config)

    async def save_weights_and_get_sampling_client_async(
        self, name: str | None = None, retry_config: object | None = None
    ) -> SamplingClient:
        checkpoint_name = name or f"ephemeral-{uuid.uuid4().hex}"
        future = await self.save_weights_for_sampler_async(checkpoint_name)
        path = (await future).path
        return await self.create_sampling_client_async(path, retry_config)

    def get_info(self) -> tinker.types.GetInfoResponse:
        return tinker.types.GetInfoResponse(
            model_data={"model_name": self._spec.base_model},
            model_id=self.model_id,
            is_lora=True,
            lora_rank=self._spec.adapter.rank,
            model_name=self._spec.base_model,
        )

    async def get_info_async(self) -> tinker.types.GetInfoResponse:
        return self.get_info()

    def get_tokenizer(self) -> Any:
        if self._service._tokenizer_factory is None:
            raise UnsupportedCapabilityError(
                "Remote Training does not expose a pinned tokenizer revision; "
                "configure tokenizer_factory"
            )
        return self._service._tokenizer_factory(self._spec.base_model)

    def get_telemetry(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._runtime.submit(self._close_remote()).result(
            timeout=process_shutdown_timeout(2)
        )

    async def close_async(self) -> None:
        if self._closed:
            return
        async with asyncio.timeout(process_shutdown_timeout(2)):
            await self._runtime.submit(self._close_remote())

    async def _close_remote(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._remote.shutdown()

    def _schedule(
        self,
        make_request: Callable[[int], Any],
        submit: Callable[[Any], Awaitable[Any]],
        convert: Callable[[Any], T],
    ) -> APIFuture[T]:
        with self._submission_lock:
            if self._closed:
                raise RuntimeError("TrainingClient is closed")
            predecessor = self._admission_tail
            admitted: Future[None] = Future()
            self._admission_tail = admitted

        async def execute() -> T:
            try:
                await asyncio.wrap_future(predecessor)
                operation = await submit(make_request(self._remote.next_sequence_id))
            finally:
                if not admitted.done():
                    admitted.set_result(None)
            return convert(await operation.result())

        return self._runtime.submit(execute())


def _checkpoint_path(
    result: SaveStateResult | SamplerWeightsResult | LoadStateResult,
    kind: str,
) -> str:
    checkpoint = result.checkpoint
    return f"tinker://{checkpoint.run_id}/{kind}/{checkpoint.checkpoint_id}"


def _parse_checkpoint_path(path: str) -> tuple[str, str, str]:
    parsed = tinker.ParsedCheckpointTinkerPath.from_tinker_path(path)
    checkpoint = path[9:].split("/", 2)[2]
    return parsed.training_run_id, parsed.checkpoint_type, checkpoint


def _to_load_response(
    result: LoadStateResult, *, path: str, restore_optimizer: bool
) -> tinker.types.LoadWeightsResponse:
    if result.optimizer_restored != restore_optimizer:
        raise RuntimeError(
            "Remote Training returned the wrong optimizer restoration state"
        )
    return tinker.types.LoadWeightsResponse(path=path)


def _validate_checkpoint_name(name: str) -> None:
    if not name or "/" in name:
        raise ValueError("checkpoint name must be one nonempty URI segment")
