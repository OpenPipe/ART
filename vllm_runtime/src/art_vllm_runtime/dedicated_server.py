"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
from collections import OrderedDict
from functools import lru_cache
import hashlib
from http import HTTPStatus
from ipaddress import ip_address
import json
import os
import socket
import time
from typing import Any
import uuid

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

from art_vllm_runtime.binary_routes import (
    PIPELINE_ROUTES_ENV,
    PIPELINE_ROUTES_PROTOCOL,
    _register_model_route_layout,
)
from art_vllm_runtime.fast_metrics import FastMetricsSidecar
from art_vllm_runtime.patches import apply_vllm_runtime_patches

ART_SERVING_PROTOCOL_VERSION = 5
_LORA_PREPARATION_POLL_S = 0.05
_LORA_PREPARATION_TIMEOUT_S = 300.0
_LORA_UPDATE_RECEIPT_CAPACITY = 4096
_runtime_state: dict[str, object] = {}
_auth_tokens: list[str] = []
_fast_metrics_port: int | None = None


class _CompletedLoraUpdateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str
    result: dict[str, object]


class _CompletedLoraUpdates:
    """Bound response-loss replay for completed update operations."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("LoRA update receipt capacity must be positive")
        self.capacity = capacity
        self._lock = asyncio.Lock()
        self._receipts: OrderedDict[str, _CompletedLoraUpdateReceipt] = OrderedDict()
        self._active: dict[str, tuple[str, asyncio.Future[dict[str, object]]]] = {}

    async def reserve(
        self, operation_id: str, fingerprint: str
    ) -> tuple[bool, asyncio.Future[dict[str, object]]]:
        async with self._lock:
            receipt = self._receipts.get(operation_id)
            if receipt is not None:
                if receipt.fingerprint != fingerprint:
                    raise ValueError("LoRA update operation identity changed")
                self._receipts.move_to_end(operation_id)
                completed = asyncio.get_running_loop().create_future()
                completed.set_result(dict(receipt.result))
                return False, completed
            active = self._active.get(operation_id)
            if active is not None:
                active_fingerprint, completion = active
                if active_fingerprint != fingerprint:
                    raise ValueError("LoRA update operation identity changed")
                return False, completion
            completion = asyncio.get_running_loop().create_future()
            self._active[operation_id] = (fingerprint, completion)
            return True, completion

    async def settle(
        self, operation_id: str, fingerprint: str, result: dict[str, object]
    ) -> None:
        async with self._lock:
            existing = self._receipts.get(operation_id)
            if existing is not None:
                if existing.fingerprint != fingerprint or existing.result != result:
                    raise RuntimeError("completed LoRA update receipt changed")
                self._receipts.move_to_end(operation_id)
                return
            active = self._active.pop(operation_id, None)
            if active is None or active[0] != fingerprint:
                raise RuntimeError("LoRA update operation was not reserved")
            if len(self._receipts) >= self.capacity:
                self._receipts.popitem(last=False)
            self._receipts[operation_id] = _CompletedLoraUpdateReceipt(
                fingerprint=fingerprint, result=dict(result)
            )
            active[1].set_result(dict(result))

    async def abandon(self, operation_id: str, fingerprint: str) -> None:
        async with self._lock:
            active = self._active.get(operation_id)
            if active is None or active[0] != fingerprint:
                return
            self._active.pop(operation_id)
            active[1].cancel()


_completed_lora_updates = _CompletedLoraUpdates(_LORA_UPDATE_RECEIPT_CAPACITY)


def _patch_prebound_listener_tcp_nodelay(api_server: Any) -> None:
    create_server_socket = api_server.create_server_socket

    def create_tcp_server_socket(*args: Any, **kwargs: Any) -> socket.socket:
        listener = create_server_socket(*args, **kwargs)
        # vLLM pre-binds before Uvicorn; accepted sockets inherit this option.
        listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return listener

    api_server.create_server_socket = create_tcp_server_socket


def _art_metrics_snapshot() -> dict[str, Any]:
    from art_vllm_runtime.metrics import get_art_metrics_snapshot

    snapshot = get_art_metrics_snapshot()
    snapshot.update(
        process_uuid=_runtime_state["process_uuid"],
        generation=_runtime_state["generation"],
    )
    return snapshot


def _fast_metrics_url(request: Any) -> str:
    if _fast_metrics_port is None:
        raise RuntimeError("ART fast metrics listener is not running")
    host = request.url.hostname
    if host is None:
        raise RuntimeError("ART capabilities request has no host")
    try:
        address = ip_address(host.strip("[]"))
        unspecified = address.is_unspecified
        loopback = address.is_loopback
    except ValueError:
        unspecified = False
        loopback = host.casefold() == "localhost"
    nnodes = _runtime_state.get("nnodes", 1)
    if isinstance(nnodes, bool) or not isinstance(nnodes, int):
        raise RuntimeError("ART runtime state has invalid nnodes")
    if unspecified or (nnodes > 1 and loopback):
        raise RuntimeError(
            f"ART fast metrics cannot advertise unroutable host {host!r}"
        )
    return str(
        request.url.replace(
            scheme="http",
            port=_fast_metrics_port,
            path="/art/metrics",
            query="",
            fragment="",
        )
    )


class _ArtAuthenticationMiddleware(AuthenticationMiddleware):
    def __init__(self, app: Any) -> None:
        super().__init__(app, tokens=_auth_tokens)

    def __call__(self, scope: Scope, receive: Receive, send: Send):
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if (
            scope.get("type") in {"http", "websocket"}
            and scope.get("method") != "OPTIONS"
            and path.startswith("/art/")
            and not self.verify_token(Headers(scope=scope))
        ):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


class _ResetPrefixCacheRequest(BaseModel):
    reset_running_requests: bool = False
    reset_connector: bool = True


class _InFlightLoraUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    expected_generation_id: str = Field(min_length=1)
    policy_version: int = Field(ge=0)
    lora_slot: str | None = Field(default=None, min_length=1)
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False


def _valid_preparation_result(value: object, *, ready: bool) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("workers")) is int
        and value["workers"] > 0
        and value.get("ready") is ready
    )


def _lora_update_fingerprint(body: _InFlightLoraUpdateRequest) -> str:
    payload = json.dumps(
        body.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


async def _reserve_lora_update(
    body: _InFlightLoraUpdateRequest, fingerprint: str
) -> tuple[bool, JSONResponse | None]:
    try:
        owner, completion = await _completed_lora_updates.reserve(
            body.operation_id, fingerprint
        )
    except ValueError as error:
        return False, JSONResponse(
            content={"error": str(error), "type": "idempotency_conflict"},
            status_code=HTTPStatus.CONFLICT.value,
        )
    if owner:
        return True, None
    try:
        result = await asyncio.shield(completion)
    except asyncio.CancelledError:
        if not completion.cancelled():
            raise
        return False, JSONResponse(
            content={
                "error": "The original LoRA update did not complete; retry the operation",
                "type": "operation_retry",
            },
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )
    return False, JSONResponse(content=result)


def _index_shared_pp_partition(config: Any, pp_size: int) -> tuple[int, ...] | None:
    if pp_size <= 1 or not hasattr(config, "index_topk"):
        return None
    layer_count = int(config.num_hidden_layers)
    pattern = getattr(config, "index_topk_pattern", None)
    offset = int(getattr(config, "index_skip_topk_offset", 2))
    frequency = int(getattr(config, "index_topk_freq", 1))

    def computes_index(layer: int) -> bool:
        if pattern is not None and layer < len(pattern):
            return pattern[layer] != "S"
        return max(layer - offset + 1, 0) % frequency == 0

    boundaries = tuple(
        layer for layer in range(1, layer_count) if computes_index(layer)
    )

    @lru_cache
    def solve(start: int, remaining: int) -> tuple[int, int, tuple[int, ...]] | None:
        if remaining == 1:
            length = layer_count - start
            return length + 1, length * length, (length,)
        candidates = []
        for end in boundaries:
            if end <= start:
                continue
            suffix = solve(end, remaining - 1)
            if suffix is None:
                continue
            length = end - start
            candidates.append(
                (
                    max(length + (start == 0), suffix[0]),
                    length * length + suffix[1],
                    (length, *suffix[2]),
                )
            )
        return min(candidates) if candidates else None

    result = solve(0, pp_size)
    if result is None:
        raise ValueError(
            f"cannot partition {layer_count} index-sharing layers across PP{pp_size}"
        )
    return result[2]


def _configure_index_shared_pp(model: str, engine_args: dict[str, Any]) -> str | None:
    pp_size = int(engine_args.get("pipeline_parallel_size", 1))
    if pp_size <= 1:
        return None
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model,
        revision=engine_args.get("revision"),
        trust_remote_code=bool(engine_args.get("trust_remote_code", False)),
    )
    partition = _index_shared_pp_partition(config, pp_size)
    if partition is None:
        return os.environ.get("VLLM_PP_LAYER_PARTITION")
    value = ",".join(map(str, partition))
    configured = os.environ.setdefault("VLLM_PP_LAYER_PARTITION", value)
    if configured != value:
        raise ValueError(
            "VLLM_PP_LAYER_PARTITION conflicts with ART's index-sharing-safe "
            f"partition: configured={configured!r}, required={value!r}"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ART dedicated vLLM server")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--replica-generation", type=int, default=0)
    parser.add_argument("--process-uuid")
    parser.add_argument("--update-identity")
    parser.add_argument("--initial-generation-id")
    parser.add_argument("--initial-policy-version", type=int)
    parser.add_argument("--lora-path", help="Optional initial checkpoint path")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--engine-args-json", default="{}", help="Additional engine args as JSON"
    )
    parser.add_argument(
        "--server-args-json",
        default="{}",
        help="Additional server args as JSON (tool_call_parser, etc.)",
    )
    return parser.parse_args(argv)


def _patch_art_runtime_routes() -> None:
    from fastapi import APIRouter, Depends, FastAPI, Query, Request
    from fastapi.responses import JSONResponse, Response
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.chat_completion.api_router import (
        create_chat_completion,
    )
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.serve.utils.api_utils import validate_json_request

    from art_vllm_runtime.binary_routes import (
        capture_routed_experts,
        encode_routed_experts_response,
    )

    if getattr(api_server, "_art_runtime_routes_patched", False):
        return

    original_build_app = api_server.build_app
    original_init_app_state = api_server.init_app_state

    def art_build_app(*build_args: object, **build_kwargs: object) -> FastAPI:
        app = original_build_app(*build_args, **build_kwargs)
        router = APIRouter()

        def engine(request: Request):
            return request.app.state.engine_client

        @router.post("/sleep")
        async def sleep(
            raw_request: Request,
            level: int = Query(default=1, ge=0, le=2),
            mode: str = Query(default="abort", pattern="^(abort|wait|keep)$"),
        ) -> JSONResponse:
            try:
                await engine(raw_request).sleep(level=level, mode=mode)
            except ValueError as err:
                return JSONResponse(
                    content={"error": str(err)},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            return JSONResponse(
                content={"status": "sleeping", "level": level, "mode": mode}
            )

        @router.post("/wake_up")
        async def wake_up(raw_request: Request) -> JSONResponse:
            await engine(raw_request).wake_up()
            return JSONResponse(content={"status": "awake"})

        @router.get("/is_sleeping")
        async def is_sleeping(raw_request: Request) -> JSONResponse:
            return JSONResponse(
                content={"is_sleeping": await engine(raw_request).is_sleeping()}
            )

        @router.get("/art/state")
        async def art_state() -> JSONResponse:
            return JSONResponse(content=dict(_runtime_state))

        @router.get("/art/metrics")
        async def art_metrics() -> JSONResponse:
            return JSONResponse(content=_art_metrics_snapshot())

        @router.get("/art/capabilities")
        async def art_capabilities(raw_request: Request) -> JSONResponse:
            return JSONResponse(
                content={
                    "runtime": "art_vllm",
                    "protocol_version": ART_SERVING_PROTOCOL_VERSION,
                    "binary_routed_experts": True,
                    "fast_metrics": {"url": _fast_metrics_url(raw_request)},
                    "inplace_lora_load": True,
                    "in_flight_lora_updates": True,
                    "policy_token_spans": True,
                }
            )

        @router.post(
            "/art/v1/chat/completions",
            dependencies=[Depends(validate_json_request)],
        )
        async def binary_chat_completion(
            request: ChatCompletionRequest, raw_request: Request
        ) -> Response:
            if request.stream:
                return JSONResponse(
                    content={"error": "ART binary routed experts require stream=false"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            with capture_routed_experts() as routes:
                response = await create_chat_completion(request, raw_request)
            if response is None:
                return Response(status_code=499)
            if response.status_code >= HTTPStatus.BAD_REQUEST.value:
                return response
            if not routes:
                return JSONResponse(
                    content={"error": "vLLM returned no routed experts"},
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                )
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in {"content-length", "content-type"}
            }
            return Response(
                content=encode_routed_experts_response(response.body, routes),
                media_type="application/vnd.art.routed-experts-v2",
                headers=headers,
            )

        @router.post("/art/reset_prefix_cache")
        async def reset_prefix_cache(
            body: _ResetPrefixCacheRequest, raw_request: Request
        ) -> JSONResponse:
            success = await engine(raw_request).reset_prefix_cache(
                reset_running_requests=body.reset_running_requests,
                reset_connector=body.reset_connector,
            )
            return JSONResponse(content={"success": success})

        @router.post("/art/in_flight_lora_update")
        async def in_flight_lora_update(
            body: _InFlightLoraUpdateRequest, raw_request: Request
        ) -> JSONResponse:
            from vllm.entrypoints.openai.engine.protocol import ErrorResponse
            from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest

            from art_vllm_runtime.policy_spans import (
                PolicyLoRARequest,
                _complete_task,
                lora_update_coordinator,
                policy_lora_request_payload,
                publish_lora_slot_policy,
                register_lora_alias,
            )

            public_model_name = body.model_name
            lora_path = body.lora_path
            generation_id = body.generation_id
            policy_version = body.policy_version
            lora_slot = body.lora_slot or public_model_name.rsplit("@", 1)[0]
            models = raw_request.app.state.openai_serving_models
            engine_client = engine(raw_request)
            coordinator = lora_update_coordinator(models, engine_client)
            request_fingerprint = _lora_update_fingerprint(body)
            if generation_id == body.expected_generation_id:
                return JSONResponse(
                    content={"error": "generation_id must advance"},
                    status_code=HTTPStatus.CONFLICT.value,
                )
            owns_operation, reserved_response = await _reserve_lora_update(
                body, request_fingerprint
            )
            if not owns_operation:
                assert reserved_response is not None
                return reserved_response

            update_seq: int | None = None
            mutation_started = False
            preparation_attempted = False
            published = False
            try:
                loaded = models.lora_requests.get(lora_slot)
                if loaded is None:
                    await _completed_lora_updates.abandon(
                        body.operation_id, request_fingerprint
                    )
                    return JSONResponse(
                        content={"error": f"LoRA slot {lora_slot!r} is not loaded"},
                        status_code=HTTPStatus.NOT_FOUND.value,
                    )
                lora_int_id = loaded.lora_int_id
                prepare_request = PolicyLoRARequest(
                    lora_name=lora_slot,
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                    base_model_name=(
                        body.base_model_name
                        if body.base_model_name is not None
                        and models.is_base_model(body.base_model_name)
                        else None
                    ),
                    load_inplace=True,
                    is_3d_lora_weight=body.is_3d_lora_weight,
                    generation_id=generation_id,
                    policy_version=policy_version,
                    update_seq=0,
                )
                prepare_payload = policy_lora_request_payload(prepare_request)
                preparation_attempted = True
                preparation = await engine_client.engine_core.call_utility_async(
                    "art_prepare_lora_policy", body.operation_id, prepare_payload
                )
                deadline = time.monotonic() + _LORA_PREPARATION_TIMEOUT_S
                while not _valid_preparation_result(preparation, ready=True):
                    if not _valid_preparation_result(preparation, ready=False):
                        raise RuntimeError(
                            "EngineCore returned an invalid LoRA preparation state"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out preparing policy LoRA")
                    await asyncio.sleep(_LORA_PREPARATION_POLL_S)
                    preparation = await engine_client.engine_core.call_utility_async(
                        "art_lora_policy_status", body.operation_id, prepare_payload
                    )

                async with models.lora_resolver_lock[lora_slot]:
                    load_request = LoadLoRAAdapterRequest(
                        lora_name=lora_slot,
                        lora_path=lora_path,
                        load_inplace=True,
                        is_3d_lora_weight=body.is_3d_lora_weight,
                    )
                    load_error = await models._check_load_lora_adapter_request(
                        load_request
                    )
                    if isinstance(load_error, ErrorResponse):
                        await engine_client.engine_core.call_utility_async(
                            "art_abort_prepared_lora_policy", body.operation_id
                        )
                        preparation_attempted = False
                        await _completed_lora_updates.abandon(
                            body.operation_id, request_fingerprint
                        )
                        return JSONResponse(
                            content=load_error.model_dump(mode="python"),
                            status_code=load_error.error.code,
                        )
                    try:
                        update_seq = await coordinator.begin_update(
                            lora_slot,
                            expected_generation_id=body.expected_generation_id,
                        )
                    except RuntimeError as error:
                        await engine_client.engine_core.call_utility_async(
                            "art_abort_prepared_lora_policy", body.operation_id
                        )
                        preparation_attempted = False
                        await _completed_lora_updates.abandon(
                            body.operation_id, request_fingerprint
                        )
                        return JSONResponse(
                            content={
                                "error": str(error),
                                "type": "generation_conflict",
                            },
                            status_code=HTTPStatus.CONFLICT.value,
                        )
                    lora_request = PolicyLoRARequest(
                        lora_name=lora_slot,
                        lora_int_id=lora_int_id,
                        lora_path=lora_path,
                        base_model_name=(
                            body.base_model_name
                            if body.base_model_name is not None
                            and models.is_base_model(body.base_model_name)
                            else None
                        ),
                        load_inplace=True,
                        is_3d_lora_weight=body.is_3d_lora_weight,
                        generation_id=generation_id,
                        policy_version=policy_version,
                        update_seq=update_seq,
                    )
                    mutation_started = True
                    transaction = await engine_client.engine_core.call_utility_async(
                        "art_commit_prepared_lora_policy_update",
                        body.operation_id,
                        policy_lora_request_payload(lora_request),
                    )
                    if not isinstance(transaction, dict) or not isinstance(
                        transaction.get("cache_transition"), dict
                    ):
                        raise RuntimeError(
                            "EngineCore returned an invalid LoRA update acknowledgement"
                        )
                    cache_transition = transaction["cache_transition"]
                    serving_request = PolicyLoRARequest(
                        **{
                            **policy_lora_request_payload(lora_request),
                            "load_inplace": False,
                        }
                    )
                    models.lora_requests[lora_slot] = serving_request
                    register_lora_alias(
                        models,
                        public_model_name=public_model_name,
                        lora_slot=lora_slot,
                    )
                    publish_lora_slot_policy(
                        models,
                        lora_slot=lora_slot,
                        generation_id=generation_id,
                        policy_version=policy_version,
                        update_seq=update_seq,
                    )
                    await coordinator.commit_update(lora_slot, serving_request)
                    published = True
                    mutation_started = False
                    preparation_attempted = False
                _runtime_state.update(
                    loaded_adapter=public_model_name,
                    generation_id=generation_id,
                    policy_version=policy_version,
                    update_identity=f"lora:{lora_slot}:{generation_id}:{update_seq}",
                )
                result: dict[str, object] = {
                    "status": "updated",
                    "model_name": public_model_name,
                    "lora_slot": lora_slot,
                    "generation_id": generation_id,
                    "policy_version": policy_version,
                    "update_seq": update_seq,
                    "cache_transition": cache_transition,
                }
                interrupted = await _complete_task(
                    asyncio.create_task(
                        _completed_lora_updates.settle(
                            body.operation_id, request_fingerprint, result
                        )
                    )
                )
                if interrupted is not None:
                    raise interrupted
                return JSONResponse(content=result)
            except BaseException:
                if preparation_attempted and not mutation_started and not published:
                    try:
                        await asyncio.shield(
                            engine_client.engine_core.call_utility_async(
                                "art_abort_prepared_lora_policy", body.operation_id
                            )
                        )
                    except BaseException:
                        pass
                if mutation_started:
                    try:
                        await asyncio.shield(
                            engine_client.engine_core.call_utility_async(
                                "pause_scheduler", "abort", True
                            )
                        )
                    finally:
                        await asyncio.shield(
                            coordinator.fail_update(lora_slot, update_seq)
                        )
                if update_seq is not None and not mutation_started and not published:
                    await asyncio.shield(
                        coordinator.cancel_update(lora_slot, update_seq)
                    )
                if not published:
                    await asyncio.shield(
                        _completed_lora_updates.abandon(
                            body.operation_id, request_fingerprint
                        )
                    )
                raise

        app.include_router(router)
        return app

    async def art_init_app_state(
        engine_client: Any, state: Any, *args: Any, **kwargs: Any
    ) -> None:
        await original_init_app_state(engine_client, state, *args, **kwargs)
        policy_version = _runtime_state.get("initial_policy_version")
        if policy_version is None:
            return
        from art_vllm_runtime.policy_spans import declare_initial_lora_policy

        await declare_initial_lora_policy(
            state.openai_serving_models,
            engine_client,
            lora_slot=str(_runtime_state["loaded_adapter"]),
            generation_id=str(_runtime_state["initial_generation_id"]),
            policy_version=int(policy_version),
        )

    setattr(api_server, "build_app", art_build_app)
    setattr(api_server, "init_app_state", art_init_app_state)
    setattr(api_server, "_art_runtime_routes_patched", True)


def _append_cli_arg(vllm_args: list[str], key: str, value: object) -> None:
    cli_key = f"--{key.replace('_', '-')}"
    match value:
        case True:
            vllm_args.append(cli_key)
        case False:
            vllm_args.append(f"--no-{key.replace('_', '-')}")
        case None:
            return
        case str() | int() | float():
            vllm_args.append(f"{cli_key}={value}")
        case dict():
            vllm_args.append(f"{cli_key}={json.dumps(value)}")
        case list():
            if key == "lora_target_modules":
                vllm_args.append(cli_key)
                for item in value:
                    match item:
                        case str() | int() | float():
                            vllm_args.append(str(item))
                        case dict():
                            vllm_args.append(json.dumps(item))
                        case _:
                            assert False, (
                                f"Unsupported CLI list item for {key}: {type(item)}"
                            )
                return
            for item in value:
                match item:
                    case str() | int() | float():
                        vllm_args.append(f"{cli_key}={item}")
                    case dict():
                        vllm_args.append(f"{cli_key}={json.dumps(item)}")
                    case _:
                        assert False, (
                            f"Unsupported CLI list item for {key}: {type(item)}"
                        )
        case _:
            assert False, f"Unsupported CLI arg for {key}: {type(value)}"


def _patch_engine_config(
    engine_args_type: Any,
    *,
    pipeline_route_capture: bool,
) -> None:
    current = engine_args_type.create_engine_config
    create_engine_config = getattr(current, "__art_original__", current)
    if not pipeline_route_capture:
        setattr(engine_args_type, "create_engine_config", create_engine_config)
        return

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        config = create_engine_config(self, *args, **kwargs)
        config.model_config.enable_return_routed_experts = True
        _register_model_route_layout(config.model_config)
        _validate_pipeline_route_config(config)
        return config

    create.__art_original__ = create_engine_config  # type: ignore[attr-defined]
    setattr(engine_args_type, "create_engine_config", create)


def _validate_pipeline_route_config(config: Any) -> None:
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size <= 1
        or parallel.distributed_executor_backend != "mp"
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or config.use_v2_model_runner
    ):
        raise ValueError(
            "pipeline routed-expert capture requires V1 mp execution, PP > 1, "
            "DP = 1, and prefill/decode CP = 1"
        )
    transfer = config.kv_transfer_config
    if transfer is not None and transfer.is_kv_transfer_instance:
        raise ValueError(
            "pipeline routed-expert capture is incompatible with KV connectors"
        )


def main(argv: list[str] | None = None) -> None:
    global _fast_metrics_port

    args = parse_args(argv)
    if (args.initial_generation_id is None) != (args.initial_policy_version is None):
        raise ValueError(
            "--initial-generation-id and --initial-policy-version must be set together"
        )
    engine_args = json.loads(args.engine_args_json)
    server_args = json.loads(args.server_args_json)
    route_capture = engine_args.get("enable_return_routed_experts", False)
    pp_size = engine_args.get("pipeline_parallel_size", 1)
    if not isinstance(route_capture, bool):
        raise ValueError("enable_return_routed_experts must be a boolean")
    if isinstance(pp_size, bool) or not isinstance(pp_size, int):
        raise ValueError("pipeline_parallel_size must be an integer")
    pp_layer_partition = _configure_index_shared_pp(args.model, engine_args)
    critical_engine_args = {
        "data_parallel_size",
        "decode_context_parallel_size",
        "distributed_executor_backend",
        "enable_return_routed_experts",
        "kv_transfer_config",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
    }
    misplaced = critical_engine_args.intersection(server_args)
    if misplaced:
        raise ValueError(
            f"engine arguments passed as server arguments: {sorted(misplaced)}"
        )
    pipeline_route_capture = route_capture and pp_size > 1
    if pipeline_route_capture:
        engine_args["enable_return_routed_experts"] = False
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0").lower() not in {
            "0",
            "false",
        }:
            raise ValueError("pipeline routed-expert capture requires vLLM V1")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ[PIPELINE_ROUTES_ENV] = PIPELINE_ROUTES_PROTOCOL
    else:
        os.environ.pop(PIPELINE_ROUTES_ENV, None)

    process_uuid = args.process_uuid or uuid.uuid4().hex

    _runtime_state.update(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        process_uuid=process_uuid,
        generation=args.replica_generation,
        node_rank=args.node_rank,
        nnodes=args.nnodes,
        headless=args.headless,
        loaded_adapter=args.served_model_name if args.lora_path else None,
        generation_id=args.initial_generation_id,
        policy_version=args.initial_policy_version
        if args.initial_policy_version is not None
        else (
            int(args.served_model_name.rsplit("@", 1)[1])
            if "@" in args.served_model_name
            and args.served_model_name.rsplit("@", 1)[1].isdigit()
            else None
        ),
        update_identity=args.update_identity,
        initial_generation_id=args.initial_generation_id,
        initial_policy_version=args.initial_policy_version,
        pp_layer_partition=pp_layer_partition,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    apply_vllm_runtime_patches()

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    _patch_prebound_listener_tcp_nodelay(api_server)
    _patch_art_runtime_routes()
    _patch_engine_config(
        AsyncEngineArgs,
        pipeline_route_capture=pipeline_route_capture,
    )

    vllm_args = [
        f"--model={args.model}",
        f"--port={args.port}",
        f"--host={args.host}",
        f"--served-model-name={args.served_model_name}",
        "--enable-lora",
    ]
    if args.nnodes > 1:
        vllm_args.extend(
            [
                f"--nnodes={args.nnodes}",
                f"--node-rank={args.node_rank}",
                f"--master-addr={args.master_addr}",
                f"--master-port={args.master_port}",
            ]
        )
        if args.headless:
            vllm_args.append("--headless")
    if args.lora_path:
        vllm_args.append(f"--lora-modules={args.served_model_name}={args.lora_path}")
    for extra_args in (engine_args, server_args):
        for key, value in extra_args.items():
            _append_cli_arg(vllm_args, key, value)

    vllm_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    vllm_parser = make_arg_parser(vllm_parser)
    namespace = vllm_parser.parse_args(vllm_args)
    if api_key := os.environ.pop("VLLM_API_KEY", None):
        namespace.api_key = [api_key]
    _auth_tokens[:] = namespace.api_key or []
    if _auth_tokens:
        namespace.middleware = [
            *namespace.middleware,
            "art_vllm_runtime.dedicated_server._ArtAuthenticationMiddleware",
        ]
    validate_parsed_serve_args(namespace)
    if args.headless:
        from vllm.entrypoints.cli.serve import run_headless

        namespace.api_server_count = 0
        run_headless(namespace)
    else:
        from art_vllm_runtime.metrics import set_fast_metrics_writer

        metrics_sidecar = FastMetricsSidecar.start(
            args.host,
            _auth_tokens,
            process_uuid=process_uuid,
            generation=args.replica_generation,
        )
        _fast_metrics_port = metrics_sidecar.port
        try:
            set_fast_metrics_writer(metrics_sidecar.writer)
            asyncio.run(api_server.run_server(namespace))
        finally:
            _fast_metrics_port = None
            set_fast_metrics_writer(None)
            metrics_sidecar.close()


if __name__ == "__main__":
    main()
