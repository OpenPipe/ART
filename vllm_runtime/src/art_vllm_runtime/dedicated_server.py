"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
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
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from vllm.entrypoints.serve.utils.server_utils import (
    GUARDED_PREFIX,
    AuthenticationMiddleware,
)

from art_vllm_runtime.binary_routes import (
    PIPELINE_ROUTES_ENV,
    PIPELINE_ROUTES_PROTOCOL,
    _register_model_route_layout,
)
from art_vllm_runtime.fast_metrics import FastMetricsSidecar
from art_vllm_runtime.patches import apply_vllm_runtime_patches

ART_SERVING_PROTOCOL_VERSION = 8
_runtime_state: dict[str, object] = {}
_auth_tokens: list[str] = []
_fast_metrics_port: int | None = None
_LORA_UPDATE_PATH = "/art/in_flight_lora_update"
_ASGI_STARTED_AT = "art_lora_update_asgi_started_at"
_BODY_RECEIVED_AT = "art_lora_update_body_received_at"
_ROUTE_UPLOAD_MANAGER_STATE = "art_route_upload_manager"
_ROUTE_UPLOAD_ALLOWED_HOSTS_ENV = "ART_ROUTE_UPLOAD_ALLOWED_HOST_SUFFIXES"
_ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV = "ART_ROUTE_UPLOAD_TRUSTED_PRINCIPAL_PROVIDER"
_ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV = "ART_ROUTE_UPLOAD_LOCAL_PRINCIPAL"
_AUTHENTICATED_PRINCIPAL_SCOPE_KEY = "art.authenticated_principal"
_MAX_ROUTE_REQUEST_IDENTITY_BYTES = 16 << 20


class _RouteUploadPrincipalError(RuntimeError):
    pass


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


def _route_upload_owner(request: Any) -> str:
    """Read the stable principal established by trusted deployment middleware."""

    principal = request.scope.get(_AUTHENTICATED_PRINCIPAL_SCOPE_KEY)
    if isinstance(principal, str) and principal:
        return principal
    raise _RouteUploadPrincipalError(
        "route upload requires a trusted authenticated principal"
    )


def _route_upload_manager(request: Any) -> Any | None:
    return getattr(request.app.state, _ROUTE_UPLOAD_MANAGER_STATE, None)


def _route_upload_allowed_host_suffixes() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get(_ROUTE_UPLOAD_ALLOWED_HOSTS_ENV, "").split(",")
        if value.strip()
    )


def _trusted_route_principal_provider_configured() -> bool:
    value = os.environ.get(_ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV, "").strip().casefold()
    if value in {"1", "true"}:
        return True
    if value in {"", "0", "false"}:
        return False
    raise ValueError(f"{_ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV} must be true/false or 1/0")


def _local_route_upload_principal() -> str | None:
    """Return an explicit principal for unauthenticated local development only."""

    principal = os.environ.get(_ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV, "").strip()
    if not principal:
        return None
    if len(principal) > 512:
        raise ValueError(f"{_ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV} is too long")
    if _auth_tokens:
        raise ValueError(
            f"{_ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV} requires unauthenticated local mode"
        )
    if _trusted_route_principal_provider_configured():
        raise ValueError(
            f"{_ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV} cannot be combined with "
            f"{_ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV}"
        )
    return principal


class _ArtLocalRoutePrincipalMiddleware:
    """Populate the same trusted scope boundary for explicit local-only use."""

    def __init__(self, app: Any, *, principal: str) -> None:
        self.app = app
        self.principal = principal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") in {"http", "websocket"}:
            scope[_AUTHENTICATED_PRINCIPAL_SCOPE_KEY] = self.principal
        await self.app(scope, receive, send)


def _route_request_fingerprint(request: Any) -> str:
    """Hash bounded request semantics, never route or adapter object bytes."""

    payload = request.model_dump(
        mode="json",
        exclude={"route_upload", "request_id"},
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(encoded) > _MAX_ROUTE_REQUEST_IDENTITY_BYTES:
        raise ValueError("route-upload inference request exceeds identity byte limit")
    return hashlib.blake2b(encoded, digest_size=32).hexdigest()


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
        authenticated = False
        if scope.get("type") in {"http", "websocket"}:
            headers = Headers(scope=scope)
            authenticated = self.verify_token(headers)
        guarded = path.startswith("/art/") or path.startswith(GUARDED_PREFIX)
        if (
            scope.get("type") in {"http", "websocket"}
            and scope.get("method") != "OPTIONS"
            and guarded
            and not authenticated
        ):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


class _ArtRequestTimingMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if scope.get("type") != "http" or path != _LORA_UPDATE_PATH:
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state[_ASGI_STARTED_AT] = time.perf_counter()

        async def timed_receive() -> Any:
            message = await receive()
            if message["type"] == "http.request" and not message.get(
                "more_body", False
            ):
                state[_BODY_RECEIVED_AT] = time.perf_counter()
            return message

        await self.app(scope, timed_receive, send)


class _ResetPrefixCacheRequest(BaseModel):
    reset_running_requests: bool = False
    reset_connector: bool = True


class _InFlightLoraUpdateRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=256)
    lora_path: str = Field(min_length=1, max_length=16_384)
    operation_id: str = Field(min_length=1, max_length=256)
    adapter_source: str = Field(min_length=1, max_length=4_096)
    generation_id: str = Field(min_length=1, max_length=256)
    expected_generation_id: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    policy_version: int = Field(ge=0)
    lora_slot: str | None = Field(default=None, min_length=1, max_length=256)
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False


class _AppliedLoraUpdate(BaseModel):
    identity: dict[str, object]
    response: dict[str, object]


def _lora_update_identity(body: _InFlightLoraUpdateRequest) -> dict[str, object]:
    # generation_id is policy/cache identity. adapter_source is the separate,
    # immutable transport/catalog reference. The rank-local materialization path
    # may differ across holders and is intentionally not part of replay identity.
    return body.model_dump(mode="python", exclude={"lora_path"})


def _runtime_policy_version() -> int | None:
    value = _runtime_state.get("policy_version")
    return None if isinstance(value, bool) or not isinstance(value, int) else value


def _launch_policy_version_for_slot(
    *, lora_slot: str, public_model_name: str
) -> int | None:
    loaded = _runtime_state.get("loaded_adapter")
    if not isinstance(loaded, str):
        return None
    loaded_slot = (
        loaded.rsplit("@", 1)[0]
        if "@" in loaded and loaded.rsplit("@", 1)[1].isdigit()
        else loaded
    )
    if loaded in {lora_slot, public_model_name} or loaded_slot == lora_slot:
        return _runtime_policy_version()
    return None


def _launch_generation_id_for_slot(
    *, lora_slot: str, public_model_name: str
) -> str | None:
    loaded = _runtime_state.get("loaded_adapter")
    if not isinstance(loaded, str):
        return None
    loaded_slot = (
        loaded.rsplit("@", 1)[0]
        if "@" in loaded and loaded.rsplit("@", 1)[1].isdigit()
        else loaded
    )
    if loaded in {lora_slot, public_model_name} or loaded_slot == lora_slot:
        generation_id = _runtime_state.get("generation_id")
        return generation_id if isinstance(generation_id, str) else None
    return None


def _admit_lora_update(
    body: _InFlightLoraUpdateRequest,
    applied: _AppliedLoraUpdate | None,
    *,
    launch_policy_version: int | None,
    launch_generation_id: str | None,
) -> dict[str, object] | None:
    """Return a prior response for exact replay; reject ambiguous lineage."""

    identity = _lora_update_identity(body)
    if applied is not None:
        if applied.identity["operation_id"] == body.operation_id:
            if applied.identity != identity:
                raise ValueError("LoRA operation identity was reused")
            return applied.response
        prior_generation = str(applied.identity["generation_id"])
        if prior_generation == body.generation_id:
            raise ValueError("LoRA generation identity was reused by another operation")
        if body.expected_generation_id != prior_generation:
            raise ValueError("LoRA expected generation does not match holder state")
        prior_policy_version = applied.identity["policy_version"]
        if isinstance(prior_policy_version, bool) or not isinstance(
            prior_policy_version, int
        ):
            raise RuntimeError("applied LoRA update has invalid policy version")
        if body.policy_version <= prior_policy_version:
            raise ValueError("LoRA policy update is not newer")
        return None

    if body.expected_generation_id != launch_generation_id:
        raise ValueError("LoRA expected generation does not match launch state")
    if (
        launch_policy_version is not None
        and body.policy_version < launch_policy_version
    ):
        raise ValueError("LoRA policy update regresses launch policy")
    return None


def _applied_lora_updates(models: Any) -> dict[str, _AppliedLoraUpdate]:
    updates = getattr(models, "_art_applied_lora_updates", None)
    if updates is None:
        updates = {}
        setattr(models, "_art_applied_lora_updates", updates)
    return updates


def _lora_mutation_lock(models: Any, _lora_slot: str) -> asyncio.Lock:
    # Scheduler pause/resume is process-global, so adapter mutations cannot overlap.
    lock = getattr(models, "_art_lora_mutation_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(models, "_art_lora_mutation_lock", lock)
    return lock


class _UnloadLoraPolicyRequest(BaseModel):
    lora_slot: str = Field(min_length=1)


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
    parser.add_argument("--initial-policy-version", type=int)
    parser.add_argument("--initial-generation-id")
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
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.chat_completion.api_router import (
        create_chat_completion,
    )
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.api_router import create_completion
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.serve.utils.api_utils import validate_json_request

    from art_vllm_runtime.binary_routes import (
        capture_routed_experts,
        routed_experts_object_chunks,
        routed_experts_response_chunks,
    )
    from art_vllm_runtime.route_uploads import (
        PresignedPutUploader,
        RouteUploadBusy,
        RouteUploadError,
        RouteUploadForbidden,
        RouteUploadManager,
        RouteUploadNotFound,
        RouteUploadReplay,
        RouteUploadReplayResponse,
    )

    if getattr(api_server, "_art_runtime_routes_patched", False):
        return

    original_build_app = api_server.build_app
    original_init_app_state = api_server.init_app_state

    def art_build_app(*build_args: object, **build_kwargs: object) -> FastAPI:
        app = original_build_app(*build_args, **build_kwargs)
        allowed_hosts = _route_upload_allowed_host_suffixes()
        local_principal = _local_route_upload_principal()
        principal_provider_configured = (
            _trusted_route_principal_provider_configured()
            or local_principal is not None
        )
        if local_principal is not None:
            app.add_middleware(
                _ArtLocalRoutePrincipalMiddleware,
                principal=local_principal,
            )
        if allowed_hosts and principal_provider_configured:
            route_upload_manager = RouteUploadManager(
                PresignedPutUploader(allowed_host_suffixes=allowed_hosts)
            )
            setattr(app.state, _ROUTE_UPLOAD_MANAGER_STATE, route_upload_manager)
            app.router.add_event_handler("shutdown", route_upload_manager.close)
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
                    "binary_routed_experts": bool(
                        _runtime_state.get("binary_routed_experts", False)
                    ),
                    "fast_metrics": {"url": _fast_metrics_url(raw_request)},
                    "inplace_lora_load": bool(
                        _runtime_state.get("in_flight_lora_updates", False)
                    ),
                    "in_flight_lora_updates": bool(
                        _runtime_state.get("in_flight_lora_updates", False)
                    ),
                    "policy_token_spans": bool(
                        _runtime_state.get("policy_token_spans", False)
                    ),
                    "presigned_route_uploads": (
                        _route_upload_manager(raw_request) is not None
                        and bool(_runtime_state.get("binary_routed_experts", False))
                    ),
                    "model_backend": _runtime_state.get("model_backend"),
                }
            )

        @router.post(
            "/v1/chat/completions",
            dependencies=[Depends(validate_json_request)],
        )
        async def public_chat_completion(
            request: ChatCompletionRequest, raw_request: Request
        ) -> Response:
            if getattr(request, "return_policy_spans", False) and request.stream:
                return JSONResponse(
                    content={"error": "policy spans require stream=false"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            if getattr(request, "route_upload", None) is None:
                response = await create_chat_completion(request, raw_request)
                return response if response is not None else Response(status_code=499)
            if request.stream:
                return JSONResponse(
                    content={"error": "route upload requires stream=false"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            return await _generate_routed_response(
                request,
                raw_request,
                lambda: create_chat_completion(request, raw_request),
            )

        @router.post(
            "/v1/completions",
            dependencies=[Depends(validate_json_request)],
        )
        async def public_completion(
            request: CompletionRequest, raw_request: Request
        ) -> Response:
            if getattr(request, "return_policy_spans", False) and request.stream:
                return JSONResponse(
                    content={"error": "policy spans require stream=false"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            if getattr(request, "route_upload", None) is None:
                response = await create_completion(request, raw_request)
                return response if response is not None else Response(status_code=499)
            if request.stream:
                return JSONResponse(
                    content={"error": "route upload requires stream=false"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            return await _generate_routed_response(
                request,
                raw_request,
                lambda: create_completion(request, raw_request),
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
            return await _generate_routed_response(
                request,
                raw_request,
                lambda: create_chat_completion(request, raw_request),
            )

        @router.post(
            "/art/v1/completions",
            dependencies=[Depends(validate_json_request)],
        )
        async def binary_completion(
            request: CompletionRequest, raw_request: Request
        ) -> Response:
            prompt = request.prompt
            if (
                request.stream
                or request.n != 1
                or not isinstance(prompt, list)
                or not prompt
                or not all(type(token) is int for token in prompt)
                or request.add_special_tokens
                or request.return_token_ids is not True
            ):
                return JSONResponse(
                    content={
                        "error": "ART binary completions require one non-streaming "
                        "choice, exact prompt token IDs, add_special_tokens=false, "
                        "and return_token_ids=true"
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            return await _generate_routed_response(
                request,
                raw_request,
                lambda: create_completion(request, raw_request),
            )

        async def _generate_routed_response(
            request: Any,
            raw_request: Request,
            generate: Any,
        ) -> Response:
            grant = getattr(request, "route_upload", None)
            lease = None
            if grant is not None:
                if not bool(_runtime_state.get("binary_routed_experts", False)):
                    return JSONResponse(
                        content={"error": "routed-expert capture is disabled"},
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    )
                manager = _route_upload_manager(raw_request)
                if manager is None:
                    return JSONResponse(
                        content={"error": "presigned route uploads are disabled"},
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    )
                try:
                    owner_id = _route_upload_owner(raw_request)
                    request_fingerprint = _route_request_fingerprint(request)
                    reservation = await manager.reserve(
                        owner_id=owner_id,
                        request_id=grant.client_reference,
                        request_fingerprint=request_fingerprint,
                        grant=grant,
                    )
                    if isinstance(reservation, RouteUploadReplay):
                        return _restore_route_upload_response(
                            await reservation.response()
                        )
                    lease = reservation
                except ValueError as error:
                    return JSONResponse(
                        content={"error": str(error)},
                        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
                    )
                except _RouteUploadPrincipalError as error:
                    return JSONResponse(
                        content={"error": str(error)},
                        status_code=HTTPStatus.FORBIDDEN.value,
                    )
                except RouteUploadBusy as error:
                    return JSONResponse(
                        content={"error": str(error)},
                        status_code=HTTPStatus.TOO_MANY_REQUESTS.value,
                    )
                except RouteUploadError as error:
                    return JSONResponse(
                        content={"error": str(error)},
                        status_code=HTTPStatus.BAD_REQUEST.value,
                    )
            if lease is None:
                with capture_routed_experts() as routes:
                    response = await generate()
                return _routed_response(response, routes)
            assert grant is not None

            async def remember_or_reject(response: Response) -> Response | None:
                try:
                    await lease.remember_response(
                        _store_route_upload_response(response)
                    )
                except RouteUploadError as error:
                    rejection = JSONResponse(
                        content={
                            "error": str(error),
                            "route_upload": lease.future.model_dump(mode="json"),
                        },
                        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
                    )
                    # Admission reserves enough metadata for this compact,
                    # deterministic replay even when the original body is too big.
                    await lease.remember_response(
                        _store_route_upload_response(rejection)
                    )
                    await lease.fail(str(error))
                    return rejection
                return None

            async with lease:
                with capture_routed_experts() as routes:
                    response = await generate()
                if response is None:
                    cancelled = Response(status_code=499)
                    if rejection := await remember_or_reject(cancelled):
                        return rejection
                    await lease.fail("generation was cancelled")
                    return cancelled
                if response.status_code >= HTTPStatus.BAD_REQUEST.value:
                    if rejection := await remember_or_reject(response):
                        return rejection
                    await lease.fail(
                        f"generation failed with HTTP {response.status_code}"
                    )
                    return response
                if not routes:
                    missing = JSONResponse(
                        content={"error": "vLLM returned no routed experts"},
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    )
                    if rejection := await remember_or_reject(missing):
                        return rejection
                    await lease.fail("vLLM returned no routed experts")
                    return missing
                route_chunks = tuple(routed_experts_object_chunks(routes))
                actual_route_bytes = sum(map(len, route_chunks))
                if actual_route_bytes > grant.max_bytes:
                    oversize = JSONResponse(
                        content={
                            "error": "route payload exceeds its signed byte bound",
                            "route_upload": lease.future.model_dump(mode="json"),
                        },
                        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
                    )
                    if rejection := await remember_or_reject(oversize):
                        return rejection
                    await lease.fail("route payload exceeds its signed byte bound")
                    return oversize
                routed_response = _route_upload_response(response, lease.future)
                if rejection := await remember_or_reject(routed_response):
                    return rejection
                await lease.publish(route_chunks)
                return routed_response

        def _routed_response(response: Response | None, routes: Any) -> Response:
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
            chunks = routed_experts_response_chunks(bytes(response.body), routes)
            headers["content-length"] = str(sum(map(len, chunks)))

            async def body_chunks():
                for chunk in chunks:
                    yield chunk

            return StreamingResponse(
                content=body_chunks(),
                media_type="application/vnd.art.routed-experts-v2",
                headers=headers,
            )

        def _route_upload_response(response: Response, future: Any) -> Response:
            try:
                content = json.loads(bytes(response.body))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "vLLM returned a non-JSON response for route upload"
                ) from error
            content["route_upload"] = future.model_dump(mode="json")
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() != "content-length"
            }
            return JSONResponse(
                content=content,
                status_code=response.status_code,
                headers=headers,
            )

        def _store_route_upload_response(
            response: Response,
        ) -> RouteUploadReplayResponse:
            return RouteUploadReplayResponse(
                status_code=response.status_code,
                headers=tuple(response.headers.items()),
                body=bytes(response.body),
            )

        def _restore_route_upload_response(
            stored: RouteUploadReplayResponse,
        ) -> Response:
            return Response(
                content=stored.body,
                status_code=stored.status_code,
                headers=dict(stored.headers),
            )

        async def _route_upload_status(
            operation_id: str, raw_request: Request, wait_s: float
        ) -> JSONResponse:
            manager = _route_upload_manager(raw_request)
            if manager is None:
                return JSONResponse(
                    content={"error": "presigned route uploads are disabled"},
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                )
            try:
                owner_id = _route_upload_owner(raw_request)
                future = await manager.wait(
                    owner_id=owner_id,
                    operation_id=operation_id,
                    timeout_s=wait_s,
                )
            except _RouteUploadPrincipalError as error:
                return JSONResponse(
                    content={"error": str(error)},
                    status_code=HTTPStatus.FORBIDDEN.value,
                )
            except RouteUploadBusy as error:
                return JSONResponse(
                    content={"error": str(error)},
                    status_code=HTTPStatus.TOO_MANY_REQUESTS.value,
                )
            except RouteUploadNotFound as error:
                return JSONResponse(
                    content={"error": str(error)},
                    status_code=HTTPStatus.NOT_FOUND.value,
                )
            except RouteUploadForbidden as error:
                return JSONResponse(
                    content={"error": str(error)},
                    status_code=HTTPStatus.FORBIDDEN.value,
                )
            return JSONResponse(content=future.model_dump(mode="json"))

        @router.get("/v1/route_uploads/{operation_id}")
        async def route_upload_status(
            operation_id: str,
            raw_request: Request,
        ) -> JSONResponse:
            return await _route_upload_status(operation_id, raw_request, 0.0)

        @router.get("/v1/route_uploads/{operation_id}/wait")
        async def wait_for_route_upload(
            operation_id: str,
            raw_request: Request,
            timeout: float = Query(default=30.0, ge=0.0, le=30.0),
        ) -> JSONResponse:
            return await _route_upload_status(operation_id, raw_request, timeout)

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
            # This is a trusted holder-local mutation. The external inference
            # control plane authorizes tenant/slot ownership, fans the operation
            # out to eligible holders, and quarantines ambiguous failures.
            endpoint_started = time.perf_counter()
            from vllm.entrypoints.openai.engine.protocol import ErrorResponse
            from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest

            from art_vllm_runtime.policy_spans import (
                PolicyLoRARequest,
                lora_update_coordinator,
                policy_lora_request_payload,
                publish_lora_slot_policy,
                register_lora_alias,
            )

            public_model_name = body.model_name
            lora_path = body.lora_path
            policy_version = body.policy_version
            lora_slot = body.lora_slot or public_model_name.rsplit("@", 1)[0]
            models = raw_request.app.state.openai_serving_models
            engine_client = engine(raw_request)
            coordinator = lora_update_coordinator(models, engine_client)
            mutation_lock_started = time.perf_counter()
            async with _lora_mutation_lock(models, lora_slot):
                state = raw_request.scope["state"]
                asgi_started = float(state[_ASGI_STARTED_AT])
                body_received = float(state[_BODY_RECEIVED_AT])
                timings = {
                    "asgi_body_receive_s": body_received - asgi_started,
                    "body_parse_dispatch_s": endpoint_started - body_received,
                    "asgi_to_handler_s": endpoint_started - asgi_started,
                    "handler_setup_s": mutation_lock_started - endpoint_started,
                    "mutation_lock_wait_s": time.perf_counter() - mutation_lock_started,
                }
                identity = _lora_update_identity(body)
                applied = _applied_lora_updates(models).get(lora_slot)
                try:
                    replay = _admit_lora_update(
                        body,
                        applied,
                        launch_policy_version=_launch_policy_version_for_slot(
                            lora_slot=lora_slot,
                            public_model_name=public_model_name,
                        ),
                        launch_generation_id=_launch_generation_id_for_slot(
                            lora_slot=lora_slot,
                            public_model_name=public_model_name,
                        ),
                    )
                except ValueError as error:
                    return JSONResponse(
                        content={"error": str(error)},
                        status_code=HTTPStatus.CONFLICT.value,
                    )
                if replay is not None:
                    return JSONResponse(content=replay)
                phase_started = time.perf_counter()
                update_seq = await coordinator.begin_update(lora_slot)
                timings["admission_drain_s"] = time.perf_counter() - phase_started
                mutation_started = False
                try:
                    async with models.lora_resolver_lock[lora_slot]:
                        phase_started = time.perf_counter()
                        load_request = LoadLoRAAdapterRequest(
                            lora_name=lora_slot,
                            lora_path=lora_path,
                            load_inplace=lora_slot in models.lora_requests,
                            is_3d_lora_weight=body.is_3d_lora_weight,
                        )
                        load_error = await models._check_load_lora_adapter_request(
                            load_request
                        )
                        if isinstance(load_error, ErrorResponse):
                            await coordinator.cancel_update(lora_slot, update_seq)
                            return JSONResponse(
                                content=load_error.model_dump(mode="python"),
                                status_code=load_error.error.code,
                            )
                        timings["adapter_validation_s"] = (
                            time.perf_counter() - phase_started
                        )
                        lora_int_id = (
                            models.lora_requests[lora_slot].lora_int_id
                            if lora_slot in models.lora_requests
                            else models.lora_id_counter.inc(1)
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
                            policy_version=policy_version,
                            update_seq=update_seq,
                        )
                        mutation_started = True
                        phase_started = time.perf_counter()
                        await engine_client.engine_core.call_utility_async(
                            "pause_scheduler", "keep", False
                        )
                        timings["scheduler_pause_s"] = (
                            time.perf_counter() - phase_started
                        )
                        phase_started = time.perf_counter()
                        cache_transition = (
                            await engine_client.engine_core.call_utility_async(
                                "art_apply_lora_policy_update",
                                policy_lora_request_payload(lora_request),
                            )
                        )
                        timings["worker_update_s"] = time.perf_counter() - phase_started
                        phase_started = time.perf_counter()
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
                            policy_version=policy_version,
                            update_seq=update_seq,
                        )
                        timings["serving_metadata_s"] = (
                            time.perf_counter() - phase_started
                        )
                        phase_started = time.perf_counter()
                        await engine_client.engine_core.call_utility_async(
                            "resume_scheduler"
                        )
                        timings["scheduler_resume_s"] = (
                            time.perf_counter() - phase_started
                        )
                        phase_started = time.perf_counter()
                        await coordinator.commit_update(lora_slot, serving_request)
                        timings["coordinator_commit_s"] = (
                            time.perf_counter() - phase_started
                        )
                        mutation_started = False
                        timings["total_s"] = time.perf_counter() - endpoint_started
                        response = {
                            "status": "updated",
                            "operation_id": body.operation_id,
                            "model_name": public_model_name,
                            "lora_slot": lora_slot,
                            "generation_id": body.generation_id,
                            "policy_version": policy_version,
                            "update_seq": update_seq,
                            "cache_transition": cache_transition,
                            "timings_s": timings,
                        }
                        _applied_lora_updates(models)[lora_slot] = _AppliedLoraUpdate(
                            identity=identity,
                            response=response,
                        )
                        _runtime_state.update(
                            loaded_adapter=public_model_name,
                            policy_version=policy_version,
                            update_identity=(
                                f"lora:{lora_slot}:{policy_version}:{update_seq}"
                            ),
                        )
                except BaseException:
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
                    else:
                        await asyncio.shield(
                            coordinator.cancel_update(lora_slot, update_seq)
                        )
                    raise
                return JSONResponse(content=response)

        @router.post("/art/unload_lora_policy")
        async def unload_lora_policy(
            body: _UnloadLoraPolicyRequest, raw_request: Request
        ) -> JSONResponse:
            from art_vllm_runtime.policy_spans import (
                lora_update_coordinator,
                policy_lora_request_payload,
                unregister_lora_slot,
            )

            models = raw_request.app.state.openai_serving_models
            engine_client = engine(raw_request)
            coordinator = lora_update_coordinator(models, engine_client)
            async with _lora_mutation_lock(models, body.lora_slot):
                update_seq = await coordinator.begin_update(body.lora_slot)
                mutation_started = False
                try:
                    async with models.lora_resolver_lock[body.lora_slot]:
                        lora_request = models.lora_requests.get(body.lora_slot)
                        if lora_request is None:
                            await coordinator.cancel_update(body.lora_slot, update_seq)
                            return JSONResponse(
                                content={
                                    "error": f"LoRA slot {body.lora_slot!r} is not loaded"
                                },
                                status_code=HTTPStatus.NOT_FOUND.value,
                            )
                        active_requests = (
                            await engine_client.engine_core.call_utility_async(
                                "art_count_lora_policy_requests", body.lora_slot
                            )
                        )
                        if active_requests:
                            await coordinator.cancel_update(body.lora_slot, update_seq)
                            return JSONResponse(
                                content={
                                    "error": (
                                        f"LoRA slot {body.lora_slot!r} has "
                                        f"{active_requests} active requests"
                                    )
                                },
                                status_code=HTTPStatus.CONFLICT.value,
                            )
                        await engine_client.engine_core.call_utility_async(
                            "pause_scheduler", "keep", False
                        )
                        mutation_started = True
                        removed = await engine_client.engine_core.call_utility_async(
                            "art_remove_lora_policy",
                            policy_lora_request_payload(lora_request),
                        )
                        removed_aliases = unregister_lora_slot(models, body.lora_slot)
                        _applied_lora_updates(models).pop(body.lora_slot, None)
                        await engine_client.engine_core.call_utility_async(
                            "resume_scheduler"
                        )
                        await coordinator.commit_removal(body.lora_slot, update_seq)
                        mutation_started = False
                    if _runtime_state.get("loaded_adapter") in removed_aliases:
                        _runtime_state.update(
                            loaded_adapter=None,
                            policy_version=None,
                            update_identity=None,
                        )
                    return JSONResponse(
                        content={
                            "status": "unloaded",
                            "lora_slot": body.lora_slot,
                            "aliases": removed_aliases,
                            **removed,
                        }
                    )
                except BaseException:
                    if mutation_started:
                        try:
                            await asyncio.shield(
                                engine_client.engine_core.call_utility_async(
                                    "pause_scheduler", "abort", True
                                )
                            )
                        finally:
                            await asyncio.shield(
                                coordinator.fail_update(body.lora_slot, update_seq)
                            )
                    else:
                        await asyncio.shield(
                            coordinator.cancel_update(body.lora_slot, update_seq)
                        )
                    raise

        for path in ("/v1/chat/completions", "/v1/completions"):
            matches = [
                route
                for route in app.router.routes
                if getattr(route, "path", None) == path
                and "POST" in (getattr(route, "methods", None) or ())
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"ART expected one vLLM generation route for {path}, "
                    f"found {len(matches)}"
                )
            app.router.routes.remove(matches[0])
        app.include_router(router)
        return app

    async def art_init_app_state(
        engine_client: Any, state: Any, *args: Any, **kwargs: Any
    ) -> None:
        await original_init_app_state(engine_client, state, *args, **kwargs)
        policy_version = _runtime_state.get("initial_policy_version")
        if policy_version is None or _runtime_state.get("loaded_adapter") is None:
            return
        from art_vllm_runtime.policy_spans import declare_initial_lora_policy

        await declare_initial_lora_policy(
            state.openai_serving_models,
            engine_client,
            lora_slot=str(_runtime_state["loaded_adapter"]),
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

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        config = create_engine_config(self, *args, **kwargs)
        from art_vllm_runtime.model_capabilities import model_backend_capabilities

        if pipeline_route_capture:
            _validate_pipeline_route_config(config)
            config.model_config.enable_return_routed_experts = True
            _register_model_route_layout(config.model_config)
        _runtime_state["model_backend"] = model_backend_capabilities(
            config.model_config,
            binary_route_capture=bool(
                _runtime_state.get("binary_routed_experts", False)
            ),
        )
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
        policy_version=args.initial_policy_version
        if args.initial_policy_version is not None
        else (
            int(args.served_model_name.rsplit("@", 1)[1])
            if "@" in args.served_model_name
            and args.served_model_name.rsplit("@", 1)[1].isdigit()
            else None
        ),
        update_identity=args.update_identity,
        generation_id=args.initial_generation_id,
        initial_policy_version=args.initial_policy_version,
        pp_layer_partition=pp_layer_partition,
        binary_routed_experts=route_capture,
        in_flight_lora_updates=True,
        policy_token_spans=True,
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
    namespace.middleware = [
        *namespace.middleware,
        "art_vllm_runtime.dedicated_server._ArtRequestTimingMiddleware",
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
