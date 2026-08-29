"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import hmac
from http import HTTPStatus
from ipaddress import ip_address
import json
import os
import re
import socket
from typing import Any
import uuid

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
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

ART_SERVING_PROTOCOL_VERSION = 7
_PRIVATE_CACHE_IDENTITY_HEADER = "x-art-cache-identity"
_PRIVATE_DISPATCH_PATH = "/art/internal/v1/chat/completions"
_PRIVATE_EXECUTION_RECEIPT_CAPACITY = 4096
_PRIVATE_EXECUTION_RECEIPT_PREFIX = "/art/internal/v1/requests"
_PRIVATE_REQUEST_IDENTITY_HEADER = "x-art-request-identity"
_RUNTIME_TARGET_HEADER = "x-art-runtime-target"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_runtime_state: dict[str, object] = {}
_auth_tokens: list[str] = []
_fast_metrics_port: int | None = None
_private_dispatch_token: str | None = None
_runtime_target_id: str | None = None


@dataclass(frozen=True)
class _PrivateExecutionReceipt:
    fingerprint: str
    execution: str


class _PrivateExecutionReceipts:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("private execution receipt capacity must be positive")
        self.capacity = capacity
        self._lock = asyncio.Lock()
        self._receipts: OrderedDict[str, _PrivateExecutionReceipt] = OrderedDict()

    async def claim(
        self, request_identity: str, fingerprint: str
    ) -> _PrivateExecutionReceipt | None:
        async with self._lock:
            existing = self._receipts.get(request_identity)
            if existing is not None:
                return existing
            if len(self._receipts) >= self.capacity:
                terminal = next(
                    (
                        identity
                        for identity, receipt in self._receipts.items()
                        if receipt.execution != "started"
                    ),
                    None,
                )
                if terminal is None:
                    raise RuntimeError("private execution receipt capacity exhausted")
                self._receipts.pop(terminal)
            self._receipts[request_identity] = _PrivateExecutionReceipt(
                fingerprint=fingerprint,
                execution="started",
            )
            return None

    async def settle(
        self, request_identity: str, fingerprint: str, execution: str
    ) -> None:
        async with self._lock:
            current = self._receipts.get(request_identity)
            if current is None or current.fingerprint != fingerprint:
                raise RuntimeError("private execution receipt identity changed")
            self._receipts[request_identity] = _PrivateExecutionReceipt(
                fingerprint=fingerprint,
                execution=execution,
            )
            self._receipts.move_to_end(request_identity)

    async def release_not_started(
        self, request_identity: str, fingerprint: str
    ) -> None:
        async with self._lock:
            current = self._receipts.get(request_identity)
            if current is not None and current.fingerprint == fingerprint:
                self._receipts.pop(request_identity)

    async def get(self, request_identity: str) -> _PrivateExecutionReceipt | None:
        async with self._lock:
            return self._receipts.get(request_identity)


_private_execution_receipts = _PrivateExecutionReceipts(
    _PRIVATE_EXECUTION_RECEIPT_CAPACITY
)


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


def _config_string(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).removeprefix("torch.")


async def _serving_profile(engine_client: Any) -> dict[str, Any] | None:
    identity = _runtime_state.get("serving_profile_identity")
    if identity is None:
        return None
    if not isinstance(identity, dict):
        raise RuntimeError("ART serving profile identity is malformed")
    config = engine_client.vllm_config
    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    lora = config.lora_config
    if lora is None:
        raise RuntimeError("ART serving profile requires LoRA configuration")
    speculative = config.speculative_config
    speculative_method = (
        None
        if speculative is None or speculative.method is None
        else _config_string(speculative.method)
    )
    from art_vllm_runtime.engine_core import query_engine_cores

    geometry_reports = await query_engine_cores(engine_client, "art_runtime_geometry")
    if not geometry_reports or any(
        report != geometry_reports[0] for report in geometry_reports[1:]
    ):
        raise RuntimeError("vLLM engine cores returned inconsistent KV geometry")
    geometry = geometry_reports[0]
    if not isinstance(geometry, dict):
        raise RuntimeError("vLLM returned malformed KV geometry")
    return {
        "schema_version": 2,
        "identity": identity,
        "runtime_model": model.model,
        "runtime_revision": model.revision,
        "tokenizer": model.tokenizer,
        "tokenizer_revision": model.tokenizer_revision,
        "model_dtype": _config_string(model.dtype),
        "quantization": (
            None if model.quantization is None else _config_string(model.quantization)
        ),
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "data_parallel_size": parallel.data_parallel_size,
        "prefill_context_parallel_size": parallel.prefill_context_parallel_size,
        "enable_expert_parallel": parallel.enable_expert_parallel,
        "max_model_len": model.max_model_len,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_partial_prefills": scheduler.max_num_partial_prefills,
        "kv_cache_dtype": _config_string(cache.cache_dtype),
        **geometry,
        "prefix_caching": cache.enable_prefix_caching,
        "prefix_hash_algorithm": _config_string(cache.prefix_caching_hash_algo),
        "max_loras": lora.max_loras,
        "max_lora_rank": lora.max_lora_rank,
        "lora_dtype": _config_string(lora.lora_dtype),
        "speculative_method": speculative_method,
        "multi_token_prediction": speculative_method == "mtp",
        "exact_token_ids": True,
        "selected_token_logprobs": True,
        "policy_span_schema": "prompt_completion_v1",
        "cache_transition": "policy_history_route_salt_v1",
        "lora_update_semantics": "holder_local_in_flight_v1",
        "route_capture_format": (
            "art_inference_route_bundle_v1"
            if _runtime_state.get("route_capture") is True
            else None
        ),
    }


class _ArtAuthenticationMiddleware(AuthenticationMiddleware):
    def __init__(self, app: Any) -> None:
        super().__init__(app, tokens=_auth_tokens)

    def __call__(self, scope: Scope, receive: Receive, send: Send):
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if (
            scope.get("type") not in {"http", "websocket"}
            or scope.get("method") == "OPTIONS"
        ):
            return self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if path == _PRIVATE_DISPATCH_PATH or path.startswith(
            f"{_PRIVATE_EXECUTION_RECEIPT_PREFIX}/"
        ):
            if _verify_bearer(headers, _private_dispatch_token):
                return self.app(scope, receive, send)
        elif not _auth_tokens:
            return self.app(scope, receive, send)
        elif path.startswith("/art/"):
            if self.verify_token(headers):
                return self.app(scope, receive, send)
        else:
            return super().__call__(scope, receive, send)
        if path.startswith("/art/"):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


def _verify_bearer(headers: Headers, token: str | None) -> bool:
    values = headers.getlist("authorization")
    if token is None or len(values) != 1:
        return False
    scheme, separator, provided = values[0].partition(" ")
    return (
        separator == " "
        and scheme.casefold() == "bearer"
        and hmac.compare_digest(provided, token)
    )


def _private_runtime_target_error(
    request: Any, *, execution: str = "not_started"
) -> JSONResponse | None:
    headers = request.headers
    target_values = headers.getlist(_RUNTIME_TARGET_HEADER)
    if (
        _runtime_target_id is None
        or len(target_values) != 1
        or not hmac.compare_digest(target_values[0], _runtime_target_id)
    ):
        return JSONResponse(
            content={
                "error": "Runtime target is no longer active",
                "type": "stale_runtime_target",
                "execution": execution,
            },
            status_code=HTTPStatus.CONFLICT.value,
        )
    return None


def _private_dispatch_context(request: Any) -> tuple[str, str] | JSONResponse:
    target_error = _private_runtime_target_error(request)
    if target_error is not None:
        return target_error
    headers = request.headers
    identities = []
    for header in (
        _PRIVATE_REQUEST_IDENTITY_HEADER,
        _PRIVATE_CACHE_IDENTITY_HEADER,
    ):
        values = headers.getlist(header)
        if len(values) != 1 or _SHA256_RE.fullmatch(values[0]) is None:
            return JSONResponse(
                content={
                    "error": f"Invalid {header}",
                    "type": "invalid_private_context",
                    "execution": "not_started",
                },
                status_code=HTTPStatus.BAD_REQUEST.value,
            )
        identities.append(values[0])
    return identities[0], identities[1]


def _private_request_fingerprint(request: Any) -> str:
    return hashlib.sha256(
        request.model_dump_json(exclude_none=False).encode()
    ).hexdigest()


def _private_duplicate_response(
    receipt: _PrivateExecutionReceipt,
    *,
    fingerprint: str,
) -> JSONResponse:
    if receipt.fingerprint != fingerprint:
        return JSONResponse(
            content={
                "error": "Request identity was reused for different content",
                "type": "request_identity_conflict",
                "execution": "not_started",
                "prior_execution": receipt.execution,
            },
            status_code=HTTPStatus.CONFLICT.value,
        )
    return JSONResponse(
        content={
            "error": "Request identity already reached the paired runtime",
            "type": "duplicate_request_identity",
            "execution": receipt.execution,
        },
        status_code=HTTPStatus.CONFLICT.value,
    )


async def _track_private_stream(
    request_identity: str,
    fingerprint: str,
    body_iterator: Any,
):
    try:
        async for chunk in body_iterator:
            yield chunk
    except BaseException:
        await _private_execution_receipts.settle(
            request_identity, fingerprint, "ambiguous"
        )
        raise
    else:
        await _private_execution_receipts.settle(
            request_identity, fingerprint, "completed"
        )


class _ResetPrefixCacheRequest(BaseModel):
    reset_running_requests: bool = False
    reset_connector: bool = True


class _InFlightLoraUpdateRequest(BaseModel):
    model_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    expected_generation_id: str = Field(min_length=1)
    policy_version: int = Field(ge=0)
    lora_slot: str | None = Field(default=None, min_length=1)
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False


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
    parser.add_argument("--runtime-target-id")
    parser.add_argument("--update-identity")
    parser.add_argument("--initial-generation-id")
    parser.add_argument("--initial-policy-version", type=int)
    parser.add_argument("--serving-profile-identity-json")
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
        mark_route_request,
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
            profile = await _serving_profile(engine(raw_request))
            return JSONResponse(
                content={
                    "runtime": "art_vllm",
                    "protocol_version": ART_SERVING_PROTOCOL_VERSION,
                    "binary_routed_experts": _runtime_state.get("route_capture")
                    is True,
                    "fast_metrics": {"url": _fast_metrics_url(raw_request)},
                    "inplace_lora_load": True,
                    "in_flight_lora_updates": True,
                    "policy_token_spans": True,
                    "profile": profile,
                }
            )

        @router.post(
            _PRIVATE_DISPATCH_PATH,
            dependencies=[Depends(validate_json_request)],
        )
        async def private_chat_completion(
            request: ChatCompletionRequest, raw_request: Request
        ) -> Response:
            context = _private_dispatch_context(raw_request)
            if isinstance(context, JSONResponse):
                return context
            request_identity, cache_identity = context
            request.cache_salt = f"art-private-cache-v1:{cache_identity}"
            fingerprint = _private_request_fingerprint(request)
            try:
                existing = await _private_execution_receipts.claim(
                    request_identity, fingerprint
                )
            except RuntimeError:
                return JSONResponse(
                    content={
                        "error": "Private execution receipt capacity is exhausted",
                        "type": "receipt_capacity_exhausted",
                        "execution": "not_started",
                    },
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                )
            if existing is not None:
                return _private_duplicate_response(
                    existing,
                    fingerprint=fingerprint,
                )
            try:
                response = await create_chat_completion(request, raw_request)
            except BaseException:
                await _private_execution_receipts.settle(
                    request_identity, fingerprint, "ambiguous"
                )
                raise
            if response is None:
                await _private_execution_receipts.settle(
                    request_identity, fingerprint, "ambiguous"
                )
                return Response(status_code=499)
            if response.status_code >= HTTPStatus.BAD_REQUEST.value:
                await _private_execution_receipts.release_not_started(
                    request_identity, fingerprint
                )
                return response
            body_iterator = getattr(response, "body_iterator", None)
            if body_iterator is not None:
                response.body_iterator = _track_private_stream(
                    request_identity,
                    fingerprint,
                    body_iterator,
                )
            else:
                await _private_execution_receipts.settle(
                    request_identity, fingerprint, "completed"
                )
            return response

        @router.get(f"{_PRIVATE_EXECUTION_RECEIPT_PREFIX}/{{request_identity}}")
        async def private_execution_receipt(
            request_identity: str, raw_request: Request
        ) -> JSONResponse:
            target_error = _private_runtime_target_error(
                raw_request, execution="unknown"
            )
            if target_error is not None:
                return target_error
            if _SHA256_RE.fullmatch(request_identity) is None:
                return JSONResponse(
                    content={
                        "error": "Invalid private request identity",
                        "type": "invalid_private_context",
                        "execution": "not_started",
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            receipt = await _private_execution_receipts.get(request_identity)
            if receipt is None:
                return JSONResponse(
                    content={
                        "type": "execution_receipt_missing",
                        "execution": "unknown",
                    },
                    status_code=HTTPStatus.NOT_FOUND.value,
                )
            return JSONResponse(
                content={
                    "type": "execution_receipt",
                    "execution": receipt.execution,
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
            mark_route_request(request)
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
            if generation_id == body.expected_generation_id:
                return JSONResponse(
                    content={"error": "generation_id must advance"}, status_code=409
                )
            try:
                update_seq = await coordinator.begin_update(
                    lora_slot, expected_generation_id=body.expected_generation_id
                )
            except RuntimeError as error:
                return JSONResponse(
                    content={"error": str(error), "type": "generation_conflict"},
                    status_code=409,
                )
            mutation_started = False
            try:
                async with models.lora_resolver_lock[lora_slot]:
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
                        generation_id=generation_id,
                        policy_version=policy_version,
                        update_seq=update_seq,
                    )
                    mutation_started = True
                    await engine_client.engine_core.call_utility_async(
                        "pause_scheduler", "keep", False
                    )
                    cache_transition = (
                        await engine_client.engine_core.call_utility_async(
                            "art_apply_lora_policy_update",
                            policy_lora_request_payload(lora_request),
                        )
                    )
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
                    await engine_client.engine_core.call_utility_async(
                        "resume_scheduler"
                    )
                    await coordinator.commit_update(lora_slot, serving_request)
                    mutation_started = False
                _runtime_state.update(
                    loaded_adapter=public_model_name,
                    generation_id=generation_id,
                    policy_version=policy_version,
                    update_identity=f"lora:{lora_slot}:{generation_id}:{update_seq}",
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
            return JSONResponse(
                content={
                    "status": "updated",
                    "model_name": public_model_name,
                    "lora_slot": lora_slot,
                    "generation_id": generation_id,
                    "policy_version": policy_version,
                    "update_seq": update_seq,
                    "cache_transition": cache_transition,
                }
            )

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
    global _fast_metrics_port, _private_dispatch_token, _runtime_target_id

    args = parse_args(argv)
    if (args.initial_generation_id is None) != (args.initial_policy_version is None):
        raise ValueError(
            "--initial-generation-id and --initial-policy-version must be set together"
        )
    private_dispatch_token = os.environ.pop("ART_PRIVATE_DISPATCH_TOKEN", None)
    if (args.runtime_target_id is None) != (private_dispatch_token is None):
        raise ValueError(
            "--runtime-target-id and ART_PRIVATE_DISPATCH_TOKEN must be set together"
        )
    if (
        args.runtime_target_id is not None
        and _SHA256_RE.fullmatch(args.runtime_target_id) is None
    ):
        raise ValueError("--runtime-target-id must be a lowercase SHA-256")
    if private_dispatch_token is not None and len(private_dispatch_token) < 32:
        raise ValueError(
            "ART_PRIVATE_DISPATCH_TOKEN must contain at least 32 characters"
        )
    _runtime_target_id = args.runtime_target_id
    _private_dispatch_token = private_dispatch_token
    engine_args = json.loads(args.engine_args_json)
    server_args = json.loads(args.server_args_json)
    serving_profile_identity = (
        None
        if args.serving_profile_identity_json is None
        else json.loads(args.serving_profile_identity_json)
    )
    if serving_profile_identity is not None and not isinstance(
        serving_profile_identity, dict
    ):
        raise ValueError("serving profile identity must be a JSON object")
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
        route_capture=route_capture,
        serving_profile_identity=serving_profile_identity,
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
    if _auth_tokens or _private_dispatch_token is not None:
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
