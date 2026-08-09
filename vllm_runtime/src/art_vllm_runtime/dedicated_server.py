"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
from functools import lru_cache
from http import HTTPStatus
import json
import os
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
from art_vllm_runtime.patches import apply_vllm_runtime_patches

ART_SERVING_PROTOCOL_VERSION = 3
_runtime_state: dict[str, object] = {}
_auth_tokens: list[str] = []


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


class _SetServedModelNameRequest(BaseModel):
    name: str = Field(min_length=1)


class _ResetPrefixCacheRequest(BaseModel):
    reset_running_requests: bool = False
    reset_connector: bool = True


class _InFlightLoraUpdateRequest(BaseModel):
    model_name: str = Field(min_length=1)
    lora_path: str = Field(min_length=1)
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
    parser.add_argument("--update-identity")
    parser.add_argument("--initial-policy-version", type=int)
    parser.add_argument("--lora-path", help="Optional initial checkpoint path")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--rollout-weights-mode",
        choices=("lora", "merged"),
        default="lora",
        help="Whether the dedicated server serves LoRA adapters or merged weights",
    )
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

        @router.post("/art/set_served_model_name")
        async def set_served_model_name(
            body: _SetServedModelNameRequest, raw_request: Request
        ) -> JSONResponse:
            models = raw_request.app.state.openai_serving_models
            if not models.base_model_paths:
                raise RuntimeError("vLLM runtime has no registered base model")
            models.base_model_paths[0].name = body.name
            _runtime_state["loaded_adapter"] = body.name
            if "@" in body.name and body.name.rsplit("@", 1)[1].isdigit():
                _runtime_state["policy_version"] = int(body.name.rsplit("@", 1)[1])
            return JSONResponse(content={"name": body.name})

        @router.get("/art/state")
        async def art_state() -> JSONResponse:
            return JSONResponse(content=dict(_runtime_state))

        @router.get("/art/metrics")
        async def art_metrics() -> JSONResponse:
            from art_vllm_runtime.metrics import get_art_metrics_snapshot

            snapshot = get_art_metrics_snapshot()
            snapshot.update(
                process_uuid=_runtime_state["process_uuid"],
                generation=_runtime_state["generation"],
            )
            return JSONResponse(content=snapshot)

        @router.get("/art/capabilities")
        async def art_capabilities() -> JSONResponse:
            return JSONResponse(
                content={
                    "runtime": "art_vllm",
                    "protocol_version": ART_SERVING_PROTOCOL_VERSION,
                    "binary_routed_experts": True,
                    "fast_metrics": True,
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
            update_seq = await coordinator.begin_update(lora_slot)
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
                    policy_version=policy_version,
                    update_identity=(f"lora:{lora_slot}:{policy_version}:{update_seq}"),
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
    args = parse_args(argv)
    if args.rollout_weights_mode == "merged" and not args.lora_path:
        raise SystemExit("--lora-path is required for merged rollout weights")
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
        initial_policy_version=args.initial_policy_version,
        pp_layer_partition=pp_layer_partition,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if args.rollout_weights_mode == "merged":
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    apply_vllm_runtime_patches()

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

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
    if args.rollout_weights_mode == "lora":
        vllm_args.append("--enable-lora")
        if args.lora_path:
            vllm_args.append(
                f"--lora-modules={args.served_model_name}={args.lora_path}"
            )
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
        asyncio.run(api_server.run_server(namespace))


if __name__ == "__main__":
    main()
