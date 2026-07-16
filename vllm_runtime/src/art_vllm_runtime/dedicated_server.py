"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
from http import HTTPStatus
import json
import os

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

from art_vllm_runtime.patches import apply_vllm_runtime_patches


class _ArtAuthenticationMiddleware(AuthenticationMiddleware):
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
    policy_version: int
    lora_slot: str | None = Field(default=None, min_length=1)
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ART dedicated vLLM server")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cuda-visible-devices", required=True)
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
    from fastapi.responses import Response
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

    def art_build_app(*build_args: object, **build_kwargs: object) -> FastAPI:
        app = original_build_app(*build_args, **build_kwargs)
        from vllm import envs

        args = app.state.args
        tokens = [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]
        if tokens:
            app.add_middleware(_ArtAuthenticationMiddleware, tokens=tokens)
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
            return JSONResponse(content={"name": body.name})

        @router.get("/art/metrics")
        async def art_metrics() -> JSONResponse:
            from art_vllm_runtime.metrics import get_art_metrics_snapshot

            return JSONResponse(content=get_art_metrics_snapshot())

        @router.get("/art/capabilities")
        async def art_capabilities() -> JSONResponse:
            return JSONResponse(
                content={
                    "runtime": "art_vllm",
                    "protocol_version": 1,
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
                media_type="application/vnd.art.routed-experts-v1",
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
                lora_update_coordinator,
            )

            public_model_name = body.model_name
            lora_path = body.lora_path
            policy_version = body.policy_version
            lora_slot = body.lora_slot or public_model_name.rsplit("@", 1)[0]
            models = raw_request.app.state.openai_serving_models
            engine_client = engine(raw_request)
            coordinator = lora_update_coordinator(models, engine_client)
            await coordinator.begin_update(lora_slot)
            try:
                load_result = await models.load_lora_adapter(
                    LoadLoRAAdapterRequest(
                        lora_name=lora_slot,
                        lora_path=lora_path,
                        load_inplace=lora_slot in models.lora_requests,
                        is_3d_lora_weight=body.is_3d_lora_weight,
                    ),
                    base_model_name=body.base_model_name,
                )
                if isinstance(load_result, ErrorResponse):
                    await coordinator.fail_update(lora_slot)
                    return JSONResponse(
                        content=load_result.model_dump(mode="python"),
                        status_code=load_result.error.code,
                    )
                waiting_cache_salt = await engine_client.engine_core.call_utility_async(
                    "art_update_waiting_lora_cache_salt",
                    lora_slot,
                    policy_version,
                )
                await coordinator.commit_update(
                    lora_slot,
                    policy_version,
                    models.lora_requests[lora_slot],
                )
                from art_vllm_runtime.metrics import record_policy_cache_waiting_update

                record_policy_cache_waiting_update(
                    updated=int(waiting_cache_salt["updated_waiting_requests"]),
                    skipped_started=int(
                        waiting_cache_salt["skipped_started_waiting_requests"]
                    ),
                )
            except BaseException:
                await coordinator.fail_update(lora_slot)
                raise
            return JSONResponse(
                content={
                    "status": "updated",
                    "model_name": public_model_name,
                    "lora_slot": lora_slot,
                    "policy_version": policy_version,
                    "waiting_cache_salt": waiting_cache_salt,
                }
            )

        app.include_router(router)
        return app

    setattr(api_server, "build_app", art_build_app)
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rollout_weights_mode == "merged" and not args.lora_path:
        raise SystemExit("--lora-path is required for merged rollout weights")
    engine_args = json.loads(args.engine_args_json)
    server_args = json.loads(args.server_args_json)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if args.rollout_weights_mode == "merged":
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    apply_vllm_runtime_patches()

    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    _patch_art_runtime_routes()

    vllm_args = [
        f"--model={args.model}",
        f"--port={args.port}",
        f"--host={args.host}",
        f"--served-model-name={args.served_model_name}",
    ]
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
    validate_parsed_serve_args(namespace)
    asyncio.run(api_server.run_server(namespace))


if __name__ == "__main__":
    main()
