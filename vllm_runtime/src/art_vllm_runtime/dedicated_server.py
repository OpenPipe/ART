"""Dedicated vLLM subprocess entry point for the ART-owned runtime package."""

import argparse
import asyncio
from http import HTTPStatus
import json
import os

from art_vllm_runtime.patches import apply_vllm_runtime_patches


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
    from fastapi import APIRouter, FastAPI, Query, Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai import api_server

    if getattr(api_server, "_art_runtime_routes_patched", False):
        return

    original_build_app = api_server.build_app

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
        async def set_served_model_name(raw_request: Request) -> JSONResponse:
            body = await raw_request.json()
            name = body["name"]
            assert isinstance(name, str) and name
            models = raw_request.app.state.openai_serving_models
            assert models.base_model_paths
            models.base_model_paths[0].name = name
            return JSONResponse(content={"name": name})

        @router.get("/art/metrics")
        async def art_metrics() -> JSONResponse:
            from art_vllm_runtime.metrics import get_art_metrics_snapshot

            return JSONResponse(content=get_art_metrics_snapshot())

        @router.post("/art/reset_prefix_cache")
        async def reset_prefix_cache(raw_request: Request) -> JSONResponse:
            body = await raw_request.json()
            success = await engine(raw_request).reset_prefix_cache(
                reset_running_requests=bool(body.get("reset_running_requests", False)),
                reset_connector=bool(body.get("reset_connector", True)),
            )
            return JSONResponse(content={"success": success})

        @router.post("/art/in_flight_lora_update")
        async def in_flight_lora_update(raw_request: Request) -> JSONResponse:
            from vllm.entrypoints.openai.engine.protocol import ErrorResponse
            from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest

            from art_vllm_runtime.policy_spans import register_lora_alias

            body = await raw_request.json()
            public_model_name = body["model_name"]
            lora_path = body["lora_path"]
            policy_version = body.get("policy_version")
            lora_slot = body.get("lora_slot") or public_model_name.rsplit("@", 1)[0]
            assert isinstance(public_model_name, str) and public_model_name
            assert isinstance(lora_path, str) and lora_path
            assert isinstance(lora_slot, str) and lora_slot
            if policy_version is not None:
                policy_version = int(policy_version)

            models = raw_request.app.state.openai_serving_models
            load_result = await models.load_lora_adapter(
                LoadLoRAAdapterRequest(
                    lora_name=lora_slot,
                    lora_path=lora_path,
                    load_inplace=lora_slot in models.lora_requests,
                    is_3d_lora_weight=bool(body.get("is_3d_lora_weight", False)),
                ),
                base_model_name=body.get("base_model_name"),
            )
            if isinstance(load_result, ErrorResponse):
                return JSONResponse(
                    content=load_result.model_dump(mode="python"),
                    status_code=load_result.error.code,
                )
            register_lora_alias(
                models,
                public_model_name=public_model_name,
                lora_slot=lora_slot,
                policy_version=policy_version,
            )
            waiting_cache_salt = None
            if policy_version is not None:
                engine_client = engine(raw_request)
                waiting_cache_salt = await engine_client.engine_core.call_utility_async(
                    "art_update_waiting_lora_cache_salt",
                    lora_slot,
                    policy_version,
                )
                from art_vllm_runtime.metrics import record_policy_cache_waiting_update

                record_policy_cache_waiting_update(
                    updated=int(waiting_cache_salt["updated_waiting_requests"]),
                    skipped_started=int(
                        waiting_cache_salt["skipped_started_waiting_requests"]
                    ),
                )
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
