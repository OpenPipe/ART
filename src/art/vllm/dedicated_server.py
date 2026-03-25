"""Dedicated vLLM subprocess entry point.

Launched by UnslothService in dedicated mode as:
    python -m art.vllm.dedicated_server --model <base_model> --port <port> ...

Sets CUDA_VISIBLE_DEVICES and applies ART patches before starting vLLM.
Must be imported/run as a standalone process — not imported into the main training process.
"""

import argparse
import asyncio
import json
import os


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ART dedicated vLLM server")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--lora-path", required=True, help="Initial checkpoint path")
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


def _patch_art_dedicated_routes() -> None:
    from fastapi import APIRouter, FastAPI, Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai import api_server
    from vllm.tasks import SupportedTask

    if getattr(api_server, "_art_dedicated_routes_patched", False):
        return

    original_build_app = api_server.build_app

    def art_build_app(
        args: argparse.Namespace,
        supported_tasks: tuple[SupportedTask, ...] | None = None,
    ) -> FastAPI:
        app = original_build_app(args, supported_tasks)
        router = APIRouter()

        @router.post("/art/set_served_model_name")
        async def set_served_model_name(raw_request: Request) -> JSONResponse:
            body = await raw_request.json()
            name = body["name"]
            assert isinstance(name, str) and name
            models = raw_request.app.state.openai_serving_models
            assert models.base_model_paths
            models.base_model_paths[0].name = name
            return JSONResponse(content={"name": name})

        app.include_router(router)
        return app

    setattr(api_server, "build_app", art_build_app)
    setattr(api_server, "_art_dedicated_routes_patched", True)


def _append_cli_arg(vllm_args: list[str], key: str, value: object) -> None:
    cli_key = f"--{key.replace('_', '-')}"
    match value:
        case True:
            vllm_args.append(cli_key)
        case False | None:
            return
        case str() | int() | float():
            vllm_args.append(f"{cli_key}={value}")
        case dict():
            vllm_args.append(f"{cli_key}={json.dumps(value)}")
        case list():
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

    # Must set CUDA_VISIBLE_DEVICES before any torch/CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if args.rollout_weights_mode == "merged":
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"

    # Patches must be applied before vLLM's api_server is imported
    from .patches import (
        patch_listen_for_disconnect,
        patch_tool_parser_manager,
        subclass_chat_completion_request,
    )

    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()

    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    engine_args = json.loads(args.engine_args_json)
    server_args = json.loads(args.server_args_json)

    if args.rollout_weights_mode == "merged":
        _patch_art_dedicated_routes()

    vllm_args = [
        f"--model={args.model}",
        f"--port={args.port}",
        f"--host={args.host}",
        f"--served-model-name={args.served_model_name}",
    ]
    if args.rollout_weights_mode == "lora":
        vllm_args.extend(
            [
                "--enable-lora",
                f"--lora-modules={args.served_model_name}={args.lora_path}",
            ]
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

    # stdout/stderr are captured to a log file by the parent process,
    # so no separate uvicorn file handler is needed here.
    asyncio.run(api_server.run_server(namespace))


if __name__ == "__main__":
    main()
