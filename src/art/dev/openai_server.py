from typing import Literal

from typing_extensions import TypedDict

from .engine import EngineArgs


def get_openai_server_config(
    model_name: str,
    base_model: str,
    log_file: str,
    lora_path: str | None = None,
    config: "OpenAIServerConfig | None" = None,
) -> "OpenAIServerConfig":
    import os

    if config is None:
        config = OpenAIServerConfig()
    log_file = config.get("log_file", log_file)

    # Build LoRA modules list for multi-checkpoint support
    # Register under both model_name (for "current" model) and model_name@step (for specific checkpoint)
    lora_modules: list[str] | None = None
    if lora_path:
        step = int(os.path.basename(lora_path))
        lora_modules = [
            f'{{"name": "{model_name}", "path": "{lora_path}"}}',
            f'{{"name": "{model_name}@{step}", "path": "{lora_path}"}}',
        ]

    server_args = ServerArgs(
        api_key="default",
        lora_modules=lora_modules,
        return_tokens_as_token_ids=True,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    server_args.update(config.get("server_args", {}))
    engine_args = EngineArgs(
        model=base_model,
        served_model_name=model_name,
        generation_config="vllm",
    )
    engine_args.update(config.get("engine_args", {}))
    return OpenAIServerConfig(
        log_file=log_file, server_args=server_args, engine_args=engine_args
    )


class OpenAIServerConfig(TypedDict, total=False):
    """
    Server configuration.

    Args:
        log_file: Path to the log file.
        server_args: Arguments for the vLLM OpenAI-compatible server.
        engine_args: Additional vLLM engine arguments for the OpenAI-compatible server.
                     Note that since the vLLM engine is initialized with Unsloth,
                     these additional arguments will only have an effect if the
                     OpenAI-compatible server uses them elsewhere.
    """

    log_file: str
    server_args: "ServerArgs"
    engine_args: "EngineArgs"


class ServerArgs(TypedDict, total=False):
    """Arguments for the vLLM OpenAI-compatible server, not including AsyncEngineArgs.

    Fields:
        host: Host name for the server. If None, will listen on all available interfaces (0.0.0.0).
        port: Port number for the server to listen on.
        uvicorn_log_level: Log level for the uvicorn server.
        allow_credentials: Whether to allow credentials in CORS requests.
        allowed_origins: JSON string that will be parsed into a list of allowed CORS origins. Defaults to ["*"].
        allowed_methods: JSON string that will be parsed into a list of allowed HTTP methods. Defaults to ["*"].
        allowed_headers: JSON string that will be parsed into a list of allowed HTTP headers. Defaults to ["*"].
        api_key: API key required for authentication. If None, no authentication is required.
        lora_modules: List of LoRA module configurations. Each string is either in 'name=path' format or a JSON object.
        prompt_adapters: List of prompt adapter configurations. Each string is in 'name=path' format.
        chat_template: Path to chat template file or the template in single-line form.
        chat_template_content_format: Format to render message content within chat templates.
        response_role: Role name to return when request.add_generation_prompt is true.
        ssl_keyfile: Path to SSL key file for HTTPS.
        ssl_certfile: Path to SSL certificate file for HTTPS.
        ssl_ca_certs: Path to CA certificates file for HTTPS.
        enable_ssl_refresh: Whether to refresh SSL context when certificate files change.
        ssl_cert_reqs: SSL certificate requirements (see stdlib ssl module's CERT_* constants).
        root_path: FastAPI root_path when app is behind a path-based routing proxy.
        middleware: List of additional ASGI middleware to apply to the app. Each string is an import path.
        return_tokens_as_token_ids: When max_logprobs is specified, represent tokens as 'token_id:{token_id}' strings.
        disable_frontend_multiprocessing: Run OpenAI frontend server in same process as model serving engine.
        enable_request_id_headers: Add X-Request-Id header to responses (may impact performance at high QPS).
        enable_auto_tool_choice: Enable automatic tool choice for supported models.
        tool_call_parser: Parser to use for model-generated tool calls into OpenAI API format.
        tool_parser_plugin: Plugin for registering custom tool call parsers.
        max_log_len: Maximum number of prompt characters or prompt ID numbers to print in logs.
        disable_fastapi_docs: Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint.
        enable_prompt_tokens_details: Include prompt_tokens_details in usage statistics.
        enable_server_load_tracking: Track server_load_metrics in the app state.
        enable_reasoning: Enable reasoning capabilities for supported models.
        reasoning_parser: Parser to use for model-generated reasoning into OpenAI API format.
    """

    host: str | None
    port: int
    uvicorn_log_level: Literal["debug", "info", "warning", "error", "critical", "trace"]
    allow_credentials: bool
    allowed_origins: str  # JSON string that will be parsed into list[str]
    allowed_methods: str  # JSON string that will be parsed into list[str]
    allowed_headers: str  # JSON string that will be parsed into list[str]
    api_key: str | None
    lora_modules: list[str] | None  # Each string is either 'name=path' or a JSON object
    prompt_adapters: list[str] | None  # Each string is in 'name=path' format
    chat_template: str | None
    chat_template_content_format: Literal["auto", "string", "openai"]
    response_role: str | None
    ssl_keyfile: str | None
    ssl_certfile: str | None
    ssl_ca_certs: str | None
    enable_ssl_refresh: bool
    ssl_cert_reqs: int
    root_path: str | None
    middleware: list[str]  # Each string is an import path
    return_tokens_as_token_ids: bool
    disable_frontend_multiprocessing: bool
    enable_request_id_headers: bool
    enable_auto_tool_choice: bool
    tool_call_parser: str | None
    tool_parser_plugin: str
    max_log_len: int | None
    disable_fastapi_docs: bool
    enable_prompt_tokens_details: bool
    enable_server_load_tracking: bool
    enable_reasoning: bool
    reasoning_parser: str | None
