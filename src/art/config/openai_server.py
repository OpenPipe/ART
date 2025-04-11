from typing import Any, Literal, Tuple, TypedDict

from .. import types


def get_openai_server_config(
    model_name: str,
    base_model: types.BaseModel,
    log_file: str,
    lora_path: str,
    config: "OpenAIServerConfig | None" = None,
) -> "OpenAIServerConfig":
    if config is None:
        config = OpenAIServerConfig()
    log_file = config.get("log_file", log_file)
    server_args = ServerArgs(
        api_key="default",
        lora_modules=[f'{{"name": "{model_name}", "path": "{lora_path}"}}'],
        return_tokens_as_token_ids=True,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    server_args.update(config.get("server_args", {}))
    engine_args = EngineArgs(
        model=base_model,
        num_scheduler_steps=16,
        served_model_name=base_model,
        disable_log_requests=True,
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


class EngineArgs(TypedDict, total=False):
    model: str
    served_model_name: str | list[str] | None
    tokenizer: str | None
    # task: TaskOption
    skip_tokenizer_init: bool
    tokenizer_mode: str
    trust_remote_code: bool
    allowed_local_media_path: str
    download_dir: str | None
    load_format: str
    # config_format: ConfigFormat
    dtype: str
    kv_cache_dtype: str
    seed: int
    max_model_len: int | None
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # distributed_executor_backend: str | type[ExecutorBase] | None
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int
    tensor_parallel_size: int
    max_parallel_loading_workers: int | None
    block_size: int | None
    enable_prefix_caching: bool | None
    disable_sliding_window: bool
    use_v2_block_manager: bool
    swap_space: float  # GiB
    cpu_offload_gb: float  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int | None
    max_num_partial_prefills: int | None
    max_long_partial_prefills: int | None
    long_prefill_token_threshold: int | None
    max_num_seqs: int | None
    max_logprobs: int  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool
    revision: str | None
    code_revision: str | None
    rope_scaling: dict[str, Any] | None
    rope_theta: float | None
    # hf_overrides: HfOverrides | None
    tokenizer_revision: str | None
    quantization: str | None
    enforce_eager: bool | None
    max_seq_len_to_capture: int
    disable_custom_all_reduce: bool
    tokenizer_pool_size: int
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # tokenizer_pool_type: str | type["BaseTokenizerGroup"]
    tokenizer_pool_extra_config: dict[str, Any] | None
    limit_mm_per_prompt: dict[str, int] | None
    mm_processor_kwargs: dict[str, Any] | None
    disable_mm_preprocessor_cache: bool
    enable_lora: bool
    enable_lora_bias: bool
    max_loras: int
    max_lora_rank: int
    enable_prompt_adapter: bool
    max_prompt_adapters: int
    max_prompt_adapter_token: int
    fully_sharded_loras: bool
    lora_extra_vocab_size: int
    long_lora_scaling_factors: Tuple[float] | None
    lora_dtype: str | None
    max_cpu_loras: int | None
    device: str
    num_scheduler_steps: int
    multi_step_stream_outputs: bool
    ray_workers_use_nsight: bool
    num_gpu_blocks_override: int | None
    num_lookahead_slots: int
    model_loader_extra_config: dict | None
    ignore_patterns: str | list[str] | None
    preemption_mode: str | None

    scheduler_delay_factor: float
    enable_chunked_prefill: bool | None

    guided_decoding_backend: str
    logits_processor_pattern: str | None
    # Speculative decoding configuration.
    speculative_model: str | None
    speculative_model_quantization: str | None
    speculative_draft_tensor_parallel_size: int | None
    num_speculative_tokens: int | None
    speculative_disable_mqa_scorer: bool | None
    speculative_max_model_len: int | None
    speculative_disable_by_batch_size: int | None
    ngram_prompt_lookup_max: int | None
    ngram_prompt_lookup_min: int | None
    spec_decoding_acceptance_method: str
    typical_acceptance_sampler_posterior_threshold: float | None
    typical_acceptance_sampler_posterior_alpha: float | None
    qlora_adapter_name_or_path: str | None
    disable_logprobs_during_spec_decoding: bool | None

    otlp_traces_endpoint: str | None
    collect_detailed_traces: str | None
    disable_async_output_proc: bool
    scheduling_policy: Literal["fcfs", "priority"]
    scheduler_cls: str | type[object]

    override_neuron_config: dict[str, Any] | None
    # override_pooler_config: PoolerConfig | None
    # compilation_config: CompilationConfig | None
    worker_cls: str

    # kv_transfer_config: KVTransferConfig | None

    generation_config: str | None
    override_generation_config: dict[str, Any] | None
    enable_sleep_mode: bool
    model_impl: str

    calculate_kv_scales: bool | None

    additional_config: dict[str, Any] | None

    disable_log_requests: bool
