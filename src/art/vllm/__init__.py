"""vLLM integration module for art."""

# Server functionality
from .server import (
    openai_server_task,
    get_uvicorn_logging_config,
    set_vllm_log_file,
)

# Engine and worker management
from .engine import (
    get_llm,
    create_engine_pause_and_resume_functions,
    run_on_workers,
    get_worker,
    WorkerExtension,
)

# Patches - these are typically imported for their side effects
from .patches import (
    patch_allocator,
    subclass_chat_completion_request,
    patch_lora_request,
    patch_get_lora_tokenizer_async,
    patch_listen_for_disconnect,
    patch_tool_parser_manager,
    patch_multi_step_model_runner,
)

__all__ = [
    # Server
    "openai_server_task",
    "get_uvicorn_logging_config",
    "set_vllm_log_file",
    # Engine
    "get_llm",
    "create_engine_pause_and_resume_functions",
    "run_on_workers",
    "get_worker",
    "WorkerExtension",
    # Patches
    "patch_allocator",
    "subclass_chat_completion_request",
    "patch_lora_request",
    "patch_get_lora_tokenizer_async",
    "patch_listen_for_disconnect",
    "patch_tool_parser_manager",
    "patch_multi_step_model_runner",
]
