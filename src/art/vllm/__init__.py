"""vLLM integration module for art."""

# Server functionality
# Engine and worker management
from .engine import (
    WorkerExtension,
    create_engine_pause_and_resume_functions,
    get_llm,
    get_worker,
    run_on_workers,
)

# Patches - these are typically imported for their side effects
from .patches import (
    patch_allocator,
    patch_get_lora_tokenizer_async,
    patch_listen_for_disconnect,
    patch_lora_request,
    patch_multi_step_model_runner,
    patch_tool_parser_manager,
    subclass_chat_completion_request,
)
from .server import (
    get_uvicorn_logging_config,
    openai_server_task,
    set_vllm_log_file,
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
