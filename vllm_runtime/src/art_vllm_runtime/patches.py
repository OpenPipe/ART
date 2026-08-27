"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

from typing import Any


def apply_vllm_runtime_patches() -> None:
    from art_vllm_runtime.dsv4_patches import apply_dsv4_vllm_runtime_patches
    from art_vllm_runtime.gemma4_moe_lora_patch import (
        patch_gemma4_moe_lora_support,
    )
    from art_vllm_runtime.glm52_patches import apply_glm52_vllm_runtime_patches
    from art_vllm_runtime.moe_lora_patches import (
        patch_local_3d_moe_dummy_lora,
        patch_small_batch_moe_lora_intermediate_dtype,
    )
    from art_vllm_runtime.policy_spans import patch_policy_token_spans
    from art_vllm_runtime.qwen35_patches import apply_qwen35_vllm_runtime_patches

    patch_policy_token_spans()
    patch_gemma4_moe_lora_support()
    subclass_chat_completion_request()
    patch_nonstreaming_chat_response_offload()
    patch_local_3d_moe_dummy_lora()
    patch_small_batch_moe_lora_intermediate_dtype()
    apply_glm52_vllm_runtime_patches()
    apply_dsv4_vllm_runtime_patches()
    apply_qwen35_vllm_runtime_patches()
    from art_vllm_runtime.binary_routes import (
        patch_binary_routed_experts_response,
        patch_pipeline_routed_experts,
        patch_pipeline_routed_experts_validation,
    )

    patch_pipeline_routed_experts_validation()
    patch_pipeline_routed_experts()
    patch_binary_routed_experts_response()


def subclass_chat_completion_request() -> None:
    from vllm.entrypoints.openai.chat_completion import protocol
    from vllm.entrypoints.openai.completion import protocol as completion_protocol

    from art_vllm_runtime.route_uploads import S3PutGrant

    if not getattr(protocol, "_art_chat_completion_request_patched", False):

        class ChatCompletionRequest(protocol.ChatCompletionRequest):
            # Training callers opt into exact metadata. Ordinary OpenAI-compatible
            # requests retain vLLM's defaults and response shape.
            return_policy_spans: bool = False
            route_upload: S3PutGrant | None = None

        protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]
        setattr(protocol, "_art_chat_completion_request_patched", True)

    if not getattr(completion_protocol, "_art_completion_request_patched", False):

        class CompletionRequest(completion_protocol.CompletionRequest):
            return_policy_spans: bool = False
            route_upload: S3PutGrant | None = None

        completion_protocol.CompletionRequest = CompletionRequest  # ty:ignore[invalid-assignment]
        setattr(completion_protocol, "_art_completion_request_patched", True)


def patch_nonstreaming_chat_response_offload() -> None:
    import asyncio

    from starlette.responses import JSONResponse as StarletteJSONResponse
    from starlette.responses import Response
    from vllm.entrypoints.openai.chat_completion import api_router
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    marker = "_art_nonstreaming_response_offload_patched"
    if getattr(OpenAIServingChat, marker, False):
        return
    original = OpenAIServingChat.chat_completion_full_generator

    class PreencodedContent:
        def __init__(self, body: bytes) -> None:
            self.body = body

    original_model_dump = ChatCompletionResponse.model_dump

    def model_dump(self: Any, *args: Any, **kwargs: Any) -> Any:
        cached = getattr(self, "_art_preencoded_content", None)
        if cached is not None and not args and not kwargs:
            return cached
        return original_model_dump(self, *args, **kwargs)

    async def build_response(
        self: Any, request: Any, result_generator: Any, *args: Any, **kwargs: Any
    ) -> Any:
        final_result = None
        try:
            async for result in result_generator:
                final_result = result
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        async def materialize() -> Any:
            async def replay_final_result():
                if final_result is not None:
                    yield final_result

            result = await original(
                self, request, replay_final_result(), *args, **kwargs
            )
            if not isinstance(result, ChatCompletionResponse):
                return result
            content = original_model_dump(result)
            object.__setattr__(
                result,
                "_art_preencoded_content",
                PreencodedContent(StarletteJSONResponse(content).body),
            )
            return result

        return await asyncio.to_thread(
            asyncio.run,
            materialize(),
        )

    class PreencodedJSONResponse(StarletteJSONResponse):
        media_type = "application/json"

        def render(self, content: Any) -> bytes:
            if isinstance(content, bytes):
                return content
            return super().render(content)

        def __init__(
            self: Any,
            content: Any,
            status_code: int = 200,
            headers: Any = None,
            media_type: str | None = None,
            background: Any = None,
        ) -> None:
            if isinstance(content, PreencodedContent):
                Response.__init__(
                    self,
                    content.body,
                    status_code=status_code,
                    headers=headers,
                    media_type=media_type or self.media_type,
                    background=background,
                )
            else:
                super().__init__(
                    content,
                    status_code=status_code,
                    headers=headers,
                    media_type=media_type,
                    background=background,
                )

    setattr(build_response, "__art_offloaded__", True)
    setattr(build_response, "__art_original__", original)
    ChatCompletionResponse.model_dump = model_dump  # ty:ignore[invalid-assignment]
    OpenAIServingChat.chat_completion_full_generator = build_response
    api_router.JSONResponse = PreencodedJSONResponse  # ty:ignore[invalid-assignment]
    setattr(OpenAIServingChat, marker, True)
