"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

from types import SimpleNamespace
from typing import Any


def apply_vllm_runtime_patches() -> None:
    from art_vllm_runtime.dsv4_patches import apply_dsv4_vllm_runtime_patches
    from art_vllm_runtime.gemma4_moe_lora_patch import (
        patch_gemma4_moe_lora_support,
    )
    from art_vllm_runtime.glm52_patches import apply_glm52_vllm_runtime_patches
    from art_vllm_runtime.moe_lora_patches import (
        patch_small_batch_moe_lora_intermediate_dtype,
    )
    from art_vllm_runtime.policy_spans import patch_policy_token_spans
    from art_vllm_runtime.qwen35_patches import apply_qwen35_vllm_runtime_patches

    patch_policy_token_spans()
    patch_gemma4_moe_lora_support()
    subclass_chat_completion_request()
    patch_nonstreaming_chat_response_offload()
    patch_small_batch_moe_lora_intermediate_dtype()
    apply_glm52_vllm_runtime_patches()
    apply_dsv4_vllm_runtime_patches()
    apply_qwen35_vllm_runtime_patches()
    patch_weight_update_lifecycle()
    patch_art_lora_delta_weight_update()
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

    if getattr(protocol, "_art_chat_completion_request_patched", False):
        return

    class ChatCompletionRequest(protocol.ChatCompletionRequest):
        logprobs: bool | None = True
        top_logprobs: int | None = 0
        return_token_ids: bool | None = True

    protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]
    setattr(protocol, "_art_chat_completion_request_patched", True)


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


def _is_gemma4_conditional_worker(worker: Any) -> bool:
    hf_config = worker.model_config.hf_config
    return hf_config.architectures == ["Gemma4ForConditionalGeneration"]


def patch_weight_update_lifecycle() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    original_start_weight_update = Worker.start_weight_update
    if getattr(original_start_weight_update, "__art_patched__", False):
        return
    original_finish_weight_update = Worker.finish_weight_update

    def start_weight_update(self: Any) -> None:
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )
        # vLLM 0.25 removed format selection from this endpoint. Defer the
        # checkpoint reload lifecycle until update_weights reveals the payload.
        self._art_weight_update_mode = None
        self._art_weight_transfer_started = False
        self._weight_update_active = True

    def finish_weight_update(self: Any) -> None:
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        if not self._weight_update_active:
            raise RuntimeError(
                "finish_weight_update called without a matching start_weight_update."
            )
        if self._art_weight_transfer_started:
            self.weight_transfer_engine.finish_weight_update()
        self._weight_update_active = False
        self._art_weight_update_mode = None
        self._art_weight_transfer_started = False

    start_weight_update.__art_patched__ = True  # type: ignore[attr-defined]
    start_weight_update.__art_original__ = original_start_weight_update  # type: ignore[attr-defined]
    finish_weight_update.__art_patched__ = True  # type: ignore[attr-defined]
    finish_weight_update.__art_original__ = original_finish_weight_update  # type: ignore[attr-defined]
    Worker.start_weight_update = start_weight_update  # type: ignore[method-assign]
    Worker.finish_weight_update = finish_weight_update  # type: ignore[method-assign]


def patch_art_lora_delta_weight_update() -> None:
    import torch
    from vllm.v1.worker.gpu_worker import Worker

    from art_vllm_runtime.lora_delta import (
        ART_LORA_DELTA_UPDATE_KIND,
        apply_lora_delta_update,
    )

    original_update_weights = Worker.update_weights
    if getattr(original_update_weights, "__art_lora_delta_patched__", False):
        return

    def update_weights(self: Any, update_info: dict) -> None:
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before update_weights."
            )

        is_lora_delta = (
            update_info.get("art_weight_update_kind") == ART_LORA_DELTA_UPDATE_KIND
        )
        mode = ART_LORA_DELTA_UPDATE_KIND if is_lora_delta else "checkpoint"
        active_mode = getattr(self, "_art_weight_update_mode", None)
        if active_mode not in (None, mode):
            raise RuntimeError(
                f"Cannot mix {active_mode!r} and {mode!r} in one weight update"
            )
        self._art_weight_update_mode = mode

        if not is_lora_delta:
            if active_mode is None and not _is_gemma4_conditional_worker(self):
                self.weight_transfer_engine.start_weight_update()
                self._art_weight_transfer_started = True
            return original_update_weights(self, update_info)

        adapter_config = update_info["art_lora_config"]
        transfer_update_info = dict(update_info)
        del transfer_update_info["art_weight_update_kind"]
        del transfer_update_info["art_lora_config"]
        typed_update_info = self.weight_transfer_engine.parse_update_info(
            transfer_update_info
        )
        lora_tensors: dict[str, torch.Tensor] = {}

        def collect_lora_tensors(weights: list[tuple[str, torch.Tensor]]) -> None:
            for name, tensor in weights:
                if name in lora_tensors:
                    raise RuntimeError(f"Duplicate LoRA tensor in update: {name}")
                lora_tensors[name] = tensor.detach().contiguous().clone()

        engine = self.weight_transfer_engine
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        if not isinstance(engine, NCCLWeightTransferEngine):
            raise RuntimeError("ART LoRA delta updates require vLLM's NCCL transport")
        model = engine.model
        engine.model = SimpleNamespace(load_weights=collect_lora_tensors)
        try:
            with torch.device(self.device):
                engine.receive_weights(typed_update_info)
                self._art_previous_lora_tensors = apply_lora_delta_update(
                    model=self.model_runner.model,
                    lora_tensors=lora_tensors,
                    adapter_config=adapter_config,
                    previous_lora_tensors=getattr(
                        self,
                        "_art_previous_lora_tensors",
                        None,
                    ),
                )
            torch.accelerator.synchronize()
        except BaseException:
            self._weight_update_active = False
            raise
        finally:
            engine.model = model

    update_weights.__art_lora_delta_patched__ = True  # type: ignore[attr-defined]
    update_weights.__art_original__ = original_update_weights  # type: ignore[attr-defined]
    Worker.update_weights = update_weights  # type: ignore[method-assign]
