"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
import importlib
from types import SimpleNamespace
from typing import Any


def apply_vllm_runtime_patches() -> None:
    _patch_openai_namespace_tool_import()

    from art_vllm_runtime.dsv4_patches import apply_dsv4_vllm_runtime_patches
    from art_vllm_runtime.gemma4_moe_lora_patch import (
        patch_gemma4_moe_lora_support,
    )
    from art_vllm_runtime.glm52_patches import apply_glm52_vllm_runtime_patches
    from art_vllm_runtime.policy_spans import patch_policy_token_spans

    _patch_qwen3_vl_moe_tie_word_embeddings()
    patch_policy_token_spans()
    patch_gemma4_moe_lora_support()
    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_nccl_unique_id_bootstrap()
    apply_glm52_vllm_runtime_patches()
    apply_dsv4_vllm_runtime_patches()
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


def _patch_openai_namespace_tool_import() -> None:
    import openai.types.responses as responses

    if hasattr(responses, "NamespaceTool"):
        return

    class NamespaceTool:
        pass

    responses.NamespaceTool = NamespaceTool  # type: ignore[attr-defined]


def _patch_qwen3_vl_moe_tie_word_embeddings() -> None:
    from transformers import Qwen3VLMoeTextConfig

    setattr(Qwen3VLMoeTextConfig, "tie_word_embeddings", False)


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


def patch_listen_for_disconnect() -> None:
    try:
        api_utils = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")
    except ModuleNotFoundError:
        api_utils = importlib.import_module("vllm.entrypoints.utils")

    if getattr(api_utils, "_art_listen_for_disconnect_patched", False):
        return

    async def patched_listen_for_disconnect(request: Any) -> None:
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    if getattr(
                        request.app.state, "enable_server_load_tracking", False
                    ) and hasattr(request.app.state, "server_load_metrics"):
                        request.app.state.server_load_metrics -= 1
                    break
        except UnboundLocalError:
            pass

    api_utils.listen_for_disconnect = patched_listen_for_disconnect  # ty:ignore[invalid-assignment]
    setattr(api_utils, "_art_listen_for_disconnect_patched", True)


def _restore_nccl_unique_id_payload(
    payload: object,
    template: object | None,
) -> object:
    from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId

    if not isinstance(payload, (bytes, bytearray)) or not isinstance(
        template, ncclUniqueId
    ):
        return payload
    raw = bytes(payload)
    assert len(raw) == ctypes.sizeof(ncclUniqueId)
    unique_id = ncclUniqueId()
    ctypes.memmove(ctypes.byref(unique_id), raw, len(raw))
    return unique_id


def _normalize_nccl_comm_init_rank_unique_id(library: Any, unique_id: object) -> object:
    if isinstance(unique_id, (bytes, bytearray)):
        return library.unique_id_from_bytes(bytes(unique_id))
    return unique_id


def patch_nccl_unique_id_bootstrap() -> None:
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.distributed.utils import StatelessProcessGroup

    original_broadcast = StatelessProcessGroup.broadcast_obj
    if not getattr(original_broadcast, "__art_patched__", False):

        def patched_broadcast(self: Any, obj: Any | None, src: int) -> Any:
            return _restore_nccl_unique_id_payload(
                original_broadcast(self, obj, src), obj
            )

        patched_broadcast.__art_patched__ = True  # type: ignore[attr-defined]
        StatelessProcessGroup.broadcast_obj = patched_broadcast  # type: ignore[method-assign]

    original_comm_init_rank = NCCLLibrary.ncclCommInitRank
    if getattr(original_comm_init_rank, "__art_patched__", False):
        return

    def patched_comm_init_rank(
        self: Any,
        world_size: int,
        unique_id: object,
        rank: int,
    ) -> Any:
        unique_id = _normalize_nccl_comm_init_rank_unique_id(self, unique_id)
        return original_comm_init_rank(self, world_size, unique_id, rank)

    patched_comm_init_rank.__art_patched__ = True  # type: ignore[attr-defined]
    NCCLLibrary.ncclCommInitRank = patched_comm_init_rank  # type: ignore[method-assign]


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
