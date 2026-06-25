"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
import importlib
import inspect
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_vllm_runtime_patches() -> None:
    from art_vllm_runtime.gemma4_moe_lora_patch import (
        patch_gemma4_moe_lora_support,
    )

    patch_transformers_v5_compat()
    patch_gemma4_moe_lora_support()
    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    patch_nccl_unique_id_bootstrap()
    patch_layerwise_reload_shadow_attrs()
    patch_dsv4_attn_sink_layerwise_reload()
    patch_dsv4_mhc_pre_fixed_split()
    patch_dsv4_lora_support()
    patch_dsv4_mla_lora_aliases()
    patch_dsv4_fast_path_lora()
    patch_lora_linear_base_attr_proxy()
    patch_marlin_lora_swiglu_limit()
    patch_art_lora_delta_weight_update()
    patch_gemma4_checkpoint_weight_update_reload()
    patch_routed_experts_prefix_cache_sidecar()


def patch_transformers_v5_compat() -> None:
    _patch_rope_validation_ignore_keys()
    _patch_qwen3_vl_moe_tie_word_embeddings()
    _patch_gemma4_moe_experts_per_tok_alias()


def _patch_rope_validation_ignore_keys() -> None:
    from transformers.configuration_utils import PretrainedConfig

    original = PretrainedConfig.convert_rope_params_to_dict
    if getattr(original, "__art_patched__", False):
        return

    def patched(self: Any, ignore_keys_at_rope_validation: Any = None, **kwargs: Any):
        if ignore_keys_at_rope_validation is not None:
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return original(
            self,
            ignore_keys_at_rope_validation=ignore_keys_at_rope_validation,
            **kwargs,
        )

    patched.__art_patched__ = True  # type: ignore[attr-defined]
    PretrainedConfig.convert_rope_params_to_dict = patched  # type: ignore[method-assign]


def _patch_qwen3_vl_moe_tie_word_embeddings() -> None:
    from transformers import Qwen3VLMoeTextConfig

    setattr(Qwen3VLMoeTextConfig, "tie_word_embeddings", False)


def _patch_gemma4_moe_experts_per_tok_alias() -> None:
    from transformers import Gemma4TextConfig

    if hasattr(Gemma4TextConfig, "num_experts_per_tok"):
        return

    def num_experts_per_tok(self: Any) -> Any:
        return self.top_k_experts

    Gemma4TextConfig.num_experts_per_tok = property(num_experts_per_tok)  # type: ignore[attr-defined]


def subclass_chat_completion_request() -> None:
    from vllm.entrypoints.openai.chat_completion import protocol

    if getattr(protocol, "_art_chat_completion_request_patched", False):
        return

    class ChatCompletionRequest(protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0
            self.return_token_ids = True

    protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]
    setattr(protocol, "_art_chat_completion_request_patched", True)


def patch_listen_for_disconnect() -> None:
    from vllm.entrypoints.serve.utils import api_utils

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


def patch_tool_parser_manager() -> None:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    original = ToolParserManager.get_tool_parser
    if getattr(original, "__art_patched__", False):
        return

    def patched_get_tool_parser(name: str) -> type:
        tool_parser_class = original(name)
        current = tool_parser_class.extract_tool_calls_streaming
        if getattr(current, "__art_patched__", False):
            return tool_parser_class

        def patch(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return current(*args, **kwargs) or DeltaMessage()

        patch.__art_patched__ = True  # type: ignore[attr-defined]
        tool_parser_class.extract_tool_calls_streaming = patch  # ty:ignore[invalid-assignment]
        return tool_parser_class

    patched_get_tool_parser.__art_patched__ = True  # type: ignore[attr-defined]
    ToolParserManager.get_tool_parser = patched_get_tool_parser  # ty:ignore[invalid-assignment]


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


def patch_gemma4_checkpoint_weight_update_reload() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    original_start_weight_update = Worker.start_weight_update
    if getattr(original_start_weight_update, "__art_patched__", False):
        return
    original_finish_weight_update = Worker.finish_weight_update

    def start_weight_update(
        self: Any,
        is_checkpoint_format: bool = True,
    ) -> None:
        if not is_checkpoint_format or not _is_gemma4_conditional_worker(self):
            return original_start_weight_update(
                self,
                is_checkpoint_format=is_checkpoint_format,
            )
        self._check_weight_transfer_engine()
        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )
        self._is_checkpoint_format = True
        self._weight_update_active = True

    def finish_weight_update(self: Any) -> None:
        if not _is_gemma4_conditional_worker(self):
            return original_finish_weight_update(self)
        self._check_weight_transfer_engine()
        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before finish_weight_update."
            )
        if not self._is_checkpoint_format:
            return original_finish_weight_update(self)
        self._weight_update_active = False
        self._is_checkpoint_format = True

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
        if update_info.get("art_weight_update_kind") != ART_LORA_DELTA_UPDATE_KIND:
            return original_update_weights(self, update_info)

        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before update_weights."
            )

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

        with torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=collect_lora_tensors,
            )
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

    update_weights.__art_lora_delta_patched__ = True  # type: ignore[attr-defined]
    update_weights.__art_original__ = original_update_weights  # type: ignore[attr-defined]
    Worker.update_weights = update_weights  # type: ignore[method-assign]


def _drop_reload_shadow_attrs(layer: Any, names: Any) -> None:
    for name in names:
        if (
            name in getattr(layer, "__dict__", {})
            and name not in layer._parameters
            and name not in layer._buffers
            and name not in layer._modules
        ):
            delattr(layer, name)


def patch_layerwise_reload_shadow_attrs() -> None:
    """Allow vLLM layerwise reload to restore processed DSV4 MegaMoE params.

    DeepSeek V4 MegaMoE drops loader-side Parameters after transforming them for
    DeepGEMM. Some vLLM builds leave same-name plain attributes behind; PyTorch
    then rejects register_parameter during the next checkpoint-format reload.
    """
    from vllm.model_executor.model_loader.reload import layerwise, meta

    if getattr(meta, "_art_reload_shadow_attrs_patched", False):
        return

    original_restore_layer_on_meta = meta.restore_layer_on_meta
    original_place_kernel_tensors = layerwise._place_kernel_tensors

    def restore_layer_on_meta(layer: Any, info: Any) -> None:
        restore_params, restore_buffers = info.restore_metadata
        _drop_reload_shadow_attrs(layer, tuple(restore_params) + tuple(restore_buffers))
        return original_restore_layer_on_meta(layer, info)

    def _place_kernel_tensors(layer: Any, info: Any) -> None:
        assert info.kernel_tensors is not None
        parameters, buffers = info.kernel_tensors
        _drop_reload_shadow_attrs(layer, tuple(parameters) + tuple(buffers))
        return original_place_kernel_tensors(layer, info)

    restore_layer_on_meta.__art_patched__ = True  # type: ignore[attr-defined]
    _place_kernel_tensors.__art_patched__ = True  # type: ignore[attr-defined]
    meta.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise._place_kernel_tensors = _place_kernel_tensors  # type: ignore[method-assign]
    setattr(meta, "_art_reload_shadow_attrs_patched", True)


def _import_dsv4_model_module() -> Any | None:
    for module_name in (
        "vllm.model_executor.models.deepseek_v4",
        "vllm.models.deepseek_v4.nvidia.model",
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def patch_dsv4_attn_sink_layerwise_reload() -> None:
    """Route DSV4 attention-sink loads through vLLM's layerwise loader.

    Merged-weight transfer uses vLLM checkpoint-format reload. During that path,
    every loadable parameter must be applied through its `weight_loader`; direct
    `copy_` into `attn_sink` bypasses layerwise accounting and finalize restores
    the old kernel tensor. With `load_format=dummy`, that old tensor is the
    initialized sink, not the checkpoint sink.
    """
    dsv4_model = _import_dsv4_model_module()
    if dsv4_model is None:
        return
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    model_cls = getattr(dsv4_model, "DeepseekV4Model", None)
    if model_cls is None:
        return
    original = model_cls.load_weights
    if getattr(original, "__art_patched__", False):
        return

    def load_weights(self: Any, weights: Any) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = dsv4_model.get_tensor_model_parallel_world_size()
        tp_rank = dsv4_model.get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == dsv4_model.torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(dsv4_model.torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        success = param.weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                if "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    padded_weight = loaded_weight.new_full(
                        tuple(param.shape), -float("inf")
                    )
                    padded_weight[: narrow_weight.shape[0]].copy_(narrow_weight)
                    weight_loader = getattr(
                        param, "weight_loader", dsv4_model.default_weight_loader
                    )
                    weight_loader(param, padded_weight)
                    loaded_params.add(name)
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", dsv4_model.default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    load_weights.__art_patched__ = True  # type: ignore[attr-defined]
    model_cls.load_weights = load_weights  # type: ignore[method-assign]


def patch_dsv4_mhc_pre_fixed_split() -> None:
    """Make DSV4 mHC pre reductions invariant to total prefill length.

    vLLM's current DSV4 mHC pre path chooses split-K from ``num_tokens``. The
    same prompt prefix can therefore use a different reduction tree when a
    suffix is appended, producing float32-level post/comb mix drift that becomes
    bf16 output drift and later MoE route changes. Pinning the DSV4 prenorm
    shape to the TileLang kernel default split count keeps the reduction plan
    stable without changing model math.
    """
    try:
        mhc = importlib.import_module("vllm.model_executor.layers.mhc")
    except ImportError:
        return
    original = getattr(mhc, "compute_num_split", None)
    if original is None or getattr(original, "__art_dsv4_fixed_split_patched__", False):
        return

    def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
        if block_k == 64 and k == 16_384:
            return 16
        return original(block_k, k, grid_size)

    compute_num_split.__art_dsv4_fixed_split_patched__ = True  # type: ignore[attr-defined]
    mhc.compute_num_split = compute_num_split


def patch_dsv4_lora_support() -> None:
    """Enable vLLM's existing LoRA manager for ART-served DSV4.

    DSV4 itself does not need a custom LoRA executor here. Once the model
    advertises packed MLA/shared-expert modules and MoE expert children, vLLM
    wraps the same FusedMoE module it already uses for serving. With LoRA
    enabled, vLLM's modular MoE selector picks Marlin, whose expert backend
    supports fused MoE LoRA. Do not point this patch at the FlashInfer TRTLLM
    MXFP4 backend; that backend currently has no LoRA hooks.
    """
    dsv4_model = _import_dsv4_model_module()
    if dsv4_model is None:
        return
    model_cls = getattr(dsv4_model, "DeepseekV4ForCausalLM", None)
    if model_cls is None or getattr(model_cls, "_art_dsv4_lora_patched", False):
        return
    model_cls.supports_lora = True
    model_cls.embedding_modules = {}
    model_cls.packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls.is_3d_moe_weight = False
    model_cls.is_non_gated_moe = False
    model_cls.lora_skip_prefixes = ["mtp", "indexer"]
    model_cls._art_dsv4_lora_patched = True
    _patch_dsv4_lora_manager_indexer_skip(model_cls)


def _patch_dsv4_lora_manager_indexer_skip(model_cls: type) -> None:
    from vllm.lora.model_manager import LoRAModelManager

    original = LoRAModelManager._match_target_modules
    if getattr(original, "__art_dsv4_indexer_skip_patched__", False):
        return

    duplicate_mla_aliases = {"fused_wqa_wkv", "wq_b", "wo_a", "wo_b"}

    def _match_target_modules(self: Any, module_name: str) -> bool:
        if isinstance(self.model, model_cls) and ".indexer." in module_name:
            return False
        if (
            isinstance(self.model, model_cls)
            and ".attn.mla_attn." in module_name
            and module_name.rsplit(".", 1)[-1] in duplicate_mla_aliases
        ):
            return False
        return original(self, module_name)

    _match_target_modules.__art_dsv4_indexer_skip_patched__ = True  # type: ignore[attr-defined]
    LoRAModelManager._match_target_modules = _match_target_modules  # type: ignore[method-assign]


def patch_dsv4_mla_lora_aliases() -> None:
    """Keep DSV4 MLA wrapper references aligned with vLLM LoRA replacements.

    DeepSeek V4 stores direct references to several attention linears inside
    ``mla_attn`` during model construction. vLLM's LoRA manager later replaces
    the canonical modules on ``attn``. Without refreshing the aliases, the DSV4
    custom attention path keeps calling the original unwrapped linears.
    """
    from vllm.lora.model_manager import LoRAModelManager

    original = LoRAModelManager.register_module
    if getattr(original, "__art_dsv4_mla_alias_patched__", False):
        return

    alias_names = {"fused_wqa_wkv", "wq_b", "wo_a", "wo_b"}

    def register_module(self: Any, module_name: str, module: Any) -> Any:
        result = original(self, module_name, module)
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in alias_names:
            return result
        parent_name, _, _ = module_name.rpartition(".")
        if parent_name.endswith(".mla_attn"):
            mla_name = parent_name
        else:
            mla_name = f"{parent_name}.mla_attn"
        try:
            mla_attn = self.model.get_submodule(mla_name)
        except AttributeError:
            return result
        if mla_attn is not None and hasattr(mla_attn, leaf):
            setattr(mla_attn, leaf, module)
        return result

    register_module.__art_dsv4_mla_alias_patched__ = True  # type: ignore[attr-defined]
    LoRAModelManager.register_module = register_module  # type: ignore[method-assign]


def _is_lora_wrapped_linear(module: Any) -> bool:
    return all(
        hasattr(module, name)
        for name in ("lora_a_stacked", "lora_b_stacked", "punica_wrapper")
    )


def _apply_lora_to_existing_linear_output(
    module: Any,
    x: Any,
    output: Any,
) -> Any:
    if not _is_lora_wrapped_linear(module):
        return output
    wrapper = module.punica_wrapper
    if getattr(wrapper, "no_lora", False):
        return output
    if getattr(wrapper, "indices_len", [None])[0] is None:
        return output
    return module._apply_lora_to_output(x, output)


def _register_dsv4_lora_expand_fp32_output_op() -> None:
    if getattr(_register_dsv4_lora_expand_fp32_output_op, "_registered", False):
        return

    import torch
    from vllm import envs
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs, supports_pdl
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op

    @triton.jit
    def _kernel(
        input_ptr,
        lora_b_ptr,
        output_ptr,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        M,
        slice_offset,
        input_stride_m,
        input_stride_k,
        lora_stride_lora,
        lora_stride_out,
        lora_stride_k,
        output_stride_m,
        output_stride_n,
        RANK: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_GDC: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        cta_n_num = tl.cdiv(OUT_WIDTH, BLOCK_N)
        cta_m_num = tl.cdiv(M, BLOCK_M)
        pid_mn = tl.program_id(0)
        pid_m = pid_mn % cta_m_num
        pid_n = (pid_mn // cta_m_num) % cta_n_num
        lora_idx = tl.program_id(1)

        lora_id = tl.load(lora_ids + lora_idx)
        if lora_id == -1:
            return

        lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
        cta_m_offset = pid_m * BLOCK_M
        if cta_m_offset >= lora_m_size:
            return

        cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
        lora_m_start = tl.load(lora_token_start_loc + lora_idx)
        row_ptr = token_indices_sorted_by_lora_ids + lora_m_start + cta_m_offset
        row_offsets = tl.arange(0, BLOCK_M) % cta_m_len
        rows = tl.load(row_ptr + row_offsets)

        offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, RANK, BLOCK_K):
            offs_k = tl.arange(0, BLOCK_K) + k_start
            x = tl.load(
                input_ptr
                + rows[:, None] * input_stride_m
                + offs_k[None, :] * input_stride_k,
                mask=(row_offsets[:, None] < cta_m_len) & (offs_k[None, :] < RANK),
                other=0.0,
            )
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            b = tl.load(
                lora_b_ptr
                + lora_id * lora_stride_lora
                + offs_n[None, :] * lora_stride_out
                + offs_k[:, None] * lora_stride_k,
                mask=(offs_k[:, None] < RANK) & (offs_n[None, :] < OUT_WIDTH),
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(x, b)

        out_cols = slice_offset + offs_n
        out_ptrs = (
            output_ptr
            + rows[:, None] * output_stride_m
            + out_cols[None, :] * output_stride_n
        )
        mask = (row_offsets[:, None] < cta_m_len) & (offs_n[None, :] < OUT_WIDTH)
        old = tl.load(out_ptrs, mask=mask)
        tl.store(out_ptrs, old + accumulator, mask=mask)

    @torch.inference_mode()
    def _impl(
        inputs: torch.Tensor,
        lora_b_weight: torch.Tensor,
        output_tensor: torch.Tensor,
        token_indices_sorted_by_lora_ids: torch.Tensor,
        num_tokens_per_lora: torch.Tensor,
        lora_token_start_loc: torch.Tensor,
        lora_ids: torch.Tensor,
        no_lora_flag_cpu: torch.Tensor,
        num_active_loras: torch.Tensor,
        slice_offset: int,
    ) -> None:
        assert no_lora_flag_cpu.numel() == 1
        if no_lora_flag_cpu.item():
            return
        assert inputs.dtype == torch.float32
        assert output_tensor.dtype == torch.float32
        assert output_tensor.is_contiguous()
        assert lora_b_weight.ndim == 4
        assert lora_b_weight.size(1) == 1
        assert lora_b_weight.is_contiguous()

        m = inputs.size(0)
        out_width = lora_b_weight.size(2)
        rank = lora_b_weight.size(3)
        kernel_config = get_lora_op_configs(
            op_type="expand",
            max_loras=lora_ids.size(0),
            batch=m,
            hidden_size=out_width,
            rank=rank,
            num_slices=1,
            add_inputs=True,
        )
        block_m = kernel_config["block_m"]
        block_n = kernel_config["block_n"]
        block_k = kernel_config["block_k"]
        use_gdc = supports_pdl(inputs.device) and envs.VLLM_LORA_ENABLE_DUAL_STREAM
        grid = (
            triton.cdiv(m, block_m) * triton.cdiv(out_width, block_n),
            num_active_loras.item(),
        )
        _kernel[grid](
            inputs,
            lora_b_weight,
            output_tensor,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            m,
            slice_offset,
            inputs.stride(0),
            inputs.stride(1),
            lora_b_weight.stride(0),
            lora_b_weight.stride(2),
            lora_b_weight.stride(3),
            output_tensor.stride(0),
            output_tensor.stride(1),
            RANK=rank,
            OUT_WIDTH=out_width,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_GDC=use_gdc,
            num_warps=kernel_config["num_warps"],
            num_ctas=kernel_config["num_ctas"],
            num_stages=kernel_config["num_stages"],
            launch_pdl=use_gdc,
        )

    def _fake(
        inputs: torch.Tensor,
        lora_b_weight: torch.Tensor,
        output_tensor: torch.Tensor,
        token_indices_sorted_by_lora_ids: torch.Tensor,
        num_tokens_per_lora: torch.Tensor,
        lora_token_start_loc: torch.Tensor,
        lora_ids: torch.Tensor,
        no_lora_flag_cpu: torch.Tensor,
        num_active_loras: torch.Tensor,
        slice_offset: int,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="art_dsv4_lora_expand_fp32_output",
        op_func=_impl,
        mutates_args=["output_tensor"],
        fake_impl=_fake,
    )
    _register_dsv4_lora_expand_fp32_output_op._registered = True  # type: ignore[attr-defined]


def _apply_dsv4_compressor_lora_to_existing_output(
    module: Any,
    x: Any,
    output: Any,
) -> Any:
    if not _is_active_lora_wrapped_linear(module):
        return output

    import torch
    from vllm.platforms import current_platform

    _register_dsv4_lora_expand_fp32_output_op()
    original_shape = output.shape if output.ndim == 3 else None
    if x.ndim == 3 and output.ndim == 3:
        x = x.flatten(0, 1)
        output = output.flatten(0, 1)
    if x.ndim != 2 or output.ndim != 2:
        raise RuntimeError(
            "DSV4 compressor LoRA expects 2D hidden/output tensors, got "
            f"x={tuple(x.shape)}, output={tuple(output.shape)}."
        )
    if output.dtype != torch.float32:
        raise RuntimeError(
            "DSV4 compressor LoRA must preserve the fp32 compressor projection, "
            f"got output dtype {output.dtype}."
        )
    if getattr(module, "tp_size", 1) != 1:
        raise RuntimeError(
            "DSV4 compressor LoRA expects disable_tp compressor linears."
        )

    wrapper = module.punica_wrapper
    local_rank = module.lora_a_stacked[0].shape[2]
    buffers = torch.empty_strided(
        (len(module.output_slices), x.shape[0], local_rank),
        (x.shape[0] * local_rank, local_rank, 1),
        dtype=torch.float32,
        device=x.device,
    )
    shrunk = wrapper.add_shrink(buffers, x, module.lora_a_stacked, 1.0)
    if not current_platform.can_update_inplace():
        buffers = shrunk

    (
        _,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    ) = wrapper.token_mapping_meta.meta_args(
        x.shape[0], wrapper.lora_config.specialize_active_lora
    )
    offset = 0
    for idx, width in enumerate(module.output_slices):
        torch.ops.vllm.art_dsv4_lora_expand_fp32_output(
            buffers[idx],
            module.lora_b_stacked[idx],
            output,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            no_lora_flag_cpu,
            num_active_loras,
            offset,
        )
        offset += width
    if offset != output.shape[-1]:
        raise RuntimeError(
            "DSV4 compressor LoRA output slice width mismatch: "
            f"slices={module.output_slices}, output={tuple(output.shape)}."
        )
    return output.reshape(original_shape) if original_shape is not None else output


def _is_active_lora_wrapped_linear(module: Any) -> bool:
    if not _is_lora_wrapped_linear(module):
        return False
    wrapper = module.punica_wrapper
    return not getattr(wrapper, "no_lora", False) and (
        getattr(wrapper, "indices_len", [None])[0] is not None
    )


def _register_dsv4_inv_rope_lora_input_op() -> None:
    if getattr(_register_dsv4_inv_rope_lora_input_op, "_registered", False):
        return

    import torch
    from vllm.platforms import current_platform
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op

    @triton.jit
    def _kernel(
        o_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        fp8_ptr,
        scale_ptr,
        lora_input_ptr,
        num_tokens,
        heads_per_group: tl.constexpr,
        o_stride_token,
        o_stride_head,
        cache_stride_pos,
        fp8_stride_group,
        fp8_stride_token,
        scale_stride_group,
        scale_stride_k,
        lora_stride_group,
        lora_stride_token,
        fp8_max: tl.constexpr,
        eps: tl.constexpr,
        QUANT_GROUP_SIZE: tl.constexpr,
        CHUNKS_PER_HEAD: tl.constexpr,
        ROPE_START: tl.constexpr,
        HALF_ROPE: tl.constexpr,
        TMA_ALIGNED_SCALES: tl.constexpr,
    ):
        pid_token = tl.program_id(0).to(tl.int64)
        pid_gh = tl.program_id(1).to(tl.int64)
        g = pid_gh // heads_per_group
        head_in_group = pid_gh % heads_per_group
        qb_start = head_in_group * CHUNKS_PER_HEAD

        if pid_token >= num_tokens:
            if TMA_ALIGNED_SCALES:
                scale_addr = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + head_in_group * scale_stride_k
                )
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
            else:
                block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
                qb_indices = qb_start + block_offsets
                scale_addrs = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + qb_indices * scale_stride_k
                )
                tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
            return

        head_dim: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
        offsets = tl.arange(0, head_dim)
        input_base = o_ptr + pid_token * o_stride_token + pid_gh * o_stride_head
        x = tl.load(input_base + offsets).to(tl.float32)

        rope_abs_start: tl.constexpr = (
            CHUNKS_PER_HEAD - 1
        ) * QUANT_GROUP_SIZE + ROPE_START
        pos = tl.load(positions_ptr + pid_token)
        cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
        is_rope = offsets >= rope_abs_start
        rope_local = offsets - rope_abs_start

        x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
            tl.float32
        )
        cs_idx = tl.maximum(rope_local >> 1, 0)
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
        x_add = x * cos_v + x_partner * sin_v
        x_sub = x * cos_v - x_partner * sin_v
        is_even = (rope_local & 1) == 0
        x = tl.where(is_rope, tl.where(is_even, x_add, x_sub), x)

        group_head_offset = head_in_group * head_dim
        lora_base = (
            lora_input_ptr
            + g * lora_stride_group
            + pid_token * lora_stride_token
            + group_head_offset
        )
        tl.store(lora_base + offsets, x)

        x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scale_raw = block_absmax * (1.0 / fp8_max)
        scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))
        scales_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
            ),
            (head_dim,),
        )
        x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

        fp8_base = (
            fp8_ptr
            + g * fp8_stride_group
            + pid_token * fp8_stride_token
            + qb_start * QUANT_GROUP_SIZE
        )
        tl.store(fp8_base + offsets, x_quant)

        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        if TMA_ALIGNED_SCALES:
            scale_bits = scales.to(tl.int32, bitcast=True)
            ue8m0_bytes = (scale_bits >> 23) & 0xFF
            packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, packed_val)
        else:
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, scales)

    def _impl(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        fp8_buf: torch.Tensor,
        scale_buf: torch.Tensor,
        lora_input: torch.Tensor,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        tma_aligned_t: int,
        num_tokens: int,
    ) -> None:
        grid = (tma_aligned_t, fp8_buf.shape[0] * heads_per_group)
        pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
        _kernel[grid](
            o,
            positions,
            cos_sin_cache,
            fp8_buf,
            scale_buf,
            lora_input,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            fp8_stride_group=fp8_buf.stride(0),
            fp8_stride_token=fp8_buf.stride(1),
            scale_stride_group=scale_buf.stride(0),
            scale_stride_k=scale_buf.stride(2),
            lora_stride_group=lora_input.stride(0),
            lora_stride_token=lora_input.stride(1),
            fp8_max=fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=rope_start,
            HALF_ROPE=half_rope,
            TMA_ALIGNED_SCALES=tma_aligned_scales,
            num_stages=1,
            num_warps=1,
            **pdl_kwargs,
        )

    def _fake(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        fp8_buf: torch.Tensor,
        scale_buf: torch.Tensor,
        lora_input: torch.Tensor,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        tma_aligned_t: int,
        num_tokens: int,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="art_dsv4_inv_rope_fp8_quant_lora_input",
        op_func=_impl,
        mutates_args=["fp8_buf", "scale_buf", "lora_input"],
        fake_impl=_fake,
    )
    _register_dsv4_inv_rope_lora_input_op._registered = True  # type: ignore[attr-defined]


def _dsv4_fused_inv_rope_fp8_quant_with_lora_input(
    dsv4_attn: Any,
    o: Any,
    positions: Any,
    cos_sin_cache: Any,
    *,
    n_groups: int,
    heads_per_group: int,
    lora_dtype: Any,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[Any, Any, Any]:
    import torch
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    fp8_dtype = torch.float8_e4m3fn
    tma_aligned_t = get_tma_aligned_size(num_tokens, 4)
    scale_inner = (
        (num_scale_blocks + 3) // 4 if tma_aligned_scales else num_scale_blocks
    )

    fp8_buf = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )
    lora_input = torch.empty(
        (n_groups, num_tokens, d),
        dtype=lora_dtype,
        device=o.device,
    )
    dsv4_attn.torch.ops.vllm.art_dsv4_inv_rope_fp8_quant_lora_input(
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        lora_input,
        heads_per_group,
        quant_group_size,
        chunks_per_head,
        nope_dim % quant_group_size,
        rope_dim // 2,
        tma_aligned_scales,
        torch.finfo(fp8_dtype).max,
        tma_aligned_t,
        num_tokens,
    )
    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1), lora_input


def _dsv4_wo_a_lora_b_group_cache(
    wo_a: Any,
    *,
    n_local_groups: int,
    out_per_group: int,
) -> tuple[Any, ...]:
    source = wo_a.lora_b_stacked[0]
    key = (
        source.data_ptr(),
        getattr(source, "_version", None),
        n_local_groups,
        out_per_group,
        source.shape[-1],
    )
    if getattr(wo_a, "_art_wo_a_lora_b_group_cache_key", None) != key:
        wo_a._art_wo_a_lora_b_group_cache = tuple(
            source[
                :, :, group * out_per_group : (group + 1) * out_per_group, :
            ].contiguous()
            for group in range(n_local_groups)
        )
        wo_a._art_wo_a_lora_b_group_cache_key = key
    return wo_a._art_wo_a_lora_b_group_cache


def _apply_dsv4_wo_a_lora_fast(
    wo_a: Any,
    z: Any,
    *,
    lora_input: Any,
    n_local_groups: int,
) -> Any:
    if not _is_active_lora_wrapped_linear(wo_a):
        return z

    import torch
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.platforms import current_platform

    wrapper = wo_a.punica_wrapper
    out_per_group = z.shape[-1]
    z_flat = z.view(z.shape[0], n_local_groups * out_per_group)
    group_b = _dsv4_wo_a_lora_b_group_cache(
        wo_a,
        n_local_groups=n_local_groups,
        out_per_group=out_per_group,
    )
    local_rank = wo_a.lora_a_stacked[0].shape[2]
    buffers = torch.empty_strided(
        (n_local_groups, z.shape[0], local_rank),
        (z.shape[0] * local_rank, local_rank, 1),
        dtype=torch.float32,
        device=z.device,
    )
    for group, lora_b in enumerate(group_b):
        buffer = buffers[group : group + 1]
        buffer.zero_()
        shrunk = wrapper.add_shrink(buffer, lora_input[group], wo_a.lora_a_stacked, 1.0)
        if not current_platform.can_update_inplace():
            buffer = shrunk
        buffer = tensor_model_parallel_all_gather(buffer)
        expanded = wrapper.add_expand(
            z_flat,
            buffer,
            (lora_b,),
            (out_per_group,),
            offset_start=group * out_per_group,
            add_inputs=True,
        )
        if not current_platform.can_update_inplace():
            z_flat = expanded
            z = z_flat.view_as(z)
    return z


def patch_dsv4_fast_path_lora() -> None:
    """Apply LoRA deltas on DSV4 paths that read base weights directly.

    vLLM's generic LoRA manager can wrap DSV4 linear modules, but the DSV4
    Flash runtime bypasses some wrapped forwards for performance: compressor
    projections are direct ``hidden @ fused_wkv_wgate.weight.T`` calls, and
    ``wo_a`` is a custom inverse-RoPE/FP8/einsum path. Without this patch vLLM
    accepts and activates these adapter tensors while silently omitting their
    deltas from generation.
    """
    dsv4_attn = importlib.import_module(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    wrapper_cls = getattr(dsv4_attn, "DeepseekV4MultiHeadLatentAttentionWrapper", None)
    if wrapper_cls is None:
        return
    _register_dsv4_inv_rope_lora_input_op()
    _register_dsv4_lora_expand_fp32_output_op()
    if getattr(wrapper_cls, "_art_fast_path_lora_patched", False):
        return

    original_attn_gemm_parallel_execute = wrapper_cls.attn_gemm_parallel_execute
    original_forward = wrapper_cls.forward

    def attn_gemm_parallel_execute(self: Any, hidden_states: Any) -> tuple[Any, ...]:
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            original_attn_gemm_parallel_execute(self, hidden_states)
        )
        if self.compressor is not None:
            kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.compressor.fused_wkv_wgate,
                hidden_states,
                kv_score,
            )
        if self.indexer is not None:
            indexer_kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.indexer.compressor.fused_wkv_wgate,
                hidden_states,
                indexer_kv_score,
            )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def forward(
        self: Any,
        positions: Any,
        hidden_states: Any,
        llama_4_scaling: Any | None = None,
    ) -> Any:
        if dsv4_attn.current_platform.is_rocm():
            return original_forward(self, positions, hidden_states, llama_4_scaling)

        num_tokens = hidden_states.shape[0]
        o_padded = dsv4_attn.torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        dsv4_attn.torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        wo_a_lora_input = None
        if _is_active_lora_wrapped_linear(self.wo_a):
            o_fp8, o_scale, wo_a_lora_input = (
                _dsv4_fused_inv_rope_fp8_quant_with_lora_input(
                    dsv4_attn,
                    o,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    n_groups=self.n_local_groups,
                    heads_per_group=self.n_local_heads // self.n_local_groups,
                    lora_dtype=self.wo_a.lora_a_stacked[0].dtype,
                    nope_dim=self.nope_head_dim,
                    rope_dim=self.rope_head_dim,
                    tma_aligned_scales=self._tma_aligned_scales,
                )
            )
        else:
            o_fp8, o_scale = dsv4_attn.fused_inv_rope_fp8_quant(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                tma_aligned_scales=self._tma_aligned_scales,
            )

        z = dsv4_attn.torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=dsv4_attn.torch.bfloat16,
        )
        dsv4_attn.torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            self.wo_a.weight,
            self.wo_a.weight_scale_inv,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )
        if wo_a_lora_input is not None:
            z = _apply_dsv4_wo_a_lora_fast(
                self.wo_a,
                z,
                lora_input=wo_a_lora_input,
                n_local_groups=self.n_local_groups,
            )
        return self.wo_b(z.flatten(1))

    attn_gemm_parallel_execute.__art_patched__ = True  # type: ignore[attr-defined]
    forward.__art_patched__ = True  # type: ignore[attr-defined]
    wrapper_cls.attn_gemm_parallel_execute = attn_gemm_parallel_execute
    wrapper_cls.forward = forward
    wrapper_cls._art_fast_path_lora_patched = True


def _base_layer_attr_proxy(name: str) -> property:
    def attr(self: Any) -> Any:
        return getattr(self.base_layer, name)

    return property(attr)


def patch_lora_linear_base_attr_proxy() -> None:
    """Expose DSV4 base metadata through vLLM linear LoRA wrappers.

    DeepSeek V4's output attention path calls a custom FP8 einsum directly and
    reads ``wo_a.weight_scale_inv`` next to ``wo_a.weight``. vLLM's linear LoRA
    wrappers already proxy ``weight`` but not the quant scale. Its router also
    reads dynamic gate metadata from ``self.gate`` after that gate can be LoRA
    wrapped. Keep these tensors owned by the base layer instead of copying or
    re-registering them on every wrapper.
    """
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    if getattr(BaseLinearLayerWithLoRA, "_art_base_attr_proxy_patched", False):
        return

    for name in ("weight_scale_inv", "tid2eid", "e_score_correction_bias"):
        if not hasattr(BaseLinearLayerWithLoRA, name):
            setattr(BaseLinearLayerWithLoRA, name, _base_layer_attr_proxy(name))
    BaseLinearLayerWithLoRA._art_base_attr_proxy_patched = True


def patch_marlin_lora_swiglu_limit() -> None:
    """Keep Marlin MoE LoRA active when DSV4 uses a SwiGLU clamp limit.

    vLLM's Marlin LoRA path injects W13 LoRA inside the activation callback and
    stores that activated cache for W2 LoRA. DSV4 sets ``gemm1_clamp_limit``;
    upstream Marlin bypasses the callback in that case and calls the clamp op
    directly, so W13 LoRA is skipped and W2 LoRA later misses ``cache2``. Route
    the callback through the same clamp op while preserving Marlin execution.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

    original_apply = MarlinExperts.apply
    if getattr(original_apply, "__art_patched__", False):
        return

    sentinel = object()

    def apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        clamp_limit = getattr(self, "gemm1_clamp_limit", None)
        if getattr(self, "_lora_context", None) is None or clamp_limit is None:
            return original_apply(self, *args, **kwargs)

        original_activation = self.activation
        previous_activation = self.__dict__.get("activation", sentinel)
        previous_clamp_limit = self.gemm1_clamp_limit

        def activation_with_clamp(
            activation: Any,
            output: Any,
            input: Any,
        ) -> None:
            if activation == MoEActivation.SILU:
                swiglu_limit_func(output, input, clamp_limit)
            else:
                original_activation(activation, output, input)

        self.activation = activation_with_clamp
        self.gemm1_clamp_limit = None
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            self.gemm1_clamp_limit = previous_clamp_limit
            if previous_activation is sentinel:
                delattr(self, "activation")
            else:
                self.activation = previous_activation

    apply.__art_patched__ = True  # type: ignore[attr-defined]
    MarlinExperts.apply = apply  # type: ignore[method-assign]


def _lora_cache_key(lora_request: Any) -> tuple[Any, ...]:
    if lora_request is None:
        return ()
    return (
        getattr(lora_request, "adapter_id", None),
        getattr(lora_request, "name", None),
        getattr(lora_request, "path", None),
    )


def _request_token_ids(req_state: Any) -> list[int] | None:
    prompt_token_ids = getattr(req_state, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return None
    return list(prompt_token_ids) + list(getattr(req_state, "output_token_ids", ()))


def _route_block_key(
    token_ids: list[int],
    end: int,
    lora_key: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (lora_key, tuple(token_ids[:end]))


def _runner_block_size(runner: Any) -> int:
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups and len(groups) == 1:
        return int(groups[0].kv_cache_spec.block_size)
    return int(getattr(runner.cache_config, "block_size", 16))


def _request_snapshots(
    runner: Any, ordered: dict[str, int]
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for req_id in ordered:
        req_state = runner.requests.get(req_id)
        if req_state is None:
            continue
        token_ids = _request_token_ids(req_state)
        if token_ids is None:
            continue
        snapshots[req_id] = {
            "token_ids": token_ids,
            "lora_key": _lora_cache_key(getattr(req_state, "lora_request", None)),
            "num_computed_tokens": int(getattr(req_state, "num_computed_tokens", 0)),
        }
    return snapshots


def patch_routed_experts_prefix_cache_sidecar() -> None:
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer

    if getattr(routed_experts_capturer, "_art_prefix_route_sidecar_patched", False):
        return

    host_cls = routed_experts_capturer._RoutedExpertsHostCache
    capturer_cls = routed_experts_capturer._RoutedExpertsCapturerReal

    original_host_init = host_cls.__init__
    original_get_or_grow_buffer = host_cls.get_or_grow_buffer
    original_free_request = host_cls.free_request
    original_scatter_to_host = capturer_cls._scatter_to_host
    original_get_routed_experts = capturer_cls.get_routed_experts
    original_issue_routing_d2h_copy = routed_experts_capturer.issue_routing_d2h_copy

    def host_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_host_init(self, *args, **kwargs)
        self._art_req_filled_masks: dict[str, np.ndarray] = {}
        self._art_prefix_route_blocks: dict[tuple[Any, ...], np.ndarray] = {}
        self._art_prefix_route_waiters: dict[
            tuple[Any, ...], list[tuple[str, int, int]]
        ] = {}
        self._art_prefix_route_needs_by_req: dict[str, set[tuple[Any, ...]]] = {}
        self._art_prefix_route_hydrated_tokens = 0
        self._art_prefix_route_cache_misses = 0
        self._art_prefix_route_cache_conflicts = 0

    def get_or_grow_buffer(self: Any, req_id: str, max_pos: int) -> np.ndarray:
        buf = original_get_or_grow_buffer(self, req_id, max_pos)
        mask = self._art_req_filled_masks.get(req_id)
        if mask is None:
            self._art_req_filled_masks[req_id] = np.zeros(buf.shape[0], dtype=np.bool_)
        elif mask.shape[0] < buf.shape[0]:
            new_mask = np.zeros(buf.shape[0], dtype=np.bool_)
            new_mask[: mask.shape[0]] = mask
            self._art_req_filled_masks[req_id] = new_mask
        return buf

    def free_request(self: Any, req_id: str) -> None:
        original_free_request(self, req_id)
        self._art_req_filled_masks.pop(req_id, None)
        for key in self._art_prefix_route_needs_by_req.pop(req_id, set()):
            waiters = self._art_prefix_route_waiters.get(key)
            if waiters is None:
                continue
            waiters = [waiter for waiter in waiters if waiter[0] != req_id]
            if waiters:
                self._art_prefix_route_waiters[key] = waiters
            else:
                self._art_prefix_route_waiters.pop(key, None)

    def mark_filled(self: Any, req_id: str, positions: np.ndarray) -> None:
        if positions.size == 0:
            return
        self.get_or_grow_buffer(req_id, int(positions.max()))
        self._art_req_filled_masks[req_id][positions] = True

    def require_filled(self: Any, req_id: str, seqlen: int) -> None:
        mask = self._art_req_filled_masks.get(req_id)
        if mask is None or mask.shape[0] < seqlen or not bool(mask[:seqlen].all()):
            available = (
                mask[:seqlen] if mask is not None else np.zeros(0, dtype=np.bool_)
            )
            missing = np.flatnonzero(~available)[:16].tolist()
            raise RuntimeError(
                "Routed expert capture is incomplete for request "
                f"{req_id}: seqlen={seqlen}, first_missing_positions={missing}"
            )

    def fill_prefix_block(
        self: Any,
        req_id: str,
        start: int,
        end: int,
        value: np.ndarray,
        key: tuple[Any, ...] | None = None,
    ) -> bool:
        buf = self.get_or_grow_buffer(req_id, end - 1)
        mask = self._art_req_filled_masks[req_id]
        if bool(mask[start:end].all()):
            if key is not None:
                needs = self._art_prefix_route_needs_by_req.get(req_id)
                if needs is not None:
                    needs.discard(key)
                    if not needs:
                        self._art_prefix_route_needs_by_req.pop(req_id, None)
            return False
        buf[start:end] = value
        mask[start:end] = True
        self.update_filled_len(req_id, end - 1)
        if key is not None:
            needs = self._art_prefix_route_needs_by_req.get(req_id)
            if needs is not None:
                needs.discard(key)
                if not needs:
                    self._art_prefix_route_needs_by_req.pop(req_id, None)
        return True

    def store_prefix_block(
        self: Any,
        key: tuple[Any, ...],
        value: np.ndarray,
    ) -> None:
        existing = self._art_prefix_route_blocks.get(key)
        if existing is None:
            existing = value.copy()
            self._art_prefix_route_blocks[key] = existing
        elif not np.array_equal(existing, value):
            self._art_prefix_route_cache_conflicts += 1
        hydrated = 0
        for req_id, start, end in self._art_prefix_route_waiters.pop(key, []):
            if self._art_fill_prefix_block(req_id, start, end, existing, key):
                hydrated += end - start
        if hydrated:
            self._art_prefix_route_hydrated_tokens += hydrated
            logger.info(
                "Hydrated %s routed-expert prefix-cache tokens from materialized "
                "route block",
                hydrated,
            )

    def store_prefix_blocks(
        self: Any,
        req_id: str,
        token_ids: list[int],
        lora_key: tuple[Any, ...],
        block_size: int,
        max_pos_exclusive: int,
    ) -> None:
        if block_size <= 0:
            return
        upper = min(max_pos_exclusive, len(token_ids))
        upper -= upper % block_size
        if upper <= 0:
            return
        buf = self.get_buffer(req_id)
        mask = self._art_req_filled_masks.get(req_id)
        if buf is None or mask is None:
            return
        for end in range(block_size, upper + 1, block_size):
            start = end - block_size
            if end > mask.shape[0] or not bool(mask[start:end].all()):
                continue
            key = _route_block_key(token_ids, end, lora_key)
            value = buf[start:end].copy()
            self._art_store_prefix_block(key, value)

    def need_cached_prefix(
        self: Any,
        req_id: str,
        token_ids: list[int],
        lora_key: tuple[Any, ...],
        cached_len: int,
        block_size: int,
    ) -> None:
        if block_size <= 0 or cached_len <= 0:
            return
        upper = min(cached_len, len(token_ids))
        upper -= upper % block_size
        if upper <= 0:
            return
        hydrated = 0
        for end in range(block_size, upper + 1, block_size):
            start = end - block_size
            mask = self._art_req_filled_masks.get(req_id)
            if (
                mask is not None
                and end <= mask.shape[0]
                and bool(mask[start:end].all())
            ):
                continue
            key = _route_block_key(token_ids, end, lora_key)
            value = self._art_prefix_route_blocks.get(key)
            if value is None:
                needs = self._art_prefix_route_needs_by_req.setdefault(req_id, set())
                if key not in needs:
                    self._art_prefix_route_waiters.setdefault(key, []).append(
                        (req_id, start, end)
                    )
                    needs.add(key)
                    self._art_prefix_route_cache_misses += block_size
                continue
            if self._art_fill_prefix_block(req_id, start, end, value, key):
                hydrated += block_size
        if hydrated:
            self._art_prefix_route_hydrated_tokens += hydrated
            logger.info(
                "Hydrated %s routed-expert prefix-cache tokens for request %s",
                hydrated,
                req_id,
            )

    def require_no_unmet_prefix_route_needs(self: Any, req_id: str) -> None:
        needs = self._art_prefix_route_needs_by_req.get(req_id)
        if needs:
            raise RuntimeError(
                "Routed expert capture is missing materialized prefix-cache "
                f"route blocks for request {req_id}: unmet_blocks={len(needs)}"
            )

    def scatter_to_host(self: Any) -> None:
        positions = self._pending_positions.copy()
        scheduled = dict(self._pending_num_scheduled or {})
        metadata = getattr(self, "_art_pending_route_metadata", None)
        original_scatter_to_host(self)
        host_cache = self.host_cache
        if host_cache is None:
            return
        block_size = int((metadata or {}).get("block_size", 0))
        snapshots = (metadata or {}).get("snapshots", {})
        offset = 0
        for req_id, n_tokens in scheduled.items():
            pos = positions[offset : offset + n_tokens]
            host_cache._art_mark_filled(req_id, pos)
            snapshot = snapshots.get(req_id)
            if snapshot is not None and pos.size:
                host_cache._art_store_prefix_blocks(
                    req_id,
                    snapshot["token_ids"],
                    snapshot["lora_key"],
                    block_size,
                    int(pos.max()) + 1,
                )
            offset += n_tokens
        self._art_pending_route_metadata = None

    def get_routed_experts(
        self: Any,
        req_id: str,
        seqlen: int | None = None,
        free_slot: bool = True,
    ) -> np.ndarray | None:
        if self.host_cache is not None:
            filled = self.host_cache.get_filled_len(req_id)
            effective_len = min(filled, seqlen) if seqlen is not None else filled
            if effective_len > 0:
                self.host_cache._art_require_no_unmet_prefix_route_needs(req_id)
                self.host_cache._art_require_filled(req_id, effective_len)
        return original_get_routed_experts(self, req_id, seqlen, free_slot)

    def issue_routing_d2h_copy(
        input_batch_req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        positions: Any,
        positions_cpu: Any,
    ) -> None:
        capturer = routed_experts_capturer.get_global_experts_capturer()
        host_cache = capturer.get_host_cache() if capturer is not None else None
        frame = inspect.currentframe()
        runner = frame.f_back.f_locals.get("self") if frame and frame.f_back else None
        ordered = {
            req_id: num_scheduled_tokens[req_id]
            for req_id in input_batch_req_ids
            if req_id in num_scheduled_tokens
        }
        metadata: dict[str, Any] | None = None
        if host_cache is not None and runner is not None:
            block_size = _runner_block_size(runner)
            snapshots = _request_snapshots(runner, ordered)
            for req_id, snapshot in snapshots.items():
                host_cache._art_need_cached_prefix(
                    req_id,
                    snapshot["token_ids"],
                    snapshot["lora_key"],
                    snapshot["num_computed_tokens"],
                    block_size,
                )
            metadata = {"block_size": block_size, "snapshots": snapshots}
        original_issue_routing_d2h_copy(
            input_batch_req_ids,
            num_scheduled_tokens,
            positions,
            positions_cpu,
        )
        if capturer is not None and metadata is not None and sum(ordered.values()) > 0:
            capturer._art_pending_route_metadata = metadata

    host_cls.__init__ = host_init  # type: ignore[method-assign]
    host_cls.get_or_grow_buffer = get_or_grow_buffer  # type: ignore[method-assign]
    host_cls.free_request = free_request  # type: ignore[method-assign]
    host_cls._art_mark_filled = mark_filled  # type: ignore[attr-defined]
    host_cls._art_require_filled = require_filled  # type: ignore[attr-defined]
    host_cls._art_fill_prefix_block = fill_prefix_block  # type: ignore[attr-defined]
    host_cls._art_store_prefix_block = store_prefix_block  # type: ignore[attr-defined]
    host_cls._art_store_prefix_blocks = store_prefix_blocks  # type: ignore[attr-defined]
    host_cls._art_need_cached_prefix = need_cached_prefix  # type: ignore[attr-defined]
    host_cls._art_require_no_unmet_prefix_route_needs = (  # type: ignore[attr-defined]
        require_no_unmet_prefix_route_needs
    )
    capturer_cls._scatter_to_host = scatter_to_host  # type: ignore[method-assign]
    capturer_cls.get_routed_experts = get_routed_experts  # type: ignore[method-assign]
    from vllm.v1.worker import gpu_model_runner

    gpu_model_runner_issue_routing_d2h_copy = getattr(
        gpu_model_runner, "issue_routing_d2h_copy", None
    )
    if gpu_model_runner_issue_routing_d2h_copy is not original_issue_routing_d2h_copy:
        raise RuntimeError(
            "ART routed-expert prefix-cache patch expected "
            "vllm.v1.worker.gpu_model_runner.issue_routing_d2h_copy to reference "
            "vllm.model_executor.layers.fused_moe.routed_experts_capturer."
            "issue_routing_d2h_copy. vLLM internals changed; update the patch."
        )

    routed_experts_capturer.issue_routing_d2h_copy = issue_routing_d2h_copy
    gpu_model_runner.issue_routing_d2h_copy = issue_routing_d2h_copy
    setattr(routed_experts_capturer, "_art_prefix_route_sidecar_patched", True)
