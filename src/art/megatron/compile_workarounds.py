from __future__ import annotations

from copy import copy
from functools import wraps
import os
from typing import Any
import weakref

import torch
from torch._C import _current_graph_task_id
from torch._C._autograd import _get_current_graph_task_keep_graph

from art.megatron.model_support.spec import CompileWorkaroundConfig

_INSTALLED_CONFIG: tuple[frozenset[str], str] | None = None
_SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG = (
    "disable_compile_self_attn_linear_proj_reduce_scatter"
)


def install_te_reusable_backward() -> None:
    """Preserve TE's saved-tensor metadata until the final backward."""
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
    from transformer_engine.pytorch.module.layernorm_mlp import _LayerNormMLP
    from transformer_engine.pytorch.module.linear import _Linear
    from transformer_engine.pytorch.ops.fuser import _OperationFuserAutogradFunction

    for function in (
        _OperationFuserAutogradFunction,
        _Linear,
        _LayerNormLinear,
        _LayerNormMLP,
    ):
        _preserve_te_backward_metadata(function)


def _preserve_te_backward_metadata(function) -> None:
    original = function.backward
    if getattr(original, "__art_reusable_backward__", False):
        return

    @wraps(original)
    def backward(ctx, *gradients):
        if not _get_current_graph_task_keep_graph():
            return original(ctx, *gradients)
        tensor_objects = ctx.tensor_objects
        if any(value is not None for value in tensor_objects):
            raise RuntimeError(
                "Retained backward is not supported for Transformer Engine quantized saved tensors"
            )
        contexts = getattr(ctx, "basic_op_ctxs", ())
        ranges = [op_ctx._saved_tensors_range for op_ctx in contexts]
        try:
            return original(ctx, *gradients)
        finally:
            ctx.tensor_objects = tensor_objects
            for op_ctx, tensor_range in zip(contexts, ranges, strict=True):
                op_ctx._saved_tensors_range = tensor_range
                # Do not keep unpacked tensors alive between backward calls,
                # including when an operation raises before TE's own cleanup.
                op_ctx.saved_tensors = None

    setattr(backward, "__art_reusable_backward__", True)
    setattr(function, "backward", staticmethod(backward))


def install_reusable_checkpoint_backward() -> None:
    """Rebuild selective checkpoint outputs separately for each backward."""
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    original = CheckpointWithoutOutput._recompute
    if getattr(original, "__art_reusable_backward__", False):
        return
    fields = ("run_function", "rng_states", "outputs", "ctx")
    original_discard = CheckpointWithoutOutput.discard_output_and_register_recompute

    @wraps(original_discard)
    def discard(self, hook_tensor):
        # TransformerLayer retains the controller on the module. Transfer its
        # recipe to the graph hook so eviction can release the physical graph.
        if self.ctx is None:
            owned_ref = getattr(self, "_art_recompute_owner", None)
            if owned_ref is None:
                return original_discard(self, hook_tensor)
            owned = owned_ref()
            if owned is None:
                return
        else:
            owned = copy(self)
            self._art_recompute_owner = weakref.ref(owned)
        try:
            return original_discard(owned, hook_tensor)
        except BaseException:
            for field in fields:
                setattr(owned, field, None)
            raise
        finally:
            for field in fields:
                setattr(self, field, None)

    @wraps(original)
    def recompute(self, gradient):
        if not _get_current_graph_task_keep_graph():
            return original(self, gradient)
        task = _current_graph_task_id()
        if getattr(self, "_art_recompute_task", None) == task:
            return
        # The inner autograd graph is consumed normally. Preserve the forward
        # recipe so the next outer backward recomputes a fresh inner graph.
        state = tuple(getattr(self, field) for field in fields)
        try:
            result = original(self, gradient)
            self._art_recompute_task = task
            return result
        except BaseException:
            state = (None,) * len(fields)
            raise
        finally:
            for field, value in zip(fields, state, strict=True):
                setattr(self, field, value)

    setattr(recompute, "__art_reusable_backward__", True)
    setattr(CheckpointWithoutOutput, "_recompute", recompute)
    setattr(CheckpointWithoutOutput, "discard_output_and_register_recompute", discard)


def _require_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        raise RuntimeError(
            f"Required compile workaround target is missing: {obj}.{name}"
        )
    return value


def _disable(fn):
    if getattr(fn, "__art_compile_disabled__", False):
        return fn
    fn = getattr(fn, "_torchdynamo_orig_callable", fn)
    if getattr(fn, "__art_compile_disabled__", False):
        return fn
    wrapped = torch.compiler.disable(fn)
    setattr(wrapped, "__art_compile_disabled__", True)
    return wrapped


def _disable_attr(obj: Any, name: str) -> None:
    setattr(obj, name, _disable(_require_attr(obj, name)))


def _selected_workaround_flags(
    config: CompileWorkaroundConfig | None,
) -> set[str]:
    raw = os.environ.get("ART_MEGATRON_COMPILE_WORKAROUNDS", "").strip()
    if not raw:
        return set(() if config is None else config.flags)
    if raw.lower() in {"none", "off"}:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _install_context_parallel_attention_workaround() -> None:
    from art.megatron.context_parallel import core_attention, executor

    # CP attention owns custom comm and side-stream lifetime management. Keep
    # that wrapper eager; the inner flex attention kernels compile separately.
    executor.run_context_parallel = _disable(executor.run_context_parallel)
    core_attention.run_context_parallel = _disable(core_attention.run_context_parallel)
    core_attention.ArtContextParallelCoreAttention.forward = _disable(
        core_attention.ArtContextParallelCoreAttention.forward
    )


def _install_self_attn_linear_proj_reduce_scatter_workaround() -> None:
    from megatron.core.tensor_parallel import mappings

    from art.megatron import lora as art_lora

    # SelfAttentionLinearProjLoRA imports this symbol directly from
    # art.megatron.lora, so rebinding only megatron.core.tensor_parallel.mappings
    # leaves the compiled LoRA path untouched.
    wrapped = _disable(mappings.reduce_scatter_to_sequence_parallel_region)
    mappings.reduce_scatter_to_sequence_parallel_region = wrapped  # type: ignore[assignment]
    art_lora.reduce_scatter_to_sequence_parallel_region = wrapped  # type: ignore[assignment]


def _install_moe_postprocess_workaround(moe_layer: Any) -> None:
    # Routed token counts change across packed RL steps. Megatron's MoE
    # postprocess reshapes through dispatcher-owned shape state, which makes
    # this small boundary recompile repeatedly while the surrounding layer still
    # benefits from compilation.
    moe_layer.MoELayer.postprocess = _disable(moe_layer.MoELayer.postprocess)


def _install_gemma4_moe_postprocess_workaround() -> None:
    from megatron.bridge.models.gemma import gemma4_provider

    # Gemma4 overrides MoELayer.postprocess to normalize routed and shared
    # expert outputs. That override sees dynamic routed-token counts, so it
    # needs the same small eager boundary as the base MoE postprocess.
    gemma4_provider.Gemma4MoELayer.postprocess = _disable(
        gemma4_provider.Gemma4MoELayer.postprocess
    )


def _install_te_triton_mask_map_workaround() -> None:
    from transformer_engine.pytorch.triton import permutation

    # TE's mask-map path launches custom Triton kernels. Keep their thin Python
    # wrappers eager while the surrounding MoE layer and grouped GEMMs compile.
    for name in (
        "make_row_id_map",
        "permute_with_mask_map",
        "unpermute_with_mask_map",
    ):
        _disable_attr(permutation, name)


def _install_shared_expert_handoff_workaround() -> None:
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    # AOT drops standalone CUDA wait_stream graphs. Keep both ownership
    # handoffs eager, but leave shared-expert math compiled on its side stream.
    for name in ("pre_forward_comm", "get_output"):
        _disable_attr(SharedExpertMLP, name)


def install_torch_compile_workarounds(
    config: CompileWorkaroundConfig | None = None,
) -> None:
    global _INSTALLED_CONFIG
    flags = _selected_workaround_flags(config)
    shared_expert_state = "none" if config is None else config.shared_expert_state
    installed_config = (frozenset(flags), shared_expert_state)
    if _INSTALLED_CONFIG is not None:
        if _INSTALLED_CONFIG != installed_config:
            raise RuntimeError(
                "torch.compile workarounds already installed with a different config"
            )
        return
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.transformer.moe import experts as moe_experts
    from megatron.core.transformer.moe import moe_layer, moe_utils, token_dispatcher

    if "fake_sync_dealloc" in flags:
        try:

            @torch.library.register_fake("streams::sync_dealloc")
            def _sync_dealloc_fake(
                wait_event_index: int,
                src_stream_index: int,
                to_dealloc: torch.Tensor,
            ) -> None:
                del wait_event_index, src_stream_index, to_dealloc
                return None
        except RuntimeError as exc:
            if "already has a fake impl registered" not in str(exc):
                raise

    if "context_parallel_attention" in flags:
        _install_context_parallel_attention_workaround()
    if "shared_expert_stream_handoffs" in flags:
        _install_shared_expert_handoff_workaround()
    if _SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG in flags:
        _install_self_attn_linear_proj_reduce_scatter_workaround()
    if "moe_postprocess" in flags:
        _install_moe_postprocess_workaround(moe_layer)
    if "gemma4_moe_postprocess" in flags:
        _install_gemma4_moe_postprocess_workaround()

    if "te_moe_permute_with_probs" in flags:
        from transformer_engine.pytorch import permutation as te_permutation

        te_permutation.moe_permute_with_probs = _disable(
            te_permutation.moe_permute_with_probs
        )
        if te_ext.fused_permute_with_probs is not None:
            te_ext.fused_permute_with_probs = _disable(te_ext.fused_permute_with_probs)
        if moe_utils.fused_permute_with_probs is not None:
            moe_utils.fused_permute_with_probs = _disable(
                moe_utils.fused_permute_with_probs
            )
    if "te_triton_permute_with_mask_map" in flags:
        _install_te_triton_mask_map_workaround()
    if "te_moe_unpermute" in flags:
        from transformer_engine.pytorch import permutation as te_permutation

        te_permutation.moe_unpermute = _disable(te_permutation.moe_unpermute)
        if te_ext.fused_unpermute is not None:
            te_ext.fused_unpermute = _disable(te_ext.fused_unpermute)
        if moe_utils.fused_unpermute is not None:
            moe_utils.fused_unpermute = _disable(moe_utils.fused_unpermute)
    if "moe_utils_permute" in flags:
        moe_utils.permute = _disable(moe_utils.permute)
    if "moe_utils_unpermute" in flags:
        moe_utils.unpermute = _disable(moe_utils.unpermute)
    if "te_moe_unpermute_backward" in flags:
        from transformer_engine.pytorch import permutation as te_permutation

        setattr(
            te_permutation._moe_unpermute_mask_map,
            "backward",
            staticmethod(_disable(te_permutation._moe_unpermute_mask_map.backward)),
        )
    if "te_triton_unpermute_bwd_with_merging_probs" in flags:
        from transformer_engine.pytorch.triton import (
            permutation as te_triton_permutation,
        )

        te_triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs = _disable(
            te_triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs
        )
    if "alltoall_dispatch_dtoh" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = (
            _disable(
                token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
            )
        )
    if "flex_token_dispatch_combine" in flags:
        token_dispatcher.MoEFlexTokenDispatcher.token_dispatch = _disable(
            token_dispatcher.MoEFlexTokenDispatcher.token_dispatch
        )
        token_dispatcher.MoEFlexTokenDispatcher.token_combine = _disable(
            token_dispatcher.MoEFlexTokenDispatcher.token_combine
        )
    if "moe_preprocess" in flags:
        moe_layer.MoELayer.preprocess = _disable(moe_layer.MoELayer.preprocess)
    if "moe_forward" in flags:
        moe_layer.MoELayer.forward = _disable(moe_layer.MoELayer.forward)
    if "mlp_forward" in flags:
        from megatron.core.transformer import mlp

        mlp.MLP.forward = _disable(mlp.MLP.forward)
    if "te_grouped_mlp_forward" in flags:
        moe_experts.TEGroupedMLP.forward = _disable(moe_experts.TEGroupedMLP.forward)
    _INSTALLED_CONFIG = installed_config
