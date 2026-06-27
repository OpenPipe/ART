from __future__ import annotations

import os
from typing import Any, cast

import torch

from art.megatron.model_support.spec import CompileWorkaroundConfig

_INSTALLED_CONFIG: tuple[frozenset[str], str] | None = None
_SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG = (
    "disable_compile_self_attn_linear_proj_reduce_scatter"
)


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


class _WeightedSwiGLUNoInnerForwardCast(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
    ) -> torch.Tensor:
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        x_glu, x_linear = torch.chunk(input, 2, dim=-1)
        return torch.nn.functional.silu(x_glu) * x_linear * weights

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        from megatron.core.fusions import fused_bias_swiglu

        grad_output = cast(torch.Tensor, grad_outputs[0])
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, weights_grad = fused_bias_swiglu.weighted_swiglu_back(
            grad_output,
            input,
            weights,
        )
        return input_grad, weights_grad, None


def _install_weighted_bias_swiglu_no_inner_forward_cast_workaround() -> None:
    from megatron.core.fusions import fused_bias_swiglu
    from megatron.core.transformer import mlp
    from megatron.core.transformer.moe import experts

    if getattr(
        fused_bias_swiglu.weighted_bias_swiglu_impl,
        "__art_no_inner_forward_cast__",
        False,
    ):
        return

    def _empty_weighted_swiglu_output(
        input: torch.Tensor,
        bias: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        output_shape = (*input.shape[:-1], int(input.shape[-1]) // 2)
        zero = input.sum() * 0.0 + weights.to(dtype=input.dtype).sum() * 0.0
        if bias is not None:
            zero = zero + bias.to(dtype=input.dtype).sum() * 0.0
        return zero.expand(output_shape).clone()

    def _weighted_bias_swiglu_no_inner_forward_cast(
        input: torch.Tensor,
        bias: torch.Tensor | None,
        weights: torch.Tensor,
        fp8_input_store: bool = False,
    ) -> torch.Tensor:
        if int(input.numel()) == 0:
            return _empty_weighted_swiglu_output(input, bias=bias, weights=weights)
        if bias is not None:
            raise NotImplementedError(
                "Bias is not supported for weighted swiglu fusion"
            )
        original_shape = input.shape
        output = _WeightedSwiGLUNoInnerForwardCast.apply(
            input.view(-1, original_shape[-1]),
            weights,
            fp8_input_store,
        ).to(input.dtype)
        return (
            output
            if len(original_shape) == 2
            else output.view(*original_shape[:-1], -1)
        )

    setattr(
        _weighted_bias_swiglu_no_inner_forward_cast,
        "__art_no_inner_forward_cast__",
        True,
    )
    setattr(
        fused_bias_swiglu,
        "weighted_bias_swiglu_impl",
        _weighted_bias_swiglu_no_inner_forward_cast,
    )
    setattr(
        mlp, "weighted_bias_swiglu_impl", _weighted_bias_swiglu_no_inner_forward_cast
    )
    setattr(
        experts,
        "weighted_bias_swiglu_impl",
        _weighted_bias_swiglu_no_inner_forward_cast,
    )


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
    if _SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG in flags:
        _install_self_attn_linear_proj_reduce_scatter_workaround()
    if "weighted_bias_swiglu_no_inner_forward_cast" in flags:
        _install_weighted_bias_swiglu_no_inner_forward_cast_workaround()
    deepep_flags = {"deepep_permute_restore", "deepep_dispatch_combine"} & flags
    if deepep_flags:
        deepep_manager = _require_attr(token_dispatcher, "_DeepepManager")
        if "deepep_permute_restore" in flags:
            _disable_attr(deepep_manager, "get_permuted_hidden_states_by_experts")
            _disable_attr(deepep_manager, "get_restored_hidden_states_by_experts")
        if "deepep_dispatch_combine" in flags:
            _disable_attr(deepep_manager, "dispatch")
            _disable_attr(deepep_manager, "combine")
    if "alltoall_dtoh" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = (
            _disable(
                token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
            )
        )
    if "alltoall_dispatch_preprocess" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher.dispatch_preprocess = _disable(
            token_dispatcher.MoEAlltoAllTokenDispatcher.dispatch_preprocess
        )
    if "alltoall_combine_postprocess" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher.combine_postprocess = _disable(
            token_dispatcher.MoEAlltoAllTokenDispatcher.combine_postprocess
        )
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
        from transformer_engine.pytorch.triton import (
            permutation as te_triton_permutation,
        )

        te_triton_permutation.permute_with_mask_map = _disable(
            te_triton_permutation.permute_with_mask_map
        )
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
    if "te_grouped_mlp_forward" in flags:
        moe_experts.TEGroupedMLP.forward = _disable(moe_experts.TEGroupedMLP.forward)
    _INSTALLED_CONFIG = installed_config
