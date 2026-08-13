from __future__ import annotations

import os
from typing import Any

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
