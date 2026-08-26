from __future__ import annotations

from types import MethodType
from typing import Any, Sequence, cast

import torch

from .exchange import (
    MambaShardShape,
    canonical_head_shard_to_token_layout,
    projected_tokens_to_canonical_head_shard,
)
from .operator import MambaParameters, run_mamba_tree
from .plan import MambaExecutionPlan

MAMBA_STATE_KEY = "mamba_2"
_ACTIVE_STATE = "_art_mamba_prefix_tree_state"


def install_mamba_prefix_tree_hooks(model_chunks: Sequence[Any]) -> None:
    """Route only Nemotron Mamba/attention layers through ART prefix-tree state."""

    from megatron.core.ssm.mamba_block import MambaStack
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    for chunk in model_chunks:
        for module in chunk.modules():
            if isinstance(module, MambaLayer) and not getattr(
                module, "_art_mamba_layer_hooked", False
            ):
                original = module.forward

                def layer_forward(
                    self: Any,
                    *args: Any,
                    _original=original,
                    **kwargs: Any,
                ) -> Any:
                    state = kwargs.get("attention_mask")
                    if not _has_mamba_plan(state):
                        return _original(*args, **kwargs)
                    if kwargs.get("packed_seq_params") is not None:
                        raise ValueError(
                            "ART Mamba tree execution owns sequence packing"
                        )
                    setattr(self.mixer, _ACTIVE_STATE, state)
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        delattr(self.mixer, _ACTIVE_STATE)

                module.forward = MethodType(layer_forward, module)
                module._art_mamba_layer_hooked = True
            elif isinstance(module, MambaMixer) and not getattr(
                module, "_art_mamba_mixer_hooked", False
            ):
                original = module.forward

                def mixer_forward(
                    self: Any,
                    hidden_states: torch.Tensor,
                    *args: Any,
                    _original=original,
                    **kwargs: Any,
                ) -> Any:
                    state = getattr(self, _ACTIVE_STATE, None)
                    if state is None:
                        return _original(hidden_states, *args, **kwargs)
                    if args or kwargs.get("inference_context") is not None:
                        raise ValueError(
                            "ART Mamba prefix-tree execution is training-only"
                        )
                    return mamba_prefix_tree_forward(self, hidden_states, state)

                module.forward = MethodType(mixer_forward, module)
                module._art_mamba_mixer_hooked = True
            elif isinstance(module, TransformerLayer) and not getattr(
                module, "_art_mamba_attention_hooked", False
            ):
                original = module.forward

                def attention_forward(
                    self: Any,
                    *args: Any,
                    _original=original,
                    **kwargs: Any,
                ) -> Any:
                    state = kwargs.get("attention_mask")
                    if _has_mamba_plan(state):
                        kwargs = dict(kwargs)
                        attention_bias = kwargs.get("attention_bias")
                        if attention_bias is not None and attention_bias is not state:
                            raise ValueError(
                                "Nemotron attention received two mask states"
                            )
                        kwargs["attention_bias"] = state
                    return _original(*args, **kwargs)

                module.forward = MethodType(attention_forward, module)
                module._art_mamba_attention_hooked = True
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, MambaStack) or getattr(
                module, "_art_mamba_stack_hooked", False
            ):
                continue
            original = module.forward

            def stack_forward(
                self: Any,
                hidden_states: Any,
                attention_mask: Any,
                inference_context: Any | None = None,
                rotary_pos_emb: torch.Tensor | None = None,
                *,
                inference_params: Any | None = None,
                packed_seq_params: Any | None = None,
                padding_mask: torch.Tensor | None = None,
                _original=original,
            ) -> torch.Tensor:
                if not _has_mamba_plan(attention_mask):
                    return _original(
                        hidden_states,
                        attention_mask,
                        inference_context,
                        rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        padding_mask=padding_mask,
                    )
                return mamba_prefix_tree_stack_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )

            module.forward = MethodType(stack_forward, module)
            module._art_mamba_stack_hooked = True


def mamba_prefix_tree_stack_forward(
    stack: Any,
    hidden_states: Any,
    attention_state: Any,
    *,
    inference_context: Any | None = None,
    rotary_pos_emb: torch.Tensor | None = None,
    inference_params: Any | None = None,
    packed_seq_params: Any | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a native hybrid stack with the recompute contract MCore omits."""
    if inference_context is not None or inference_params is not None:
        raise ValueError(
            "ART Mamba prefix-tree execution does not support inference state"
        )
    if packed_seq_params is not None:
        raise ValueError("ART Mamba tree execution owns sequence packing")
    if stack.config.fp8 or stack.config.fp4:
        raise ValueError("ART Mamba full-stack execution currently requires BF16/FP32")
    if not stack.pre_process:
        hidden_states = stack.input_tensor

    from megatron.core import tensor_parallel
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.utils import WrappedTensor, make_viewless_tensor

    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("MambaStack requires tensor hidden states")

    def run_range(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end):
            layer = stack.layers[index]
            physical = getattr(layer, "_orig_mod", layer)
            if isinstance(physical, TransformerLayer):
                value = layer(
                    hidden_states=value,
                    attention_mask=attention_state,
                    inference_context=None,
                    rotary_pos_emb=rotary_pos_emb,
                    sequence_len_offset=None,
                    packed_seq_params=None,
                    padding_mask=padding_mask,
                )
            else:
                value = layer(
                    hidden_states=value,
                    attention_mask=attention_state,
                    inference_context=None,
                    packed_seq_params=None,
                )
            if isinstance(value, tuple):
                value = value[0]
        return value

    granularity = stack.config.recompute_granularity
    if granularity not in (None, "full"):
        raise ValueError("ART Mamba supports eager or full activation recomputation")
    if granularity == "full" and stack.training:
        method = stack.config.recompute_method
        count = int(stack.config.recompute_num_layers or 0)
        if method not in ("uniform", "block") or count <= 0:
            raise ValueError(
                "Mamba full recompute requires uniform/block and a layer count"
            )
        if torch.is_grad_enabled() and not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        def checkpoint_range(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
            return tensor_parallel.checkpoint(
                lambda tensor: run_range(tensor, start, end),
                bool(stack.config.distribute_saved_activations),
                value,
            )

        if method == "uniform":
            for start in range(0, len(stack.layers), count):
                hidden_states = checkpoint_range(
                    hidden_states, start, min(start + count, len(stack.layers))
                )
        else:
            for index in range(len(stack.layers)):
                hidden_states = (
                    checkpoint_range(hidden_states, index, index + 1)
                    if index < count
                    else run_range(hidden_states, index, index + 1)
                )
    else:
        hidden_states = run_range(hidden_states, 0, len(stack.layers))
    if stack.post_process and stack.post_layer_norm:
        hidden_states = stack.final_norm(hidden_states)
    return make_viewless_tensor(
        inp=hidden_states,
        requires_grad=hidden_states.requires_grad,
        keep_graph=True,
    )


def mamba_prefix_tree_forward(
    mixer: Any,
    hidden_states: torch.Tensor,
    attention_state: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    plan = cast(MambaExecutionPlan, attention_state.model_state[MAMBA_STATE_KEY])
    if int(mixer.chunk_size) != plan.chunk_size:
        raise ValueError(
            f"Mamba kernel chunk size {mixer.chunk_size} disagrees with plan {plan.chunk_size}"
        )
    projected, _ = mixer.in_proj(hidden_states)
    shape = MambaShardShape(
        inner=int(mixer.d_inner_local_tp),
        heads=int(mixer.nheads_local_tp),
        groups=int(mixer.ngroups_local_tp),
        state_dim=int(mixer.d_state),
    )
    canonical = projected_tokens_to_canonical_head_shard(
        projected,
        plan.exchange,
        shape,
        mixer.pg_collection.cp,
    )
    cp = mixer.cp
    recurrent = run_mamba_tree(
        canonical,
        plan,
        MambaParameters(
            conv_weight=cp.get_conv1d_weight().squeeze(1),
            conv_bias=cp.get_conv1d_bias(),
            dt_bias=cp.get_dt_bias(),
            a_log=cp.get_A_log(),
            d=cp.get_D(),
            head_dim=int(mixer.headdim),
            state_dim=int(mixer.d_state),
            num_groups=int(cp.ngroups_local_tpcp),
        ),
    )
    gate = canonical[:, : recurrent.shape[-1]]
    local = canonical_head_shard_to_token_layout(
        torch.cat((recurrent, gate), dim=-1),
        tuple(projected.shape),
        plan.exchange,
        shape,
        mixer.pg_collection.cp,
    )
    if not mixer.rmsnorm:
        raise RuntimeError("ART Mamba tree execution requires gated RMSNorm")
    local, gate = local.chunk(2, dim=-1)
    local = mixer.norm(local, gate)
    return mixer.out_proj(local)


def _has_mamba_plan(state: Any) -> bool:
    return isinstance(getattr(state, "model_state", None), dict) and isinstance(
        state.model_state.get(MAMBA_STATE_KEY), MambaExecutionPlan
    )
