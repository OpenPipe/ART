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
