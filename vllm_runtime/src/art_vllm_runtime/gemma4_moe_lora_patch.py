"""Gemma4 MoE LoRA compatibility for ART's vLLM runtime."""

from typing import Any


def patch_gemma4_moe_lora_support() -> None:
    """Expose Gemma4's FusedMoE metadata to vLLM's native LoRA path."""
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

    # Remove this shim when upstream vLLM Gemma4 MoE defines these natively.
    Gemma4ForCausalLM.is_3d_moe_weight = True
    Gemma4ForConditionalGeneration.is_3d_moe_weight = True

    if not hasattr(Gemma4ForCausalLM, "get_expert_mapping"):

        def get_causal_expert_mapping(
            self: Any,
        ) -> list[tuple[str, str, int, str]]:
            return fused_moe_make_expert_params_mapping(
                self.model,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=int(getattr(self.config, "num_experts", 0) or 0),
                num_redundant_experts=0,
            )

        get_causal_expert_mapping.__art_patched__ = True  # type: ignore[attr-defined]
        Gemma4ForCausalLM.get_expert_mapping = get_causal_expert_mapping  # type: ignore[attr-defined,method-assign]

    if not hasattr(Gemma4ForConditionalGeneration, "get_expert_mapping"):

        def get_conditional_expert_mapping(
            self: Any,
        ) -> list[tuple[str, str, int, str]]:
            return self.language_model.get_expert_mapping()

        get_conditional_expert_mapping.__art_patched__ = True  # type: ignore[attr-defined]
        Gemma4ForConditionalGeneration.get_expert_mapping = (  # type: ignore[attr-defined,method-assign]
            get_conditional_expert_mapping
        )
