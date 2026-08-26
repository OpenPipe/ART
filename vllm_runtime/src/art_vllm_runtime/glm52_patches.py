"""GLM-5.2 adaptations for the ART-owned vLLM runtime."""


def apply_glm52_vllm_runtime_patches() -> None:
    patch_glm52_lora_metadata()


def patch_glm52_lora_metadata() -> None:
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

    GlmMoeDsaForCausalLM.is_3d_moe_weight = True
    GlmMoeDsaForCausalLM.lora_skip_prefixes = ["indexer"]
