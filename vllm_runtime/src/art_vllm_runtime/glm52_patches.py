"""GLM-5.2 adaptations for the ART-owned vLLM runtime."""

from functools import wraps
from typing import Any


class _SharedTopkBuffer:
    def __init__(self, topk_indices_buffer: Any, topk_tokens: int) -> None:
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = topk_tokens


def apply_glm52_vllm_runtime_patches() -> None:
    patch_glm52_lora_metadata()
    patch_glm52_shared_indexers()
    patch_glm52_indexer_rope()


def patch_glm52_lora_metadata() -> None:
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

    GlmMoeDsaForCausalLM.is_3d_moe_weight = True
    GlmMoeDsaForCausalLM.lora_skip_prefixes = ["indexer"]


def patch_glm52_shared_indexers() -> None:
    """Avoid constructing unused IndexShare indexers on pinned vLLM 0.23."""
    from vllm.model_executor.models import deepseek_v2

    original = deepseek_v2.Indexer
    if getattr(original, "__art_glm52_index_share_patched__", False):
        return

    class Glm52Indexer(original):
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            config = args[1]
            prefix = args[7] if len(args) > 7 else kwargs.get("prefix", "")
            if getattr(config, "model_type", None) != "glm_moe_dsa":
                return super().__new__(cls)

            layer_idx = deepseek_v2.extract_layer_index(prefix)
            if layer_idx >= config.num_hidden_layers:
                return super().__new__(cls)
            mode = config.indexer_types[layer_idx]
            pattern = getattr(config, "index_topk_pattern", None)
            if pattern is None:
                freq = getattr(config, "index_topk_freq", 1)
                offset = getattr(config, "index_skip_topk_offset", 2)
                skip_topk = max(layer_idx - offset + 1, 0) % freq != 0
            else:
                skip_topk = pattern[layer_idx] == "S"
            if skip_topk != (mode == "shared"):
                raise RuntimeError(
                    f"GLM index schedules disagree at layer {layer_idx}: {mode=}"
                )
            return (
                _SharedTopkBuffer(args[6], config.index_topk)
                if skip_topk
                else super().__new__(cls)
            )

    Glm52Indexer.__art_glm52_index_share_patched__ = True
    deepseek_v2.Indexer = Glm52Indexer


def patch_glm52_indexer_rope() -> None:
    """Match GLM's half-split indexer RoPE without changing main MLA RoPE."""
    from vllm.model_executor.models import deepseek_v2

    original = deepseek_v2.DeepseekV2MLAAttention.__init__
    if getattr(original, "__art_glm52_indexer_rope_patched__", False):
        return

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        config = kwargs["config"] if "config" in kwargs else args[1]
        if getattr(config, "model_type", None) != "glm_moe_dsa":
            return
        rope = deepseek_v2.get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )
        self.indexer_rope_emb = rope
        self.mla_attn.indexer_rope_emb = rope
        indexer = self.mla_attn.indexer
        if hasattr(indexer, "is_inplace_rope"):
            indexer.is_inplace_rope = rope.enabled()

    init.__art_glm52_indexer_rope_patched__ = True  # type: ignore[attr-defined]
    deepseek_v2.DeepseekV2MLAAttention.__init__ = init
