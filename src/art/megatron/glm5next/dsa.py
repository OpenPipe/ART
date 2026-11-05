"""GLM-5.3 Flash no-RoPE MLA using ART's GLM-5.2 sparse forward/backward."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any

from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import get_pg_size
import torch
from torch.nn import functional as F

from art.megatron.glm5next.kpool_indexer import KPoolIndexer
from art.megatron.glm52.attention import Glm52SelfAttention, Glm52SparseCore
from art.megatron.glm52.state import Glm52PrefixTreeState

DSA_LORA_TARGETS = frozenset({"q_a_proj", "kv_a_proj_with_mqa", "o_proj"})


def apply_glm5next_dsa_lora(
    attention: Any,
    *,
    adapter_model_prefix: str,
    provider: Any,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    """M5 adapter hook: preserve GLM-5.2 slots/export, restrict Flash targets."""
    if not 1 <= rank <= 32:
        raise ValueError("Flash's Tinker adapter contract requires rank in [1,32].")
    # ART interprets an empty target set as all targets. Never pass an empty
    # intersection to GLM-5.2, where that would also install q_b LoRA.
    targets = set(
        DSA_LORA_TARGETS if not target_modules else DSA_LORA_TARGETS & target_modules
    )
    if not targets:
        return
    from art.megatron.glm52.lora import apply_glm52_attention_lora

    apply_glm52_attention_lora(
        attention,
        adapter_model_prefix=adapter_model_prefix,
        provider=provider,
        target_modules=targets,
        rank=rank,
        alpha=alpha,
    )


class Glm5NextSparseCore(Glm52SparseCore):
    def __init__(
        self,
        *,
        config: Any,
        layer_number: int,
        pg_collection: Any,
        linear_builder: Any,
        norm_builder: Any,
        **_: Any,
    ) -> None:
        torch.nn.Module.__init__(self)
        del layer_number
        if get_pg_size(pg_collection.tp) != 1 or config.sequence_parallel:
            raise ValueError(
                "Flash DSA currently supports TP1 without sequence parallelism."
            )
        if getattr(config, "dsa_indexer_loss_coeff", 0.0) != 0.0:
            raise ValueError(
                "Flash's frozen k-pool indexer requires dsa_indexer_loss_coeff=0."
            )
        pool_size = int(getattr(config, "dsa_indexer_kpool", 4))
        if pool_size != 4:
            raise ValueError("Flash DSA requires four-token k-pools.")
        norm_config = deepcopy(config)
        norm_config.normalization = "LayerNorm"

        def linear(n: int, m: int) -> Any:
            return build_module(
                linear_builder,
                n,
                m,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )

        self.indexer = KPoolIndexer(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.dsa_indexer_n_heads,
            head_dim=config.dsa_indexer_head_dim,
            topk=config.dsa_indexer_topk,
            pool_size=pool_size,
            linear_factory=linear,
            norm_factory=lambda n: build_module(
                norm_builder, config=norm_config, hidden_size=n, eps=1e-6
            ),
        )

    def topk(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        state: Glm52PrefixTreeState,
    ) -> Any:
        if not isinstance(self.indexer, KPoolIndexer):
            raise TypeError("Flash DSA requires KPoolIndexer.")
        return self.indexer(hidden_states.detach(), q_residual.detach(), state)


def glm5next_core_builder(linear_builder: Any, norm_builder: Any) -> Any:
    return partial(
        Glm5NextSparseCore, linear_builder=linear_builder, norm_builder=norm_builder
    )


class Glm5NextSelfAttention(Glm52SelfAttention):
    """Same MCore/LoRA interface as Glm52SelfAttention, with exact NoPE weights.

    q_a:4096->1536; q_b:1536->16384; kv_a:4096->512;
    kv_b:512->32768; o:16384->4096. Only the kernel input gets 64 zero
    dimensions (12.5% extra score arithmetic); the scale remains 256**-0.5.
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        if config.qk_pos_emb_head_dim != 0 or not 0 < config.kv_lora_rank <= 512:
            raise ValueError(
                "Flash DSA requires no RoPE and a KV latent rank in [1,512]."
            )
        super().__init__(config, *args, **kwargs)

    def _attention_state(self, attention_bias: Any) -> Glm52PrefixTreeState:
        from art.megatron.glm5next.dsa_cp import Glm5NextDsaState

        model_state = getattr(attention_bias, "model_state", None)
        state = (
            model_state.get("glm5next_dsa") if isinstance(model_state, dict) else None
        )
        if not isinstance(state, Glm5NextDsaState):
            raise RuntimeError(
                "Build glm5next_dsa state once per packed batch before Flash DSA."
            )
        return state

    def _normalize_latent(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(
            xf.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon
        )
        return weight * normalized.to(x.dtype)

    def _rotary_slices(
        self, q_rope: torch.Tensor, k_rope: torch.Tensor, state: Glm52PrefixTreeState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del state
        padding = 576 - self.config.kv_lora_rank
        return F.pad(q_rope, (0, padding)), F.pad(k_rope, (0, padding))

    def _latent_output(self, latent_out: torch.Tensor) -> torch.Tensor:
        # Small HF oracle fixtures keep their native latent rank.
        return latent_out[..., : self.config.kv_lora_rank]
