from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.spec import CompileWorkaroundConfig, LayerFamilyInstance

_ORACLE_HIDDEN_SIZE = 512
_ORACLE_Q_LORA_RANK = 128
_ORACLE_NUM_ATTENTION_HEADS = 1
_ORACLE_NUM_EXPERTS = 2
_ORACLE_NUM_EXPERTS_PER_TOK = 1
_ORACLE_FFN_HIDDEN_SIZE = 128
_ORACLE_INDEX_HEADS = 1
_ORACLE_INDEX_TOPK = 1024
_VALIDATION_NUM_LAYERS_ENV = "ART_DSV4_VALIDATION_NUM_LAYERS"
_ORACLE_EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.experts\..*\.weight(?P<expert>\d+)$")
_DSV4_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_DSV4_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "te_triton_permute_with_mask_map",
)


class Dsv4Handler(DefaultMoeHandler):
    key = "dsv4"
    is_moe = True
    cp_supported = False
    native_vllm_lora_status = "disabled"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        self.ensure_hf_reference_registered()
        return base_config

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        from art.megatron.dsv4.spec import get_dsv4_decoder_block_spec

        provider.transformer_layer_spec = get_dsv4_decoder_block_spec
        if int(getattr(provider, "context_parallel_size", 1) or 1) != 1:
            raise RuntimeError(
                "DSV4 model support in this worktree does not implement context parallelism."
            )

    def configure_provider_for_runtime(self, provider: Any) -> None:
        provider.mtp_num_layers = None
        raw_num_layers = os.environ.get(_VALIDATION_NUM_LAYERS_ENV)
        if raw_num_layers is None:
            return
        num_layers = int(raw_num_layers)
        if num_layers < 1:
            raise ValueError(f"{_VALIDATION_NUM_LAYERS_ENV} must be positive")
        provider.num_layers = num_layers
        provider.moe_layer_freq = [1] * num_layers
        ratios = list(getattr(provider, "dsv4_compress_ratios", ()) or ())
        if ratios:
            if num_layers > len(ratios):
                raise ValueError(
                    f"{_VALIDATION_NUM_LAYERS_ENV}={num_layers} exceeds "
                    f"dsv4_compress_ratios length {len(ratios)}"
                )
            provider.dsv4_compress_ratios = ratios[:num_layers]

    def default_chat_template(self) -> str | None:
        return None

    def configure_tokenizer(
        self,
        tokenizer: Any,
        *,
        internal_config: Any,
    ) -> Any:
        from art.megatron.dsv4.tokenizer import (
            get_dsv4_tokenizer,
            has_configured_chat_template,
        )

        if has_configured_chat_template(internal_config):
            return tokenizer
        return get_dsv4_tokenizer(tokenizer)

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]:
        target_set = set(target_modules)

        def include(name: str) -> bool:
            if ".self_attn.compressor.indexer." in name:
                return False
            if "q_a_proj" in target_set and name.endswith(".self_attn.q_a_proj.weight"):
                return True
            if "q_b_proj" in target_set and name.endswith(".self_attn.q_b_proj.weight"):
                return True
            if "kv_proj" in target_set and name.endswith(".self_attn.kv_proj.weight"):
                return True
            if "o_a_proj" in target_set and name.endswith(".self_attn.o_a_proj.weight"):
                return True
            if "o_b_proj" in target_set and name.endswith(".self_attn.o_b_proj.weight"):
                return True
            if "compressor.kv_proj" in target_set and name.endswith(
                ".self_attn.compressor.kv_proj.weight"
            ):
                return True
            if "compressor.gate_proj" in target_set and name.endswith(
                ".self_attn.compressor.gate_proj.weight"
            ):
                return True
            if (
                "gate_proj" in target_set
                and ".mlp." in name
                and name.endswith(".gate_proj.weight")
            ):
                return True
            if (
                "up_proj" in target_set
                and ".mlp." in name
                and name.endswith(".up_proj.weight")
            ):
                return True
            if (
                "down_proj" in target_set
                and ".mlp." in name
                and name.endswith(".down_proj.weight")
            ):
                return True
            if "experts" in target_set and name.endswith(
                (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
            ):
                return True
            return False

        return [name for name, _ in model.named_parameters() if include(name)]

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from megatron.core.models.gpt.gpt_model import GPTModel

        from art.megatron.dsv4.deepseek_v4 import DeepSeekV4Attention
        from art.megatron.dsv4.layer import Dsv4MoELayer
        from art.megatron.dsv4.rope import materialize_rope_cache

        for chunk in list(model_chunks):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
            for child in gpt_module.modules():
                materialize_rope_cache(child)
            preprocess = gpt_module._preprocess

            def preprocess_hook(
                *args: Any, _preprocess=preprocess, _gpt=gpt_module, **kwargs: Any
            ):
                input_ids = kwargs.get("input_ids")
                position_ids = kwargs.get("position_ids")
                for child in _gpt.decoder.modules():
                    if isinstance(child, Dsv4MoELayer):
                        child.set_input_ids(
                            input_ids if isinstance(input_ids, torch.Tensor) else None
                        )
                    if isinstance(child, DeepSeekV4Attention):
                        child.set_position_ids(
                            position_ids
                            if isinstance(position_ids, torch.Tensor)
                            else None
                        )
                preproc_output = list(_preprocess(*args, **kwargs))
                decoder_input = cast(torch.Tensor, preproc_output[0])
                if not decoder_input.requires_grad and decoder_input.is_leaf:
                    decoder_input.requires_grad_(True)
                table = preproc_output[1]
                if isinstance(position_ids, torch.Tensor) and torch.is_tensor(table):
                    embedding_dim = int(table.shape[-1])
                    batch_size, sequence_length = position_ids.shape
                    gathered = table.view(table.shape[0], embedding_dim).index_select(
                        0, position_ids.reshape(-1)
                    )
                    preproc_output[1] = (
                        gathered.view(batch_size, sequence_length, embedding_dim)
                        .permute(1, 0, 2)
                        .contiguous()
                        .unsqueeze(2)
                    )
                return tuple(preproc_output)

            gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        ratios = list(getattr(provider, "dsv4_compress_ratios", ()) or ())

        def first_layer_index(ratio: int) -> int | None:
            try:
                return ratios.index(ratio)
            except ValueError:
                return None

        return [
            LayerFamilyInstance(
                key="dsv4_sliding_attention", layer_index=first_layer_index(0)
            ),
            LayerFamilyInstance(
                key="dsv4_csa_attention", layer_index=first_layer_index(4)
            ),
            LayerFamilyInstance(
                key="dsv4_hca_attention", layer_index=first_layer_index(128)
            ),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
            LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
        ]

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from art.megatron.dsv4.layer import Dsv4TransformerLayer
        from art.megatron.dsv4.lora import apply_dsv4_attention_lora
        from art.megatron.lora import (
            _adapter_model_prefix,
            wrap_grouped_moe_experts,
            wrap_shared_experts_mlp,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module in chunk.modules():
                if not isinstance(module, Dsv4TransformerLayer):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                apply_dsv4_attention_lora(
                    module.self_attention,
                    adapter_model_prefix=adapter_model_prefix,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                wrap_grouped_moe_experts(
                    _require_moe_experts(module),
                    adapter_model_prefix=adapter_model_prefix,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                if getattr(module.mlp, "shared_experts", None) is not None:
                    wrap_shared_experts_mlp(
                        module.mlp.shared_experts,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        from art.megatron.dsv4.layer import Dsv4TransformerLayer
        from art.megatron.dsv4.lora import (
            add_dsv4_attention_adapter_weights,
            add_dsv4_shared_experts_adapter_weights,
        )
        from art.megatron.weights.adapter_export import (
            add_grouped_moe_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, Dsv4TransformerLayer):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                add_dsv4_attention_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    attention=module.self_attention,
                )
                add_grouped_moe_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    experts=_require_moe_experts(module),
                )
                if getattr(module.mlp, "shared_experts", None) is not None:
                    add_dsv4_shared_experts_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        shared_experts=module.mlp.shared_experts,
                    )
        return adapter_weights_by_base

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _dsv4_from_vllm_lora_tensors(
            tensors,
            adapter_config=adapter_config,
        )

    def compile_workaround_config(self, provider: Any) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _DSV4_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state=self._shared_expert_compile_state(provider),
        )

    def ensure_hf_reference_registered(self) -> None:
        from art.megatron.dsv4.hf_config import ensure_dsv4_hf_model_registered

        ensure_dsv4_hf_model_registered()

    def prepare_hf_reference_config(self, config: Any) -> None:
        """Puts the HF parity oracle in eager training mode with reduced fit-only axes."""
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        config._experts_implementation = "eager"
        self._apply_oracle_shape_overrides(config)

    def hf_reference_from_pretrained_kwargs(
        self, *, config: Any, dtype: torch.dtype
    ) -> dict[str, Any]:
        del config, dtype
        return {
            "experts_implementation": "eager",
            "ignore_mismatched_sizes": True,
            "key_mapping": _dsv4_source_to_hf_key_mapping(),
        }

    def use_hf_reference_state_for_hf_parity(self) -> bool:
        """DSV4 parity seeds Megatron from the reduced canonical HF oracle state.

        The public checkpoint uses Miles/RadixArk source names and full model
        shapes, while the validation oracle uses canonical HF names and reduced
        fit-only axes. This hook is validation-only; production loading remains
        tied to the normal Bridge checkpoint source.
        """
        return True

    def configure_oracle_provider(self, provider: Any, *, case_config: Any) -> None:
        """Mirrors HF oracle reductions while keeping DSV4 hard kernel invariants."""
        hooks = list(getattr(provider, "_pre_wrap_hooks", []))
        kept = [hook for hook in hooks if not self._is_bridge_hf_load_hook(hook)]
        if len(kept) != len(hooks):
            provider._pre_wrap_hooks = kept
        provider.register_pre_wrap_hook(
            lambda chunks: self._initialize_oracle_base_weights(
                chunks,
                seed=int(case_config.seed),
            )
        )
        self._apply_oracle_shape_overrides(provider)
        provider.kv_lora_rank = 512
        provider.kv_channels = 512
        provider.qk_pos_emb_head_dim = 64
        provider.num_query_groups = 1
        provider.num_moe_experts = _ORACLE_NUM_EXPERTS
        provider.moe_ffn_hidden_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.ffn_hidden_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.moe_shared_expert_intermediate_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.moe_router_topk = _ORACLE_NUM_EXPERTS_PER_TOK
        provider.dsv4_o_groups = 1
        provider.dsv4_o_lora_rank = 1024
        provider.dsa_indexer_n_heads = _ORACLE_INDEX_HEADS
        provider.dsa_indexer_head_dim = 128
        provider.dsa_indexer_topk = _ORACLE_INDEX_TOPK
        provider.dsv4_oracle_freeze_attn_sink = True

    @staticmethod
    def _is_bridge_hf_load_hook(hook: Any) -> bool:
        fn = getattr(hook, "func", hook)
        name = getattr(fn, "__name__", "")
        qualname = getattr(fn, "__qualname__", "")
        return name in {
            "load_weights_hf_to_megatron",
            "_optimized_load_weights_hf_to_megatron",
        } or qualname.endswith(".load_weights_hf_to_megatron")

    def _apply_oracle_shape_overrides(self, config: Any) -> None:
        """Reduces memory-heavy axes only; head_dim/window/o-rank stay production-sized."""
        config.hidden_size = _ORACLE_HIDDEN_SIZE
        config.q_lora_rank = _ORACLE_Q_LORA_RANK
        config.num_attention_heads = _ORACLE_NUM_ATTENTION_HEADS
        config.n_routed_experts = _ORACLE_NUM_EXPERTS
        config.num_experts_per_tok = _ORACLE_NUM_EXPERTS_PER_TOK
        config.moe_intermediate_size = _ORACLE_FFN_HIDDEN_SIZE
        config.o_groups = 1
        config.index_n_heads = _ORACLE_INDEX_HEADS
        config.index_head_dim = 128
        config.index_topk = _ORACLE_INDEX_TOPK
        config.dsv4_oracle_freeze_attn_sink = True
        config.dsv4_oracle_source_aliases = True
        config.dsv4_oracle_precision_aligned_compressor = True

    def _initialize_oracle_base_weights(
        self,
        model_chunks: Sequence[Any],
        *,
        seed: int,
    ) -> Sequence[Any]:
        """Seeds reduced DSV4 oracle base tensors after meta-device materialization.

        DSV4 correctness runs strip the full-checkpoint Bridge load because the
        public tensors are production-sized and quantized. Since ART materializes
        meta models with empty storage, every base tensor must be explicitly
        initialized before freeze/LoRA hooks run; otherwise the oracle exercises a
        zero model and cannot produce meaningful adapter gradients.
        """
        from megatron.core import parallel_state as ps

        ep_rank = ps.get_expert_model_parallel_rank()
        ep_size = ps.get_expert_model_parallel_world_size()
        with torch.no_grad():
            for chunk in model_chunks:
                for name, param in chunk.named_parameters():
                    if self._is_oracle_lora_tensor(name):
                        continue
                    init_name = self._oracle_base_tensor_name(
                        name,
                        ep_rank=ep_rank,
                        ep_size=ep_size,
                    )
                    self._initialize_oracle_tensor(
                        init_name,
                        param,
                        seed=seed,
                    )
                for name, buffer in chunk.named_buffers():
                    self._initialize_oracle_buffer(name, buffer, seed=seed)
        return model_chunks

    @staticmethod
    def _is_oracle_lora_tensor(name: str) -> bool:
        return "_lora." in name or ".lora." in name

    @staticmethod
    def _oracle_base_tensor_name(name: str, *, ep_rank: int, ep_size: int) -> str:
        if ep_size <= 1:
            return name
        match = _ORACLE_EXPERT_WEIGHT_RE.search(name)
        if match is None:
            return name
        local_expert = int(match.group("expert"))
        local_expert_count = max(1, _ORACLE_NUM_EXPERTS // ep_size)
        global_expert = ep_rank * local_expert_count + local_expert
        return f"{name[: match.start('expert')]}{global_expert}"

    def _initialize_oracle_buffer(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        seed: int,
    ) -> None:
        if name.endswith("freqs_cis"):
            return
        if name.endswith("tid2eid"):
            self._initialize_oracle_tid2eid(tensor)
            return
        if not torch.is_floating_point(tensor):
            return
        self._initialize_oracle_tensor(name, tensor, seed=seed)

    @staticmethod
    def _initialize_oracle_tid2eid(tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise RuntimeError(
                f"Expected DSV4 tid2eid to be 2D, got {tuple(tensor.shape)}"
            )
        token_ids = torch.arange(tensor.shape[0], device=tensor.device).unsqueeze(1)
        offsets = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
        tensor.copy_(
            (token_ids + offsets).remainder(_ORACLE_NUM_EXPERTS).to(tensor.dtype)
        )

    def _initialize_oracle_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        seed: int,
    ) -> None:
        if tensor.is_meta:
            raise RuntimeError(f"DSV4 oracle tensor was not materialized: {name}")
        if not torch.is_floating_point(tensor):
            return
        if self._is_oracle_identity_weight(name):
            tensor.fill_(1)
            return
        if name.endswith(
            ("bias", "attn_sink", "_base", "_scale", "e_score_correction_bias")
        ):
            tensor.zero_()
            return
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
        key_seed = int.from_bytes(digest[:8], "little") % (2**31)
        generator = torch.Generator(device=tensor.device).manual_seed(key_seed)
        values = torch.randn(
            tensor.shape,
            generator=generator,
            device=tensor.device,
            dtype=torch.float32,
        )
        tensor.copy_((0.02 * values).to(dtype=tensor.dtype))

    @staticmethod
    def _is_oracle_identity_weight(name: str) -> bool:
        return name.endswith(".weight") and any(
            token in name for token in ("layernorm", "_norm", ".norm.")
        )


def ensure_dsv4_bridge_registered() -> None:
    from art.megatron.dsv4.bridge import ensure_dsv4_bridge_registered as _ensure

    _ensure()


def _ensure_dsv4_hf_config_registered() -> None:
    from art.megatron.dsv4.hf_config import ensure_dsv4_hf_config_registered

    ensure_dsv4_hf_config_registered()


_ensure_dsv4_hf_config_registered()
DSV4_HANDLER = Dsv4Handler()


def _dsv4_source_to_hf_key_mapping() -> dict[str, str]:
    layer = r"layers\.(\d+)"
    target = r"model.layers.\1"
    return {
        r"^embed\.weight$": "model.embed_tokens.weight",
        r"^head\.weight$": "lm_head.weight",
        r"^norm\.weight$": "model.norm.weight",
        r"^hc_head_fn$": "model.hc_head.hc_fn",
        r"^hc_head_base$": "model.hc_head.hc_base",
        r"^hc_head_scale$": "model.hc_head.hc_scale",
        rf"^{layer}\.attn_norm\.weight$": rf"{target}.input_layernorm.weight",
        rf"^{layer}\.ffn_norm\.weight$": rf"{target}.post_attention_layernorm.weight",
        rf"^{layer}\.hc_attn_fn$": rf"{target}.attn_hc.fn",
        rf"^{layer}\.hc_attn_base$": rf"{target}.attn_hc.base",
        rf"^{layer}\.hc_attn_scale$": rf"{target}.attn_hc.scale",
        rf"^{layer}\.hc_ffn_fn$": rf"{target}.ffn_hc.fn",
        rf"^{layer}\.hc_ffn_base$": rf"{target}.ffn_hc.base",
        rf"^{layer}\.hc_ffn_scale$": rf"{target}.ffn_hc.scale",
        rf"^{layer}\.attn\.wq_a\.weight$": rf"{target}.self_attn.q_a_proj.weight",
        rf"^{layer}\.attn\.q_norm\.weight$": rf"{target}.self_attn.q_a_norm.weight",
        rf"^{layer}\.attn\.wq_b\.weight$": rf"{target}.self_attn.q_b_proj.weight",
        rf"^{layer}\.attn\.wkv\.weight$": rf"{target}.self_attn.kv_proj.weight",
        rf"^{layer}\.attn\.kv_norm\.weight$": rf"{target}.self_attn.kv_norm.weight",
        rf"^{layer}\.attn\.wo_a\.weight$": rf"{target}.self_attn.o_a_proj.weight",
        rf"^{layer}\.attn\.wo_b\.weight$": rf"{target}.self_attn.o_b_proj.weight",
        rf"^{layer}\.attn\.attn_sink$": rf"{target}.self_attn.sinks",
        rf"^{layer}\.ffn\.gate\.weight$": rf"{target}.mlp.gate.weight",
        rf"^{layer}\.ffn\.gate\.tid2eid$": rf"{target}.mlp.gate.tid2eid",
        rf"^{layer}\.ffn\.gate\.bias$": (rf"{target}.mlp.gate.e_score_correction_bias"),
        rf"^{layer}\.ffn\.shared_experts\.w1\.weight$": (
            rf"{target}.mlp.shared_experts.gate_proj.weight"
        ),
        rf"^{layer}\.ffn\.shared_experts\.w3\.weight$": (
            rf"{target}.mlp.shared_experts.up_proj.weight"
        ),
        rf"^{layer}\.ffn\.shared_experts\.w2\.weight$": (
            rf"{target}.mlp.shared_experts.down_proj.weight"
        ),
        rf"^{layer}\.attn\.compressor\.ape$": (
            rf"{target}.self_attn.compressor.position_bias"
        ),
        rf"^{layer}\.attn\.compressor\.wkv\.weight$": (
            rf"{target}.self_attn.compressor.kv_proj.weight"
        ),
        rf"^{layer}\.attn\.compressor\.wgate\.weight$": (
            rf"{target}.self_attn.compressor.gate_proj.weight"
        ),
        rf"^{layer}\.attn\.compressor\.norm\.weight$": (
            rf"{target}.self_attn.compressor.kv_norm.weight"
        ),
        rf"^{layer}\.attn\.indexer\.wq_b\.weight$": (
            rf"{target}.self_attn.compressor.indexer.q_b_proj.weight"
        ),
        rf"^{layer}\.attn\.indexer\.weights_proj\.weight$": (
            rf"{target}.self_attn.compressor.indexer.weights_proj.weight"
        ),
        rf"^{layer}\.attn\.indexer\.compressor\.ape$": (
            rf"{target}.self_attn.compressor.indexer.position_bias"
        ),
        rf"^{layer}\.attn\.indexer\.compressor\.wkv\.weight$": (
            rf"{target}.self_attn.compressor.indexer.kv_proj.weight"
        ),
        rf"^{layer}\.attn\.indexer\.compressor\.wgate\.weight$": (
            rf"{target}.self_attn.compressor.indexer.gate_proj.weight"
        ),
        rf"^{layer}\.attn\.indexer\.compressor\.norm\.weight$": (
            rf"{target}.self_attn.compressor.indexer.kv_norm.weight"
        ),
    }


def _dsv4_unpack_vllm_3d_lora_b(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _dsv4_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().contiguous()


def _dsv4_from_vllm_lora_key(key: str) -> str:
    return key.replace(".mlp.shared_experts.", ".mlp.shared_expert.", 1)


def _dsv4_from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _DSV4_VLLM_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor
    if not grouped:
        return {
            _dsv4_from_vllm_lora_key(key): tensor for key, tensor in tensors.items()
        }

    rank = int(adapter_config["r"])
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, slots in grouped.items():
        try:
            gate_up_a = slots["base_layer.lora_A"]
            gate_up_b = slots["base_layer.lora_B"]
            down_a = slots["lora_A"]
            down_b = slots["lora_B"]
        except KeyError as exc:
            raise RuntimeError(
                f"Incomplete DSV4 vLLM MoE LoRA block for {prefix}"
            ) from exc
        if gate_up_a.shape[0] % rank != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_A rows {gate_up_a.shape[0]} are not "
                f"divisible by rank {rank}"
            )
        if gate_up_b.shape[0] % 2 != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_B rows {gate_up_b.shape[0]} are not even"
            )
        num_experts = gate_up_a.shape[0] // rank
        gate_up_b_by_expert = _dsv4_unpack_vllm_3d_lora_b(
            gate_up_b,
            num_experts=num_experts,
            rank=rank,
        )
        down_b_by_expert = _dsv4_unpack_vllm_3d_lora_b(
            down_b,
            num_experts=num_experts,
            rank=rank,
        )
        for expert in range(num_experts):
            row = expert * rank
            gate_up_a_block = gate_up_a[row : row + rank]
            down_a_block = down_a[row : row + rank]
            gate_b, up_b = gate_up_b_by_expert[expert].chunk(2, dim=0)
            transformed[f"{prefix}.{expert}.gate_proj.lora_A.weight"] = _dsv4_clone(
                gate_up_a_block
            )
            transformed[f"{prefix}.{expert}.gate_proj.lora_B.weight"] = _dsv4_clone(
                gate_b
            )
            transformed[f"{prefix}.{expert}.up_proj.lora_A.weight"] = _dsv4_clone(
                gate_up_a_block
            )
            transformed[f"{prefix}.{expert}.up_proj.lora_B.weight"] = _dsv4_clone(up_b)
            transformed[f"{prefix}.{expert}.down_proj.lora_A.weight"] = _dsv4_clone(
                down_a_block
            )
            transformed[f"{prefix}.{expert}.down_proj.lora_B.weight"] = _dsv4_clone(
                down_b_by_expert[expert]
            )
        used_keys.update(
            {
                f"{prefix}.base_layer.lora_A.weight",
                f"{prefix}.base_layer.lora_B.weight",
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            }
        )
    for key, tensor in tensors.items():
        if key not in used_keys:
            transformed[_dsv4_from_vllm_lora_key(key)] = tensor
    return transformed
