from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Any, Callable, Iterator, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    LayerFamilyInstance,
    PrefixTreeModelStateContext,
)

_DSV4_ART_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_up_proj|down_proj)\."
    r"(?P<lora>lora_[AB])\.weight$"
)
_DSV4_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_DSV4_SPLIT_MOE_EXPERT_KEY_RE = re.compile(
    r"^.*(?:\.mlp\.experts\.\d+\.(?:gate_proj|up_proj)|"
    r"\.ffn\.experts\.\d+\.w[123])\.lora_[AB]\.weight$"
)
_DSV4_MOE_COMPILE_WORKAROUND_FLAGS = ("te_triton_permute_with_mask_map",)


def _dsv4_input_activator(
    model: Any,
) -> Callable[[torch.Tensor | None, torch.Tensor | None], None]:
    from art.megatron.dsv4.deepseek_v4 import DeepSeekV4Attention
    from art.megatron.dsv4.layer import Dsv4MoELayer

    modules = tuple(model.modules())
    input_setters = tuple(
        child.set_input_ids for child in modules if isinstance(child, Dsv4MoELayer)
    )
    position_setters = tuple(
        child.set_position_ids
        for child in modules
        if isinstance(child, DeepSeekV4Attention)
    )

    def activate(
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        for setter in input_setters:
            setter(input_ids)
        for setter in position_setters:
            setter(position_ids)

    return activate


class Dsv4Handler(DefaultMoeHandler):
    key = "dsv4"
    is_moe = True
    cp_supported = False
    native_vllm_lora_status = "validated"

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
        provider.moe_shared_expert_overlap = False
        provider.art_pipeline_activation_multiplier = provider.dsv4_hc_mult

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

    def build_prefix_tree_model_state(
        self,
        context: PrefixTreeModelStateContext,
    ) -> dict[str, Any]:
        if context.input_pos is None:
            raise RuntimeError(
                "DSV4 prefix-tree compression layouts require input_pos."
            )
        from art.megatron.dsv4.compressor import (
            Dsv4PrefixTreeState,
            build_prefix_tree_compression_layouts,
        )

        return {
            "dsv4": Dsv4PrefixTreeState(
                compression_layouts=build_prefix_tree_compression_layouts(
                    position_ids=context.input_pos,
                    group_ids=context.group_ids,
                    parent_ids=context.parent_ids,
                    device=context.device,
                )
            )
        }

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

        for chunk in list(model_chunks):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
            preprocess = gpt_module._preprocess
            activate = _dsv4_input_activator(gpt_module.decoder)

            def preprocess_hook(
                *args: Any,
                _preprocess=preprocess,
                _activate=activate,
                **kwargs: Any,
            ):
                input_ids = kwargs.get("input_ids")
                position_ids = kwargs.get("position_ids")
                _activate(
                    input_ids if isinstance(input_ids, torch.Tensor) else None,
                    position_ids if isinstance(position_ids, torch.Tensor) else None,
                )
                preproc_output = list(_preprocess(*args, **kwargs))
                decoder_input = cast(torch.Tensor | None, preproc_output[0])
                if (
                    decoder_input is not None
                    and not decoder_input.requires_grad
                    and decoder_input.is_leaf
                ):
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

            setattr(gpt_module, "_preprocess", preprocess_hook)

    def build_pipeline_microbatch_activator(
        self,
        model_chunks: Sequence[Any],
    ) -> Callable[[Any, int], None]:
        activators = tuple(_dsv4_input_activator(chunk) for chunk in model_chunks)

        def activate(prepared: Any, chunk_index: int) -> None:
            input_ids = getattr(prepared, "model_tokens", None)
            position_ids = getattr(prepared, "model_input_pos", None)
            if input_ids is None:
                input_ids = prepared.input_ids
                position_ids = prepared.position_ids
            activators[chunk_index](input_ids, position_ids)

        return activate

    @contextmanager
    def preserve_pipeline_microbatch_activation(
        self,
        model_chunks: Sequence[Any],
    ) -> Iterator[None]:
        states = [
            (module, name, getattr(module, name))
            for chunk in model_chunks
            for module in chunk.modules()
            for name in ("_dsv4_input_ids", "_dsv4_position_ids")
            if hasattr(module, name)
        ]
        try:
            yield
        finally:
            for module, name, value in states:
                setattr(module, name, value)

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        ratios: list[int] = list(getattr(provider, "dsv4_compress_ratios", ()) or ())

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
        from art.megatron.dsv4.lora import (
            apply_dsv4_attention_lora,
            disable_dsv4_etp_shared_expert_lora_compile,
            install_dsv4_te_permutation_static_configs,
        )
        from art.megatron.lora import (
            _adapter_model_prefix,
            wrap_grouped_moe_experts_3d,
            wrap_shared_experts_mlp,
        )

        target_set = set(target_modules)
        etp_enabled = int(getattr(provider, "expert_tensor_parallel_size", 1) or 1) > 1
        if etp_enabled:
            install_dsv4_te_permutation_static_configs()
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
                wrap_grouped_moe_experts_3d(
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
                    if etp_enabled:
                        disable_dsv4_etp_shared_expert_lora_compile(
                            module.mlp.shared_experts
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

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _dsv4_to_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def to_vllm_lora_config(self, adapter_config: dict[str, Any]) -> dict[str, Any]:
        """Translate ART training targets only for restrictive vLLM launches.

        A vLLM-format DSV4 adapter can be loaded by a vLLM server whose
        ``target_modules`` filter is unset. ART-managed vLLM launches set that
        filter for performance/memory control, so the filter must use
        vLLM/Miles module names rather than ART/Megatron training target names.
        """
        return _dsv4_vllm_lora_config(adapter_config)

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]:
        return (
            ExpertPackedLoraGroup(
                art_group_suffix=".mlp.experts",
                slots=(
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_A",
                        output_suffix="base_layer.lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_B",
                        output_suffix="base_layer.lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_A",
                        output_suffix="lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_B",
                        output_suffix="lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                ),
            ),
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

def ensure_dsv4_bridge_registered() -> None:
    from art.megatron.dsv4.bridge import ensure_dsv4_bridge_registered as _ensure

    _ensure()


def _ensure_dsv4_hf_config_registered() -> None:
    from art.megatron.dsv4.hf_config import ensure_dsv4_hf_config_registered

    ensure_dsv4_hf_config_registered()


def _sanitize_dsv4_child_process_env() -> None:
    from art.megatron.dsv4.kernel.tilelang_import import sanitize_tilelang_env

    sanitize_tilelang_env()


_sanitize_dsv4_child_process_env()
_ensure_dsv4_hf_config_registered()
DSV4_HANDLER = Dsv4Handler()


def _dsv4_unpack_vllm_3d_lora_b(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _dsv4_pack_vllm_3d_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _dsv4_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().contiguous()


def _dsv4_to_vllm_lora_key(key: str) -> str:
    replacements = (
        (".self_attn.compressor.kv_proj.", ".attn.compressor.wkv."),
        (".self_attn.compressor.gate_proj.", ".attn.compressor.wgate."),
        (".self_attn.q_a_proj.", ".attn.wq_a."),
        (".self_attn.q_b_proj.", ".attn.wq_b."),
        (".self_attn.kv_proj.", ".attn.wkv."),
        (".self_attn.o_a_proj.", ".attn.wo_a."),
        (".self_attn.o_b_proj.", ".attn.wo_b."),
        (".mlp.experts", ".ffn.experts"),
        (".mlp.shared_expert.", ".ffn.shared_experts."),
        (".mlp.shared_experts.", ".ffn.shared_experts."),
    )
    for old, new in replacements:
        if old in key:
            return key.replace(old, new, 1)
    return key


def _dsv4_from_vllm_lora_key(key: str) -> str:
    replacements = (
        (".attn.compressor.wkv.", ".self_attn.compressor.kv_proj."),
        (".attn.compressor.wgate.", ".self_attn.compressor.gate_proj."),
        (".attn.wq_a.", ".self_attn.q_a_proj."),
        (".attn.wq_b.", ".self_attn.q_b_proj."),
        (".attn.wkv.", ".self_attn.kv_proj."),
        (".attn.wo_a.", ".self_attn.o_a_proj."),
        (".attn.wo_b.", ".self_attn.o_b_proj."),
        (".ffn.experts", ".mlp.experts"),
        (".ffn.shared_experts.", ".mlp.shared_expert."),
        (".mlp.shared_experts.", ".mlp.shared_expert."),
    )
    for old, new in replacements:
        if old in key:
            return key.replace(old, new, 1)
    return key


def _dsv4_vllm_lora_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, (list, tuple, set)):
        return adapter_config
    transformed: list[str] = []
    ordered_target_modules = (
        sorted(target_modules) if isinstance(target_modules, set) else target_modules
    )
    for module in ordered_target_modules:
        if module in {"q_a_proj", "kv_proj"}:
            transformed.append("fused_wqa_wkv")
        elif module == "q_b_proj":
            transformed.append("wq_b")
        elif module == "o_a_proj":
            transformed.append("wo_a")
        elif module == "o_b_proj":
            transformed.append("wo_b")
        elif module in {"compressor.kv_proj", "compressor.gate_proj"}:
            transformed.append("fused_wkv_wgate")
        elif module in {"gate_proj", "up_proj"}:
            transformed.extend(("gate_up_proj", "experts"))
        elif module == "down_proj":
            transformed.extend(("down_proj", "experts"))
        elif module == "experts":
            transformed.append("experts")
        else:
            transformed.append(module)
    config = dict(adapter_config)
    config["target_modules"] = list(dict.fromkeys(transformed))
    return config


def _dsv4_to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    canonical = _dsv4_from_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
        split_experts=False,
    )
    fused_prefixes: set[str] = set()
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in canonical.items():
        fused_match = _DSV4_VLLM_MOE_KEY_RE.match(key)
        if fused_match is not None:
            fused_prefixes.add(fused_match.group("prefix"))
            continue
        match = _DSV4_ART_MOE_EXPERT_KEY_RE.match(key)
        if match is not None:
            grouped.setdefault(match.group("prefix"), {}).setdefault(
                int(match.group("expert")), {}
            ).setdefault(match.group("module"), {})[match.group("lora")] = tensor

    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    mixed_prefixes = fused_prefixes.intersection(grouped)
    if mixed_prefixes:
        raise RuntimeError(
            f"Mixed fused and split DSV4 MoE LoRA block for {min(mixed_prefixes)}"
        )

    for prefix, experts in grouped.items():
        vllm_prefix = _dsv4_to_vllm_lora_key(prefix)
        blocks = {
            slot: []
            for slot in (
                ("gate_up_proj", "lora_A"),
                ("gate_up_proj", "lora_B"),
                ("down_proj", "lora_A"),
                ("down_proj", "lora_B"),
            )
        }
        for expert in sorted(experts):
            modules = experts[expert]
            try:
                expert_tensors = {slot: modules[slot[0]][slot[1]] for slot in blocks}
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete DSV4 MoE LoRA block for {prefix}.{expert}"
                ) from exc
            for slot, tensor in expert_tensors.items():
                blocks[slot].append(tensor.contiguous())
                used_keys.add(f"{prefix}.{expert}.{slot[0]}.{slot[1]}.weight")
        transformed[f"{vllm_prefix}.base_layer.lora_A.weight"] = torch.cat(
            blocks[("gate_up_proj", "lora_A")], dim=0
        ).contiguous()
        transformed[f"{vllm_prefix}.base_layer.lora_B.weight"] = (
            _dsv4_pack_vllm_3d_lora_b(blocks[("gate_up_proj", "lora_B")])
        )
        transformed[f"{vllm_prefix}.lora_A.weight"] = torch.cat(
            blocks[("down_proj", "lora_A")], dim=0
        ).contiguous()
        transformed[f"{vllm_prefix}.lora_B.weight"] = _dsv4_pack_vllm_3d_lora_b(
            blocks[("down_proj", "lora_B")]
        )

    for key, tensor in canonical.items():
        if key in used_keys:
            continue
        vllm_key = _dsv4_to_vllm_lora_key(key)
        if vllm_key in transformed:
            raise RuntimeError(
                f"Duplicate DSV4 LoRA tensor after conversion: {vllm_key}"
            )
        transformed[vllm_key] = tensor
    return transformed, _dsv4_vllm_lora_config(adapter_config)


def _dsv4_from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    split_experts: bool = True,
) -> dict[str, torch.Tensor]:
    split_key = next(
        (key for key in tensors if _DSV4_SPLIT_MOE_EXPERT_KEY_RE.match(key)), None
    )
    if split_key is not None:
        raise RuntimeError(
            "DSV4 only supports fused 3D MoE LoRA tensors; got split expert "
            f"tensor {split_key}"
        )
    canonical = {
        _dsv4_from_vllm_lora_key(key): tensor for key, tensor in tensors.items()
    }
    if len(canonical) != len(tensors):
        raise RuntimeError("Duplicate DSV4 LoRA tensor after key canonicalization")
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in canonical.items():
        match = _DSV4_VLLM_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor
    if not grouped:
        return canonical

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
        non_2d = next(
            (slot for slot, tensor in slots.items() if tensor.ndim != 2), None
        )
        if non_2d is not None:
            raise RuntimeError(
                f"{prefix}: {non_2d} must be 2D, got {tuple(slots[non_2d].shape)}"
            )
        if rank <= 0 or gate_up_a.shape[0] == 0 or gate_up_a.shape[0] % rank != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_A rows {gate_up_a.shape[0]} are not "
                f"divisible by rank {rank}"
            )
        num_experts = gate_up_a.shape[0] // rank
        expected = {
            "gate/up lora_B": (
                tuple(gate_up_b.shape),
                (2 * down_a.shape[1], gate_up_a.shape[0]),
            ),
            "down lora_A": (down_a.shape[0], gate_up_a.shape[0]),
            "down lora_B": (
                tuple(down_b.shape),
                (gate_up_a.shape[1], gate_up_a.shape[0]),
            ),
        }
        for slot, (actual, wanted) in expected.items():
            if actual != wanted:
                raise RuntimeError(
                    f"{prefix}: {slot} shape {actual} does not match {wanted}"
                )
        if not split_experts:
            continue
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
            transformed[f"{prefix}.{expert}.gate_up_proj.lora_A.weight"] = _dsv4_clone(
                gate_up_a_block
            )
            transformed[f"{prefix}.{expert}.gate_up_proj.lora_B.weight"] = _dsv4_clone(
                gate_up_b_by_expert[expert]
            )
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
    for key, tensor in canonical.items():
        if key not in used_keys:
            transformed[key] = tensor
    return transformed if split_experts else canonical
