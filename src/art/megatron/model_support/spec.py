from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.gpt_provider import GPTModelProvider

RolloutWeightsMode = Literal["lora", "merged"]
NativeVllmLoraStatus = Literal["disabled", "wip", "validated"]
SharedExpertCompileState = Literal[
    "none",
    "shared_experts",
    "shared_expert_overlap",
]
ExpertPackedLoraLayout = Literal[
    "expert_rows",
    "rank_major_expert_cols",
    "interleaved_gate_up_rank_major_expert_cols",
]
HfWeightSourceKind = Literal["direct", "bridge_materialized"]


class DependencyFloor(BaseModel):
    transformers: str | None = None
    vllm: str | None = None
    megatron_bridge: str | None = None


class LayerFamilyInstance(BaseModel):
    key: str
    count: int = 1
    layer_index: int | None = None
    module_path: str | None = None
    module_type: str | None = None


class ArchitectureReport(BaseModel):
    base_model: str
    model_key: str
    handler_key: str
    bridge_type: str | None = None
    provider_type: str | None = None
    layer_families: list[LayerFamilyInstance] = Field(default_factory=list)
    recommended_min_layers: int = 1
    unresolved_risks: list[str] = Field(default_factory=list)


class PrefixTreeModelStateContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_ids: Any
    parent_ids: Any
    input_pos: Any | None = None
    device: Any
    attention_token_layout_index: Any | None = None
    attention_head_dim: int | None = None
    attention_value_head_dim: int | None = None


class CompileWorkaroundConfig(BaseModel):
    flags: tuple[str, ...] = ()
    unconditional_flags: tuple[str, ...] = ()
    shared_expert_state: SharedExpertCompileState = "none"
    disable_compile: bool = False


class FlexAttentionCompileCrashConfig(BaseModel):
    # Fatal compile workarounds only. Do not add entries for autotuning noise or
    # performance tuning; entries require Inductor to raise after config search.
    triton_num_stages_2_head_dims: tuple[int, ...] = ()


class ExpertPackedLoraSlot(BaseModel):
    source_projection: str
    source_lora: Literal["lora_A", "lora_B"]
    output_suffix: str
    pack_layout: ExpertPackedLoraLayout


class ExpertPackedLoraGroup(BaseModel):
    art_group_suffix: str
    slots: tuple[ExpertPackedLoraSlot, ...]


class HfWeightSource(BaseModel):
    logical_key: str
    physical_key_options: tuple[tuple[str, ...], ...]
    kind: HfWeightSourceKind = "direct"


class ModelSupportSpec(BaseModel):
    key: str
    handler_key: str
    is_moe: bool = False
    model_names: tuple[str, ...] = ()
    default_target_modules: tuple[str, ...]
    default_rollout_weights_mode: RolloutWeightsMode = "lora"
    native_vllm_lora_status: NativeVllmLoraStatus = "disabled"
    dependency_floor: DependencyFloor = Field(default_factory=DependencyFloor)


@runtime_checkable
class ModelSupportHandler(Protocol):
    key: str
    is_moe: bool
    build_gdn_execution_spec: bool
    cp_supported: bool
    native_vllm_lora_status: NativeVllmLoraStatus

    def identity_lora_model_config(self, base_config: Any) -> Any: ...

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]: ...

    def patch_bridge(self, bridge: "AutoBridge") -> None: ...

    def hf_weight_source(
        self,
        bridge: "AutoBridge",
        hf_param: str,
        *,
        task: Any | None = None,
    ) -> HfWeightSource | None: ...

    def patch_provider(
        self,
        provider: "GPTModelProvider",
        bridge: "AutoBridge",
    ) -> None: ...

    def configure_provider_for_runtime(self, provider: "GPTModelProvider") -> None: ...

    def default_chat_template(self) -> str | None: ...

    def configure_tokenizer(
        self,
        tokenizer: Any,
        *,
        internal_config: Any,
    ) -> Any: ...

    def vllm_engine_args(
        self,
        *,
        rollout_weights_mode: RolloutWeightsMode,
    ) -> dict[str, object]: ...

    def vllm_server_args(self) -> dict[str, object]: ...

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None: ...

    def build_prefix_tree_model_state(
        self,
        context: PrefixTreeModelStateContext,
    ) -> dict[str, Any]: ...

    def correctness_precision(self) -> Literal["bf16", "fp32"]: ...

    def correctness_use_fp32_lora_reference(self) -> bool: ...

    def correctness_phase_pass_fns(
        self, oracle_harness: Any
    ) -> dict[str, Any] | None: ...

    def collect_layer_families(
        self,
        provider: "GPTModelProvider",
    ) -> list[LayerFamilyInstance]: ...

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: "GPTModelProvider",
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None: ...

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]: ...

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, Any],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def to_vllm_lora_config(
        self,
        adapter_config: dict[str, Any],
    ) -> dict[str, Any]: ...

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]: ...

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, Any],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, Any]: ...

    def compile_workaround_config(
        self,
        provider: "GPTModelProvider",
    ) -> CompileWorkaroundConfig: ...

    def flex_attention_compile_crash_config(
        self,
        provider: "GPTModelProvider",
    ) -> FlexAttentionCompileCrashConfig: ...

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]: ...
