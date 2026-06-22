from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

RolloutWeightsMode = Literal["lora", "merged"]
NativeVllmLoraStatus = Literal["disabled", "wip", "validated"]
SharedExpertCompileState = Literal[
    "none",
    "shared_experts",
    "shared_expert_overlap",
]
ExpertPackedLoraLayout = Literal["expert_rows", "rank_major_expert_cols"]


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


class CompileWorkaroundConfig(BaseModel):
    flags: tuple[str, ...] = ()
    unconditional_flags: tuple[str, ...] = ()
    shared_expert_state: SharedExpertCompileState = "none"
    disable_compile: bool = False


class ExpertPackedLoraSlot(BaseModel):
    source_projection: str
    source_lora: Literal["lora_A", "lora_B"]
    output_suffix: str
    pack_layout: ExpertPackedLoraLayout


class ExpertPackedLoraGroup(BaseModel):
    art_group_suffix: str
    slots: tuple[ExpertPackedLoraSlot, ...]


class ModelSupportSpec(BaseModel):
    key: str
    handler_key: str
    is_moe: bool = False
    model_names: tuple[str, ...] = ()
    default_target_modules: tuple[str, ...]
    default_rollout_weights_mode: RolloutWeightsMode = "lora"
    default_vllm_max_loras: int = 2
    native_vllm_lora_status: NativeVllmLoraStatus = "disabled"
    dependency_floor: DependencyFloor = Field(default_factory=DependencyFloor)


class ModelSupportHandler(Protocol):
    key: str
    is_moe: bool
    cp_supported: bool
    native_vllm_lora_status: NativeVllmLoraStatus

    def identity_lora_model_config(self, base_config: Any) -> Any: ...

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]: ...

    def patch_bridge(self, bridge: Any) -> None: ...

    def patch_provider(self, provider: Any, bridge: Any) -> None: ...

    def configure_provider_for_runtime(self, provider: Any) -> None: ...

    def default_chat_template(self) -> str | None: ...

    def configure_tokenizer(
        self,
        tokenizer: Any,
        *,
        internal_config: Any,
    ) -> Any: ...

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None: ...

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]: ...

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
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
        provider: Any,
    ) -> CompileWorkaroundConfig: ...

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]: ...
