from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

RolloutWeightsMode = Literal["lora", "merged"]
NativeVllmLoraStatus = Literal["disabled", "wip", "validated"]


class HandlerMeta(BaseModel):
    """Single source of truth for a registered handler's key, status, and
    lazy-load address. Kept megatron-free so the registry and spec layer can
    be imported without `megatron-core` installed."""

    model_config = ConfigDict(frozen=True)

    key: str
    native_vllm_lora_status: NativeVllmLoraStatus
    module: str
    attr: str


DEFAULT_DENSE_META = HandlerMeta(
    key="default_dense",
    native_vllm_lora_status="disabled",
    module="art.megatron.model_support.handlers.default_dense",
    attr="DEFAULT_DENSE_HANDLER",
)

QWEN3_DENSE_META = HandlerMeta(
    key="qwen3_dense",
    native_vllm_lora_status="validated",
    module="art.megatron.model_support.handlers.qwen3_dense",
    attr="QWEN3_DENSE_HANDLER",
)

QWEN3_MOE_META = HandlerMeta(
    key="qwen3_moe",
    native_vllm_lora_status="validated",
    module="art.megatron.model_support.handlers.qwen3_moe",
    attr="QWEN3_MOE_HANDLER",
)

QWEN3_5_DENSE_META = HandlerMeta(
    key="qwen3_5_dense",
    native_vllm_lora_status="validated",
    module="art.megatron.model_support.handlers.qwen3_5",
    attr="QWEN3_5_DENSE_HANDLER",
)

QWEN3_5_MOE_META = HandlerMeta(
    key="qwen3_5_moe",
    native_vllm_lora_status="validated",
    module="art.megatron.model_support.handlers.qwen3_5",
    attr="QWEN3_5_MOE_HANDLER",
)

ALL_HANDLER_METAS: tuple[HandlerMeta, ...] = (
    DEFAULT_DENSE_META,
    QWEN3_DENSE_META,
    QWEN3_MOE_META,
    QWEN3_5_DENSE_META,
    QWEN3_5_MOE_META,
)
SharedExpertCompileState = Literal[
    "none",
    "shared_experts",
    "shared_expert_overlap",
]


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


class MinimalLayerCoverageReport(BaseModel):
    base_model: str
    model_key: str
    requested_num_layers: int
    recommended_min_layers: int
    covered: bool
    missing_layer_families: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)


class ValidationStageResult(BaseModel):
    name: str
    passed: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str | None = None


class ValidationReport(BaseModel):
    base_model: str
    model_key: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    stages: list[ValidationStageResult] = Field(default_factory=list)


class CompileWorkaroundConfig(BaseModel):
    flags: tuple[str, ...] = ()
    shared_expert_state: SharedExpertCompileState = "none"
    disable_compile: bool = False


class ModelSupportSpec(BaseModel):
    key: str
    handler_key: str
    model_names: tuple[str, ...] = ()
    default_target_modules: tuple[str, ...]
    default_rollout_weights_mode: RolloutWeightsMode = "lora"
    native_vllm_lora_status: NativeVllmLoraStatus = "disabled"
    dependency_floor: DependencyFloor = Field(default_factory=DependencyFloor)


class ModelSupportHandler(Protocol):
    key: str
    is_moe: bool
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

    def hf_tensor_map_to_art_canonical(
        self,
        hf_tensor_map: dict[str, Any],
        *,
        expected_keys: set[str],
    ) -> dict[str, Any]:
        """
        Testing-only hook for canonicalizing raw HuggingFace tensor maps into the
        ART tensor-map keyspace expected by model-support probes.

        This currently exists to support validations such as HF parity, where the
        raw HF model can expose fused parameter names or layouts that differ from
        the canonical names ART compares against.
        """
        ...

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, Any],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

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
