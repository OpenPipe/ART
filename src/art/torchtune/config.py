from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComponentConfig(BaseModel):
    """Base config for components that can be instantiated via config.instantiate"""

    component: str = Field(
        ..., alias="_component_", description="Path to the component class"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for component-specific params
        populate_by_name=True,  # Allow using either field name or alias
    )


class ModelConfig(ComponentConfig):
    """Configuration for model instantiation"""

    pass  # Additional fields handled by extra="allow"


class OptimizerConfig(ComponentConfig):
    """Configuration for optimizer instantiation"""

    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class MetricLoggerConfig(ComponentConfig):
    """Configuration for metric logger instantiation"""

    pass  # Additional fields handled by extra="allow"


class CheckpointerConfig(BaseModel):
    """Configuration for checkpointer"""

    model_type: str = Field(..., description="Model type for checkpointing")

    model_config = ConfigDict(extra="allow")


class CompileConfig(BaseModel):
    """Configuration for torch.compile settings"""

    model: bool = Field(default=True)
    loss: bool = Field(default=True)
    optimizer_step: bool = Field(default=False)
    scale_grads: bool = Field(default=True)


class ProfilerConfig(ComponentConfig):
    """Configuration for profiler"""

    enabled: bool = Field(default=False)
    profile_memory: bool = Field(default=False)
    wait_steps: int = Field(default=0)
    warmup_steps: int = Field(default=1)
    active_steps: int = Field(default=1)

    @field_validator("component", mode="before")
    @classmethod
    def set_default_component(cls, v):
        # v can be None when profiler is not configured
        return v if v is not None else "torchtune.training.setup_torch_profiler"


class TensorParallelPlanConfig(ComponentConfig):
    """Configuration for tensor parallel plan"""

    pass  # Additional fields handled by extra="allow"


class RecipeConfig(BaseModel):
    """Main configuration for FullFinetuneRecipeDistributed"""

    # Core components
    model: ModelConfig
    optimizer: OptimizerConfig
    metric_logger: MetricLoggerConfig
    checkpointer: CheckpointerConfig

    # Device and precision settings
    device: str = Field(default="cuda")
    dtype: Literal["fp32", "bf16"] = Field(default="bf16")

    # Training parameters
    seed: Optional[int] = Field(default=None)
    epochs: int = Field(default=1, gt=0)
    max_steps_per_epoch: Optional[int] = Field(default=None, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)

    # Output and logging
    output_dir: str = Field(default="./outputs")
    log_level: str = Field(default="INFO")
    log_every_n_steps: int = Field(default=1, gt=0)
    log_peak_memory_stats: bool = Field(default=False)

    # Checkpointing and resumption
    resume_from_checkpoint: bool = Field(default=False)
    enable_async_checkpointing: bool = Field(default=False)

    # Distributed training settings
    fsdp_cpu_offload: bool = Field(default=False)
    fsdp_reshard_after_forward: bool = Field(default=True)
    tensor_parallel_dim: int = Field(default=1, ge=1)
    tensor_parallel_plan: Optional[TensorParallelPlanConfig] = Field(default=None)
    context_parallel_dim: int = Field(default=1, ge=1)
    data_parallel_shard_dim: int = Field(default=-1)
    data_parallel_replicate_dim: int = Field(default=1, ge=1)

    # Optimization settings
    optimizer_in_bwd: bool = Field(default=False)
    clip_grad_norm: Optional[float] = Field(default=None, gt=0)

    # Activation checkpointing
    enable_activation_checkpointing: bool = Field(default=False)
    enable_activation_offloading: bool = Field(default=False)
    activation_offloading_use_streams: bool = Field(default=True)
    ac_mode: Optional[str] = Field(default=None)
    ac_option: Optional[int] = Field(default=None)

    # Float8 training
    enable_fp8_training: bool = Field(default=False)
    fp8_recipe_name: Optional[str] = Field(default=None)

    # Compilation settings
    compile: Union[bool, CompileConfig] = Field(default=False)

    # Optional components
    profiler: Optional[ProfilerConfig] = Field(default=None)

    # Additional settings
    custom_sharded_layers: Optional[List[str]] = Field(default=None)
    cudnn_deterministic_mode: Optional[bool] = Field(default=None)
