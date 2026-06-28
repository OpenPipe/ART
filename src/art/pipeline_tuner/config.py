from __future__ import annotations

from typing import Literal

import pydantic

PACKED_GROUP_PHYSICAL_TOKENS_KEY = "_art_packed_group_physical_tokens"
PACKED_GROUP_PROMPT_TOKENS_KEY = "_art_packed_group_prompt_tokens"
PACKED_GROUP_COMPLETION_TOKENS_KEY = "_art_packed_group_completion_tokens"


class PipelineRuntimeConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    num_rollout_workers: int = pydantic.Field(default=16, ge=1)
    min_batch_size: int = pydantic.Field(default=4, ge=1)
    max_batch_size: int | None = pydantic.Field(default=None, ge=1)
    max_steps_off_policy: int | None = pydantic.Field(default=4, ge=0)
    queue_maxsize: int | None = pydantic.Field(default=None, ge=1)
    score_reference_groups_per_step: float | None = pydantic.Field(default=None, gt=0.0)

    @pydantic.model_validator(mode="after")
    def validate_batch_bounds(self) -> "PipelineRuntimeConfig":
        if (
            self.max_batch_size is not None
            and self.max_batch_size < self.min_batch_size
        ):
            raise ValueError("max_batch_size must be >= min_batch_size")
        return self


class PipelineAutotuneConfig(pydantic.BaseModel):
    mode: Literal["off", "online", "profile"] = "off"
    profile: str | None = None
    output_name: str = "latest"
    window_steps: int = pydantic.Field(default=4, ge=1)
    warmup_ignore_steps: int = pydantic.Field(default=3, ge=0)
    target_spill_probability: float = pydantic.Field(default=0.03, ge=0.0, le=1.0)
    worker_step: int = pydantic.Field(default=4, ge=1)
    worker_move_fraction: float = pydantic.Field(default=0.10, gt=0.0, le=1.0)
    max_worker_move: int = pydantic.Field(default=16, ge=4)
    initial_model_calls_per_inference_gpu: int = pydantic.Field(default=8, ge=1)
    initial_min_batch_size: int = pydantic.Field(default=8, ge=1)
    initial_max_batch_size: int = pydantic.Field(default=8, ge=1)
    bootstrap_samples: int = pydantic.Field(default=256, ge=16)
    queue_running_reserve_fraction: float = pydantic.Field(default=0.75, ge=0.0, le=1.0)
    trainer_load_under_score: float = pydantic.Field(default=0.08, ge=0.0)
    trainer_load_severe_under_score: float = pydantic.Field(default=0.50, ge=0.0)
    trainer_load_over_score: float = pydantic.Field(default=0.04, ge=0.0)
    vllm_pressure_over_ratio: float = pydantic.Field(default=0.80, ge=0.0)
    vllm_pressure_under_ratio: float = pydantic.Field(default=0.50, ge=0.0)
    queue_put_high_frac: float = pydantic.Field(default=0.20, ge=0.0, le=1.0)
    queue_put_severe_frac: float = pydantic.Field(default=0.50, ge=0.0, le=1.0)
    stale_high_frac: float = pydantic.Field(default=0.20, ge=0.0, le=1.0)
    padding_high_frac: float = pydantic.Field(default=0.25, ge=0.0, le=1.0)
    recommendation_min_windows: int = pydantic.Field(default=5, ge=1)
    recommendation_consecutive_holds: int = pydantic.Field(default=2, ge=1)
    batch_underfill_frac: float = pydantic.Field(default=0.95, ge=0.0, le=1.0)
    policy_age_high_fraction: float = pydantic.Field(default=0.60, ge=0.0, le=1.0)
    policy_age_severe_fraction: float = pydantic.Field(default=1.0, ge=0.0, le=1.0)
    min_batch_floor_fraction: float = pydantic.Field(default=0.75, gt=0.0, le=1.0)
    freshness_min_batch_floor_fraction: float = pydantic.Field(
        default=0.50, gt=0.0, le=1.0
    )
    target_group_change_windows: int = pydantic.Field(default=1, ge=1)
    target_group_increase_fraction: float = pydantic.Field(default=0.25, gt=0.0, le=1.0)
    target_group_max_increase: int = pydantic.Field(default=64, ge=1)
    target_group_min_relative_change: float = pydantic.Field(
        default=0.10, ge=0.0, le=1.0
    )
    target_group_immediate_decrease_fraction: float = pydantic.Field(
        default=0.25, ge=0.0, le=1.0
    )
    vllm_metric_interval_s: float = pydantic.Field(default=1.0, gt=0.0)


class PipelineTuneSettings(pydantic.BaseModel):
    num_rollout_workers: int = pydantic.Field(ge=1)
    min_batch_size: int = pydantic.Field(ge=1)
    max_batch_size: int = pydantic.Field(ge=1)
    queue_maxsize: int = pydantic.Field(ge=1)
    target_groups_per_step: int = pydantic.Field(ge=1)


class PipelineMetric(pydantic.BaseModel):
    name: str
    value: float
    t_s: float
    step: int | None = None
    tags: dict[str, str] = pydantic.Field(default_factory=dict)


class PackedGroupObservation(pydantic.BaseModel):
    step: int
    physical_tokens: int = pydantic.Field(ge=1)
    prompt_tokens: int = pydantic.Field(ge=0)
    completion_tokens: int = pydantic.Field(ge=0)


class TunerWindowStats(pydantic.BaseModel):
    start_step: int
    end_step: int
    score_mean: float = 0.0
    accepted_tok_per_s_mean: float = 0.0
    trainer_idle_frac: float = 0.0
    trainer_load_score: float = 0.0
    vllm_capacity_wait_frac: float = 0.0
    vllm_active_frac: float = 0.0
    vllm_capacity_wait_request_s: float = 0.0
    vllm_running_request_s: float = 0.0
    vllm_pressure: float = 0.0
    vllm_capacity_wait_area: float = 0.0
    vllm_running_area: float = 0.0
    vllm_idle_frac: float = 0.0
    vllm_max_num_seqs_mean: float = 0.0
    queue_put_wait_frac: float = 0.0
    stale_frac: float = 0.0
    predicted_stale_frac: float = 0.0
    queue_freshness_pressure: float = 0.0
    token_weighted_policy_age_steps_mean: float = 0.0
    freshness_discount_mean: float = 1.0
    padding_ratio_mean: float = 0.0
    groups_per_step_mean: float = 0.0
    step_wall_s_mean: float = 0.0
    collect_batch_s_mean: float = 0.0
    train_work_s_mean: float = 0.0
    train_capacity_tokens_mean: float = 0.0
    group_pack_token_samples: list[float] = pydantic.Field(
        default_factory=list, exclude=True
    )


class TunerDecision(pydantic.BaseModel):
    step: int
    state: str
    action: str
    reason: str
    previous: PipelineTuneSettings
    updated: PipelineTuneSettings
    stats: TunerWindowStats | None = None
    recommendations: list[str] = pydantic.Field(default_factory=list)


class PipelineAutotunerProfile(pydantic.BaseModel):
    schema_version: int = 1
    model_name: str | None = None
    backend: str | None = None
    packed_sequence_length: int | None = None
    inference_gpu_count: int | None = None
    policy_age_limit_steps: float | None = None
    settings: PipelineTuneSettings
    config: PipelineAutotuneConfig
    decisions: list[TunerDecision] = pydantic.Field(default_factory=list)
    notes: list[str] = pydantic.Field(default_factory=list)
