from pydantic import BaseModel, ConfigDict, Field


class TrainingMicrobatchWorkload(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_nonpadding_tokens: int = Field(ge=0)
    loss_bearing_tokens: int = Field(ge=0)
    executed_token_equivalents: int = Field(ge=0)
    nominal_schedule_capacity_tokens: int = Field(ge=0)


class TrainingStepWorkload(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_nonpadding_tokens: int = Field(ge=0)
    loss_bearing_tokens: int = Field(ge=0)
    executed_token_equivalents: int = Field(ge=0)
    nominal_schedule_capacity_tokens: int = Field(ge=0)
    dummy_executed_token_equivalents: int = Field(ge=0)
    dummy_schedule_capacity_tokens: int = Field(ge=0)
    real_microbatches: int = Field(ge=0)
    dummy_microbatches: int = Field(ge=0)
