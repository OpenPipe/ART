from pydantic import BaseModel, ConfigDict, Field


class RankDistributedLoraStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    source_bytes: int = Field(ge=0)
    sent_bytes: int = Field(ge=0)
    received_bytes: int = Field(ge=0)
    owned_tensor_bytes: int = Field(ge=0)
    peak_accounted_owner_bytes: int = Field(ge=0)
    owned_upload_bytes: int = Field(ge=0)
    owned_tensor_count: int = Field(ge=0)
    owned_block_count: int = Field(ge=0)
