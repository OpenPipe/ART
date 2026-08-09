from pydantic import BaseModel, ConfigDict


class MergedWeightTransferInitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


class MergedWeightTransferSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    init_info: MergedWeightTransferInitInfo
    vllm_base_url: str
    served_model_name: str
    api_key: str | None = None
    nccl_so_path: str | None = None
