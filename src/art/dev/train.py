from typing_extensions import TypedDict


class TrainConfig(TypedDict, total=False):
    logprob_calculation_chunk_size: int
