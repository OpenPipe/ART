from typing_extensions import TypedDict


class TrainConfig(TypedDict, total=False):
    enable_assistant_message_training: bool
    epsilon: float  # clip epsilon, using the same name as TRL
    epsilon_high: (
        float | None
    )  # asymmetric clip upper bound. Defaults to epsilon when None
    logprob_calculation_chunk_size: int
