from typing import Literal, TypedDict
from typing_extensions import Required


# Update from torchtune.models.{model_family}.__init__.py files
# Only include argument-less TransformerDecoder factory functions
TorchtuneModel = Literal[
    # Gemma
    "gemma_2b",
    "gemma_7b",
    # Gemma2
    "gemma2_2b",
    "gemma2_9b",
    "gemma2_27b",
    # Llama2
    "llama2_7b",
    "llama2_13b",
    "llama2_70b",
    "llama2_reward_7b",
    # Llama3
    "llama3_8b",
    "llama3_70b",
    # Llama3.1
    "llama3_1_8b",
    "llama3_1_70b",
    "llama3_1_405b",
    # Llama3.2
    "llama3_2_1b",
    "llama3_2_3b",
    # Llama3.2 Vision
    "llama3_2_vision_11b",
    "llama3_2_vision_90b",
    # Llama3.3
    "llama3_3_70b",
    # Llama4
    "llama4_scout_17b_16e",
    "llama4_maverick_17b_128e",
    # Mistral
    "mistral_7b",
    "mistral_reward_7b",
    # Phi3
    "phi3_mini",
    # Phi4
    "phi4_14b",
    # Qwen2
    "qwen2_0_5b",
    "qwen2_1_5b",
    "qwen2_7b",
    # Qwen2.5
    "qwen2_5_0_5b",
    "qwen2_5_1_5b_base",
    "qwen2_5_1_5b_instruct",
    "qwen2_5_3b",
    "qwen2_5_7b_base",
    "qwen2_5_7b_instruct",
    "qwen2_5_14b_base",
    "qwen2_5_14b_instruct",
    "qwen2_5_32b_base",
    "qwen2_5_32b_instruct",
    "qwen2_5_72b_base",
    "qwen2_5_72b_instruct",
    # Qwen3
    "qwen3_0_6b_base",
    "qwen3_0_6b_instruct",
    "qwen3_1_7b_base",
    "qwen3_1_7b_instruct",
    "qwen3_4b_base",
    "qwen3_4b_instruct",
    "qwen3_8b_base",
    "qwen3_8b_instruct",
    "qwen3_14b_base",
    "qwen3_14b_instruct",
    "qwen3_32b",
]

# Update from torchtune.training.checkpointing._utils.ModelType
TorchtuneModelType = Literal[
    "GEMMA",
    "GEMMA2",
    "LLAMA2",
    "LLAMA3",
    "LLAMA3_2",
    "LLAMA3_VISION",
    "LLAMA4",
    "MISTRAL",
    "PHI3_MINI",
    "PHI4",
    "REWARD",
    "QWEN2",
    "CLIP_TEXT",
    "T5_ENCODER",
    "QWEN3",
]


class TorchtuneArgs(TypedDict, total=False):
    model: Required[TorchtuneModel]
    model_type: Required[TorchtuneModelType]
    tensor_parallel_dim: int
    context_parallel_dim: int
    async_weight_syncing: bool
