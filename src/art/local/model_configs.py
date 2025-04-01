from dataclasses import dataclass
import torch
from torchtune.models.qwen2_5 import (
    qwen2_5_7b_base,
    qwen2_5_14b_base,
    qwen2_5_14b_instruct,
    qwen2_5_32b_base,
    qwen2_5_32b_instruct,
    qwen2_5_72b_instruct,
)
from torchtune.models.llama3_1 import llama3_1_8b, llama3_1_70b
from torchtune.modules import TransformerDecoder
from typing import Callable


@dataclass
class ModelConfig:
    """Basic language model configuration"""

    base_model: str
    min_gpus: int
    tune_model_type: str
    tune_model: Callable[[], TransformerDecoder]
    tune_num_output_chunks: int
    vllm_tool_call_parser: str | None

    def __post_init__(self) -> None:
        assert (
            torch.cuda.device_count() >= self.min_gpus
        ), f"{self.base_model} requires at least {self.min_gpus} GPUs"


def qwen_7b() -> ModelConfig:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-7B model config."""
    return ModelConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        min_gpus=1,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_7b_base,
        tune_num_output_chunks=8,
        vllm_tool_call_parser=None,
    )


def distilled_qwen_7b() -> ModelConfig:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-7B model config."""
    return ModelConfig(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        min_gpus=1,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_7b_base,
        tune_num_output_chunks=8,
        vllm_tool_call_parser=None,
    )


def llama_8b() -> ModelConfig:
    """unsloth/Meta-Llama-3.1-8B-Instruct model config."""
    return ModelConfig(
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        min_gpus=1,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_8b,
        tune_num_output_chunks=8,
        vllm_tool_call_parser=None,
    )


def theta_8b() -> ModelConfig:
    """NousResearch/Hermes-2-Theta-Llama-3-8B model config."""
    return ModelConfig(
        base_model="NousResearch/Hermes-2-Theta-Llama-3-8B",
        min_gpus=1,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_8b,
        tune_num_output_chunks=8,
        vllm_tool_call_parser="hermes",
    )


def hermes_8b() -> ModelConfig:
    """NousResearch/Hermes-3-Llama-3.1-8B model config."""
    return ModelConfig(
        base_model="NousResearch/Hermes-3-Llama-3.1-8B",
        min_gpus=1,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_8b,
        tune_num_output_chunks=8,
        vllm_tool_call_parser="hermes",
    )


def qwen_14b() -> ModelConfig:
    """Qwen/Qwen2.5-14B-Instruct model config."""
    return ModelConfig(
        base_model="Qwen/Qwen2.5-14B-Instruct",
        min_gpus=2,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_instruct,
        tune_num_output_chunks=2,
        vllm_tool_call_parser="hermes",
    )


def distilled_qwen_14b() -> ModelConfig:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-14B model config."""
    return ModelConfig(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        min_gpus=2,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_base,
        tune_num_output_chunks=2,
        vllm_tool_call_parser=None,
    )


def qwen_32b() -> ModelConfig:
    """Qwen/Qwen2.5-32B-Instruct model config."""
    return ModelConfig(
        base_model="Qwen/Qwen2.5-32B-Instruct",
        min_gpus=4,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_instruct,
        tune_num_output_chunks=2,
        vllm_tool_call_parser="hermes",
    )


def distilled_qwen_32b() -> ModelConfig:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-32B model config."""
    return ModelConfig(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        min_gpus=4,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_base,
        tune_num_output_chunks=2,
        vllm_tool_call_parser=None,
    )


def llama_70b() -> ModelConfig:
    """unsloth/Llama-3.3-70B-Instruct model config."""
    return ModelConfig(
        base_model="unsloth/Llama-3.3-70B-Instruct",
        min_gpus=8,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_num_output_chunks=2,
        vllm_tool_call_parser=None,
    )


def distilled_llama_70b() -> ModelConfig:
    """deepseek-ai/DeepSeek-R1-Distill-Llama-70B model config."""
    return ModelConfig(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        min_gpus=8,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_num_output_chunks=8,
        vllm_tool_call_parser=None,
    )


def qwen_72b() -> ModelConfig:
    """Qwen/Qwen2.5-72B-Instruct model config."""
    return ModelConfig(
        base_model="Qwen/Qwen2.5-72B-Instruct",
        min_gpus=8,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_72b_instruct,
        tune_num_output_chunks=2,
        vllm_tool_call_parser="hermes",
    )


model_configs = {
    "Qwen/Qwen2.5-7B-Instruct": qwen_7b,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": distilled_qwen_7b,
    "unsloth/Meta-Llama-3.1-8B-Instruct": llama_8b,
    "NousResearch/Hermes-2-Theta-Llama-3-8B": theta_8b,
    "NousResearch/Hermes-3-Llama-3.1-8B": hermes_8b,
    "Qwen/Qwen2.5-14B-Instruct": qwen_14b,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": distilled_qwen_14b,
    "Qwen/Qwen2.5-32B-Instruct": qwen_32b,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": distilled_qwen_32b,
    "unsloth/Llama-3.3-70B-Instruct": llama_70b,
}
