from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.transformer.enums import AttnBackend
import torch


def get_provider(model: str) -> GPTModelProvider:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, Qwen3MoEBridge), (
        "Only Qwen3 MoE models are supported"
    )
    provider = bridge.to_megatron_provider()
    provider.attention_backend = AttnBackend.fused
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.tensor_model_parallel_size = min(2, torch.cuda.device_count())
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = torch.cuda.device_count()
    provider.expert_tensor_parallel_size = 1
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    return provider
