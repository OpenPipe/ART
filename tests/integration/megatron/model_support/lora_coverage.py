from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from pydantic import BaseModel, Field
import torch
from torch.distributed import (
    destroy_process_group,
    is_initialized,
)

from art.megatron import train as megatron_train
from art.megatron.lora import LoRA

from .base_megatron_session import initialize_single_rank_process_group
from .fp32_grouped_gemm import (
    allow_fp32_grouped_gemm_fallback_for_model_support_tests,
)
from .oracle_harness import OracleCaseConfig, oracle_topology
from .oracle_worker import _configure_provider, provider_topology_env

allow_fp32_grouped_gemm_fallback_for_model_support_tests()

_WRAPPED_TARGET_SUFFIXES: dict[str, tuple[str, ...]] = {
    "q_a_proj": (".self_attn.q_a_proj",),
    "q_b_proj": (".self_attn.q_b_proj",),
    "kv_a_proj_with_mqa": (".self_attn.kv_a_proj_with_mqa",),
    "kv_b_proj": (".self_attn.kv_b_proj",),
    "kv_proj": (".self_attn.kv_proj",),
    "o_a_proj": (".self_attn.o_a_proj",),
    "o_b_proj": (".self_attn.o_b_proj",),
    "compressor.kv_proj": (".self_attn.compressor.kv_proj",),
    "compressor.gate_proj": (".self_attn.compressor.gate_proj",),
    "q_proj": (".self_attn.q_proj", ".mixer.q_proj"),
    "k_proj": (".self_attn.k_proj", ".mixer.k_proj"),
    "v_proj": (".self_attn.v_proj", ".mixer.v_proj"),
    "o_proj": (".self_attn.o_proj", ".mixer.o_proj"),
    "in_proj": (".mixer.in_proj",),
    "in_proj_qkv": (".linear_attn.in_proj_qkv",),
    "in_proj_z": (".linear_attn.in_proj_z",),
    "out_proj": (".linear_attn.out_proj", ".mixer.out_proj"),
    "gate_proj": (".gate_proj",),
    "up_proj": (".up_proj",),
    "down_proj": (".down_proj",),
    "experts": (
        ".mlp.experts.{expert}.gate_up_proj",
        ".mlp.experts.{expert}.down_proj",
    ),
}


class LoraCoverageReport(BaseModel):
    base_model: str
    target_modules: list[str]
    wrapped_target_modules: list[str] = Field(default_factory=list)
    exported_target_modules: list[str] = Field(default_factory=list)
    missing_wrapped_target_modules: list[str] = Field(default_factory=list)
    missing_exported_target_modules: list[str] = Field(default_factory=list)
    wrapped_adapter_prefix_count: int = 0
    export_base_count: int = 0
    export_adapter_count: int = 0
    trainable_lora_parameter_count: int = 0
    unexpected_trainable_parameter_names: list[str] = Field(default_factory=list)


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Megatron LoRA coverage.")
    if is_initialized():
        raise RuntimeError("torch.distributed is already initialized in this process.")
    torch.cuda.set_device(0)
    initialize_single_rank_process_group()
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        if is_initialized():
            destroy_process_group()


def _covered_wrapped_target_modules(adapter_prefixes: set[str]) -> set[str]:
    covered: set[str] = set()
    for target_module, suffixes in _WRAPPED_TARGET_SUFFIXES.items():
        if any(
            prefix.endswith(suffix)
            for prefix in adapter_prefixes
            for suffix in suffixes
        ):
            covered.add(target_module)
        if target_module == "experts" and any(
            any(
                namespace in prefix
                for namespace in (".mlp.experts.", ".mixer.experts.")
            )
            for prefix in adapter_prefixes
        ):
            covered.add(target_module)
    return covered


def _covered_exported_target_modules(
    adapter_weights_by_base: dict[str, list[Any | str | None]],
) -> set[str]:
    covered: set[str] = set()
    for base_name, adapter_weights in adapter_weights_by_base.items():
        if base_name.endswith(".self_attention.linear_q_down_proj.weight"):
            covered.add("q_a_proj")
            continue
        if base_name.endswith(".self_attention.linear_q_up_proj.weight"):
            covered.add("q_b_proj")
            continue
        if base_name.endswith(".self_attention.linear_kv_down_proj.weight"):
            covered.add("kv_a_proj_with_mqa")
            continue
        if base_name.endswith(".self_attention.linear_kv_up_proj.weight"):
            covered.add("kv_b_proj")
            continue
        if base_name.endswith(".self_attention.wq_a.weight"):
            covered.add("q_a_proj")
            continue
        if base_name.endswith(".self_attention.wq_b.weight"):
            covered.add("q_b_proj")
            continue
        if base_name.endswith(".self_attention.wkv.weight"):
            covered.add("kv_proj")
            continue
        if base_name.endswith(".self_attention.wo_a.weight"):
            covered.add("o_a_proj")
            continue
        if base_name.endswith(".self_attention.wo_b.weight"):
            covered.add("o_b_proj")
            continue
        if base_name.endswith(".self_attention.compressor.wkv.weight"):
            covered.add("compressor.kv_proj")
            continue
        if base_name.endswith(".self_attention.compressor.wgate.weight"):
            covered.add("compressor.gate_proj")
            continue
        if base_name.endswith(".self_attention.linear_qkv.weight"):
            for adapter_weight in adapter_weights:
                adapter_key = (
                    adapter_weight
                    if isinstance(adapter_weight, str) or adapter_weight is None
                    else getattr(adapter_weight, "adapter_key", None)
                )
                if adapter_key == "adapter_q":
                    covered.add("q_proj")
                elif adapter_key == "adapter_k":
                    covered.add("k_proj")
                elif adapter_key == "adapter_v":
                    covered.add("v_proj")
            continue
        if base_name.endswith(".self_attention.linear_proj.weight"):
            covered.add("o_proj")
            continue
        if base_name.endswith(".self_attention.in_proj.weight"):
            covered.update({"in_proj_qkv", "in_proj_z"})
            continue
        if base_name.endswith(".self_attention.out_proj.weight"):
            covered.add("out_proj")
            continue
        if base_name.endswith(".mixer.in_proj.weight"):
            covered.add("in_proj")
            continue
        if base_name.endswith(".mixer.out_proj.weight"):
            covered.add("out_proj")
            continue
        if ".mlp.experts.linear_fc1" in base_name:
            covered.update({"experts", "gate_proj", "up_proj"})
            continue
        if ".mlp.experts.linear_fc2" in base_name:
            covered.update({"experts", "down_proj"})
            continue
        if ".mlp.experts.linear_fc" in base_name:
            covered.add("experts")
            continue
        if ".linear_fc1.weight" in base_name:
            covered.update({"gate_proj", "up_proj"})
            continue
        if ".linear_fc2.weight" in base_name:
            covered.add("down_proj")
    return covered


def build_lora_coverage_report(
    *,
    base_model: str,
    target_modules: list[str],
    adapter_prefixes: set[str],
    adapter_weights_by_base: dict[str, list[Any | str | None]],
    trainable_lora_parameter_names: set[str] | None = None,
    unexpected_trainable_parameter_names: set[str] | None = None,
) -> LoraCoverageReport:
    wrapped = sorted(_covered_wrapped_target_modules(adapter_prefixes))
    exported = sorted(_covered_exported_target_modules(adapter_weights_by_base))
    return LoraCoverageReport(
        base_model=base_model,
        target_modules=target_modules,
        wrapped_target_modules=wrapped,
        exported_target_modules=exported,
        missing_wrapped_target_modules=sorted(set(target_modules) - set(wrapped)),
        missing_exported_target_modules=sorted(set(target_modules) - set(exported)),
        wrapped_adapter_prefix_count=len(adapter_prefixes),
        export_base_count=len(adapter_weights_by_base),
        export_adapter_count=sum(map(len, adapter_weights_by_base.values())),
        trainable_lora_parameter_count=len(trainable_lora_parameter_names or ()),
        unexpected_trainable_parameter_names=sorted(
            unexpected_trainable_parameter_names or ()
        ),
    )


def run_lora_coverage(case_config: OracleCaseConfig) -> LoraCoverageReport:
    topology = oracle_topology(is_moe=case_config.is_moe)
    with _single_rank_model_parallel():
        with provider_topology_env(topology):
            runtime = megatron_train.build_training_runtime(
                model_identifier=case_config.base_model,
                provider_torch_dtype=torch.float32,
                provider_configure=lambda provider: _configure_provider(
                    provider, topology, case_config
                ),
                print_env=False,
                build_optimizer=False,
                allow_unvalidated_arch=case_config.allow_unvalidated_arch,
            )
        adapter_prefixes = {
            module.adapter_model_prefix
            for chunk in runtime.model
            for module in chunk.modules()
            if isinstance(module, LoRA)
        }
        adapter_weights_by_base = (
            runtime.provider_bundle.handler.build_adapter_weights_by_base(runtime.model)
        )

    return build_lora_coverage_report(
        base_model=case_config.base_model,
        target_modules=list(runtime.provider_bundle.spec.default_target_modules),
        adapter_prefixes=adapter_prefixes,
        adapter_weights_by_base=adapter_weights_by_base,
    )
