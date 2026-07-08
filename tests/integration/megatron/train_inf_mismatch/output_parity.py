from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.megatron.prefix_tree import parse_prefix_tree_row

from ..model_support.workflow_resources import (
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_current_host,
)

# These gates are intentionally bf16-scale, not fp32 oracle-scale. A 2026-05-18
# Qwen/Qwen3.5-35B-A3B diagnostic on the exact same real generated tokens found:
# vLLM generation vs Megatron: 2.916% mean_abs_pct, 0.0123 MAE, 0.883 top1,
# 0.976 top20; vLLM prompt_logprobs vs Megatron: 7.896%, 0.0334 MAE, 0.969
# top1, 0.941 top20; vLLM generation vs vLLM prompt_logprobs: 7.517%, 0.0322
# MAE, 0.879 top1, 0.941 top20. The real ART path also canonicalizes shared
# prefix routes when vLLM produced different routes for the same prefix. Do not
# tighten these thresholds without rechecking both vLLM self-mismatch and shared
# prefix route-conflict behavior on the measured path. With the workflow's
# 16-token completions, Qwen3.5 MoE reruns on 2026-05-25 measured 4.169% and
# 4.606% mean_abs_pct while staying under the KL gate, so its gate is 5%.
# DeepSeek-V4-Flash uses vLLM quantized DSV4 kernels on the serving side while
# Megatron materializes train-time bf16/fp32 tensors. A 2026-06-18 diagnostic
# measured non-QAT Megatron vs vLLM generation at 19.016% mean_abs_pct and
# 0.02603 candidate->target top20 KL; vLLM generation vs exact vLLM prompt
# rescore was already 15.176% mean_abs_pct and 0.04424 KL.
BF16_FWD_MEAN_ABS_PCT_LIMIT = 4.0
BF16_FWD_MEAN_ABS_PCT_LIMIT_BY_MODEL_KEY = {
    "dsv4": 20.0,
    # Gemma 4 MoE long-prompt SWA native-LoRA runs showed high variation, with
    # repeated samples reaching 7.6% mean_abs_pct and 0.0076 KL.
    "gemma4_dense": 8.0,
    "gemma4_moe": 8.0,
    "qwen3_moe": 7.0,
    "qwen3_5_moe": 5.0,
}
TOP20_KL_CANDIDATE_TO_TARGET_LIMIT = 0.002
TOP20_KL_CANDIDATE_TO_TARGET_LIMIT_BY_MODEL_KEY = {
    "dsv4": 0.07,
    "gemma4_dense": 0.003,
    "gemma4_moe": 0.008,
    # GPT OSS MXFP4/native-LoRA repeats on 2026-07-06 stayed under the 4%
    # mean_abs_pct gate but measured 0.00186-0.00256 top20 KL.
    "gpt_oss_moe": 0.003,
}
MEAN_ABS_PCT_DENOMINATOR_EPS = 1e-18
TOP_K = 20
ScoreRecord = tuple[int, float, list[int], list[float]]

RolloutMode = Literal["native_lora", "merged"]
EngineSide = Literal["megatron", "vllm"]
WeightState = Literal["base", "lora"]


class Topology(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int = 1
    ep: int = 2
    etp: int = 1
    dp: int = 1
    cp: int = 2
    pp: int = 1

    def world_size(self) -> int:
        dense_world = self.tp * self.cp * self.pp * self.dp
        expert_model_size = self.etp * self.ep * self.pp
        if dense_world % expert_model_size != 0:
            raise ValueError(
                "Invalid Megatron MoE topology: "
                f"tp*cp*pp*dp={dense_world} must be divisible by "
                f"etp*ep*pp={expert_model_size}"
            )
        return dense_world

    def env(self) -> dict[str, str]:
        return {
            "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE": str(self.tp),
            "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE": str(self.ep),
            "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE": str(self.etp),
            "ART_MEGATRON_CONTEXT_PARALLEL_SIZE": str(self.cp),
            "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE": str(self.pp),
        }

    def slug(self) -> str:
        return (
            f"tp{self.tp}_ep{self.ep}_etp{self.etp}_dp{self.dp}_cp{self.cp}_pp{self.pp}"
        )


class ProbePackedConfig(BaseModel):
    num_sequences: int = 4
    sequence_length: int = 1024
    prefill_tokens: int = 256
    completion_branches_per_prefix: int = 2
    decode_tokens: int = 128
    decode_tokens_jitter: int = 32
    vocab_high: int = 8192
    packing_mode: Literal["stop_early", "truncate"] = "stop_early"


class TrainInfOutputParityConfig(BaseModel):
    base_model: str = "Qwen/Qwen3.5-35B-A3B"
    seed: int = 20260512
    topology: Topology = Field(default_factory=Topology)
    packed: ProbePackedConfig = Field(default_factory=ProbePackedConfig)
    rollout_modes: list[RolloutMode] = Field(default_factory=list)
    trainer_gpu_ids: list[int] = Field(default_factory=lambda: [0, 1])
    inference_gpu_ids: list[int] = Field(default_factory=lambda: [2, 3])
    allow_unvalidated_arch: bool = False
    lora_target_modules: list[str] | None = None
    engine_args: dict[str, Any] = Field(default_factory=dict)
    server_args: dict[str, Any] = Field(default_factory=dict)
    megatron_env: dict[str, str] = Field(default_factory=dict)
    replay_vllm_routing: bool = False
    external_vllm_server_url: str | None = None
    external_vllm_api_key: str | None = None

    @model_validator(mode="after")
    def _set_default_rollout_modes(self) -> "TrainInfOutputParityConfig":
        if not self.rollout_modes:
            self.rollout_modes = default_rollout_modes_for_model(
                self.base_model,
                allow_unvalidated_arch=self.allow_unvalidated_arch,
            )
        return self


class LogicalPrompt(BaseModel):
    prompt_id: int
    sample_id: int
    family_id: int
    completion_id: int
    # ART stores the final context token at the start of each leaf segment, so
    # vLLM's generated-token logprobs start one token after the ancestor path.
    packed_prompt_length: int
    scored_token_start_index: int
    token_ids: list[int]


class LogicalToken(BaseModel):
    token_id: int
    sample_id: int
    family_id: int
    completion_id: int
    prompt_id: int
    art_packed_token_index: int
    art_logit_index: int
    vllm_prompt_token_index: int


class LogicalTokenMap(BaseModel):
    prompts: list[LogicalPrompt]
    tokens: list[LogicalToken]


class TokenTopK(BaseModel):
    token_ids: list[int]
    logprobs: list[float]


class ScoreBundle(BaseModel):
    side: EngineSide
    weight_state: WeightState
    rollout_mode: RolloutMode | None = None
    target_logprobs: list[float]
    topk: list[TokenTopK]


class MeanAbsPctSummary(BaseModel):
    mean_abs_pct: float
    sequence_count: int
    source_numel: int
    trimmed_numel: int


class PairComparison(BaseModel):
    mean_abs_pct: float
    sequence_count: int
    source_numel: int
    trimmed_numel: int
    mae: float
    max_abs: float
    p50_abs: float
    p95_abs: float
    p99_abs: float


class TopKComparison(BaseModel):
    top1_match_rate: float
    top20_overlap_rate: float
    top20_intersection_logprob_mae: float
    top20_intersection_kl_target_to_candidate: float
    top20_intersection_kl_candidate_to_target: float
    compared_intersection_count: int


class RolloutComparison(BaseModel):
    rollout_mode: RolloutMode
    base: PairComparison
    lora: PairComparison
    delta: PairComparison
    base_topk: TopKComparison
    lora_topk: TopKComparison


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _parse_gpu_ids(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one comma-separated value")
    return parts


def _parse_rollout_modes(value: str) -> list[RolloutMode]:
    modes = _parse_str_list(value)
    invalid = sorted(set(modes) - {"native_lora", "merged"})
    if invalid:
        raise ValueError(f"Unsupported rollout modes: {invalid}")
    return cast(list[RolloutMode], modes)


def default_rollout_modes_for_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> list[RolloutMode]:
    from art.megatron.model_support.registry import native_vllm_lora_status_for_model

    modes: list[RolloutMode] = []
    if (
        native_vllm_lora_status_for_model(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
        != "disabled"
    ):
        modes.append("native_lora")
    modes.append("merged")
    return modes


def fwd_mean_abs_pct_limit_for_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> float:
    from art.megatron.model_support.registry import get_model_support_spec

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return BF16_FWD_MEAN_ABS_PCT_LIMIT_BY_MODEL_KEY.get(
        spec.key,
        BF16_FWD_MEAN_ABS_PCT_LIMIT,
    )


def top20_kl_candidate_to_target_limit_for_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> float:
    from art.megatron.model_support.registry import get_model_support_spec

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return TOP20_KL_CANDIDATE_TO_TARGET_LIMIT_BY_MODEL_KEY.get(
        spec.key,
        TOP20_KL_CANDIDATE_TO_TARGET_LIMIT,
    )


def model_support_is_moe(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> bool:
    from art.megatron.model_support.registry import (
        get_model_support_handler_for_spec,
        get_model_support_spec,
    )

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return get_model_support_handler_for_spec(spec).is_moe


def model_supports_context_parallel(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> bool:
    from art.megatron.model_support.registry import (
        get_model_support_handler_for_spec,
        get_model_support_spec,
    )

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return bool(getattr(get_model_support_handler_for_spec(spec), "cp_supported", True))


def config_from_env() -> TrainInfOutputParityConfig:
    train_inf_external_url_env = "ART_TRAIN_INF_MISMATCH_EXTERNAL_VLLM_URL"
    model_support_external_url_env = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL"
    train_inf_external_key_env = "ART_TRAIN_INF_MISMATCH_EXTERNAL_VLLM_API_KEY"
    model_support_external_key_env = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_API_KEY"
    config = TrainInfOutputParityConfig(
        base_model=os.environ.get(
            "ART_TRAIN_INF_MISMATCH_BASE_MODEL",
            os.environ.get("BASE_MODEL", TrainInfOutputParityConfig().base_model),
        ),
        trainer_gpu_ids=_parse_gpu_ids(
            os.environ.get("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS"),
            [0, 1],
        ),
        inference_gpu_ids=_parse_gpu_ids(
            os.environ.get("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS"),
            [2, 3],
        ),
        allow_unvalidated_arch=os.environ.get(
            "ART_TRAIN_INF_MISMATCH_ALLOW_UNVALIDATED_ARCH", "0"
        )
        == "1",
    )
    workflow_resources = handler_workflow_resources_for_base_model(
        config.base_model,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    stage_resources = (
        workflow_resources.train_inf_mismatch
        if workflow_resources is not None
        else None
    )
    if stage_resources is not None:
        stage_resources = resolve_stage_resources_for_current_host(
            "train_inf_mismatch",
            stage_resources,
        )
    if stage_resources is not None:
        if (
            stage_resources.megatron is not None
            and "ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS" not in os.environ
        ):
            config.trainer_gpu_ids = list(stage_resources.megatron.gpu_ids)
        if (
            stage_resources.vllm is not None
            and "ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS" not in os.environ
        ):
            config.inference_gpu_ids = list(stage_resources.vllm.gpu_ids)
        if stage_resources.megatron is not None:
            config.topology = config.topology.model_copy(
                update=stage_resources.megatron.topology.to_train_inf_topology_kwargs()
            )
        if stage_resources.vllm is not None:
            config.engine_args = {
                **stage_resources.vllm.engine_args(),
                **config.engine_args,
            }
        config.megatron_env = {
            **stage_resources.megatron_env,
            **config.megatron_env,
        }
    if raw_modes := os.environ.get("ART_TRAIN_INF_MISMATCH_ROLLOUT_MODES"):
        config.rollout_modes = _parse_rollout_modes(raw_modes)
    if raw_seq_len := os.environ.get("ART_TRAIN_INF_MISMATCH_SEQUENCE_LENGTH"):
        config.packed.sequence_length = int(raw_seq_len)
    if raw_prefill := os.environ.get("ART_TRAIN_INF_MISMATCH_PREFILL_TOKENS"):
        config.packed.prefill_tokens = int(raw_prefill)
    if raw_decode := os.environ.get("ART_TRAIN_INF_MISMATCH_DECODE_TOKENS"):
        config.packed.decode_tokens = int(raw_decode)
    for env_name, attr in (
        ("ART_TRAIN_INF_MISMATCH_TP", "tp"),
        ("ART_TRAIN_INF_MISMATCH_EP", "ep"),
        ("ART_TRAIN_INF_MISMATCH_ETP", "etp"),
        ("ART_TRAIN_INF_MISMATCH_DP", "dp"),
        ("ART_TRAIN_INF_MISMATCH_CP", "cp"),
        ("ART_TRAIN_INF_MISMATCH_PP", "pp"),
    ):
        if raw_value := os.environ.get(env_name):
            config.topology = config.topology.model_copy(update={attr: int(raw_value)})
    is_moe = model_support_is_moe(
        config.base_model,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    cp_supported = model_supports_context_parallel(
        config.base_model,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    if not is_moe:
        config.topology = config.topology.model_copy(update={"ep": 1, "etp": 1})
    if not cp_supported and "ART_TRAIN_INF_MISMATCH_CP" not in os.environ:
        updates = {"cp": 1}
        if stage_resources is None and "ART_TRAIN_INF_MISMATCH_DP" not in os.environ:
            updates["dp"] = config.topology.dp * config.topology.cp
        config.topology = config.topology.model_copy(update=updates)
    if raw_targets := os.environ.get("ART_TRAIN_INF_MISMATCH_LORA_TARGET_MODULES"):
        config.lora_target_modules = _parse_str_list(raw_targets)
    if raw_vllm_memory := os.environ.get(
        "ART_TRAIN_INF_MISMATCH_VLLM_GPU_MEMORY_UTILIZATION"
    ):
        config.engine_args["gpu_memory_utilization"] = float(raw_vllm_memory)
    if raw_gdn_backend := os.environ.get("ART_TRAIN_INF_MISMATCH_GDN_PREFILL_BACKEND"):
        raw_additional_config = config.engine_args.get("additional_config")
        additional_config: dict[str, Any] = {}
        if isinstance(raw_additional_config, dict):
            additional_config.update(cast(dict[str, Any], raw_additional_config))
        additional_config["gdn_prefill_backend"] = raw_gdn_backend
        config.engine_args["additional_config"] = additional_config
    raw_url = os.environ.get(train_inf_external_url_env)
    if raw_url is None and stage_resources is not None:
        raw_url = os.environ.get(model_support_external_url_env)
    if raw_url:
        config.external_vllm_server_url = raw_url
        config.external_vllm_api_key = os.environ.get(
            train_inf_external_key_env,
            os.environ.get(
                model_support_external_key_env,
                "art-external-vllm",
            ),
        )
    if (
        stage_resources is not None
        and stage_resources.requires_external_vllm
        and not config.external_vllm_server_url
    ):
        raise RuntimeError(
            "train_inf_mismatch for this model requires an external vLLM server. "
            f"Set {train_inf_external_url_env} or {model_support_external_url_env}."
        )
    return config


def _prefix_tree_leaf_paths(
    group_ids: Any,
    parent_ids: Any,
    *,
    required_leaf_count: int = 1,
) -> list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int]]]:
    tree = parse_prefix_tree_row(group_ids=group_ids, parent_ids=parent_ids)
    segment_by_group = {segment.group_id: segment for segment in tree.segments}
    child_count_by_group: dict[int, int] = {}
    for segment in tree.segments:
        if segment.group_id == segment.parent_id:
            continue
        child_count_by_group[segment.parent_id] = (
            child_count_by_group.get(segment.parent_id, 0) + 1
        )
    paths = [
        (
            leaf.family_index,
            tuple(
                (segment_by_group[group_id].start, segment_by_group[group_id].end)
                for group_id in leaf.ancestors
            ),
            (leaf.start, leaf.end),
        )
        for leaf in tree.segments
        if leaf.group_id != leaf.parent_id
        and child_count_by_group.get(leaf.group_id, 0) == 0
        and leaf.end - leaf.start >= 2
    ]
    return paths if len(paths) >= required_leaf_count else []


def build_logical_token_map(packed_tensors: dict[str, Any]) -> LogicalTokenMap:
    tokens = packed_tensors["tokens"]
    group_ids = packed_tensors["group_ids"]
    parent_ids = packed_tensors["parent_ids"]
    assistant_mask = packed_tensors.get("assistant_mask")
    logprobs = packed_tensors.get("logprobs")
    prompts: list[LogicalPrompt] = []
    logical_tokens: list[LogicalToken] = []
    prompt_id_by_tokens: dict[tuple[int, ...], int] = {}

    def scored_token(sample_id: int, packed_i: int) -> bool:
        if assistant_mask is not None and not bool(assistant_mask[sample_id, packed_i]):
            return False
        if logprobs is not None:
            value = float(logprobs[sample_id, packed_i])
            if math.isnan(value):
                return False
        return True

    for sample_id in range(int(tokens.shape[0])):
        leaf_paths = _prefix_tree_leaf_paths(
            group_ids[sample_id], parent_ids[sample_id]
        )
        for completion_id, (family_id, ancestor_segments, leaf_segment) in enumerate(
            leaf_paths
        ):
            leaf_start, leaf_end = leaf_segment
            last_scored_i = None
            for packed_i in range(leaf_start + 1, leaf_end):
                if scored_token(sample_id, packed_i):
                    last_scored_i = packed_i
            if last_scored_i is None:
                continue
            effective_leaf_end = last_scored_i + 1
            prompt_len = sum(end - start for start, end in ancestor_segments)
            reference_segments = (*ancestor_segments, (leaf_start, effective_leaf_end))
            flat = [
                int(value)
                for start, end in reference_segments
                for value in tokens[sample_id, start:end].tolist()
            ]
            flat_key = tuple(flat)
            prompt_id = prompt_id_by_tokens.get(flat_key)
            if prompt_id is None:
                prompt_id = len(prompts)
                prompt_id_by_tokens[flat_key] = prompt_id
                prompts.append(
                    LogicalPrompt(
                        prompt_id=prompt_id,
                        sample_id=sample_id,
                        family_id=family_id,
                        completion_id=completion_id,
                        packed_prompt_length=prompt_len,
                        scored_token_start_index=prompt_len + 1,
                        token_ids=flat,
                    )
                )
            for packed_i in range(leaf_start + 1, effective_leaf_end):
                if not scored_token(sample_id, packed_i):
                    continue
                logical_tokens.append(
                    LogicalToken(
                        token_id=int(tokens[sample_id, packed_i].item()),
                        sample_id=sample_id,
                        family_id=family_id,
                        completion_id=completion_id,
                        prompt_id=prompt_id,
                        art_packed_token_index=packed_i,
                        art_logit_index=packed_i - 1,
                        vllm_prompt_token_index=prompt_len + (packed_i - leaf_start),
                    )
                )

    if not prompts or not logical_tokens:
        raise RuntimeError("Prefix-tree probe produced no comparable logical tokens")
    return LogicalTokenMap(prompts=prompts, tokens=logical_tokens)


def aggregate_mean_abs_pct(
    *,
    candidate: Any,
    target: Any,
    sequence_ids: list[int],
) -> MeanAbsPctSummary:
    import torch

    cand = candidate.detach().float().reshape(-1)
    ref = target.detach().float().reshape(-1)
    if cand.shape != ref.shape:
        raise RuntimeError(f"Shape mismatch: candidate={cand.shape} target={ref.shape}")
    if cand.numel() != len(sequence_ids):
        raise RuntimeError(
            f"sequence_ids length mismatch: {len(sequence_ids)} != {cand.numel()}"
        )
    if cand.numel() == 0:
        return MeanAbsPctSummary(
            mean_abs_pct=0.0,
            sequence_count=0,
            source_numel=0,
            trimmed_numel=0,
        )
    sequence_count = len({int(sequence_id) for sequence_id in sequence_ids})
    mean_abs_diff = float((cand - ref).abs().mean().item())
    mean_abs_reference = float(ref.abs().mean().item())
    return MeanAbsPctSummary(
        mean_abs_pct=(
            mean_abs_diff / (mean_abs_reference + MEAN_ABS_PCT_DENOMINATOR_EPS)
        )
        * 100.0,
        sequence_count=sequence_count,
        source_numel=int(cand.numel()),
        trimmed_numel=0,
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[index])


def compare_pair(
    *,
    candidate: Any,
    target: Any,
    sequence_ids: list[int],
) -> PairComparison:
    import torch

    cand = candidate.detach().float().reshape(-1)
    ref = target.detach().float().reshape(-1)
    pct = aggregate_mean_abs_pct(
        candidate=cand,
        target=ref,
        sequence_ids=sequence_ids,
    )
    diff = (cand - ref).abs()
    sorted_diff = sorted(float(value) for value in diff.tolist())
    return PairComparison(
        mean_abs_pct=pct.mean_abs_pct,
        sequence_count=pct.sequence_count,
        source_numel=pct.source_numel,
        trimmed_numel=pct.trimmed_numel,
        mae=float(diff.mean().item()) if diff.numel() else 0.0,
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
        p50_abs=_percentile(sorted_diff, 0.50),
        p95_abs=_percentile(sorted_diff, 0.95),
        p99_abs=_percentile(sorted_diff, 0.99),
    )


def _logsumexp(values: list[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _restricted_kl(
    left_by_id: dict[int, float],
    right_by_id: dict[int, float],
    token_ids: set[int],
) -> float:
    if not token_ids:
        return 0.0
    ordered_ids = sorted(token_ids)
    left_values = [left_by_id[token_id] for token_id in ordered_ids]
    right_values = [right_by_id[token_id] for token_id in ordered_ids]
    left_log_z = _logsumexp(left_values)
    right_log_z = _logsumexp(right_values)
    kl = 0.0
    for left_value, right_value in zip(left_values, right_values, strict=True):
        left_logprob = left_value - left_log_z
        right_logprob = right_value - right_log_z
        kl += math.exp(left_logprob) * (left_logprob - right_logprob)
    return float(kl)


def compare_topk(candidate: ScoreBundle, target: ScoreBundle) -> TopKComparison:
    if len(candidate.topk) != len(target.topk):
        raise RuntimeError("top-k score length mismatch")
    top1_matches = 0
    overlap_sum = 0.0
    intersection_abs_sum = 0.0
    intersection_count = 0
    target_to_candidate_kl_sum = 0.0
    candidate_to_target_kl_sum = 0.0
    kl_count = 0
    for cand_topk, ref_topk in zip(candidate.topk, target.topk, strict=True):
        cand_ids = cand_topk.token_ids[:TOP_K]
        ref_ids = ref_topk.token_ids[:TOP_K]
        if cand_ids and ref_ids and cand_ids[0] == ref_ids[0]:
            top1_matches += 1
        cand_set = set(cand_ids)
        ref_set = set(ref_ids)
        intersection = cand_set & ref_set
        overlap_sum += len(intersection) / max(TOP_K, 1)
        cand_by_id = dict(zip(cand_topk.token_ids, cand_topk.logprobs, strict=True))
        ref_by_id = dict(zip(ref_topk.token_ids, ref_topk.logprobs, strict=True))
        for token_id in intersection:
            intersection_abs_sum += abs(cand_by_id[token_id] - ref_by_id[token_id])
            intersection_count += 1
        if intersection:
            target_to_candidate_kl_sum += _restricted_kl(
                ref_by_id, cand_by_id, intersection
            )
            candidate_to_target_kl_sum += _restricted_kl(
                cand_by_id, ref_by_id, intersection
            )
            kl_count += 1
    count = max(len(candidate.topk), 1)
    return TopKComparison(
        top1_match_rate=top1_matches / count,
        top20_overlap_rate=overlap_sum / count,
        top20_intersection_logprob_mae=(
            intersection_abs_sum / intersection_count if intersection_count else 0.0
        ),
        top20_intersection_kl_target_to_candidate=(
            target_to_candidate_kl_sum / kl_count if kl_count else 0.0
        ),
        top20_intersection_kl_candidate_to_target=(
            candidate_to_target_kl_sum / kl_count if kl_count else 0.0
        ),
        compared_intersection_count=intersection_count,
    )


def compare_rollout(
    *,
    rollout_mode: RolloutMode,
    megatron_base: ScoreBundle,
    megatron_lora: ScoreBundle,
    vllm_base: ScoreBundle,
    vllm_lora: ScoreBundle,
    logical_map: LogicalTokenMap,
) -> RolloutComparison:
    import torch

    sequence_ids = [token.prompt_id for token in logical_map.tokens]
    mb = torch.tensor(megatron_base.target_logprobs, dtype=torch.float32)
    ml = torch.tensor(megatron_lora.target_logprobs, dtype=torch.float32)
    vb = torch.tensor(vllm_base.target_logprobs, dtype=torch.float32)
    vl = torch.tensor(vllm_lora.target_logprobs, dtype=torch.float32)
    return RolloutComparison(
        rollout_mode=rollout_mode,
        base=compare_pair(candidate=mb, target=vb, sequence_ids=sequence_ids),
        lora=compare_pair(candidate=ml, target=vl, sequence_ids=sequence_ids),
        delta=compare_pair(
            candidate=ml - mb,
            target=vl - vb,
            sequence_ids=sequence_ids,
        ),
        base_topk=compare_topk(megatron_base, vllm_base),
        lora_topk=compare_topk(megatron_lora, vllm_lora),
    )


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_provider(provider: Any, config: TrainInfOutputParityConfig) -> None:
    provider.tensor_model_parallel_size = config.topology.tp
    provider.expert_model_parallel_size = config.topology.ep
    provider.expert_tensor_parallel_size = config.topology.etp
    provider.context_parallel_size = config.topology.cp
    provider.pipeline_model_parallel_size = config.topology.pp
    if hasattr(provider, "attention_dropout"):
        provider.attention_dropout = 0.0
    if hasattr(provider, "hidden_dropout"):
        provider.hidden_dropout = 0.0


def _gather_context_parallel_logits(logits: Any, *, full_sequence_length: int) -> Any:
    from megatron.core import parallel_state as ps
    import torch
    import torch.distributed as dist

    if int(ps.get_context_parallel_world_size()) <= 1:
        return logits
    if int(logits.shape[1]) == full_sequence_length:
        return logits
    cp_size = int(ps.get_context_parallel_world_size())
    local_chunks = [torch.empty_like(logits) for _ in range(cp_size)]
    dist.all_gather(  # ty: ignore[possibly-missing-attribute]
        local_chunks, logits.contiguous(), group=ps.get_context_parallel_group()
    )
    local_sequence_length = int(logits.shape[1])
    if local_sequence_length % 2 != 0:
        raise RuntimeError(
            "Cannot reconstruct context-parallel logits with odd local sequence "
            f"length {local_sequence_length}"
        )
    half = local_sequence_length // 2
    ordered = [chunk[:, :half] for chunk in local_chunks]
    ordered.extend(chunk[:, half:] for chunk in reversed(local_chunks))
    gathered = torch.cat(ordered, dim=1)
    if int(gathered.shape[1]) != full_sequence_length:
        raise RuntimeError(
            "Context-parallel logit gather produced unexpected sequence length: "
            f"{int(gathered.shape[1])} != {full_sequence_length}"
        )
    return gathered


def _packed_valid_lengths(packed_tensors: dict[str, Any]) -> list[int]:
    return [
        int((packed_tensors["group_ids"][row_index] != -1).sum().item())
        for row_index in range(int(packed_tensors["group_ids"].shape[0]))
    ]


def logical_logit_uids(
    *,
    packed_tensors: dict[str, Any],
    logical_tokens: Sequence[LogicalToken],
    sample_id_to_row: dict[int, int] | None = None,
) -> list[int]:
    valid_lengths = _packed_valid_lengths(packed_tensors)
    row_offsets: list[int] = []
    cursor = 0
    for valid_length in valid_lengths:
        row_offsets.append(cursor)
        cursor += valid_length
    uids: list[int] = []
    for token in logical_tokens:
        row_index = (
            sample_id_to_row[token.sample_id]
            if sample_id_to_row is not None
            else token.sample_id
        )
        if row_index < 0 or row_index >= len(valid_lengths):
            raise RuntimeError(
                "Logical token sample does not map to a packed row: "
                f"sample_id={token.sample_id}, row={row_index}"
            )
        if (
            token.art_logit_index < 0
            or token.art_logit_index >= valid_lengths[row_index]
        ):
            raise RuntimeError(
                "Logical token logit index is outside packed valid tokens: "
                f"sample_id={token.sample_id}, row={row_index}, "
                f"logit_index={token.art_logit_index}, "
                f"valid_length={valid_lengths[row_index]}"
            )
        uids.append(row_offsets[row_index] + token.art_logit_index)
    return uids


def _lora_target_modules(config: TrainInfOutputParityConfig) -> list[str]:
    from art.dev.get_model_config import default_target_modules

    return list(config.lora_target_modules or default_target_modules(config.base_model))


def _configure_lora_target_modules(
    provider_bundle: Any, target_modules: list[str]
) -> None:
    if not target_modules:
        raise ValueError("LoRA target module override cannot be empty")
    spec = provider_bundle.spec.model_copy(
        update={"default_target_modules": tuple(target_modules)}
    )
    provider_bundle.spec = spec
    setattr(provider_bundle.provider, "_art_model_support_spec", spec)


def _build_deterministic_nonzero_lora(
    initial_state: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    import torch

    initialized: dict[str, Any] = {}
    for key in sorted(initial_state):
        value = initial_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor for LoRA key {key!r}")
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        key_seed = int.from_bytes(digest[:8], "little") % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(key_seed)
        random_values = torch.randn(value.shape, generator=generator)
        initialized[key] = (0.01 * random_values).to(value.dtype).contiguous()
    return initialized


def _merge_sharded_lora(shards_by_rank: list[dict[str, Any]]) -> dict[str, Any]:
    from art.megatron.weights.lora_publish import merge_sharded_adapter_entries

    entries_by_key: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    for rank_entry in shards_by_rank:
        state = rank_entry["state"]
        manifest = rank_entry["manifest"]
        for key, tensor in state.items():
            entries_by_key.setdefault(key, []).append((manifest[key], tensor))
    return merge_sharded_adapter_entries(entries_by_key)


def _collect_full_lora_state(model_chunks: list[Any]) -> dict[str, Any] | None:
    import torch

    local_state: dict[str, Any] = {}
    local_manifest: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "sharded_lora_manifest"):
                local_manifest.update(module.sharded_lora_manifest())
            if hasattr(module, "sharded_lora_state_dict"):
                local_state.update(
                    {
                        key: value.detach().cpu()
                        for key, value in module.sharded_lora_state_dict().items()
                    }
                )
    rank = torch.distributed.get_rank()  # type: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # type: ignore[possibly-missing-attribute]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(  # type: ignore[possibly-missing-attribute]
        {"state": local_state, "manifest": local_manifest},
        gathered,
        dst=0,
    )
    if rank != 0:
        return None
    assert gathered is not None
    return _merge_sharded_lora([entry for entry in gathered if entry is not None])


def _adapter_config(config: TrainInfOutputParityConfig) -> dict[str, Any]:
    from peft.tuners.lora.config import LoraConfig

    from art.megatron.lora import LORA_ALPHA, default_lora_rank_for_handler
    from art.megatron.model_support import get_model_support_handler

    return LoraConfig(
        base_model_name_or_path=config.base_model,
        r=default_lora_rank_for_handler(
            get_model_support_handler(
                config.base_model,
                allow_unvalidated_arch=config.allow_unvalidated_arch,
            )
        ),
        lora_alpha=LORA_ALPHA,
        target_modules=_lora_target_modules(config),
        bias="none",
    ).to_dict()


def _save_vllm_lora_adapter(
    *,
    lora_path: Path,
    state: dict[str, Any],
    runtime: Any,
    config: TrainInfOutputParityConfig,
) -> None:
    import torch

    from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

    if not state:
        raise RuntimeError("Refusing to save empty LoRA state")
    zero_keys = [
        key
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
        and int(torch.count_nonzero(value).item()) == 0
    ]
    if zero_keys:
        raise RuntimeError(f"Refusing zero LoRA tensors: {zero_keys[:5]}")
    adapter_config = _adapter_config(config)
    tensors, adapter_config = runtime.model_support_handler.to_vllm_lora_tensors(
        state,
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(lora_path, tensors, adapter_config)


def _run_logits(
    *,
    runtime: Any,
    packed_tensors: dict[str, Any],
) -> Any:
    import torch

    from art.megatron.prefix_tree_state import create_prefix_tree_state
    from art.megatron.training.trace import (
        packed_sequence_token_uids,
        prepare_replay_local_input_token_uids,
    )
    from art.preprocessing.pack import PackedTensors

    device = next(runtime.model[0].parameters()).device
    input_ids = packed_tensors["tokens"].to(device=device)
    position_ids = packed_tensors["input_pos"].to(device=device)
    group_ids = packed_tensors["group_ids"].to(device=device)
    parent_ids = packed_tensors["parent_ids"].to(device=device)
    attention_state = create_prefix_tree_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=position_ids,
        sliding_windows=tuple(
            int(window)
            for window in getattr(runtime.provider, "art_flex_sliding_windows", ())
        ),
        build_gdn_execution_spec=bool(
            getattr(runtime.model_support_handler, "build_gdn_execution_spec", False)
        ),
        model_support_handler=runtime.model_support_handler,
        attention_head_dim=getattr(runtime.provider, "kv_channels", None),
        attention_value_head_dim=getattr(runtime.provider, "kv_channels", None),
    )
    prepare_replay_local_input_token_uids(
        runtime.moe_routing_replay_controller,
        packed_sequence_token_uids(cast(PackedTensors, packed_tensors), device=device),
        attention_state,
    )
    with torch.no_grad():
        logits = runtime.model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device),
            labels=None,
            **runtime.model_support_handler.get_forward_kwargs(
                runtime.model[0],
                attention_bias=attention_state,
            ),
        )
        from megatron.core import parallel_state, tensor_parallel

        if (
            parallel_state.model_parallel_is_initialized()
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        logits = _gather_context_parallel_logits(
            logits,
            full_sequence_length=int(input_ids.shape[1]),
        )
        return logits


def _batch_seq_logits(logits: Any, labels: Any) -> Any:
    if int(logits.ndim) != 3:
        raise RuntimeError(
            f"Expected logits [B, S, V] or [S, B, V], got {logits.shape}"
        )
    if tuple(logits.shape[:2]) == tuple(labels.shape):
        return logits
    if tuple(logits.shape[:2]) == (int(labels.shape[1]), int(labels.shape[0])):
        return logits.transpose(0, 1).contiguous()
    raise RuntimeError(
        "Logits do not align with local labels: "
        f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
    )


def _local_score_records_from_logits(
    *,
    logits: Any,
    labels: Any,
    token_uids: Any,
    desired_uids: set[int],
) -> dict[int, ScoreRecord]:
    import torch

    if token_uids is None:
        raise RuntimeError("CP train/inf scoring requires local token_uids")
    logits = _batch_seq_logits(logits, labels)
    if tuple(token_uids.shape) != tuple(labels.shape):
        raise RuntimeError(
            "CP token uid shape does not match labels: "
            f"uids={tuple(token_uids.shape)}, labels={tuple(labels.shape)}"
        )
    if not desired_uids:
        return {}
    records: dict[int, ScoreRecord] = {}
    log_probs = torch.log_softmax(logits.detach().float(), dim=-1)
    labels_cpu = labels.detach().to(device="cpu")
    token_uids_cpu = token_uids.detach().to(device="cpu")
    mask = (labels_cpu != -100) & (token_uids_cpu >= 0)
    for batch_index, seq_index in torch.nonzero(mask, as_tuple=False).tolist():
        uid = int(token_uids_cpu[batch_index, seq_index].item())
        if uid not in desired_uids:
            continue
        row = log_probs[batch_index, seq_index]
        token_id = int(labels_cpu[batch_index, seq_index].item())
        values, indices = torch.topk(row, TOP_K)
        records[uid] = (
            token_id,
            float(row[token_id].item()),
            [int(value) for value in indices.tolist()],
            [float(value) for value in values.tolist()],
        )
    return records


def _merge_score_records(
    shards: Sequence[dict[int, ScoreRecord]],
) -> dict[int, ScoreRecord]:
    merged: dict[int, ScoreRecord] = {}
    for shard in shards:
        for uid, record in shard.items():
            previous = merged.get(uid)
            if previous is not None and previous != record:
                raise RuntimeError(f"Duplicate CP score record for uid={uid}")
            merged[uid] = record
    return merged


def _score_bundle_from_records(
    *,
    records: dict[int, ScoreRecord],
    logical_tokens: Sequence[LogicalToken],
    logical_uids: Sequence[int],
    side: EngineSide,
    weight_state: WeightState,
    rollout_mode: RolloutMode | None,
) -> ScoreBundle:
    target_logprobs: list[float] = []
    topk: list[TokenTopK] = []
    missing: list[int] = []
    for token, uid in zip(logical_tokens, logical_uids, strict=True):
        record = records.get(uid)
        if record is None:
            missing.append(uid)
            continue
        token_id, target_logprob, topk_ids, topk_logprobs = record
        if token_id != token.token_id:
            raise RuntimeError(
                "CP score record target token does not match logical token: "
                f"uid={uid}, record={token_id}, logical={token.token_id}"
            )
        target_logprobs.append(target_logprob)
        topk.append(TokenTopK(token_ids=topk_ids, logprobs=topk_logprobs))
    if missing:
        raise RuntimeError(
            "Missing CP score records for logical tokens: "
            f"{missing[:16]} of {len(missing)} missing"
        )
    return ScoreBundle(
        side=side,
        weight_state=weight_state,
        rollout_mode=rollout_mode,
        target_logprobs=target_logprobs,
        topk=topk,
    )


def _score_context_parallel_once(
    *,
    runtime: Any,
    packed_tensors: dict[str, Any],
    logical_tokens: Sequence[LogicalToken],
    sample_id_to_row: dict[int, int] | None,
    side: EngineSide,
    weight_state: WeightState,
    rollout_mode: RolloutMode | None,
) -> ScoreBundle:
    from megatron.core import parallel_state as ps
    from megatron.core import tensor_parallel
    import torch
    import torch.distributed as dist

    dist_any = cast(Any, dist)
    from art.megatron.context_parallel.types import ParallelTopology
    from art.megatron.training.microbatches import _prepare_current_rl_micro
    from art.megatron.training.trace import (
        attach_trace_token_uids,
        prepare_replay_local_input_token_uids,
    )

    model_chunks = cast(list[Any], runtime.model)
    device = next(model_chunks[0].parameters()).device
    topology = ParallelTopology(
        tp=ps.get_tensor_model_parallel_world_size(),
        cp=ps.get_context_parallel_world_size(),
        dp=ps.get_data_parallel_world_size(),
        pp=ps.get_pipeline_model_parallel_world_size(),
        sp=bool(getattr(runtime.provider, "sequence_parallel", False)),
    )
    prepared_micro, pending = _prepare_current_rl_micro(
        cast(Any, packed_tensors),
        device=device,
        topology=topology,
        provider=runtime.provider,
        model_support_handler=runtime.model_support_handler,
        ref_logprobs=None,
        trace_token_uids=True,
        pending_prepared_micro=None,
    )
    if pending is not None:
        raise RuntimeError("CP train/inf scoring unexpectedly returned lookahead state")
    prepare_replay_local_input_token_uids(
        runtime.moe_routing_replay_controller,
        prepared_micro.local_token_uids,
        prepared_micro.attention_state,
    )
    with (
        torch.no_grad(),
        attach_trace_token_uids(
            model_chunks,
            prepared_micro.local_token_uids,
        ),
    ):
        logits = model_chunks[0](
            input_ids=prepared_micro.model_tokens,
            position_ids=prepared_micro.model_input_pos,
            attention_mask=torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device),
            labels=None,
            packed_seq_params=prepared_micro.packed_seq_params,
            **runtime.model_support_handler.get_forward_kwargs(
                model_chunks[0],
                attention_bias=prepared_micro.attention_state,
            ),
        )
    if ps.get_tensor_model_parallel_world_size() > 1:
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
    logical_uids = logical_logit_uids(
        packed_tensors=packed_tensors,
        logical_tokens=logical_tokens,
        sample_id_to_row=sample_id_to_row,
    )
    local_records: dict[int, ScoreRecord] = {}
    if ps.get_tensor_model_parallel_rank() == 0:
        local_records = _local_score_records_from_logits(
            logits=logits,
            labels=prepared_micro.model_labels,
            token_uids=prepared_micro.local_token_uids,
            desired_uids=set(logical_uids),
        )
    gathered_records: list[dict[int, ScoreRecord]] = [
        {} for _ in range(dist_any.get_world_size())
    ]
    dist_any.all_gather_object(gathered_records, local_records)
    return _score_bundle_from_records(
        records=_merge_score_records(gathered_records),
        logical_tokens=logical_tokens,
        logical_uids=logical_uids,
        side=side,
        weight_state=weight_state,
        rollout_mode=rollout_mode,
    )


def score_context_parallel_runtime(
    *,
    runtime: Any,
    packed_tensors: dict[str, Any],
    logical_map: LogicalTokenMap,
    weight_state: WeightState,
    rollout_mode: RolloutMode | None = "native_lora",
    global_grad_accumulation_sequences: int,
) -> ScoreBundle:
    from art.megatron.training.microbatches import (
        _clone_packed_tensors,
        _zero_contribution_inputs,
        build_micro_sample_indices,
        select_indexed_inputs,
        select_micro_inputs,
    )

    controller = runtime.moe_routing_replay_controller
    target_logprobs: list[float] = []
    topk: list[TokenTopK] = []
    tokens_by_sample: dict[int, list[LogicalToken]] = {}
    for token in logical_map.tokens:
        tokens_by_sample.setdefault(token.sample_id, []).append(token)
    num_sequences = int(packed_tensors["tokens"].shape[0])
    template = _clone_packed_tensors(
        select_indexed_inputs(cast(Any, packed_tensors), 0)
    )
    zero_template = _zero_contribution_inputs(template)
    num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
    for step_index in range(num_steps):
        micro_indices = build_micro_sample_indices(
            step_index=step_index,
            num_sequences=num_sequences,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        micro_inputs = select_micro_inputs(
            cast(Any, packed_tensors),
            micro_indices,
            zero_template,
        )
        if controller is not None:
            controller.set_step(
                step_index=step_index,
                sample_index=micro_indices,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
        for micro_order, (sample_index, micro_input) in enumerate(
            zip(micro_indices, micro_inputs, strict=True)
        ):
            if controller is not None:
                controller.begin_micro(sample_index, micro_order)
            sample_score = _score_context_parallel_once(
                runtime=runtime,
                packed_tensors=cast(dict[str, Any], micro_input),
                logical_tokens=(
                    []
                    if sample_index is None
                    else tokens_by_sample.get(sample_index, [])
                ),
                sample_id_to_row=({} if sample_index is None else {sample_index: 0}),
                side="megatron",
                weight_state=weight_state,
                rollout_mode=rollout_mode,
            )
            target_logprobs.extend(sample_score.target_logprobs)
            topk.extend(sample_score.topk)
        if controller is not None:
            controller.finalize_step()
    return ScoreBundle(
        side="megatron",
        weight_state=weight_state,
        rollout_mode=rollout_mode,
        target_logprobs=target_logprobs,
        topk=topk,
    )


def _extract_scores_from_logits(
    *,
    logits: Any,
    logical_map: LogicalTokenMap,
    side: EngineSide,
    weight_state: WeightState,
    rollout_mode: RolloutMode | None = None,
) -> ScoreBundle:
    import torch

    log_probs = torch.log_softmax(logits.detach().float(), dim=-1).cpu()
    target_logprobs: list[float] = []
    topk: list[TokenTopK] = []
    for token in logical_map.tokens:
        row = log_probs[token.sample_id, token.art_logit_index]
        target_logprobs.append(float(row[token.token_id].item()))
        values, indices = torch.topk(row, TOP_K)
        topk.append(
            TokenTopK(
                token_ids=[int(value) for value in indices.tolist()],
                logprobs=[float(value) for value in values.tolist()],
            )
        )
    return ScoreBundle(
        side=side,
        weight_state=weight_state,
        rollout_mode=rollout_mode,
        target_logprobs=target_logprobs,
        topk=topk,
    )
