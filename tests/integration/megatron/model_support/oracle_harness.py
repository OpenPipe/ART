from __future__ import annotations

from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from rich import box
from rich.console import Console
from rich.table import Table
import torch

from art.megatron.routing_replay import ROUTER_KEY_FORMAT_VERSION
from art.megatron.training.streaming_weight_offload import StreamingWeightOffloadConfig

from ..artifacts import GitRepoState, pinned_git_state
from ..metrics import DEFAULT_MEAN_ABS_PCT_THRESHOLD, mean_abs_pct_from_sums
from .forward_trace import ForwardTraceCapture

REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = Path(REPO_ROOT / ".local/megatron_lora_correctness")
LIVE_TRAINING_LOG_PATH = REPO_ROOT / ".local" / "live_training.log"
ORACLE_MOE_ROUTING_BUNDLE_DIRNAME = "oracle_moe_routing_replay"

REGENERATE_ENV = "ART_REGENERATE_ORACLE"
SENSITIVITY_MUTATION_ENV = "ART_SENSITIVITY_MUTATIONS"
ORACLE_OBJECTIVE_ENV = "ART_ORACLE_OBJECTIVE"
ORACLE_BASE_MODEL_ENV = "ART_ORACLE_BASE_MODEL"
KEEP_TOPOLOGY_ARTIFACTS_ENV = "ART_ORACLE_KEEP_TOPOLOGY_ARTIFACTS"
ORACLE_ARTIFACT_SUITE_NAME = "Megatron oracle artifacts"

OracleObjective = Literal["rl", "sft"]
SUPPORTED_ORACLE_OBJECTIVES: tuple[OracleObjective, ...] = ("rl", "sft")
SensitivityMutation = str
FlexBackend = Literal[
    "FLASH",
    "TRITON",
    "TRITON_LEGACY",
    "TRITON_LEGACY_INNER_FP32",
    "TRITON_LEGACY_FULL_FP32",
]
TEST_DEFAULT_FLEX_BACKEND: FlexBackend = "TRITON"

DEFAULT_SENSITIVITY_MUTATION = "skip_finalize"
CP_ATTENTION_SENSITIVITY_MUTATIONS = (
    "attn_kv_fetch_pack_on_comm_stream",
    "attn_skip_nested_grad_sanitize",
    "attn_skip_flash_lse_normalize",
)
SHARED_SENSITIVITY_MUTATIONS = (
    DEFAULT_SENSITIVITY_MUTATION,
    "fwd_skip_o_proj_tp_reduce",
    "fwd_o_proj_tp_reduce_avg_not_sum",
    "bwd_skip_sync_qkv_a",
    "bwd_skip_sync_o_proj_b",
    "bwd_skip_sync_fc1_a",
    "save_drop_nonzero_ranked_tp_shards",
    "save_duplicate_replicated_entries",
    "dp_grad_accumulation_seqs",
    *CP_ATTENTION_SENSITIVITY_MUTATIONS,
)
RL_ONLY_SENSITIVITY_MUTATIONS = ("dp_local_token_normalization",)
SFT_ONLY_SENSITIVITY_MUTATIONS = ("sft_local_token_normalization",)
SUPPORTED_SENSITIVITY_MUTATIONS = (
    *SHARED_SENSITIVITY_MUTATIONS,
    *RL_ONLY_SENSITIVITY_MUTATIONS,
    *SFT_ONLY_SENSITIVITY_MUTATIONS,
)
OBJECTIVE_SENSITIVITY_MUTATIONS: dict[OracleObjective, tuple[SensitivityMutation, ...]]
OBJECTIVE_SENSITIVITY_MUTATIONS = {
    "rl": (*SHARED_SENSITIVITY_MUTATIONS, *RL_ONLY_SENSITIVITY_MUTATIONS),
    "sft": (*SHARED_SENSITIVITY_MUTATIONS, *SFT_ONLY_SENSITIVITY_MUTATIONS),
}
REQUIRED_PACKED_TENSOR_FILES = (
    "tokens.pt",
    "group_ids.pt",
    "parent_ids.pt",
    "input_pos.pt",
    "assistant_mask.pt",
    "logprobs.pt",
    "advantages.pt",
    "weights.pt",
)
NON_FINITE_METRIC_VALUE = 1e30
ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT = DEFAULT_MEAN_ABS_PCT_THRESHOLD
ROUTER_SCORE_MEAN_ABS_PCT_LIMIT = 5e-4
FORWARD_EXPERT_LORA_TRACE_NOISE_RELATIVE_L2_LIMIT = 3e-4
FORWARD_EXPERT_LORA_TRACE_NOISE_REASON = "forward_expert_lora_trace_noise"
EXPERT_TABLE_ROW_LIMIT = 8
EXPERT_TRIPLET_PARAM_RE = re.compile(
    r"layers\.(?P<layer>\d+|__layer_avg__)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|gate_up_proj|down_proj)\."
)
LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\.")
PHASE_PRINT_ORDER = {
    "forward": 0,
    "router_scores": 1,
    "router_topk_ids": 2,
    "outputs": 3,
    "losses": 4,
    "grads": 5,
    "deltas": 6,
}


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def oracle_output_slug(
    objective: OracleObjective,
    topology: "Topology",
    suffix: str | None = None,
) -> str:
    slug = f"{objective}__{topology.slug()}"
    if suffix is not None:
        slug = f"{slug}__{suffix}"
    return slug


def supported_sensitivity_mutations_for_objective(
    objective: OracleObjective,
    *,
    is_moe: bool = True,
) -> tuple[SensitivityMutation, ...]:
    del is_moe
    return OBJECTIVE_SENSITIVITY_MUTATIONS[objective]


def objective_supports_sensitivity_mutation(
    objective: OracleObjective,
    mutation: SensitivityMutation,
    *,
    is_moe: bool = True,
) -> bool:
    return mutation in supported_sensitivity_mutations_for_objective(
        objective, is_moe=is_moe
    )


def selected_oracle_objectives() -> list[OracleObjective]:
    raw = os.environ.get(ORACLE_OBJECTIVE_ENV)
    if raw is None or raw.strip() == "":
        return list(SUPPORTED_ORACLE_OBJECTIVES)
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(SUPPORTED_ORACLE_OBJECTIVES)
    if normalized in SUPPORTED_ORACLE_OBJECTIVES:
        return [cast(OracleObjective, normalized)]
    supported = ", ".join((*SUPPORTED_ORACLE_OBJECTIVES, "all"))
    raise ValueError(
        f"Unsupported {ORACLE_OBJECTIVE_ENV} value '{raw}'. "
        f"Supported values: {supported}."
    )


def _resolve_test_flex_backend(
    case_config: "OracleCaseConfig",
    flex_backend: FlexBackend | None,
) -> FlexBackend | None:
    if flex_backend is not None:
        return flex_backend
    if case_config.precision == "fp32":
        return TEST_DEFAULT_FLEX_BACKEND
    return None


class Topology(BaseModel):
    """Defines distributed topology settings for one Megatron run variant."""

    model_config = ConfigDict(frozen=True)

    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: bool = False
    cp: int = 1
    pp: int = 1
    vpp: int = 1

    def resolved_expert_dp(self) -> int:
        """Derives expert data parallel size from topology/world-size constraints."""
        attention_world = self.tp * self.cp * self.pp * self.dp
        expert_divisor = self.etp * self.ep * self.pp
        if attention_world % expert_divisor != 0:
            raise ValueError(
                "Invalid topology for Megatron expert parallelism: "
                f"world_size={attention_world} is not divisible by "
                f"etp*ep*pp={expert_divisor}."
            )
        return attention_world // expert_divisor

    def slug(self) -> str:
        """Builds a deterministic topology identifier used for output directories."""
        return (
            f"tp{self.tp}_ep{self.ep}_etp{self.etp}"
            f"_dp{self.dp}_edp{self.resolved_expert_dp()}"
            f"_cp{self.cp}_pp{self.pp}_vpp{self.vpp}_sp{int(self.sp)}"
        )

    def world_size(self) -> int:
        # Mirrors Megatron parallel-state sizing:
        # attention side: world = tp * pp * cp * dp
        # expert side must also divide this world size (validated in resolved_expert_dp()).
        attention_world = self.tp * self.cp * self.pp * self.dp
        self.resolved_expert_dp()
        return attention_world


TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=1, sp=False),
    Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False),
    Topology(tp=2, ep=2, etp=1, dp=1, cp=2, sp=True),
    Topology(tp=2, ep=4, etp=2, dp=2, cp=2, sp=True),
]


def _without_context_parallel(topology: Topology) -> Topology:
    return topology.model_copy(update={"dp": topology.dp * topology.cp, "cp": 1})


CP_UNSUPPORTED_MOE_TOPOLOGIES = [
    _without_context_parallel(topology) for topology in TOPOLOGIES[:-1]
] + [
    Topology(tp=2, ep=2, etp=2, dp=2, cp=1, sp=True),
]
DENSE_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=1, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=1, sp=True),
    Topology(tp=1, ep=1, etp=1, dp=2, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=2, sp=True),
    Topology(tp=1, ep=1, etp=1, dp=1, cp=2, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=1, cp=2, sp=True),
    Topology(tp=2, ep=1, etp=1, dp=2, cp=2, sp=True),
]
ORACLE_TOPOLOGY = TOPOLOGIES[0]
DENSE_ORACLE_TOPOLOGY = DENSE_TOPOLOGIES[0]
SENSITIVITY_TOPOLOGY = Topology(tp=2, ep=2, etp=1, dp=1, sp=True)
CP_ATTENTION_SENSITIVITY_TOPOLOGY = Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False)
DENSE_SENSITIVITY_TOPOLOGY = Topology(tp=2, ep=1, etp=1, dp=1, sp=True)
DENSE_DP_SENSITIVITY_TOPOLOGY = Topology(tp=1, ep=1, etp=1, dp=2, sp=False)
DENSE_CP_ATTENTION_SENSITIVITY_TOPOLOGY = Topology(
    tp=1, ep=1, etp=1, dp=1, cp=2, sp=False
)
SENSITIVITY_TOPOLOGY_BY_MUTATION: dict[SensitivityMutation, Topology] = {
    mutation: SENSITIVITY_TOPOLOGY for mutation in SUPPORTED_SENSITIVITY_MUTATIONS
}
SENSITIVITY_TOPOLOGY_BY_MUTATION |= {
    mutation: CP_ATTENTION_SENSITIVITY_TOPOLOGY
    for mutation in CP_ATTENTION_SENSITIVITY_MUTATIONS
}
SENSITIVITY_TOPOLOGY_BY_MUTATION["bwd_skip_sync_fc1_a"] = Topology(
    tp=2, ep=1, etp=2, dp=1, sp=True
)
SENSITIVITY_TOPOLOGY_BY_MUTATION |= {
    k: Topology(tp=1, ep=2, etp=1, dp=2, sp=False)
    for k in [
        "dp_grad_accumulation_seqs",
        "dp_local_token_normalization",
        "sft_local_token_normalization",
    ]
}


class PackedTensorConfig(BaseModel):
    """Controls synthetic packed tensor generation used by oracle harness runs."""

    num_sequences: int = 4
    sequence_length: int = 1024
    prefill_tokens: int = 256
    completion_branches_per_prefix: int = Field(default=2, ge=1)
    decode_tokens_jitter: int = Field(default=32, ge=0)
    decode_tokens: int = 128
    packing_mode: Literal["stop_early", "truncate"] = "stop_early"
    vocab_high: int = 8192


class LoraConfig(BaseModel):
    """Configures LoRA adapter dimensions and targeted module families."""

    rank: int = 1
    alpha: int = 32
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


MetricSummary = dict[str, float]
PhasePassFn = Callable[[MetricSummary], bool]


class MetricThresholdRule(BaseModel):
    """Callable row pass rule that AND-checks configured metric upper bounds."""

    limits: dict[str, float] = Field(default_factory=dict)
    minimums: dict[str, float] = Field(default_factory=dict)

    def failure_reasons(self, summary: MetricSummary) -> list[str]:
        """Builds readable failure reasons for this threshold rule."""
        reasons: list[str] = []
        for key, limit in sorted(self.limits.items()):
            value = summary.get(key)
            if not isinstance(value, (int, float)):
                reasons.append(f"{key}=missing")
                continue
            if float(value) > float(limit):
                reasons.append(f"{key}={float(value):.6g}>{float(limit):.6g}")
        for key, minimum in sorted(self.minimums.items()):
            value = summary.get(key)
            if not isinstance(value, (int, float)):
                reasons.append(f"{key}=missing")
                continue
            if float(value) <= float(minimum):
                reasons.append(f"{key}={float(value):.6g}<={float(minimum):.6g}")
        return reasons

    def __call__(self, summary: MetricSummary) -> bool:
        """Evaluates whether the summary satisfies all configured bounds."""
        return len(self.failure_reasons(summary)) == 0


class OracleCaseConfig(BaseModel):
    """Contains all deterministic run parameters for one oracle case."""

    base_model: str
    precision: Literal["bf16", "fp32"] = "fp32"
    num_layers: int = 4
    seed: int = 20260304
    num_steps: int = 1
    grad_accumulation_sequences: int = Field(default=4, ge=1)
    learning_rate: float = 5e-6
    beta: float = 0.0
    loss_scale: float = 1
    packed_tensors: PackedTensorConfig = Field(default_factory=PackedTensorConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    allow_unvalidated_arch: bool = False

    @property
    def is_moe(self) -> bool:
        from art.megatron.model_support import model_uses_expert_parallel

        return model_uses_expert_parallel(
            self.base_model,
            allow_unvalidated_arch=self.allow_unvalidated_arch,
        )


class DiskPackedTensorsSpec(BaseModel):
    """Describes packed tensor artifacts persisted on disk for reuse."""

    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: tuple[int, list[int]] | None = None
    image_grid_thw: tuple[int, list[int]] | None = None


class CaseArtifacts(BaseModel):
    """Holds stable case-level artifact paths used by all variants."""

    case_id: str
    case_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str


class WorkerRunRequest(BaseModel):
    """Defines one distributed worker invocation for generating variant artifacts."""

    git: GitRepoState
    case_id: str
    objective: OracleObjective
    case_config: OracleCaseConfig
    topology: Topology
    topology_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str
    mutation: SensitivityMutation | None = None
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True
    capture_moe_routing_bundle_path: str | None = None
    flex_backend: FlexBackend | None = None
    offload_between_jobs: bool = True
    streaming_weight_offload: StreamingWeightOffloadConfig = Field(
        default_factory=StreamingWeightOffloadConfig
    )
    use_fp32_lora_reference: bool = True


class StepTrace(BaseModel):
    """Tracks per-step trace artifact filenames and loss metadata."""

    step_index: int
    loss: float
    probs_corr: float
    micro_sample_indices: list[int | None] = Field(default_factory=list)
    micro_losses: list[float] = Field(default_factory=list)
    debug_files: dict[str, str] = Field(default_factory=dict)
    output_file: str
    grads_file: str
    deltas_file: str
    lora_file: str


class RunManifest(BaseModel):
    """Records run metadata and per-step trace references for one topology output."""

    git: GitRepoState
    case_id: str
    objective: OracleObjective
    base_model: str
    num_layers: int
    topology: str
    world_size: int
    seed: int
    num_steps: int
    packed_tensors: DiskPackedTensorsSpec
    offload_between_jobs: bool = True
    streaming_weight_offload: StreamingWeightOffloadConfig = Field(
        default_factory=StreamingWeightOffloadConfig
    )
    use_fp32_lora_reference: bool = True
    steps: list[StepTrace]


class MetricRow(BaseModel):
    """Represents one comparable unit (param/module/global) for one phase and step."""

    case_id: str
    variant: str
    topology: str
    oracle_topology: str
    step_index: int
    phase: str
    param: str
    numel: float
    mean_abs_diff: float
    relative_l2: float
    typical_abs_scale: float
    mean_abs_pct: float
    topk_mismatch_fraction: float | None = None
    top1_mismatch_fraction: float | None = None
    pass_signal: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class VariantSpec(BaseModel):
    """Declares how to execute and evaluate one candidate variant against the oracle."""

    name: str
    objective: OracleObjective = "rl"
    topology: Topology
    pass_fn_by_phase: dict[str, PhasePassFn] = Field(
        default_factory=dict,
        repr=False,
        exclude=True,
    )
    output_slug: str | None = None
    reference_slug: str | None = None
    mutation: SensitivityMutation | None = None
    expected_signal: Literal["pass", "fail"] = "pass"
    force_regenerate: bool = True
    flex_backend: FlexBackend | None = None
    offload_between_jobs: bool = True
    streaming_weight_offload: StreamingWeightOffloadConfig = Field(
        default_factory=StreamingWeightOffloadConfig
    )

    def resolved_output_slug(self) -> str:
        """Resolves the artifact slug for this run, including mutation suffix when present."""
        if self.output_slug is not None:
            return self.output_slug
        return oracle_output_slug(self.objective, self.topology, self.mutation)

    def resolved_reference_slug(self) -> str:
        """Resolves which topology slug should be treated as the comparison oracle."""
        if self.reference_slug is not None:
            return self.reference_slug
        return oracle_output_slug(self.objective, ORACLE_TOPOLOGY)


class VariantReport(BaseModel):
    """Captures full comparison output for one variant run."""

    git: GitRepoState
    case_id: str
    variant: str
    topology: str
    reference_topology: str
    expected_signal: Literal["pass", "fail"]
    signal: Literal["pass", "fail"]
    pass_count: int
    fail_count: int
    step_summaries: dict[int, dict[str, Any]] = Field(repr=False)
    metrics: list[MetricRow] = Field(repr=False)


class DiffAccumulator:
    """Accumulates diff statistics across tensors and router-id mismatch counters."""

    def __init__(self) -> None:
        self.numel = 0
        self.abs_sum = 0.0
        self.diff_sq_sum = 0.0
        self.ref_sq_sum = 0.0
        self.ref_abs_sum = 0.0
        self.candidate_abs_sum = 0.0
        self.router_topk_total = 0
        self.router_topk_mismatch = 0
        self.router_top1_total = 0
        self.router_top1_mismatch = 0

    def update(self, reference, candidate) -> None:  # type: ignore[no-untyped-def]
        """Adds one tensor pair into the accumulator."""
        ref = reference.detach().float()
        cand = candidate.detach().float()
        diff = (cand - ref).abs()
        if diff.numel() == 0:
            return
        self.numel += int(diff.numel())
        self.abs_sum += float(diff.sum().item())
        self.diff_sq_sum += float((cand - ref).square().sum().item())
        self.ref_sq_sum += float(ref.square().sum().item())
        self.ref_abs_sum += float(ref.abs().sum().item())
        self.candidate_abs_sum += float(cand.abs().sum().item())

    @staticmethod
    def layer_averaged_summary(reference_stack, candidate_stack) -> dict[str, float]:  # type: ignore[no-untyped-def]
        """Computes normal per-layer summaries, then averages those summaries."""
        ref = reference_stack.detach().float()
        cand = candidate_stack.detach().float()
        layer_count = int(ref.shape[0])
        averaged_metrics = {
            k: 0.0
            for k in [
                "numel",
                "mean_abs_diff",
                "relative_l2",
                "typical_abs_scale",
                "candidate_abs_scale",
                "mean_abs_pct",
            ]
        }
        for layer_index in range(layer_count):
            layer_accumulator = DiffAccumulator()
            layer_accumulator.update(ref[layer_index], cand[layer_index])
            layer_summary = layer_accumulator.as_summary()
            averaged_metrics = {
                k: averaged_metrics[k] + layer_summary[k]
                for k in averaged_metrics.keys()
            }
        return {
            k: _finite_metric(averaged_metrics[k] / layer_count)
            for k in averaged_metrics.keys()
        }

    def update_router_ids(self, reference_ids, candidate_ids) -> None:  # type: ignore[no-untyped-def]
        """Adds router top-k id mismatch counts into the accumulator."""
        self.numel += int(reference_ids.numel())
        if reference_ids.ndim >= 2 and reference_ids.shape[1] > 0:
            self.router_topk_total += int(reference_ids.shape[0])
            self.router_topk_mismatch += int(
                (
                    torch.sort(reference_ids, dim=1).values
                    != torch.sort(candidate_ids, dim=1).values
                )
                .any(dim=1)
                .sum()
                .item()
            )
            self.router_top1_total += int(reference_ids.shape[0])
            self.router_top1_mismatch += int(
                (reference_ids[:, 0] != candidate_ids[:, 0]).sum().item()
            )
            return
        self.router_topk_total += int(reference_ids.numel())
        self.router_topk_mismatch += int((reference_ids != candidate_ids).sum().item())

    def as_summary(self) -> dict[str, float]:
        """Returns normalized summary values for one row."""
        if self.numel == 0:
            topk_fraction = 0.0
            top1_fraction = 0.0
        else:
            topk_fraction = (
                self.router_topk_mismatch / self.router_topk_total
                if self.router_topk_total > 0
                else 0.0
            )
            top1_fraction = (
                self.router_top1_mismatch / self.router_top1_total
                if self.router_top1_total > 0
                else 0.0
            )
        if self.numel == 0:
            return {
                "numel": 0.0,
                "mean_abs_diff": 0.0,
                "relative_l2": 0.0,
                "typical_abs_scale": 0.0,
                "candidate_abs_scale": 0.0,
                "mean_abs_pct": 0.0,
                "topk_mismatch_fraction": topk_fraction,
                "top1_mismatch_fraction": top1_fraction,
            }
        mean_abs = self.abs_sum / self.numel
        typical_abs = self.ref_abs_sum / self.numel
        candidate_abs = self.candidate_abs_sum / self.numel
        return {
            "numel": _finite_metric(float(self.numel), default=0.0),
            "mean_abs_diff": _finite_metric(mean_abs),
            "relative_l2": _finite_metric(
                (self.diff_sq_sum**0.5) / max(self.ref_sq_sum**0.5, 1e-12)
            ),
            "typical_abs_scale": _finite_metric(typical_abs, default=0.0),
            "candidate_abs_scale": _finite_metric(candidate_abs, default=0.0),
            "mean_abs_pct": _finite_metric(
                mean_abs_pct_from_sums(self.abs_sum, self.ref_abs_sum, self.numel)
            ),
            "topk_mismatch_fraction": _finite_metric(topk_fraction, default=1.0),
            "top1_mismatch_fraction": _finite_metric(top1_fraction, default=1.0),
        }


T = TypeVar("T")


def _require_not_none(value: T | None, name: str) -> T:
    """Asserts non-None values for required artifacts and raises a named runtime error."""
    if value is None:
        raise RuntimeError(f"{name} is None")
    return value


def _truthy(value: str | None) -> bool:
    """Parses env-var style booleans using a small accepted truthy set."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sensitivity_mutations() -> list[SensitivityMutation]:
    """Parses sensitivity mutation selectors from env as a CSV list."""
    raw = os.environ.get(SENSITIVITY_MUTATION_ENV)
    if raw is None or raw.strip() == "":
        return []
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(SUPPORTED_SENSITIVITY_MUTATIONS)
    if normalized in {"1", "true", "yes", "on"}:
        return [DEFAULT_SENSITIVITY_MUTATION]
    mutations = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unsupported = [
        mutation
        for mutation in mutations
        if mutation not in SUPPORTED_SENSITIVITY_MUTATIONS
    ]
    if not unsupported:
        return mutations
    supported = ", ".join(SUPPORTED_SENSITIVITY_MUTATIONS)
    raise ValueError(
        f"Unsupported {SENSITIVITY_MUTATION_ENV} value '{raw}'. "
        f"Supported values: {supported}, CSV of supported values, all, 1/true/yes/on."
    )


def sensitivity_enabled() -> bool:
    """Returns whether any sensitivity mutation has been requested via environment."""
    return bool(sensitivity_mutations())


def selected_sensitivity_mutations_for_objective(
    objective: OracleObjective,
    mutations: list[SensitivityMutation],
    *,
    is_moe: bool = True,
) -> list[SensitivityMutation]:
    return [
        mutation
        for mutation in mutations
        if mutation
        in supported_sensitivity_mutations_for_objective(objective, is_moe=is_moe)
    ]


def sensitivity_topology_for_mutation(
    mutation: SensitivityMutation,
    *,
    is_moe: bool = True,
) -> Topology:
    """Returns the sensitivity topology required for one mutation."""
    if not is_moe:
        if mutation in {
            "dp_grad_accumulation_seqs",
            "dp_local_token_normalization",
            "sft_local_token_normalization",
        }:
            return DENSE_DP_SENSITIVITY_TOPOLOGY
        if mutation in CP_ATTENTION_SENSITIVITY_MUTATIONS:
            return DENSE_CP_ATTENTION_SENSITIVITY_TOPOLOGY
        return DENSE_SENSITIVITY_TOPOLOGY
    return SENSITIVITY_TOPOLOGY_BY_MUTATION[mutation]


def sensitivity_required_world_size(
    mutations: list[SensitivityMutation],
    *,
    is_moe: bool = True,
) -> int:
    """Returns the max world-size required by a selected mutation set."""
    return max(
        sensitivity_topology_for_mutation(mutation, is_moe=is_moe).world_size()
        for mutation in mutations
    )


def regenerate_requested() -> bool:
    """Returns whether regeneration mode is enabled for oracle artifacts."""
    return _truthy(os.environ.get(REGENERATE_ENV))


def keep_topology_artifacts() -> bool:
    """Returns whether oracle topology tensor artifacts should be retained."""
    return _truthy(os.environ.get(KEEP_TOPOLOGY_ARTIFACTS_ENV))


DEFAULT_ORACLE_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"


def case_config(base_model: str | None = None) -> OracleCaseConfig:
    """Builds the deterministic default oracle case config."""
    return OracleCaseConfig(
        base_model=base_model
        or os.environ.get(ORACLE_BASE_MODEL_ENV, DEFAULT_ORACLE_BASE_MODEL)
    )


def available_gpu_count() -> int:
    """Reports visible CUDA device count for topology scheduling and test skips."""
    import torch

    return int(torch.cuda.device_count())


def oracle_topology(*, is_moe: bool = True) -> Topology:
    """Returns the canonical single-rank oracle topology for a model family."""
    return ORACLE_TOPOLOGY if is_moe else DENSE_ORACLE_TOPOLOGY


def _filter_context_parallel_support(
    topologies: list[Topology],
    *,
    is_moe: bool,
    cp_supported: bool,
) -> list[Topology]:
    if cp_supported:
        return topologies
    if is_moe:
        return list(CP_UNSUPPORTED_MOE_TOPOLOGIES)
    return [_without_context_parallel(topology) for topology in topologies]


def selected_suite_topologies(
    *,
    is_moe: bool = True,
    cp_supported: bool = True,
) -> list[Topology]:
    """Returns the correctness topology list for a model family."""
    return _filter_context_parallel_support(
        list(TOPOLOGIES if is_moe else DENSE_TOPOLOGIES),
        is_moe=is_moe,
        cp_supported=cp_supported,
    )


def stable_case_id(case_config: OracleCaseConfig) -> str:
    """Builds a deterministic case id from case config contents."""
    payload = case_config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    model_tag = (
        case_config.base_model.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )
    return f"{model_tag}_{digest}"


def _write_json(path: Path, payload: Any) -> None:
    """Writes canonical pretty JSON to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    """Loads a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _current_git_state() -> GitRepoState:
    return pinned_git_state(ORACLE_ARTIFACT_SUITE_NAME)


def _manifest_matches_current_commit(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    git_payload = payload.get("git")
    return (
        isinstance(git_payload, dict)
        and git_payload.get("commit") == _current_git_state().commit
    )


def _build_packed_tensors(
    config: PackedTensorConfig,
    seed: int,
) -> dict[str, Any]:
    """Generates deterministic synthetic packed tensors used in integration runs."""
    import torch

    if config.num_sequences <= 1:
        raise ValueError("num_sequences must be greater than 1")
    shape = (config.num_sequences, config.sequence_length)
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(shape, dtype=torch.long)
    token_low = 10
    token_span = max(1, config.vocab_high - token_low)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    input_pos = torch.zeros(shape, dtype=torch.long)
    logprobs = torch.full(shape, float("nan"), dtype=torch.float32)
    advantages = torch.zeros(shape, dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    prefix_length = max(1, min(config.sequence_length - 1, config.prefill_tokens))
    max_completion_tokens = max(1, config.sequence_length - prefix_length)
    base_completion_tokens = max(1, min(config.decode_tokens, max_completion_tokens))
    jitter_width = min(config.decode_tokens_jitter, max_completion_tokens - 1)

    def _sample_completion_length() -> int:
        if jitter_width > 0:
            jitter = int(
                torch.randint(
                    low=-jitter_width,
                    high=jitter_width + 1,
                    size=(1,),
                    generator=generator,
                    dtype=torch.long,
                ).item()
            )
        else:
            jitter = 0
        return max(
            1,
            min(max_completion_tokens, base_completion_tokens + jitter),
        )

    def _sample_token_block(length: int) -> torch.Tensor:
        return torch.randint(
            low=token_low,
            high=config.vocab_high,
            size=(length,),
            dtype=torch.long,
            generator=generator,
        )

    def _sample_logprob_block(length: int) -> torch.Tensor:
        return (
            torch.randn(
                (length,),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
            - 1.75
        )

    def _sample_advantage_value() -> float:
        return float(
            (
                torch.randn(
                    (1,),
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.5
            ).item()
        )

    for sequence_index in range(config.num_sequences):
        cursor = 0
        next_group_id = 0
        while cursor < config.sequence_length:
            prompt_group_id = next_group_id
            next_group_id += 1
            completion_lengths = [
                _sample_completion_length()
                for _ in range(config.completion_branches_per_prefix)
            ]
            remaining = config.sequence_length - cursor

            if config.packing_mode == "stop_early":
                included_completion_lengths = list(completion_lengths)
                while (
                    included_completion_lengths
                    and (prefix_length + sum(included_completion_lengths)) > remaining
                ):
                    included_completion_lengths.pop()
                if not included_completion_lengths:
                    break

                prompt_end = cursor + prefix_length
                tokens[sequence_index, cursor:prompt_end] = _sample_token_block(
                    prefix_length
                )
                group_ids[sequence_index, cursor:prompt_end] = prompt_group_id
                parent_ids[sequence_index, cursor:prompt_end] = prompt_group_id
                input_pos[sequence_index, cursor:prompt_end] = torch.arange(
                    prefix_length, dtype=torch.long
                )
                cursor = prompt_end

                for completion_length in included_completion_lengths:
                    completion_group_id = next_group_id
                    next_group_id += 1
                    completion_end = cursor + completion_length
                    tokens[sequence_index, cursor:completion_end] = _sample_token_block(
                        completion_length
                    )
                    group_ids[sequence_index, cursor:completion_end] = (
                        completion_group_id
                    )
                    parent_ids[sequence_index, cursor:completion_end] = prompt_group_id
                    input_pos[sequence_index, cursor:completion_end] = torch.arange(
                        prefix_length,
                        prefix_length + completion_length,
                        dtype=torch.long,
                    )
                    assistant_mask[sequence_index, cursor:completion_end] = True
                    logprobs[sequence_index, cursor:completion_end] = (
                        _sample_logprob_block(completion_length)
                    )
                    advantages[sequence_index, cursor:completion_end] = (
                        _sample_advantage_value()
                    )
                    weights[sequence_index, cursor:completion_end] = 1.0
                    cursor = completion_end
                continue

            prompt_take = min(prefix_length, remaining)
            prompt_end = cursor + prompt_take
            tokens[sequence_index, cursor:prompt_end] = _sample_token_block(prompt_take)
            group_ids[sequence_index, cursor:prompt_end] = prompt_group_id
            parent_ids[sequence_index, cursor:prompt_end] = prompt_group_id
            input_pos[sequence_index, cursor:prompt_end] = torch.arange(
                prompt_take, dtype=torch.long
            )
            cursor = prompt_end
            if cursor >= config.sequence_length:
                break

            for completion_length in completion_lengths:
                if cursor >= config.sequence_length:
                    break
                completion_group_id = next_group_id
                next_group_id += 1
                remaining = config.sequence_length - cursor
                completion_take = min(completion_length, remaining)
                completion_end = cursor + completion_take
                tokens[sequence_index, cursor:completion_end] = _sample_token_block(
                    completion_take
                )
                group_ids[sequence_index, cursor:completion_end] = completion_group_id
                parent_ids[sequence_index, cursor:completion_end] = prompt_group_id
                input_pos[sequence_index, cursor:completion_end] = torch.arange(
                    prefix_length,
                    prefix_length + completion_take,
                    dtype=torch.long,
                )
                assistant_mask[sequence_index, cursor:completion_end] = True
                logprobs[sequence_index, cursor:completion_end] = _sample_logprob_block(
                    completion_take
                )
                advantages[sequence_index, cursor:completion_end] = (
                    _sample_advantage_value()
                )
                weights[sequence_index, cursor:completion_end] = 1.0
                cursor = completion_end
                if completion_take < completion_length:
                    break

    # Ensure paired cross-DP rows are never token-identical across valid tokens.
    half = config.num_sequences // 2
    if half > 0 and config.num_sequences % 2 == 0:
        valid_lengths = (group_ids != -1).sum(dim=1)
        for pair_index in range(half):
            left_index = pair_index
            right_index = pair_index + half
            left_valid = int(valid_lengths[left_index].item())
            right_valid = int(valid_lengths[right_index].item())
            if left_valid != right_valid or left_valid == 0:
                continue
            if torch.equal(
                tokens[left_index, :left_valid], tokens[right_index, :right_valid]
            ):
                tokens[right_index, 0] = (
                    (tokens[right_index, 0] - token_low + 1) % token_span
                ) + token_low
    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * config.num_sequences,
        "image_grid_thw": [None] * config.num_sequences,
    }


def _create_packed_tensors(
    case_config: OracleCaseConfig,
    packed_dir: Path,
) -> DiskPackedTensorsSpec:
    """Persists packed tensors to disk and returns their descriptor."""
    from art.preprocessing.pack import PackedTensors, packed_tensors_to_dir

    packed_tensors = cast(
        PackedTensors,
        _build_packed_tensors(case_config.packed_tensors, case_config.seed),
    )
    descriptor = packed_tensors_to_dir(packed_tensors, str(packed_dir))
    return DiskPackedTensorsSpec.model_validate(descriptor)


def ensure_case_artifacts(case_config: OracleCaseConfig) -> CaseArtifacts:
    """Ensures stable case-level artifacts (input tensors) are present and reusable."""
    case_id = stable_case_id(case_config)
    case_dir = ARTIFACT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "case_config.json", case_config.model_dump(mode="json"))
    regenerate = regenerate_requested()

    descriptor_path = case_dir / "packed_tensors.json"
    packed_dir = case_dir / "packed_tensors"
    if descriptor_path.exists() and not regenerate:
        packed_spec = DiskPackedTensorsSpec.model_validate(_read_json(descriptor_path))
    else:
        if packed_dir.exists():
            shutil.rmtree(packed_dir)
        packed_spec = _create_packed_tensors(case_config, packed_dir)
        _write_json(descriptor_path, packed_spec.model_dump(mode="json"))

    shared_init_path = case_dir / "shared_init" / "adapter_model.safetensors"
    shared_init_path.parent.mkdir(parents=True, exist_ok=True)
    return CaseArtifacts(
        case_id=case_id,
        case_dir=str(case_dir),
        packed_tensors=packed_spec,
        shared_init_adapter_path=str(shared_init_path),
    )


def _replace_topology_dir(path: Path) -> None:
    """Resets one topology output directory before regeneration."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "traces").mkdir(parents=True, exist_ok=True)


def _prune_topology_artifacts(path: Path) -> None:
    """Keeps small diagnostics and removes tensors that are only needed for comparison."""
    if keep_topology_artifacts() or not path.exists():
        return
    for child in path.iterdir():
        if child.name in {
            "manifest.json",
            "variant_report.json",
            "run_request.json",
            "worker.log",
        }:
            continue
        if child.is_dir():
            shutil.rmtree(child)
            continue
        child.unlink()


def _prune_case_artifacts(case_dir: Path) -> None:
    """Drops reusable generated inputs after tests have written reports."""
    if keep_topology_artifacts() or not case_dir.exists():
        return
    for name in ("packed_tensors", "packed_tensors.json", "shared_init"):
        path = case_dir / name
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _load_manifest(topology_dir: Path) -> RunManifest:
    """Loads one run manifest for a topology output directory."""
    manifest_path = topology_dir / "manifest.json"
    return RunManifest.model_validate(_read_json(manifest_path))


def _load_output_tensor(topology_dir: Path, step: StepTrace):
    """Loads one output trace tensor referenced by a step trace entry."""
    import torch

    path = topology_dir / step.output_file
    return torch.load(path, map_location="cpu")


def _load_safetensor_map(path: Path) -> dict[str, Any]:
    """Loads one safetensor map from disk."""
    from safetensors.torch import load_file  # ty: ignore[unresolved-import]

    return load_file(str(path))


def _align_sequence_parallel(reference, candidate):  # type: ignore[no-untyped-def]
    """Aligns sequence-parallel-shaped tensors so diff computation is topology-agnostic."""
    if reference.shape == candidate.shape:
        return candidate
    if (
        candidate.ndim == reference.ndim + 1
        and candidate.shape[0] * candidate.shape[1] == reference.shape[0]
        and tuple(candidate.shape[2:]) == tuple(reference.shape[1:])
    ):
        return candidate.reshape(reference.shape)
    return None


def _load_forward_trace(
    topology_dir: Path, step_index: int
) -> dict[str, list[dict[str, Any]]]:
    """Loads one merged forward-trace file for a given step."""
    trace_path = topology_dir / "traces" / f"forward_trace_step_{step_index:03d}.pt"
    return ForwardTraceCapture.load_trace(trace_path)


def _finite_metric(value: float, *, default: float = NON_FINITE_METRIC_VALUE) -> float:
    """Maps NaN/Inf metric values to a large finite sentinel for JSON-safe reports."""
    value_f = float(value)
    if math.isnan(value_f):
        return default
    if math.isinf(value_f):
        return default if value_f > 0 else -default
    return value_f


def _triplet_expert_key(param: str) -> tuple[str, int] | None:
    """Returns (projection, expert_id) for expert gate/up/down params."""
    match = EXPERT_TRIPLET_PARAM_RE.search(param)
    if match is None:
        return None
    return match.group("proj"), int(match.group("expert"))


def _layer_agnostic_param_key(param: str) -> str | None:
    """Normalizes one parameter name by stripping the explicit layer index."""
    if LAYER_INDEX_RE.search(param) is None:
        return None
    return LAYER_INDEX_RE.sub("layers.__layer_avg__.", param, count=1)


def _expert_agnostic_param_key(param: str) -> str:
    """Normalizes expert-triplet params by stripping the explicit expert index."""
    match = EXPERT_TRIPLET_PARAM_RE.search(param)
    if match is None:
        return param
    start, end = match.span("expert")
    return f"{param[:start]}__expert_avg__{param[end:]}"


def _is_forward_expert_lora_trace(param: str) -> bool:
    """Returns whether one forward-trace row is an expert LoRA internal."""
    return ".mlp.experts." in param and (
        ".lora." in param or ".gate_lora." in param or ".up_lora." in param
    )


def _stacked_layers(
    pairs: list[tuple[str, Any, Any]],
) -> list[tuple[str, Any, Any]]:
    """Builds layer-stacked tensor pairs keyed without explicit layer index."""
    import torch

    grouped: dict[str, list[tuple[Any, Any]]] = {}
    original_names_by_group: dict[str, list[str]] = {}
    for name, reference, candidate in pairs:
        normalized = _layer_agnostic_param_key(name)
        if normalized is None:
            raise RuntimeError(
                f"Expected all compared params to include a layer index, got '{name}'."
            )
        grouped.setdefault(normalized, []).append(
            (reference.detach().float(), candidate.detach().float())
        )
        original_names_by_group.setdefault(normalized, []).append(name)

    stacked_pairs: list[tuple[str, Any, Any]] = []
    for normalized in sorted(grouped):
        group = grouped[normalized]
        reference_shapes = {tuple(reference.shape) for reference, _ in group}
        candidate_shapes = {tuple(candidate.shape) for _, candidate in group}
        if len(reference_shapes) != 1 or len(candidate_shapes) != 1:
            original_names = original_names_by_group[normalized]
            for original_name, (reference, candidate) in zip(original_names, group):
                # Keep one synthetic layer axis so layer-averaged comparison
                # does not treat tensor rows/features as layer entries.
                stacked_pairs.append(
                    (original_name, reference.unsqueeze(0), candidate.unsqueeze(0))
                )
            continue
        stacked_pairs.append(
            (
                normalized,
                torch.stack([reference for reference, _ in group], dim=0),
                torch.stack([candidate for _, candidate in group], dim=0),
            )
        )
    return stacked_pairs


class VariantRunner:
    """Runs oracle/candidate variants and emits row-level comparison reports."""

    def __init__(
        self,
        *,
        objective: OracleObjective = "rl",
        case_config: OracleCaseConfig,
        oracle_flex_backend: FlexBackend | None = None,
        variant_flex_backend: FlexBackend | None = None,
        oracle_topology_override: Topology | None = None,
        oracle_slug_override: str | None = None,
        oracle_offload_between_jobs: bool = True,
        oracle_streaming_weight_offload: StreamingWeightOffloadConfig | None = None,
        use_fp32_lora_reference: bool = True,
        console: Console | None = None,
    ) -> None:
        self.objective = objective
        self.case_config = case_config
        self.git = _current_git_state()
        self.case_artifacts = ensure_case_artifacts(case_config)
        self.case_id = self.case_artifacts.case_id
        self.case_dir = Path(self.case_artifacts.case_dir)
        self.oracle_topology = oracle_topology_override or oracle_topology(
            is_moe=case_config.is_moe
        )
        self.oracle_slug = oracle_slug_override or oracle_output_slug(
            objective, self.oracle_topology
        )
        self.oracle_dir = self.case_dir / self.oracle_slug
        self.oracle_routing_bundle_dir = (
            self.case_dir / f"{objective}__{ORACLE_MOE_ROUTING_BUNDLE_DIRNAME}"
        )
        self.oracle_offload_between_jobs = oracle_offload_between_jobs
        self.oracle_streaming_weight_offload = (
            oracle_streaming_weight_offload or StreamingWeightOffloadConfig()
        )
        self.use_fp32_lora_reference = use_fp32_lora_reference
        self.shared_init_path = Path(self.case_artifacts.shared_init_adapter_path)
        self.oracle_flex_backend = _resolve_test_flex_backend(
            case_config, oracle_flex_backend
        )
        self.variant_flex_backend = _resolve_test_flex_backend(
            case_config, variant_flex_backend
        )
        self.console = console or Console(width=140)
        self._oracle_initialized = False
        self._oracle_regenerated = False
        self._sample_valid_lengths_cache: tuple[int, ...] | None = None

    def _sample_valid_lengths(self) -> tuple[int, ...]:
        if self._sample_valid_lengths_cache is not None:
            return self._sample_valid_lengths_cache
        from art.megatron.context_parallel.builder import (
            build_shared_prefix_attention_spec,
        )
        from art.preprocessing.pack import packed_tensors_from_dir

        packed_tensors = packed_tensors_from_dir(
            **self.case_artifacts.packed_tensors.model_dump(exclude_none=True)
        )
        group_ids = packed_tensors["group_ids"]
        parent_ids = packed_tensors["parent_ids"]
        self._sample_valid_lengths_cache = tuple(
            int(
                build_shared_prefix_attention_spec(
                    group_ids=group_ids[row_index : row_index + 1],
                    parent_ids=parent_ids[row_index : row_index + 1],
                )
                .rows[0]
                .valid_tokens
            )
            for row_index in range(int(group_ids.shape[0]))
        )
        return self._sample_valid_lengths_cache

    def _step_micro_sample_indices(self, step: StepTrace) -> list[int | None]:
        base_sample_index = (
            step.step_index * self.case_config.grad_accumulation_sequences
        )
        expected = [
            sample_index
            if sample_index < self.case_artifacts.packed_tensors.num_sequences
            else None
            for sample_index in range(
                base_sample_index,
                base_sample_index + self.case_config.grad_accumulation_sequences,
            )
        ]
        if step.micro_sample_indices and len(step.micro_sample_indices) == len(
            expected
        ):
            return list(step.micro_sample_indices)
        return expected

    def _load_output_tensor_map(
        self,
        topology_dir: Path,
        step: StepTrace,
    ) -> dict[str, torch.Tensor]:
        tensor = _load_output_tensor(topology_dir, step)
        if isinstance(tensor, list):
            outputs = tensor
        elif isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            outputs = [tensor[index] for index in range(int(tensor.shape[0]))]
        else:
            return {"logprobs": tensor}

        sample_indices = self._step_micro_sample_indices(step)
        valid_lengths = self._sample_valid_lengths()
        output_map: dict[str, torch.Tensor] = {}
        for output_index, output in enumerate(outputs):
            key = f"logprobs.micro_{output_index:03d}"
            if not isinstance(output, torch.Tensor):
                output_map[key] = output
                continue
            if output_index < len(sample_indices):
                sample_index = sample_indices[output_index]
                if isinstance(sample_index, int):
                    valid_length = int(valid_lengths[sample_index])
                    target_length = max(valid_length - 1, 0)
                    if output.ndim > 0 and int(output.shape[-1]) > target_length:
                        output = output[..., :target_length].contiguous()
            output_map[key] = output
        return output_map

    @staticmethod
    def _load_loss_tensor_map(step: StepTrace) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor([step.loss], dtype=torch.float32)}

    def _trim_trace_padding(
        self,
        trace: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        valid_lengths = self._sample_valid_lengths()
        sequence_length = int(self.case_config.packed_tensors.sequence_length)
        if sequence_length <= 0:
            return trace

        for calls in trace.values():
            for call in calls:
                sample_index = call.get("micro_sample_index")
                if not isinstance(sample_index, int):
                    continue
                valid_length = int(valid_lengths[sample_index])
                if valid_length >= sequence_length:
                    continue
                row_token_uids = call.get("row_token_uids")
                if (
                    isinstance(row_token_uids, torch.Tensor)
                    and row_token_uids.ndim == 1
                ):
                    local_token_uids = torch.remainder(row_token_uids, sequence_length)
                    keep_rows = torch.nonzero(
                        (row_token_uids >= 0) & (local_token_uids < valid_length),
                        as_tuple=False,
                    ).reshape(-1)
                    if int(keep_rows.numel()) > 0 and int(keep_rows.numel()) < int(
                        row_token_uids.numel()
                    ):
                        call["row_token_uids"] = row_token_uids.index_select(
                            0, keep_rows
                        ).contiguous()
                        for key in (
                            "primary_output",
                            "router_topk_scores",
                            "router_topk_ids",
                        ):
                            tensor = call.get(key)
                            if (
                                isinstance(tensor, torch.Tensor)
                                and tensor.ndim > 0
                                and int(tensor.shape[0]) == int(row_token_uids.numel())
                            ):
                                call[key] = tensor.index_select(
                                    0, keep_rows
                                ).contiguous()
                        continue
                for key in ("primary_output", "router_topk_scores", "router_topk_ids"):
                    tensor = call.get(key)
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                        continue
                    leading_dim = int(tensor.shape[0])
                    if leading_dim <= valid_length:
                        continue
                    if leading_dim % sequence_length == 0:
                        row_multiplier = leading_dim // sequence_length
                        target_rows = valid_length * row_multiplier
                    elif leading_dim <= sequence_length:
                        target_rows = valid_length
                    else:
                        continue
                    if 0 < target_rows < leading_dim:
                        call[key] = tensor[:target_rows].contiguous()
        return trace

    def _run_topology(
        self,
        *,
        topology: Topology,
        output_slug: str,
        mutation: SensitivityMutation | None,
        replay_bundle_dir: Path | None,
        capture_bundle_dir: Path | None,
        regenerate: bool,
        flex_backend: FlexBackend | None = None,
        offload_between_jobs: bool = True,
        streaming_weight_offload: StreamingWeightOffloadConfig | None = None,
    ) -> Path:
        """Executes one topology worker run and returns its output directory."""
        topology_dir = self.case_dir / output_slug
        manifest_path = topology_dir / "manifest.json"
        if (
            manifest_path.exists()
            and not regenerate
            and _manifest_matches_current_commit(manifest_path)
        ):
            return topology_dir
        _replace_topology_dir(topology_dir)
        run_case_config = self.case_config
        request = WorkerRunRequest(
            git=self.git,
            case_id=self.case_id,
            objective=self.objective,
            case_config=run_case_config,
            topology=topology,
            topology_dir=str(topology_dir),
            packed_tensors=self.case_artifacts.packed_tensors,
            shared_init_adapter_path=str(self.shared_init_path),
            mutation=mutation,
            moe_routing_replay_path=(
                None if replay_bundle_dir is None else str(replay_bundle_dir)
            ),
            moe_routing_replay_strict=True,
            capture_moe_routing_bundle_path=(
                None if capture_bundle_dir is None else str(capture_bundle_dir)
            ),
            flex_backend=flex_backend,
            offload_between_jobs=offload_between_jobs,
            streaming_weight_offload=(
                streaming_weight_offload or StreamingWeightOffloadConfig()
            ),
            use_fp32_lora_reference=self.use_fp32_lora_reference,
        )
        from .oracle_worker import run_worker_subprocess

        run_worker_subprocess(request, topology_dir, repo_root=REPO_ROOT)
        return topology_dir

    def ensure_oracle(self) -> Path:
        """Ensures routing capture and the canonical replay-backed oracle exist once."""
        regenerate = regenerate_requested()
        if self._oracle_initialized and (not regenerate or self._oracle_regenerated):
            return self.oracle_dir
        if regenerate and self.shared_init_path.exists():
            self.shared_init_path.unlink()
        bundle_manifest = self.oracle_routing_bundle_dir / "manifest.json"
        oracle_manifest = self.oracle_dir / "manifest.json"
        capture_manifest = (
            self.case_dir / f"{self.oracle_slug}__oracle_capture" / "manifest.json"
        )
        bundle_format_current = False
        if bundle_manifest.exists():
            try:
                bundle_format_current = (
                    _read_json(bundle_manifest).get("format_version")
                    == ROUTER_KEY_FORMAT_VERSION
                )
            except Exception:
                bundle_format_current = False
        need_capture = (
            regenerate
            or not bundle_manifest.exists()
            or not bundle_format_current
            or not self.shared_init_path.exists()
            or not _manifest_matches_current_commit(capture_manifest)
        )
        run_oracle_topology = partial(
            self._run_topology,
            topology=self.oracle_topology,
            mutation=None,
            flex_backend=self.oracle_flex_backend,
            offload_between_jobs=self.oracle_offload_between_jobs,
            streaming_weight_offload=self.oracle_streaming_weight_offload,
            regenerate=True,
        )
        if self.case_config.is_moe and need_capture:
            run_oracle_topology(
                output_slug=f"{self.oracle_slug}__oracle_capture",
                replay_bundle_dir=None,
                capture_bundle_dir=self.oracle_routing_bundle_dir,
            )
        if (
            regenerate
            or not oracle_manifest.exists()
            or not self.shared_init_path.exists()
            or not _manifest_matches_current_commit(oracle_manifest)
        ):
            run_oracle_topology(
                output_slug=self.oracle_slug,
                replay_bundle_dir=(
                    self.oracle_routing_bundle_dir if self.case_config.is_moe else None
                ),
                capture_bundle_dir=None,
            )
        self._oracle_initialized = True
        self._oracle_regenerated = self._oracle_regenerated or regenerate
        return self.oracle_dir

    def ensure_variant_artifacts(
        self,
        variant: VariantSpec,
    ) -> Path:
        """Ensures oracle prerequisites and candidate artifacts for one variant."""
        self.ensure_oracle()
        output_slug = variant.resolved_output_slug()
        if output_slug == self.oracle_slug and variant.mutation is None:
            return self.oracle_dir
        return self._run_topology(
            topology=variant.topology,
            output_slug=output_slug,
            mutation=variant.mutation,
            flex_backend=variant.flex_backend or self.variant_flex_backend,
            offload_between_jobs=variant.offload_between_jobs,
            streaming_weight_offload=variant.streaming_weight_offload,
            replay_bundle_dir=(
                self.oracle_routing_bundle_dir if self.case_config.is_moe else None
            ),
            capture_bundle_dir=None,
            regenerate=variant.force_regenerate,
        )

    @staticmethod
    def _apply_phase_pass(
        *,
        row: MetricRow,
        phase: str,
        summary: MetricSummary,
        pass_fn_by_phase: dict[str, PhasePassFn],
    ) -> None:
        """Evaluates a per-phase pass function against one summary payload."""
        pass_fn = pass_fn_by_phase.get(phase)
        if pass_fn is None:
            row.pass_signal = True
            row.failure_reasons = []
            return
        row.pass_signal = bool(pass_fn(summary))
        if row.pass_signal:
            row.failure_reasons = []
            return
        explain = getattr(pass_fn, "failure_reasons", None)
        if callable(explain):
            reasons = explain(summary)
            row.failure_reasons = (
                reasons if reasons else ["phase pass function returned false"]
            )
            return
        row.failure_reasons = ["phase pass function returned false"]

    @staticmethod
    def _inf_summary() -> dict[str, float]:
        """Builds a large-error finite summary for structural mismatches."""
        return {
            "numel": 0.0,
            "mean_abs_diff": NON_FINITE_METRIC_VALUE,
            "relative_l2": NON_FINITE_METRIC_VALUE,
            "typical_abs_scale": 0.0,
            "candidate_abs_scale": 0.0,
            "mean_abs_pct": NON_FINITE_METRIC_VALUE,
            "topk_mismatch_fraction": 1.0,
            "top1_mismatch_fraction": 1.0,
        }

    def _build_metric_row(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        param: str,
        summary: dict[str, float],
        structural_failure: str | None = None,
    ) -> MetricRow:
        """Builds one metric row and applies per-phase pass evaluation."""
        row = MetricRow(
            case_id=self.case_id,
            variant=variant.name,
            topology=variant.resolved_output_slug(),
            oracle_topology=variant.resolved_reference_slug(),
            step_index=step_index,
            phase=phase,
            param=param,
            numel=summary["numel"],
            mean_abs_diff=summary["mean_abs_diff"],
            relative_l2=summary["relative_l2"],
            typical_abs_scale=summary["typical_abs_scale"],
            mean_abs_pct=summary["mean_abs_pct"],
            topk_mismatch_fraction=summary.get("topk_mismatch_fraction"),
            top1_mismatch_fraction=summary.get("top1_mismatch_fraction"),
        )
        self._apply_phase_pass(
            row=row,
            phase=phase,
            summary=summary,
            pass_fn_by_phase=variant.pass_fn_by_phase,
        )
        if phase in {"grads", "deltas"} and _triplet_expert_key(param) is not None:
            row.pass_signal = True
            row.failure_reasons = []
        if structural_failure is not None:
            row.pass_signal = False
            row.failure_reasons = [structural_failure, *row.failure_reasons]
        return row

    def _build_metric_rows_from_tensor_pairs(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        pairs: list[tuple[str, Any, Any]],
        router_ids: bool = False,
        layer_averaged: bool = False,
    ) -> list[MetricRow]:
        """Builds rows from named tensor pairs with one shared diff path."""
        rows: list[MetricRow] = []
        for name, reference, candidate in pairs:
            reference_aligned = reference
            candidate_aligned = candidate
            aligned_candidate = _align_sequence_parallel(
                reference_aligned, candidate_aligned
            )
            if aligned_candidate is None:
                rows.append(
                    self._build_metric_row(
                        variant=variant,
                        step_index=step_index,
                        phase=phase,
                        param=name,
                        summary=self._inf_summary(),
                        structural_failure="shape mismatch",
                    )
                )
                continue
            summary: dict[str, float]
            if router_ids:
                accumulator = DiffAccumulator()
                accumulator.update_router_ids(reference_aligned, aligned_candidate)
                summary = accumulator.as_summary()
            elif layer_averaged:
                summary = DiffAccumulator.layer_averaged_summary(
                    reference_aligned,
                    aligned_candidate,
                )
            else:
                accumulator = DiffAccumulator()
                accumulator.update(reference_aligned, aligned_candidate)
                summary = accumulator.as_summary()
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=step_index,
                    phase=phase,
                    param=name,
                    summary=summary,
                )
            )
        return rows

    def _check_matching_keys(
        self,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        variant: VariantSpec,
        step_index: int,
        phase: str,
    ) -> tuple[bool, list[MetricRow] | None]:
        """Checks if the keys of two tensor maps match and builds a metric row if they don't."""
        reference_keys = set(reference.keys())
        candidate_keys = set(candidate.keys())
        if reference_keys != candidate_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            return False, [
                self._build_metric_row(
                    variant=variant,
                    step_index=step_index,
                    phase=phase,
                    param="__keys__",
                    summary=self._inf_summary(),
                    structural_failure=f"missing={missing[:5]} extra={extra[:5]}",
                )
            ]
        return True, None

    def _build_metric_rows_from_tensor_maps(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        router_ids: bool = False,
    ) -> list[MetricRow]:
        """Builds rows from two keyed tensor maps through a unified compare path."""
        matching, rows = self._check_matching_keys(
            reference, candidate, variant, step_index, phase
        )
        if not matching:
            return rows if rows is not None else []
        pairs = [
            (key, reference[key], candidate[key])
            for key in sorted(set(reference.keys()))
        ]
        if phase in {"forward", "grads", "deltas"}:
            pairs = _stacked_layers(pairs)
        rows = self._build_metric_rows_from_tensor_pairs(
            variant=variant,
            step_index=step_index,
            phase=phase,
            pairs=pairs,
            router_ids=router_ids,
            layer_averaged=phase in {"forward", "grads", "deltas"},
        )
        if phase in {"grads", "deltas"}:
            rows.extend(
                self._build_metric_rows_from_tensor_pairs(
                    variant=variant,
                    step_index=step_index,
                    phase=phase,
                    pairs=_stacked_layers(
                        [
                            (
                                _expert_agnostic_param_key(key),
                                reference[key],
                                candidate[key],
                            )
                            for key in sorted(set(reference.keys()))
                            if _triplet_expert_key(key) is not None
                        ]
                    ),
                    router_ids=router_ids,
                    layer_averaged=True,
                )
            )
        return rows

    @staticmethod
    def _build_step_summaries(rows: list[MetricRow]) -> dict[int, dict[str, Any]]:
        """Builds step-indexed payloads directly from row model dumps."""
        step_summaries: dict[int, dict[str, Any]] = {}
        for row in rows:
            step_entry = step_summaries.setdefault(row.step_index, {})
            phase_entry = cast(dict[str, Any], step_entry.setdefault(row.phase, {}))
            phase_entry[row.param] = row.model_dump(mode="json")
        return step_summaries

    @staticmethod
    def _step_phase_rows(
        rows: list[MetricRow], step_index: int, phase: str
    ) -> list[MetricRow]:
        return [
            row for row in rows if row.step_index == step_index and row.phase == phase
        ]

    @classmethod
    def _phase_rows_pass(
        cls, rows: list[MetricRow], step_index: int, phase: str
    ) -> bool:
        phase_rows = cls._step_phase_rows(rows, step_index, phase)
        return bool(phase_rows) and all(row.pass_signal for row in phase_rows)

    @classmethod
    def _router_topk_exact(cls, rows: list[MetricRow], step_index: int) -> bool:
        topk_rows = cls._step_phase_rows(rows, step_index, "router_topk_ids")
        return bool(topk_rows) and all(
            row.pass_signal
            and row.topk_mismatch_fraction == 0.0
            and row.top1_mismatch_fraction == 0.0
            for row in topk_rows
        )

    @classmethod
    def _apply_forward_expert_lora_trace_noise_passes(
        cls, rows: list[MetricRow]
    ) -> None:
        """Reclassifies proven near-null expert LoRA forward trace noise only."""
        steps = {row.step_index for row in rows}
        gate_by_step = {
            step: (
                cls._phase_rows_pass(rows, step, "outputs")
                and cls._phase_rows_pass(rows, step, "router_scores")
                and cls._router_topk_exact(rows, step)
            )
            for step in steps
        }
        for row in rows:
            if row.pass_signal:
                continue
            if row.phase != "forward" or not _is_forward_expert_lora_trace(row.param):
                continue
            if not gate_by_step.get(row.step_index, False):
                continue
            if row.relative_l2 > FORWARD_EXPERT_LORA_TRACE_NOISE_RELATIVE_L2_LIMIT:
                continue
            row.pass_signal = True
            row.failure_reasons = [FORWARD_EXPERT_LORA_TRACE_NOISE_REASON]

    def compare_variant(self, variant: VariantSpec) -> VariantReport:
        """Compares one candidate variant against its reference topology."""
        reference_slug = variant.resolved_reference_slug()
        topology_slug = variant.resolved_output_slug()
        reference_dir = self.case_dir / reference_slug
        topology_dir = self.case_dir / topology_slug
        reference_manifest = _load_manifest(reference_dir)
        topology_manifest = _load_manifest(topology_dir)
        rows: list[MetricRow] = []
        if reference_manifest.objective != variant.objective:
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=0,
                    phase="objective",
                    param="__reference_objective__",
                    summary=self._inf_summary(),
                    structural_failure=(
                        f"reference={reference_manifest.objective} "
                        f"expected={variant.objective}"
                    ),
                )
            )
        if topology_manifest.objective != variant.objective:
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=0,
                    phase="objective",
                    param="__candidate_objective__",
                    summary=self._inf_summary(),
                    structural_failure=(
                        f"candidate={topology_manifest.objective} "
                        f"expected={variant.objective}"
                    ),
                )
            )
        if len(reference_manifest.steps) != len(topology_manifest.steps):
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=0,
                    phase="step_count",
                    param="__step_count__",
                    summary=self._inf_summary(),
                    structural_failure=(
                        f"reference={len(reference_manifest.steps)} "
                        f"candidate={len(topology_manifest.steps)}"
                    ),
                )
            )

        import torch

        for reference_step, topology_step in zip(
            reference_manifest.steps, topology_manifest.steps
        ):
            step_index = reference_step.step_index
            reference_trace = self._trim_trace_padding(
                _load_forward_trace(reference_dir, step_index)
            )
            topology_trace = self._trim_trace_padding(
                _load_forward_trace(topology_dir, step_index)
            )
            map_phase_inputs = [
                (
                    "outputs",
                    self._load_output_tensor_map(reference_dir, reference_step),
                    self._load_output_tensor_map(topology_dir, topology_step),
                    False,
                ),
                (
                    "losses",
                    self._load_loss_tensor_map(reference_step),
                    self._load_loss_tensor_map(topology_step),
                    False,
                ),
                (
                    "grads",
                    _load_safetensor_map(reference_dir / reference_step.grads_file),
                    _load_safetensor_map(topology_dir / topology_step.grads_file),
                    False,
                ),
                (
                    "deltas",
                    _load_safetensor_map(reference_dir / reference_step.deltas_file),
                    _load_safetensor_map(topology_dir / topology_step.deltas_file),
                    False,
                ),
                *[
                    (
                        phase,
                        ForwardTraceCapture.flatten_trace_tensors(
                            reference_trace,
                            value_key=value_key,
                        ),
                        ForwardTraceCapture.flatten_trace_tensors(
                            topology_trace,
                            value_key=value_key,
                        ),
                        phase == "router_topk_ids",
                    )
                    for phase, value_key in (
                        ("forward", "primary_output"),
                        ("router_scores", "router_topk_scores"),
                        ("router_topk_ids", "router_topk_ids"),
                    )
                ],
            ]
            for phase, reference_map, candidate_map, router_ids in map_phase_inputs:
                rows.extend(
                    self._build_metric_rows_from_tensor_maps(
                        variant=variant,
                        step_index=step_index,
                        phase=phase,
                        reference=reference_map,
                        candidate=candidate_map,
                        router_ids=router_ids,
                    )
                )
        self._apply_forward_expert_lora_trace_noise_passes(rows)
        pass_count = sum(1 for row in rows if row.pass_signal)
        fail_count = len(rows) - pass_count
        signal: Literal["pass", "fail"] = "pass" if fail_count == 0 else "fail"
        return VariantReport(
            git=self.git,
            case_id=self.case_id,
            variant=variant.name,
            topology=topology_slug,
            reference_topology=reference_slug,
            expected_signal=variant.expected_signal,
            signal=signal,
            pass_count=pass_count,
            fail_count=fail_count,
            step_summaries=self._build_step_summaries(rows),
            metrics=rows,
        )

    @staticmethod
    def assert_expected_signal(
        report: VariantReport,
        context: str,
        *,
        report_path: Path,
    ) -> None:
        """Raises when observed run signal diverges from variant expectation."""
        if report.signal == report.expected_signal:
            return
        if report.signal == "fail":
            first_failure = next(row for row in report.metrics if not row.pass_signal)
            raise AssertionError(
                f"{context}: topology={report.topology} phase={first_failure.phase} "
                f"step={first_failure.step_index} param={first_failure.param} "
                f"reasons={'; '.join(first_failure.failure_reasons)} "
                f"report={report_path}"
            )
        raise AssertionError(
            f"{context}: expected_signal={report.expected_signal} "
            f"observed_signal={report.signal} topology={report.topology} "
            f"report={report_path}"
        )

    def _write_variant_report(self, topology_dir: Path, report: VariantReport) -> None:
        """Persists full variant report JSON for debugging and regression inspection."""
        _write_json(
            topology_dir / "variant_report.json", report.model_dump(mode="json")
        )

    def _prune_reference_artifacts(self) -> None:
        """Drops oracle-only tensors after all comparisons that need them are complete."""
        _prune_topology_artifacts(self.oracle_dir)
        if self.case_config.is_moe:
            _prune_topology_artifacts(self.oracle_routing_bundle_dir)
            _prune_topology_artifacts(
                self.case_dir / f"{self.oracle_slug}__oracle_capture"
            )

    def print_report(self, report: VariantReport) -> None:
        """Prints a row-level table excluding expert-specific rows."""
        table_rows = [
            row for row in report.metrics if _triplet_expert_key(row.param) is None
        ]
        detail_table = Table(
            title=f"Variant Report | variant={report.variant}",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        detail_table.add_column("Step", justify="right")
        detail_table.add_column("Phase", style="cyan")
        detail_table.add_column("Param", overflow="fold")
        detail_table.add_column("Status")
        detail_table.add_column("relative_l2", justify="right")
        detail_table.add_column("mean_abs_pct", justify="right")
        detail_table.add_column("typical_abs", justify="right")
        detail_table.add_column("mean_abs_diff", justify="right")
        detail_table.add_column("Failure")
        sorted_rows = sorted(
            table_rows,
            key=lambda row: (
                row.step_index,
                PHASE_PRINT_ORDER.get(row.phase, 999),
                row.param,
                row.pass_signal,
            ),
        )
        for row in sorted_rows:
            status_text = (
                "[green]PASS[/green]" if row.pass_signal else "[red]FAIL[/red]"
            )
            failure_text = "" if row.pass_signal else "; ".join(row.failure_reasons)
            detail_table.add_row(
                str(row.step_index),
                row.phase,
                row.param,
                status_text,
                f"{row.relative_l2:.6g}",
                f"{row.mean_abs_pct:.6g}%",
                f"{row.typical_abs_scale:.6g}",
                f"{row.mean_abs_diff:.6g}",
                failure_text,
            )
        self.console.print(detail_table)

    def run_variant(
        self,
        variant: VariantSpec,
    ) -> VariantReport:
        """Runs a variant end-to-end, writes JSON report, and prints row table."""
        topology_dir = self.ensure_variant_artifacts(variant)
        report = self.compare_variant(variant)
        self._write_variant_report(topology_dir, report)
        _prune_topology_artifacts(topology_dir)
        self.print_report(report)
        return report

    def run_suite(
        self,
        variants: list[VariantSpec],
        *,
        prune_reference_artifacts: bool = True,
        prune_case_artifacts: bool = True,
    ) -> list[VariantReport]:
        """Runs variants in order and stops at the first unexpected signal.

        Reference and case artifacts are normally pruned when the suite exits.
        Callers that immediately run another comparison suite against the same
        reference can defer that pruning so the second suite does not have to
        regenerate or fail on missing forward traces.
        """
        reports: list[VariantReport] = []
        try:
            for variant in variants:
                report = self.run_variant(variant)
                reports.append(report)
                self.assert_expected_signal(
                    report,
                    "Megatron correctness suite mismatch",
                    report_path=self.case_dir
                    / variant.resolved_output_slug()
                    / "variant_report.json",
                )
        finally:
            if prune_reference_artifacts:
                self._prune_reference_artifacts()
            if prune_case_artifacts:
                _prune_case_artifacts(self.case_dir)
        return reports


def _default_phase_pass_fns() -> dict[str, PhasePassFn]:
    """Builds default per-phase pass functions over diff summaries."""
    # note the metrics get averaged across layers to reduce noise
    # we also average across experts to reduce noise
    # we don't expect particular layers to see errors as opposed to the others so this is helpful
    non_zero_scales = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
    fwd_out_loss = MetricThresholdRule(
        limits={"mean_abs_pct": ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT}
    )
    fwd_out = MetricThresholdRule(
        limits={"mean_abs_pct": ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT},
        minimums=non_zero_scales,
    )
    grads_deltas = MetricThresholdRule(
        limits={"mean_abs_pct": ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT},
        minimums=non_zero_scales,
    )
    router_scores_rule = MetricThresholdRule(
        # Production RouterReplay replays top-k ids and gathers probabilities from
        # live candidate scores, so scores are close but not bit-exact.
        limits={"mean_abs_pct": ROUTER_SCORE_MEAN_ABS_PCT_LIMIT}
    )
    router_topk_rule = (
        MetricThresholdRule(  # should be no mismatch due to router replay
            limits={
                "topk_mismatch_fraction": 0.0,
                "top1_mismatch_fraction": 0.0,
            }
        )
    )
    return {"forward": fwd_out, "outputs": fwd_out, "losses": fwd_out_loss} | {
        "grads": grads_deltas,
        "deltas": grads_deltas,
        "router_scores": router_scores_rule,
        "router_topk_ids": router_topk_rule,
    }


def _suite_variants(
    objective: OracleObjective,
    *,
    is_moe: bool = True,
    cp_supported: bool = True,
    max_world_size: int | None = None,
    variant_flex_backend: FlexBackend | None = None,
    phase_pass_fns: dict[str, PhasePassFn] | None = None,
) -> list[VariantSpec]:
    """Builds the standard oracle suite variant ordering."""
    phase_pass = phase_pass_fns or _default_phase_pass_fns()
    variants: list[VariantSpec] = []
    for topology in selected_suite_topologies(
        is_moe=is_moe,
        cp_supported=cp_supported,
    )[1:]:
        if max_world_size is not None and topology.world_size() > max_world_size:
            continue
        variants.append(
            VariantSpec(
                name=f"{objective}_topology_{topology.slug()}",
                objective=objective,
                topology=topology,
                pass_fn_by_phase=phase_pass,
                flex_backend=variant_flex_backend,
            )
        )
    return variants


def run_suite(
    *,
    case_config: OracleCaseConfig,
    max_world_size: int | None = None,
    oracle_flex_backend: FlexBackend | None = None,
    variant_flex_backend: FlexBackend | None = None,
    cp_supported: bool = True,
    phase_pass_fns: dict[str, PhasePassFn] | None = None,
    use_fp32_lora_reference: bool = True,
    prune_reference_artifacts: bool = True,
    prune_case_artifacts: bool = True,
) -> list[VariantReport]:
    """Runs non-oracle topologies against the canonical replay-backed oracle."""
    reports: list[VariantReport] = []
    for objective in selected_oracle_objectives():
        runner = VariantRunner(
            objective=objective,
            case_config=case_config,
            oracle_flex_backend=oracle_flex_backend,
            variant_flex_backend=variant_flex_backend,
            use_fp32_lora_reference=use_fp32_lora_reference,
        )
        reports.extend(
            runner.run_suite(
                _suite_variants(
                    objective,
                    is_moe=case_config.is_moe,
                    cp_supported=cp_supported,
                    max_world_size=max_world_size,
                    variant_flex_backend=variant_flex_backend,
                    phase_pass_fns=phase_pass_fns,
                ),
                prune_reference_artifacts=prune_reference_artifacts,
                prune_case_artifacts=prune_case_artifacts,
            )
        )
    return reports


def run_sensitivity_suite(
    *,
    case_config: OracleCaseConfig,
    mutations: list[SensitivityMutation],
    max_world_size: int | None = None,
    oracle_flex_backend: FlexBackend | None = None,
    variant_flex_backend: FlexBackend | None = None,
) -> list[VariantReport]:
    """Runs a list of sensitivity mutations and expects each to fail."""
    phase_pass = _default_phase_pass_fns()
    reports: list[VariantReport] = []
    ran_any_variants = False
    for objective in selected_oracle_objectives():
        runner = VariantRunner(
            objective=objective,
            case_config=case_config,
            oracle_flex_backend=oracle_flex_backend,
            variant_flex_backend=variant_flex_backend,
        )
        objective_mutations = selected_sensitivity_mutations_for_objective(
            objective,
            mutations,
            is_moe=case_config.is_moe,
        )
        if not objective_mutations:
            continue
        variants = []
        for mutation in objective_mutations:
            topology = sensitivity_topology_for_mutation(
                mutation,
                is_moe=case_config.is_moe,
            )
            if max_world_size is not None and topology.world_size() > max_world_size:
                continue
            variants.append(
                VariantSpec(
                    name=f"{objective}_sensitivity_{mutation}",
                    objective=objective,
                    topology=topology,
                    mutation=mutation,
                    expected_signal="fail",
                    pass_fn_by_phase=phase_pass,
                    flex_backend=variant_flex_backend,
                )
            )
        if not variants:
            continue
        ran_any_variants = True
        reports.extend(runner.run_suite(variants))
    if ran_any_variants:
        return reports
    requested = ", ".join(mutations)
    supported = ", ".join(
        f"{objective}: "
        f"{', '.join(supported_sensitivity_mutations_for_objective(objective, is_moe=case_config.is_moe))}"
        for objective in selected_oracle_objectives()
    )
    raise ValueError(
        "No sensitivity variants matched the selected objectives. "
        f"Requested mutations: {requested}. Supported by objective: {supported}."
    )
