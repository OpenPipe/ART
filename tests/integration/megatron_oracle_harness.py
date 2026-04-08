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

from .megatron_forward_trace import ForwardTraceCapture

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(REPO_ROOT / ".local/megatron_lora_correctness")
ORACLE_MOE_ROUTING_BUNDLE_DIRNAME = "oracle_moe_routing_replay"
ORACLE_REPLAY_TOPOLOGY_SUFFIX = "oracle_replay"

REGENERATE_ENV = "ART_REGENERATE_ORACLE"
EXTENDED_TOPOLOGIES_ENV = "ART_ENABLE_EXTENDED_TOPOLOGIES"
SENSITIVITY_MUTATION_ENV = "ART_SENSITIVITY_MUTATIONS"
ORACLE_OBJECTIVE_ENV = "ART_ORACLE_OBJECTIVE"

OracleObjective = Literal["rl", "sft"]
SUPPORTED_ORACLE_OBJECTIVES: tuple[OracleObjective, ...] = ("rl", "sft")
SensitivityMutation = str

DEFAULT_SENSITIVITY_MUTATION = "skip_finalize"
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
EXPERT_TABLE_ROW_LIMIT = 8
EXPERT_TRIPLET_PARAM_RE = re.compile(
    r"layers\.(?P<layer>\d+|__layer_avg__)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
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
) -> tuple[SensitivityMutation, ...]:
    return OBJECTIVE_SENSITIVITY_MUTATIONS[objective]


def objective_supports_sensitivity_mutation(
    objective: OracleObjective,
    mutation: SensitivityMutation,
) -> bool:
    return mutation in supported_sensitivity_mutations_for_objective(objective)


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
    Topology(tp=2, ep=1, etp=1, dp=1, sp=True),
    Topology(tp=2, ep=2, etp=1, dp=1, sp=True),
    Topology(tp=2, ep=1, etp=2, dp=1, sp=True),
]
EXTENDED_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=2, sp=False),
    Topology(tp=1, ep=2, etp=1, dp=2, sp=False),
    Topology(tp=1, ep=1, etp=2, dp=2, sp=True),
]
ORACLE_TOPOLOGY = TOPOLOGIES[0]
SENSITIVITY_TOPOLOGY = Topology(tp=2, ep=2, etp=1, dp=1, sp=True)
SENSITIVITY_TOPOLOGY_BY_MUTATION: dict[SensitivityMutation, Topology] = {
    mutation: SENSITIVITY_TOPOLOGY for mutation in SUPPORTED_SENSITIVITY_MUTATIONS
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
    sequence_length: int = 256
    prefill_tokens: int = 64
    decode_tokens: int = 64
    decode_tokens_jitter: int = Field(default=32, ge=0)
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


class StepTrace(BaseModel):
    """Tracks per-step trace artifact filenames and loss metadata."""

    step_index: int
    loss: float
    probs_corr: float
    output_file: str
    grads_file: str
    deltas_file: str
    lora_file: str


class RunManifest(BaseModel):
    """Records run metadata and per-step trace references for one topology output."""

    case_id: str
    objective: OracleObjective
    base_model: str
    num_layers: int
    topology: str
    world_size: int
    seed: int
    num_steps: int
    packed_tensors: DiskPackedTensorsSpec
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
    objective: OracleObjective
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

    @staticmethod
    def layer_averaged_summary(reference_stack, candidate_stack) -> dict[str, float]:  # type: ignore[no-untyped-def]
        """Computes normal per-layer summaries, then averages those summaries."""
        ref = reference_stack.detach().float()
        cand = candidate_stack.detach().float()
        layer_count = int(ref.shape[0])
        metrics = {
            k: 0.0
            for k in [
                "numel",
                "mean_abs_diff",
                "relative_l2",
                "typical_abs_scale",
                "mean_abs_pct",
            ]
        }
        for layer_index in range(layer_count):
            layer_accumulator = DiffAccumulator()
            layer_accumulator.update(ref[layer_index], cand[layer_index])
            layer_summary = layer_accumulator.as_summary()
            metrics = {k: metrics[k] + layer_summary[k] for k in metrics.keys()}
        return {k: _finite_metric(metrics[k] / layer_count) for k in metrics.keys()}

    def update_router_ids(self, reference_ids, candidate_ids) -> None:  # type: ignore[no-untyped-def]
        """Adds router top-k id mismatch counts into the accumulator."""
        self.router_topk_total += int(reference_ids.numel())
        self.router_topk_mismatch += int((reference_ids != candidate_ids).sum().item())
        if reference_ids.ndim >= 2 and reference_ids.shape[1] > 0:
            self.router_top1_total += int(reference_ids.shape[0])
            self.router_top1_mismatch += int(
                (reference_ids[:, 0] != candidate_ids[:, 0]).sum().item()
            )

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
                "mean_abs_pct": 0.0,
                "topk_mismatch_fraction": topk_fraction,
                "top1_mismatch_fraction": top1_fraction,
            }
        mean_abs = self.abs_sum / self.numel
        typical_abs = self.ref_abs_sum / self.numel
        mean_abs_pct = (mean_abs / (typical_abs + 1e-12)) * 100.0
        return {
            "numel": _finite_metric(float(self.numel), default=0.0),
            "mean_abs_diff": _finite_metric(mean_abs),
            "relative_l2": _finite_metric(
                (self.diff_sq_sum**0.5) / max(self.ref_sq_sum**0.5, 1e-12)
            ),
            "typical_abs_scale": _finite_metric(typical_abs, default=0.0),
            "mean_abs_pct": _finite_metric(mean_abs_pct),
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
) -> list[SensitivityMutation]:
    return [
        mutation
        for mutation in mutations
        if objective_supports_sensitivity_mutation(objective, mutation)
    ]


def sensitivity_topology_for_mutation(mutation: SensitivityMutation) -> Topology:
    """Returns the sensitivity topology required for one mutation."""
    return SENSITIVITY_TOPOLOGY_BY_MUTATION[mutation]


def sensitivity_required_world_size(mutations: list[SensitivityMutation]) -> int:
    """Returns the max world-size required by a selected mutation set."""
    return max(
        sensitivity_topology_for_mutation(mutation).world_size()
        for mutation in mutations
    )


def extended_topologies_enabled() -> bool:
    """Returns whether extended topologies are enabled for the suite."""
    return _truthy(os.environ.get(EXTENDED_TOPOLOGIES_ENV))


def regenerate_requested() -> bool:
    """Returns whether regeneration mode is enabled for oracle artifacts."""
    return _truthy(os.environ.get(REGENERATE_ENV))


def case_config(
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> OracleCaseConfig:
    """Builds the deterministic default oracle case config."""
    return OracleCaseConfig(base_model=base_model)


def available_gpu_count() -> int:
    """Reports visible CUDA device count for topology scheduling and test skips."""
    import torch

    return int(torch.cuda.device_count())


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
    tokens = torch.randint(
        low=10,
        high=config.vocab_high,
        size=shape,
        dtype=torch.long,
        generator=generator,
    )
    # Ensure paired cross-DP rows are never token-identical.
    half = config.num_sequences // 2
    if half > 0 and config.num_sequences % 2 == 0:
        for pair_index in range(half):
            left_index = pair_index
            right_index = pair_index + half
            if torch.equal(tokens[left_index], tokens[right_index]):
                token_span = max(1, config.vocab_high - 10)
                tokens[right_index] = ((tokens[right_index] - 10 + 1) % token_span) + 10
    group_ids = torch.zeros(shape, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = (
        torch.arange(config.sequence_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(config.num_sequences, -1)
        .clone()
    )
    prefix_length = max(1, min(config.sequence_length - 1, config.prefill_tokens))
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    max_decode_tokens = max(1, config.sequence_length - prefix_length)
    base_decode_tokens = max(1, min(config.decode_tokens, max_decode_tokens))
    jitter_width = min(config.decode_tokens_jitter, max_decode_tokens - 1)
    candidate_decode_lengths: list[int] = []
    for _ in range(config.num_sequences):
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
        decode_length = max(
            1,
            min(max_decode_tokens, base_decode_tokens + jitter),
        )
        candidate_decode_lengths.append(decode_length)
    # Keep jitter local around the configured decode length, but force pairwise
    # differences across halves so default DP rank shards see different lengths.
    if half > 0 and config.num_sequences % 2 == 0:
        for pair_index in range(half):
            left_index = pair_index
            right_index = pair_index + half
            if (
                candidate_decode_lengths[left_index]
                != candidate_decode_lengths[right_index]
            ):
                continue
            if candidate_decode_lengths[right_index] < max_decode_tokens:
                candidate_decode_lengths[right_index] += 1
            elif candidate_decode_lengths[right_index] > 1:
                candidate_decode_lengths[right_index] -= 1

    for sequence_index, decode_length in enumerate(candidate_decode_lengths):
        active_stop = prefix_length + decode_length
        assistant_mask[sequence_index, prefix_length:active_stop] = True
        decode_span = max(1, min(config.decode_tokens, decode_length))
        cursor = prefix_length
        branch = 1
        while cursor < active_stop:
            end = min(active_stop, cursor + decode_span)
            group_ids[sequence_index, cursor:end] = branch
            parent_ids[sequence_index, cursor:end] = 0
            cursor = end
            branch += 1
    logprobs = (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
        - 1.75
    )
    advantages = (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
        + 1.0
    )
    weights = torch.ones(shape, dtype=torch.float32)
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
                stacked_pairs.append((original_name, reference, candidate))
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
        objective: OracleObjective,
        case_config: OracleCaseConfig,
        console: Console | None = None,
    ) -> None:
        self.objective = objective
        self.case_config = case_config
        self.case_artifacts = ensure_case_artifacts(case_config)
        self.case_id = self.case_artifacts.case_id
        self.case_dir = Path(self.case_artifacts.case_dir)
        self.oracle_slug = oracle_output_slug(objective, ORACLE_TOPOLOGY)
        self.oracle_dir = self.case_dir / self.oracle_slug
        self.oracle_routing_bundle_dir = (
            self.case_dir / f"{objective}__{ORACLE_MOE_ROUTING_BUNDLE_DIRNAME}"
        )
        self.shared_init_path = Path(self.case_artifacts.shared_init_adapter_path)
        self.console = console or Console(width=140)
        self._oracle_initialized = False
        self._oracle_regenerated = False

    def _run_topology(
        self,
        *,
        topology: Topology,
        output_slug: str,
        mutation: SensitivityMutation | None,
        replay_bundle_dir: Path | None,
        capture_bundle_dir: Path | None,
        regenerate: bool,
    ) -> Path:
        """Executes one topology worker run and returns its output directory."""
        topology_dir = self.case_dir / output_slug
        manifest_path = topology_dir / "manifest.json"
        if manifest_path.exists() and not regenerate:
            return topology_dir
        _replace_topology_dir(topology_dir)
        run_case_config = self.case_config
        request = WorkerRunRequest(
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
        )
        from .megatron_oracle_worker import run_worker_subprocess

        run_worker_subprocess(request, topology_dir, repo_root=REPO_ROOT)
        return topology_dir

    def ensure_oracle(self) -> Path:
        """Ensures oracle capture and canonical replay artifacts exist exactly once per session."""
        regenerate = regenerate_requested()
        if self._oracle_initialized and (not regenerate or self._oracle_regenerated):
            return self.oracle_dir
        if regenerate and self.shared_init_path.exists():
            self.shared_init_path.unlink()
        bundle_manifest = self.oracle_routing_bundle_dir / "manifest.json"
        oracle_manifest = self.oracle_dir / "manifest.json"
        need_capture = (
            regenerate
            or not bundle_manifest.exists()
            or not self.shared_init_path.exists()
        )
        run_oracle_topology = partial(
            self._run_topology,
            topology=ORACLE_TOPOLOGY,
            mutation=None,
            regenerate=True,
        )
        if need_capture:
            run_oracle_topology(
                output_slug=f"{self.oracle_slug}__oracle_capture",
                replay_bundle_dir=None,
                capture_bundle_dir=self.oracle_routing_bundle_dir,
            )
        if regenerate or not oracle_manifest.exists():
            run_oracle_topology(
                output_slug=self.oracle_slug,
                replay_bundle_dir=self.oracle_routing_bundle_dir,
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
            replay_bundle_dir=self.oracle_routing_bundle_dir,
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
                    reference_aligned, aligned_candidate
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
            reference_trace = _load_forward_trace(reference_dir, step_index)
            topology_trace = _load_forward_trace(topology_dir, step_index)
            map_phase_inputs = [
                (
                    "outputs",
                    {"logprobs": _load_output_tensor(reference_dir, reference_step)},
                    {"logprobs": _load_output_tensor(topology_dir, topology_step)},
                    False,
                ),
                (
                    "losses",
                    {"loss": torch.tensor([reference_step.loss], dtype=torch.float32)},
                    {"loss": torch.tensor([topology_step.loss], dtype=torch.float32)},
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
        pass_count = sum(1 for row in rows if row.pass_signal)
        fail_count = len(rows) - pass_count
        signal: Literal["pass", "fail"] = "pass" if fail_count == 0 else "fail"
        return VariantReport(
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
        self.print_report(report)
        return report

    def run_suite(
        self,
        variants: list[VariantSpec],
    ) -> list[VariantReport]:
        """Runs variants in order and stops at the first unexpected signal."""
        reports: list[VariantReport] = []
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
        return reports


def _default_phase_pass_fns() -> dict[str, PhasePassFn]:
    """Builds default per-phase pass functions over diff summaries."""
    # note the metrics get averaged across layers to reduce noise
    # we also average across experts to reduce noise
    # we don't expect particular layers to see errors as opposed to the others so this is helpful
    fwd_out_loss = MetricThresholdRule(
        limits={"relative_l2": 1e-2, "mean_abs_pct": 1.0}
    )
    grads_deltas = MetricThresholdRule(limits={"mean_abs_pct": 3.0})
    router_topk_rule = (
        MetricThresholdRule(  # should be no mismatch due to router replay
            limits={
                "topk_mismatch_fraction": 0.0,
                "top1_mismatch_fraction": 0.0,
            }
        )
    )
    return {key: fwd_out_loss for key in ["forward", "outputs", "losses"]} | {
        "grads": grads_deltas,
        "deltas": grads_deltas,
        "router_topk_ids": router_topk_rule,
    }


def _suite_variants(objective: OracleObjective) -> list[VariantSpec]:
    """Builds the standard oracle suite variant ordering."""
    phase_pass = _default_phase_pass_fns()
    variants = [
        VariantSpec(
            name=f"{objective}_oracle_replay_parity",
            objective=objective,
            topology=ORACLE_TOPOLOGY,
            output_slug=oracle_output_slug(
                objective,
                ORACLE_TOPOLOGY,
                ORACLE_REPLAY_TOPOLOGY_SUFFIX,
            ),
            pass_fn_by_phase=phase_pass,
            force_regenerate=regenerate_requested(),
        )
    ]
    for topology in TOPOLOGIES[1:] + (
        EXTENDED_TOPOLOGIES if extended_topologies_enabled() else []
    ):
        variants.append(
            VariantSpec(
                name=f"{objective}_topology_{topology.slug()}",
                objective=objective,
                topology=topology,
                pass_fn_by_phase=phase_pass,
            )
        )
    return variants


def run_suite(
    *,
    case_config: OracleCaseConfig,
) -> list[VariantReport]:
    """Runs replay parity and topology variants with fail-fast assertions."""
    reports: list[VariantReport] = []
    for objective in selected_oracle_objectives():
        runner = VariantRunner(objective=objective, case_config=case_config)
        reports.extend(runner.run_suite(_suite_variants(objective)))
    return reports


def run_sensitivity_suite(
    *,
    case_config: OracleCaseConfig,
    mutations: list[SensitivityMutation],
) -> list[VariantReport]:
    """Runs a list of sensitivity mutations and expects each to fail."""
    phase_pass = _default_phase_pass_fns()
    reports: list[VariantReport] = []
    ran_any_variants = False
    for objective in selected_oracle_objectives():
        runner = VariantRunner(objective=objective, case_config=case_config)
        objective_mutations = selected_sensitivity_mutations_for_objective(
            objective,
            mutations,
        )
        if not objective_mutations:
            continue
        variants = [
            VariantSpec(
                name=f"{objective}_sensitivity_{mutation}",
                objective=objective,
                topology=sensitivity_topology_for_mutation(mutation),
                mutation=mutation,
                expected_signal="fail",
                pass_fn_by_phase=phase_pass,
            )
            for mutation in objective_mutations
        ]
        ran_any_variants = True
        reports.extend(runner.run_suite(variants))
    if ran_any_variants:
        return reports
    requested = ", ".join(mutations)
    supported = ", ".join(
        f"{objective}: {', '.join(supported_sensitivity_mutations_for_objective(objective))}"
        for objective in selected_oracle_objectives()
    )
    raise ValueError(
        "No sensitivity variants matched the selected objectives. "
        f"Requested mutations: {requested}. Supported by objective: {supported}."
    )
