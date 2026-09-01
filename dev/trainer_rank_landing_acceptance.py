"""Holistic TrainerRank planner landing acceptance driver.

One self-contained script covering the four acceptance phases for the planner
landing. Gate thresholds are set from the research thread's sealed evidence
(final acceptance campaign, 2026-08-31/09-01) with conservative margins; the
sealed values are recorded next to each gate.

Phases
------
contract        CPU. Asserts the knob-free public API. Fails fast on any tree
                where the holistic planner has not landed. Every other phase
                runs this first.
census          CPU. Plans all 44 real Ellavox groups (88 rank workloads) from
                the pinned corpus and requires zero refusals.
                Sealed: 88/88 feasible, 0 refusals.
measure         GPU. Paired measurement of the automatic planner against a
                reference arm on one predeclared cell, emitting evidence JSONL.
                Reference arms are forced via the test-only hook (see below).
validate        CPU. Applies the phase gates to measured evidence JSONL and
                exits nonzero on any gate failure.

Predeclared GPU cells (from the sealed research recipes)
--------------------------------------------------------
grpo-gdn-cp4    Qwen/Qwen3.5-4B, 2 layers, CP4, hierarchical GRPO
                ``primary_long_g8`` shape (2 groups x 8 completions,
                system 2048, prompt 8192, completion 512).
                Sealed result: +47.2% paired median complete-call gain vs the
                depth-one arm (CI95 29.9-61.0%), 55.8% lower p50 peak
                allocation (5.85 vs 13.23 GiB), median selected max depth 3.
cp1-regression  Qwen/Qwen3.5-4B CP1: the heterogeneous control and the real
                Ellavox stream. Sealed result: -0.58% and +0.07% (ties), zero
                unsafe admissions, planning fraction 1.5% / 4.2%.

Acceptance interface required from the landed implementation
-------------------------------------------------------------
1. Test-only anchor forcing: when ``ART_TRAINER_RANK_TEST_HOOKS=1``, the
   planner honors ``ART_TRAINER_RANK_TEST_ANCHOR`` in
   {"depth_one", "full_sharing"} by pinning selection to that anchor. It must
   be inert (and ideally rejected) without the opt-in, and must never be
   reachable through public constructor or method arguments.
2. Concise plan telemetry: ``TrainerRank.last_forward_telemetry()`` returning
   at least ``selected_max_depth`` (int) and ``planning_ms`` (float) for the
   most recent public forward on this rank.
These are part of the landing contract; if names differ at landing, adapt the
single ADAPTATION POINT block below, not the gates.

These phases define the landing contract; they were written (and
fail-verified) before the implementation and must pass on the landed tree.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ELLAVOX_CORPUS = REPO_ROOT / "dev" / "_trainer_rank_ellavox_qwen35_4b_tokens.json"
ELLAVOX_CORPUS_SHA256 = (
    "b2528f067065c20cea81a2868f39a8cdd008c30ee54d5e90b68ddf598cfcd41b"
)

# Gate thresholds (sealed measurement -> conservative landing gate).
GATES: dict[str, dict[str, float]] = {
    "grpo-gdn-cp4": {
        # Sealed: 47.2% (bootstrap CI95 lower bound 29.9%).
        "min_paired_median_gain_pct": 20.0,
        # Sealed: 55.8% lower p50 peak allocation.
        "min_peak_reduction_pct": 30.0,
        # Sealed: median selected max depth 3. Tail segments count toward
        # maximum_depth, so the depth-one reference arm itself reports 2; the
        # gate must sit at the sealed value to detect selection collapse.
        "min_median_selected_max_depth": 3.0,
        # This cell is the sealed 2-layer throughput screen: execution is
        # deliberately tiny, so planning is bounded absolutely here (sealed
        # steady planning was 82 ms p50). The 10% planning *fraction* gate is
        # measured on the full-height cp1-regression cell, matching the
        # sealed protocol (its CP1 acceptance cells ran the full model).
        "max_planning_ms": 150.0,
    },
    "cp1-regression": {
        # Sealed: -0.58% / +0.07% (ties). Gate: no worse than a 2% loss.
        "max_paired_median_regression_pct": 2.0,
        # Sealed: 1.5% / 4.2% on full-height CP1 cells.
        "max_planning_fraction": 0.10,
    },
}

# GRPO primary_long_g8 predeclared shape (groups, group size, system, prompt,
# completion) from the sealed sweep preregistration.
GRPO_PRIMARY_LONG_G8 = (2, 8, 2048, 8192, 512)

# --- ADAPTATION POINT (names only; gates above must not change) -------------
TEST_HOOKS_ENV = "ART_TRAINER_RANK_TEST_HOOKS"
TEST_ANCHOR_ENV = "ART_TRAINER_RANK_TEST_ANCHOR"
TELEMETRY_METHOD = "last_forward_telemetry"
TELEMETRY_DEPTH_KEY = "selected_max_depth"
TELEMETRY_PLANNING_MS_KEY = "planning_ms"
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    print(f"LANDING ACCEPTANCE FAILURE: {message}", file=sys.stderr)
    raise SystemExit(1)


def _public_parameters(callable_: Any) -> dict[str, inspect.Parameter]:
    return {
        name: parameter
        for name, parameter in inspect.signature(callable_).parameters.items()
        if name not in ("self", "cls")
    }


def phase_contract() -> None:
    """Fail fast unless the knob-free public contract has landed."""

    import art.trainer_rank as trainer_rank

    problems: list[str] = []
    constructor = _public_parameters(trainer_rank.TrainerRank.__init__)
    if list(constructor) != ["runtime"]:
        problems.append(
            f"TrainerRank must accept exactly (runtime); found {sorted(constructor)}"
        )
    for method_name in ("forward_micro_batches", "dp_rank_forward"):
        parameters = _public_parameters(getattr(trainer_rank.TrainerRank, method_name))
        extra = set(parameters) - {"inputs", "checkpoint", "no_grad"}
        if extra:
            problems.append(f"{method_name} has extra parameters {sorted(extra)}")
    if not hasattr(trainer_rank.TrainerRank, TELEMETRY_METHOD):
        problems.append(
            f"TrainerRank.{TELEMETRY_METHOD}() telemetry surface is missing"
        )
    try:
        from art.trainer_rank import _prefix_tree_planner
    except ImportError:
        problems.append("planner surface art.trainer_rank._prefix_tree_planner missing")
    else:
        for required in (
            "build_canonical_prefix_tree",
            "prefix_tree_layout_candidates",
            "select_prefix_tree_layout",
        ):
            if not hasattr(_prefix_tree_planner, required):
                problems.append(f"planner surface is missing {required}")
    if problems:
        _fail("contract phase:\n  - " + "\n  - ".join(problems))
    print("contract phase: PASS")


@functools.lru_cache(maxsize=1)
def _load_corpus() -> dict[str, Any]:
    import hashlib

    data = ELLAVOX_CORPUS.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ELLAVOX_CORPUS_SHA256:
        _fail(f"census corpus digest mismatch: {digest}")
    return json.loads(data)


def _load_census_sequences() -> dict[str, tuple[tuple[int, ...], ...]]:
    corpus = _load_corpus()
    workloads: dict[str, tuple[tuple[int, ...], ...]] = {}
    for group in corpus["groups"]:
        workloads[f"group-{group['slice_id']}-{group['group_id']}"] = tuple(
            tuple(history["tokens"]) for history in group["histories"]
        )
    return workloads


def phase_census() -> None:
    """Every real Ellavox group must plan without refusal. Sealed: 88/88."""

    phase_contract()
    from art.trainer_rank import _prefix_tree_planner as planner

    refusals: list[str] = []
    planned = 0
    for workload_id, sequences in _load_census_sequences().items():
        tree = planner.build_canonical_prefix_tree(sequences)
        try:
            selected = planner.select_prefix_tree_layout(
                tree,
                cp_size=1,
                layers=36,
                uses_gdn=True,
                refinement_work_budget=2_000,
            )
        except Exception as error:  # noqa: BLE001 - any refusal fails the gate
            refusals.append(f"{workload_id}: {type(error).__name__}: {error}")
            continue
        if selected is None:
            refusals.append(f"{workload_id}: planner returned no layout")
            continue
        planned += 1
    if refusals:
        _fail(
            f"census phase: {len(refusals)} refusals (gate: 0):\n  - "
            + "\n  - ".join(refusals[:10])
        )
    print(f"census phase: PASS ({planned} workloads planned, 0 refusals)")


def _grpo_requests(seed: int) -> list[Any]:
    """Hierarchical GRPO requests (verbatim shape from the sealed sweep)."""

    import torch

    from art.trainer_rank import ForwardInput

    prompt_groups, group_size, system_tokens, prompt_tokens, completion_tokens = (
        GRPO_PRIMARY_LONG_G8
    )

    def _tokens(token_seed: int, count: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(token_seed)
        return torch.randint(low=10, high=64_000, size=(count,), generator=generator)

    system = _tokens(seed * 100_003, system_tokens)
    requests: list[Any] = []
    for prompt_group in range(prompt_groups):
        prompt = _tokens(seed * 1_000_003 + prompt_group * 100_019, prompt_tokens)
        prefix = torch.cat((system, prompt))
        for branch in range(group_size):
            completion = _tokens(
                seed * 10_000_019 + prompt_group * 1_000_033 + branch * 10_007,
                completion_tokens,
            )
            tokens = torch.cat((prefix, completion))
            labels = torch.roll(tokens, shifts=-1).clone()
            labels[-1] = -100
            labels[: max(int(prefix.numel()) - 1, 0)] = -100
            requests.append(ForwardInput(input_tokens=tokens, target_tokens=labels))
    return requests


def _ellavox_requests(sample: int) -> list[Any]:
    """Real Ellavox rows for the depth-one-is-best regression cell."""

    import torch

    from art.trainer_rank import ForwardInput

    groups = _load_corpus()["groups"]
    group = groups[sample % len(groups)]
    requests: list[Any] = []
    for history in group["histories"]:
        tokens = torch.tensor(history["tokens"], dtype=torch.long)
        labels = torch.roll(tokens, shifts=-1).clone()
        labels[-1] = -100
        requests.append(ForwardInput(input_tokens=tokens, target_tokens=labels))
    return requests


def _cell_requests(cell: str, sample: int) -> list[Any]:
    if cell == "grpo-gdn-cp4":
        return _grpo_requests(seed=6_001 + sample)
    if cell == "cp1-regression":
        return _ellavox_requests(sample)
    raise AssertionError(f"unknown cell {cell!r}")


def _output_loss(outputs: Any) -> Any:
    import torch

    terms = [
        -output.target_logprobs.float().sum()
        for output in outputs
        if output.target_logprobs is not None
    ]
    if not terms:
        raise RuntimeError("no differentiable outputs")
    return torch.stack(terms).sum()


def phase_measure(cell: str, arm: str, output_jsonl: str, repeat: int) -> None:
    """Run one paired-measurement arm on GPU and append evidence rows."""

    phase_contract()
    if cell not in GATES:
        _fail(f"unknown cell {cell!r}; expected one of {sorted(GATES)}")
    if arm not in ("automatic", "depth_one", "full_sharing"):
        _fail(f"unknown arm {arm!r}")
    if arm != "automatic":
        os.environ[TEST_HOOKS_ENV] = "1"
        os.environ[TEST_ANCHOR_ENV] = arm

    import torch
    import torch.distributed as dist

    from art.megatron import train as megatron_train
    from art.trainer_rank import TrainerRank, TrainerRankMemoryError

    if not torch.cuda.is_available():
        _fail("measure phase requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        torch.manual_seed(1234)
        # The gdn-cp4 throughput screen uses exactly two transformer layers
        # (sealed preregistration); the cp1 regression/planning cell runs the
        # full-height model, matching the sealed CP1 acceptance protocol.
        runtime = megatron_train.build_training_runtime(
            model_identifier="Qwen/Qwen3.5-4B",
            provider_configure=(
                (lambda provider: setattr(provider, "num_layers", 2))
                if cell == "grpo-gdn-cp4"
                else None
            ),
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.train()
        rank = TrainerRank(runtime)
        dp_rank, dp_size = rank._dp_rank_and_size()  # noqa: SLF001

        # Behavior smoke from the contract: empty inputs are valid zero-work
        # calls that must not disturb subsequent planning.
        empty = rank.dp_rank_forward([])
        assert len(list(empty)) == 0, "dp_rank_forward([]) must return no outputs"

        rows: list[dict[str, object]] = []
        for sample in range(repeat + 4):  # 1 cold + 3 warmup + repeat measured
            requests = _cell_requests(cell, sample)[dp_rank::dp_size]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            admission_failed = False
            try:
                outputs = rank.dp_rank_forward(requests)
                _output_loss(outputs).backward()
            except TrainerRankMemoryError as error:
                admission_failed = True
                rows.append(
                    {
                        "record_type": "admission_failure",
                        "cell": cell,
                        "arm": arm,
                        "sample_index": sample,
                        "error": str(error),
                    }
                )
            if not admission_failed:
                end.record()
                torch.cuda.synchronize()
                rank.zero_grad()
            # Every rank must join this world collective even after a local
            # admission refusal; skipping it would hang peers under DP > 1.
            peak = torch.tensor(
                [0 if admission_failed else torch.cuda.max_memory_allocated()],
                dtype=torch.long,
                device="cuda",
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if admission_failed:
                continue
            telemetry = getattr(rank, TELEMETRY_METHOD)()
            rows.append(
                {
                    "record_type": "sample",
                    "cell": cell,
                    "arm": arm,
                    "sample_index": sample,
                    "measured": sample >= 4,
                    "complete_call_ms": float(start.elapsed_time(end)),
                    "peak_allocated_bytes": int(peak.item()),
                    "selected_max_depth": int(telemetry[TELEMETRY_DEPTH_KEY]),
                    "planning_ms": float(telemetry[TELEMETRY_PLANNING_MS_KEY]),
                    "world_rank": dist.get_rank(),
                }
            )
        if dist.get_rank() == 0:
            with open(output_jsonl, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"measure phase: wrote {len(rows)} rows to {output_jsonl}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _measured(rows: list[dict[str, Any]], arm: str, key: str) -> list[float]:
    return [
        float(row[key]) for row in rows if row.get("arm") == arm and row.get("measured")
    ]


def phase_validate(cell: str, evidence: str) -> None:
    """Apply the sealed gates to measured evidence rows."""

    gates = GATES.get(cell)
    if gates is None:
        _fail(f"unknown cell {cell!r}; expected one of {sorted(GATES)}")
    rows = [
        json.loads(line)
        for line in Path(evidence).read_text().splitlines()
        if line.strip()
    ]
    admission_failures = [
        row for row in rows if row.get("record_type") == "admission_failure"
    ]
    if admission_failures:
        _fail(
            f"{cell}: {len(admission_failures)} admission failures recorded"
            " (gate: 0 unsafe/failed admissions)"
        )

    def _samples(arm: str) -> dict[int, float]:
        values: dict[int, float] = {}
        for row in rows:
            if row.get("arm") == arm and row.get("measured"):
                index = int(row["sample_index"])
                if index in values:
                    _fail(
                        f"{cell}: duplicate {arm} sample_index {index}; use a"
                        " fresh evidence file per run"
                    )
                values[index] = float(row["complete_call_ms"])
        return values

    automatic_samples = _samples("automatic")
    reference_samples = _samples("depth_one")
    if set(automatic_samples) != set(reference_samples) or not automatic_samples:
        _fail(
            f"{cell}: arms must contain identical measured sample indices"
            f" (automatic={sorted(automatic_samples)},"
            f" depth_one={sorted(reference_samples)})"
        )
    automatic_ms = [automatic_samples[i] for i in sorted(automatic_samples)]
    reference_ms = [reference_samples[i] for i in sorted(reference_samples)]
    paired = [
        100.0 * (reference - automatic) / reference
        for automatic, reference in zip(automatic_ms, reference_ms, strict=True)
    ]
    median_gain = statistics.median(paired)
    planning_fraction = max(
        statistics.median(_measured(rows, arm, "planning_ms"))
        / statistics.median(_measured(rows, arm, "complete_call_ms"))
        for arm in ("automatic", "depth_one")
    )
    if "max_planning_fraction" not in gates:
        planning_fraction = 0.0
    failures: list[str] = []
    if "max_planning_ms" in gates:
        planning_ms = max(
            statistics.median(_measured(rows, arm, "planning_ms"))
            for arm in ("automatic", "depth_one")
        )
        if planning_ms > gates["max_planning_ms"]:
            failures.append(
                f"planning p50 {planning_ms:.1f}ms > {gates['max_planning_ms']}ms gate"
            )
    if "min_paired_median_gain_pct" in gates:
        if median_gain < gates["min_paired_median_gain_pct"]:
            failures.append(
                f"paired median gain {median_gain:.1f}% <"
                f" {gates['min_paired_median_gain_pct']}% gate"
            )
        automatic_peak = statistics.median(
            _measured(rows, "automatic", "peak_allocated_bytes")
        )
        reference_peak = statistics.median(
            _measured(rows, "depth_one", "peak_allocated_bytes")
        )
        peak_reduction = 100.0 * (reference_peak - automatic_peak) / reference_peak
        if peak_reduction < gates["min_peak_reduction_pct"]:
            failures.append(
                f"peak reduction {peak_reduction:.1f}% <"
                f" {gates['min_peak_reduction_pct']}% gate"
            )
        median_depth = statistics.median(
            _measured(rows, "automatic", "selected_max_depth")
        )
        if median_depth < gates["min_median_selected_max_depth"]:
            failures.append(
                f"median selected max depth {median_depth} <"
                f" {gates['min_median_selected_max_depth']} gate"
            )
    if "max_paired_median_regression_pct" in gates:
        if median_gain < -gates["max_paired_median_regression_pct"]:
            failures.append(
                f"paired median regression {median_gain:.2f}% exceeds"
                f" -{gates['max_paired_median_regression_pct']}% gate"
            )
    if planning_fraction > gates.get("max_planning_fraction", 1.0):
        failures.append(
            f"planning fraction {planning_fraction:.3f} >"
            f" {gates['max_planning_fraction']} gate"
        )
    if failures:
        _fail(f"{cell} gates:\n  - " + "\n  - ".join(failures))
    print(
        f"validate phase: PASS ({cell}: median gain {median_gain:.1f}%,"
        f" planning fraction {planning_fraction:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("contract", "census", "measure", "validate"),
    )
    parser.add_argument("--cell", default="grpo-gdn-cp4")
    parser.add_argument(
        "--arm", default="automatic", choices=("automatic", "depth_one", "full_sharing")
    )
    parser.add_argument("--evidence", default="")
    parser.add_argument("--repeat", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        phase_contract()
    elif arguments.phase == "census":
        phase_census()
    elif arguments.phase == "measure":
        if not arguments.evidence:
            _fail("--evidence output path is required for measure")
        phase_measure(
            arguments.cell, arguments.arm, arguments.evidence, arguments.repeat
        )
    else:
        if not arguments.evidence:
            _fail("--evidence input path is required for validate")
        phase_validate(arguments.cell, arguments.evidence)


if __name__ == "__main__":
    main()
