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
import logging
import os
from pathlib import Path
import statistics
import sys
from typing import Any, cast

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


SPLIT_MEMORY_LIMIT_ENV = "ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"


def phase_split_conversion(evidence: str | None, pressure: str) -> None:
    """GPU gate for best-effort internal splitting (sealed cell shape).

    Mirrors the research thread's sealed split-conversion cell: Qwen3.5-4B,
    four transformer layers, CP1, four inputs. ``pressure`` selects how memory
    pressure is induced:

    - ``cap``: the test-only usable-memory cap. Deterministic; exercises the
      control flow, reconstruction, parity and backwardability, but changes
      nothing about what CUDA can actually allocate.
    - ``ballast``: live ballast tensors consume real device memory, so the
      split runs under genuinely reduced headroom and the caller's combined
      backward must fit physically — allocator fragmentation, reserve and
      reusable-cache handling, a backward workspace larger than the forward's
      ephemeral memory, and later-subforward OOM behavior are all real. The
      ballast stays live through the combined backward. Uses no test hooks.

    Gates:
    - unlimited: the call runs unsplit (subforward_count == 1);
    - conversion: under pressure sized between the unsplit and split
      requirements, the call splits (subforward_count >= 2), outputs match the
      unsplit run (parity), and a single combined backward succeeds with every
      graph live. ``cap`` adds a reverse-order per-subforward backward;
      ``ballast`` requires the observed forward+backward peak to stay within
      both the budget the planner admitted against and its predicted peak
      (every retained graph plus the largest subforward's ephemeral share);
    - bounded-decline: under pressure below the smallest single request, the
      call refuses before any model execution with the honest wording.
    """

    phase_contract()
    import torch
    import torch.distributed as dist

    from art.megatron import train as megatron_train
    from art.trainer_rank import ForwardInput, TrainerRank, TrainerRankMemoryError

    if not torch.cuda.is_available():
        _fail("split-conversion phase requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    rows: list[dict[str, object]] = []
    ballast: list[torch.Tensor] = []
    try:
        torch.manual_seed(1234)
        runtime = megatron_train.build_training_runtime(
            model_identifier="Qwen/Qwen3.5-4B",
            provider_configure=lambda provider: setattr(provider, "num_layers", 4),
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.train()
        rank = TrainerRank(runtime)
        if pressure == "cap":
            os.environ[TEST_HOOKS_ENV] = "1"

        def requests() -> list[ForwardInput]:
            generator = torch.Generator().manual_seed(6_101)
            items = []
            for _ in range(4):
                tokens = torch.randint(10, 64_000, (12_288,), generator=generator)
                labels = torch.roll(tokens, shifts=-1).clone()
                labels[-1] = -100
                items.append(ForwardInput(input_tokens=tokens, target_tokens=labels))
            return items

        def forward(
            *, no_grad: bool = False
        ) -> tuple[list[torch.Tensor], dict[str, object]]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            outputs = rank.dp_rank_forward(requests(), no_grad=no_grad)
            telemetry = rank.last_forward_telemetry()
            logprobs = [
                output.target_logprobs.detach().float().clone() for output in outputs
            ]
            return logprobs, {**telemetry, "outputs": outputs}

        def combined_backward(info: dict[str, object]) -> None:
            outputs = cast(list, info["outputs"])
            torch.stack(
                [output.target_logprobs.float().sum() for output in outputs]
            ).sum().backward()
            torch.cuda.synchronize()

        def unsplit_requirement() -> int:
            plan = rank._plan_flat_forward(requests())
            return rank._estimate_required_memory_bytes_from_values(
                packed_tokens=plan.packed_tokens,
                output_bytes=plan.output_bytes,
                signature=plan.signature,
                logical_tokens=plan.logical_tokens,
            )

        def parity(
            reference: list[torch.Tensor], logprobs: list[torch.Tensor]
        ) -> tuple[float, float]:
            # Metric matches dev/trainer_rank_check.py: mean absolute difference
            # relative to the reference's mean magnitude (bf16 kernels reorder
            # reductions across packings), with a loose max-abs backstop.
            split_all = torch.cat([a.reshape(-1) for a in logprobs])
            reference_all = torch.cat([b.reshape(-1) for b in reference])
            mean_abs_pct = float(
                (split_all - reference_all).abs().mean()
                / max(float(reference_all.abs().mean()), 1e-6)
                * 100.0
            )
            max_abs = float((split_all - reference_all).abs().max())
            rows.append(
                {
                    "arm": f"{pressure}-conversion-parity",
                    "mean_abs_pct": mean_abs_pct,
                    "max_abs": max_abs,
                }
            )
            if mean_abs_pct > 0.5 or max_abs > 1.0:
                _fail(
                    "split outputs diverge from unsplit reference:"
                    f" mean_abs_pct={mean_abs_pct:.4f}% max_abs={max_abs:.4f}"
                )
            return mean_abs_pct, max_abs

        def expect_decline(arm: str) -> None:
            torch.cuda.synchronize()
            before = int(torch.cuda.memory_allocated())
            try:
                rank.dp_rank_forward(requests())
            except TrainerRankMemoryError as error:
                message = str(error).lower()
                if "unable to find a feasible split" not in message:
                    _fail(f"decline is not worded as a bounded-search refusal: {error}")
                if int(torch.cuda.memory_allocated()) > before + (1 << 20):
                    _fail("decline allocated model state before refusing")
                rows.append({"arm": arm, "refused": True, "message": str(error)})
            else:
                _fail(f"{arm} arm did not refuse")

        # 1. Unlimited: unsplit reference with combined backward (warms kernels
        # and autotuning, profiles bytes/token and the retained fraction).
        torch.cuda.synchronize()
        baseline_unlimited = int(torch.cuda.memory_allocated())
        reference, info = forward()
        forward_peak_unlimited = int(torch.cuda.max_memory_allocated())
        combined_backward(info)
        rank.zero_grad()
        unsplit_peak = int(torch.cuda.max_memory_allocated())
        rows.append(
            {
                "arm": "unlimited",
                "subforward_count": info["subforward_count"],
                "peak": unsplit_peak,
                "forward_peak_bytes": forward_peak_unlimited - baseline_unlimited,
                "forward_backward_peak_bytes": unsplit_peak - baseline_unlimited,
            }
        )
        if info["subforward_count"] != 1:
            _fail("unlimited arm must run unsplit")
        del info
        # Second unlimited pass so the memory profile is warm.
        forward()
        rank.zero_grad()
        rows.append(
            {
                "arm": "profile",
                "profiles": [
                    {
                        "bytes_per_token": profile.bytes_per_token,
                        "packed_tokens": profile.packed_tokens,
                        "retained_fraction": profile.retained_fraction,
                    }
                    for profile in rank._memory_profiles.values()
                ],
            }
        )

        if pressure == "cap":
            # 2. Conversion: cap the usable budget just below the planner's own
            # unsplit requirement. The unsplit plan then fails admission by
            # construction, while any split whose profiled retained fraction is
            # below 1.0 has a cumulative requirement strictly under the unsplit
            # one (k-way: f*R + (1-f)*R/k < R), so the ladder converts.
            def pressured_cap() -> int:
                # Recompute at call time: the allocator state and the learned
                # memory profile both move between runs.
                required_unsplit = unsplit_requirement()
                torch.cuda.synchronize()
                return int(torch.cuda.memory_allocated()) + required_unsplit - 1

            os.environ[SPLIT_MEMORY_LIMIT_ENV] = str(pressured_cap())
            logprobs, info = forward()
            rows.append(
                {
                    "arm": "cap-conversion",
                    "subforward_count": info["subforward_count"],
                    "cap": os.environ[SPLIT_MEMORY_LIMIT_ENV],
                }
            )
            if cast(int, info["subforward_count"]) < 2:
                _fail(f"conversion arm did not split ({info['subforward_count']=})")
            parity(reference, logprobs)
            combined_backward(info)
            rank.zero_grad()
            del info
            # Reverse-order per-subforward backward with every graph live (the
            # sealed research protocol). Outputs within one subforward share a
            # graph, so each subforward's outputs are reduced to one loss.
            os.environ[SPLIT_MEMORY_LIMIT_ENV] = str(pressured_cap())
            _, info = forward()
            outputs = cast(list, info["outputs"])
            partition = cast(
                tuple[tuple[int, ...], ...], info["subforward_request_indices"]
            )
            if len(partition) < 2:
                _fail("reverse-order arm expected a split plan")
            for indices in reversed(partition):
                torch.stack(
                    [outputs[index].target_logprobs.float().sum() for index in indices]
                ).sum().backward()
            rank.zero_grad()
            del info, outputs

            # 3. Bounded decline: below the smallest single request.
            torch.cuda.synchronize()
            os.environ[SPLIT_MEMORY_LIMIT_ENV] = str(
                int(torch.cuda.memory_allocated()) + 1
            )
            expect_decline("cap-bounded-decline")
            os.environ.pop(SPLIT_MEMORY_LIMIT_ENV, None)
            os.environ.pop(TEST_HOOKS_ENV, None)
        else:

            def add_ballast(target_available: int) -> int:
                """Allocate ballast until the planner's real usable budget is at most target."""

                for _ in range(64):
                    torch.cuda.synchronize()
                    excess = rank._available_memory_bytes() - target_available
                    if excess <= 0:
                        break
                    ballast.append(
                        torch.empty(
                            min(excess, 8 << 30), dtype=torch.uint8, device="cuda"
                        )
                    )
                torch.cuda.synchronize()
                return rank._available_memory_bytes()

            def release_ballast() -> None:
                ballast.clear()
                torch.cuda.synchronize()

            def retained_fraction(*, no_grad: bool) -> float:
                with torch.set_grad_enabled(not no_grad):
                    signature = rank._plan_flat_forward(requests()).signature
                fraction = rank._memory_profiles[signature].retained_fraction
                if fraction is None:
                    _fail("retained fraction was not profiled by the unlimited passes")
                    raise AssertionError("unreachable")
                return fraction

            def conversion(
                arm: str,
                *,
                no_grad: bool,
                reference: list[torch.Tensor],
                target_available: int,
            ) -> None:
                available = add_ballast(target_available)
                device_free, device_total = torch.cuda.mem_get_info()
                torch.cuda.synchronize()
                baseline = int(torch.cuda.memory_allocated())
                reserved_baseline = int(torch.cuda.memory_reserved())
                logprobs, info = forward(no_grad=no_grad)
                forward_peak = int(torch.cuda.max_memory_allocated()) - baseline
                if cast(int, info["subforward_count"]) < 2:
                    _fail(f"{arm} did not split ({info['subforward_count']=})")
                mean_abs_pct, max_abs = parity(reference, logprobs)
                if not no_grad:
                    combined_backward(info)  # every graph live, ballast live
                total_peak = int(torch.cuda.max_memory_allocated()) - baseline
                reserved_peak = (
                    int(torch.cuda.max_memory_reserved()) - reserved_baseline
                )
                predicted = cast(int, info["predicted_peak_bytes"])
                budget = cast(int, info["usable_limit_bytes"])
                rows.append(
                    {
                        "arm": arm,
                        "ballast_bytes": sum(int(b.numel()) for b in ballast),
                        "available_bytes": available,
                        "device_free_bytes": int(device_free),
                        "device_total_bytes": int(device_total),
                        "subforward_count": info["subforward_count"],
                        "subforward_request_indices": info[
                            "subforward_request_indices"
                        ],
                        "predicted_peak_bytes": predicted,
                        "usable_limit_bytes": budget,
                        "forward_peak_bytes": forward_peak,
                        "forward_backward_peak_bytes": total_peak,
                        "reserved_growth_bytes": reserved_peak,
                        "mean_abs_pct": mean_abs_pct,
                        "max_abs": max_abs,
                    }
                )
                if total_peak > budget:
                    _fail(
                        f"{arm}: observed peak exceeded the budget the planner admitted"
                        f" against: {total_peak} > {budget}"
                    )
                if total_peak > predicted:
                    _fail(
                        f"{arm}: observed peak exceeded the predicted peak (all retained"
                        f" graphs plus the largest ephemeral share): {total_peak} >"
                        f" {predicted}; the backward headroom heuristic did not hold"
                    )
                if not no_grad:
                    rank.zero_grad()
                release_ballast()

            # 2. Training-forward conversion under real pressure. Every graph is
            # retained for backward, so with retained fraction f a 2-way split
            # needs f*R + (1-f)*R/2 against the unsplit requirement R: the
            # conversion window is (1-f)*R/2 wide. Size the ballast from the
            # measured f (midway inside the window) and report the width.
            required_unsplit = unsplit_requirement()
            fraction = retained_fraction(no_grad=False)
            predicted_two_way = int(
                fraction * required_unsplit + (1.0 - fraction) * required_unsplit / 2
            )
            rows.append(
                {
                    "arm": "ballast-window",
                    "required_unsplit_bytes": required_unsplit,
                    "retained_fraction": fraction,
                    "predicted_two_way_bytes": predicted_two_way,
                    "window_bytes": required_unsplit - predicted_two_way,
                }
            )
            if predicted_two_way >= required_unsplit:
                _fail("no conversion window: the profile retains everything")
            conversion(
                "ballast-conversion",
                no_grad=False,
                reference=reference,
                target_available=(predicted_two_way + required_unsplit) // 2,
            )

            # 3. Bounded decline under real pressure: below one request's share.
            add_ballast(required_unsplit // 8)
            expect_decline("ballast-bounded-decline")
            release_ballast()

            # 4. no_grad conversion under real pressure: nothing but the outputs
            # is retained, so splitting pays in proportion. Budget = 60% of the
            # no_grad unsplit requirement; a 2-way split must fit with margin.
            reference_no_grad, info = forward(no_grad=True)
            if info["subforward_count"] != 1:
                _fail("unlimited no_grad arm must run unsplit")
            del info
            forward(no_grad=True)
            with torch.no_grad():
                required_no_grad = unsplit_requirement()
            fraction_no_grad = retained_fraction(no_grad=True)
            rows.append(
                {
                    "arm": "ballast-no-grad-window",
                    "required_unsplit_bytes": required_no_grad,
                    "retained_fraction": fraction_no_grad,
                }
            )
            conversion(
                "ballast-no-grad-conversion",
                no_grad=True,
                reference=reference_no_grad,
                target_available=required_no_grad * 3 // 5,
            )
            rows.append(
                {
                    "arm": "profile",
                    "profiles": [
                        {
                            "grad_enabled": signature.grad_enabled,
                            "bytes_per_token": profile.bytes_per_token,
                            "packed_tokens": profile.packed_tokens,
                            "retained_fraction": profile.retained_fraction,
                        }
                        for signature, profile in rank._memory_profiles.items()
                    ],
                }
            )

        if evidence and dist.get_rank() == 0:
            with open(evidence, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"split-conversion phase ({pressure}): PASS {rows}")
    finally:
        ballast.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


# --- Tensor-parallel gates ---------------------------------------------------
#
# Written before lifting the TP>1 constructor refusal (test-first): both GPU
# phases below fail on the pre-lift tree at ``TrainerRank(runtime)``.
#
# ``tp2-public`` (2 ranks, DP1 x TP2 x CP1) is the first public-API execution
# of the automatic planner under tensor parallelism: planner-selected prefix
# sharing -> GDN prefix state -> sequence parallelism -> TP output head ->
# backward through an active LoRA slot, compared against the depth-one arm.
# ``dp2-tp2-waves`` (4 ranks, DP2 x TP2) exercises the global wave planner's
# collectives (world scope) composed with TP execution collectives (pair scope)
# through public ``forward_micro_batches``, including an empty DP slot.

TP_GATES: dict[str, float] = {
    # bf16 kernels reorder reductions across packings (same metric/tolerance as
    # the split-conversion gate and dev/trainer_rank_check.py).
    "max_output_mean_abs_pct": 0.5,
    "max_output_abs": 1.0,
    "max_loss_rel_pct": 0.5,
    # LoRA gradients across two different packings of the same tokens.
    "max_grad_rel_l2": 2e-2,
    "min_grad_cosine": 0.999,
    # Plan-cache-stable measured rows: identical content plans as a cache hit.
    "max_measured_planning_ms": 10.0,
    # Cross-layout divergence at TP may not exceed this multiple of the TP1
    # control's divergence (and is ignored below an absolute floor).
    "max_cross_layout_ratio": 1.5,
}
# Cross-layout divergence below these absolute floors is never gated.
TP_CROSS_LAYOUT_FLOOR: dict[str, float] = {"mean_abs_pct": 0.05, "grad_rel_l2": 0.005}

# Hierarchical GRPO shape for the TP2 public cell: (groups, group size, system,
# prompt, completion). The odd system and completion lengths make every
# planner layout's packed length odd, so sequence-parallel padding
# (local_length % TP != 0) is exercised with outputs at the final real token.
GRPO_TP2 = (2, 8, 1023, 2048, 255)


class _ForwardCompileWatch(logging.Handler):
    """Collect per-forward compile status from the trainer telemetry log."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.statuses: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        marker = "ART_TRAINER_EVENT "
        if marker not in message:
            return
        try:
            event = json.loads(message.split(marker, 1)[1])
        except json.JSONDecodeError:
            return
        if event.get("event") == "phase" and event.get("phase") == "forward":
            self.statuses.append(str(event.get("compile_status", "unknown")))

    def take(self) -> list[str]:
        statuses, self.statuses = self.statuses, []
        return statuses


def _install_compile_watch() -> _ForwardCompileWatch:
    logger = logging.getLogger("art.trainer_rank._telemetry")
    logger.setLevel(logging.INFO)
    watch = _ForwardCompileWatch()
    logger.addHandler(watch)
    return watch


def _grpo_groups(seed: int, shape: tuple[int, int, int, int, int]) -> list[list[Any]]:
    """Hierarchical GRPO groups: one nested item per prompt group."""

    import torch

    from art.trainer_rank import ForwardInput

    prompt_groups, group_size, system_tokens, prompt_tokens, completion_tokens = shape

    def _tokens(token_seed: int, count: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(token_seed)
        return torch.randint(low=10, high=64_000, size=(count,), generator=generator)

    system = _tokens(seed * 100_003, system_tokens)
    groups: list[list[Any]] = []
    for prompt_group in range(prompt_groups):
        prompt = _tokens(seed * 1_000_003 + prompt_group * 100_019, prompt_tokens)
        prefix = torch.cat((system, prompt))
        requests: list[Any] = []
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
        groups.append(requests)
    return groups


def _anchor_env(arm: str) -> None:
    if arm == "automatic":
        os.environ.pop(TEST_ANCHOR_ENV, None)
        os.environ.pop(TEST_HOOKS_ENV, None)
    else:
        os.environ[TEST_HOOKS_ENV] = "1"
        os.environ[TEST_ANCHOR_ENV] = arm


def _tp_group() -> Any:
    from megatron.core import parallel_state as ps

    return ps.get_tensor_model_parallel_group(check_initialized=False)


def _gather_objects(value: Any, group: Any = None) -> list[Any]:
    import torch.distributed as dist

    size = dist.get_world_size(group)
    values: list[Any] = [None] * size
    dist.all_gather_object(values, value, group=group)
    return values


def _plan_fingerprint(
    rank: Any, requests: list[Any], checkpoint: Any
) -> dict[str, Any]:
    """Content fingerprint of the plan this rank would execute (a cache hit)."""

    import hashlib

    plan = rank._plan_flat_forward(requests, checkpoint=checkpoint)
    telemetry = rank.last_forward_telemetry()
    return {
        "packed_tokens": int(plan.packed_tokens),
        "logical_tokens": int(plan.logical_tokens),
        "selected_max_depth": int(plan.selected_max_depth),
        "subforward_request_indices": telemetry["subforward_request_indices"],
        "groups": [
            {
                "tokens": hashlib.sha256(
                    group.packed.tokens.cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "group_ids": hashlib.sha256(
                    group.packed.group_ids.cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "segments": len(group.packed.segments),
            }
            for group in plan.groups
        ],
    }


def _assert_tp_peers_agree(fingerprint: dict[str, object], label: str) -> None:
    peers = _gather_objects(fingerprint, _tp_group())
    if any(peer != peers[0] for peer in peers):
        _fail(f"{label}: TP peers planned different physical layouts: {peers}")


def _lora_gradients(rank: Any, slot: str) -> list[Any]:
    import torch

    return [
        torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        if parameter.grad is None
        else parameter.grad.detach().float().cpu().clone()
        for parameter in rank._checkpoint_slots[slot].params
    ]


def _compare_gradients(actual: list[Any], expected: list[Any]) -> tuple[float, float]:
    """Return (relative L2 error, cosine similarity) over all LoRA gradients."""

    import torch

    flat_actual = torch.cat([g.reshape(-1).double() for g in actual])
    flat_expected = torch.cat([g.reshape(-1).double() for g in expected])
    denominator = max(float(flat_expected.norm()), 1e-300)
    rel_l2 = float((flat_actual - flat_expected).norm()) / denominator
    cosine = float(
        (flat_actual @ flat_expected)
        / max(float(flat_actual.norm()) * float(flat_expected.norm()), 1e-300)
    )
    return rel_l2, cosine


def _compare_logprobs(actual: list[Any], expected: list[Any]) -> tuple[float, float]:
    import torch

    flat_actual = torch.cat([a.reshape(-1) for a in actual])
    flat_expected = torch.cat([b.reshape(-1) for b in expected])
    mean_abs_pct = float(
        (flat_actual - flat_expected).abs().mean()
        / max(float(flat_expected.abs().mean()), 1e-6)
        * 100.0
    )
    return mean_abs_pct, float((flat_actual - flat_expected).abs().max())


def _check_pair(
    rows: list[dict[str, object]],
    label: str,
    automatic: dict[str, Any],
    depth_one: dict[str, Any],
) -> None:
    """Gate automatic-vs-depth-one outputs, loss and LoRA gradients."""

    mean_abs_pct, max_abs = _compare_logprobs(
        automatic["logprobs"], depth_one["logprobs"]
    )
    loss_rel_pct = (
        abs(automatic["loss"] - depth_one["loss"])
        / max(abs(depth_one["loss"]), 1e-6)
        * 100.0
    )
    grad_rel_l2, grad_cosine = _compare_gradients(
        automatic["gradients"], depth_one["gradients"]
    )
    rows.append(
        {
            "arm": f"{label}-parity",
            "mean_abs_pct": mean_abs_pct,
            "max_abs": max_abs,
            "loss_rel_pct": loss_rel_pct,
            "grad_rel_l2": grad_rel_l2,
            "grad_cosine": grad_cosine,
        }
    )
    problems = []
    if mean_abs_pct > TP_GATES["max_output_mean_abs_pct"]:
        problems.append(f"output mean_abs_pct={mean_abs_pct:.4f}")
    if max_abs > TP_GATES["max_output_abs"]:
        problems.append(f"output max_abs={max_abs:.4f}")
    if loss_rel_pct > TP_GATES["max_loss_rel_pct"]:
        problems.append(f"loss_rel_pct={loss_rel_pct:.4f}")
    if grad_rel_l2 > TP_GATES["max_grad_rel_l2"]:
        problems.append(f"grad_rel_l2={grad_rel_l2:.5f}")
    if grad_cosine < TP_GATES["min_grad_cosine"]:
        problems.append(f"grad_cosine={grad_cosine:.6f}")
    if problems:
        _fail(f"{label}: automatic vs depth-one diverge: " + ", ".join(problems))


def _init_gpu_phase(name: str) -> None:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        _fail(f"{name} phase requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")


def _require_topology(rank: Any, *, tp: int, dp: int) -> None:
    from megatron.core import parallel_state as ps

    actual_tp = int(ps.get_tensor_model_parallel_world_size())
    actual_dp = int(ps.get_data_parallel_world_size())
    if (actual_tp, actual_dp) != (tp, dp):
        _fail(
            f"expected TP{tp} x DP{dp}, runtime initialized TP{actual_tp} x DP{actual_dp}"
        )


def _write_rows(evidence: str | None, rows: list[dict[str, object]], name: str) -> None:
    import torch.distributed as dist

    if evidence and dist.get_rank() == 0:
        with open(evidence, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if dist.get_rank() == 0:
        print(f"{name} phase: PASS {rows}")


def phase_tp2_public(
    evidence: str | None, repeat: int, *, tp: int, dump_dir: str | None
) -> None:
    """DP1 x TP{tp} x CP1 public ``dp_rank_forward`` cell (Qwen3.5-4B full model).

    Run at ``--tp 2`` (the gate) and at ``--tp 1`` (the control: the identical
    cell on one GPU). Structural gates run here; the numerics gates compare
    the two dumps in ``--phase tp-compare`` (CPU), because bf16 divergence
    between two packings of the same tokens is only meaningful relative to
    the TP1 control, and TP correctness itself is a same-layout question.

    Gates:
    - all TP peers plan the same physical layout on every call;
    - the automatic planner shares more deeply than depth-one on the
      hierarchical GRPO shape (fewer packed tokens);
    - packed lengths are odd, so sequence-parallel padding is exercised at
      TP>1 with outputs at the final real token;
    - the loss is finite for both arms;
    - measured rows are compile-free and plan-cache-stable (3 warmups per arm,
      then ``repeat`` alternating measured pairs); paired timing is reported.
    Rows are printed as they are produced and written even on failure.
    """

    phase_contract()
    _init_gpu_phase("tp2-public")
    import math
    import statistics

    import torch
    import torch.distributed as dist

    from art.megatron import train as megatron_train
    from art.trainer_rank import TrainerRank

    sys.path.insert(0, str(REPO_ROOT / "dev"))
    from trainer_rank_support import load_random_checkpoints

    label = f"tp{tp}-public"
    rows: list[dict[str, object]] = []

    def note(row: dict[str, object]) -> None:
        rows.append(row)
        if dist.get_rank() == 0:
            print(
                f"{label} row: {json.dumps(row, sort_keys=True, default=str)}",
                flush=True,
            )

    try:
        torch.manual_seed(1234)
        runtime = megatron_train.build_training_runtime(
            model_identifier="Qwen/Qwen3.5-4B",
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.train()
        rank = TrainerRank(runtime)
        _require_topology(rank, tp=tp, dp=1)
        [slot] = load_random_checkpoints(runtime, rank, 1, base_model="Qwen/Qwen3.5-4B")
        watch = _install_compile_watch()
        requests = [
            request for group in _grpo_groups(6_301, GRPO_TP2) for request in group
        ]

        def run(arm: str) -> dict[str, Any]:
            _anchor_env(arm)
            watch.take()
            rank.zero_grad()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = rank.dp_rank_forward(requests, checkpoint=slot)
            loss = _output_loss(outputs)
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            telemetry = rank.last_forward_telemetry()
            fingerprint = _plan_fingerprint(rank, requests, slot)
            _assert_tp_peers_agree(fingerprint, f"{label} {arm}")
            result = {
                "arm": arm,
                "ms": float(start.elapsed_time(end)),
                "loss": float(loss.detach().float().item()),
                "logprobs": [o.target_logprobs.detach().float().cpu() for o in outputs],
                "gradients": _lora_gradients(rank, slot),
                "selected_max_depth": int(telemetry["selected_max_depth"]),
                "planning_ms": float(telemetry["planning_ms"]),
                "packed_tokens": int(fingerprint["packed_tokens"]),
                "compile_statuses": watch.take(),
            }
            del outputs, loss
            rank.zero_grad()
            _anchor_env("automatic")
            return result

        # Warmups (compile, autotune, memory profile) for both arms.
        for _ in range(3):
            for arm in ("automatic", "depth_one"):
                run(arm)
        # Correctness runs: both arms, plus a repeat of automatic for the
        # same-layout run-to-run noise floor (bf16 backward is nondeterministic).
        automatic = run("automatic")
        depth_one = run("depth_one")
        automatic_again = run("automatic")
        noise_out = _compare_logprobs(
            automatic_again["logprobs"], automatic["logprobs"]
        )
        noise_grad = _compare_gradients(
            automatic_again["gradients"], automatic["gradients"]
        )
        note(
            {
                "arm": f"{label}-layouts",
                "tp": tp,
                "automatic_selected_max_depth": automatic["selected_max_depth"],
                "automatic_packed_tokens": automatic["packed_tokens"],
                "depth_one_selected_max_depth": depth_one["selected_max_depth"],
                "depth_one_packed_tokens": depth_one["packed_tokens"],
                "logical_tokens": sum(int(r.input_tokens.numel()) for r in requests),
                "automatic_loss": automatic["loss"],
                "depth_one_loss": depth_one["loss"],
                "same_layout_noise_output_mean_abs_pct": noise_out[0],
                "same_layout_noise_output_max_abs": noise_out[1],
                "same_layout_noise_grad_rel_l2": noise_grad[0],
                "same_layout_noise_grad_cosine": noise_grad[1],
            }
        )
        cross_out = _compare_logprobs(automatic["logprobs"], depth_one["logprobs"])
        cross_grad = _compare_gradients(automatic["gradients"], depth_one["gradients"])
        note(
            {
                "arm": f"{label}-cross-layout",
                "tp": tp,
                "mean_abs_pct": cross_out[0],
                "max_abs": cross_out[1],
                "loss_rel_pct": abs(automatic["loss"] - depth_one["loss"])
                / max(abs(depth_one["loss"]), 1e-6)
                * 100.0,
                "grad_rel_l2": cross_grad[0],
                "grad_cosine": cross_grad[1],
            }
        )
        if automatic["packed_tokens"] >= depth_one["packed_tokens"]:
            _fail(
                "automatic planner did not share more deeply than depth-one "
                f"({automatic['packed_tokens']} vs {depth_one['packed_tokens']} packed tokens)"
            )
        if tp > 1 and (
            automatic["packed_tokens"] % tp == 0 or depth_one["packed_tokens"] % tp == 0
        ):
            _fail("cell shape did not exercise sequence-parallel padding")
        if not (math.isfinite(automatic["loss"]) and math.isfinite(depth_one["loss"])):
            _fail("non-finite loss")
        if dump_dir and dist.get_rank() == 0:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "tp": tp,
                    "arms": {
                        arm: {
                            "logprobs": result["logprobs"],
                            "loss": result["loss"],
                            "packed_tokens": result["packed_tokens"],
                            "selected_max_depth": result["selected_max_depth"],
                        }
                        for arm, result in (
                            ("automatic", automatic),
                            ("depth_one", depth_one),
                        )
                    },
                    "noise": {
                        "output_mean_abs_pct": noise_out[0],
                        "output_max_abs": noise_out[1],
                        "grad_rel_l2": noise_grad[0],
                        "grad_cosine": noise_grad[1],
                    },
                    "cross_layout": {
                        "mean_abs_pct": cross_out[0],
                        "max_abs": cross_out[1],
                        "grad_rel_l2": cross_grad[0],
                        "grad_cosine": cross_grad[1],
                    },
                },
                Path(dump_dir, f"tp{tp}_public.pt"),
            )

        # Alternating measured pairs: compile-free and plan-cache-stable.
        samples: dict[str, list[float]] = {"automatic": [], "depth_one": []}
        for _ in range(repeat):
            for arm in ("automatic", "depth_one"):
                result = run(arm)
                samples[arm].append(result["ms"])
                if any(status != "none" for status in result["compile_statuses"]):
                    _fail(f"measured {arm} row compiled: {result['compile_statuses']}")
                if result["planning_ms"] > TP_GATES["max_measured_planning_ms"]:
                    _fail(
                        f"measured {arm} row was not a plan-cache hit: {result['planning_ms']:.2f} ms"
                    )
                reference = automatic if arm == "automatic" else depth_one
                if result["packed_tokens"] != reference["packed_tokens"]:
                    _fail(f"measured {arm} row changed layout")
        paired_gain = [
            (d - a) / d * 100.0
            for a, d in zip(samples["automatic"], samples["depth_one"], strict=True)
        ]
        note(
            {
                "arm": f"{label}-timing",
                "tp": tp,
                "automatic_median_ms": statistics.median(samples["automatic"]),
                "depth_one_median_ms": statistics.median(samples["depth_one"]),
                "paired_median_gain_pct": statistics.median(paired_gain),
                "measured_pairs": repeat,
            }
        )
        _write_rows(evidence, rows, label)
    except SystemExit:
        if evidence and dist.get_rank() == 0:
            with open(evidence, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _difference_profile(actual: list[Any], expected: list[Any]) -> dict[str, float]:
    """Per-target-token difference statistics (masked positions are exact zeros)."""

    import torch

    per_request: list[float] = []
    tails: list[float] = []
    bodies: list[float] = []
    signed: list[float] = []
    for a, b in zip(actual, expected, strict=True):
        a = a.reshape(-1).double()
        b = b.reshape(-1).double()
        mask = (a != 0) & (b != 0)
        d = (a - b)[mask]
        if int(d.numel()) < 8:
            continue
        per_request.append(float(d.abs().mean()))
        tails.append(float(d[-4:].abs().mean()))
        bodies.append(float(d[:-4].abs().mean()))
        signed.append(float(d.mean()))
    if not per_request:
        _fail("difference profile: no unmasked target tokens")
    ordered = sorted(per_request)
    return {
        "mean_abs_per_token": sum(per_request) / len(per_request),
        "max_request_mean_abs": ordered[-1],
        "median_request_mean_abs": ordered[len(ordered) // 2],
        "tail_mean_abs": sum(tails) / len(tails),
        "body_mean_abs": sum(bodies) / len(bodies),
        "signed_mean": sum(signed) / len(signed),
    }


def phase_tp_compare(control_dump: str, dump: str, evidence: str | None) -> None:
    """CPU numerics gates for the TP public cell against its TP1 control.

    bf16 rounding differs whenever the reduction order changes, and both a
    different packing (cross-layout) and a different tensor-parallel degree
    (sharded GEMMs, bf16 all-reduces) change it. Over 36 layers that noise is
    about 1.3% of the mean logprob magnitude on this cell, so absolute
    tolerances borrowed from same-packing comparisons cannot separate a TP
    defect from noise. The reference is therefore the control's own
    cross-layout divergence (automatic vs depth-one at TP1), measured on the
    identical cell in the same run:

    - same-layout TP correctness: for each arm, TP-vs-TP1 output divergence
      (mean and max) must stay within ``max_cross_layout_ratio`` of the
      control's cross-layout divergence, and the losses must agree;
    - cross-layout divergence at TP (outputs and LoRA gradients) must stay
      within the same ratio of the control's;
    - no structured error: no request's mean difference exceeds twice the
      median request's, the last four target tokens (where sequence-parallel
      padding sits) differ no more than twice the body, and the signed mean
      difference is small relative to the absolute one.
    """

    import torch

    control = torch.load(control_dump, weights_only=False)
    trial = torch.load(dump, weights_only=False)
    if int(control["tp"]) != 1:
        _fail(f"control dump must be TP1 (got tp={control['tp']})")
    ratio = TP_GATES["max_cross_layout_ratio"]
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    reference_mean = float(control["cross_layout"]["mean_abs_pct"])
    reference_max = float(control["cross_layout"]["max_abs"])
    reference_profile = _difference_profile(
        control["arms"]["automatic"]["logprobs"],
        control["arms"]["depth_one"]["logprobs"],
    )
    rows.append(
        {
            "arm": "tp1-cross-layout-reference",
            **control["cross_layout"],
            **reference_profile,
        }
    )

    def structured(label: str, profile: dict[str, float]) -> None:
        if profile["max_request_mean_abs"] > 2.0 * profile["median_request_mean_abs"]:
            problems.append(
                f"{label}: one request diverges ({profile['max_request_mean_abs']:.3f} vs "
                f"median {profile['median_request_mean_abs']:.3f})"
            )
        if profile["tail_mean_abs"] > 2.0 * max(profile["body_mean_abs"], 1e-9):
            problems.append(
                f"{label}: final tokens diverge ({profile['tail_mean_abs']:.3f} vs body "
                f"{profile['body_mean_abs']:.3f})"
            )
        if abs(profile["signed_mean"]) > 0.25 * profile["mean_abs_per_token"]:
            problems.append(
                f"{label}: biased differences (signed {profile['signed_mean']:+.4f} vs "
                f"abs {profile['mean_abs_per_token']:.4f})"
            )

    for arm in ("automatic", "depth_one"):
        c, tr = control["arms"][arm], trial["arms"][arm]
        if c["packed_tokens"] != tr["packed_tokens"]:
            problems.append(
                f"{arm}: TP{trial['tp']} packed {tr['packed_tokens']} tokens vs TP1 "
                f"{c['packed_tokens']} (layouts differ)"
            )
            continue
        mean_abs_pct, max_abs = _compare_logprobs(tr["logprobs"], c["logprobs"])
        loss_rel_pct = abs(tr["loss"] - c["loss"]) / max(abs(c["loss"]), 1e-6) * 100.0
        profile = _difference_profile(tr["logprobs"], c["logprobs"])
        rows.append(
            {
                "arm": f"tp{trial['tp']}-vs-tp1-{arm}",
                "packed_tokens": tr["packed_tokens"],
                "mean_abs_pct": mean_abs_pct,
                "max_abs": max_abs,
                "loss_rel_pct": loss_rel_pct,
                "mean_abs_pct_ratio_to_reference": mean_abs_pct
                / max(reference_mean, 1e-12),
                **profile,
            }
        )
        if mean_abs_pct > ratio * reference_mean:
            problems.append(
                f"{arm}: same-layout output divergence {mean_abs_pct:.4f}% exceeds "
                f"{ratio}x the TP1 cross-layout reference {reference_mean:.4f}%"
            )
        if max_abs > ratio * reference_max:
            problems.append(
                f"{arm}: same-layout max_abs {max_abs:.3f} exceeds {ratio}x the reference {reference_max:.3f}"
            )
        if loss_rel_pct > TP_GATES["max_loss_rel_pct"]:
            problems.append(f"{arm}: same-layout loss_rel_pct={loss_rel_pct:.4f}")
        structured(f"tp{trial['tp']}-vs-tp1-{arm}", profile)

    trial_profile = _difference_profile(
        trial["arms"]["automatic"]["logprobs"], trial["arms"]["depth_one"]["logprobs"]
    )
    cross: dict[str, object] = {
        "arm": f"tp{trial['tp']}-cross-layout-vs-control",
        **trial_profile,
    }
    for key in ("mean_abs_pct", "grad_rel_l2"):
        c, tr = float(control["cross_layout"][key]), float(trial["cross_layout"][key])
        cross[f"{key}_control"] = c
        cross[f"{key}_trial"] = tr
        cross[f"{key}_ratio"] = tr / max(c, 1e-12)
        if tr > ratio * c and tr > TP_CROSS_LAYOUT_FLOOR[key]:
            problems.append(
                f"cross-layout {key} at TP{trial['tp']} is {tr:.5f} vs {c:.5f} at TP1 "
                f"(ratio {tr / max(c, 1e-12):.2f} > {ratio})"
            )
    cross["control_noise"] = control["noise"]
    cross["trial_noise"] = trial["noise"]
    rows.append(cross)
    structured(f"tp{trial['tp']}-cross-layout", trial_profile)
    if evidence:
        with open(evidence, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    for row in rows:
        print(json.dumps(row, sort_keys=True, default=str))
    if problems:
        _fail("tp-compare:\n  - " + "\n  - ".join(problems))
    print("tp-compare phase: PASS")


def phase_dp2_tp2_waves(evidence: str | None) -> None:
    """DP2 x TP2 public ``forward_micro_batches`` gate (4 ranks, Qwen3.5-4B).

    Arm A (branchy): six hierarchical GRPO groups as top-level items, the
    test-only memory cap sized so the stream needs at least two waves, forward
    and backward per wave through an active LoRA slot. Gates: every global input
    is returned exactly once in original order; both ranks of each TP pair
    execute identical wave shapes while DP replicas carry different payloads;
    the automatic planner selects depth > 1; automatic vs depth-one outputs,
    loss and LoRA gradients agree; no collective hangs (the run completes).

    Arm B (empty slot): a single heterogeneous item, so one DP replica has
    nothing to do in the only wave and must still complete the protocol.
    """

    phase_contract()
    _init_gpu_phase("dp2-tp2-waves")
    import torch
    import torch.distributed as dist

    from art.megatron import train as megatron_train
    from art.trainer_rank import ForwardInput, TrainerRank

    sys.path.insert(0, str(REPO_ROOT / "dev"))
    from trainer_rank_support import load_random_checkpoints

    rows: list[dict[str, object]] = []
    try:
        torch.manual_seed(1234)
        runtime = megatron_train.build_training_runtime(
            model_identifier="Qwen/Qwen3.5-4B",
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.train()
        rank = TrainerRank(runtime)
        _require_topology(rank, tp=2, dp=2)
        dp_rank, dp_size = rank._dp_rank_and_size()
        [slot] = load_random_checkpoints(runtime, rank, 1, base_model="Qwen/Qwen3.5-4B")
        items = [
            group
            for seed in range(6)
            for group in _grpo_groups(7_001 + seed, (1, 8, 1023, 2048, 255))
        ]

        def stream(arm: str, cap_bytes: int | None) -> dict[str, Any]:
            _anchor_env(arm)
            if cap_bytes is None:
                os.environ.pop(SPLIT_MEMORY_LIMIT_ENV, None)
            else:
                os.environ[TEST_HOOKS_ENV] = "1"
                os.environ[SPLIT_MEMORY_LIMIT_ENV] = str(cap_bytes)
            rank.zero_grad()
            seen: list[int] = []
            wave_shapes: list[dict[str, object]] = []
            logprobs: dict[int, list[torch.Tensor]] = {}
            loss_total = 0.0
            depths: list[int] = []
            for batch in rank.forward_micro_batches(items, checkpoint=slot):
                seen.extend(int(index) for index in batch.indices)
                telemetry = rank.last_forward_telemetry()
                depths.append(int(telemetry["selected_max_depth"]))
                wave_shapes.append(
                    {
                        "global_start": batch.stats.global_start,
                        "global_stop": batch.stats.global_stop,
                        "global_count": batch.stats.global_count,
                        "local_count": batch.stats.local_count,
                        "packed_tokens": batch.stats.packed_tokens,
                        "logical_tokens": batch.stats.logical_tokens,
                        "selected_max_depth": int(telemetry["selected_max_depth"]),
                        "subforward_count": batch.stats.subforward_count,
                    }
                )
                flat = [output for group in batch.outputs for output in group]
                for index, group in zip(batch.indices, batch.outputs, strict=True):
                    logprobs[int(index)] = [
                        o.target_logprobs.detach().float().cpu() for o in group
                    ]
                if flat:
                    loss = _output_loss(flat)
                    loss_total += float(loss.detach().float().item())
                    loss.backward()
            torch.cuda.synchronize()
            result = {
                "arm": arm,
                "seen": seen,
                "waves": wave_shapes,
                "loss": loss_total,
                "logprobs": logprobs,
                "gradients": _lora_gradients(rank, slot),
                "depths": depths,
            }
            rank.zero_grad()
            _anchor_env("automatic")
            os.environ.pop(SPLIT_MEMORY_LIMIT_ENV, None)
            return result

        # Warm-up: one uncapped automatic stream (compile, profile).
        stream("automatic", None)
        # Cap so at most two items fit per DP rank per wave (three waves).
        plan = rank._plan_flat_forward(items[0], checkpoint=slot)
        per_item = rank._estimate_required_memory_bytes_from_values(
            packed_tokens=plan.packed_tokens,
            output_bytes=plan.output_bytes,
            signature=plan.signature,
            logical_tokens=plan.logical_tokens,
        )
        torch.cuda.synchronize()
        cap = int(torch.cuda.memory_allocated()) + int(per_item * 2.5)
        cap_all = _gather_objects(cap)
        cap = min(cap_all)  # one cap on every rank; MIN-reduced budgets anyway

        automatic = stream("automatic", cap)
        depth_one = stream("depth_one", cap)

        # Every global input exactly once, in original order, across DP ranks
        # (ranks in one TP pair hold the same slice, so dedupe by DP rank).
        for result in (automatic, depth_one):
            by_dp: dict[int, list[int]] = {}
            for peer_dp, seen in _gather_objects((dp_rank, result["seen"])):
                by_dp.setdefault(int(peer_dp), list(seen))
            merged = sorted(index for seen in by_dp.values() for index in seen)
            if merged != list(range(len(items))):
                _fail(f"{result['arm']}: inputs not returned exactly once: {merged}")
            if any(seen != sorted(seen) for seen in by_dp.values()):
                _fail(
                    f"{result['arm']}: a DP rank returned inputs out of order: {by_dp}"
                )
            if len(result["waves"]) < 2:
                _fail(
                    f"{result['arm']}: expected at least two waves, got {len(result['waves'])}"
                )
            peers = _gather_objects(result["waves"], _tp_group())
            if any(peer != peers[0] for peer in peers):
                _fail(
                    f"{result['arm']}: TP peers executed different wave shapes: {peers}"
                )
            if len({tuple(seen) for seen in by_dp.values()}) < len(by_dp):
                _fail(
                    "DP replicas received identical payloads; the gate needs distinct slices"
                )
        if not all(
            wave["packed_tokens"] < wave["logical_tokens"]
            for wave in automatic["waves"]
        ):
            _fail(
                f"automatic planner shared nothing in some wave: {automatic['waves']}"
            )
        rows.append(
            {
                "arm": "dp2-tp2-branchy",
                "waves": len(automatic["waves"]),
                "wave_shapes": automatic["waves"],
                "depth_one_waves": len(depth_one["waves"]),
                "local_indices": sorted(automatic["seen"]),
                "cap_bytes": cap,
            }
        )
        # Parity per item held locally (both arms hold the same local slice).
        local = sorted(automatic["logprobs"])
        if local != sorted(depth_one["logprobs"]):
            _fail("arms held different local item sets")
        _check_pair(
            rows,
            "dp2-tp2-branchy",
            {
                **automatic,
                "logprobs": [
                    t for index in local for t in automatic["logprobs"][index]
                ],
            },
            {
                **depth_one,
                "logprobs": [
                    t for index in local for t in depth_one["logprobs"][index]
                ],
            },
        )

        # Arm B: empty DP slot.
        single = [
            ForwardInput(
                input_tokens=(
                    tokens := torch.randint(
                        10,
                        64_000,
                        (1_537,),
                        generator=torch.Generator().manual_seed(9_301),
                    )
                ),
                target_tokens=torch.cat((tokens[1:], tokens.new_tensor([-100]))),
            )
        ]
        rank.zero_grad()
        waves = 0
        local_outputs = 0
        for batch in rank.forward_micro_batches([single], checkpoint=slot):
            waves += 1
            flat = [output for group in batch.outputs for output in group]
            local_outputs += len(flat)
            if flat:
                _output_loss(flat).backward()
        torch.cuda.synchronize()
        counts = _gather_objects((dp_rank, waves, local_outputs))
        rows.append({"arm": "dp2-tp2-empty-slot", "per_rank": counts})
        per_dp = {}
        for dp, wave_count, outputs in counts:
            per_dp.setdefault(dp, set()).add((wave_count, outputs))
        if any(len(v) != 1 for v in per_dp.values()):
            _fail(f"TP peers disagree on the empty-slot wave: {counts}")
        totals = sorted(next(iter(v)) for v in per_dp.values())
        if totals != [(1, 0), (1, 1)]:
            _fail(
                f"empty-slot arm expected one wave with (0, 1) outputs across DP ranks: {counts}"
            )
        rank.zero_grad()
        _write_rows(evidence, rows, "dp2-tp2-waves")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "contract",
            "census",
            "measure",
            "validate",
            "split-conversion",
            "tp2-public",
            "dp2-tp2-waves",
            "tp-compare",
        ),
    )
    parser.add_argument("--cell", default="grpo-gdn-cp4")
    parser.add_argument(
        "--arm", default="automatic", choices=("automatic", "depth_one", "full_sharing")
    )
    parser.add_argument("--evidence", default="")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--pressure",
        default="cap",
        choices=("cap", "ballast"),
        help="split-conversion only: induce memory pressure with the test-only cap or real ballast",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="tp2-public only: expected tensor-parallel size (1 runs the control cell)",
    )
    parser.add_argument(
        "--dump-dir", default="", help="tp2-public only: save per-arm outputs here"
    )
    parser.add_argument(
        "--control-dump", default="", help="tp-compare: TP1 control dump (.pt)"
    )
    parser.add_argument("--dump", default="", help="tp-compare: TP>1 dump (.pt)")
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
    elif arguments.phase == "split-conversion":
        phase_split_conversion(arguments.evidence or None, arguments.pressure)
    elif arguments.phase == "tp2-public":
        phase_tp2_public(
            arguments.evidence or None,
            arguments.repeat,
            tp=arguments.tp,
            dump_dir=arguments.dump_dir or None,
        )
    elif arguments.phase == "tp-compare":
        if not (arguments.control_dump and arguments.dump):
            _fail("tp-compare requires --control-dump and --dump")
        phase_tp_compare(
            arguments.control_dump, arguments.dump, arguments.evidence or None
        )
    elif arguments.phase == "dp2-tp2-waves":
        phase_dp2_tp2_waves(arguments.evidence or None)
    else:
        if not arguments.evidence:
            _fail("--evidence input path is required for validate")
        phase_validate(arguments.cell, arguments.evidence)


if __name__ == "__main__":
    main()
