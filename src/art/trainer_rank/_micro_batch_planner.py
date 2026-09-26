"""TrainerRank micro-batch planning, split search and admission.

These are ``TrainerRank`` method bodies moved out of ``_impl`` verbatim: each
function takes the owning rank as ``self`` and ``TrainerRank`` binds them as
methods, so ``self._x(...)`` dispatch and per-instance overrides keep working.

Module globals the bodies used to read from ``_impl`` (``torch``, ``dist``,
``_telemetry_phase``, sibling helpers, plan/cost types) are still resolved
through ``_impl`` at call time, so tests that patch ``_impl.dist`` and friends
keep intercepting them; only pure stdlib helpers are imported here directly.
Referencing ``_impl`` as a module also lets the circular import resolve lazily.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import (
    asdict,
    replace,
)
import hashlib
import time
from typing import TYPE_CHECKING

from art.trainer_rank import _impl
from art.trainer_rank._backward_work import region as _backward_region

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Sequence,
    )
    from typing import Any

    import torch

    from art.trainer_rank import _planner_evidence
    from art.trainer_rank._impl import (
        AdapterSelection,
        AnyForwardInput,
        AnyForwardOutput,
        ForwardInputs,
        ForwardInputsT,
        ForwardOutputs,
        MicroBatch,
        TrainerRank,
        TrainerRankMemoryError,
        _AnyForwardPlan,
        _CandidateMicroBatch,
        _FlatForwardPlan,
        _ForwardGroupPlan,
        _ForwardRefusal,
        _LayoutKey,
        _MemoryCheck,
        _MemorySignature,
        _PlannerFacts,
        _SplitForwardPlan,
        _SubforwardCost,
    )
    from art.trainer_rank._prefix_tree_planner import (
        CanonicalPrefixTree,
        PrefixTreeLayout,
    )


def _forward_micro_batches(
    self: TrainerRank,
    inputs: Iterable[ForwardInputs],
    *,
    checkpoint: AdapterSelection,
    yield_empty: bool,
) -> Generator[MicroBatch[ForwardInputs, ForwardOutputs], None, None]:
    backward = self._backward_work()
    if backward is not None:
        backward.harvest()
    items = [_impl._materialize(item) for item in inputs]
    requests = list(_impl._flatten(items))
    self._validate_replicated_top_level_count(len(items), yield_empty=yield_empty)
    for _, indices in self._group_active_request_indices(
        requests, checkpoint=checkpoint
    ):
        for index in indices:
            self._forward_item(requests[index])
    start = 0
    self._reset_planning_telemetry()
    while start < len(items):
        with _impl._telemetry_phase(
            "plan",
            {"global_start": start, "global_remaining": len(items) - start},
        ):
            candidate = self._select_next_micro_batch(
                items, start, checkpoint=checkpoint
            )
        self._snapshot_planning_telemetry(candidate.plan, candidate.check)
        tracked_outputs: list[AnyForwardOutput] = []
        outputs: list[Any] = []
        flat_outputs = iter(tracked_outputs)
        error: BaseException | None = None
        try:
            if isinstance(candidate.plan, _impl._FlatForwardPlan):
                tracked_outputs, memory_baseline = (
                    self._run_flat_plan_with_memory_tracking(
                        candidate.plan,
                        check=candidate.check,
                        context="forward_micro_batches",
                    )
                )
            else:
                tracked_outputs, memory_baseline, forward_peak = (
                    self._execute_split_plan_with_memory_tracking(
                        candidate.plan,
                        check=candidate.check,
                        context="forward_micro_batches",
                    )
                )
            flat_outputs = iter(tracked_outputs)
            outputs = [
                _impl._unflatten(item, flat_outputs) for item in candidate.inputs
            ]
        except BaseException as exc:
            error = exc
        try:
            self._release_cached_memory_for_backward(candidate.plan, error=error)
        except BaseException:
            # Do not retain our completed graph through a new handoff traceback.
            del tracked_outputs, flat_outputs, outputs
            raise
        if backward is not None:
            backward.attach(tracked_outputs)
        stop = start + candidate.stats_global_count
        if stop < len(items):
            self._last_global_micro_batch_size = max(
                self._last_global_micro_batch_size or 0,
                candidate.stats_global_count,
            )
            # Overlap next-wave planning with the caller's GPU time: the
            # width search seeds from the last wave's width, so pre-plan
            # that slice while this generator is suspended at the yield.
            self._submit_speculative_wave_planning(items, stop, checkpoint=checkpoint)
        self._snapshot_planning_telemetry(candidate.plan, candidate.check)
        with _impl._telemetry_phase(
            # This interval is controlled by the caller and normally contains
            # loss construction and backward for the yielded microbatch.
            "caller",
            self._telemetry_signature(candidate.plan),
            dedup_signature=self._telemetry_plan_signature(candidate.plan),
        ):
            yield _impl.MicroBatch(
                inputs=candidate.inputs,
                outputs=outputs,
                indices=candidate.indices,
                stats=_impl.MicroBatchStats(
                    global_start=start,
                    global_stop=stop,
                    global_count=candidate.stats_global_count,
                    local_count=len(candidate.inputs),
                    packed_tokens=candidate.plan.packed_tokens,
                    logical_tokens=candidate.plan.logical_tokens,
                    estimated_required_bytes=candidate.check.estimated_required_bytes,
                    available_bytes=candidate.check.available_bytes,
                    rejected_candidates=candidate.rejected_candidates,
                    cold_start=candidate.cold_start,
                    subforward_count=candidate.plan.subforward_count,
                ),
            )
        if backward is not None:
            backward.harvest()
        # The caller normally runs backward while the micro-batch is yielded.
        # Include that peak in future planning; forward-only profiling can
        # otherwise admit a later micro-batch that leaves no collective or
        # optimizer headroom. Peak only: the retained observation belongs
        # to the forward's return, already recorded for this same plan.
        if isinstance(candidate.plan, _impl._FlatForwardPlan):
            self._update_peak_memory_profile(candidate.plan, memory_baseline)
        elif memory_baseline is not None:
            self._record_split_memory_floor(
                candidate.plan, memory_baseline, forward_peak
            )
        # Only the caller may retain completed outputs into the next wave.
        self._complete_planner_observation()
        del tracked_outputs, flat_outputs, outputs
        start = stop


def _plan_admissible_forward(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    checkpoint: AdapterSelection,
    context: str,
) -> tuple[_AnyForwardPlan, _MemoryCheck]:
    # DP-local recovery may retry asymmetrically; checkpoint setup is WORLD.
    self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
    result = self._recover_admission(
        lambda: self._find_admissible_forward(
            requests,
            checkpoint=checkpoint,
            refusal_prefix="forward is predicted to exceed available memory",
            ensure_slots=False,
        ),
        lambda value: value,
        lambda value, check: (value[0], check),
        context=context,
        sync_across_dp=False,
        admit_refusal=lambda refusal: (refusal.plan, refusal.check),
    )
    self._snapshot_planning_telemetry(*result)
    return result


def _find_admissible_forward(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    checkpoint: AdapterSelection,
    refusal_prefix: str,
    ensure_slots: bool = True,
) -> tuple[_AnyForwardPlan, _MemoryCheck] | _ForwardRefusal:
    """Find an admissible plan: unsplit first, then the bounded split ladder.

    The unsplit plan is tried cost-optimal, then memory-minimal. If neither
    is admitted, the ladder tries 2, 4, ... subforwards (at most one
    request each), cutting the requests in prefix-local depth-first order
    into token-balanced chunks, and stops at the fewest subforwards whose
    rung check passes (``_admit_split_rung``). Exhausting the ladder is a
    refusal worded as "unable to find a feasible split": the search is
    bounded, so this is not a claim that none exists.

    Checkpoint slots are ensured up front unless the caller already did
    so before entering a DP-local retry loop. Everything after plans with
    ``ensure_slots=False`` so the number of collectives this rank performs
    does not depend on its (DP-local) inputs.
    """

    if ensure_slots:
        self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
    plan = self._plan_flat_forward(requests, checkpoint=checkpoint, ensure_slots=False)
    check = self._memory_check(plan)
    if check.fits:
        return plan, check
    best = (plan, check)
    # Best effort before splitting: the memory-minimal (full sharing)
    # layouts may fit where the cost-optimal ones do not.
    plan = self._plan_flat_forward(
        requests, checkpoint=checkpoint, memory_minimal=True, ensure_slots=False
    )
    check = self._memory_check(plan)
    if check.fits:
        return plan, check
    if check.estimated_required_bytes < best[1].estimated_required_bytes:
        best = (plan, check)
    request_count = len(requests)
    if request_count == 1:
        return _impl._ForwardRefusal(
            *(
                best
                if getattr(self, "_allow_oversized_batches", False)
                else (plan, check)
            ),
            f"{refusal_prefix}; a single request cannot be split",
        )
    if self._expert_parallel_active():
        return _impl._ForwardRefusal(
            plan,
            check,
            f"{refusal_prefix}; unable to find a feasible split: internal "
            "splitting is disabled under expert parallelism in this release",
            overridable=False,
        )
    # A rejected lower bound normally avoids materializing the rung. If
    # ranks disagree on the opt-in, keep that original behavior everywhere
    # so the exact-pricing collectives cannot diverge within TP x CP.
    keep_rejected = (
        self._recovery_reduce(
            [float(getattr(self, "_allow_oversized_batches", False))],
            op="MIN",
            sync_across_dp=False,
        )[0]
        == 1
    )
    rows = tuple(
        request.input_tokens.detach().reshape(-1).to("cpu", _impl.torch.long)
        for request in requests
    )
    order = self._split_request_order(requests, rows, checkpoint=checkpoint)
    subforward_count = 2
    while True:
        chunks = _impl._split_chunks(order, requests, subforward_count)
        split, check = self._admit_split_rung(
            chunks,
            requests,
            rows,
            checkpoint=checkpoint,
            keep_rejected=keep_rejected,
        )
        if split is not None and check.fits:
            return split, check
        if (
            split is not None
            and check.estimated_required_bytes < best[1].estimated_required_bytes
        ):
            best = (split, check)
        if subforward_count >= request_count:
            break
        subforward_count = min(request_count, subforward_count * 2)
    return _impl._ForwardRefusal(
        *(best if getattr(self, "_allow_oversized_batches", False) else (plan, check)),
        f"{refusal_prefix}; unable to find a feasible split: every rung of "
        "the bounded ladder (2, 4, ..., one request per subforward) is "
        "predicted to exceed available memory once all returned graphs are "
        "live together",
        # Without the override the rejected rung is not retained. Its
        # check must not be attributed to the unsplit context plan.
        check_matches_plan=getattr(self, "_allow_oversized_batches", False),
    )


def _admit_split_rung(
    self: TrainerRank,
    chunks: Sequence[tuple[int, ...]],
    requests: Sequence[AnyForwardInput],
    rows: Sequence[torch.Tensor],
    *,
    checkpoint: AdapterSelection,
    keep_rejected: bool | None = None,
) -> tuple[_SplitForwardPlan | None, _MemoryCheck]:
    """Admit one rung of the ladder, or return its binding memory check.

    Every returned graph stays live, so subforward ``j`` needs its own
    transient peak plus the memory retained by the subforwards before it.
    Each of those sums is bounded by the rung check — all retained memory
    plus the largest ephemeral share — which therefore decides the rung by
    itself, in any execution order. The same quantity is the headroom the
    caller's backward can count on: every graph live and one subforward's
    forward-ephemeral memory free again. That is a heuristic for backward
    workspace, not a bound (a backward may need more than its forward's
    ephemeral memory); the ballast arm of the split-conversion gate
    measures it on a real cell.

    Cost: the cheap full-sharing lower bound (one O(tokens) CPU scan per
    chunk) can reject a rung without planning it. A surviving rung is
    priced exactly with cost-optimal and then memory-minimal layouts.
    Retained-profile trust boundaries can make the bound optimistic, so
    more than one rung may need exact planning before one executes.
    """

    lower = [
        self._split_chunk_lower_cost(
            [requests[index] for index in chunk],
            [rows[index] for index in chunk],
            checkpoint=checkpoint,
        )
        for chunk in chunks
    ]
    check = self._split_rung_check(lower)
    if keep_rejected is None:
        keep_rejected = getattr(self, "_allow_oversized_batches", False)
    if not check.fits and not keep_rejected:
        return None, check
    best: tuple[_SplitForwardPlan | None, _MemoryCheck] = (None, check)
    for memory_minimal in (False, True):
        plans = [
            self._plan_flat_forward(
                [requests[index] for index in chunk],
                checkpoint=checkpoint,
                memory_minimal=memory_minimal,
                ensure_slots=False,
            )
            for chunk in chunks
        ]
        costs = [self._plan_cost(plan) for plan in plans]
        # Bind original request mappings; floor keys normalize execution order.
        order = sorted(
            range(len(plans)),
            key=lambda i: (
                -(costs[i].ephemeral - costs[i].checkpoint_peak_increment),
                i,
            ),
        )
        split = _impl._SplitForwardPlan(
            subforwards=tuple(plans[i] for i in order),
            request_indices=tuple(tuple(chunks[i]) for i in order),
            request_count=len(requests),
        )
        check = self._split_plan_memory_check(split, costs)
        if check.fits:
            return split, check
        if (
            best[0] is None
            or check.estimated_required_bytes < best[1].estimated_required_bytes
        ):
            best = (split, check)
    return best if keep_rejected else (None, check)


def _split_rung_check(
    self: TrainerRank, costs: Sequence[_SubforwardCost]
) -> _MemoryCheck:
    return self._memory_check_required(self._split_required_memory(costs))


def _split_chunk_lower_cost(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    rows: Sequence[torch.Tensor],
    *,
    checkpoint: AdapterSelection,
) -> _SubforwardCost:
    """Optimistic cost across layout and profile-trust boundaries."""

    groups = self._group_active_request_indices(
        requests, checkpoint=checkpoint, ensure_slots=False
    )
    packed_tokens = 0
    unshared_packed_tokens = 0
    head_workspace_bytes = 0
    group_rows: list[tuple[int, bool]] = []
    for (_slot, grad_enabled), group_indices in groups:
        estimated = _impl.estimate_prefix_tree_packed_tokens(
            (rows[index] for index in group_indices),
            max_depth=len(group_indices),
        )
        assert estimated is not None  # rows are CPU copies
        physical_rows = self._physical_tokens(estimated)
        packed_tokens += physical_rows
        # The most loaded CP rank holds at least an even share.
        cp = max(1, self._topology_key()[2])
        group_rows.append((-(-physical_rows // cp), grad_enabled))
        head_requests = tuple(requests[index] for index in group_indices)
        head_workspace_bytes = max(
            head_workspace_bytes,
            self._group_head_workspace_bytes(
                self._head_projection_rows(head_requests, lower_bound=True),
                head_requests,
                grad_enabled=grad_enabled,
                lower_bound=True,
            ),
        )
        unshared_packed_tokens += self._physical_tokens(
            sum(int(rows[index].numel()) for index in group_indices)
        )
    output_bytes = self._estimate_group_request_output_bytes(requests)
    signature = self._memory_signature_from_requests(
        requests,
        slot_group_count=len(groups),
        grad_modes=tuple(mode for (_, mode), _ in groups),
        slot_groups=tuple(key for key, _ in groups),
    )
    logical_tokens = _impl._active_logical_tokens(requests)
    cost = self._subforward_cost(
        packed_tokens=packed_tokens,
        output_bytes=output_bytes,
        signature=signature,
        logical_tokens=logical_tokens,
        group_rows=tuple(group_rows),
        slot_refs=tuple(ref for (ref, _), _ in groups),
        head_workspace_bytes=head_workspace_bytes,
        # The average CP load is an optimistic bound, not an admission cost.
        retained_tokens=(packed_tokens + signature.topology[2] - 1)
        // signature.topology[2],
    )
    profile = self._memory_profiles.get(signature)
    if (
        profile is not None
        and profile.retained_compute_bytes_per_token is not None
        and logical_tokens / max(1, packed_tokens)
        > profile.logical_per_packed * _impl._MEMORY_PROFILE_TRUST_GROWTH
        and logical_tokens
        / max(
            1,
            min(
                unshared_packed_tokens,
                profile.packed_tokens * _impl._MEMORY_PROFILE_TRUST_GROWTH,
            ),
        )
        <= profile.logical_per_packed * _impl._MEMORY_PROFILE_TRUST_GROWTH
    ):
        # A larger layout may trust retained compute where full sharing
        # cannot. Its full-required retention is not a pruning lower bound.
        # Keep outputs and the independent source retention floor; exact
        # plan costs keep both trust guards.
        return replace(
            cost,
            retained=min(
                cost.retained,
                int(
                    (output_bytes + self._checkpoint_memory_floor(tuple(group_rows))[0])
                    * _impl._MEMORY_SAFETY_FACTOR
                ),
            ),
        )
    return cost


def _plan_group_rows(
    self: TrainerRank, plan: _FlatForwardPlan
) -> tuple[tuple[int, bool], ...]:
    """Physical rows per group on the most loaded context-parallel rank."""
    topology = self._topology() if plan.signature.topology[2] > 1 else None
    return tuple(
        (
            self._physical_tokens(
                int(group.packed.tokens.numel())
                if topology is None
                else max(
                    1,
                    self._max_rank_model_tokens(
                        _impl._pad_packed_batch(
                            group.packed, multiple=int(topology.tp)
                        ),
                        topology=topology,
                    ),
                )
            ),
            group.grad_enabled,
        )
        for group in plan.groups
    )


def _plan_cost(self: TrainerRank, plan: _FlatForwardPlan) -> _SubforwardCost:
    return self._subforward_cost(
        packed_tokens=plan.packed_tokens,
        output_bytes=plan.output_bytes,
        signature=plan.signature,
        logical_tokens=plan.active_logical_tokens,
        gdn_segments=plan.grad_segment_count,
        group_rows=self._plan_group_rows(plan),
        slot_refs=tuple(g.slot_ref for g in plan.groups),
        head_workspace_bytes=self._plan_head_workspace_bytes(plan),
        checkpoint_floor=_impl._gdn_memory.plan_floor(self, plan),
        retained_tokens=self._plan_retained_tokens(plan),
        hybridep_growth_bytes=self._plan_hybridep_growth_bytes(plan),
    )


def _split_request_order(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    rows: Sequence[torch.Tensor],
    *,
    checkpoint: AdapterSelection,
) -> tuple[int, ...]:
    """Order requests in prefix-local depth-first order.

    Within each checkpoint group, requests follow the canonical tree's
    depth-first leaf order, so prefix-sharing requests are adjacent and
    contiguous cuts keep most sharing inside one chunk; requests that
    produce no outputs (and so join no group) follow in input order.
    """

    ordered: list[int] = []
    seen: set[int] = set()
    for (_slot, grad_enabled), group_indices in self._group_active_request_indices(
        requests, checkpoint=checkpoint, ensure_slots=False
    ):
        tree, _layout = self._select_group_layout(
            tuple(rows[index] for index in group_indices),
            grad_enabled=grad_enabled,
        )
        for sequence_indices in tree.sequence_indices_by_terminal:
            for position in sequence_indices:
                ordered.append(group_indices[position])
                seen.add(group_indices[position])
    ordered.extend(index for index in range(len(requests)) if index not in seen)
    return tuple(ordered)


def _reset_planning_telemetry(self: TrainerRank) -> None:
    self._planning_seconds_accum = 0.0
    with self._layout_cache_lock:
        self._speculative_planning_seconds = 0.0


def _snapshot_planning_telemetry(
    self: TrainerRank, plan: _AnyForwardPlan, check: _MemoryCheck
) -> None:
    with self._layout_cache_lock:
        speculative_seconds = self._speculative_planning_seconds
    partition = (
        plan.request_indices
        if isinstance(plan, _impl._SplitForwardPlan)
        else (tuple(range(plan.request_count)),)
    )
    self._last_forward_telemetry_snapshot = {
        "planning_ms": self._planning_seconds_accum * 1_000.0,
        "speculative_planning_ms": speculative_seconds * 1_000.0,
        "selected_max_depth": plan.selected_max_depth,
        "subforward_count": plan.subforward_count,
        # Flat request indices executed by each subforward, in execution
        # order; lets callers backward per subforward if they wish.
        "subforward_request_indices": partition,
        "predicted_peak_bytes": check.estimated_required_bytes,
        "usable_limit_bytes": check.available_bytes,
    }


def _select_next_micro_batch(
    self: TrainerRank,
    items: Sequence[ForwardInputsT],
    start: int,
    *,
    checkpoint: AdapterSelection = _impl.Unset,
) -> _CandidateMicroBatch[ForwardInputsT]:
    def admit(refusal: _ForwardRefusal) -> _CandidateMicroBatch[ForwardInputsT]:
        dp_rank, dp_size = self._dp_rank_and_size()
        width = min(len(items) - start, dp_size)
        indices = _impl._local_wave_indices(start, width, dp_rank, dp_size)
        return _impl._CandidateMicroBatch(
            inputs=[items[index] for index in indices],
            indices=indices,
            plan=refusal.plan,
            check=refusal.check,
            stats_global_count=width,
            rejected_candidates=1,
            cold_start=True,
        )

    return self._recover_admission(
        lambda: self._search_next_micro_batch(items, start, checkpoint=checkpoint),
        lambda value: (value.plan, value.check),
        lambda value, check: replace(value, check=check),
        context="forward_micro_batches",
        sync_across_dp=True,
        admit_refusal=admit,
    )


def _search_next_micro_batch(
    self: TrainerRank,
    items: Sequence[ForwardInputsT],
    start: int,
    *,
    checkpoint: AdapterSelection = _impl.Unset,
) -> _CandidateMicroBatch[ForwardInputsT] | _ForwardRefusal:
    dp_rank, dp_size = self._dp_rank_and_size()
    remaining, min_width, granularity = _impl._wave_geometry(len(items), start, dp_size)
    if min_width <= 0:
        raise RuntimeError("cannot select an empty microbatch window")

    def normalize(width: int) -> int:
        return _impl._normalize_wave_width(width, min_width, remaining, granularity)

    def local_slice(width: int) -> tuple[tuple[int, ...], list[ForwardInputsT]]:
        indices = _impl._local_wave_indices(start, width, dp_rank, dp_size)
        return indices, [items[index] for index in indices]

    estimates: dict[int, tuple[_MemoryCheck, bool, bool] | None] = {}
    plans: dict[int, _FlatForwardPlan] = {}
    checked_plans: dict[int, _MemoryCheck] = {}
    # Per-width layout mode chosen by admission: False = cost-optimal,
    # True = memory-minimal (full sharing). Materialization must build the
    # same layouts the admitted estimate priced.
    layout_modes: dict[int, bool] = {}
    exact_failed_width: int | None = None

    def estimate(width: int) -> tuple[_MemoryCheck, bool, bool] | None:
        nonlocal exact_failed_width
        width = normalize(width)
        if width in estimates:
            return estimates[width]
        indices, local_inputs = local_slice(width)
        local_requests = list(_impl._flatten(local_inputs))
        cheap_segments: list[int] = []
        values = self._estimate_flat_forward(
            local_requests,
            checkpoint=checkpoint,
            sync_planning_errors=True,
            gdn_segments=cheap_segments,
        )
        if not self._all_ranks_true(values is not None):
            estimates[width] = None
            return None
        assert values is not None
        logical_tokens = _impl._active_logical_tokens(local_requests)

        def priced(
            packed_tokens: int,
            output_bytes: int,
            signature: _MemorySignature,
            group_rows: tuple[tuple[int, bool], ...],
            head_workspace_bytes: int,
            *,
            gdn_segments: int,
        ) -> tuple[_MemoryCheck, int, int, _MemorySignature]:
            with self._planning_status(True):
                required = self._estimate_required_memory_bytes_from_values(
                    packed_tokens=packed_tokens,
                    output_bytes=output_bytes,
                    signature=signature,
                    logical_tokens=logical_tokens,
                    # Gradient groups' segments: exact layouts' counts, else
                    # a bound matching the estimate's (_estimate_flat_forward).
                    gdn_segments=gdn_segments,
                    group_rows=group_rows,
                    head_workspace_bytes=head_workspace_bytes,
                )
            return (
                self._memory_check_required(required, sync_across_dp=True),
                packed_tokens,
                output_bytes,
                signature,
            )

        def priced_estimate(
            *, exact: bool, memory_minimal: bool
        ) -> tuple[_MemoryCheck, int, int, _MemorySignature] | None:
            segments: list[int] = []
            estimated = self._estimate_flat_forward(
                local_requests,
                checkpoint=checkpoint,
                exact=exact,
                memory_minimal=memory_minimal,
                sync_planning_errors=True,
                gdn_segments=segments,
            )
            return (
                None
                if estimated is None
                else priced(*estimated, gdn_segments=sum(segments))
            )

        def trusted(packed_tokens: int, signature: _MemorySignature) -> bool:
            return self._all_ranks_have_memory_profile(
                packed_tokens=packed_tokens, signature=signature
            )

        # The cheap no-sharing count is an upper bound on any planner
        # layout: valid for accepting a width (memory and profile trust),
        # never for rejecting one. Exact pricing runs when the bound would
        # reject on memory, or when it would reject on profile trust while
        # a profile exists — the selected layout may be far smaller than
        # the bound and squarely inside the profiled regime.
        selected = priced(*values, gdn_segments=sum(cheap_segments))
        profiled = self._all_ranks_true(selected[3] in self._memory_profiles)
        needs_exact = not selected[0].fits or (
            profiled and not trusted(selected[1], selected[3])
        )
        if needs_exact and (exact_failed_width is None or width < exact_failed_width):
            # Feasibility must be monotone in width for the outer search:
            # the cost-optimal layout can decline sharing at one width and
            # accept it at a wider one, but the memory-minimal (full
            # sharing) layout's packed count is monotone by construction.
            # Its cheap bound decides feasibility; planner pricing then
            # picks cost-optimal when that fits and memory-minimal
            # otherwise, recording the mode so materialization executes
            # exactly the layouts that were priced.
            minimal_bound = priced_estimate(exact=False, memory_minimal=True)
            if minimal_bound is not None and minimal_bound[0].fits:
                for memory_minimal in (False, True):
                    exact = priced_estimate(exact=True, memory_minimal=memory_minimal)
                    assert exact is not None
                    selected = exact
                    # The minimum wave may execute cold after trust fails.
                    # Keep its materialization paired with the retained check.
                    layout_modes[width] = memory_minimal
                    if selected[0].fits and (
                        not profiled or trusted(selected[1], selected[3])
                    ):
                        break
            elif minimal_bound is not None:
                selected = minimal_bound
                layout_modes[width] = True
            if not selected[0].fits:
                exact_failed_width = (
                    width
                    if exact_failed_width is None
                    else min(exact_failed_width, width)
                )
        check, packed_tokens, _output_bytes, signature = selected
        result = (
            check,
            trusted(packed_tokens, signature),
            self._all_ranks_true(signature in self._memory_profiles),
        )
        estimates[width] = result
        return result

    rejected_widths: set[int] = set()

    def fits(width: int) -> tuple[bool, bool]:
        width = normalize(width)
        result = estimate(width)
        if result is None:
            # Estimator unavailable (device inputs, or CP per-rank floors):
            # admit on the materialized plan, trying the cost-optimal
            # layouts first and the memory-minimal layouts if those do not
            # fit or fall outside the profile's trust window.
            def price(plan: _FlatForwardPlan) -> tuple[_MemoryCheck, bool, bool]:
                check = self._memory_check(
                    plan, sync_across_dp=True, sync_planning_errors=True
                )
                trusted = self._all_ranks_have_memory_profile(
                    packed_tokens=plan.packed_tokens,
                    signature=plan.signature,
                )
                profiled = self._all_ranks_true(plan.signature in self._memory_profiles)
                return check, trusted, profiled

            plan = materialize(width)
            check, trusted, profiled = price(plan)
            if (not check.fits or (profiled and not trusted)) and not layout_modes.get(
                width, False
            ):
                layout_modes[width] = True
                plans.pop(width, None)
                plan = materialize(width)
                check, trusted, profiled = price(plan)
        else:
            check, trusted, profiled = result
        if width in plans:
            checked_plans[width] = check
        if not check.fits:
            rejected_widths.add(width)
        return check.fits and (trusted or not profiled), trusted

    def materialize(width: int) -> _FlatForwardPlan:
        width = normalize(width)
        plan = plans.get(width)
        if plan is None:
            _, local_inputs = local_slice(width)
            plan = self._plan_flat_forward(
                list(_impl._flatten(local_inputs)),
                checkpoint=checkpoint,
                memory_minimal=layout_modes.get(width, False),
                sync_planning_errors=True,
            )
            plans[width] = plan
        return plan

    def candidate(width: int) -> _CandidateMicroBatch[ForwardInputsT]:
        width = normalize(width)
        indices, local_inputs = local_slice(width)
        plan = materialize(width)
        estimated = estimates.get(width)
        check = (
            estimated[0]
            if estimated is not None
            else self._memory_check(
                plan, sync_across_dp=True, sync_planning_errors=True
            )
        )
        cold_start = not self._all_ranks_have_memory_profile(
            packed_tokens=plan.packed_tokens,
            signature=plan.signature,
        )
        selected = _impl._CandidateMicroBatch(
            inputs=local_inputs,
            indices=indices,
            plan=plan,
            check=check,
            stats_global_count=width,
            rejected_candidates=len(rejected_widths),
            cold_start=cold_start,
        )
        checked_plans[width] = check
        if getattr(self, "_allow_oversized_batches", False):
            smallest = min(
                checked_plans,
                key=lambda w: (checked_plans[w].estimated_required_bytes, w),
            )
            if smallest != width:
                fallback_indices, fallback_inputs = local_slice(smallest)
                selected = replace(
                    selected,
                    fallback=_impl._CandidateMicroBatch(
                        inputs=fallback_inputs,
                        indices=fallback_indices,
                        plan=plans[smallest],
                        check=checked_plans[smallest],
                        stats_global_count=smallest,
                        rejected_candidates=len(rejected_widths),
                        cold_start=True,
                    ),
                )
        return selected

    first_estimate = estimate(min_width)
    if first_estimate is None or not (first_estimate[0].fits and first_estimate[1]):
        first = candidate(min_width)
        if (
            first_estimate is None
            and first.check.fits
            and first.cold_start
            and not layout_modes.get(min_width, False)
            and self._all_ranks_true(first.plan.signature in self._memory_profiles)
        ):
            # Materialized pricing (DP-uniform): a profiled cost-optimal
            # layout outside trust; full sharing may be trusted, as the
            # estimator path would find.
            layout_modes[min_width] = True
            plans.pop(min_width, None)
            first = candidate(min_width)
        if not first.check.fits:
            # The smallest wave cannot run unsplit: best effort is the
            # bounded split ladder. Each DP rank runs it on its own share,
            # then all ranks agree on the outcome (one collective, always)
            # so a refusal is raised everywhere or nowhere.
            indices, local_inputs = local_slice(min_width)
            refusal_prefix = (
                "smallest DP microbatch is predicted to exceed available memory"
            )
            admission_error: BaseException | None = None
            try:
                found = self._find_admissible_forward(
                    list(_impl._flatten(local_inputs)),
                    checkpoint=checkpoint,
                    refusal_prefix=refusal_prefix,
                )
            except BaseException as exc:
                admission_error, found = exc, None
            try:
                outcome = self._admission_outcome(
                    0
                    if admission_error is not None
                    else 1
                    if isinstance(found, _impl._ForwardRefusal)
                    else 2
                )
            except BaseException:
                if admission_error is not None:
                    raise admission_error
                raise
            if admission_error is not None:
                raise admission_error
            assert found is not None
            if outcome == 0:
                raise RuntimeError("Memory admission failed on another DP rank")
            if isinstance(found, _impl._ForwardRefusal):
                return found
            if outcome == 1:
                return _impl._ForwardRefusal(
                    found[0],
                    found[1],
                    f"{refusal_prefix} on another DP rank, which was unable "
                    "to find a feasible split for its share",
                )
            split_plan, split_check = found
            return _impl._CandidateMicroBatch(
                inputs=local_inputs,
                indices=indices,
                plan=split_plan,
                check=split_check,
                stats_global_count=min_width,
                rejected_candidates=len(rejected_widths),
                cold_start=True,
            )
        if first.cold_start:
            return first

    best = min_width
    failed: int | None = None
    width = normalize(self._last_global_micro_batch_size or min_width)
    if width > best:
        fit, trusted = fits(width)
        if fit and trusted:
            best = width
        # A cached width from another checkpoint/output mix is only a
        # shortcut when it fits a trusted profile. Otherwise grow below;
        # a cached binary-search bound can also jump to an unprofiled mix.

    while failed is None and best < remaining:
        width = normalize(max(best + 1, best * 2))
        if width == best:
            break
        fit, trusted = fits(width)
        if fit:
            best = width
            if not trusted:
                break
        else:
            failed = width

    if failed is not None:
        while failed - best > 1:
            width = normalize((best + failed) // 2)
            if width in (best, failed):
                break
            if fits(width)[0]:
                best = width
            else:
                failed = width

    return candidate(best)


def _planner_topology_facts(
    self: TrainerRank, *, grad_enabled: bool = True
) -> "_PlannerFacts":
    _dp, tp_size, cp_size, _pp = self._topology_key()
    uses_gdn = bool(
        getattr(self.runtime.model_support_handler, "build_gdn_execution_spec", False)
    )
    return _impl._PlannerFacts(
        cp_size=cp_size,
        tp_size=tp_size,
        layers=self._num_layers,
        gdn_layers=self._gdn_layers if uses_gdn else 0,
        uses_gdn=uses_gdn,
        ep_size=self._parallel_shape.ep,
        etp_size=self._parallel_shape.etp,
        coefficient_version=(
            self._coefficient_version
            if grad_enabled
            else _impl.COEFFICIENT_VERSION_FALLBACK
        ),
        coefficient_table=self._coefficient_table if grad_enabled else None,
        grad_enabled=grad_enabled,
    )


def _layout_anchor(self: TrainerRank, *, memory_minimal: bool) -> str | None:
    """Resolve the layout anchor: test forcing wins, else memory policy."""

    forced = self._forced_test_anchor()
    if forced is not None:
        return forced
    return _impl._MEMORY_MINIMAL_ANCHOR if memory_minimal else None


def _layout_cache_key(
    self: TrainerRank,
    input_ids: Sequence[torch.Tensor],
    *,
    memory_minimal: bool = False,
    grad_enabled: bool = True,
) -> "_LayoutKey":
    facts = self._planner_topology_facts(grad_enabled=grad_enabled)
    hasher = hashlib.sha256()
    for tensor in input_ids:
        row = tensor.detach().reshape(-1).cpu().contiguous()
        hasher.update(str(row.dtype).encode("ascii"))
        hasher.update(_impl._U64_STRUCT.pack(int(row.numel())))
        hasher.update(row.numpy())
    return (
        hasher.hexdigest(),
        facts,
        self._layout_anchor(memory_minimal=memory_minimal),
    )


def _cached_group_layout(
    self: TrainerRank,
    key: "_LayoutKey",
) -> tuple[CanonicalPrefixTree, PrefixTreeLayout] | None:
    with self._layout_cache_lock:
        cached = self._layout_selection_cache.get(key)
        if cached is not None:
            self._layout_selection_cache.move_to_end(key)
        return cached


def _compute_group_layout(
    self: TrainerRank,
    input_ids: Sequence[torch.Tensor],
    key: "_LayoutKey",
) -> tuple[CanonicalPrefixTree, PrefixTreeLayout]:
    """Plan one group and memoize the result.

    Pure with respect to TrainerRank state apart from the caches, so it is
    safe to run on the speculative planning thread; a concurrent duplicate
    computation of the same key is deterministic and harmless. The
    canonical tree is cached by content alone so the cost-optimal and
    memory-minimal layouts of one group share a single construction.
    """

    content_key, facts, anchor = key
    with self._layout_cache_lock:
        tree = self._tree_cache.get(content_key)
        if tree is not None:
            self._tree_cache.move_to_end(content_key)
    if tree is None:
        tree = _impl.build_canonical_prefix_tree(input_ids)
        with self._layout_cache_lock:
            self._tree_cache[content_key] = tree
            while len(self._tree_cache) > _impl._LAYOUT_SELECTION_CACHE_LIMIT:
                self._tree_cache.popitem(last=False)
    if anchor is not None:
        candidates = _impl.prefix_tree_layout_candidates(tree)
        matching = [candidate for candidate in candidates if anchor in candidate.labels]
        if len(matching) != 1:
            raise ValueError(f"unknown forced layout anchor {anchor!r}")
        layout = matching[0].layout
    else:
        layout = _impl.select_prefix_tree_layout(
            tree,
            cp_size=facts.cp_size,
            layers=facts.layers,
            uses_gdn=facts.uses_gdn,
            tp_size=facts.tp_size,
            gdn_layers=facts.gdn_layers,
            coefficient_version=facts.coefficient_version,
            coefficients=(
                self._planner_coefficients
                if facts.coefficient_table is not None
                else None
            ),
            refinement_work_budget=_impl._PLANNER_REFINEMENT_BUDGET,
            reranker=(
                self._planner_reranker if facts.coefficient_table is not None else None
            ),
            plan_structure=(
                None
                if self._planner_reranker is None or facts.coefficient_table is None
                else lambda candidate: self._plan_structure(input_ids, tree, candidate)
            ),
        ).layout
    cached = (tree, layout)
    with self._layout_cache_lock:
        self._layout_selection_cache[key] = cached
        self._layout_selection_cache.move_to_end(key)
        while len(self._layout_selection_cache) > _impl._LAYOUT_SELECTION_CACHE_LIMIT:
            self._layout_selection_cache.popitem(last=False)
    return cached


def _plan_structure(
    self: TrainerRank,
    input_ids: Sequence[torch.Tensor],
    tree: CanonicalPrefixTree,
    layout: PrefixTreeLayout,
) -> tuple[int, int]:
    """(remote wave count, largest per-rank token load) of the context-
    parallel plan this layout produces, through the runtime's planning
    bundle cache: the layout finally selected reuses its bundle."""

    from art.megatron.context_parallel.runtime import summarize_prefix_tree_plan
    from art.megatron.training.microbatches import (
        _context_parallel_config_for_provider,
    )

    packed = _impl.materialize_prefix_tree_layout(
        input_ids, tree, layout, verify_shared_tokens=False
    )
    handler = self.runtime.model_support_handler
    summary = summarize_prefix_tree_plan(
        group_ids=packed.group_ids,
        parent_ids=packed.parent_ids,
        topology=self._topology(),
        config=_context_parallel_config_for_provider(
            self.runtime.provider, self.device, handler
        ),
        original_seq_len=int(packed.tokens.shape[1]),
        build_gdn_execution_spec=bool(
            getattr(handler, "build_gdn_execution_spec", False)
        ),
    )
    return summary.wave_count, summary.max_rank_tokens


def _select_group_layout(
    self: TrainerRank,
    input_ids: Sequence[torch.Tensor],
    *,
    memory_minimal: bool = False,
    grad_enabled: bool = True,
) -> tuple[CanonicalPrefixTree, PrefixTreeLayout]:
    """Select one group's prefix-sharing layout, cached by content identity.

    The cache key is a raw-bytes content hash plus the topology, cost
    coefficients, grad mode and layout anchor, so identical steady-state
    groups (or groups pre-planned speculatively during the caller's GPU
    work) skip canonicalization and search entirely. ``memory_minimal``
    selects the full-sharing layout instead of the cost-optimal one; the
    width search uses it when the cost-optimal layout cannot be admitted.
    Groups without gradients are scored by the version-1 fallback: the
    calibrated tables measured forward + backward.
    """

    started = time.perf_counter()
    key = self._layout_cache_key(
        input_ids, memory_minimal=memory_minimal, grad_enabled=grad_enabled
    )
    cached = self._cached_group_layout(key)
    if cached is None:
        cached = self._compute_group_layout(input_ids, key)
    self._planning_seconds_accum += time.perf_counter() - started
    return cached


def _speculative_planning_executor(self: TrainerRank) -> ThreadPoolExecutor:
    if self._speculative_planner is None:
        self._speculative_planner = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="trainer-rank-speculative-planner",
        )
    return self._speculative_planner


def _submit_speculative_wave_planning(
    self: TrainerRank,
    items: Sequence[ForwardInputs],
    start: int,
    *,
    checkpoint: AdapterSelection,
) -> None:
    """Pre-plan the predicted next wave while the caller uses the GPU.

    Runs while this generator is suspended at a yield (the caller's
    forward/backward time). The prediction mirrors the width search
    exactly: the next wave seeds from the largest width so far and plans
    this DP rank's strided local slice. Grouping, immutable CPU token
    snapshots, and cache keys are produced on the calling thread; the
    worker only runs the pure, memoized planner over those snapshots, so
    speculation can never change a selected plan and cannot be poisoned by
    the caller mutating its tensors afterwards. A wrong prediction merely
    leaves an unused LRU entry. Speculation is skipped for CUDA inputs so
    the worker never touches the device.

    The synchronous submission cost here is on the critical path (it
    delays the yield) and is charged to planning telemetry; the worker's
    hidden CPU time is reported separately as ``speculative_planning_ms``.
    """

    started = time.perf_counter()
    try:
        dp_rank, dp_size = self._dp_rank_and_size()
        remaining, min_width, granularity = _impl._wave_geometry(
            len(items), start, dp_size
        )
        if min_width <= 0:
            return
        width = _impl._normalize_wave_width(
            self._last_global_micro_batch_size or min_width,
            min_width,
            remaining,
            granularity,
        )
        indices = _impl._local_wave_indices(start, width, dp_rank, dp_size)
        requests = list(_impl._flatten([items[index] for index in indices]))
        if not requests:
            return
        if any(request.input_tokens.device.type != "cpu" for request in requests):
            return
        # Slots were ensured for every input when the call began; skip the
        # ensure-collective so speculation adds no communication.
        groups = self._group_active_request_indices(
            requests, checkpoint=checkpoint, ensure_slots=False
        )
        pending: list[
            tuple[
                tuple[torch.Tensor, ...],
                _LayoutKey,
            ]
        ] = []
        for (_slot, grad_enabled), group_indices in groups:
            snapshots = tuple(
                requests[index]
                .input_tokens.detach()
                .reshape(-1)
                .to(dtype=_impl.torch.long)
                .clone()
                for index in group_indices
            )
            key = self._layout_cache_key(snapshots, grad_enabled=grad_enabled)
            if self._cached_group_layout(key) is None:
                pending.append((snapshots, key))
    except Exception:
        # Prediction is best-effort; the real wave surfaces any genuine
        # input problem on the main thread.
        return
    finally:
        self._planning_seconds_accum += time.perf_counter() - started
    if not pending:
        return

    def warm() -> None:
        worker_started = time.perf_counter()
        for snapshots, key in pending:
            if self._cached_group_layout(key) is None:
                self._compute_group_layout(snapshots, key)
        with self._layout_cache_lock:
            self._speculative_planning_seconds += time.perf_counter() - worker_started

    self._speculative_planning_future = self._speculative_planning_executor().submit(
        warm
    )


def _plan_flat_forward(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    checkpoint: AdapterSelection = _impl.Unset,
    memory_minimal: bool = False,
    ensure_slots: bool = True,
    sync_planning_errors: bool = False,
) -> _FlatForwardPlan:
    with self._planning_status(sync_planning_errors):
        plans: list[_ForwardGroupPlan] = []
        output_bytes = self._estimate_group_request_output_bytes(requests)
        logical_tokens = sum(int(request.input_tokens.numel()) for request in requests)
    if sync_planning_errors and ensure_slots:
        self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
    with self._planning_status(sync_planning_errors):
        groups = self._group_active_request_indices(
            requests,
            checkpoint=checkpoint,
            ensure_slots=ensure_slots and not sync_planning_errors,
        )
        selected_max_depth = 0
        for (slot_ref, grad_enabled), group_indices in groups:
            items = tuple(
                self._forward_item(requests[index]) for index in group_indices
            )
            group_input_ids = tuple(item.input_ids for item in items)
            tree, layout = self._select_group_layout(
                group_input_ids,
                memory_minimal=memory_minimal,
                grad_enabled=grad_enabled,
            )
            selected_max_depth = max(selected_max_depth, layout.maximum_depth)
            started = time.perf_counter()
            packed = _impl.materialize_prefix_tree_layout(
                group_input_ids, tree, layout, verify_shared_tokens=False
            )
            self._planning_seconds_accum += time.perf_counter() - started
            plans.append(
                _impl._ForwardGroupPlan(
                    slot_ref=slot_ref,
                    grad_enabled=grad_enabled,
                    request_indices=tuple(group_indices),
                    items=items,
                    packed=packed,
                    layout=layout
                    if getattr(
                        getattr(self, "_planner_reporter", None),
                        "threshold_pct",
                        None,
                    )
                    is not None
                    else None,
                )
            )

        return _impl._FlatForwardPlan(
            request_count=len(requests),
            output_metadata=tuple(
                self._forward_output_metadata(request, checkpoint=checkpoint)
                for request in requests
            ),
            groups=tuple(plans),
            packed_tokens=sum(
                self._physical_tokens(int(plan.packed.tokens.numel())) for plan in plans
            ),
            logical_tokens=logical_tokens,
            output_bytes=output_bytes,
            signature=self._memory_signature_from_requests(
                requests,
                slot_group_count=len(plans),
                grad_modes=tuple(mode for (_, mode), _ in groups),
                slot_groups=tuple(key for key, _ in groups),
            ),
            selected_max_depth=selected_max_depth,
            inactive_logical_tokens=logical_tokens
            - _impl._active_logical_tokens(requests),
        )


def _fill_planner_snapshot(
    self: TrainerRank,
    plan: _AnyForwardPlan,
    check: _MemoryCheck,
    observation: dict[str, Any],
) -> None:
    """Freeze the selected estimator without opening a measurement window."""
    try:
        children = (
            plan.subforwards if isinstance(plan, _impl._SplitForwardPlan) else (plan,)
        )
        # Freeze scalar calibration before forward updates it. Keep owned
        # request references, not new token copies or autograd outputs.
        costs = [self._plan_cost(child) for child in children]
        estimates: list[dict[str, Any]] = []
        for child, cost in zip(children, costs, strict=True):
            estimates.append(
                {
                    "signature": asdict(child.signature),
                    "profile": asdict(profile)
                    if (profile := self._memory_profiles.get(child.signature))
                    else None,
                    "arguments": {
                        "packed_tokens": child.packed_tokens,
                        "output_bytes": child.output_bytes,
                        "logical_tokens": child.active_logical_tokens,
                        "gdn_segments": child.grad_segment_count,
                        "retained_tokens": self._plan_retained_tokens(child),
                        "group_rows": self._plan_group_rows(child),
                        "hybridep_growth_bytes": (
                            self._plan_hybridep_growth_bytes(child)
                        ),
                    },
                    "expected_required_bytes": cost.required,
                    "retained_bytes": cost.retained,
                    "cost_components": asdict(cost),
                    # Observed costs do not reconstruct model/slot eligibility
                    # or the head/GDN/checkpoint inputs used to derive them.
                    "missing_inputs": [
                        "immutable runtime group/slot, head, checkpoint and GDN facts"
                    ]
                    if child.groups
                    else [],
                }
            )
        floor = 0
        if isinstance(plan, _impl._SplitForwardPlan):
            key = self._split_memory_key(plan)
            floor = self._split_memory_floors.get(key, 0) if key is not None else 0
        local_required = max(
            self._split_required_memory(costs),
            int(floor * _impl._MEMORY_SAFETY_FACTOR),
        )
        rank_fields = {
            name: getattr(self, "_" + name)
            for name in (
                "num_layers",
                "hidden_size",
                "param_dtype_size",
                "recompute_granularity",
                "sequence_parallel",
                "attention_output_gate",
                "mlp_activation_factor",
                "gdn_layers",
                "checkpointed_moe_layers",
                "moe_output_bytes_per_token",
            )
        }
        rank_fields["recompute_modules"] = sorted(self._recompute_modules)
        rank_fields["one_layer_recompute"] = self._one_layer_recompute()
        rank_fields["moe_forward_stages"] = getattr(self, "_moe_forward_stages", ())
        rank_fields["geometry"] = asdict(self._geometry)
        rank_fields["topology"] = list(plan.signature.topology)

        def version(tensor: torch.Tensor | None) -> int | None:
            try:
                return None if tensor is None else tensor._version
            except RuntimeError:
                return None

        tensors = [
            (
                item,
                version(item.input_ids),
                version(item.labels),
                {
                    "top_k": item.request.top_k,
                    "logits": item.request.logits,
                    "hidden_states": item.request.hidden_states,
                    "no_grad": item.request.no_grad,
                    "checkpoint": str(item.request.checkpoint),
                },
            )
            for group in plan.groups
            for item in group.items
        ]
        slots = [str(group.slot_ref) for group in plan.groups]
        selected_names = {
            getattr(group.slot_ref, "name", None) for group in plan.groups
        }
        checkpoint_sources = {
            name: self._checkpoint_prefetch_sources[name]
            for name in selected_names
            if name in self._checkpoint_prefetch_sources
        }
        spec = getattr(self.runtime, "model_support_spec", None)
        bridge = getattr(getattr(self.runtime, "provider_bundle", None), "bridge", None)
        pretrained = getattr(bridge, "hf_pretrained", None)
        config = getattr(pretrained, "config", pretrained)
        model = {
            "support_key": getattr(spec, "key", None),
            "name_or_path": getattr(config, "_name_or_path", None),
            "revision": getattr(config, "_commit_hash", None),
            "dtype": str(next(self.runtime.model[0].parameters()).dtype),
        }

        def replay() -> dict[str, Any]:
            remaining = 1_000_000
            incomplete = {
                reason for item in estimates for reason in item["missing_inputs"]
            }

            def tensor_data(
                tensor: torch.Tensor | None, original_version: int | None
            ) -> Any:
                nonlocal remaining
                if tensor is None:
                    return None
                reason = None
                if (
                    tensor.device.type != "cpu"
                    or original_version is None
                    or version(tensor) != original_version
                ):
                    reason = "device_or_modified_input"
                elif tensor.numel() > remaining:
                    reason = "token_inventory_over_limit"
                if reason is not None:
                    incomplete.add(reason)
                    return {
                        "unavailable": reason,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                remaining -= tensor.numel()
                return tensor.tolist()

            requests = [
                {
                    **options,
                    "input_tokens": tensor_data(item.input_ids, input_version),
                    "target_tokens": tensor_data(item.labels, label_version),
                }
                for item, input_version, label_version, options in tensors
            ]
            layouts = []
            cursor = 0
            for group in plan.groups:
                rows = [
                    item["input_tokens"]
                    for item in requests[cursor : cursor + len(group.items)]
                ]
                cursor += len(group.items)
                if group.layout is None:
                    incomplete.add("selected_layout_unavailable")
                elif not all(isinstance(row, list) for row in rows):
                    incomplete.add("layout_inputs_unavailable")
                else:
                    layouts.append(
                        {
                            "input_tokens": rows,
                            "selected_decisions": sorted(
                                group.layout.selected_decisions
                            ),
                            "expected_fingerprint": group.layout.fingerprint,
                            "expected_packed_tokens": group.layout.packed_tokens,
                        }
                    )
            return {
                "memory_replay": {"rank": rank_fields, "estimates": estimates},
                "layouts": layouts,
                "incomplete_reasons": sorted(incomplete),
                "model": model["name_or_path"],
                "model_identity": model,
                "requests": requests,
                "plan": self._telemetry_signature(plan),
                "subforward_request_indices": list(plan.request_indices)
                if isinstance(plan, _impl._SplitForwardPlan)
                else [list(range(plan.request_count))],
                "group_request_indices": [
                    list(group.request_indices) for group in plan.groups
                ],
                "subforward_group_counts": [len(child.groups) for child in children],
                "checkpoint_slots": slots,
                "checkpoint_sources": checkpoint_sources,
                "split_memory_floor_bytes": floor,
                "local_admission_peak_bytes": local_required,
                "reduced_admission_peak_bytes": check.estimated_required_bytes,
                "available_bytes": check.available_bytes,
                "safety_factor": _impl._MEMORY_SAFETY_FACTOR,
                "rank": _impl.dist.get_rank()
                if _impl.dist.is_available() and _impl.dist.is_initialized()
                else 0,
                "device": {
                    "device": str(self.device),
                    **self._planner_device_identity,
                },
            }

        observation.update(
            {
                "predicted": round(local_required / _impl._MEMORY_SAFETY_FACTOR),
                "admission": check.estimated_required_bytes,
                "replay": replay,
                "comparable": True,
            }
        )
    except Exception:
        # Reporting is auxiliary; a diagnostic failure cannot change the
        # model's ordinary execution or replace an OOM.
        _impl._planner_misses._warn("could not prepare planner-miss observation")


def _complete_planner_observation(
    self: TrainerRank, *, phase: str | None = None
) -> None:
    """Close at the existing profiling boundary, retaining only OOM context."""
    try:
        observation = getattr(self, "_planner_observation", None)
        if observation is None or not observation["window_open"]:
            return
        observation["window_open"] = False
        # Later caller/backward work can still perturb another execution's
        # allocator window, so retain weak overlap custody until cleanup.
        phase = observation["phase"] if phase is None else phase
        observation["phase"] = "caller_after_profile"
        if observation["baseline"] is None or not observation["comparable"]:
            return
        if observation["generation"] <= self._planner_overlap_generation:
            _impl._planner_misses._warn(
                "overlapping forwards invalidated the allocator peak interval"
            )
            return
        peak = max(
            observation["peak"], int(_impl.torch.cuda.max_memory_allocated(self.device))
        )
        self._planner_reporter.report(
            predicted_peak_bytes=observation["predicted"],
            observed_peak_bytes=max(0, peak - observation["baseline"]),
            phase=phase,
            replay_factory=observation["replay"],
            admission_peak_bytes=observation["admission"],
            decision=observation.get("decision"),
        )
    except Exception:
        _impl._planner_misses._warn("could not finish planner-miss observation")


def finish_planner_observation(self: TrainerRank) -> None:
    """Release execution context without sampling an unbounded caller peak.

    ART compares completed peaks at its existing profiling boundaries:
    dp_rank_forward's return or forward_micro_batches' iterator resume.
    Caladan calls this at execution end. Direct callers can report a caught
    caller OOM before cleanup, then call this method to release the context.
    An abandoned microbatch iterator cannot mint a completed comparison.
    """
    try:
        self.discard_planner_observation()
    except Exception:
        _impl._planner_misses._warn("could not finish planner-miss execution")


def report_planner_oom(self: TrainerRank, error: BaseException) -> None:
    """Persist a caught CUDA OOM before caller cleanup, then leave it alone.

    This does not suppress, retry, or recover the original failure. An OOM
    after the profiled window retains inputs but cannot claim a partial peak
    for that window, let alone a completed error percentage.
    """
    try:
        original: BaseException | None = error
        seen: set[int] = set()
        while original is not None and id(original) not in seen:
            if getattr(original, "_art_planner_admission_attempt", None) is not None:
                # This error escaped planning, before a new forward window.
                # A prior caller context cannot supply its input/peak facts.
                return
            if isinstance(original, _impl.torch.cuda.OutOfMemoryError):
                break
            seen.add(id(original))
            original = original.__cause__
        observation = getattr(self, "_planner_observation", None)
        if (
            original is None
            or not isinstance(original, _impl.torch.cuda.OutOfMemoryError)
            or observation is None
        ):
            return
        self.discard_planner_observation()
        reasons = []
        if observation["generation"] <= self._planner_overlap_generation:
            reasons.append("overlapping_forward_memory_window")
        if not observation["window_open"]:
            reasons.append("oom_after_profiled_window")
        partial = None
        try:
            if observation["baseline"] is not None and not reasons:
                partial = max(
                    0,
                    max(
                        observation["peak"],
                        int(_impl.torch.cuda.max_memory_allocated(self.device)),
                    )
                    - observation["baseline"],
                )
        except Exception:
            pass
        oom_facts: dict[str, Any] = {
            "type": type(original).__name__,
            "message": str(original)[:4096],
            "baseline_bytes": observation["baseline"],
        }
        try:
            stats = _impl.torch.cuda.memory_stats(self.device)
            oom_facts["allocator"] = {
                key: stats[key]
                for key in (
                    "allocated_bytes.all.current",
                    "reserved_bytes.all.current",
                    "active_bytes.all.current",
                    "num_ooms",
                    "num_alloc_retries",
                )
                if key in stats
            }
            oom_facts["allocator_phase"] = "post_unwind"
        except Exception:
            oom_facts["allocator"] = None

        def replay() -> dict[str, Any]:
            return {
                **observation["replay"](),
                "oom": oom_facts,
                "measurement_valid": not reasons,
                "window_reasons": reasons,
            }

        self._planner_reporter.report(
            predicted_peak_bytes=observation["predicted"],
            observed_peak_bytes=None,
            oom=True,
            phase=observation["phase"],
            replay_factory=replay,
            admission_peak_bytes=observation["admission"],
            partial_peak_bytes=partial,
            decision=observation.get("decision"),
            failure=_impl._planner_evidence.failure(
                original, phase=observation["phase"]
            ),
        )
    except Exception:
        _impl._planner_misses._warn("could not persist planner OOM report")


def discard_planner_observation(self: TrainerRank) -> None:
    try:
        observation = getattr(self, "_planner_observation", None)
        if observation is not None:
            self._planner_active_observations.pop(observation["generation"], None)
        self._planner_observation = None
    except Exception:
        _impl._planner_misses._warn("could not discard planner-miss observation")


def _admission_outcome(self: TrainerRank, local: int) -> int:
    """Existing world fallback MIN: error=0, refusal=1, fit=2."""
    if not (_impl.dist.is_available() and _impl.dist.is_initialized()):
        return local
    value = _impl.torch.tensor(
        local,
        device=self.device if self.device.type == "cuda" else "cpu",
        dtype=_impl.torch.int32,
    )
    _impl.dist.all_reduce(value, op=_impl.dist.ReduceOp.MIN)
    return int(value.item())


@_backward_region
def _recover_admission(
    self: TrainerRank,
    search: Callable[[], Any],
    describe: Callable[[Any], tuple[_AnyForwardPlan, _MemoryCheck]],
    update: Callable[[Any, _MemoryCheck], Any],
    *,
    context: str,
    sync_across_dp: bool,
    admit_refusal: Callable[[_ForwardRefusal], Any] | None = None,
) -> Any:
    reporter = getattr(self, "_planner_reporter", None)
    if reporter is None or reporter.threshold_pct is None:
        return self._recover_admission_impl(
            search,
            describe,
            update,
            context=context,
            sync_across_dp=sync_across_dp,
            admit_refusal=admit_refusal,
        )
    try:
        decision = _impl._planner_evidence.Decision(
            context, sync_across_dp=sync_across_dp, owner=self
        )
    except Exception:
        return self._recover_admission_impl(
            search,
            describe,
            update,
            context=context,
            sync_across_dp=sync_across_dp,
            admit_refusal=admit_refusal,
        )
    with _impl._planner_evidence.scope(decision):
        try:
            result = self._recover_admission_impl(
                search,
                describe,
                update,
                context=context,
                sync_across_dp=sync_across_dp,
                admit_refusal=admit_refusal,
            )
        except BaseException as error:
            # Cancellation/termination keeps its original behavior. Do not
            # turn abandoned planning into a fabricated completed event.
            if isinstance(error, Exception):
                try:
                    setattr(error, "_art_planner_admission_attempt", decision.id)
                except Exception:
                    pass
                self._report_planning_failure(decision, error)
            raise
        try:
            _, check = describe(result)
            decision.selected = check.sample
            if decision.outcome == "planning":
                decision.outcome = "admitted"
            return update(result, replace(check, decision=decision.snapshot()))
        except Exception:
            _impl._planner_misses._warn("could not retain admission decision")
            return result


def _report_planning_failure(
    self: TrainerRank, decision: _planner_evidence.Decision, error: Exception
) -> None:
    try:
        refusal = decision.refusal
        refused = decision.outcome == "refused" and refusal is not None
        decision.outcome = "refused" if refused else "planning_error"
        observation: dict[str, Any] = {
            "predicted": None,
            "admission": None,
            "replay": lambda: {"incomplete_reasons": ["no_selected_plan"]},
        }
        if refused:
            decision.selected = refusal.check.sample
            observation.update(
                admission=refusal.check.estimated_required_bytes,
                replay=lambda: {"incomplete_reasons": ["planner_snapshot_unavailable"]},
            )
            self._fill_planner_snapshot(refusal.plan, refusal.check, observation)
            snapshot = observation["replay"]

            def replay() -> dict[str, Any]:
                payload = snapshot()
                return {
                    **payload,
                    "candidate_matches_check": refusal.check_matches_plan,
                    "incomplete_reasons": [
                        *payload.get("incomplete_reasons", []),
                        *(
                            []
                            if refusal.check_matches_plan
                            else ["candidate does not describe denying check"]
                        ),
                    ],
                }

            observation["replay"] = replay
            if not refusal.check_matches_plan:
                observation["predicted"] = None
        self._planner_reporter.report(
            predicted_peak_bytes=observation["predicted"],
            observed_peak_bytes=None,
            admission_peak_bytes=observation["admission"],
            phase="planning",
            event="admission_refused" if refused else "planning_error",
            decision=decision.snapshot(),
            failure=_impl._planner_evidence.failure(error, phase="planning"),
            replay_factory=observation["replay"],
        )
    except Exception:
        _impl._planner_misses._warn("could not retain planning failure")


def _recover_admission_impl(
    self: TrainerRank,
    search: Callable[[], Any],
    describe: Callable[[Any], tuple[_AnyForwardPlan, _MemoryCheck]],
    update: Callable[[Any, _MemoryCheck], Any],
    *,
    context: str,
    sync_across_dp: bool,
    admit_refusal: Callable[[_ForwardRefusal], Any] | None = None,
) -> Any:
    """Pure search, at most one smaller-plan refresh, then one recovery."""
    original: TrainerRankMemoryError | None = None
    refused: _ForwardRefusal | None = None
    best: _ForwardRefusal | None = None

    def reject() -> Any:
        assert refused is not None
        if admit_refusal is not None:
            # Only the exhausted memory-refusal path changes. Every peer
            # must have a supported candidate; never override an EP or
            # failed planning/runtime capability guard.
            allowed = (
                getattr(self, "_allow_oversized_batches", False)
                and refused.overridable
                and best is not None
            )
            selected = None
            if allowed:
                assert best is not None
                selected = (
                    update(best.candidate, best.check)
                    if best.candidate is not None
                    else admit_refusal(best)
                )
            width = (
                selected.stats_global_count
                if isinstance(selected, _impl._CandidateMicroBatch)
                else 0
            )
            agreed = self._recovery_reduce(
                [float(allowed), float(width), -float(width)],
                op="MIN",
                sync_across_dp=sync_across_dp,
            )
            if agreed[0] == 1 and agreed[1] == -agreed[2]:
                decision = _impl._planner_evidence.current(self)
                if decision is not None:
                    decision.outcome = "admitted_oversized"
                return selected
        decision = _impl._planner_evidence.current(self)
        if decision is not None:
            decision.outcome, decision.refusal = "refused", refused
        raise refused.error(context) from original

    def finish(value: Any) -> Any:
        nonlocal refused, best
        if isinstance(value, _impl._ForwardRefusal):
            if sync_across_dp and admit_refusal is not None:
                # Minimum-wave split checks are TP x CP-local. Compare
                # them against larger world-priced waves on the same basis
                # so every DP rank retains the same logical wave width.
                value = replace(
                    value,
                    check=self._refresh_memory_check(
                        value.check,
                        sync_across_dp=True,
                    ),
                )
            refused = value
            if value.overridable and (
                best is None
                or value.check.estimated_required_bytes
                < best.check.estimated_required_bytes
            ):
                best = value
            return None
        plan, check = describe(value)
        if sync_across_dp:
            check = self._refresh_memory_check(check, sync_across_dp=True)
        if check.fits:
            return update(value, check)
        refused = _impl._ForwardRefusal(
            plan,
            check,
            "selected plan exceeds freshly sampled available memory",
            candidate=value,
        )
        if (
            best is None
            or check.estimated_required_bytes < best.check.estimated_required_bytes
        ):
            best = refused
        if isinstance(value, _impl._CandidateMicroBatch) and value.fallback is not None:
            fallback = value.fallback
            if (
                best is None
                or fallback.check.estimated_required_bytes
                < best.check.estimated_required_bytes
            ):
                best = _impl._ForwardRefusal(
                    fallback.plan,
                    fallback.check,
                    refused.message,
                    candidate=fallback,
                )
        return None

    value = search()
    result = finish(value)
    if result is not None:
        return result
    assert refused is not None
    original = refused.error(context)
    with self._cache_recovery_episode() as (owner, started):
        if not isinstance(value, _impl._ForwardRefusal):
            # A formerly fitting width is not proof that the minimum cannot fit.
            value = search()
            result = finish(value)
            if result is not None:
                return result
            if not isinstance(value, _impl._ForwardRefusal):
                # Counters moved again: do not loop or reclaim for a large width.
                assert refused is not None
                self._snapshot_planning_telemetry(refused.plan, refused.check)
                return reject()
        assert refused is not None
        if self._try_cache_recovery(
            refused.check,
            sync_across_dp=sync_across_dp,
            owner=owner,
            started=started,
        ):
            value = search()
            result = finish(value)
            if result is not None:
                return result
        assert refused is not None
        self._snapshot_planning_telemetry(refused.plan, refused.check)
        return reject()


def _plan_retained_tokens(self: TrainerRank, plan: _FlatForwardPlan) -> int:
    if (
        plan.signature.topology[2] <= 1
        or not plan.signature.grad_enabled
        or self._recompute_granularity == "full"
        or self._geometry.moe_experts
    ):
        return plan.packed_tokens
    topology = self._topology()
    # Bound each group's largest attention/GDN layout, including TP padding.
    # Groups can place their peak on different ranks; summing is conservative.
    return sum(
        self._physical_tokens(
            max(
                1,
                self._max_rank_model_tokens(
                    _impl._pad_packed_batch(group.packed, multiple=int(topology.tp)),
                    topology=topology,
                ),
            )
        )
        for group in plan.groups
    )


@contextmanager
def _planning_status(self: TrainerRank, enabled: bool) -> Generator[None, None, None]:
    """Exchange pure local planning errors before any later WORLD check."""
    if not enabled:
        yield
        return
    primary: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary = exc
    exchange_error: BaseException | None = None
    try:
        succeeded = self._all_ranks_true(primary is None)
    except BaseException as exc:
        exchange_error = exc
    # Leave the exchange's except block before raising the original error.
    if primary is not None:
        raise primary
    if exchange_error is not None:
        raise exchange_error
    if not succeeded:
        raise RuntimeError("Local planning failed on another DP rank")
