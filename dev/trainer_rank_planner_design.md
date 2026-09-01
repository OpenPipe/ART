# Holistic TrainerRank planner: landing design brief

Phase 0 deliverable for the single-PR landing. Sources: the research thread's
frozen behavior contract (2026-08-31), its sealed acceptance evidence, and
direct empirical verification of the research planner's behavior (this
document records the verified facts the acceptance suite pins).

## What main already has (reuse, do not rebuild)

- `art.megatron.prefix_tree_packing.prefix_tree_pack` — the packing primitive
  (retains `max_depth`; per contract it stays for tests/preprocessing only).
- `art.megatron.gdn.gdn_prefix_tree` — GDN execution planning/lowering.
- `TrainerRank` execution machinery: `_select_next_micro_batch` (adaptive
  width), `_plan_flat_forward` (grouping + packing), `_memory_check` +
  `_MemoryProfile` (cross-rank memory agreement), `_project_head` (head
  chunking), CP/GDN/HybridEP forward paths, checkpoint slots (#821).

## What the PR adds

1. **Planner core** (`_prefix_tree_planner.py`, `_planner_cost.py`,
   `_prefix_tree_performance_search.py`): canonical radix tree, mandatory
   candidate family, calibrated integer cost model, bounded deterministic
   Pareto-beam search. The tree/candidates/search modules are adopted from the
   research implementation — they were its clean, oracle-validated core — with
   the induced-forest bridge and research-only surfaces removed.
2. **Selection policy**: `select_prefix_tree_layout` = mandatory candidates +
   bounded refinement search under the calibrated production score.
3. **Knob-free public API**: `TrainerRank(runtime)`. Scope, precisely: the
   planner decides the prefix-sharing layout (arbitrary depth, per-subtree
   share/replay); microbatch width reuses main's adaptive selector, made
   sharing-aware (a no-sharing token count accepts a width, and the planner's
   actual layouts are priced only when that bound would reject one); head
   chunking and memory margins are internal calibrated constants, not planner
   decisions; `dp_rank_forward` plans once and raises
   `TrainerRankMemoryError(predicted_peak_bytes, usable_limit_bytes,
   suggestion)` when the unsplit plan cannot be admitted (best-effort internal
   splitting is a follow-up PR); `TrainerRankRuntimeSupportError` at
   TP>1/PP>1 (follow-up widens the seam).
4. **Distributed identity WITHOUT a leader protocol** (deliberate deviation
   from the research design, in the spirit of "or whatever's simplest"):
   layout selection is a pure deterministic function of (content identity,
   topology, coefficient version) — it never reads rank-local memory facts —
   so every rank in a model-parallel replica computes the identical plan from
   identical inputs, and steady state is a content-hash cache hit (~1 ms).
   Memory admission and width selection consume facts that are already
   collectively agreed via the existing MAX/MIN all-reduces. The research
   needed a leader because its planning path cost seconds (exact lowering,
   preflight, proofs); none of that machinery exists here, so a leader plus
   recipe wire format would add latency and code while preventing nothing.
   The goals the leader served (no digest votes, no proofs, minimal
   collectives, bounded planning fraction) are enforced directly by the
   acceptance gates.
5. **Telemetry**: `last_forward_telemetry()` with `selected_max_depth`,
   `planning_ms` (critical path, including speculative submission cost), and
   `speculative_planning_ms` (hidden worker time). Env-gated test anchor
   forcing (`ART_TRAINER_RANK_TEST_HOOKS` + `ART_TRAINER_RANK_TEST_ANCHOR`).

## Verified facts the acceptance suite pins (empirical, research planner)

- Sealed GPU win-cell shape (GRPO 2x8, system 2048 / prompt 8192 /
  completion 512): production score selects depth 3, 26,624 physical tokens
  for 172,032 logical — identical at layers=2 and layers=12; matches the
  sealed cold witness exactly.
- Heterogeneous control (16 unique 4k rows): selects depth 1, no decisions.
- Tiny sealed-corpus families (grpo_like/deep_comb/mixed_branch): production
  score correctly selects NO sharing (tiny segments cannot pay GDN/CP costs).
  Nonuniform selection in the sealed gate came from the *search-quality*
  harness under an injected adversarial scorer — a search-capability result,
  not production policy. The acceptance gate was corrected accordingly
  (2026-09-01, pre-implementation).
- Candidate family on those trees retains all anchors: 0-decision, full-
  decision, and depth-1 layouts present; exhaustive layout counts 4/2048/1024.

## Calibrated production score (provenance: research `_impl.py` frozen source,
mirrored and test-locked by the sealed gate harness)

```
cp = max(1, cp_size); L = max(1, layers)
transformer = packed_tokens * 1024
imbalance   = ceil(packed_tokens / cp) * (96 + 32*cp)
launch      = segment_count * (96 + 32*cp) * 1024
exchanges   = selected_decision_count * (64 + 32*cp) * 1024
gdn         = uses_gdn * ( min(1, max(0, depth-1)) * L * 768 * 1024
                         + max(0, depth-2)         * L * 256 * 1024 )
total = L * transformer + imbalance + launch + exchanges + gdn
score = (total, packed_tokens, segment_count, maximum_depth)   # lexicographic
```

Known limitation carried from research: this undervalues full sharing on some
GRPO cells (sealed: full-sharing arm 875.7 ms vs automatic 1,132.8 ms on the
win cell). Constants are versioned (`coefficient_version`) for future
recalibration; not addressed in this PR.

## Width feasibility is decided by the memory-minimal layout

The cost-optimal layout can decline sharing at one width and accept it at a
wider one, so its packed-token count — and therefore "does the cost-optimal
plan fit" — is not monotone in wave width (research review reproduced: fits
at width 1, fails at 2, fits at 3). The width search's exponential/binary
structure requires a monotone predicate, so feasibility is defined as "the
memory-minimal (full-sharing) layout fits": full sharing minimizes packed
tokens and its count is monotone in width by construction. Admission then
executes the cost-optimal layout when it fits and the memory-minimal layout
otherwise; the chosen mode is recorded per width so materialization builds
exactly the layouts that were priced. `dp_rank_forward` applies the same
fallback before refusing. Both bounds are cheap O(tokens) walks of the packing
primitive (no-sharing and unlimited-depth sharing); planner pricing runs only
inside the band where they disagree.

## Planning cost engineering

Two behavior-preserving optimizations keep exact width pricing cheap:
- `build_canonical_prefix_tree` scans each shared segment with one vectorized
  tensor comparison over the active rows' span (tokens only ever matter for
  equality) and hashes row content from tensor bytes — 31 ms -> 2 ms on the
  sealed win-cell shape, byte-identical output (300-seed equivalence test
  against the scalar reference algorithm).
- The bounded search caches each candidate's dominance vector and beam key at
  construction instead of recomputing them per Pareto comparison — 28 ms ->
  5 ms per search on an 8-decision tree, identical results (210 baseline
  fingerprints).
- Width probing skips exact pricing at or above a width whose memory-minimal
  layout already failed (the monotone predicate above).
Benchmark (2-layer, fresh tokens, forced multi-wave): per-step planning
67.7 ms -> 42.3 ms, step wall 185.8 ms -> 160.7 ms with sharing-aware widths
(2 waves) — within ~5% of the no-sharing-bound 4-wave step (153.5 ms) while
keeping the memory-to-throughput crossover.

## Overlapped (speculative) next-wave planning

``forward_micro_batches`` pre-plans the predicted next wave (exactly the
width the search will seed with — the largest width so far — over this DP
rank's strided slice) on a single background thread while the generator is
suspended at the yield — i.e. during the caller's forward/backward GPU time.
Because selection is a pure memoized function, speculation can never change a
plan: a correct prediction turns the next wave's selection into a cache hit,
a wrong one leaves an unused LRU entry. No cancellation or stale-state
machinery is needed (the hazard that kept this out of the research freeze).
Token snapshots for the worker are immutable CPU clones taken on the calling
thread (the same bytes that produced the cache key), so a caller mutating its
tensors after the yield cannot poison the cache; CUDA inputs skip speculation
so the worker never touches the device. The synchronous submission cost is
charged to `planning_ms`; hidden worker time is reported separately as
`speculative_planning_ms`. See the acceptance README for measured numbers.

## Best-effort internal splitting (follow-up PR)

Contract (relaxed, 2026-09-01): try not to raise when splitting would make
execution feasible; account for every returned graph staying live together;
if finding out is too expensive or fragile, refuse — worded as "unable to find
a feasible split", never as a claim that none exists.

Mechanism:
- `dp_rank_forward` (and the minimum wave of `forward_micro_batches`) plans
  unsplit first (cost-optimal, then memory-minimal). If neither is admitted, a
  bounded, deterministic ladder tries 2, 4, ... subforwards, partitioned along
  the canonical prefix tree so shared-prefix siblings stay together, stopping
  at the fewest subforwards that fit; one request per subforward is the last
  rung.
- Chunk pricing mirrors width probing (cheap no-sharing bound accepts, cheap
  full-sharing bound rejects, planner pricing only in between). Chunks execute
  larger-ephemeral-first (deterministic), and subforward `j` is admitted
  against the retained memory of every earlier subforward plus its own
  transient estimate; the backward upper bound (all retained graphs plus the
  largest reusable workspace) must also fit. This is the research thread's
  cumulative memory invariant: retained adds, ephemeral does not, backward
  workspaces take a max.
- Retained fraction (memory still allocated after a forward returns, as a
  fraction of its estimate) is learned online per signature; the default
  before any observation is 1.0 (everything retained), so a cold call that
  cannot fit unsplit refuses until a profile exists.
- The complete ordered split is admitted before any model execution; there is
  no retry after the first forward. A failure in a later subforward raises
  `TrainerRankPartialExecutionError` (a `TrainerRankMemoryError`) so partial
  execution is reported distinctly.
- Splitting is disabled under expert parallelism in this release (HybridEP
  capacity must not be resized between subforwards while earlier graphs are
  live); the refusal says so.
- Telemetry: `subforward_count` in `last_forward_telemetry()` and
  `MicroBatchStats`. Test-only `ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES`
  (gated by `ART_TRAINER_RANK_TEST_HOOKS`) caps usable memory so GPU gates
  induce conversion/decline deterministically without ballast tensors.

GPU gate (`--phase split-conversion`), mirroring the sealed research cell
(Qwen3.5-4B, 4 layers, CP1, 4 inputs): unlimited runs unsplit; a cap between
the split and unsplit requirements converts (>= 2 subforwards) with outputs
matching the unsplit reference, a single combined backward, and a
reverse-order per-output backward all with every graph live; a cap below the
smallest request refuses before any model execution.

## Explicitly out of scope (follow-ups)

Head chunking and memory margins as data-dependent planner decisions; TP>1 admission seam; cost-model
recalibration. Not planned: infeasibility proofs, all-rank planning/digest
agreement, HybridEP/CUDA instrumentation from the research diff.
