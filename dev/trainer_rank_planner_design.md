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
   splitting is a follow-up PR); `TrainerRankRuntimeSupportError` at PP>1
   (TP>1 was refused at landing as a calibration caution and is admitted
   again by the TP-support follow-up; see "Tensor parallelism" below).
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
  bounded, deterministic ladder tries 2, 4, ... subforwards (at most one
  request each), cutting the requests in prefix-local depth-first order into
  token-balanced chunks — so most sharing stays inside one chunk, though a
  cut can still divide a sibling subtree — and stops at the fewest
  subforwards whose rung check passes.
- Rung check. Every returned graph stays live, so subforward `j` needs its
  own transient peak plus the memory retained by the subforwards before it.
  Each of those sums is bounded by *all retained memory plus the largest
  ephemeral share*, which therefore decides a rung by itself in any order
  (this is the research thread's cumulative invariant: retained adds,
  ephemeral does not). Chunks execute larger-ephemeral-first, which minimizes
  the running forward peak. The same quantity is the headroom the caller's
  backward can count on — every graph live plus one subforward's
  forward-ephemeral memory free again. That is a *heuristic* for backward
  workspace, not a bound: a backward may need more than its forward's
  ephemeral memory (e.g. kernel autotune workspaces). The ballast arm of the
  GPU gate measures it on a real cell instead of claiming it.
- Cost. The cheap full-sharing lower bound (one O(tokens) CPU scan per chunk)
  rejects a rung without planning anything; a surviving rung is priced
  exactly with cost-optimal layouts and, failing that, memory-minimal ones
  (whose packed tokens equal the lower bound). The planner therefore runs
  for at most one rung — the one that executes — and the whole ladder is
  O(tokens log n) cheap scans plus one exact pass.
- Retained fraction (memory still allocated after a forward returns, as a
  fraction of that forward's observed peak — a physical ratio, so it needs no
  trusted denominator; the first, cold call's static estimate is far below
  the real peak) is learned online per signature: `None` until observed, then
  max-merged (an observed 1.0 is distinct from "unobserved"). Admission
  applies it to a subforward's estimated peak, which is at least the real
  peak whenever the estimate is trusted.
  It is trusted only within the profile's packed-token trust range and near
  its observed logical/packed ratio, so a small profiled forward cannot
  authorize a much larger split. Unobserved means 1.0 (everything retained),
  so a cold call that cannot fit unsplit refuses until a profile exists.
  Limitation: the observation is taken at forward return and says nothing
  about backward; TrainerRank cannot see the caller's backward peak for
  `dp_rank_forward` (the micro-batch path folds the post-yield peak into
  `bytes_per_token`, not into the retained fraction).
- Collectives. Ensuring checkpoint slots is a world collective; the ladder's
  length depends on this rank's DP-local inputs, so slots are ensured exactly
  once per call and all further planning skips the ensure. Memory checks
  all-reduce only within the TP×CP group (identical inputs). In the
  minimum-wave path every DP rank runs its own ladder and then all ranks
  agree on the outcome with one collective, so a refusal is raised everywhere
  or nowhere.
- The complete ordered split is admitted before any model execution; there is
  no retry after the first forward. Any execution-time memory failure of an
  admitted split raises `TrainerRankPartialExecutionError` (a
  `TrainerRankMemoryError`) naming how many subforwards completed, so it is
  never mistaken for an up-front refusal. Each subforward's outputs carry
  their own slot-graph sentinel, so slot load/step stays blocked until every
  subforward's graph is released.
- Splitting is disabled under expert parallelism in this release (HybridEP
  capacity must not be resized between subforwards while earlier graphs are
  live); the refusal says so.
- Telemetry: `subforward_count`, `subforward_request_indices`,
  `predicted_peak_bytes` and `usable_limit_bytes` in
  `last_forward_telemetry()`; `subforward_count` in `MicroBatchStats`.
  Test-only `ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES` (gated by
  `ART_TRAINER_RANK_TEST_HOOKS`) caps usable memory so the deterministic GPU
  arm can induce conversion/decline without ballast.

GPU gate (`--phase split-conversion`), mirroring the sealed research cell
(Qwen3.5-4B, 4 layers, CP1, 4 inputs), in two arms. `--pressure cap`
(deterministic control flow): unlimited runs unsplit; a cap between the split
and unsplit requirements converts (>= 2 subforwards) with outputs matching the
unsplit reference, a single combined backward and a reverse-order
per-subforward backward with every graph live; a cap below the smallest
request refuses before any model execution. `--pressure ballast` (physical
memory safety, no test hooks): live ballast tensors bring the real usable
budget under the unsplit requirement; the call converts with parity, the
combined backward runs with the ballast still live, and the observed
forward+backward peak must stay within both the budget the planner admitted
against and its predicted peak; deeper ballast refuses before execution.

What the ballast arm taught us: on this cell a training forward retains
~99% of its peak for backward (layer activations plus the chunked head's
saved logits), so a 2-way split lowers the requirement by only
(1−f)·R/2 ≈ 0.5% — splitting cannot shrink retained activations, only the
transient share. The arm therefore sizes its ballast from the measured
fraction and reports the window width honestly. `no_grad` forwards
(reference/old-policy logprobs; retained ≈ outputs only) are the
demonstrated high-value case: they convert at a fraction of the unsplit
requirement, which the arm also shows under real pressure. The training
benefit is workload-dependent and small in this sealed landing cell; the
research thread's full-height cell retained closer to 92%, so CP/GDN,
output-heavy or workspace-heavy training shapes may have several gigabytes
of splittable transient memory, and grad-enabled support is kept for them.
Callers that could backward per subforward would gain more, but the public
contract keeps every graph live, so that is not modeled.

Accepted limitations: the full-sharing lower bound can conservatively reject
a rung whose cost-optimal layouts would have fit if retained-profile trust
changes with the sharing ratio (a false refusal, never an unsafe admission —
within the bounded-search contract); and a cold oversized `no_grad` call
still refuses until a compatible profile exists (a later simplification could
model `no_grad` retained memory directly from the known output bytes).

## Tensor parallelism (follow-up PR)

TrainerRank accepted TP>1 before the planner landed; #826 refused it as a
calibration caution, not because anything was missing. The machinery is
unchanged and pre-dates the planner: the vocab-parallel output head (log-Z,
target logprobs, top-k and full logits reduced/gathered across the sharded
vocabulary in the same head chunks), the sequence-parallel hidden gather and
output-layer SP toggle, TP padding of packed batches (appended singleton tree
nodes, after planning, so padding can never become a shared subtree), sharded
vs replicated LoRA gradient reduction, memory checks all-reduced within the
TP×CP group, and memory-profile signatures keyed by (dp, tp, cp, pp) so a TP
topology calibrates itself online.

Known, accepted limitations: the cost model carries CP terms but no TP terms
(layout ranking is expected to survive since sharing shrinks tokens uniformly,
but constants are uncalibrated — folded into the cost-model recalibration
follow-up, to be fitted from fresh TP2 telemetry rather than by dividing the
estimate by TP, which would turn conservative refusals into unsafe
admissions); the cold static estimate ignores sharding (conservative); TP
padding happens after planning, so estimates and telemetry count fewer than
TP tokens too few; the all-shards-`-inf` output-head case is a documented
non-goal (unreachable for supported models without a vocabulary-wide mask).

Gates (test-first; all failed on the refusing tree):
- `tests/unit/test_trainer_rank_topology.py`: TP>1 constructs, PP>1 and
  multi-chunk runtimes still refuse.
- `--phase tp2-public` (2× H200, Qwen3.5-4B full model, DP1×TP2×CP1, public
  `dp_rank_forward`, active LoRA slot), plus the identical cell at `--tp 1`
  as the control: both TP peers plan the same physical layout on every call;
  the automatic planner shares more deeply than depth-one on the hierarchical
  GRPO shape; odd packed lengths exercise sequence-parallel padding with
  outputs at the final real token; losses finite; measured rows compile-free
  and plan-cache-stable; paired timing reported, not gated. Numerics are
  gated by `--phase tp-compare` (CPU) on the two dumps. bf16 rounding differs
  whenever the reduction order changes — a different packing and a different
  TP degree both change it — and over 36 layers that noise is ~1.3% of the
  mean logprob magnitude (0.19–0.20 nats per target token) on this cell, so
  absolute tolerances borrowed from same-packing comparisons cannot separate
  a TP defect from noise. The reference is therefore the control's own
  cross-layout divergence measured in the same run: same-layout TP-vs-TP1
  divergence (mean and max) and cross-layout divergence at TP (outputs and
  LoRA gradients) must stay within 1.5× of it, losses must agree, and the
  differences must be unstructured (no request outlier, final tokens like
  the body, no bias). Measured: same-layout TP2-vs-TP1 1.39% vs the 1.31%
  reference (ratio 1.06), cross-layout ratios 1.05 (outputs) and 1.06
  (gradients), losses within 0.06%, flat per-request profile, tail = body.
- `--phase dp2-tp2-waves` (4× H200, DP2×TP2, public `forward_micro_batches`):
  at least two waves under the test-only cap, DP replicas with different
  payloads, identical wave shapes within each TP pair, every input returned
  exactly once in order, forward and backward per wave, automatic vs
  depth-one parity, and an empty-DP-slot arm; completing is the no-hang gate.
- CI: `dev/trainer_rank_check.py` at TP=2 (16 request combinations, two
  slots) next to the CP=2 run; the GDN TP2 kernel parity test now also runs
  with LoRA.

## Explicitly out of scope (follow-ups)

Head chunking and memory margins as data-dependent planner decisions;
cost-model recalibration (including TP terms). Not planned: infeasibility
proofs, all-rank planning/digest agreement, HybridEP/CUDA instrumentation from
the research diff.
