# ART fixed workspace plus token slope: Phase 1 CPU prototype

Owner: Performance investigation, Aragorn perf fork. Admission/review owner:
Aragorn. Base ART ee91943cd; Caladan inspected at 2d9a19489. No GPU runs,
deployment, or PR are part of this phase. AMENDMENT-1 applies.

## Decision and scope

Opt into replacement of empirical peak/token extrapolation for newly measured profiles with
`fixed + slope * priced_tokens`. Keep every existing signature field, model-local
profile ownership, output charge, analytic/static floor, checkpoint floor,
packed/logical-row pricing, and admission/agreement site. Do not infer an unseen
GPU guarantee from a CPU replay. The hard result sought here is coverage of the
supplied measurements after fitting; the 1.5x aspiration is evaluated separately.

The failing regression was run before changing production code. Its transcript
is `dev/memory_fixed_slope/poisoning-before.txt`: a synthetic 8-token workspace
observation reconstructed from run6's rate makes an already calibrated signature
price 8192 tokens at exactly 568,361,327,001 bytes. The synthetic observation is
not claimed to be a recorded GPU sample. Run6 itself records successful CP8
81920-token forwards at ranks 1, 8, and 32 before refusing 8192 and 32768.

## Model and merge

For each existing signature retain compute observations `(n, c)`, where `n` is
physical packed tokens and `c = max(0, peak_delta - requested_output_bytes)`.
Repeated lengths max-merge their compute peak. Keep the upper concave hull of
these points, including both endpoints. This is a sufficient statistic for
`max(c - s*n)` for every nonnegative slope; unlike an eight-point eviction rule,
it cannot forget a constraint needed after a slope change.

The slope anchor is the longest observation, provided it has at least **8192
packed tokens**. Its slope is `anchor_compute / anchor_tokens`; before such an
observation the empirical slope is zero. This includes fixed workspace in the
largest root's rate, deliberately conservatively, instead of attempting a
fragile two-sample derivative. Tiny waves never choose the anchor or the slope.
The fixed term is `max(0, max_i(c_i - slope*n_i))`, rounded upward to whole bytes.
Use integer numerator/denominator arithmetic for integer token counts.

8192 is the smallest supplied sweep root (1024 rows/rank for balanced CP8),
not a hardware-independent law. A roughly 0.5 GiB fixed workspace contributes
about 64 KiB/token at this threshold versus tens of MiB/token on tiny roots.
The report includes threshold sensitivity. Calibration should reach an eligible
root; a MoE 4096-token-only calibration does not establish an empirical slope.
Analytic floors and the unchanged 8x profile trust window still apply.

When the longest root changes, refit both slope and fixed from the hull. Do not
max-merge fitted coefficients: a smaller fixed rate observed first must not
become permanent. Union plus hull is associative, commutative and idempotent.
The hull may grow in pathological workloads; this prototype does not silently
cap it and lose safety. A production storage bound needs a separately justified
conservative coarsening rule.

## Safety, monotonicity and rank agreement

For every retained or hull-dominated observation,
`fixed >= c_i - slope*n_i`, hence `fixed + slope*n_i >= c_i`. Restoring outputs
and taking the maximum with unchanged physical floors preserves that bound.
Multiplying by the unchanged 1.10 safety factor gives at least measured memory
at all supplied fitted points. This is an in-sample bound; it does not establish
that an unmeasured shape, routing distribution, cold kernel or future length
cannot exceed the estimate. In particular, the first unmeasured 8K root may still
be underpredicted by the existing cold floors. GPU validation remains required.

Slope and fixed are nonnegative. Priced tokens are the existing maximum of
packed tokens and normalized logical tokens; this is nondecreasing. Adding
logical-row and segment charges, taking maxima with monotone physical floors,
and multiplying by a positive safety factor preserves monotonicity for otherwise
fixed geometry. Changes of geometry are governed by existing planner bounds,
not a claim that unrelated layouts have identical costs.

Predictions depend on the set of observations, not their arrival order. Tests
permute observations and merge partitions, including tiny, repeated and
nonmonotone peaks. The legacy warm diagnostic fields may still describe which
caller phase ran first, but do not choose version-2 admission prices.

Profiles remain rank-local. No new collectives are introduced. Existing MAX of
required bytes and MIN of available bytes/local fit booleans still decide
admission; diagnostics can combine extrema from different ranks, but the fit
predicate remains the conjunction of each rank's own demand versus budget.
The profile-trust collective and 8x extent/sharing windows remain unchanged.

## Admission-effective arithmetic (AMENDMENT-1)

The scalar path computes
`required = int((output + max(static_compute, empirical_compute)) * 1.10)`.
Packed pricing adds `_PACKED_PRICED_LOGICAL_ROW_BYTES * logical_tokens` and
GDN per-segment state inside empirical compute, as before. CP adds its output
coexistence charge inside static compute. `_memory_check` then adds
`int(HybridEP_growth * 1.10)` outside that product.

Split paths retain the existing sum of retained child costs plus largest
workspace, charge HybridEP growth once, and max with the checkpoint/backward
floor times 1.10. `_subforward_cost` also preserves the input-gradient,
adapter-gradient, head-workspace and cold-transient floor. Learned split floors
are safety-scaled. These paths already supply a safety-scaled operand to
`_memory_check_required`; that method does not scale it again.

For the native allocator, `usable = max(0, physical_free - int(total * 0.03))`.
For other allocators, reusable reserved bytes are added before that subtraction.
The test-only cap may further lower usable to `cap - current_allocated`.
Eviction/recovery compares required plus the same 3% reserve against free
memory; it does not add a second reserve to a value already compared with usable.
Therefore the effective/measured ratios here use **the actual required operand
against usable**. Relative to physical free memory the equivalent requirement
is `required + reserve`; adding reserve again when comparing against usable
would double-count it. The sweep lacks device-total/reserve samples, so a
physical-free ratio cannot be reconstructed exactly. No reserve is invented.

Recorded `ART_predicted_increment_bytes` is already safety-scaled. OLD raw is
reconstructed by dividing it by 1.10 (less than two bytes rounding uncertainty
for these single-root cases); OLD effective is the recorded value. NEW reports
both the raw empirical line and the full raw/effective admission calculation.
No second 1.10 factor is applied to recorded predictions.

## Existing OOM-risk bug: OLD effective under measured

Seven of the 60 supplied cases have an OLD admission operand below the largest
measured rank increment. This is an existing estimation risk, not proof of an
actual OOM: all these cases had headroom. It must not be hidden by adding the
budget reserve to the demand. Model 0 is Qwen3.8-27B; model 1 is
Qwen3.6-35B-A3B. All rows below have CP8 and length 8192.

| Model | Operation | LoRA | OLD effective bytes | Measured bytes | Effective/measured |
|---|---|---:|---:|---:|---:|
| dense | forward_backward | 1 | 1476467097 | 1561926144 | 0.945286 |
| dense | forward_backward | 32 | 1476467097 | 1476984832 | 0.999649 |
| MoE | forward_backward | 1 | 753604403 | 1225593856 | 0.614889 |
| MoE | forward_backward | 8 | 791012147 | 994050560 | 0.795746 |
| MoE | forward | 1 | 516810342 | 771460608 | 0.669912 |
| MoE | forward | 8 | 516810342 | 771460608 | 0.669912 |
| MoE | forward | 32 | 516810342 | 771460608 | 0.669912 |

Full OLD raw/effective ratios for all 60 cases: `dev/memory_fixed_slope/OLD.md`
and `old.csv`. The canonical source hash and per-rank attempts are vendored in
`tests/unit/fixtures/memory_fixed_slope/`. The fit uses measured rank maxima,
not averages. Rank-local fitting is also replayed to test actual collective
semantics; max/min reductions do not imply identical local coefficients.

## Existing profile paths and persistence

Version 2 is explicit in each prototype profile. Collection is enabled by the internal
`fixed_slope_memory_profiles()` context; existing version-2 profiles keep using
their fit outside that scope. The default stays version 1 in this Phase 1 CPU
prototype, so baseline contracts remain executable and production activation
requires Aragorn's review and GPU validation. The companion Caladan patch handles
v4 encoding and merges; activating collection across startup and calibration is
a subsequent coordinated step. No environment/user-facing tuning knob is added.
Keep the existing dataclass names and fields so old records can still be constructed with defaults and encoded by
Caladan's `asdict`. Missing version means version 1, with its exact legacy
pricing behavior until an update. On update, keep the legacy peak/token rate as
an immutable conservative floor: the original observations cannot be recovered
from a scalar rate. Clearing/recalibrating under a new cache key is required to
remove historical poisoning; do not guess it away during migration.

Ordinary version-2 profiles use the fixed+slope bound over all observations;
the old order-dependent warm rate no longer discounts it. This intentionally
retains first-call workspace in the fixed term and covers cold observations.
Legacy warm fields remain readable; version-1 profiles retain their old warm
formula. Retained-graph pricing remains its separate conservative zero-fixed
plus max-rate line, including its existing trust gates and checkpoint floor.
It is not replaced by a peak-derived retained fraction.

`_CallerMemoryProfile` keeps its caller/CP evidence and whole-caller observation
gates. Version 2 uses the all-observation fixed+slope bound plus its existing
compute peak floor; it does not grant new CP ownership discounts. Version 1
keeps componentwise CP ownership coverage and its existing covering-root curve.
`_BoundedMemoryProfile` keeps its separate scoped signature and diagnostic
root envelope; version 2 uses fixed+slope instead of the old fitted marginal
rate/horizon, while version 1 remains replayable. Analytic floors still apply.

Caladan's encode/decode already transports additional dataclass fields, but its
generic scalar-max merge is **not valid** for a refitted slope/intercept. The
legacy bytes/token diagnostic stays separate from the exact rational slope,
so a zero empirical slope does not require weakening its positive-rate check.
Required coordinated changes: (1) key version 3 -> 4; (2) validate v2 observation
hulls and exact coefficients; (3) merge observation sets using ART's helper and
refit, never max the new slope/fixed independently; (4) preserve legacy floors
and scoped evidence; (5) round-trip and merge tests. A reviewable companion patch
is supplied as `dev/memory_fixed_slope/caladan-v4.patch`; the Caladan worktree is read-only here.
Old v3 rows remain decodable with ART's v1 defaults, but must not seed a v4
calibration intended to discard poisoning. ART source/runtime pins must move
together during subsequent admission, preserving explicitly frozen runs.

Planner `capable()` consumes replica compatibility, calibration status,
`max_context`, and adapter residency capacity. `room()` consumes resident
adapter bytes and calibration state; neither parses ART profile coefficients.
No schema change to these consumers is needed. Publish pending/provisional
until coordinated recalibration completes, using existing gates.

## Refusal semantics and remaining risks

Do not modify `_memory_check_required`, `_all_ranks_have_memory_profile`, the
public memory-error type, or `_recover_admission_impl.reject`. In particular,
leave the collective exhausted-refusal/oversized bypass site that #2699 relies
on unchanged. Tests cover local refusal before execution, rank disagreement,
MAX/MIN order, and the existing recovery/collective contracts.

The 1.5x aspiration is not guaranteed by a single line or preserved static
floors. For example, the dense forward 32K recorded estimate is 4.192x measured,
and its static packed activation floor alone can retain that conservatism.
A lower cost after discarding a physical floor would violate this lane's scope.
Other risks: convex growth beyond the largest root, sparse signature coverage,
route-dependent MoE workspace, shape-dependent CP imbalance, unbounded worst
case hull storage, stale v3 caches, and first-call costs not yet observed.

## Phase 1 results

All **60/60** supplied measured rank-max increments are covered after fitting.
`dev/memory_fixed_slope/COMPARISON.md`, `comparison.csv` and `comparison.json`
report OLD and NEW raw/measured and effective/measured for every case, flag
all seven OLD effective underestimates, and retain per-case usable limits.
NEW raw is obtained from the same scalar consumer with safety set to 1 in the
CPU harness; OLD raw can only be reconstructed from recorded effective bytes.

| Component | Minimum raw/measured | Maximum raw/measured | Minimum effective/measured | Maximum effective/measured |
|---|---:|---:|---:|---:|
| NEW empirical fixed+slope line plus outputs | 1.000000 | 1.660974 | 1.100000 | 1.827071 |
| NEW scalar admission replay including available static floors | 1.000000 | 3.810755 | 1.100000 | 4.191831 |

**20/60** scalar replay cases exceed 1.5x. This aspiration is not met. Even an
optimal single effective line covering just dense forward's 8192 and 131072
measurements must predict at least 1.613x the measured 16384 case: the chord's
value there is about 1.282 GB, versus 0.795 GB measured. Thus a 1.5x bound at
every point is incompatible with a single line covering these observations,
independently of ART's additional static floors. The chosen longest-root slope
is deliberately more conservative than an optimized chord.

This is an **in-sample scalar consumer replay**, not an exact reconstruction of
the historical native GPU admission plan. The fixture has admission operands,
measurements and per-rank attempts, but no serialized internal signature flags,
operator graphs or floor inputs. The replay uses the existing ordinary signature
fields, separate model/op/LoRA buckets, target outputs (4 bytes/token), balanced
CP8 rows and metadata-only models. Unsupported operator-specific checkpoint/GDN
floors cannot be reconstructed from these records and are not invented. Actual
native floors or a scoped caller's existing peak floor can increase the result.
The raw empirical line already covers every measured rank, so these additional
nonnegative floors cannot undermine the coverage result. The hard static-floor
code paths remain unchanged and are covered by baseline/CPU contract checks;
the reported tightness range is a CPU replay result, not a GPU-qualified claim.

Rank-local fits were also replayed for every rank and case, using each rank's
own attempts. Their reduced maxima cover all 60 measured maxima. No averaging
or synthetic union of CP ownership vectors is used. Threshold sensitivity is in
`thresholds.csv`: thresholds 2048–32768 produce the same completed fit because
the longest root is 131072. The supplied fixture cannot identify an optimal
sub-8K cutoff. 8192 is the first measured grid point, justified as a conservative
initial policy rather than a statistically established universal threshold.

Run6's raw/effective refusals and the extent of actual fit evidence are in
`RUN6.md`; its 81920 forward success has no recorded peak bytes. Wanda has no
same-run successful measured peak, and no fabricated ratio is reported.

The actual CPU planner selects **32 -> 384 datums** for the requested dense RL
shape (568 total, families of 16 sharing 240 prompt tokens, 213 completion tokens).
This is approximate: CP8, LoRA 8, **24 GiB usable**, matching the host-overhead
harness budget; root-sweep peak observations are transferred to the RL signature
only for this what-if. Production signature matching does not permit that
transfer. OLD includes an identical synthetic 8-token workspace observation
that sets the host harness's 2 MB/token rate. NEW keeps the same observation in
its fixed term. OLD selects 7296 packed rows at 16.389 GB effective; NEW selects
87552 packed rows at 25.166 GB effective (usable is 25.770 GB). Wave count for
568 datums changes from about 18 to 2 if these widths remain feasible. This is
not a measured RL memory bound or a production capacity promise. `wave.json`
contains selected operands, alternate widths, profiles and assumptions.

## Validation and admission status

See `dev/memory_fixed_slope/VALIDATION.md` for exact commands and limits.
Phase 1 stops at a pushed CPU prototype with no PR. GPU qualification, default
activation, and a coordinated Caladan format/key change remain admission-owner
work before production use; no deployment or pin update is performed here.
