# Phase 2: default-off PR implementation

Aragorn approved 36a46dc27's design with a 1.75 cold guard, boot anchors and
fresh v4 calibration. Both changes remain default off. GPU validation is pending.
The Phase 1 commit remains unchanged in history.

## Admission and collection

Before a signature has an observation of at least 8192 packed tokens, scalar
admission uses the maximum of the v2 fit, unchanged analytic floors and the
original max-rate envelope. The latter upper-bounds the old warm/scoped discounts
and is independent of which caller happened to run first. Inside the largest
observed extent there is no new multiplier; beyond it (including no profile)
apply 1.75, in addition to the existing 1.10 safety factor. Outputs, checkpoint
bounds, retained split costs and HybridEP growth receive the cold charge once.
The 3% device reserve stays on the usable-budget side; it is never counted twice.

Followup-1 extends the cold factor beyond the last observation **also after
anchoring**, on NEW rather than the unanchored OLD envelope. An 8K-only anchored
fit without this factor under-predicts held-out v7 peaks (effective minima
0.898638 dense and 0.961873 MoE). `anchor_choices.py` freezes boot evidence and
MAX-reduces rank-local predictions; see `ANCHOR-CHOICES.md` for all choices.

A runtime observation beyond an unanchored profile's existing 8x trust window
cannot promote an anchor. Offline/restore merges are set-based; admissible real
wave observations feed the same hull. The trust collective, MAX/MIN admission,
conjunction of local fits and exhausted collective rejection site are unchanged.

Ordinary observation updates do not import v1 state into a fresh v2 fit. The
explicit restore merge alone converts lost v1 evidence to an immutable rate
floor. Caladan v4 starts with no startup-smoke/v1 seed, rejects legacy-bearing or
unanchored saved v4 rows, and merges v2 hulls through ART's helper.

## Fixture results

`unanchored.csv` reports each recorded OLD admission operand guarded by 1.75,
with both raw and effective ratios. `phase2-ratios.json` summarizes:

| Replay | Min effective/measured | Max effective/measured |
|---|---:|---:|
| 60 recorded OLD operands under the unanchored guard | 1.076056 | 7.335704 |
| Seven OLD effective underestimates under that guard | 1.076056 | 1.749387 |
| Anchored v2 scalar replay | 1.100000 | 4.191831 |

The guard test reconstructs the recorded native OLD floor as an input to the
real scalar consumer: the sweep does not serialize native operator floor inputs.
It tests both absent and unanchored profiles. These are CPU fixture bounds, not
new GPU measurements. The anchored empirical line alone remains 1.10–1.827071x;
retaining static floors means the 1.5x aspiration is still not met in every case.

## Boot coverage and time estimate

`signature_inventory.json` comes from the real signature constructor over CPU
metadata stand-ins. For a single selected slot, segment count, individual root
lengths, sharing ratio and checkpoint name are not signature fields. The keyed
fields are topology, planner coefficients, slot group count, request-mix set,
grad enabled/modes and actual selected slot shapes. `short_requests` affects
pricing but is excluded from equality. Model identity is rank-instance local.

Dense target roots and target RL waves share the same signature. MoE training
has a special single-target caller signature: `_ScopedMemorySignature` with
`bounded_caller_peak=True`. Multiple roots use ordinary `_MemorySignature`.
Hidden versus target outputs and route presence/mixed route presence change
`request_mix`. Forward and backward change grad fields/slot shape mode. The
inventory confirms 64 and 256 roots have equal keys for each family.

Caladan's `STARK_MEMORY_FIXED_SLOPE=1` wraps startup and serving collection,
selects v4 keys, discards short smoke profiles, and by default measures an 8K
root plus 64x128 waves for each configured warm LoRA rank and operation (A).
`STARK_MEMORY_ANCHOR_TOKENS=8192,32768` selects B; an empty value selects the
configured root-only ladder (C). The choice is part of v4 keys only.
The bundle probe uses the actual target or StableReinforce head/loss/backward
paths, including all-present and mixed replay routes on MoE. It checks that the
exact expected signature actually gained an anchor; a split/unanchored rung
cannot silently publish completion. New bundles must expose the probe API.

Coverage is for the full target layout already used by `warm_config` and the
supplied production evidence. Different user adapter target masks or unconfigured
LoRA ranks remain under the cold rule until they acquire their own observations;
they are not claimed to be boot-anchored. CP1 fused waves bypass TrainerRank's
admission-profile consumer and are unchanged. GPU testing must verify all new
native probe shapes and route/hidden first-use workspaces before enabling.

`calibration_timing.json` is an estimate, using run6's warm MoE FB 8K/32K
2.3525/2.8230 s, its warm forward times, v7 dense timings, 0.9 s plan/rung, and
one full first-8K-call allowance per output family. Option B adds **465 s dense**
(24 new rungs) and **927 s MoE** (72). At today's approximately 780 s boot that
is about 1245/1707 s, below `CALIBRATION_SECONDS=3600`. If first-use cost repeats
for every LoRA rank, the estimates rise to approximately 1241/2408 s extra
(2021/3188 s total). This is not a deadline guarantee; 64/256-root GDN and
StableReinforce/route-replay compile costs have not been GPU measured.

A budgets **416 s dense / 827 s MoE** extra without crediting removed v3 root
rungs; 12/36 added RL rungs. It is selected because none of the twelve measured
131K cells is memory-refused. The enabled calibrated context bound remains the
largest actually probed root (8K for A); v9 must bypass Caladan's context cap as
v7 did. The unchanged ART 8x trust window also still applies: 131K and 262K
numbers from frozen 8K evidence are numerical extrapolations, not a trust claim.

## Validation

- 339 affected ART CPU tests pass (including Gloo, split, refusal, trust and
  recompute contracts); a focused 64-test cold/migration/order sweep also passes.
- Changed ART files and affected tests pass ruff, format and ty.
- Required repository-wide prek is run; the CPU environment's pre-existing ty
  diagnostics are reported separately in the PR, as in Phase 1.
- Caladan calibration/warm/planner tests pass with UTF8 local PostgreSQL, whose
  fixture stops and removes the server. New bundle probes exercise real Torch
  loss/autograd on CPU stand-ins and reject incomplete anchor collection.
- Caladan changed-file checks use its pinned ty 0.0.14. Deployment keeps the
  flag at 0. Its flag-off v3 ladder/key is compared with main, including a
  main-generated golden key and exact source-digest compatibility mapping.

After ART merges, Caladan must move its ART source and runtime/image pins to the
same reviewed ART commit, rebuild the trainer runtime and publish the matching
bundle before enabling. Frozen legacy images/runs are not repinned by these PRs.

Followup validation: 123 fixed/slope CPU tests pass, including 24 held-out
anchor tests. The required all-files prek rerun passes ruff/format/lock; its
lock-synchronized type environment reports 35 diagnostics, all identical to
untouched ee91943cd (`followup-typecheck-parity.json`). This is a different
dependency/tool environment from the earlier 49-diagnostic CPU snapshot, not
35 new errors. Changed files and the replay separately pass focused type checks.
Caladan #2706 has clean history: 268 affected tests against proposed ART;
249 pass / 19 enabled-only skips against the old pin, including the flag-off
boot, golden key, SQL calibration, executor and planner checks. PostgreSQL is
stopped. Enabled v4 still requires the coupled source/runtime pin bump.
An additional 391 ART tests pass (8 optional native-dependency skips), covering
warm/caller profiles, planner options, CPU Gloo planning status, EP admission,
split peaks and recompute accounting: 514 passed across the focused/broader runs.
