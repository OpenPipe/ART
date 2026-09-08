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
  completion 512): the landing's version-1 score selected depth 3, 26,624
  physical tokens for 172,032 logical (the sealed cold witness). The fitted
  version-2 score selects prompt-level sharing at CP4 (depth 2, 28,672
  physical — the measured fastest layout there, 791 ms vs 872 ms for full
  sharing) and full sharing (26,624) at CP1; both satisfy the gate (depth > 1,
  under a quarter of the logical tokens).
- Heterogeneous control (16 unique 4k rows): selects depth 1, no decisions.
- Tiny sealed-corpus families (grpo_like/deep_comb/mixed_branch): production
  score correctly selects NO sharing (tiny segments cannot pay GDN/CP costs).
  Nonuniform selection in the sealed gate came from the *search-quality*
  harness under an injected adversarial scorer — a search-capability result,
  not production policy. The acceptance gate was corrected accordingly
  (2026-09-01, pre-implementation).
- Candidate family on those trees retains all anchors: 0-decision, full-
  decision, and depth-1 layouts present; exhaustive layout counts 4/2048/1024.

## Fitted production score (coefficient version 2)

The landing shipped the research thread's layout score with constants that
were, as the research thread later confirmed, hand-set rather than fitted
(1 µs per token per layer as a structural scale; `96 + 32·cp` per segment and
`64 + 32·cp` per shared edge hand-shaped; 768 µs per GDN layer as the smallest
quantum preserving four measured winners; 256 µs as a launch-floor proxy),
applied to the total layer count rather than the GDN layer count, and blind
to TP. The recalibration replaced them with a fitted table
(`_planner_cost.COEFFICIENTS_MILLI_US`, integer milli-microseconds per
feature unit) over ten interpretable integer term functions
(`_planner_cost.TERM_FUNCTIONS`) of four O(segments) layout features
(`layout_features`: packed tokens, segment count, dependency levels, a
segment-length histogram) and the planner facts (cp, tp, layers, GDN
layers). The ten terms are the ones that carried weight in the calibration:
per-rank token work with CP-exchange and TP-collective terms, a GDN per-token
surcharge, attention KV exchange across CP ranks, tiny-per-rank segments per
layer, dependency levels crossing CP or TP ranks, and GDN level hand-offs with
their TP interaction. Candidate terms that fitted to zero (segment launches,
fan-out, shared tokens, attention area, small-M token surcharges) were dropped
from the module; a future campaign can reintroduce them.
Only quantities that differ between a call's layouts are priced; everything
a call shares (logical tokens, the output head, model size) cancels in
ranking.

```
score = (Σ_term coefficient[term] · term(features, facts), packed_tokens,
         segment_count, maximum_depth)                           # lexicographic
```

Calibration protocol (`--phase cost-calibrate`, `dev/trainer_rank_cost_fit.py`):
every mandatory candidate layout of a cell (and the production selection) is
timed through the public API — forward and backward through an active LoRA
slot, compile-free, max-rank — with its features; the fit is a non-negative
least squares on within-cell paired timing deltas (cells weighted equally,
pairs weighted by their scale), refined by a deterministic regret-minimizing
coordinate search, validated on whole held-out cells. Cells: hierarchical
GRPO g8/g16/g4x4 shapes on Qwen3.5-4B (GDN, 24 of 32 layers) and Qwen3-4B
(attention) at 2 layers and full height, three heterogeneous controls, and
real Ellavox groups, at TP1/TP2 × CP1/CP2/CP4 on H200 bf16.

What the data showed:
- The cost of an additional shared prefix level is a GDN effect: on the GDN
  model it grows when the level's state hand-offs cross CP or TP ranks (at
  CP4 and TP2 sharing a 1,023-token system across two prompt groups is a net
  loss; at TP1/CP1 it is a win), while the attention model pays almost
  nothing for it and benefits from the extra level even at CP4.
- Per-rank token work scales with `tp × cp`, GDN layers cost more per token
  than attention layers, and rows in segments that are short *per rank*
  (threshold × cp) run inefficient kernels.
- The sealed "full sharing 876 ms vs automatic 1,133 ms" gap on the win cell
  was the research run's online calibration wandering between five layouts;
  the frozen version-1 score actually selected full sharing there. Its real
  misranking on that cell was prompt-level sharing (791 ms) vs full sharing
  (872 ms) — an ~80 ms level cost the model priced at under 1 ms.

Gates (held-out cells, noise-qualified): pairwise ordering ≥ 90% on pairs
separated by more than 3%, median regret ≤ 2%, p95 ≤ 5%, none above 10%,
clear winners selected within 5%. The table is fitted on 45 cells and
evaluated on all 58 (3,849 within-cell pairs; the 11 odd Ellavox groups are
the pre-registered holdout, and the two Ellavox CP4 cells re-measured after
issue #840 are held out as well, 13 held-out cells): it ranks 98.1% of
separated pairs correctly, p95 regret 2.9%, max 4.2%, no clear misses; the
holdout passes. Ablations withholding
every TP2 cell, every CP2 cell, or the whole attention model pass; withholding
every heterogeneous cell misranks one CP4 heterogeneous cell by 9.5%, and
withholding every CP4 cell does not extrapolate (25%), so those cells stay in
the fit. Robustness: the table fitted on the first 38 cells, run through the
real selector on the 18 later cells, was already within 4.2% everywhere, and
the production selection timed in the later CP2/TP2 cells had median regret
−0.2%, max 0.4%. The hand-set version-1 score on the original 56 cells: 78.6%
pairwise, max regret 67% (an Ellavox group at CP2).

Calibrated domain: a table applies only to the execution classes it was
measured on (`CalibratedTable`, `select_scoring`): device class (compute
capability plus a memory-system bucket, so the 80 GB H100 that shares
capability 9.0 with the 141 GB H200 is not admitted), parameter dtype, model
geometry read from the Megatron config (hidden and FFN widths, attention head
geometry, GDN state shape, expert geometry — never a model name; layer counts
are scoring facts, not identity) and parallel shape (TP × CP × EP × ETP).
Admission is exact: the dense hidden-2,560 table admits the Qwen3.5-4B and
Qwen3-4B geometries at the four measured shapes (TP1 × CP1/2/4 and TP2 × CP1)
except the withheld pair Qwen3-4B × TP1 × CP4 (measured and fitted, but the
score is known to misrank real-data groups there; see the known limitation
below). A table's `withheld` pairs fall back to version 1, and the manifest
records each with its reason. TP2 × CP2, CP8, TP4, expert parallelism, other
widths and MoE models keep the version-1 score (kept verbatim) with a one-time
warning unless another table admits them. Each certificate names its table and
records the admitted device classes, dtypes, geometries and shapes; the
certificate test asserts they equal the production table's sets, that every
geometry was measured at every admitted shape, and that the withheld pairs are
exactly the manifest's.
`dev/trainer_rank_cost_calibration_manifest_<table>.json` lists the exact cells
each recipe or lattice launch of that table produces; the fitter's `--manifest` validation requires every
non-excluded cell to be present and complete, rejects unexpected cells and
duplicate cells with differing execution fingerprints (geometry included), and
the certificate test asserts the 58 fitted identities plus the 16 excluded
blind-spot cells of the Qwen3-4B real-data launch (below).

Second calibrated class (2026-09-04): Qwen3.5-35B-A3B (GDN + MoE, hidden
2,048, 256 experts top-8) on H200 bf16, measured over a shape lattice that
separates context from expert parallelism (EP1 at CP1/2/4 and TP2, EP2 at
CP2/CP4/TP2, EP4 at CP4; `dev/trainer_rank_cost_calibration_lattice.sky.yaml`)
on the synthetic families and the Ellavox groups: 73 cells, 3,658 within-cell
pairs, 13 held-out odd-Ellavox cells. The same ten terms fitted to their own
table (`gdn-moe-h2048-h200-bf16`) rank 97.1% of separated pairs correctly,
median regret 0%, p95 2.4%, max 4.9%, gates passing on every shape (EP shapes
included), so no expert-parallel term was needed; the version-1 fallback this
class used before loses up to 35% on the large Ellavox groups. Fifteen cells
of the lattice are excluded with reasons: CP1 cells the single-GPU memory
admission cannot run or refuses (issue #848) and the CP4/EP1 Ellavox cells
that segfault in the expert grouped GEMM on real routing (issue #851).

Known limitation (2026-09-04, dense controls): the O(segments) layout
features cannot see the context-parallel plan a layout produces, and on
attention-only models with real data at CP4 that plan dominates. Two layouts
of an Ellavox group with near-identical features (about 13k packed tokens,
six versus seven segments, the same 12k-token longest segment) differ by 25%
in measured time because one needs four exchange waves with rank loads
7168/3584/1536/790 and the other two waves with 6656/2560/2048/1751. The
Qwen3-1.7B, Qwen3-8B and Qwen3-14B controls therefore fail the gates at CP4
with the ten terms (held-out max regret 26%, 14% and 17%), and the same group
measured on the certified Qwen3-4B geometry at TP1 × CP4 shows a 35% regret
for the shipped table (the version-1 score reaches 15% on another group
there; neither is adequate). The dense certificate had no
attention-plus-real-data cells at CP4, so its metrics did not cover this.

Resolution (2026-09-04/05, per review; superseded for TP1 × CP4 by the planner
recalibration of 2026-09-08 below, which retired the re-ranker). The attention classes get their own
tables where the ten-term gates pass — Qwen3-1.7B (`dense-attn-h2048-h200-bf16`)
and Qwen3-8B (`dense-attn-h4096-h200-bf16`) at TP1 × CP1/CP2 and TP2 × CP1,
Qwen3-14B (`dense-attn-h5120-h200-bf16`) also at TP2 × CP2 (42/42/56 cells;
pairwise 96.5%/99.9%/99.9%, max regret ≤1.1%) — and TP1 × CP4 is admitted
through a **two-stage selection** (`ReRanker`, `CalibratedTable.reranked_shapes`,
`select_prefix_tree_layout(..., reranker, plan_structure)`) where that selection
pays for itself: the ten-term score (a shortlist table fitted on every measured
shape; its job is recall) keeps the three cheapest layouts of the search plus
the depth-one anchor, each shortlisted layout is priced by the structure of the
context-parallel plan it produces — remote wave count and largest per-rank token
load, per layer, from the CP planner's own assignment
(`summarize_prefix_tree_plan`, through the planning bundle cache so the
selected layout's plan is reused) — and the lowest second-stage score wins
(ties keep the cheaper layout). The two physical drivers came out of the
measured schedules: on equal-load layouts an extra remote wave costs 2.4 / 1.8 /
0.6 ms per layer on 1.7B / 8B / 14B (the CP planner's own model prefers the
extra wave; issue #854), and the max-rank load carries the rest.

The re-ranker's certificate (the `reranker` block, recomputed by
`tests/unit/test_planner_cost_certificate.py` on all cells and on the held-out
cells) requires: every clear measured winner in the shortlist; the ranking
gates on the two-stage selection; never more than 2% worse than the cheap-only
or the version-1 selection; and the synchronous planning cost of pricing the
shortlist covered by the mean execution saved against **both** single-stage
alternatives (cache reuse and overlap are not measured and do not count). That
last gate decides which classes ship it: Qwen3-8B (max regret 0.8%; planning
1.6% of cell time against 1.8% saved over version 1) and Qwen3-14B (4.1%; 1.0%
against 1.2%) pass on all and held-out cells; Qwen3-1.7B does not — the
re-ranker ranks its cells (max 2.6%) but on that small model pricing three
plans costs 2.6% of a cell's time against 2.0% saved over version 1 — nor does
the Qwen3-30B-A3B attention-MoE class (0.9% saved against 1.1% planning on all
cells, and version 1 is 0.3% better on its held-out cells). Both keep version 1
at TP1 × CP4, with the reason recorded per cell in their manifests. The dense
table withholds Qwen3-4B × TP1 × CP4 (version 1's 15% worst observed cell is the
less harmful fallback than the table's 35%; its 7 real-data CP4 cells are too
few to certify a re-ranker).

The tables and re-rankers were calibrated on training forwards (forward +
backward). Groups that run without gradients keep the version-1 score and are
cached under their own identity (`_PlannerFacts.grad_enabled`) until
forward-only execution is validated separately.

Fourth calibrated class (2026-09-04/05): Qwen3-30B-A3B (attention + MoE, hidden
2,048, 128 experts top-8; `attn-moe-h2048-h200-bf16`) on H200 bf16 over the same
EP-deconfounding lattice (112 cells; 3 CP1 cells excluded for the single-GPU
memory admission of issue #848; no CP>1/EP1 segfaults on this model). The
eight-round timings of TP1 × CP2 (EP1/EP2) and TP1 × CP4 (EP1/EP4) carried
sporadic 10–20 s stalls on about 4% of rounds, so those shapes were re-measured
with sixteen rounds (the stalls persist at about 4% of rounds but shrink to
about 2×, and medians hold). The class shows the attention CP4 blind spot.
Admitted directly at TP1 × CP1, TP1 × CP2 (EP1, EP2) and TP2 × CP1 (EP1, EP2)
(67 cells; pairwise 96.3%, max regret 2.2%). TP1 × CP4 keeps version 1: the
two-stage re-ranker ranks it (max 2.1% at EP1, EP2 and EP4) but does not pay
back its planning cost against version 1 (above).

Fifth calibrated class (2026-09-04): Qwen3.5-27B (dense GDN + attention,
hidden 5,120; `dense-gdn-h5120-h200-bf16`) on H200 bf16 over TP1 × CP1/2/4,
TP2 × CP1 and TP2 × CP2 (70 cells; four CP1/CP2 cells incomplete or lost to the
single-GPU memory admission of issue #848, one TP2 × CP2 cell lost to an NCCL
collective timeout during warm-up). Clean timings (median per-candidate spread
0.4–0.7% except TP2 × CP2), and the ten terms rank this GDN class at CP4 as they
do the 35B GDN MoE class, so no re-ranker: admitted directly at TP1 × CP1/2/4
and TP2 × CP1 (52 cells; pairwise 97.0%, max regret 3.4%, held-out 1.8%).
TP2 × CP2 fails its gates marginally (p95 5.1%, one clear miss) and keeps
version 1. This is the class where the version-1 fallback hurt most: on the
largest Ellavox group at CP4 it chose a layout 112% slower than the best
(4,084 ms against 1,927 ms), and 26% slower at TP2 × CP2.

### Recalibrated context-parallel planner and re-certification (2026-09-08, issue #854)

The context-parallel assignment planner (`_search_generic_chunk_assignment`)
chooses the remote wave count and the chunk owners from its own per-layer cost
model, and three of its assumptions made it choose slower plans. It priced no
host work per remote stage, although one more wave costs a measured
2.3–2.6 ms per layer (forward + backward) on the small and medium classes and
0.2–0.6 ms where layers are long enough to hide it; it priced the KV fetch at
about 14 GB/s, ten times slower than NVLink, and hid that phantom latency
behind extra waves; and it balanced attention pairs only, so on causal rows
the early rank owned several times the tokens of the last while the rest of
the layer's compute went unpriced. The recalibration adds a host cost per
remote stage (`planner_remote_stage_host_ms`, 1.2 ms per direction), prices
fetch and reduce at 30 ns per token, and adds a per-owned-token compute cost
from the provider geometry (`planner_owned_token_ms`, `estimate_owned_token_ms`;
routed-expert work is charged per owned token only without expert parallelism,
because with EP the routed rows are redistributed across the group). The
search also keeps every rank's ownership contiguous — moves shift boundary
chunks to the rank owning the neighbouring chunk — because the GDN layers chain
recurrent state along the attention layout and a rank owning two separate
ranges pays extra hops (one +15% cell in the first hardware validation, the
only fragmented plan of its cell).

Every CP > 1 cell of every calibrated class was re-measured with a paired
A/B (`--planner-ab`): each layout is timed under the current and the legacy
planner in alternating rounds on the same node; legacy rows carry
`planner_variant` and are never fitted. Per-layout median change of the
current planner against the legacy one, the change of each cell's best
layout, and the change of the timed production (automatic) selection:

| class | shape | layouts faster | per-layout median | cell-best | production |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | TP1 × CP2 | 127/138 | −5.4% | −8.8% | −7.9% |
| | TP1 × CP4 | 122/138 | −12.7% | −8.5% | −13.0% |
| | TP2 × CP2 | 113/138 | −2.1% | −3.5% | −6.4% |
| Qwen3.5-4B (GDN) | TP1 × CP2 | 120/137 | −3.4% | −3.8% | −3.9% |
| | TP1 × CP4 | 110/137 | −3.3% | −3.9% | −4.7% |
| Qwen3-8B | TP1 × CP2 | 137/138 | −23.5% | −20.7% | −20.4% |
| | TP1 × CP4 | 128/138 | −14.6% | −21.0% | −20.6% |
| | TP2 × CP2 | 135/138 | −10.2% | −10.8% | −10.4% |
| Qwen3-14B | TP1 × CP2 | 137/138 | −27.6% | −25.3% | −25.6% |
| | TP1 × CP4 | 129/138 | −16.8% | −28.7% | −29.5% |
| | TP2 × CP2 | 135/138 | −17.8% | −16.1% | −14.6% |
| Qwen3-30B-A3B | TP1 × CP2 | 128/138 | −5.0% | −5.6% | −5.6% |
| | TP1 × CP2 EP2 | 106/138 | −1.2% | −3.8% | −3.8% |
| | TP1 × CP4 | 110/138 | −6.6% | −5.8% | −6.5% |
| | TP1 × CP4 EP2 | 120/138 | −6.7% | −5.5% | −7.7% |
| Qwen3.5-35B-A3B (GDN) | TP1 × CP2 | 100/135 | −0.9% | −0.7% | −0.8% |
| | TP1 × CP2 EP2 | 100/137 | −0.5% | −0.3% | +0.2% |
| | TP1 × CP4 | 18/30 | −0.8% | −1.2% | −1.0% |
| | TP1 × CP4 EP2 | 109/135 | −1.1% | −1.2% | −1.2% |
| | TP1 × CP4 EP4 | 113/137 | −1.2% | −1.6% | −1.8% |
| Qwen3.5-27B (GDN) | TP1 × CP2 | 116/137 | −5.9% | −5.4% | −5.3% |
| | TP1 × CP4 | 111/137 | −3.3% | −5.6% | −4.7% |
| | TP2 × CP2 | 113/131 | −5.2% | −5.6% | −5.5% |

Same-plan layouts (where both planners produce the identical assignment)
change by 0.0%, which validates the pairing. On the attention classes 85 of
124 measured CP4 layouts lose remote waves and 119–120 lower their maximum
rank load; none fragments. The few layouts that got slower are the uniform
synthetic GRPO cells at CP4 (Qwen3-8B cal-grpo-g4x4 +4–5%), where the
token-balanced plan trades a lower maximum load for more attention pairs on
one rank.

**GDN classes.** On Qwen3.5-4B (24 GDN layers of 32) twelve CP4 layouts are
slower by more than 2% although every one of them has a single-wave plan at
least as balanced as before. They are GDN-planner decisions reacting to the
ownership it is handed, not attention-plan regressions. On the largest
Ellavox group (`g5 no_sharing`, +22.9%) the GDN planner chains one
12.6k-token sequence across the four ranks under the legacy layout but keeps
every sequence local under the new one, stacking two on rank 3 (25.3k GDN
tokens against about 15.8k per rank); its own model predicts the chain saves
4.18 ms per layer in the first case and 3.97 ms in the second, straddling the
hand-set 4.0 ms chain gate (`GdnPlannerConfig.cp_chain_min_runtime_delta_ms`),
while the measured penalty is about 15 ms per GDN layer — the GDN model
under-prices chaining roughly fourfold. On the 7k-token groups (+6–9%) the
greedy owner search leaves one rank empty and stacks depth-1 segments on two
ranks. Neither changes what the layout planner picks (production −3.9% /
−4.7%) and the refit absorbs the tail; a GDN-planner recalibration (chain gate,
owner search) is a separate follow-up.

**Expert-parallel classes.** On Qwen3-30B-A3B the gains are 5–7% at CP2 and
CP4 with one clear regression: the heterogeneous synthetic cell
`cal-hetero3` at TP1 × CP2 EP2 (+8.0% on its best layout, 8/8 rounds). With
routed-expert work no longer charged per owned token under expert
parallelism the search balances attention pairs and hands rank 0 10,752 of
16,756 tokens (legacy 7,168), and the measurement says the per-owned-token
work at EP2 (dispatch, permutation, router) is larger than the projections
alone; a dispatch term for EP is a follow-up. The 35B GDN-MoE class is near
neutral, as expected where 35 ms layers hide the host work an extra wave
costs. Two of its cells segfault in the expert grouped GEMM on real routing
(issue #851) under the re-measurement — `cal-ellavox g2` at CP2 (already
excluded) and now at CP4 EP2 — and are excluded with that reason.
On Qwen3.5-27B (48 GDN layers of 64) the class gains 3–6% on every shape;
its Ellavox g6 CP4 layouts show the same +3–5% GDN owner-assignment
sensitivity as the 4B class, and `cal-hetero3` at TP2 × CP2 (not admitted)
the same +8% as on the 30B class. One TP2 × CP2 cell was again lost to an
NCCL collective timeout in the tensor-parallel group during warm-up.

**Re-certification.** Every table was refit from the current-planner rows of
the paired campaign plus its unchanged CP1 and TP2 × CP1 cells, with the
checked-in recipe of its certificate:

| table | cells | pairwise | p95 regret | max regret | change |
| --- | --- | --- | --- | --- | --- |
| dense-h2560 (Qwen3.5-4B, Qwen3-4B) | 58 | 98.7% | 1.9% | 2.8% | was 98.1% / 2.9% / 4.2% |
| dense-attn-h2048 (Qwen3-1.7B) | 42 | 100% | 0.6% | 1.0% | CP exchange term → 0; three-term structure kept |
| dense-attn-h4096 (Qwen3-8B) | 56 | 99.8% | 1.1% | 1.7% | TP1 × CP4 now fitted directly (99.3%, max 1.7%) |
| dense-attn-h5120 (Qwen3-14B) | 70 | 98.8% | 0.9% | 2.9% | TP1 × CP4 now fitted directly (94.2%, max 2.9%) |
| attn-moe-h2048 (Qwen3-30B-A3B) | 67 | 99.9% | 1.0% | 2.2% | was 96.3% / max 2.2%; TP1 × CP4 stays excluded |
| gdn-moe-h2048 (Qwen3.5-35B-A3B) | 80 | 96.6% | 2.7% | 4.2% | CP4/EP1 synthetic cells re-measured; one more #851 exclusion |
| dense-gdn-h5120 (Qwen3.5-27B) | 52 | 98.6% | 3.9% | 4.6% | was 97.0% / max 3.4%; least-squares objective (the regret refinement lands on a 5.5% clear miss on the synthetic grpo-g8 CP4 cell, whose best layout changed with the planner) |

The two-stage re-ranker is retired on the shipped tables. Its two drivers
were the remote wave count and the maximum rank load of the CP plan a layout
produces; the recalibrated planner builds single-wave, balanced plans for
nearly every layout, so the re-ranker's gates now fail on both classes that
shipped it (Qwen3-8B: pairwise 0.82, saving 0.12% of cell time against 1.6%
planning; Qwen3-14B: no saving against the cheap selection and worse than
version 1 on one group) while the direct ten-term score ranks their CP4 cells.
Both classes admit TP1 × CP4 directly. No shipped table carries a `ReRanker`
any more; the machinery (`ReRanker`, `reranked_shapes`, `plan_structure`, the
fitter's two-stage gates) is kept for now and its removal is a follow-up. The
CP4 blind spot itself has largely closed: on Qwen3-1.7B the direct score ranks
CP4 at 100% (max 1.4%) and TP2 × CP2 at 99.6% (max 4.6%), on Qwen3-8B
TP2 × CP2 at 99.4% (max 1.5%) — informational; admitting shapes that were not
admitted before is deferred to a follow-up. The fitter's timed-production
block now ignores legacy-planner rows, and exported certificates list every
production term (zero when unfitted) so a certificate equals the shipped
dictionary.

The re-ranker is affordable because the context-parallel assignment search
was vectorized (`_evaluate_plans` in `art.megatron.context_parallel.runtime`):
it prices a whole batch of candidate moves in one numpy pass with the cost
formulas and stage simulations kept in the same arithmetic order, identical
results on all 446 real calibration candidates (the pre-rewrite evaluator is
the reference in `tests/unit/test_context_parallel_plan_evaluation.py`), and
the median search fell from 18.3 ms to 4.1 ms per packed row — also on the
production plan-build path of every CP>1 micro-batch.

Landing gates re-derived: the sealed win-cell shape still selects deep sharing
(prompt-level sharing at CP4, where it measures fastest; full sharing at CP1),
the heterogeneous control and the tiny sealed families still decline, and
selection stays deterministic. The coefficient version is part of the planner
facts and therefore of every layout cache key, so the new table invalidates
cached recipes.

Calibrated domain (review finding, generalized for the multi-class campaign):
the gate is capability- and geometry-based, never model-name-based. Extending
the domain means running the calibration cells on the new execution class and
either fitting it its own table or admitting it into an existing one, with a
certificate either way; no class is admitted because a width falls between
measured widths. The selected table's identity joins the coefficient version
in the planner facts and therefore in every layout cache key. CPU-only
planning (unit tests) uses the default dense table.

Reproducibility (review finding): `dev/trainer_rank_cost_calibration_certificate_<table>.json`
(one per calibrated table)
binds the shipped table to its data — per-cell candidate features, median
timings, counts, spreads and fingerprints (no tokens, no per-sample rows), the
exact fit arguments including any explicit cell exclusions, the integer table
and its hash, and the headline metrics. `tests/unit/test_planner_cost_certificate.py`
asserts the shipped table is the certified table and that the certified
metrics hold on the recorded aggregates; `--from-certificate` re-fits from it.
The runners propagate every cell failure (no masked exit codes) and the
fitter refuses to fit incomplete evidence unless the gaps are excluded
explicitly (`--require-complete`, `--exclude-cells`). Two Ellavox CP4 cells
(groups 1 and 4) were excluded this way at first because their calibration
runs deadlocked in NCCL all-to-alls in the context-parallel group (issue
#840). Tracing every collective per rank (`dev/trainer_rank_collective_trace.py`,
`dev/trainer_rank_collective_diff.py`) showed the harness, not the runtime,
at fault: the warm-up loop stopped when the *local* rank's forward was
compile-free, and one CP rank whose local shapes still recompiled ran an
extra warm-up of the previous layout while its peers moved on, so the ranks
exchanged different layouts. Warm-up completion is now decided from the
gathered world-wide compile statuses; both cells were re-measured cleanly
and folded into the certificate as held-out cells (shipped-table regret 0%
and 2.8%; the training cells and therefore the table are unchanged).

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
admissions); the cold static estimate ignores sharding (conservative); the
all-shards-`-inf` output-head case is a documented non-goal (unreachable for
supported models without a vocabulary-wide mask). Padding is accounted for:
execution pads every checkpoint/no-grad group independently to a TP multiple,
so a plan's `packed_tokens` (and the cheap bounds, the split lower bound and
the memory profile with it) is the *physical* count, each group rounded up
(`_physical_tokens`); the unpadded aggregate would have been short by up to
`groups × (TP−1)` tokens, not merely `< TP`.

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
