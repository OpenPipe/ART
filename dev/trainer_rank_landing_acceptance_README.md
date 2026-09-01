# Holistic TrainerRank planner: landing acceptance criteria

This directory contains the acceptance suite for landing the holistic
TrainerRank forward planner. It was written test-first against the research
thread's frozen behavior contract and sealed evidence (final acceptance
campaign, 2026-08-31/09-01), before any implementation landed. The intent:
implementation is complete when everything below passes, unmodified.

## Suite layout

| Piece | Runs on | Expected now | Gate |
| --- | --- | --- | --- |
| `tests/acceptance/trainer_rank_planner/test_public_contract.py` | CPU | FAIL | knob-free `TrainerRank(runtime)`; knob-free forward methods; simple `TrainerRankMemoryError` |
| `tests/acceptance/trainer_rank_planner/test_no_hardcoded_policy.py` | CPU | FAIL | no knob identifiers or literal `max_depth=` policy in production; no stale docs |
| `tests/acceptance/trainer_rank_planner/test_nonuniform_selection_gate.py` | CPU | FAIL | sealed corpus: `grpo_like`/`deep_comb`/`mixed_branch` select nonuniform depth>1; `no_sharing` stays uniform; deterministic |
| `dev/trainer_rank_landing_acceptance.py --phase contract` | CPU | exit 1 | fast contract check, run first by every other phase |
| `dev/trainer_rank_landing_acceptance.py --phase census` | CPU | exit 1 | all 44 real Ellavox groups plan with zero refusals (sealed: 88/88 feasible) |
| `dev/trainer_rank_landing_acceptance_gdn_cp4.sky.yaml` | 4x H200 (k8s) | not runnable | paired median gain >= 20% vs depth-one (sealed 47.2%, CI floor 29.9%); peak reduction >= 30% (sealed 55.8%); median selected depth >= 2 (sealed 3); planning fraction <= 10% (sealed 4.2%) |
| `dev/trainer_rank_landing_acceptance_cp1.sky.yaml` | 1x H200 (k8s) | not runnable | paired median regression <= 2% on depth-one-is-best cells (sealed: ties); planning fraction <= 10% |

Regression baseline: the existing suite (`uv run pytest tests/unit`,
`uv run prek run --all-files`) must stay green throughout; the acceptance
tests live under `tests/acceptance/` so the default suite is unaffected until
they are promoted.

## Acceptance interface the implementation must provide

1. Planner surface: `art.trainer_rank._prefix_tree_planner` exposing
   `build_canonical_prefix_tree(sequences)`,
   `prefix_tree_layout_candidates(tree)` (candidates carry `.layout` and
   `.labels`, including `uniform_depth_*` anchors), and
   `select_prefix_tree_layout(tree, *, cp_size, layers, uses_gdn,
   refinement_work_budget)`.
2. Telemetry: `TrainerRank.last_forward_telemetry()` with at least
   `selected_max_depth` and `planning_ms`.
3. Test-only anchor forcing: `ART_TRAINER_RANK_TEST_ANCHOR` in
   `{depth_one, full_sharing}` honored only when
   `ART_TRAINER_RANK_TEST_HOOKS=1`; never reachable via public arguments.

If names differ at landing, adapt the marked ADAPTATION POINT blocks (one per
file); the assertion bodies and gate thresholds are the acceptance criteria
and must not be weakened.

## Known consumers to update at landing

`dev/trainer_rank_check.py` (and siblings) construct
`TrainerRank(runtime, shared_prefix_max_depth=..., head_chunk_tokens=...)` and
will break when the knobs are removed; update them alongside the landing.

## Sealed evidence provenance

Research worktree `~/.codex/worktrees/7236/art`,
`scratch/trainer_rank_final_acceptance/` (acceptance-summary.json,
evidence-summary.md, behavior-spec.md, gpu/grpo-gdn-cp4-leader). Known
limitations carried forward, not gated here: TP2+ admission support, and the
cost model undervaluing full sharing on some GRPO cells (sealed: full-sharing
arm 875.7 ms vs automatic 1132.8 ms on the win cell).
