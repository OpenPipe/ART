# Holistic TrainerRank planner: landing acceptance criteria

Acceptance suite for the holistic TrainerRank forward planner. Written
test-first against the research thread's frozen behavior contract and sealed
evidence (final acceptance campaign, 2026-08-31/09-01), before implementation;
every gate below now passes on the landed implementation.

## Suite layout and gates

| Piece | Runs on | Gate | Landed result |
| --- | --- | --- | --- |
| `tests/acceptance/trainer_rank_planner/test_public_contract.py` | CPU | knob-free `TrainerRank(runtime)`; knob-free forward methods; simple `TrainerRankMemoryError` | pass |
| `tests/acceptance/trainer_rank_planner/test_no_hardcoded_policy.py` | CPU | no knob identifiers or literal `max_depth=` policy in production; no stale docs | pass |
| `tests/acceptance/trainer_rank_planner/test_nonuniform_selection_gate.py` | CPU | win-cell shape selects depth>1 (sealed cold witness reproduced exactly); tiny/heterogeneous workloads decline sharing; bounded search oracle-exact under adversarial scorer; deterministic | pass |
| `--phase contract` / `--phase census` | CPU | knob-free surface; all 44 real Ellavox groups plan, zero refusals | pass |
| `dev/trainer_rank_landing_acceptance_gdn_cp4.sky.yaml` | 4x H200 k8s | 2-layer sealed throughput screen: paired median gain >= 20% (sealed 47.2%); peak reduction >= 30% (sealed 55.8%); median selected depth >= 2 (sealed 3); planning p50 <= 150 ms absolute (sealed steady 82 ms) | gain 49.7%, peak 4.17 vs 9.0 GiB, depth 3, planning 36 ms |
| `dev/trainer_rank_landing_acceptance_cp1.sky.yaml` | 1x H200 | full-height model, real Ellavox stream: paired median regression <= 2% (sealed: ties); planning fraction <= 10% (sealed 1.5%/4.2%) | see evidence log |

Protocol note (2026-09-01): planning cost is gated as a *fraction* only on the
full-height cp1 cell, matching the sealed protocol — the sealed 1.5%/4.2%
fractions came from full-model CP1 acceptance cells, while the win screen
deliberately uses a 2-layer model whose ~130 ms execution would make any
fraction meaningless. The screen gates planning absolutely instead. Every
measured sample uses fresh tokens, so these planning numbers are all
cache-miss (cold) costs; steady-state identical-content calls are a content
hash plus dictionary hit, and `forward_micro_batches` additionally pre-plans
the predicted next wave in the background during the caller's GPU time.

Regression baseline: the existing suite (`uv run pytest tests/unit`,
`uv run prek run --all-files`) stays green; the trainer-rank shard runs 144
passed with the knob tests converted to knob-rejection tests. The acceptance
tests are wired into the CI Megatron shard.

## Landed interface facts

- Planner surface: `art.trainer_rank._prefix_tree_planner`
  (`build_canonical_prefix_tree`, `prefix_tree_layout_candidates`,
  `select_prefix_tree_layout`, `search_prefix_tree_layout`,
  `iter_all_prefix_tree_layouts`) plus `art.trainer_rank._planner_cost`.
- Telemetry: `TrainerRank.last_forward_telemetry()` with
  `selected_max_depth` and `planning_ms`.
- Test-only anchor forcing: `ART_TRAINER_RANK_TEST_ANCHOR` (any candidate
  label, e.g. `no_sharing`/`depth_one`/`full_sharing`), honored only when
  `ART_TRAINER_RANK_TEST_HOOKS=1`; never reachable via public arguments.
  Used by `dev/trainer_rank_check.py --anchors ...` and the measure phase.

## Data-safety note

The census corpus (`dev/_trainer_rank_ellavox_qwen35_4b_tokens.json`,
SHA-pinned in the driver) contains customer-derived token IDs and is
deliberately git-ignored. The census phase requires it locally; CI runs the
pytest acceptance suite only.

## Sealed evidence provenance

Research worktree `~/.codex/worktrees/7236/art`,
`scratch/trainer_rank_final_acceptance/`. Known limitations carried forward:
TP>1 refusal (planner admission not TP-calibrated), and the cost model
undervaluing full sharing on some GRPO cells (sealed: full-sharing arm
875.7 ms vs automatic 1,132.8 ms). Cost constants are versioned via
`COEFFICIENT_VERSION` for future recalibration.
