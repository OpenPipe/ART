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
| `tests/unit/test_trainer_rank_split.py` | CPU | best-effort splitting contract: bounded ladder (failed rungs rejected by cheap bounds; planner runs only for the executing rung), cumulative live-graph admission, retained profile trusted only near its observed scale and max-merged once observed, caller-order reconstruction, honest refusal wording, minimum-wave splitting, deterministic partitions, one slot-ensure collective per call, independent slot-graph sentinels per subforward, `TrainerRankPartialExecutionError` on execution-time failure, `subforward_count` telemetry | pass |
| `--phase split-conversion --pressure cap` | 1x H200 | sealed cell shape (Qwen3.5-4B, 4 layers, 4 inputs) under the test-only cap: unlimited runs unsplit; cap converts (>=2 subforwards) with output parity, combined and reverse-order backward; sub-request cap refuses before execution | see evidence log |
| `tests/unit/test_trainer_rank_topology.py` | CPU | TP>1 runtimes construct; PP>1 and multi-chunk runtimes still refuse | pass |
| `--phase tp2-public --tp 2` (`dev/trainer_rank_landing_acceptance_tp2.sky.yaml`) + `--tp 1` control (1x H200) + `--phase tp-compare` (CPU) | 2x H200 k8s | Qwen3.5-4B full model, DP1×TP2×CP1, public `dp_rank_forward`, active LoRA: TP peers plan identical physical layouts; automatic shares deeper than depth-one (group lengths 9,199 vs 37,871 for 53,216 logical tokens; physical 9,200 / 37,872 after per-group TP padding); odd group lengths exercise SP padding; measured rows compile-free and plan-cache-stable; numerics gated relative to the TP1 control's cross-layout divergence with source/workload fingerprints (same-layout TP2-vs-TP1 ratios 1.06/1.05, cross-layout ratios 1.05/1.06, losses within 0.06%, unstructured differences). Also runs the CI check script at TP=2 (0.0 divergence) | pass: automatic 720 ms vs depth-one 1,921 ms (62.5% paired gain at TP2; 71.9% at TP1) |
| `--phase dp2-tp2-waves` (`dev/trainer_rank_landing_acceptance_dp2_tp2.sky.yaml`) | 4x H200 k8s | DP2×TP2 public `forward_micro_batches`: ≥2 waves, distinct DP payloads, identical wave shapes within each TP pair, every input returned once in order, forward+backward per wave, automatic vs depth-one parity, empty-DP-slot arm, no hang | see evidence log |
| `--phase cost-calibrate` (`dev/trainer_rank_cost_calibration_{cp4,2gpu}.sky.yaml`, `dev/trainer_rank_cost_calibration_local.sh`) | 1x/2x/4x H200 | every mandatory candidate layout of a cell timed through the public API (forward+backward, active LoRA, compile-free, max-rank) plus the production selection; layout features and topology/model facts to JSONL. Cells: GRPO g8/g16/g4x4 (Qwen3.5-4B GDN and Qwen3-4B attention, 2-layer and full height), three heterogeneous controls, Ellavox groups, at TP1/TP2 × CP1/CP2/CP4 | 58 cells, 3,849 within-cell pairs |
| `dev/trainer_rank_cost_fit.py` | CPU | paired within-cell deltas, non-negative least squares over the production term functions, regret-minimizing refinement; gates on whole held-out cells: pairwise ordering ≥90% on pairs separated >3%, median regret ≤2%, p95 ≤5%, none >10%, clear winners selected within 5%; `--selector-check` runs the shipped table through the real selector | final table (fit on 45 cells, evaluated on all 58; 3,849 pairs; 13 held-out cells): 98.1% pairwise, p95 regret 2.9%, max 4.2%, no clear misses; pre-registered odd-Ellavox holdout passes; the two Ellavox CP4 cells re-measured after the issue #840 harness fix are held out too (shipped-table regret 0% and 2.8%); profile narrowed to the measured envelope (H200-class SM 9.0, bf16, hidden 2,560, dense) and bound to the certificate by test; expected-cell manifest validated; TP2, CP2 and attention-model ablations pass (all-heterogeneous ablation: one CP4 cell at 9.5%); campaign-1 table on the 18 later cells within 4.2%; timed production selection on the later CP2/TP2 cells: median −0.2%, max 0.4%; hand-set score: 78.6% pairwise, max regret 67% |
| `dev/trainer_rank_cost_calibration_certificate.json` (compact: one line per cell) + `tests/unit/test_planner_cost_certificate.py` | CPU | the shipped table is the certified table (hash) and the certified metrics hold on the recorded per-cell aggregates; exact 58 cell identities match the manifest with no exclusions; the profile envelope matches the certified evidence; runners fail loudly on any cell failure and the fitter refuses incomplete evidence unless exclusions are explicit (the two Ellavox CP4 cells that hung in the campaign were a harness desynchronization, issue #840, fixed and re-measured) | pass |
| `tests/unit/test_planner_cost_profile.py` | CPU | the fitted table applies only inside the calibrated capability profile, narrowed to the measured envelope (SM 9.0 with H200-class device memory, bf16, hidden 2,560, non-MoE) and bound to the certificate by test; outside it the version-1 score is used verbatim | pass |
| `--phase split-conversion --pressure ballast` | 1x H200 | same cell under real pressure (live ballast tensors, no test hooks). Training forward: ballast sized from the measured retained fraction f (budget midway inside the (1−f)·R/2 conversion window, width reported), unsplit refused before execution, split runs under the reduced headroom with parity, combined backward with ballast live, observed forward+backward peak within both the admitted budget and the predicted peak; deeper ballast refuses before execution. `no_grad` forward (retained ≈ outputs only; the demonstrated high-value case): budget 60% of the unsplit requirement converts with parity and the observed peak within budget and prediction | see evidence log |

Protocol note (2026-09-01): planning cost is gated as a *fraction* only on the
full-height cp1 cell, matching the sealed protocol — the sealed 1.5%/4.2%
fractions came from full-model CP1 acceptance cells, while the win screen
deliberately uses a 2-layer model whose ~130 ms execution would make any
fraction meaningless. The screen gates planning absolutely instead. Every
measured sample uses fresh tokens, so these planning numbers are all
cache-miss (cold) costs; steady-state identical-content calls are a content
hash plus dictionary hit, and `forward_micro_batches` additionally pre-plans
the predicted next wave in the background during the caller's GPU time
(measured benefit is marginal — about 1–2 ms/step on a 2-wave GPU benchmark —
because sharing-aware width pricing already plans the accepted width; it is
kept as insurance for planning-heavy regimes and reported separately as
`speculative_planning_ms`).

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
`scratch/trainer_rank_final_acceptance/`. The landing's cost constants were
hand-set (the research thread confirmed it) and carried no TP terms; the
cost-model recalibration replaced them with a fitted table
(`COEFFICIENT_VERSION` 2, see the calibration rows above and the design
brief). The sealed "full-sharing arm 875.7 ms vs automatic 1,132.8 ms" gap was
the research run's online calibration wandering between layouts, not a
ranking error of the frozen score; the real misranking on that cell was
prompt-level sharing (791 ms) vs full sharing (872 ms), which the fitted
table gets right. The TP>1 refusal from the landing was lifted by the
TP-support follow-up (see the `tp2-public` and `dp2-tp2-waves` rows above).
