# Phase 1 validation

All execution used `CUDA_VISIBLE_DEVICES=""` and a worktree-local `.venv`.
No cluster, GPU, service mutation, credential loading, deployment or PR.

- Before implementation: poison regression **failed**, reproducing exactly
  568361327001 effective bytes. Transcript: `poisoning-before.txt`.
- Focused CPU contracts: **259 passed**. Command:

  ```sh
  dev/memory_fixed_slope/cpu.sh python -m pytest \
    tests/unit/test_trainer_rank_fixed_slope.py \
    tests/unit/test_trainer_rank_profile_warm.py \
    tests/unit/test_trainer_rank_ep_admission_scope.py \
    tests/unit/test_trainer_rank_planning_status.py \
    tests/unit/test_trainer_rank_planner_options.py \
    tests/unit/test_moe_long_root_caller.py -q
  ```

  Includes all 120 permutations of five observations for each of 12 sweep
  buckets, 720 permutations of tiny/repeated/nonmonotone points, merge
  partitioning, threshold/trust/logical-row monotonicity, static checkpoint
  floors, legacy defaults/migration, corrupt persisted fits, MAX/MIN admission,
  refusal before execution, and the exhausted-reject EP/WORLD collective site.
  Existing distributed planning-status tests use CPU Gloo processes. A final
  run through the checked-in portable CPU wrapper also passed all **36** new
  fixed/slope tests.
- Fixture replay: **60/60 covered** in the aggregate and rank-local fit replays.
  Minimum effective/measured 1.10x; maximum scalar replay 4.191831x.
  These are post-fit CPU results; see DESIGN.md for input/floor limitations.
- Dense RL CPU planner: approximate width **32 -> 384**, 24 GiB usable, CP8,
  LoRA 8. Source measurements are transferred across shapes for a what-if;
  this is not GPU qualification or permission to do that in production.
- Caladan companion: `git apply --check` passes against 2d9a19489;
  **16 checks passed** against a temporary patched copy using real codecs.
  The Caladan worktree was not changed. See `caladan-check.json`.
- Agreement/trust/retained method ASTs and the entire microbatch planner remain
  unchanged from ee91943cd; see `agreement-parity.json`.
- Focused `ty check` passes for `_memory.py`, `_memory_model.py` and the new test.
- Required `uv run prek run --all-files`: ruff, ruff-format and uv.lock sync pass;
  repository-wide ty fails with **49 pre-existing diagnostics**. An archive of
  ee91943cd under the same environment produces the **identical 49** messages,
  locations and counts (no added or removed diagnostics). See
  `typecheck-parity.json`. Unrelated baseline type errors were not changed.
- A broader bounded-forward suite initially had 55 dependency skips and one
  native-metadata import failure (`megatron.bridge` missing). The exact failing
  test also fails on the untouched baseline in this CPU environment. This is
  not a claim that all optional native memory-floor tests executed.

The imported public-service/LoRA shims are isolated under `dev/`; none is used
by production ART. Existing source/runtime ART pins and frozen run identities
were not changed. No GPU validation is claimed.
