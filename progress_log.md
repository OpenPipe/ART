# ART Multi-Node Progress Log

## Current State

- Objective: deliver production-quality single- and multi-node ART with Monarch control, live typed communication, multi-node Megatron and vLLM, efficient replica routing, and unchanged simple single-node usage. Generalized RL is not a prerequisite.
- Baseline: `base/austin/glm52_cp` at `bc39c128`; implementation branch `austin/monarch_multinode_training`.
- Completed: start signal observed; source worktree verified clean; append-only base snapshot and matching implementation worktree created; current ART, Monarch, TorchForge/TorchStore, and Megatron PP/VPP audits completed.
- Active: freeze typed runtime/data-plane contracts and finish the vLLM plus two-node validation audits before parallel implementation.
- Blocker/risk: no external blocker. Scope is broad, so correctness and throughput gates must prevent speculative abstractions or duplicate execution paths.
- Next: commit the design freeze, then implement the typed runtime/job transport, replica runtime, and common MCore schedule adapter in isolated workstreams.
- Authoritative state: tracker `project_tracking/art/monarch_multinode_training/project.md`; source/base commit `bc39c128`; tracker commit `232b02e`.

## Work Blocks

| UTC | Elapsed | Outcome | Evidence / next action |
| --- | ---: | --- | --- |
| 2026-07-15 07:46 | 0h00 | Binding 48-hour multi-node goal started; waiting for the GLM/E2E handoff signal at 30-minute intervals. | No project work performed before the signal. |
| 2026-07-15 08:17 | 0h31 | Observed `scratch/start_multinode` and created the implementation worktree from the finalized GLM snapshot. | Clean source `austin/glm52_cp` at `bc39c128`; created `base/austin/glm52_cp` and `austin/monarch_multinode_training`. Next: current-code/API audit and design freeze. |
| 2026-07-15 08:34 | 0h48 | Audited the current E2E runtime and pinned distributed APIs. Chose direct Monarch 0.2 actor-per-rank orchestration, a run-scoped typed trainer boundary, host-local shared packed-batch leases, and one MCore schedule adapter for RL/SFT/reference execution. Rejected TorchStore as an early, version-coupled dependency and rejected nested multi-node `torchrun`. | Research notes in `scratch/multinode_research/{monarch,art_runtime,contracts_data_plane,pp_vpp}.md`. Remaining gates: pinned vLLM replica/router details and exact 2x8 H200 matrix. |
