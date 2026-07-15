# ART Multi-Node Progress Log

## Current State

- Objective: deliver production-quality single- and multi-node ART with Monarch control, live typed communication, multi-node Megatron and vLLM, efficient replica routing, and unchanged simple single-node usage. Generalized RL is not a prerequisite.
- Baseline: `base/austin/glm52_cp` at `bc39c128`; implementation branch `austin/monarch_multinode_training`.
- Completed: baseline/contracts and all research gates committed; pinned Monarch 0.2 local and real two-node actor/ProcMesh smokes passed on the intended H200 hosts without using GPUs.
- Active: four isolated implementation workstreams for runtime/data plane, trainer transport, native vLLM routing, and PP/VPP.
- Blocker/risk: no external blocker. Scope is broad, so correctness and throughput gates must prevent speculative abstractions or duplicate execution paths.
- Next: review and integrate workstream commits, run one-node parity, then provision the two-node H200 validation cluster.
- Authoritative state: tracker `project_tracking/art/monarch_multinode_training/project.md`; source/base commit `bc39c128`; tracker commit `232b02e`.

## Work Blocks

| UTC | Elapsed | Outcome | Evidence / next action |
| --- | ---: | --- | --- |
| 2026-07-15 07:46 | 0h00 | Binding 48-hour multi-node goal started; waiting for the GLM/E2E handoff signal at 30-minute intervals. | No project work performed before the signal. |
| 2026-07-15 08:17 | 0h31 | Observed `scratch/start_multinode` and created the implementation worktree from the finalized GLM snapshot. | Clean source `austin/glm52_cp` at `bc39c128`; created `base/austin/glm52_cp` and `austin/monarch_multinode_training`. Next: current-code/API audit and design freeze. |
| 2026-07-15 08:34 | 0h48 | Audited the current E2E runtime and pinned distributed APIs. Chose direct Monarch 0.2 actor-per-rank orchestration, a run-scoped typed trainer boundary, host-local shared packed-batch leases, and one MCore schedule adapter for RL/SFT/reference execution. Rejected TorchStore as an early, version-coupled dependency and rejected nested multi-node `torchrun`. | Research notes in `scratch/multinode_research/{monarch,art_runtime,contracts_data_plane,pp_vpp}.md`. Remaining gates: pinned vLLM replica/router details and exact 2x8 H200 matrix. |
| 2026-07-15 08:46 | 1h00 | Froze and committed all research gates and shared topology schemas. A real local Monarch test attached two worker loops, spawned two processes per host, and observed ranks `(global, local, world) = (0,0,4), (1,1,4), (2,0,4), (3,1,4)`. Found that Monarch 0.2 resolves a uv-venv interpreter symlink when spawning ProcMesh children; explicit child `PYTHONPATH` propagation made the smoke pass. | Commits `2bae190d`, tracker `61e2894` and `fa42db5`; probe `scratch/monarch_local_smoke.py`. Four isolated coding agents active. Require a pre-CUDA child import/build probe in production bootstrap. |
| 2026-07-15 08:52 | 1h06 | Passed a real cross-host Monarch smoke on the idle `austin-art0`/`austin-art1` 8xH200 pods: attached both worker loops, spawned one process per host, configured torch-elastic, and returned ranks 0/1 with world size 2. No GPU process was created. | Probe `scratch/monarch_remote_controller_smoke.py`. Required `enable_transport("tcp")` before every Monarch API; raw TCP liveness probes are invalid because the worker port accepts root clients. Actor health endpoints are mandatory. |
| 2026-07-15 08:59 | 1h13 | Passed a real cross-host Monarch RDMA ownership probe: one source actor exposed a 128 MiB CPU buffer, both hosts read it concurrently with exact SHA-256 equality, and the owning actor released the handle cleanly. | Probe `scratch/monarch_remote_rdma_smoke.py`. Use actor-owned `RDMABuffer` handles for one transfer per trainer host, POSIX shared memory for host-local rank fanout, and owner-actor release. Root-created or root-dropped handles are invalid in pinned Monarch 0.2. |
