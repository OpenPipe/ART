# Host checkpoint offload follow-up

The default-off host activation scope from upstream `67170e60` (cherry-pick
`82284f40`) and its split-key fix `0b4b6713` (`69d03c7d`) are reverted here.
There are no production callers. Host offload needs its own implementation and
qualification before returning; this branch retains device checkpoint storage.

The pre-forward check compares one root's boundary bytes with the total host
budget. The pack hook instead checks cumulative `stats.saved_bytes`, which is
never released. A second forward can therefore pass preflight, exhaust the
budget inside a layer on one rank, and leave a peer entering the next CP
collective. Exchanging planning exceptions cannot catch this later divergence.

Sol's CPU/Gloo repro is preserved at
`/tmp/claude-1001/art-cp-admission-review/scratch/probe_host_budget.py`.
It uses two checkpoint layers, 8 rows on rank 0 and 4 on rank 1, a budget of
163,840 bytes, two forwards retaining both graphs in one scope, and an in-layer
all-reduce. Both second-forward preflights pass; rank 0 raises
`checkpoint host activation byte budget exceeded`, while rank 1 enters the
collective and sees peer teardown (or hangs without teardown). The frozen
`scratch/head` source still reproduces this. The current source intentionally
has no `offload_saved_activations` API, so the original probe fails at import.

A replacement must collectively reserve cumulative live host bytes before any
rank executes a root, using worst-rank demand and available budget, and release
reservations only when all tensors/graphs holding that root have been released.
Alternatively, enforce a single-forward scope collectively or use a per-forward
budget with explicit lifetime release; resetting a counter while an earlier
graph remains live does not bound host memory. Pack-time limits can remain
defensive checks, but must not be the normal admission mechanism.

Required regressions include repeated forwards with unequal row ownership,
retained graphs versus completed backwards, exact budget edges, exceptions and
scope cleanup, and collective refusal before the first layer. Separate CUDA
qualification must establish saved-tensor lifetime/gradient correctness, actual
device and host peaks, bounded pinned storage, and copy synchronization. CPU
round trips alone do not establish those properties.
