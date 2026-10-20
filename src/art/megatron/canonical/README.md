Canonical dense numerics (`dense-root-v1`)
=======================================

`install(model, numerical_policy="canonical")` installs wrappers after ART's
model/LoRA hooks and returns `{module_path: "row" | "attention"}`. Reinstalling
returns the same report without rewrapping. Unsupported parameter operators,
attention layouts, MoE, TP/CP > 1 and nonzero dropout are rejected before mutation.
`numerical_policy="native"` leaves the model untouched and returns `{}`.

`RootLayout(lengths)` describes complete independent roots in physical packing
order, each padded separately to 128 rows with positions restarting at zero.
Use `layout.pad(roots)` and `layout.unpad(outputs)`; labels use padding `-100`.
Keep `execution(layout)` open through synchronous forward, backward and activation
recomputation. Require batch size one, no prefix sharing, attention sinks or
sliding windows, and disable full-layer compilation before constructing the model.
The caller owns loss masking/normalization and stable FP32 gradient accumulation
in datum order. Packed shared-parameter reductions are not generally bitwise equal.

`execution` enables deterministic algorithms and disables uninitialized-memory
fills **process-wide**, restoring both on exit (including exceptions). All other
TrainerRank, planner and checkpoint-prefetch threads inherit these flags. Use a
process dedicated to this policy, or quiesce unrelated PyTorch work for the whole
scope. Its lock coordinates only other `execution` callers; it does not isolate
arbitrary threads. Do not suspend an async task inside the scope. Native scopes
mask the root layout without modifying process flags.

Record `POLICY_VERSION` and kernel/runtime versions with a run; retain the same
policy between forward, backward and replay. The GPU regression is
`tests/integration/megatron/test_canonical_numerics.py`.
