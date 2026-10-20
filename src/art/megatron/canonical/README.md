Canonical dense numerics (`dense-root-v1`)
=======================================

This ports the qualified batch-invariance/elastic-canonical operator policy to
ART. It changes the floating-point function relative to native execution. Keep
the policy and kernel/runtime versions with a run; never change it between a
returned forward and its backward or replay. MoE uses `native`.

The trainer selects `numerical_policy="canonical"`, disables full-layer compile
before building the model (`ART_DISABLE_MEGATRON_COMPILE=1`), and installs hooks
after ART's LoRA hooks. Component kernel compilation remains enabled. Stark's
runtime applies that compile setting from the single numerical-policy flag.
All recorded GPU cohorts set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization. Preserve that bootstrap configuration when reproducing the
qualified policy; an unset workspace configuration was not separately tested.
PyTorch2.11 documents this as a per-handle/stream workspace setting, so this
module does not claim that an unset value necessarily raises an error.

```python
from art.megatron.canonical import RootLayout, execution, install

install(language_model, numerical_policy=numerical_policy)
layout = RootLayout((len(tokens),))
with execution(layout, numerical_policy=numerical_policy):
    padded_tokens = layout.pad([tokens])
    padded_labels = layout.pad([labels], value=-100)
    # ForwardInput/decoder consumes independently padded complete roots.
    # FP32 head restarts fixed 128-row calls at each logical root.
    logprobs = forward(padded_tokens, padded_labels)
    loss = loss_function(logprobs[:len(tokens)])
    loss.backward()
```

Keep the synchronous execution scope open through backward and activation
recomputation. It serializes process-wide deterministic settings across trainer
threads and restores them even after an exception. Do not suspend an async task
inside it. ART's checkpoint wrapper captures immutable root metadata for CUDA
recomputation threads, alongside the existing dynamic LoRA slot and routes.
Fused callers may pass `install(model, row_context=slice_lora_segments)` to
rebase adapter spans for each root call. This upgrades existing wrappers; later
ordinary installation preserves that callback. The optional fused module is not
a dependency of this policy.

Hooks preserve model/Parameter identities and use ordinary operators outside
the scope. Reinstalling them is harmless.

Each root is independent, has zero dropout, starts positions at zero, and is
padded separately to 128 tokens. Supply logical lengths in physical packing
order. The policy runs dense row operators on each root, forces the packed
varlen GDN algorithm even for equal-length buckets, and uses contiguous
per-root causal FLASH SDPA with deterministic backward. Singleton operator
calls bypass split/join copies. Debug uninitialized-buffer fills are disabled;
deterministic algorithms remain enabled with errors rather than warnings.

Initial support is TP1/CP1, packed batch size one, no prefix sharing, sinks or
sliding windows. Installation rejects MoE, TP/CP greater than one and nonzero
dropout. Root-split CP needs a separate numerical qualification. Arbitrary
within-root sequence splitting is not the API's chunking promise.

`target_logprobs(hidden, weight, targets, layout=layout)` is a frozen-weight FP32
head reference: fixed 128-row calls independently restarted for each root,
ignored labels `-100`, checkpointed logits to bound backward memory. It returns
logical rows per root and does not add a head adapter. Stark's trainer head adds
its own optional FP32 unembedding LoRA and supports multiple target columns.

Bitwise gradients across request chunk boundaries additionally require:

- Clear native parameter gradients before each datum.
- Export each datum's gradient to FP32 and add in stable request/datum order.
- Start additions at the existing accumulated buffer. Adding a pre-summed
  request to it changes floating-point grouping.
- Normalize by the same original labels/weights, excluding padding. A loss
  masked to one root must use that root's unchanged normalization.

Per-root forward equality alone does not make a packed shared-parameter
backward or an arbitrary collective reduction byte reproducible. The default
Stark backend executes one datum at a time. Fused/elastic execution requires its
own integration evidence before advertising that capability.

Run `dev/canonical_numerics_probe.py` under one-GPU `torchrun` for full-decoder
repeat, forward/backward, reversal, singleton, companion replacement and return
checks. The default is the full pretrained Qwen3.8 model; `--layers 4
--initialization random` is a quick kernel check. A pinned local HF snapshot can
be supplied with `--model` and the explicit `--model-support-key qwen3_5_dense`.
Evidence includes operator/driver source hashes, software versions, outputs and
every trainable gradient tensor hash. Lane REPORT.md records measured hardware
and live-suite status; CPU and reduced-model checks alone are not full-model or
H200 qualification.
