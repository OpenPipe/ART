GLM-5.3 Flash HF oracle fixtures

The payloads are shared at `/tmp/stark-gpu/glm-fixtures`; the small manifest and
pinned runtime requirements are versioned here. No GPU is needed to load R2.
Use the M1 interpreter at `/tmp/stark-gpu/glm-m1/venv/bin/python`.

```python
from art.megatron.glm5next.hf_oracle import load_fixture, cpu_reference_kernels

model = load_fixture("R2")  # verifies the manifest's SHA-256 records
with cpu_reference_kernels():
    logits = model(input_ids).logits
```

F4 and F8 use the native HF text layers and an untied head. The wrapper preserves
`model.language_model.*` checkpoint prefixes and calls `from_pretrained`, so HF
converts the convolution, mHC, forget-gate and expert keys. Vision and MTP have no
parameters in the wrapper. F8 uses HF disk offload while building to bound host
memory. All fixtures keep the source EOS IDs; `pad_token_id=None` prevents EOS
from becoming an embedding padding index. Never infer a loss mask from token IDs.

`glm5next_oracle.py` beside this directory builds/downloads fixtures, generates
the fixed seeded packs, captures references, and provides the full-model vLLM
client. Run it directly for CLI help; this avoids importing ART's optional runtime
dependencies. Its `vllm` subcommand accepts `--base-url`, `--model`, and `--output`;
M7 should run it separately for FP8 and BF16. It uses the local `prompts.json`,
sends each packed document independently, and verifies echoed token IDs. The
first token of each document has no prompt logprob. It requests one generated
token only to satisfy the completions API and excludes it from recorded prompt
scores. API credentials, if needed, come from `OPENAI_API_KEY`.

Each pack contains three documents of lengths `[13, total-42, 29]` with totals
128, 512, 2048, 4096, 8192, and 16384. Fresh HF calls reset both KDA state and DSA
pooling at document boundaries. Dump-local indices and positions start at zero;
`pack_offset` maps them back to the pack. Concatenate in document order and use
`cu_seqlens` from `prompts.json` for an ART packed comparison. Token 154820 occurs
deliberately at packed position 17 and is retained in the loss. R2 uses token 256.

Each document safetensors contains input IDs, positions, pack offsets, FP32 logit
logsumexp, top-64 values/IDs at every position, within-document next-token logprobs,
and full logits at positions 0, midpoint, last, and applicable boundary probes
2047, 2048, 2050, 2051, 8191, 8192. F4 also records layer input/output mHC streams,
attention/MLP inputs and outputs, and mHC post/comb/collapsed tensors. Every DSA
layer records indexer token indices; every MoE layer records router expert IDs
and weights. R2 exercises the same capture schema, with top-2 routing.

Gradient files cover each LoRA-target family on the 128-token pack: CE sums over
125 within-document next-token targets, divided by 125. Each document has an
independent forward. Base-weight gradients are accumulated in FP32 on CPU, one
family at a time; they include all routed experts (unused experts have zero
gradient). HF packs gate/up expert gradients along dimension 1 of `gate_up_proj`.
The files include loss, input IDs, boundaries, and target count. These are
base-weight CE gradients, rather than arbitrary randomly initialized LoRA factors.

The HF indexer scores learned complete four-token pools and selects at most
`2048//4=512` pools. It expands each selected pool into four token positions,
appends the visible incomplete tail (0–3 tokens), pads with -1 to width 2051, and
masks invalid queries. A pool is eligible only when its final token is visible.
At query 2047, 512 pools are visible; query 2050 has 512 pools plus three tail
tokens; query 2051 has 513 complete pools competing for 512 slots. Selection is
score-ordered; tied scores inherit `torch.topk` tie behavior, so compare sets.

The comparison CLI accepts a candidate safetensors or a Python file exposing
`run_oracle(inputs: dict[str, Tensor]) -> dict[str, Tensor]`:

```sh
python src/art/megatron/glm5next/hf_oracle.py DUMP --actual CANDIDATE --key target_logprobs
python src/art/megatron/glm5next/hf_oracle.py DUMP --module CANDIDATE.py --inputs INPUTS --key layers.3.streams
```

It reports max-absolute error, relative L2 error, cosine similarity, logprob
absolute-error mean/p99, and row-wise top-k set Jaccard agreement. Invalid -1
indices and duplicate indices are excluded from sets; two empty sets agree.
Numerical distances are report-only. Nonfinite tensors, missing keys, changed
artifact hashes, and incompatible shapes fail. The CPU tests enforce exact
checkpoint conversion and independently check document-local CE and gradients.
