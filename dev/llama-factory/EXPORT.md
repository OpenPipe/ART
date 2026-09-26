LLaMA‑Factory LoRA Export and Merge

Goal: produce Hugging Face PEFT‑compatible adapters and merged full model weights suitable for vLLM.

1) Export PEFT LoRA adapter (safe)

Use LLaMA‑Factory CLI export to write a PEFT adapter folder from a training output dir:

```bash
llamafactory-cli export peft \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter outputs/llamafactory/qwen3_30b_lora_sft \
  --export_dir exports/qwen3_30b_lora_peft
```

This yields a HF‑style adapter directory usable with `peft` and `transformers`.

2) Merge LoRA into base weights (for vLLM)

If you need merged weights for inference engines that prefer full weights:

```bash
llamafactory-cli export merge \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter outputs/llamafactory/qwen3_30b_lora_sft \
  --export_dir exports/qwen3_30b_merged
```

3) Load in vLLM (example)

Point vLLM to the merged model directory:

```bash
python -m vllm.entrypoints.api_server \
  --model exports/qwen3_30b_merged \
  --tensor-parallel-size 2
```

Notes

- Ensure `HF_TOKEN` is set if the base model is gated.
- For very large models, merging requires substantial CPU RAM and disk space.

