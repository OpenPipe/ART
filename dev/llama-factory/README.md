LLaMA‑Factory LoRA SFT for Qwen3‑30B (2×GPU, FP, reproducible)

This folder contains a reproducible setup to fine‑tune `Qwen/Qwen3-30B-A3B-Instruct-2507` with LoRA using LLaMA‑Factory. It supports 1‑GPU debug runs and 2‑GPU data‑parallel (torchrun) runs without quantization (full‑precision bf16). A small ONLINE dataset is wired for smoke tests.

What you get

- Isolated env under `dev/llama-factory/.venv` with pinned deps (Transformers 4.52.4, PEFT, etc.)
- Training config: `configs/qwen3_30b_lora.yaml` (template=qwen3, bf16, LoRA, ONLINE dataset)
- 2‑GPU FP run verified (both GPUs utilized)
- Artifacts in `outputs/llamafactory/<run_name>` (HF PEFT adapter, tokenizer files, checkpoints)
- Simple inference script snippet (base + adapter)

Prerequisites

- Linux + CUDA GPUs (tested on H200/Hopper). bf16 support recommended.
- Python via `uv` (https://docs.astral.sh/uv/) installed on host.
- Disk: ~40–60 GB HF cache + ~2 GB for adapter/checkpoints per short run.

Setup

```bash
cd dev/llama-factory
uv sync
. .venv/bin/activate
```

Useful env vars (optional):

```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# If hf_transfer is not installed, disable fast transfer:
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Config overview (`configs/qwen3_30b_lora.yaml`)

- `model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507`
- `template: qwen3` (Qwen3 chat template)
- LoRA: rank=8, alpha=32, dropout=0.05, targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `bf16: true`, `optim: adamw_torch`, `learning_rate: 0.0002`
- Dataset: `dataset_dir: ONLINE`, `dataset: [tatsu-lab/alpaca]` (swap with your dataset)
- Output: `output_dir: outputs/llamafactory/qwen3_30b_lora_sft_fp2g` (change per run to avoid auto‑resume)
- Quantization lines are present but commented. Leave commented for full‑precision training.

Run training

- 2 GPUs (recommended):

```bash
cd dev/llama-factory
. .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 \
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0} \
llamafactory-cli train configs/qwen3_30b_lora.yaml \
  2>&1 | tee ../../logs/llf_qwen3_30b_2g_fp16_fresh.log
```

- 1 GPU (debug):

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/qwen3_30b_lora.yaml
```

Notes

- Auto‑resume: LLaMA‑Factory/Transformers will resume if `output_dir` already contains checkpoints. To force a fresh run, set a new `output_dir` in the YAML.
- GPU utilization: verify two trainer ranks are running and memory is allocated on both GPUs:
  - `ps -ef | grep -E "torchrun|llamafactory/launcher.py"`
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`

Outputs and artifacts

- Example fresh run dir: `outputs/llamafactory/qwen3_30b_lora_sft_fp2g/`
  - `adapter_model.safetensors`, `adapter_config.json` (HF PEFT adapter)
  - `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`
  - `checkpoint-*` subfolders (if `save_steps` is set)

Inference sanity check (base + adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-30B-A3B-Instruct-2507"
adapter = "dev/llama-factory/outputs/llamafactory/qwen3_30b_lora_sft_fp2g"

model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
model.eval()

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
prompt = "You are a helpful assistant.\n\nUser: Tell me a haiku about GPUs.\nAssistant:"
inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, temperature=0.7, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

Switching datasets

- To use ONLINE HF datasets, set `dataset_dir: ONLINE` and replace the list under `dataset:` with your dataset name(s).
- For local JSON/JSONL/Parquet, point `dataset_dir` to your data folder and set `dataset:` accordingly. See LLaMA‑Factory docs for schema/columns.

Qwen3-235B (8×H200, ZeRO-3 sharded)

- Use `configs/qwen3_235b_lora_zero3.yaml` for LoRA SFT on `Qwen/Qwen3-235B-A22B-Instruct-2507`.
- DeepSpeed Stage-3 config lives at `configs/deepspeed_zero3_235b.json`; model shards across all 8 GPUs instead of replicating.
- Example launch (adjust dataset + logging paths):
  ```bash
  cd dev/llama-factory
  . .venv/bin/activate  # or your env
  mkdir -p ../../logs \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --standalone --master_port=29500 \
    $(pwd)/.venv/bin/llamafactory-cli train configs/qwen3_235b_lora_zero3.yaml \
    2>&1 | tee ../../logs/llf_qwen3_235b_8g_zero3.log
  ```
- The config keeps LoRA targets to attention/router weights to avoid instantiating adapters for every MoE expert.
  Increase `max_steps`, `max_samples`, and swap in your dataset before real runs.
- Expect peak GPU memory ~70–80 GiB per H200 for bf16 + ZeRO-3; disable CPU offload or tune JSON buckets if you see stalls.
- Ensure `nvcc` is available (or set `CUDA_HOME` accordingly) so DeepSpeed can load its prebuilt CUDA ops before launch.

Quantization (optional)

- The YAML contains commented QLoRA lines (`quantization_method: bnb`, etc.). To enable 4‑bit QLoRA:
  - Uncomment the quantization block.
  - Consider using `optim: adamw_8bit` in YAML.
  - Keep `learning_rate` explicit decimal (e.g., `0.0002`) to avoid LR parsing issues with some optimizers.

Troubleshooting

- Fast‑transfer error: if you see `HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer not available`, either install `hf_transfer` or set `HF_HUB_ENABLE_HF_TRANSFER=0`.
- Unsupported keys: remove `evaluation_strategy` from YAML (not used by this CLI path).
- Wrong LoRA targets: use explicit Qwen3 modules (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`).
- Resume unexpectedly: change `output_dir` in YAML for a fresh run.

References

- LLaMA‑Factory docs: https://github.com/hiyouga/LLaMA-Factory
- Model card: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507


