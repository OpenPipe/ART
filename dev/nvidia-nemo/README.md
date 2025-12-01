# ART NeMo Subproject

Isolated environment for running NeMo LoRA SFT on Qwen3-30B.

Setup

```bash
cd dev/nvidia-nemo
uv sync --no-python-downloads
# optional: include aligner
uv sync --extra aligner --no-python-downloads
```

Run sample SFT (small dataset)

```bash
uv run python train_lora_sft.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset data/sample_sft.jsonl \
  --output out/qwen3-30b-lora-test
```

2x H200 launcher

```bash
bash run_2x_h200.sh
```
