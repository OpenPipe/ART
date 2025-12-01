#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export HF_HOME=${HF_HOME:-$PWD/.hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PWD/.hf}

# Use both GPUs on a single H200 node (adjust as needed for multi-node)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

. .venv/bin/activate
python train_lora_sft.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset data/sample_sft.jsonl \
  --output out/qwen3-30b-lora-test \
  --gpus 2 \
  --nodes 1 \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --seq-length 4096 \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05


