#!/bin/bash

export MODEL_DIR=$(HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download Qwen/Qwen3-32B | tail -n 1)
export TORCHTUNE_DIR=$(uv run python -c "import torchtune; import os; print(os.path.dirname(torchtune.__file__))")
uv run $TORCHTUNE_DIR/_cli/tune.py run \
    --nproc-per-node 8 \
    src/art/torchtune/recipe.py \
    --config ./src/art/torchtune/config.yaml \
    tokenizer.path=$MODEL_DIR/vocab.json \
    tokenizer.merges_file=$MODEL_DIR/merges.txt \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    checkpointer.checkpoint_files="[$(ls $MODEL_DIR/*.safetensors | xargs -n1 basename | sed 's/^/"/;s/$/",/' | tr '\n' ' ' | sed 's/, $//' )]" \
    model._component_=torchtune.models.qwen3.qwen3_32b \
    "$@"
    