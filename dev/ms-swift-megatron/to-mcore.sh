#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen3-235B-A22B-Instruct-2507-mcore \
    --test_convert_precision true
    