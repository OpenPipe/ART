#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

uv run --no-sync pytest --tb=short \
  tests/integration/megatron/cp_attn/test_attention_packed_vs_flattened.py \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_packed_sibling_order_matches_cp1_oracle[2]' \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_tree_chain_matches_cp1_oracle[2]' \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_tree_trainability_updates_parameters[2]' \
  tests/integration/megatron/gdn_shared_prefix/test_real_gdn_tp_lora.py::test_real_qwen35_gdn_tp2_gradients_match_flattened \
  tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_dynamic_lora_slots_capture_recompute_context_and_step_independently \
  'tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_trainer_rank_tp_head_backward_matches_unsharded_oracle[2]'

ART_MEGATRON_CONTEXT_PARALLEL_SIZE=2 \
  uv run --no-sync torchrun --standalone --nproc-per-node=2 \
    dev/trainer_rank_check.py \
    --model Qwen/Qwen3-0.6B \
    --layers 1 \
    --depths 0,4 \
    --chunks 17,8192 \
    --slots 0
