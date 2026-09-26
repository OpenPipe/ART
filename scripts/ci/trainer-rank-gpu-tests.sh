#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export ART_GRAPH_GPU_TEST=1
runtime_python="$(
  .venv/bin/python -c 'from art.megatron.runtime.managed import ensure_megatron_runtime; print(ensure_megatron_runtime(art_build_sha256="trainer-rank-ci").python)'
)"
test -x "${runtime_python}"

"${runtime_python}" -m pytest --tb=short \
  tests/unit/test_trainer_rank_head_recompute.py \
  tests/unit/test_trainer_rank_rng.py \
  tests/unit/test_trainer_rank_custom_tensors.py \
  tests/unit/test_trainer_rank_tensors.py \
  tests/unit/test_trainer_rank_graphs_cuda.py \
  tests/unit/test_trainer_rank_memory_policy_cuda.py \
  tests/unit/test_trainer_rank_head_memory_cuda.py \
  tests/unit/test_trainer_rank_output_memory_cuda.py \
  tests/unit/test_trainer_rank_live_heads.py \
  tests/unit/test_trainer_rank_commands.py \
  tests/unit/test_trainer_command_transport.py \
  tests/unit/test_trainer_driver_transport.py \
  tests/unit/test_trainer_rank_versions.py \
  tests/integration/megatron/lora/test_lora_versions.py \
  tests/integration/megatron/lora/test_trainer_v1_versions.py \
  tests/integration/megatron/lora/test_trainer_v1_graph_cache.py \
  tests/integration/megatron/cp_attn/test_attention_packed_vs_flattened.py \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_packed_sibling_order_matches_cp1_oracle[2]' \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_tree_chain_matches_cp1_oracle[2]' \
  'tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_packed_correctness.py::test_gdn_cp_tree_trainability_updates_parameters[2]' \
  tests/integration/megatron/gdn_shared_prefix/test_real_gdn_tp_lora.py::test_real_qwen35_gdn_tp2_gradients_match_flattened \
  tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_dynamic_lora_slots_capture_recompute_context_and_step_independently \
  tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_trainer_rank_custom_objects_train_and_become_stale_on_cuda \
  tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_trainer_rank_custom_parameter_reduction_oracle \
  'tests/integration/megatron/lora/test_dynamic_lora_slots.py::test_trainer_rank_tp_head_backward_matches_unsharded_oracle[2]'

# Bound distributed retained-backward and residency regressions independently.
timeout --signal=TERM --kill-after=30s 10m "${runtime_python}" -m pytest --tb=short \
  tests/unit/test_trainer_rank_resident_memory_cuda.py \
  tests/integration/megatron/cp_attn/test_retained_backward.py \
  tests/integration/megatron/cp_attn/test_cpu_offload_residency.py

# Keep SFT distributed state and compiler workarounds in separate test processes.
"${runtime_python}" -m pytest --tb=short \
  tests/integration/megatron/test_sft_packing.py::test_sft_packing_loss_and_gradients

"${runtime_python}" -m pytest --tb=short \
  tests/integration/megatron/test_shared_expert_stream_handoff.py::test_compiled_shared_expert_handoff

ART_MEGATRON_CONTEXT_PARALLEL_SIZE=2 \
  "${runtime_python}" -m torch.distributed.run --standalone --nproc-per-node=2 \
    dev/trainer_rank_check.py \
    --model Qwen/Qwen3-0.6B \
    --layers 1 \
    --anchors no_sharing,full_sharing \
    --slots 2

# Tensor parallelism through the public API: vocab-parallel head, sequence
# parallelism (forced on at TP>1) and TP padding of packed batches, at layouts
# no_sharing vs full_sharing with two LoRA slots.
ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE=2 \
  "${runtime_python}" -m torch.distributed.run --standalone --nproc-per-node=2 \
    dev/trainer_rank_check.py \
    --model Qwen/Qwen3-0.6B \
    --layers 1 \
    --anchors no_sharing,full_sharing \
    --slots 2
