// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include "utils.cuh"
 
struct PermuteArgs {
  // The address of the input 
  void* tokens_ptr;
  float* probs_ptr;
  float* scaling_factor_ptr;
  torch::Tensor dense_chunk_layout;
  torch::Tensor dense_to_expert_map;
  torch::Tensor num_of_local_experts_tokens;

  // The address of the output (pre-allocated by caller)
  void* output_tokens_ptr = nullptr;
  float* output_probs_ptr = nullptr;
  float* output_scaling_factor_ptr = nullptr;

  // The shape message of the input
  int hidden_size;
  int scales_per_token; // Now is hidden_size/128
  int64_t num_permuted_token;
  int num_ranks_per_node; // Probs dimension 0 = num_ranks_per_node * num_of_local_experts
  int num_of_local_experts;
  int pad_multiple;

  // Misc
  int local_rank;
  bool use_fp8;
  bool with_probs;
  int num_of_blocks_permute;
  cudaStream_t stream;
};

struct UnpermuteArgs {
  // Input tensors
  torch::Tensor permuted_tokens;
  c10::optional<torch::Tensor> permuted_probs;
  torch::Tensor dense_chunk_layout;
  torch::Tensor dense_to_expert_map;

  // The address of the output
  uint16_t* tokens_ptr; 
  float* probs_ptr;

  // The shape message of the output
  int num_of_local_experts;
  int hidden_size;

  // Misc
  int local_rank;
  int num_ranks_per_node;
  bool with_probs;
  int num_of_blocks_unpermute;
  cudaStream_t stream;
};

 /**
  * @brief Pad each element of tokens_per_expert to the nearest multiple of pad_multiple, writing int64 result to dst.
  */
 void pad_tokens_per_expert(
     const int32_t* src,   // GPU raw counts [num_experts]
     int64_t* dst,         // pinned or device [num_experts]
     int num_experts,
     int pad_multiple,
     cudaStream_t stream);

 template <typename DType, typename ProbType, typename ScalarType>
 void permute_launcher(PermuteArgs args);
 
 template <typename DType, typename ProbType>
 void unpermute_launcher(UnpermuteArgs args);
