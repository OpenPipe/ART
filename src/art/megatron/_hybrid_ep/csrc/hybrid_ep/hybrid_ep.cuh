// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"
#include "hybrid_ep_backend.cuh"
#include "allocator/allocator.cuh"
#include "utils.cuh"
#include "executor/executor.cuh"
#include "extension/allgather.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>

#include "buffer/intranode.cuh"
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#include "buffer/internode.cuh"
#endif

class HybridEPBuffer {
public:
  HybridEPBuffer(pybind11::object process_group, BufferConfig config, int local_rank, int node_rank, int group_size, std::string base_path, std::string cuda_home, std::string cccl_include_dir, std::string jit_cache_dir, bool use_shared_buffer, bool enable_custom_allgather);
  ~HybridEPBuffer();
  bool update_buffer(HybridEpConfigInstance config); // True means the buffer is reallocated.

  HandleImpl metadata_preprocessing(HybridEpConfigInstance config, 
    torch::Tensor local_routing_map, 
    int64_t num_of_tokens_per_rank, 
    c10::optional<int64_t> num_permuted_tokens, 
    c10::optional<int64_t> pad_multiple, 
    bool enable_permute, 
    bool fuse_permute_dispatch,
    bool non_blocking
  );

  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
  dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
           c10::optional<torch::Tensor> scaling_factor,
           HandleImpl handle,
           bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          HandleImpl handle,
          bool with_probs);
  
  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
  dispatch_with_permute(
            torch::Tensor hidden, 
            c10::optional<torch::Tensor> probs,
            c10::optional<torch::Tensor> scaling_factor,
            HandleImpl handle,
            c10::optional<int64_t> pad_multiple,
            bool fuse_permute_dispatch,
            bool non_blocking,
            bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine_with_unpermute(
          torch::Tensor hidden, 
          c10::optional<torch::Tensor> probs,
          HandleImpl handle,
          c10::optional<int64_t> pad_multiple,
          bool fuse_unpermute_combine,
          bool with_probs);       

private:
  ExtendedMemoryAllocator remote_allocator;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifdef USE_NIXL
  NIXLCoordinator internode_coordinator;
#else
  RDMACoordinator internode_coordinator;
#endif
#endif
  NVLCoordinator nvl_coordinator;
  BufferConfig buffer_config;
  Executor executor;
  pybind11::object process_group;
  CustomAllgather allgather_obj;

  void allocate_buffer();
  void release_buffer();
};
