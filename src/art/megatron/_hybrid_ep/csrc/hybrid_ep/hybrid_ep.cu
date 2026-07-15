// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#include "hybrid_ep.cuh"
#include <iostream>
#include <sstream>
#include <vector>
#include <functional>

std::string get_comm_id(pybind11::object process_group) {
  auto torch = pybind11::module_::import("torch");
  auto torch_distributed = torch.attr("distributed");

  // Get the global id of each rank in the process group
  std::vector<int> global_ranks;
  pybind11::object get_global_rank;
  if (pybind11::hasattr(torch_distributed, "get_global_rank")) {
    get_global_rank = torch_distributed.attr("get_global_rank");
  } 
  int group_size = process_group.attr("size")().cast<int>();
  global_ranks.reserve(group_size);
  for (int i = 0; i < group_size; ++i) {
    int g = get_global_rank(process_group, i).cast<int>();
    global_ranks.push_back(g);
  }

  // Concatenate the global ranks into a string
  std::ostringstream ranks_ss;
  for (size_t i = 0; i < global_ranks.size(); ++i) {
    if (i) ranks_ss << ",";
    ranks_ss << global_ranks[i];
  }

  // Hash the string to get the comm id
  auto hashed = std::hash<std::string>{}(ranks_ss.str());
  return std::to_string(hashed);
}

HybridEPBuffer::HybridEPBuffer(
  pybind11::object process_group, 
  BufferConfig config, 
  int local_rank, 
  int node_rank, 
  int group_size, 
  std::string base_path,
  std::string cuda_home,
  std::string cccl_include_dir,
  std::string jit_cache_dir,
  bool use_shared_buffer,
  bool enable_custom_allgather
) : process_group(process_group),
    buffer_config(config),
    executor(local_rank, node_rank, base_path, cuda_home, cccl_include_dir, jit_cache_dir, get_comm_id(process_group), enable_custom_allgather)
{
    buffer_config.num_of_dispatch_chunks = (buffer_config.max_num_of_tokens_per_rank - 1) / buffer_config.num_of_tokens_per_chunk_dispatch_api + 1;
    buffer_config.num_of_combine_chunks = (buffer_config.max_num_of_tokens_per_rank - 1) / buffer_config.num_of_tokens_per_chunk_combine_api + 1;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize the allgather object
    allgather_obj.init(process_group, local_rank, buffer_config, &this->remote_allocator);
    // Initialize the nvl coordinator
    nvl_coordinator.init(process_group, node_rank, local_rank, group_size, use_shared_buffer, buffer_config, &this->remote_allocator);
    // Initialize the rdma coordinator
    if(group_size > buffer_config.num_of_ranks_per_node) {
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
      internode_coordinator.init(process_group, node_rank, local_rank, buffer_config);
#else
      fprintf(stderr, "Inter-node communication is not supported. Please rebuild with HYBRID_EP_MULTINODE flag, group_size=%d, buffer_config.num_of_ranks_per_node=%d.\n", group_size, buffer_config.num_of_ranks_per_node);
      fflush(stderr);
      assert(false); // inter-node communication is not supported.
#endif
    }

    allocate_buffer();
}

HybridEPBuffer::~HybridEPBuffer() {
    release_buffer();
}

void HybridEPBuffer::release_buffer() {
  // Synchronize the device to ensure all operations are completed.
  CUDA_CHECK(cudaDeviceSynchronize());

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  if(buffer_config.num_of_nodes > 1) {
    internode_coordinator.destroy();
  }
#endif
  nvl_coordinator.destroy();
  allgather_obj.destroy();
}

void HybridEPBuffer::allocate_buffer() {
  nvl_coordinator.allocate_buffers();
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  if(buffer_config.num_of_nodes > 1) {
    internode_coordinator.allocate_buffers();
  }
#endif
  allgather_obj.allocate_buffers();

  // Set the intra-node and inter-node buffers for the executor
  executor.set_intra_node_buffers(&nvl_coordinator.dispatch_buffers, &nvl_coordinator.combine_buffers);
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  executor.set_inter_node_buffers(&internode_coordinator.dispatch_buffers, &internode_coordinator.combine_buffers);
#endif
  CUDA_CHECK(cudaDeviceSynchronize());
}

bool HybridEPBuffer::update_buffer(HybridEpConfigInstance config) {
  // If new config requires bigger buffer, we will release the old buffer and allocate a new one.
  bool need_reallocate = false;
  need_reallocate |= nvl_coordinator.grow_buffer_config(config, buffer_config);
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  need_reallocate |= internode_coordinator.grow_buffer_config(config, buffer_config);
#endif
  need_reallocate |= allgather_obj.grow_buffer_config(config, buffer_config);

  if(buffer_config.num_of_nodes > 1 && need_reallocate) {
    TORCH_WARN("Reallocating HybridEP buffers in multi-node mode is very slow; "
               "adjust buffer_config to pre-allocate sufficient capacity.");
  }

  if(need_reallocate) {
    nvl_coordinator.update_config(buffer_config);
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    internode_coordinator.update_config(buffer_config);
#endif
    allgather_obj.update_config(buffer_config);
    release_buffer();
    allocate_buffer();
  }
  return need_reallocate;
}

HandleImpl HybridEPBuffer::metadata_preprocessing(HybridEpConfigInstance config, torch::Tensor local_routing_map, int64_t num_of_tokens_per_rank, c10::optional<int64_t> num_permuted_tokens, c10::optional<int64_t> pad_multiple, bool enable_permute, bool fuse_permute_dispatch, bool non_blocking) {
  // Basic checks
  assert(local_routing_map.device().is_cuda());
  assert(local_routing_map.is_contiguous());
  if(fuse_permute_dispatch) {
    assert(enable_permute);
  }

  // Prepare the global routing map
  auto global_routing_map = executor.allgather_routing_map(
    allgather_obj, config, local_routing_map, process_group
  );

  // Run the hybrid-ep metadata preprocessing kernel
  auto handle = executor.metadata_preprocess_core(
    config, 
    nvl_coordinator.preprocessing_tmp, 
    nvl_coordinator.preprocessing_local_experts_tmp,
    global_routing_map, 
    num_of_tokens_per_rank, 
    num_permuted_tokens.has_value() ? num_permuted_tokens.value() : -1,
    pad_multiple.has_value() ? pad_multiple.value() : 0,
    enable_permute,
    fuse_permute_dispatch,
    non_blocking
  );
  return handle;
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
HybridEPBuffer::dispatch(
         torch::Tensor hidden, 
         c10::optional<torch::Tensor> probs,
         c10::optional<torch::Tensor> scaling_factor,
         HandleImpl handle,
         bool with_probs) {
  auto config = handle.config;
  // Check the input tensors
  assert(hidden.device().is_cuda());
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
  }
  if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
    assert(scaling_factor.has_value());
    assert(scaling_factor.value().device().is_cuda());
    assert(scaling_factor.value().is_contiguous());
  }
  
  // Prepare the parameters
  Executor::DispatchArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
  args.sparse_to_dense_map = handle.sparse_to_dense_map;
  args.rdma_to_attn_map = handle.rdma_to_attn_map;
  args.attn_to_rdma_map = handle.attn_to_rdma_map;
  args.num_dispatched_tokens_tensor = handle.num_dispatched_tokens_tensor;
  args.num_of_tokens_per_rank = handle.num_of_tokens_per_rank;
  args.enable_permute = false;
  args.stream = at::cuda::getCurrentCUDAStream();
  
  // Run the full dispatch operation
  config.forward_dispatch_api = with_probs;
  executor.dispatch_preprocess(config, args);
  if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
    executor.dispatch_core<uint8_t>(config, args);
  } else if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT16) {
    executor.dispatch_core<uint16_t>(config, args);
  }else {
    throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
  }
  executor.dispatch_postprocess(config, args);

  return std::make_tuple(args.local_expert_output_token, args.local_expert_output_prob, args.local_expert_output_scaling_factor);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEPBuffer::combine(
                torch::Tensor hidden, 
                c10::optional<torch::Tensor> probs,
                HandleImpl handle,
                bool with_probs) {
  auto config = handle.config;
  // Check the input tensors
  assert(c10::elementSize(hidden.scalar_type()) == 2);
  assert(hidden.device().is_cuda());
  assert(hidden.dtype() != torch::kUInt8);
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
    assert(probs.value().numel() == 0 ||
           probs.value().size(1) == config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  }

  // Construct the output tensors
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =torch::empty({handle.num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({handle.num_of_tokens_per_rank, config.num_of_experts_per_rank *  config.num_of_ranks_per_node * config.num_of_nodes}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Prepare the parameters
  Executor::CombineArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  args.combined_tokens = reinterpret_cast<uint16_t*>(combined_tokens.data_ptr());
  if(with_probs) args.combined_probs = reinterpret_cast<float*>(combined_probs.data_ptr());
  args.sparse_to_dense_map = handle.sparse_to_dense_map;
  args.rdma_to_attn_map = handle.rdma_to_attn_map;
  args.attn_to_rdma_map = handle.attn_to_rdma_map;
  args.num_of_tokens_per_rank = handle.num_of_tokens_per_rank;
  args.enable_unpermute = false;
  args.stream = at::cuda::getCurrentCUDAStream();

  // Run the full combine operation
  config.backward_combine_api = with_probs;
  executor.combine_preprocess(config, args);
  executor.combine_core(config, args);
  executor.combine_postprocess(config, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
HybridEPBuffer::dispatch_with_permute(
          torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          c10::optional<torch::Tensor> scaling_factor,
          HandleImpl handle,
          c10::optional<int64_t> pad_multiple,
          bool fuse_permute_dispatch,
          bool non_blocking,
          bool with_probs)
{
 auto config = handle.config;
 // Check the input tensors
 assert(hidden.device().is_cuda());
 assert(hidden.is_contiguous());
 if (with_probs) {
   assert(probs.has_value());
   assert(probs.value().device().is_cuda());
   assert(probs.value().is_contiguous());
   assert(probs.value().dtype() == torch::kFloat32);
 }
 if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
   assert(scaling_factor.has_value());
   assert(scaling_factor.value().device().is_cuda());
   assert(scaling_factor.value().is_contiguous());
 }

 // Prepare the parameters
 Executor::DispatchArgs args;
 args.hidden = hidden;
 if(with_probs) args.probs = probs.value();
 if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
 args.sparse_to_dense_map = handle.sparse_to_dense_map;
 args.rdma_to_attn_map = handle.rdma_to_attn_map;
 args.attn_to_rdma_map = handle.attn_to_rdma_map;
 args.num_dispatched_tokens_tensor = handle.num_dispatched_tokens_tensor;
 args.num_permuted_tokens = handle.num_permuted_tokens;
 args.pad_multiple = (pad_multiple.has_value()) ? pad_multiple.value() : 0;
 args.fuse_permute_dispatch = fuse_permute_dispatch;
 args.non_blocking = non_blocking;
 args.num_of_tokens_per_rank = handle.num_of_tokens_per_rank;
 args.enable_permute = true;
 args.stream = at::cuda::getCurrentCUDAStream();
 args.dense_chunk_layout = handle.dense_chunk_layout;
 args.dense_to_expert_map = handle.dense_to_expert_map;
 args.tokens_per_expert = handle.tokens_per_expert;
 // Pre-allocate output tensors for both fuse and standalone permute paths
 args.local_expert_output_token = 
    torch::empty({handle.num_permuted_tokens, config.hidden_dim}, torch::dtype(hidden.dtype()).device(torch::kCUDA));
 if (with_probs) {
   args.local_expert_output_prob = torch::empty({handle.num_permuted_tokens}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
 }
 if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
   args.local_expert_output_scaling_factor = torch::empty({handle.num_permuted_tokens, config.hidden_dim / 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
 }
 
 // Run the full dispatch operation
 config.forward_dispatch_api = with_probs;
 executor.dispatch_preprocess(config, args);
 if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
   executor.dispatch_core<uint8_t>(config, args);
 } else if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT16) {
   executor.dispatch_core<uint16_t>(config, args);
 }else {
   throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
 }
 executor.dispatch_postprocess(config, args);

 return std::make_tuple(args.local_expert_output_token, args.local_expert_output_prob, args.local_expert_output_scaling_factor);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEPBuffer::combine_with_unpermute(
        torch::Tensor hidden, 
        c10::optional<torch::Tensor> probs,
        HandleImpl handle,
        c10::optional<int64_t> pad_multiple,
        bool fuse_unpermute_combine,
        bool with_probs)
{
  auto config = handle.config;
  // Check the input tensors
  assert(c10::elementSize(hidden.scalar_type()) == 2);
  assert(hidden.device().is_cuda());
  assert(hidden.dtype() != torch::kUInt8);
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
  }

  // Construct the output tensors
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =torch::empty({handle.num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({handle.num_of_tokens_per_rank, config.num_of_experts_per_rank *  config.num_of_ranks_per_node * config.num_of_nodes}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Prepare the parameters
  Executor::CombineArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  args.combined_tokens = reinterpret_cast<uint16_t*>(combined_tokens.data_ptr());
  if(with_probs) args.combined_probs = reinterpret_cast<float*>(combined_probs.data_ptr());
  args.sparse_to_dense_map = handle.sparse_to_dense_map;
  args.rdma_to_attn_map = handle.rdma_to_attn_map;
  args.attn_to_rdma_map = handle.attn_to_rdma_map;
  args.num_of_tokens_per_rank = handle.num_of_tokens_per_rank;
  args.fuse_unpermute_combine = fuse_unpermute_combine;
  args.enable_unpermute = true;
  args.stream = at::cuda::getCurrentCUDAStream();
  args.dense_chunk_layout = handle.dense_chunk_layout;
  args.dense_to_expert_map = handle.dense_to_expert_map;
  args.tokens_per_expert = handle.tokens_per_expert;
  // Run the full combine operation
  config.backward_combine_api = with_probs;
  executor.combine_preprocess(config, args);
  executor.combine_core(config, args);
  executor.combine_postprocess(config, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}
