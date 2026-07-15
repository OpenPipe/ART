// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved
#include "intranode.cuh"

NVLCoordinator::~NVLCoordinator() {
    destroy();
}

void NVLCoordinator::destroy() {
    auto free_buffer = [this](auto *&ptr, bool remote_memory) {
        if (ptr == nullptr) return;
        if (remote_memory) {
            // If the memory can be accessed by remote devices, free it from remote allocator.
            remote_allocator->free(reinterpret_cast<void*>(ptr));
        } else {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
        }
        ptr = nullptr;
    };
      
    // Clean up preprocessing buffer
    free_buffer(this->preprocessing_tmp, false);
    free_buffer(this->preprocessing_local_experts_tmp, false);

    // Clean up dispatch buffers
    if (!use_shared_buffer) {
        free_buffer(dispatch_buffers.expert_output_token, true);
        free_buffer(dispatch_buffers.expert_output_prob, true);
    }
    free_buffer(dispatch_buffers.expert_output_scaling_factor, true);
    free_buffer(dispatch_buffers.expected_intra_node_flag_value, false);
    free_buffer(dispatch_buffers.intra_node_flag_parity, false);
    if (local_rank == 0) {
        free_buffer(dispatch_buffers.intra_node_write_completion_flags, true);
    }else{
        remote_allocator->close_handle(dispatch_buffers.intra_node_write_completion_flags);
        dispatch_buffers.intra_node_write_completion_flags = nullptr;
    }
    if (dispatch_buffers.expert_output_token_all_ranks != nullptr) {
        for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
            if (i != local_rank) {
                remote_allocator->close_handle(dispatch_buffers.expert_output_token_all_ranks[i]);
                remote_allocator->close_handle(dispatch_buffers.expert_output_prob_all_ranks[i]);
                remote_allocator->close_handle(dispatch_buffers.expert_output_scaling_factor_all_ranks[i]);
            }
        }
        delete[] dispatch_buffers.expert_output_token_all_ranks;
        delete[] dispatch_buffers.expert_output_prob_all_ranks;
        delete[] dispatch_buffers.expert_output_scaling_factor_all_ranks;
        dispatch_buffers.expert_output_token_all_ranks = nullptr;
        dispatch_buffers.expert_output_prob_all_ranks = nullptr;
        dispatch_buffers.expert_output_scaling_factor_all_ranks = nullptr;
    }
    // Clean up fused permute-dispatch buffers
    free_buffer(dispatch_buffers.expected_permute_flag_value, false);
    if (dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks != nullptr) {
        for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
            if (i == local_rank) {
                remote_allocator->free(dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks[i]);
            } else {
                remote_allocator->close_handle(dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks[i]);
            }
        }
        delete[] dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks;
        dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks = nullptr;
        dispatch_buffers.intra_node_expert_output_chunk_flags = nullptr;
    }

    // Clean up combine buffers
    free_buffer(combine_buffers.expert_input_token, true);
    free_buffer(combine_buffers.expert_input_prob, true);
    free_buffer(combine_buffers.expected_intra_node_flag_value, false);
    free_buffer(combine_buffers.intra_node_flag_parity, false);
    if (local_rank == 0) {
        free_buffer(combine_buffers.intra_node_write_completion_flags, true);
    }else{
        remote_allocator->close_handle(combine_buffers.intra_node_write_completion_flags);
        combine_buffers.intra_node_write_completion_flags = nullptr;
    }
    free_buffer(combine_buffers.expected_unpermute_flag_value, false);
    // Clean up fused unpermute-combine chunk flags
    if (combine_buffers.intra_node_expert_input_chunk_flags_all_ranks != nullptr) {
        for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
            if (i == local_rank) {
                remote_allocator->free(combine_buffers.intra_node_expert_input_chunk_flags_all_ranks[i]);
            } else {
                remote_allocator->close_handle(combine_buffers.intra_node_expert_input_chunk_flags_all_ranks[i]);
            }
        }
        delete[] combine_buffers.intra_node_expert_input_chunk_flags_all_ranks;
        combine_buffers.intra_node_expert_input_chunk_flags_all_ranks = nullptr;
        combine_buffers.intra_node_expert_input_chunk_flags = nullptr;
    }
    if (combine_buffers.expert_input_token_all_ranks != nullptr) {
        for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
            if (i != local_rank) {
                remote_allocator->close_handle(combine_buffers.expert_input_token_all_ranks[i]);
                remote_allocator->close_handle(combine_buffers.expert_input_prob_all_ranks[i]);
            }
        }
        delete[] combine_buffers.expert_input_token_all_ranks;
        delete[] combine_buffers.expert_input_prob_all_ranks;
        combine_buffers.expert_input_token_all_ranks = nullptr;
        combine_buffers.expert_input_prob_all_ranks = nullptr;
    }
}

void NVLCoordinator::init(
    pybind11::object process_group, 
    int node_rank, 
    int local_rank, 
    int group_size, 
    bool use_shared_buffer, 
    BufferConfig config,
    ExtendedMemoryAllocator *remote_allocator
) {
    this->buffer_config = config;
    this->node_rank = node_rank;
    this->local_rank = local_rank;
    this->group_size = group_size;
    this->use_shared_buffer = use_shared_buffer;
    this->process_group = process_group;
    this->remote_allocator = remote_allocator;
    
    // Token number at the worst case, all tokens are routed to the same expert.
    this->max_num_of_tokens = buffer_config.max_num_of_tokens_per_rank *
                              buffer_config.num_of_ranks_per_node *
                              buffer_config.num_of_nodes;
    assert(this->max_num_of_tokens % 4 == 0); 
    // The expert-token buffer is vectorized in 4-token groups.
}

bool NVLCoordinator::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.hidden_dim, config.hidden_dim);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    changed |= grow_to(buf_config.num_of_blocks_preprocessing_api, config.num_of_blocks_preprocessing_api);
    if (buf_config.num_of_tokens_per_chunk_dispatch_api != config.num_of_tokens_per_chunk_dispatch_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_dispatch_api = config.num_of_tokens_per_chunk_dispatch_api;
    }
    if (buf_config.num_of_tokens_per_chunk_combine_api != config.num_of_tokens_per_chunk_combine_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_combine_api = config.num_of_tokens_per_chunk_combine_api;
    }
    int new_dispatch_chunks = (buf_config.max_num_of_tokens_per_rank - 1)
        / buf_config.num_of_tokens_per_chunk_dispatch_api + 1;
    int new_combine_chunks = (buf_config.max_num_of_tokens_per_rank - 1)
        / buf_config.num_of_tokens_per_chunk_combine_api + 1;
    changed |= grow_to(buf_config.num_of_dispatch_chunks, new_dispatch_chunks);
    changed |= grow_to(buf_config.num_of_combine_chunks, new_combine_chunks);
    if (!use_shared_buffer
        && get_token_data_type_size(buf_config.token_data_type) < get_token_data_type_size(config.token_data_type)) {
        changed = true;
        buf_config.token_data_type = config.token_data_type;
    }
    return changed;
}

void NVLCoordinator::update_config(BufferConfig config) {
    this->buffer_config = config;
    this->max_num_of_tokens = config.max_num_of_tokens_per_rank *
                              config.num_of_ranks_per_node *
                              config.num_of_nodes;
}

void NVLCoordinator::allocate_buffers() {
    allocate_preprocessing_buffers();
    // If use_shared_buffer is true, the combine buffers and dispatch buffers share the same memory. So we need to allocate the combine buffers first.
    allocate_combine_buffers();
    allocate_dispatch_buffers();
    exchange_remote_nvl_info();
}

void NVLCoordinator::allocate_preprocessing_buffers() {
    auto preprocessing_tmp_elts =
      buffer_config.num_of_blocks_preprocessing_api * buffer_config.num_of_ranks_per_node;
    auto preprocessing_local_experts_tmp_elts =
      buffer_config.num_of_blocks_preprocessing_api * buffer_config.num_of_experts_per_rank;

    CUDA_CHECK(
        cudaMalloc((void **)&this->preprocessing_tmp,
                   preprocessing_tmp_elts * sizeof(hybrid_ep::tmp_state_t)));
    CUDA_CHECK(
        cudaMalloc((void **)&this->preprocessing_local_experts_tmp,
                   preprocessing_local_experts_tmp_elts * sizeof(hybrid_ep::tmp_state_t)));
}
  
void NVLCoordinator::allocate_dispatch_buffers() {
    dispatch_buffers.data_type = buffer_config.token_data_type;
    size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);
  
    // Calculate buffer sizes
    auto expert_output_token_elts = max_num_of_tokens * buffer_config.hidden_dim;
    auto expert_output_prob_elts = max_num_of_tokens * 
                                   (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto expert_output_scaling_factor_elts = max_num_of_tokens * (buffer_config.hidden_dim / 128);
  
    // Allocate main buffers
    if (use_shared_buffer) {
      assert(combine_buffers.expert_input_token != nullptr);
      assert(combine_buffers.expert_input_prob != nullptr);
      dispatch_buffers.expert_output_token = combine_buffers.expert_input_token;
      dispatch_buffers.expert_output_prob = combine_buffers.expert_input_prob;
    } else {
      remote_allocator->allocate((void**)&dispatch_buffers.expert_output_token, expert_output_token_elts * sizeof_token_data_type);
      remote_allocator->allocate((void**)&dispatch_buffers.expert_output_prob, expert_output_prob_elts * sizeof(float));
    }
    remote_allocator->allocate((void**)&dispatch_buffers.expert_output_scaling_factor, expert_output_scaling_factor_elts * sizeof(float));
  
    // Allocate and initialize synchronization buffers
    if (local_rank == 0) {
      remote_allocator->allocate((void**)&dispatch_buffers.intra_node_write_completion_flags, 2 * sizeof(uint32_t));
      CUDA_CHECK(cudaMemset(dispatch_buffers.intra_node_write_completion_flags, 0, 2 * sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_intra_node_flag_value, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dispatch_buffers.expected_intra_node_flag_value, 0, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.intra_node_flag_parity, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dispatch_buffers.intra_node_flag_parity, 0, sizeof(uint32_t)));

    // Allocate fused permute-dispatch synchronization buffers
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_permute_flag_value, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dispatch_buffers.expected_permute_flag_value, 0, sizeof(uint32_t)));
    // Allocate local chunk flags buffer via remote allocator (accessible by other ranks)
    int64_t chunk_flags_numel = static_cast<int64_t>(buffer_config.num_of_dispatch_chunks) * buffer_config.num_of_ranks_per_node * buffer_config.num_of_nodes;
    remote_allocator->allocate((void**)&dispatch_buffers.intra_node_expert_output_chunk_flags, chunk_flags_numel * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(dispatch_buffers.intra_node_expert_output_chunk_flags, 0,
                           chunk_flags_numel * sizeof(uint32_t)));

    // Create memory handles for cross-rank buffer exchange
    MemHandle handles[5];
    remote_allocator->get_handle(&handles[0], dispatch_buffers.expert_output_token);
    remote_allocator->get_handle(&handles[1], dispatch_buffers.expert_output_prob);
    remote_allocator->get_handle(&handles[2], dispatch_buffers.expert_output_scaling_factor);
    if (local_rank == 0) {
      remote_allocator->get_handle(&handles[3], dispatch_buffers.intra_node_write_completion_flags);
    }
    remote_allocator->get_handle(&handles[4], dispatch_buffers.intra_node_expert_output_chunk_flags);
    
    // Pack handles into tensor
    dispatch_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                          torch::dtype(torch::kUInt8).device(torch::kCPU));
    memcpy(dispatch_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

    // Check possible errors
    CUDA_CHECK(cudaGetLastError());
}

void NVLCoordinator::allocate_combine_buffers() {
    // Calculate buffer sizes
    auto expert_input_token_elts = max_num_of_tokens * buffer_config.hidden_dim;
    auto expert_input_prob_elts = max_num_of_tokens *
                                  (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);

    // Allocate main buffers
    remote_allocator->allocate((void**)&combine_buffers.expert_input_token, expert_input_token_elts * sizeof(uint16_t));
    remote_allocator->allocate((void**)&combine_buffers.expert_input_prob, expert_input_prob_elts * sizeof(float));
  
    // Allocate and initialize synchronization buffers
    if (local_rank == 0) {
      remote_allocator->allocate((void**)&combine_buffers.intra_node_write_completion_flags, 2 * sizeof(uint32_t));
      CUDA_CHECK(cudaMemset(combine_buffers.intra_node_write_completion_flags, 0, 2 * sizeof(uint32_t)));
    }
    
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_intra_node_flag_value, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(combine_buffers.expected_intra_node_flag_value, 0, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.intra_node_flag_parity, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(combine_buffers.intra_node_flag_parity, 0, sizeof(uint32_t)));
    // Allocate fused unpermute-combine synchronization buffer
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_unpermute_flag_value, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(combine_buffers.expected_unpermute_flag_value, 0, sizeof(uint32_t)));
    // Allocate local chunk flags buffer via remote allocator (accessible by other ranks)
    int64_t chunk_flags_numel = static_cast<int64_t>(buffer_config.num_of_combine_chunks) * buffer_config.num_of_ranks_per_node * buffer_config.num_of_nodes;
    remote_allocator->allocate((void**)&combine_buffers.intra_node_expert_input_chunk_flags, chunk_flags_numel * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(combine_buffers.intra_node_expert_input_chunk_flags, 0,
                           chunk_flags_numel * sizeof(uint32_t)));

    // Create memory handles
    MemHandle handles[4];
    remote_allocator->get_handle(&handles[0], combine_buffers.expert_input_token);
    remote_allocator->get_handle(&handles[1], combine_buffers.expert_input_prob);
    if (local_rank == 0) {
      remote_allocator->get_handle(&handles[2], combine_buffers.intra_node_write_completion_flags);
    }
    remote_allocator->get_handle(&handles[3], combine_buffers.intra_node_expert_input_chunk_flags);
  
    // Pack handles into tensor
    combine_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                         torch::dtype(torch::kUInt8).device(torch::kCPU));
    memcpy(combine_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

    // Check possible errors
    CUDA_CHECK(cudaGetLastError());
}

void NVLCoordinator::exchange_remote_nvl_info() {
    // Use Python's torch.distributed APIs through py::object
    auto torch_distributed = py::module_::import("torch.distributed");
    
    // Move tensors to CUDA for communication
    auto dispatch_cuda = dispatch_memory_handles.cuda();
    auto combine_cuda = combine_memory_handles.cuda();
    
    // Get world size from process group
    int world_size = process_group.attr("size")().cast<int>();
    
    // Create empty tensors for allgather output
    py::list dispatch_output_list;
    py::list combine_output_list;
    
    for (int i = 0; i < world_size; i++) {
      dispatch_output_list.append(torch::empty_like(dispatch_cuda));
      combine_output_list.append(torch::empty_like(combine_cuda));
    }
    
    // Perform allgather using Python API
    torch_distributed.attr("all_gather")(dispatch_output_list, dispatch_cuda, process_group);
    torch_distributed.attr("all_gather")(combine_output_list, combine_cuda, process_group);
    
    // Convert back to C++ vectors and move to CPU
    std::vector<torch::Tensor> dispatch_cpu_tensors;
    std::vector<torch::Tensor> combine_cpu_tensors;
    
    for (int i = 0; i < world_size; i++) {
      dispatch_cpu_tensors.push_back(dispatch_output_list[i].cast<torch::Tensor>().cpu());
      combine_cpu_tensors.push_back(combine_output_list[i].cast<torch::Tensor>().cpu());
    }
    
    // Open handles from other ranks
    open_handles_from_other_ranks(dispatch_cpu_tensors, combine_cpu_tensors);
}
  
  
void NVLCoordinator::open_handles_from_other_ranks(
      std::vector<torch::Tensor> dispatch_handles,
      std::vector<torch::Tensor> combine_handles) {
    // Allocate the pointer arrays used in the dispatch kernel.
    dispatch_buffers.expert_output_token_all_ranks =
        new void*[buffer_config.num_of_ranks_per_node];
    dispatch_buffers.expert_output_prob_all_ranks =
        new float*[buffer_config.num_of_ranks_per_node];
    dispatch_buffers.expert_output_scaling_factor_all_ranks =
        new float*[buffer_config.num_of_ranks_per_node];
  
    // Global offset means the position in the multi-node case.
    auto global_offset = node_rank * buffer_config.num_of_ranks_per_node;
  
    // Open the dispatch handles for intra_node_write_completion_flags
    if (local_rank != 0) {
      MemHandle intra_node_write_completion_flags_handle;
      // Only rank 0 will allocate memory for this flag
      memcpy(&intra_node_write_completion_flags_handle,
             dispatch_handles[global_offset].data_ptr<uint8_t>() +
                 sizeof(MemHandle) * 3,
             sizeof(MemHandle));
      remote_allocator->open_handle((void**)(&dispatch_buffers.intra_node_write_completion_flags),
                             &intra_node_write_completion_flags_handle);
    }
  
    // Open the handles for expert_output and chunk_flags
    dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks =
        new uint32_t*[buffer_config.num_of_ranks_per_node];
    for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
      MemHandle expert_output_token_handle, expert_output_prob_handle,
          expert_output_scaling_factor_handle, chunk_flags_handle;
  
      // Extract the handles from the tensor.
      auto base_ptr = dispatch_handles[i + global_offset].data_ptr<uint8_t>();
      memcpy(&expert_output_token_handle, base_ptr, sizeof(MemHandle));
      memcpy(&expert_output_prob_handle, base_ptr + sizeof(MemHandle),
             sizeof(MemHandle));
      memcpy(&expert_output_scaling_factor_handle,
             base_ptr + sizeof(MemHandle) * 2,
             sizeof(MemHandle));
      memcpy(&chunk_flags_handle, base_ptr + sizeof(MemHandle) * 4,
             sizeof(MemHandle));
  
      if (i != local_rank) {
        remote_allocator->open_handle((void**)(&dispatch_buffers.expert_output_token_all_ranks[i]),
                               &expert_output_token_handle);
        remote_allocator->open_handle((void**)(&dispatch_buffers.expert_output_prob_all_ranks[i]),
                               &expert_output_prob_handle);
        remote_allocator->open_handle((void**)(&dispatch_buffers.expert_output_scaling_factor_all_ranks[i]), 
                               &expert_output_scaling_factor_handle);
        remote_allocator->open_handle(
            (void**)&dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks[i],
            &chunk_flags_handle);
      } else {
        // For local rank, use direct pointer assignment
        dispatch_buffers.expert_output_token_all_ranks[i] =
            dispatch_buffers.expert_output_token;
        dispatch_buffers.expert_output_prob_all_ranks[i] =
            dispatch_buffers.expert_output_prob;
        dispatch_buffers.expert_output_scaling_factor_all_ranks[i] =
            dispatch_buffers.expert_output_scaling_factor;
        dispatch_buffers.intra_node_expert_output_chunk_flags_all_ranks[i] =
            dispatch_buffers.intra_node_expert_output_chunk_flags;
      }
    }

    // Allocate the pointer arrays used in the combine kernel.
    combine_buffers.expert_input_token_all_ranks =
        new uint16_t*[buffer_config.num_of_ranks_per_node];
    combine_buffers.expert_input_prob_all_ranks =
        new float*[buffer_config.num_of_ranks_per_node];
    // Open the combine handles for intra_node_write_completion_flags
    if (local_rank != 0) {
      MemHandle intra_node_write_completion_flags_handle;
      // Only rank 0 will allocate memory for this flag
      memcpy(&intra_node_write_completion_flags_handle,
             combine_handles[global_offset].data_ptr<uint8_t>() +
                 sizeof(MemHandle) * 2,
             sizeof(MemHandle));
      remote_allocator->open_handle((void**)(&combine_buffers.intra_node_write_completion_flags),
                             &intra_node_write_completion_flags_handle);
    }
    // Open the handles for expert_input and chunk_flags
    combine_buffers.intra_node_expert_input_chunk_flags_all_ranks =
        new uint32_t*[buffer_config.num_of_ranks_per_node];
    for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
      MemHandle expert_input_token_handle, expert_input_prob_handle,
          chunk_flags_handle;
      auto base_ptr = combine_handles[i + global_offset].data_ptr<uint8_t>();
      // Extract the handles from the tensor.
      memcpy(&expert_input_token_handle, base_ptr, sizeof(MemHandle));
      memcpy(&expert_input_prob_handle, base_ptr + sizeof(MemHandle),
             sizeof(MemHandle));
      memcpy(&chunk_flags_handle, base_ptr + sizeof(MemHandle) * 3,
             sizeof(MemHandle));
      // Open the handles for expert_input and chunk_flags
      if (i != local_rank) {
        remote_allocator->open_handle((void**)(&combine_buffers.expert_input_token_all_ranks[i]),
                               &expert_input_token_handle);
        remote_allocator->open_handle((void**)(&combine_buffers.expert_input_prob_all_ranks[i]),
                               &expert_input_prob_handle);
        remote_allocator->open_handle(
            (void**)&combine_buffers.intra_node_expert_input_chunk_flags_all_ranks[i],
            &chunk_flags_handle);
      } else {
        // For local rank, use direct pointer assignment (more efficient, no IPC overhead)
        combine_buffers.expert_input_token_all_ranks[i] =
            combine_buffers.expert_input_token;
        combine_buffers.expert_input_prob_all_ranks[i] =
            combine_buffers.expert_input_prob;
        combine_buffers.intra_node_expert_input_chunk_flags_all_ranks[i] =
            combine_buffers.intra_node_expert_input_chunk_flags;
      }
    }
}
