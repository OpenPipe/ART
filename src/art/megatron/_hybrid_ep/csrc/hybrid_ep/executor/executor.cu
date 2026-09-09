// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "executor.cuh"
#include <vector>
#include <cstdint>
#include <nvtx3/nvToolsExt.h>

Executor::Executor(int local_rank, int node_rank, std::string base_path, std::string cuda_home, std::string cccl_include_dir, std::string jit_cache_dir, std::string comm_id, bool enable_custom_allgather) : local_rank(local_rank), node_rank(node_rank), kernel_cache(node_rank, local_rank, base_path, cuda_home, cccl_include_dir, jit_cache_dir, comm_id), enable_custom_allgather(enable_custom_allgather) {}

void Executor::set_intra_node_buffers(IntraNodeDispatchBuffers *intra_node_dispatch_buffers, IntraNodeCombineBuffers *intra_node_combine_buffers) {
    this->intra_node_dispatch_buffers = intra_node_dispatch_buffers;
    this->intra_node_combine_buffers = intra_node_combine_buffers;
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
void Executor::set_inter_node_buffers(InterNodeDispatchBuffers *inter_node_dispatch_buffers, InterNodeCombineBuffers *inter_node_combine_buffers) {
    this->inter_node_dispatch_buffers = inter_node_dispatch_buffers;
    this->inter_node_combine_buffers = inter_node_combine_buffers;
}
#endif

torch::Tensor Executor::allgather_routing_map(
    CustomAllgather &allgather_obj,
    HybridEpConfigInstance config,
    torch::Tensor local_routing_map,
    py::object process_group
){
    nvtxRangePushA("allgather_routing_map in hybrid-ep");

    auto torch_distributed = py::module_::import("torch.distributed");
    auto num_of_expert = local_routing_map.size(-1);
    auto num_of_tokens_per_rank = local_routing_map.size(-2);
    auto group_size = process_group.attr("size")().cast<int>();
    assert(num_of_expert == config.num_of_experts_per_rank * config.num_of_ranks_per_node * config.num_of_nodes);

    torch::Tensor global_routing_map;
    // At inter-node case, we will use NCCL allgather
    if(config.num_of_nodes > 1 || !enable_custom_allgather) {
        global_routing_map = torch::empty(
            {num_of_tokens_per_rank * group_size, num_of_expert},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)
        );
        torch_distributed.attr("all_gather_into_tensor")(global_routing_map, local_routing_map, process_group);
    } else { // At intra-node case, we will use custom allgather
        allgather_obj.launch(local_routing_map, /*NUM_OF_SMS=*/32, at::cuda::getCurrentCUDAStream());
        global_routing_map = torch::from_blob(
            allgather_obj.get_output_buffer(), 
            {num_of_tokens_per_rank * group_size, num_of_expert},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)
        );
    }

    nvtxRangePop();  // End of allgather_routing_map nvtx range
    return global_routing_map;
}

HandleImpl Executor::metadata_preprocess_core(
    HybridEpConfigInstance config, 
    hybrid_ep::tmp_state_t *preprocessing_tmp,
    hybrid_ep::tmp_state_t *preprocessing_local_experts_tmp,
    torch::Tensor global_routing_map,
    int64_t num_of_tokens_per_rank,
    int64_t num_permuted_tokens,
    int64_t pad_multiple,
    bool enable_permute,
    bool fuse_permute_dispatch,
    bool non_blocking
) {
  nvtxRangePushA("metadata_preprocess_core in hybrid-ep");

  // TMA requires (remainder_chunk_size * num_of_ranks_per_node * 4) % 16 == 0
  const int remainder_chunk_size = num_of_tokens_per_rank % config.num_of_tokens_per_chunk_dispatch_api;
  if (remainder_chunk_size != 0) {
    const int tma_load_size = remainder_chunk_size * config.num_of_ranks_per_node * sizeof(int32_t);
    TORCH_CHECK(
        tma_load_size % 16 == 0,
        "TMA 16B alignment error: tma_load_size = remainder_chunk(", remainder_chunk_size,
        ") * ranks_per_node(", config.num_of_ranks_per_node, ") * 4 = ", tma_load_size, 
        "B, must be multiple of 16B."
    );
  }

  // padding for the routing map
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  auto stream = at::cuda::getCurrentCUDAStream();

  // Construt the output tensor of the metadata preprocessing kernel.
  HandleImpl handle;
  handle.config = config;
  handle.num_of_tokens_per_rank = num_of_tokens_per_rank;
  handle.num_permuted_tokens = num_permuted_tokens;
  handle.sparse_to_dense_map =
      torch::empty({num_of_tokens_per_rank * config.num_of_nodes,
                    config.num_of_ranks_per_node},
                   torch::dtype(torch::kInt32).device(torch::kCUDA));
  handle.rdma_to_attn_map =
      torch::empty({rdma_to_attn_map_size_per_node, config.num_of_nodes},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  handle.attn_to_rdma_map =
      torch::empty({num_of_tokens_per_rank, config.num_of_nodes - 1},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  if (non_blocking) {
    handle.num_dispatched_tokens_tensor =
        torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  } else {
    handle.num_dispatched_tokens_tensor =
        torch::empty({1}, torch::dtype(torch::kInt32).pinned_memory(true));
  }
  handle.local_expert_routing_map = torch::empty(
      {num_of_tokens_per_rank * config.num_of_ranks_per_node * config.num_of_nodes, config.num_of_experts_per_rank},
      torch::dtype(torch::kBool).device(torch::kCUDA));
  
  int32_t *dense_chunk_layout_ptr = nullptr;
  int32_t *dense_to_expert_map_ptr = nullptr;
  int32_t *tokens_per_expert_ptr = nullptr;
  handle.overflow_flag =
      torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  if(enable_permute) {
    int num_of_chunks_per_rank = (num_of_tokens_per_rank - 1) / config.num_of_tokens_per_chunk_preprocessing_api + 1;
    handle.dense_chunk_layout = 
        torch::empty({num_of_chunks_per_rank * config.num_of_ranks_per_node * config.num_of_nodes},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    handle.dense_to_expert_map = 
        torch::empty({num_of_tokens_per_rank * config.num_of_ranks_per_node * config.num_of_nodes, config.num_of_experts_per_rank},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    // Raw tokens_per_expert on GPU int32, written by scan kernel, read by fuse dispatch kernel.
    handle.tokens_per_expert = 
        torch::empty({config.num_of_experts_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    dense_chunk_layout_ptr = handle.dense_chunk_layout.data_ptr<int32_t>();
    dense_to_expert_map_ptr = handle.dense_to_expert_map.data_ptr<int32_t>();
    tokens_per_expert_ptr = handle.tokens_per_expert.data_ptr<int32_t>();
  }

  kernel_cache.run_preprocess_kernel(
      config, global_routing_map.data_ptr<bool>(), 
      preprocessing_tmp, preprocessing_local_experts_tmp,
      handle.sparse_to_dense_map.data_ptr<int32_t>(),
      handle.rdma_to_attn_map.data_ptr<bool>(), handle.attn_to_rdma_map.data_ptr<bool>(),
      handle.num_dispatched_tokens_tensor.data_ptr<int32_t>(),
      handle.local_expert_routing_map.data_ptr<bool>(), 
      dense_chunk_layout_ptr, dense_to_expert_map_ptr, tokens_per_expert_ptr, 
      handle.overflow_flag.data_ptr<int>(),
      static_cast<int>(node_rank), static_cast<int>(local_rank), 
      static_cast<int>(handle.num_permuted_tokens < 0 ? std::numeric_limits<int>::max() : handle.num_permuted_tokens),
      num_of_tokens_per_rank, enable_permute, non_blocking, stream);

  if(enable_permute) {
    // Both paths: handle.tokens_per_expert is raw int32 on GPU.
    // Produce padded_tokens_per_expert (pinned or device int64) via pad kernel.
    if (non_blocking) {
        handle.padded_tokens_per_expert = torch::empty(
            {config.num_of_experts_per_rank},
            torch::dtype(torch::kInt64).device(torch::kCUDA));
    } else {
        handle.padded_tokens_per_expert = torch::empty(
            {config.num_of_experts_per_rank},
            torch::dtype(torch::kInt64).pinned_memory(true));
    }
    pad_tokens_per_expert(
        handle.tokens_per_expert.data_ptr<int32_t>(),
        handle.padded_tokens_per_expert.data_ptr<int64_t>(),
        config.num_of_experts_per_rank, pad_multiple, stream);

    // If we want to put the tokens_per_expert/num_dispatched_tokens_tensor can be used in the host, we need to synchronize the stream.
    if (!non_blocking) {
        cudaStreamSynchronize(stream);
        if (handle.num_permuted_tokens < 0) {
            int64_t num_permuted_tokens = 0;
            const int64_t* tpe_ptr = handle.padded_tokens_per_expert.data_ptr<int64_t>();
            for (int i = 0; i < config.num_of_experts_per_rank; ++i) {
                num_permuted_tokens += tpe_ptr[i];
            }
            handle.num_permuted_tokens = num_permuted_tokens;
        }
    }
  }

  nvtxRangePop();  // End of metadata_preprocess_core nvtx range
  return handle;
}

void Executor::dispatch_preprocess(HybridEpConfigInstance config, DispatchArgs& args) {
    nvtxRangePushA("dispatch_preprocess in hybrid-ep");
    if(config.num_of_nodes > 1) {
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
        auto sizeof_token_data_type = get_token_data_type_size(config.token_data_type);
        CUDA_CHECK(cudaMemcpyAsync(inter_node_dispatch_buffers->attn_input_token, args.hidden.data_ptr(), args.hidden.numel() * sizeof_token_data_type, cudaMemcpyDeviceToDevice, args.stream));
        if(config.forward_dispatch_api) {
#ifdef USE_NIXL
            // Stage prob into the dest-major NIXL send layout
            // `[num_of_nodes][max_tokens][prob_per_token]` so the dispatch
            // N2N warp coalesces prob puts per chunk (or per run in the
            // sparse path) instead of one put per token. Same total bytes
            // as the `cudaMemcpyAsync` it replaces.
            const int prob_per_token = config.num_of_experts_per_rank * config.num_of_ranks_per_node;
            restripe_prob_for_nixl_dispatch(
                static_cast<const float*>(args.probs.data_ptr()),
                static_cast<float*>(inter_node_dispatch_buffers->attn_input_prob),
                static_cast<int>(args.num_of_tokens_per_rank),
                static_cast<int>(config.max_num_of_tokens_per_rank),
                static_cast<int>(config.num_of_nodes),
                prob_per_token,
                args.stream);
#else
            CUDA_CHECK(cudaMemcpyAsync(inter_node_dispatch_buffers->attn_input_prob, args.probs.data_ptr(), args.probs.numel() * sizeof(float), cudaMemcpyDeviceToDevice, args.stream));
#endif
        }
        if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
            CUDA_CHECK(cudaMemcpyAsync(inter_node_dispatch_buffers->attn_input_scaling_factor, args.scaling_factor.data_ptr(), args.scaling_factor.numel() * sizeof(float), cudaMemcpyDeviceToDevice, args.stream));
        }
#else
        throw std::runtime_error("Multi-node support is not enabled in this build.");
#endif
    } 
    nvtxRangePop();  // End of dispatch_preprocess nvtx range
}

template void Executor::dispatch_core<uint8_t>(HybridEpConfigInstance config, DispatchArgs& args);
template void Executor::dispatch_core<uint16_t>(HybridEpConfigInstance config, DispatchArgs& args);

template<typename DType>
void Executor::dispatch_core(HybridEpConfigInstance config, DispatchArgs& args) {
    nvtxRangePushA("dispatch_core in hybrid-ep");

    hybrid_ep::dispatch_kernel_param_t<DType> param;
    param.multinode_ctx_ptr = nullptr;
    param.multinode_aux_ptr = nullptr;
    // Setup input pointers
    if(config.num_of_nodes > 1) {
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
        param.attn_input_token = reinterpret_cast<DType*>(inter_node_dispatch_buffers->attn_input_token);
        param.attn_input_prob = reinterpret_cast<float*>(inter_node_dispatch_buffers->attn_input_prob);
        param.attn_input_token_scaling_factor = reinterpret_cast<float*>(inter_node_dispatch_buffers->attn_input_scaling_factor);
#else
        throw std::runtime_error("Multi-node support is not enabled in this build.");
#endif
    } else {
        param.attn_input_token = reinterpret_cast<DType*>(args.hidden.data_ptr());
        param.attn_input_prob = (config.forward_dispatch_api) ? 
            reinterpret_cast<float*>(args.probs.data_ptr()) : nullptr;
        param.attn_input_token_scaling_factor = (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) ? 
            reinterpret_cast<float*>(args.scaling_factor.data_ptr()) : nullptr;
    }
    
    // Setup output pointers
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
      param.expert_output_token[i] = reinterpret_cast<DType*>(
          intra_node_dispatch_buffers->expert_output_token_all_ranks[i]);
      param.expert_output_prob[i] = intra_node_dispatch_buffers->expert_output_prob_all_ranks[i];
      param.expert_output_scaling_factor[i] = 
          intra_node_dispatch_buffers->expert_output_scaling_factor_all_ranks[i];
    }
    
    // Setup local buffer pointers
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    param.rdma_inter_node_group_token = reinterpret_cast<DType*>(
        inter_node_dispatch_buffers->rdma_inter_node_group_token);
    param.rdma_inter_node_group_prob = inter_node_dispatch_buffers->rdma_inter_node_group_prob;
    param.rdma_inter_node_group_scaling_factor = 
        inter_node_dispatch_buffers->rdma_inter_node_group_scaling_factor;
    param.rdma_inter_node_group_flags = inter_node_dispatch_buffers->rdma_inter_node_group_flags;
#endif
    param.intra_node_write_completion_flags = 
        intra_node_dispatch_buffers->intra_node_write_completion_flags;
    param.rdma_to_attn_map = args.rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = args.attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = args.sparse_to_dense_map.data_ptr<int32_t>();

    if(args.fuse_permute_dispatch) {
        // Set permute output buffers — point to pre-allocated output tensors from args
        param.local_expert_output_token = reinterpret_cast<DType*>(args.local_expert_output_token.data_ptr());
        param.local_expert_output_prob = args.local_expert_output_prob.has_value() ?
            args.local_expert_output_prob.value().data_ptr<float>() : nullptr;
        param.local_expert_output_scaling_factor = args.local_expert_output_scaling_factor.has_value() ?
            args.local_expert_output_scaling_factor.value().data_ptr<float>() : nullptr;
        // Set permute related metadata & intermediate buffers
        param.dense_chunk_layout = args.dense_chunk_layout.data_ptr<int32_t>();
        param.dense_to_expert_map = args.dense_to_expert_map.data_ptr<int32_t>();
        param.num_of_local_experts_tokens = args.tokens_per_expert.data_ptr<int32_t>();
        param.expected_permute_flag_value = intra_node_dispatch_buffers->expected_permute_flag_value;
        for (int i = 0; i < config.num_of_ranks_per_node; i++) {
            param.intra_node_expert_output_chunk_flags[i] = 
                intra_node_dispatch_buffers->intra_node_expert_output_chunk_flags_all_ranks[i];
        }
    }

    // Misc
    param.local_rank = local_rank;
    param.node_rank = node_rank;
    param.num_of_tokens_per_rank = args.num_of_tokens_per_rank;
    param.expected_intra_node_flag_value = intra_node_dispatch_buffers->expected_intra_node_flag_value;
    param.intra_node_flag_parity = intra_node_dispatch_buffers->intra_node_flag_parity;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    param.expected_rdma_flag_value = inter_node_dispatch_buffers->expected_rdma_flag_value;
#ifdef USE_NIXL
    param.multinode_ctx_ptr = inter_node_dispatch_buffers->nixl_gpu_ctx;
    param.multinode_aux_ptr = nullptr;
#else
    param.multinode_ctx_ptr = reinterpret_cast<void*>(inter_node_dispatch_buffers->d_qps_gpu);
    param.multinode_aux_ptr = reinterpret_cast<void*>(inter_node_dispatch_buffers->mr_info);
#endif
#endif
    // Launch kernel
    kernel_cache.run_dispatch_kernel<DType>(config, param, args.fuse_permute_dispatch, args.non_blocking, args.stream);
    nvtxRangePop();  // End of dispatch_core nvtx range
}

void Executor::dispatch_postprocess(HybridEpConfigInstance config, DispatchArgs& args) {
    nvtxRangePushA("dispatch_postprocess in hybrid-ep");

    if(args.enable_permute && !args.fuse_permute_dispatch) {
        // Standalone permute: scan already produced dense layout.
        assert(args.num_permuted_tokens >= 0);
        assert(args.dense_chunk_layout.defined());
        assert(args.dense_to_expert_map.defined());
        assert(args.tokens_per_expert.defined());
    
        // Prepare the arguments for the permute kernel
        PermuteArgs permute_args;
        permute_args.tokens_ptr = reinterpret_cast<void*>(intra_node_dispatch_buffers->expert_output_token);
        permute_args.probs_ptr = reinterpret_cast<float*>(intra_node_dispatch_buffers->expert_output_prob);
        permute_args.scaling_factor_ptr = reinterpret_cast<float*>(intra_node_dispatch_buffers->expert_output_scaling_factor);
        permute_args.dense_chunk_layout = args.dense_chunk_layout;
        permute_args.dense_to_expert_map = args.dense_to_expert_map;
        permute_args.num_of_local_experts_tokens = args.tokens_per_expert;
        // Output buffers: point to pre-allocated tensors from args
        permute_args.output_tokens_ptr = args.local_expert_output_token.data_ptr();
        permute_args.output_probs_ptr = args.local_expert_output_prob.has_value() ?
            args.local_expert_output_prob.value().data_ptr<float>() : nullptr;
        permute_args.output_scaling_factor_ptr = args.local_expert_output_scaling_factor.has_value() ?
            args.local_expert_output_scaling_factor.value().data_ptr<float>() : nullptr;
        permute_args.hidden_size = config.hidden_dim;
        permute_args.scales_per_token = config.hidden_dim / 128;
        permute_args.num_permuted_token = args.num_permuted_tokens;
        permute_args.num_ranks_per_node = config.num_of_ranks_per_node;
        permute_args.num_of_local_experts = config.num_of_experts_per_rank;
        permute_args.pad_multiple = args.pad_multiple;
        permute_args.local_rank = local_rank;
        permute_args.use_fp8 = config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8;
        permute_args.with_probs = config.forward_dispatch_api;
        permute_args.stream = args.stream;
        permute_args.num_of_blocks_permute = config.num_of_blocks_permute;
        
        if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT16) {
            permute_launcher<uint16_t, float, float>(permute_args);
        } else if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
            permute_launcher<uint8_t, float, float>(permute_args);
        }else {
            throw std::runtime_error("Unsupported token data type: " + type_to_string(config.token_data_type));
        }
    }else if(args.enable_permute && args.fuse_permute_dispatch) {
        // Fused permute: data already written to args output tensors by fused dispatch kernel. No-op.
    }else if(!args.enable_permute) {
        // No permute: allocate output tensors and D2D copy from NVLink buffer
        int num_dispatched_tokens = args.num_dispatched_tokens_tensor.value().item<int>();
        size_t sizeof_token_data_type = get_token_data_type_size(intra_node_dispatch_buffers->data_type);
        args.local_expert_output_token = torch::empty(
            {num_dispatched_tokens, config.hidden_dim}, 
            torch::dtype(args.hidden.dtype()).device(torch::kCUDA));
        auto res_sz = static_cast<size_t>(num_dispatched_tokens) * config.hidden_dim * sizeof_token_data_type;
        CUDA_CHECK(cudaMemcpyAsync(args.local_expert_output_token.data_ptr(), intra_node_dispatch_buffers->expert_output_token, res_sz, cudaMemcpyDeviceToDevice, args.stream));

        if(config.forward_dispatch_api) {
            args.local_expert_output_prob = torch::empty({num_dispatched_tokens,
                config.num_of_experts_per_rank * config.num_of_ranks_per_node},
                torch::dtype(torch::kFloat32).device(torch::kCUDA));
            auto probs_sz = static_cast<size_t>(num_dispatched_tokens) * config.num_of_experts_per_rank * config.num_of_ranks_per_node * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(args.local_expert_output_prob.value().data_ptr<float>(),
                intra_node_dispatch_buffers->expert_output_prob,
                probs_sz, cudaMemcpyDeviceToDevice, args.stream));
        }

        if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
            args.local_expert_output_scaling_factor = torch::empty({
                    num_dispatched_tokens, 
                    config.hidden_dim / 128}, 
                    torch::dtype(torch::kFloat32).device(torch::kCUDA));
            auto scaling_factor_sz = static_cast<size_t>(num_dispatched_tokens) * config.hidden_dim / 128 * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(args.local_expert_output_scaling_factor.value().data_ptr<float>(),
                intra_node_dispatch_buffers->expert_output_scaling_factor,
                scaling_factor_sz, cudaMemcpyDeviceToDevice, args.stream));
        }
    }

    nvtxRangePop();  // End of dispatch_postprocess nvtx range
}

void Executor::combine_preprocess(HybridEpConfigInstance config, CombineArgs& args) {
    nvtxRangePushA("combine_preprocess in hybrid-ep");

    if(args.enable_unpermute && !args.fuse_unpermute_combine) {
        assert(args.dense_chunk_layout.defined());
        assert(args.dense_to_expert_map.defined());
    
        UnpermuteArgs unpermute_args;
        unpermute_args.permuted_tokens = args.hidden;
        unpermute_args.permuted_probs = args.probs;
        unpermute_args.tokens_ptr = reinterpret_cast<uint16_t*>(intra_node_combine_buffers->expert_input_token);
        unpermute_args.probs_ptr = reinterpret_cast<float*>(intra_node_combine_buffers->expert_input_prob);
        unpermute_args.dense_chunk_layout = args.dense_chunk_layout;
        unpermute_args.dense_to_expert_map = args.dense_to_expert_map;
        unpermute_args.num_of_local_experts = config.num_of_experts_per_rank;
        unpermute_args.hidden_size = config.hidden_dim;
        unpermute_args.local_rank = local_rank;
        unpermute_args.num_ranks_per_node = config.num_of_ranks_per_node;
        unpermute_args.with_probs = config.backward_combine_api;
        unpermute_args.stream = args.stream;
        unpermute_args.num_of_blocks_unpermute = config.num_of_blocks_unpermute;
        
        unpermute_launcher<uint16_t, float>(unpermute_args);
    
    } else if(args.fuse_unpermute_combine) {
        // Fused unpermute: data already written to args input tensors by fused combine kernel. No-op.
    } else if(!args.enable_unpermute) {
        // Copy the input tensor to the input buffer
        auto input_sz = args.hidden.numel() * sizeof(uint16_t);
        CUDA_CHECK(
            cudaMemcpyAsync(intra_node_combine_buffers->expert_input_token,
                            reinterpret_cast<uint16_t *>(args.hidden.data_ptr()), input_sz,
                            cudaMemcpyDeviceToDevice, args.stream));
        if (config.backward_combine_api) {
            auto probs_sz = args.probs.numel() * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(intra_node_combine_buffers->expert_input_prob,
                                       reinterpret_cast<float*>(args.probs.data_ptr()), probs_sz,
                                       cudaMemcpyDeviceToDevice, args.stream));
        }
    }

    nvtxRangePop();  // End of combine_preprocess nvtx range
}

void Executor::combine_core(HybridEpConfigInstance config, CombineArgs& args) {
    nvtxRangePushA("combine_core in hybrid-ep");
    hybrid_ep::combine_kernel_param_t param;
    param.multinode_ctx_ptr = nullptr;
    param.multinode_aux_ptr = nullptr;

    // Setup input pointers
    if(args.fuse_unpermute_combine) {
        param.local_expert_input_token = reinterpret_cast<uint16_t*>(args.hidden.data_ptr());
        param.local_expert_input_prob = (config.backward_combine_api) ? 
            reinterpret_cast<float*>(args.probs.data_ptr()) : nullptr;
    } 
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
        param.expert_input_token[i] =
            intra_node_combine_buffers->expert_input_token_all_ranks[i];
        param.expert_input_prob[i] =
            intra_node_combine_buffers->expert_input_prob_all_ranks[i];
    }

    // Setup output pointers
    param.attn_output_token = reinterpret_cast<uint16_t*>(args.combined_tokens);
    param.attn_output_prob = (config.backward_combine_api) ? reinterpret_cast<float*>(args.combined_probs) : nullptr;

    // Setup local buffer pointers
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    param.rdma_intra_node_red_token =
        inter_node_combine_buffers->rdma_intra_node_red_token;
    param.rdma_intra_node_red_prob = inter_node_combine_buffers->rdma_intra_node_red_prob;
    param.rdma_inter_node_group_token =
        inter_node_combine_buffers->rdma_inter_node_group_token;
    param.rdma_inter_node_group_prob =
        inter_node_combine_buffers->rdma_inter_node_group_prob;
    param.rdma_inter_node_group_flags =
        inter_node_combine_buffers->rdma_inter_node_group_flags;
#endif
    param.intra_node_write_completion_flags =
        intra_node_combine_buffers->intra_node_write_completion_flags;
    param.rdma_to_attn_map = args.rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = args.attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = args.sparse_to_dense_map.data_ptr<int32_t>();

    // Misc
    param.node_rank = this->node_rank;
    param.local_rank = this->local_rank;
    param.num_of_tokens_per_rank = args.num_of_tokens_per_rank;
    param.expected_intra_node_flag_value =
        intra_node_combine_buffers->expected_intra_node_flag_value;
    param.intra_node_flag_parity = intra_node_combine_buffers->intra_node_flag_parity;
    if(args.fuse_unpermute_combine) {
        // Set permute related metadata & intermediate buffers
        param.dense_chunk_layout = args.dense_chunk_layout.data_ptr<int32_t>();
        param.dense_to_expert_map = args.dense_to_expert_map.data_ptr<int32_t>();
        param.expected_unpermute_flag_value = intra_node_combine_buffers->expected_unpermute_flag_value;
        for (int i = 0; i < config.num_of_ranks_per_node; i++) {
            param.intra_node_expert_input_chunk_flags[i] =
                intra_node_combine_buffers->intra_node_expert_input_chunk_flags_all_ranks[i];
        }
    }
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    param.expected_rdma_flag_value = inter_node_combine_buffers->expected_rdma_flag_value;
#ifdef USE_NIXL
    param.multinode_ctx_ptr = inter_node_combine_buffers->nixl_gpu_ctx;
    param.multinode_aux_ptr = nullptr;
#else
    param.multinode_ctx_ptr = reinterpret_cast<void*>(inter_node_combine_buffers->d_qps_gpu);
    param.multinode_aux_ptr = reinterpret_cast<void*>(inter_node_combine_buffers->mr_info);
#endif
#endif

    // Launch kernel
    kernel_cache.run_combine_kernel(config, param, args.fuse_unpermute_combine, args.non_blocking, args.stream);

    nvtxRangePop();  // End of combine_core nvtx range
}

void Executor::combine_postprocess(HybridEpConfigInstance config, CombineArgs& args) {
    nvtxRangePushA("combine_postprocess in hybrid-ep");
    // No postprocess is needed for the combine kernel now.
    nvtxRangePop();  // End of combine_postprocess nvtx range
}
