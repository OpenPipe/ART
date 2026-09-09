// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>

#include "utils.cuh"
#include "hybrid_ep_backend.cuh"
#include "jit/compiler.cuh"
#include "extension/permute.cuh"
#include "extension/allgather.cuh"
#include "buffer/intranode.cuh"
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#include "buffer/internode.cuh"
#endif

struct HandleImpl {
    // Handle for dispatch
    torch::Tensor sparse_to_dense_map;
    torch::Tensor rdma_to_attn_map;
    torch::Tensor attn_to_rdma_map;
    torch::Tensor num_dispatched_tokens_tensor;
    torch::Tensor local_expert_routing_map;
    int64_t num_of_tokens_per_rank = -1;
    HybridEpConfigInstance config;

    // Dense-layout metadata for permute/unpermute.
    torch::Tensor tokens_per_expert; 
    torch::Tensor padded_tokens_per_expert; 
    torch::Tensor overflow_flag;
    int64_t num_permuted_tokens = -1;
    torch::Tensor dense_chunk_layout;
    torch::Tensor dense_to_expert_map;
};

class Executor {
public:
    Executor(int local_rank, int node_rank, std::string base_path, std::string cuda_home, std::string cccl_include_dir, std::string jit_cache_dir, std::string comm_id, bool enable_custom_allgather);

    struct DispatchArgs {
        // Input tensors
        torch::Tensor hidden;
        torch::Tensor probs;
        torch::Tensor scaling_factor;
        // Output of Metadata Preprocessing
        torch::Tensor sparse_to_dense_map;
        torch::Tensor rdma_to_attn_map;
        torch::Tensor attn_to_rdma_map;
        c10::optional<torch::Tensor> num_dispatched_tokens_tensor;

        // Output of permute
        torch::Tensor local_expert_output_token;
        c10::optional<torch::Tensor> local_expert_output_prob;
        c10::optional<torch::Tensor> local_expert_output_scaling_factor;
        // Used in the fused permute-dispatch
        torch::Tensor dense_chunk_layout;           
        torch::Tensor dense_to_expert_map; 
        torch::Tensor tokens_per_expert; 

        int64_t num_permuted_tokens = -1;
        // Misc
        int pad_multiple;  // Used in the padding case of permute
        bool enable_permute = false;
        bool fuse_permute_dispatch = false;
        bool non_blocking = false;  // If enable this, the produced num_dispatched_tokens will be put
                                        // on the CPU pinned memory, and the tokens_per_expert will be put
                                        // on the CPU, which may reduce the times of the sync
        int64_t num_of_tokens_per_rank;  // Dynamic sequence length
        cudaStream_t stream;
    };

    struct CombineArgs {
        // Input tensors
        torch::Tensor hidden;
        torch::Tensor probs;
        // Combine output tensors
        uint16_t *combined_tokens;
        float *combined_probs;
        // Output of Metadata Preprocessing
        torch::Tensor sparse_to_dense_map;
        torch::Tensor rdma_to_attn_map;
        torch::Tensor attn_to_rdma_map;
        // Dense-layout metadata used by standalone and fused unpermute.
        torch::Tensor dense_chunk_layout;  
        torch::Tensor dense_to_expert_map;          
        torch::Tensor tokens_per_expert;            
        
        // Misc
        bool enable_unpermute = false;
        bool fuse_unpermute_combine = false;
        bool non_blocking = false; // If enable this, the HYBRID_EP_BUILD_TOKEN_DROP_ENABLE will be enabled on the fused combine-unpermute kernel.
        int64_t num_of_tokens_per_rank;  // Dynamic sequence length
        cudaStream_t stream;
    };

    torch::Tensor allgather_routing_map(
        CustomAllgather &allgather_obj,
        HybridEpConfigInstance config,
        torch::Tensor local_routing_map,
        py::object process_group
    );

    HandleImpl metadata_preprocess_core(
        HybridEpConfigInstance config,
        hybrid_ep::tmp_state_t *preprocessing_tmp,
        hybrid_ep::tmp_state_t *preprocessing_local_experts_tmp,
        torch::Tensor global_routing_map,
        int64_t num_of_tokens_per_rank,
        int64_t num_permuted_tokens,
        int64_t pad_multiple,
        bool enable_permute,
        bool fuse_unpermute_combine,
        bool non_blocking
    );

    void dispatch_preprocess(
        HybridEpConfigInstance config, DispatchArgs& args);
    template<typename DType> 
    void dispatch_core(
        HybridEpConfigInstance config, DispatchArgs& args);
    void dispatch_postprocess(
        HybridEpConfigInstance config, DispatchArgs& args); 

    void combine_preprocess(
        HybridEpConfigInstance config, CombineArgs& args);
    void combine_core(
        HybridEpConfigInstance config, CombineArgs& args);
    void combine_postprocess(
        HybridEpConfigInstance config, CombineArgs& args); 

    void set_intra_node_buffers(IntraNodeDispatchBuffers *intra_node_dispatch_buffers, IntraNodeCombineBuffers *intra_node_combine_buffers);
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    void set_inter_node_buffers(InterNodeDispatchBuffers *inter_node_dispatch_buffers, InterNodeCombineBuffers *inter_node_combine_buffers);
#endif

private:
    KernelCache kernel_cache;
    int local_rank;
    int node_rank;
    bool enable_custom_allgather;

    // Buffers for intra-node communication
    IntraNodeDispatchBuffers *intra_node_dispatch_buffers = nullptr;
    IntraNodeCombineBuffers *intra_node_combine_buffers = nullptr;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    // Buffers for inter-node communication
    InterNodeDispatchBuffers *inter_node_dispatch_buffers = nullptr;
    InterNodeCombineBuffers *inter_node_combine_buffers = nullptr;
#endif
};
