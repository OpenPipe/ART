// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include <any>
#include <string>
#include <unordered_map>
#include <dlfcn.h>
#include <any>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <iostream>

#include "config.cuh"
#include "hybrid_ep_backend.cuh"
#include "utils.cuh"

class NVCCCompiler{
public:
    // Init the flags required by nvcc compiler
    NVCCCompiler(std::string base_path, std::string cuda_home, std::string cccl_include_dir, std::string jit_cache_dir, std::string comm_id);

    // Generate the code for jit compile
    std::string get_metadata_preprocessing_code(HybridEpConfigInstance config);
    std::string get_dispatch_code(HybridEpConfigInstance config);
    std::string get_combine_code(HybridEpConfigInstance config);

    /**
    * @brief Build the code to a .so file in the runtime
    *
    * @param code The code to be compiled
    * @param signature The signature of the code, which is used to name the .so
    * file
    * @param local_rank The local rank of the current process
    * @param node_rank The node rank of the current process
    * @param num_of_nodes The number of nodes in the communication
    * @param enable_permute_fusion Whether to enable HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE macro (for fused permute-dispatch or fused unpermute-combine)
    * @return std::string The path of the compiled .so file
    */
    std::string get_or_build(std::string code, std::string signature, int local_rank, int node_rank, int num_of_nodes, bool enable_permute_fusion = false, bool enable_token_drop = false);

    /**
    * @brief Get the compiled function pointer from the compiled .so file
    *
    * @param library_path The path of the compiled .so file
    * @param kernel_key The key of the kernel, used to cache the compiled function pointer
    * @return std::any The function pointer
    */
    std::any get_instance(const std::string& library_path);


private:
    std::string comm_id;              // hash(all ranks in the process_group)
    std::string base_path;            // The path of the installed package
    std::string jit_dir;              // The path of the jit library
    std::string intra_node_flags;     // The flags required by nvcc compiler in the intra-node case
    std::string inter_node_flags;     // The flags required by nvcc compiler in the inter-node case
    std::string nvcc_path;            // The path of the nvcc compiler
    std::string objs;                 // The objects to be compiled, only used in the inter-node case
};

class KernelCache{
public:
    KernelCache(int node_rank, int local_rank, std::string base_path, std::string cuda_home, std::string cccl_include_dir, std::string jit_cache_dir, std::string comm_id);

    void run_preprocess_kernel(
        HybridEpConfigInstance config, 
        const bool* input_routing_map,
        hybrid_ep::tmp_state_t* preprocessing_tmp,
        hybrid_ep::tmp_state_t* preprocessing_local_experts_tmp,
        int32_t* sparse_to_dense_map,
        bool* rdma_to_attn_map,
        bool* attn_to_rdma_map,
        int32_t* num_of_tokens_for_experts,
        bool* local_expert_routing_map,
        int32_t* dense_chunk_layout,
        int32_t* dense_to_expert_map,
        int32_t* num_of_local_experts_tokens,
        int* token_drop_triggered,
        const int node_rank,
        const int local_rank,
        const int local_experts_tokens_limit,
        int num_of_tokens_per_rank,
        bool fuse_permute_dispatch,
        bool non_blocking,
        cudaStream_t stream
    );

    template <typename DATA_TYPE>
    void run_dispatch_kernel(
        HybridEpConfigInstance config, 
        hybrid_ep::dispatch_kernel_param_t<DATA_TYPE> param,
        bool fuse_permute_dispatch,
        bool non_blocking,
        cudaStream_t stream
    );

    void run_combine_kernel(
        HybridEpConfigInstance config, 
        hybrid_ep::combine_kernel_param_t param,
        bool fuse_unpermute_combine,
        bool non_blocking,
        cudaStream_t stream
    );

private:
    std::any get_or_compile(
        const std::string& kernel_key,
        const std::string& code,
        int num_of_nodes,
        bool enable_permute_fusion,
        bool enable_token_drop);

    NVCCCompiler nvcc_compiler;
    std::unordered_map<std::string, std::any> kernel_cache;
    std::string jit_dir;    // The path of the jit directory
    std::string base_path;  // The path of the installed package
    int local_rank;   // Used to generate the unique signature for each rank
    int node_rank;    // Used to generate the unique signature for each node
};
