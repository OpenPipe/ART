// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#pragma once

// HybridEP_NIXLConnector manages the lifecycle of NIXL-based inter-node GPU
// communication for the Hybrid-EP dispatch and combine kernels.
//
// Lifecycle:
//   1. Constructor:          Collects local peer info (IP, boot_id, device_id).
//   2. updateMemoryBuffers:  Creates a NIXL agent, registers GPU buffers with
//                            the agent, and publishes local metadata to etcd.
//   3. connectRanks:         Fetches remote metadata, exchanges peer info via
//                            NIXL notifications, performs UCX wireup, creates
//                            memory views, and builds GPU contexts.
//   4. disconnectRanks:      Invalidates remote metadata for graceful teardown.
//   5. Destructor:           Disconnects peers, destroys agents, frees GPU memory.

#ifdef USE_NIXL

#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "nixl.h"
#include "coordinator.cuh"

#define MAX_IP_LENGTH 16
#define MAX_BOOT_ID_LENGTH 37

// Peer information exchanged between ranks via NIXL notifications
struct NixlPeerInfo {
    char ip[MAX_IP_LENGTH];
    char boot_id[MAX_BOOT_ID_LENGTH];
    ino_t ipc_namespace_inode;
    // Dispatch receive buffers
    void *rdma_buffer_ptr;
    void *rdma_prob_buffer_ptr;
    void *rdma_scaling_factor_buffer_ptr;
    uint64_t *dispatch_flags_ptr;
    // Combine receive buffers
    void *combine_rdma_buffer_ptr;
    void *combine_rdma_prob_buffer_ptr;
    uint64_t *combine_flags_ptr;
    // Misc
    int device_id;
    int rank;
};

// NIXL agent information and Memory View management
struct NixlAgentInfo {
    std::shared_ptr<nixlAgent> agent;
    std::string agent_name;
    nixlBackendH* backend;
    nixl_xfer_dlist_t src_vram;
    std::vector<std::string> dst_agent_names;
    std::vector<bool> wire_up_done;
    nixl_opt_args_t extra_params;

    nixlMemViewH dispatch_local_mvh;
    nixlMemViewH dispatch_remote_data_mvh;
    nixlMemViewH dispatch_remote_signal_mvh;
    nixlMemViewH combine_local_mvh;
    nixlMemViewH combine_remote_data_mvh;
    nixlMemViewH combine_remote_signal_mvh;

    NixlAgentInfo(int num_remote_nodes, int max_num_ranks) :
        src_vram(VRAM_SEG),
        dst_agent_names(max_num_ranks),
        wire_up_done(max_num_ranks, false),
        dispatch_local_mvh(nullptr),
        dispatch_remote_data_mvh(nullptr),
        dispatch_remote_signal_mvh(nullptr),
        combine_local_mvh(nullptr),
        combine_remote_data_mvh(nullptr),
        combine_remote_signal_mvh(nullptr) {}
};

namespace hybrid_ep {

class HybridEP_NIXLConnector {
private:
    int rank_uuid;
    int local_device_id;
    int num_ranks;
    int num_experts_per_rank;
    int num_nodes;
    int ranks_per_node;
    int num_channels;

    std::vector<NixlAgentInfo> nixl_agent_infos;
    std::vector<NixlPeerInfo> nixl_peer_info;
    NixlPeerInfo my_peer_info;

    std::vector<int> connected_ranks;
    bool initialized;
    bool connected;

    InterNodeDispatchBuffers* dispatch_buf = nullptr;
    InterNodeCombineBuffers* combine_buf = nullptr;
    bool forward_dispatch = false;
    bool backward_combine = false;
    bool use_fp8 = false;

public:
    HybridEP_NIXLConnector(int rank_uuid, int local_device_id);
    ~HybridEP_NIXLConnector();

    void updateMemoryBuffers(
        int num_ranks,
        int num_experts_per_rank,
        int num_nodes,
        int ranks_per_node,
        int num_dispatch_blocks,
        int num_combine_blocks,
        InterNodeDispatchBuffers& dispatch_buffers,
        InterNodeCombineBuffers& combine_buffers);

    void connectRanks(const std::vector<int>& remote_rank_uuids);
    void disconnectRanks(const std::vector<int>& remote_rank_uuids);

    dispatch_gpu_nixl_ctx* get_dispatch_gpu_ctx();
    combine_gpu_nixl_ctx* get_combine_gpu_ctx();

private:
    void _nixl_agents_init(int num_agents);
    void _register_buffers_with_agents();
    void _nixl_agents_connect(const std::vector<int>& ranks);
    void _nixl_agents_wireup(const std::vector<int>& ranks);
    void _nixl_ucx_wireup(const std::vector<int>& ranks);
    void _nixl_agents_wiredown(const std::vector<int>& ranks);
    void _nixl_create_memory_views(const std::vector<int>& ranks);
    void _nixl_build_gpu_contexts(int num_dispatch_blocks, int num_combine_blocks);

    int current_epoch;
    std::string nixl_run_id;
    int gda_num_channels;

    dispatch_gpu_nixl_ctx *d_dispatch_nixl_ctx;
    combine_gpu_nixl_ctx *d_combine_nixl_ctx;
    uint64_t* d_dispatch_flag_counters;
    uint64_t* d_combine_flag_counters;
};

}  // namespace hybrid_ep

#endif  // USE_NIXL
