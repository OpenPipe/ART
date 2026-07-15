// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

#ifdef USE_NIXL

#include "buffer/nixl_connector.h"
#include "coordinator.cuh"
#include <cuda_runtime.h>
#include <memory>

// Copy `src_prob` (user input, layout
// `[num_tokens][num_of_nodes][prob_per_token]`) into `dst_prob` (NIXL send
// staging, layout `[num_of_nodes][max_tokens][prob_per_token]`). The
// dest-major layout lets the dispatch N2N warp issue one coalesced `nixlPut`
// per chunk (per dest) instead of one put per token. Same total HBM traffic
// as the `cudaMemcpyAsync` it replaces.
void restripe_prob_for_nixl_dispatch(
    const float* src_prob,
    float* dst_prob,
    int num_tokens,
    int max_tokens_per_rank,
    int num_of_nodes,
    int prob_per_token,
    cudaStream_t stream);

class NIXLCoordinator : public InterNodeCoordinator {
public:
    NIXLCoordinator() = default;
    ~NIXLCoordinator() override;

    void init(pybind11::object process_group, int node_rank, int local_rank, BufferConfig config) override;
    bool grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) override;
    void update_config(BufferConfig config) override;
    void allocate_buffers() override;
    void destroy() override;

    InterNodeDispatchBuffers& get_dispatch_buffers() override { return dispatch_buffers; }
    InterNodeCombineBuffers& get_combine_buffers() override { return combine_buffers; }

    InterNodeDispatchBuffers dispatch_buffers;
    InterNodeCombineBuffers combine_buffers;

private:
    void allocate_dispatch_buffers();
    void allocate_combine_buffers();
    void free_buffers();

    pybind11::object process_group;
    int node_rank = -1;
    int local_rank = -1;
    BufferConfig buffer_config;
    std::unique_ptr<hybrid_ep::HybridEP_NIXLConnector> nixl_connector;
    bool buffer_allocated = false;
};

#endif  // USE_NIXL
