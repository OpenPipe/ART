// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved
#pragma once

#include "utils.cuh"
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifndef USE_NIXL
#include "doca_gpunetio_host.h"
#include "doca_gpunetio_device.h"
#include "infiniband/verbs.h"
#include "infiniband/mlx5dv.h"
#else
#include "nixl.h"
#include "nixl_device.cuh"

namespace hybrid_ep{

// GPU-side context for NIXL RDMA transfers.
// Each memory view is indexed by remote_idx * number of buffers per remote node + buffer index.
// num_channels maps to UCX_RC_GDA_NUM_CHANNELS for QP channel distribution.

struct dispatch_gpu_nixl_ctx {
  nixlMemViewH local_mvh;                 // Local source buffers (token, prob, scaling_factor)
  nixlMemViewH remote_data_mvh;           // Remote data buffers indexed by remote_idx
  nixlMemViewH remote_signal_mvh;         // Remote signal buffers indexed by remote_idx
  uint64_t *local_flag_counters;          // [num_remote_nodes] Local completion counters
  int num_remote_nodes;
  int num_channels;                       // From UCX_RC_GDA_NUM_CHANNELS - used for channel_id distribution
  int rank;
  int local_mvh_stride;
  int remote_data_mvh_stride;
};

struct combine_gpu_nixl_ctx {
  nixlMemViewH local_mvh;                 // Local source buffers (reduced token, prob)
  nixlMemViewH remote_data_mvh;           // Remote data buffers indexed by remote_idx
  nixlMemViewH remote_signal_mvh;         // Remote signal buffers indexed by remote_idx
  uint64_t *local_flag_counters;          // [num_remote_nodes] Local completion counters
  int num_remote_nodes;
  int num_channels;                       // From UCX_RC_GDA_NUM_CHANNELS - used for channel_id distribution
  int rank;
  int local_mvh_stride;
  int remote_data_mvh_stride;
};

} // namespace hybrid_ep
#endif // USE_NIXL
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
#include <cuda_pipeline_primitives.h>
#endif

namespace hybrid_ep{

template<int NUM_OF_BOOL_TO_REDUCE>
using Reduce_t =
  typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 8 == 0, uint64_t,
    typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 4 == 0, uint32_t,
      typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 2 == 0, uint16_t, uint8_t
      >::type
    >::type
  >::type;

template<int NUM_OF_BYTES_TO_COPY>
using Copy_t =
  typename std::conditional<NUM_OF_BYTES_TO_COPY % 16 == 0, uint4,
    typename std::conditional<NUM_OF_BYTES_TO_COPY % 8 == 0, uint2,
      typename std::conditional<NUM_OF_BYTES_TO_COPY % 4 == 0, uint32_t,
        typename std::conditional<NUM_OF_BYTES_TO_COPY % 2 == 0, uint16_t, uint8_t
        >::type
      >::type
    >::type
  >::type;

enum scan_state{
  EMPTY = 0, 
  PRIV_SUM = 1 
};

struct tmp_state_t{
  scan_state state;
  int32_t value;
};

// Generic warp group for warp-specializaion.
template<int NUM_WARPS,
         int STARTING_WARPS>
struct warp_group{
  __host__ __device__ static constexpr int size(){ return 32 * NUM_WARPS; }
  __host__ __device__ static constexpr int warp_size(){ return NUM_WARPS; }

  __host__ __device__ static int thread_rank(){ return threadIdx.x - (32 * STARTING_WARPS); }
  __host__ __device__ static int warp_rank(){ return thread_rank() / 32; }
};

template<typename TOKEN_DATA_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         bool FORWARD_DISPATCH>
struct dispatch_kernel_dynamic_shared_memory_buffer_t{};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
#ifndef USE_NIXL
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
#ifndef USE_NIXL
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
#ifndef USE_NIXL
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
#ifndef USE_NIXL
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL
};
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory ping-pong buffer for sparse_to_dense map for token data chunks. Should be 128B alignment for optimal perf for TMA.
  alignas(128) int32_t sparse_to_dense_map_buffer[2][NUM_OF_TOKENS_PER_CHUNK][NUM_OF_RANKS_PER_NODE];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mbarrier that protect sparse_to_dense map. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t sparse_to_dense_map_mbarrier_buffer[2];
  // Shared memory mbarrier that perform sync within S2G warp group. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t S2G_group_mbarrier_buffer;
};

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
template<typename TOKEN_DATA_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM, 
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         bool FORWARD_DISPATCH>
struct dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t{};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t permute_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float permute_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float permute_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t permute_mbarrier_buffer[NUM_OF_STAGES][2];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t permute_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float permute_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t permute_mbarrier_buffer[NUM_OF_STAGES][2];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t permute_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float permute_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t permute_mbarrier_buffer[NUM_OF_STAGES][2];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t permute_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t permute_mbarrier_buffer[NUM_OF_STAGES][2];
};
#endif

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         bool BACKWARD_COMBINE>
struct combine_kernel_dynamic_shared_memory_buffer_t{};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer for intra node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for intra node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory prob buffer for intra node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float intra_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for intra node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float intra_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES];

  // Shared memory mbarrier that protect intra node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that maintain producer->consumer relationship between intra-node red warp group and rdma warp group. 1 per chunk, Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_to_rdma_mbarrier_buffer[NUM_OF_NODES - 1][MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK];

#ifndef USE_NIXL
  // Shared memory mr info for combine. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) combine_memory_region_info_t combine_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool intra_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};
#endif

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory prob buffer for inter node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];

  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer for intra node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for intra node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory mbarrier that protect intra node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that maintain producer->consumer relationship between intra-node red warp group and rdma warp group. 1 per chunk, Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_to_rdma_mbarrier_buffer[NUM_OF_NODES - 1][MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK];

#ifndef USE_NIXL
  // Shared memory mr info for combine. (Mr info can be cached in shared memory, while qp info can't be cached.) DOCA only.
  alignas(8) combine_memory_region_info_t combine_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages. DOCA only.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
#endif // USE_NIXL

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool intra_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};
#endif

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         bool BACKWARD_COMBINE>
struct combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t{};

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, true>{
  // Shared memory token buffer for unpermute red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t unpermute_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for unpermute red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t unpermute_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory prob buffer for unpermute red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float unpermute_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for unpermute red warp group G2S data movement. Should be 16B alignment. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float unpermute_prob_G2S_buffer[NUM_OF_STAGES_G2S];

  // Shared memory mbarrier that protect unpermute red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t unpermute_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Local expert id(which local expert of this G2S token entry belongs to) for each token entry in G2S buffer.
  int unpermute_local_expert_id_G2S_buffer[NUM_OF_STAGES_G2S];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a unpermute reduction group, otherwise not.
  bool unpermute_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, false>{
  // Shared memory token buffer for unpermute red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t unpermute_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for unpermute red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t unpermute_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory mbarrier that protect unpermute red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t unpermute_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a unpermute reduction group, otherwise not.
  bool unpermute_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};
#endif

// Data structure for kernel parameter for dispatch kernel.
template<typename TOKEN_DATA_TYPE>
struct dispatch_kernel_param_t{
  // Input buffers. These buffers are local buffers.
  const TOKEN_DATA_TYPE* attn_input_token;
  const float* attn_input_prob; // Needed by expert layer, so only valid in forward dispatch.
  const float* attn_input_token_scaling_factor; // If input token is FP8 dtype, we need scaling factor for tokens.
  // Output buffers. These buffers are both local and remote buffers.
  TOKEN_DATA_TYPE* expert_output_token[MAX_NUM_OF_RANKS_PER_NODE];
  float* expert_output_prob[MAX_NUM_OF_RANKS_PER_NODE]; // Only valid in forward dispatch.
  float* expert_output_scaling_factor[MAX_NUM_OF_RANKS_PER_NODE]; // Only valid for FP8 token type.
  TOKEN_DATA_TYPE* local_expert_output_token;
  float* local_expert_output_prob;
  float* local_expert_output_scaling_factor;
  // Internal temp buffers. These buffers are local buffers.
  const TOKEN_DATA_TYPE* rdma_inter_node_group_token;
  const float* rdma_inter_node_group_prob; // Only valid in forward dispatch.
  const float* rdma_inter_node_group_scaling_factor; // Only valid for FP8 token type.
  uint64_t* rdma_inter_node_group_flags; // For RDMA Atomic flags.
  uint32_t* intra_node_write_completion_flags; // For intra-node S2G write completion notification. Need 2 flags for different parity for dispatch kernel.
  uint32_t* intra_node_expert_output_chunk_flags[MAX_NUM_OF_RANKS_PER_NODE]; // For intra-node S2G -> permute_G2S chunk write completion notification.
  // Metadata buffers. These buffers are local buffers.
  const bool* rdma_to_attn_map;
  const bool* attn_to_rdma_map;
  const int32_t* sparse_to_dense_map;
  const int32_t* dense_chunk_layout;
  const int32_t* dense_to_expert_map;
  const int32_t* num_of_local_experts_tokens;
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value; // Need 2 expected values for different parity for dispatch kernel.
  uint32_t* expected_permute_flag_value;
  uint32_t* intra_node_flag_parity;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // Multinode context: always same layout to avoid param_t ABI mismatch between runtime and JIT.
  // NIXL: multinode_ctx_ptr = dispatch_gpu_nixl_ctx*, multinode_aux_ptr = nullptr.
  // DOCA: multinode_ctx_ptr = d_qps_gpu (void**), multinode_aux_ptr = mr_info.
  void *multinode_ctx_ptr;
  void *multinode_aux_ptr;
};

// Data structure for kernel parameter for combine kernel.
struct combine_kernel_param_t{
  // Input buffers. These buffers are both local and remote buffers.
  uint16_t* expert_input_token[MAX_NUM_OF_RANKS_PER_NODE];
  float* expert_input_prob[MAX_NUM_OF_RANKS_PER_NODE];
  const uint16_t* local_expert_input_token;
  const float* local_expert_input_prob;
  // Output buffers. These buffers are local buffers.
  uint16_t* attn_output_token;
  float* attn_output_prob;
  // Internal temp buffers. These buffers are local buffers.
  uint16_t* rdma_intra_node_red_token;
  float* rdma_intra_node_red_prob;
  const uint16_t* rdma_inter_node_group_token;
  const float* rdma_inter_node_group_prob;
  uint64_t* rdma_inter_node_group_flags;
  uint32_t* intra_node_write_completion_flags; // For intra-node src ready notification. Need 2 flags for different parity for combine kernel.
  uint32_t* intra_node_expert_input_chunk_flags[MAX_NUM_OF_RANKS_PER_NODE]; // For unpermute red -> intra_node_G2S and inter_node_G2S chunk write completion notification.
  // Metadata buffers. These buffers are local buffers.
  const bool* rdma_to_attn_map;
  const bool* attn_to_rdma_map;
  const int32_t* sparse_to_dense_map;
  const int32_t* dense_chunk_layout;
  const int32_t* dense_to_expert_map;
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value; // Need 2 expected values for different parity for combine kernel.
  uint32_t* expected_unpermute_flag_value;
  uint32_t* intra_node_flag_parity;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // Multinode context: always same layout to avoid param_t ABI mismatch between runtime and JIT.
  // NIXL: multinode_ctx_ptr = combine_gpu_nixl_ctx*, multinode_aux_ptr = nullptr.
  // DOCA: multinode_ctx_ptr = d_qps_gpu (void**), multinode_aux_ptr = mr_info.
  void *multinode_ctx_ptr;
  void *multinode_aux_ptr;
};

// Each CUDA block has sixteen named barriers numbered 0..15.
// __syncthreads(); will use the 0 named barriers, so we want to avoid that.
// We want to use 1 for intra-node reduction warp group, >= 2 for inter-node reduction warp group, 
// RDMA warp group currently only contains 1 warp so does not use named bar yet, if it need to use, it should use 2 + NUM_OF_DATA_PIPELINE_PER_BLOCK. 
// For unpermute block, we use >= 1 for unpermute reduction warp group.
inline __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifdef USE_NIXL
// NIXL inter-node dispatch warp function (1 warp per CUDA block).
// Transfers: tokens, probs (FORWARD_DISPATCH), scaling factors (FP8).
// Coalesced path: bulk token+SF puts when all tokens are dense to a remote.
// Sparse path: contiguous token runs are merged into bulk puts (reduces
// per-nixlPut overhead: atomic WQE reservation, descriptor lookup, doorbell
// logic). Prob puts remain per-token (source strided by NUM_OF_NODES).
// All data puts use nixl_gpu_flags::defer; the final atomic signal uses
// NODELAY to flush everything in one doorbell.
template<typename INTER_NODE_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void N2N_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool *attn_to_rdma_map,
                                                      struct dispatch_gpu_nixl_ctx *nixl_ctx,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(INTER_NODE_GROUP::size() == 32, "INTER_NODE_GROUP should be 1 warp.");

  const int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  bool *smem_attn_to_rdma_map_ptr = smem_buffer_ptr->attn_to_rdma_map_buffer;
  
  const size_t local_stride = nixl_ctx->local_mvh_stride;
  const size_t remote_stride = nixl_ctx->remote_data_mvh_stride;

  for (int chunk_idx = blockIdx.x; chunk_idx < NUM_OF_CHUNKS_PER_RANK; chunk_idx += NUM_OF_BLOCKS) {
    const int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = min(NUM_OF_TOKENS_PER_CHUNK, num_of_tokens_per_rank - chunk_base_token_idx);

    // Load routing map to shared memory.
    for (int m = INTER_NODE_GROUP::thread_rank(); m < token_range * (NUM_OF_NODES - 1); m += INTER_NODE_GROUP::size())
      smem_attn_to_rdma_map_ptr[m] = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + m];
    __syncwarp();

    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      const int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      const int actual_remote_node_rank = remote_idx < node_rank ? remote_idx : (remote_idx + 1);
      const int my_node_rank_in_remote = (node_rank < actual_remote_node_rank) ? node_rank : (node_rank - 1);
      const size_t flag_offset = (my_node_rank_in_remote * NUM_OF_CHUNKS_PER_RANK + chunk_idx) * sizeof(uint64_t);

      // Quick density probe: check first warp-width of tokens.
      // On 4+ nodes, per-remote density is ~70%, so this almost always fails,
      // letting us skip the full count pass and go straight to the original
      // interleaved count+put loop with near-zero overhead.
      bool try_coalesce;
      {
        int t = INTER_NODE_GROUP::thread_rank();
        bool probe = (t < token_range) && smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)];
        uint32_t ballot = __ballot_sync(0xffffffff, probe || t >= token_range);
        try_coalesce = (ballot == 0xffffffff) && (token_range > 0);
      }

      int total_tokens = 0;

      if (try_coalesce) {
        // First 32 tokens all need write — worth doing full count
        for (int t = INTER_NODE_GROUP::thread_rank(); t < NUM_OF_TOKENS_PER_CHUNK; t += INTER_NODE_GROUP::size()) {
          const bool need_write = (t < token_range) && smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)];
          total_tokens += __popc(__ballot_sync(0xffffffff, need_write));
        }
      }

      if (try_coalesce && total_tokens == token_range) {
        // All tokens in this chunk need write: coalesce token data, prob,
        // and (if FP8) SF into single puts. Prob coalescing relies on the
        // dest-major source layout `[NUM_OF_NODES][max_tokens][prob_per_token]`
        // set up by `restripe_prob_for_nixl_dispatch`.
        constexpr uint64_t DEFER = nixl_gpu_flags::defer;
        const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;

        if (INTER_NODE_GROUP::thread_rank() == 0) {
          size_t chunk_local_base = (size_t)chunk_base_token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
          size_t chunk_remote_base = (size_t)chunk_base_token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
          size_t chunk_size = (size_t)token_range * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);

          nixlMemViewElem src_desc{nixl_ctx->local_mvh, 0, chunk_local_base};
          nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 0, chunk_remote_base};

          nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, chunk_size, channel_id, DEFER);
          assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        }

        if constexpr (FORWARD_DISPATCH) {
          if (INTER_NODE_GROUP::thread_rank() == 0) {
            constexpr size_t prob_per_token = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
            // Source slot for the (actual_remote_node_rank, chunk) range under the dest-major layout.
            size_t local_offset = ((size_t)actual_remote_node_rank * MAX_NUM_OF_TOKENS_PER_RANK + chunk_base_token_idx) * prob_per_token * sizeof(float);
            size_t remote_offset = (size_t)chunk_base_token_idx * prob_per_token * sizeof(float);
            size_t put_size = (size_t)token_range * prob_per_token * sizeof(float);

            nixlMemViewElem src_desc{nixl_ctx->local_mvh, 1, local_offset};
            nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 1, remote_offset};

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, put_size, channel_id, DEFER);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }
        }

        if constexpr (std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
          if (INTER_NODE_GROUP::thread_rank() == 0) {
            size_t chunk_local_base = (size_t)chunk_base_token_idx * (HIDDEN_DIM / 128) * sizeof(float);
            size_t chunk_remote_base = (size_t)chunk_base_token_idx * (HIDDEN_DIM / 128) * sizeof(float);
            size_t chunk_size = (size_t)token_range * (HIDDEN_DIM / 128) * sizeof(float);

            nixlMemViewElem src_desc{nixl_ctx->local_mvh, (size_t)(local_stride - 1), chunk_local_base};
            nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + (remote_stride - 1), chunk_remote_base};

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, chunk_size, channel_id, DEFER);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }
        }
      } else {
        // Sparse path: separate count pass then run-merged puts.
        // Count pass: all lanes cooperatively count active tokens.
        for (int t = INTER_NODE_GROUP::thread_rank(); t < NUM_OF_TOKENS_PER_CHUNK; t += INTER_NODE_GROUP::size()) {
          const bool need_write = (t < token_range) && smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)];
          total_tokens += __popc(__ballot_sync(0xffffffff, need_write));
        }

        // Put pass: lane 0 merges contiguous token runs into bulk puts.
        // Prob puts remain per-token (source layout is strided by NUM_OF_NODES).
        if (total_tokens > 0 && INTER_NODE_GROUP::thread_rank() == 0) {
          const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
          constexpr uint64_t DEFER = nixl_gpu_flags::defer;
          int t = 0;
          while (t < token_range) {
            if (!smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)]) { t++; continue; }
            const int run_start = t;
            while (t < token_range && smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)]) t++;
            const int run_len = t - run_start;
            const int token_start = run_start + chunk_base_token_idx;

            {
              size_t local_offset = (size_t)token_start * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
              size_t remote_offset = (size_t)token_start * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
              size_t put_size = (size_t)run_len * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);

              nixlMemViewElem src_desc{nixl_ctx->local_mvh, 0, local_offset};
              nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 0, remote_offset};

              nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
                src_desc, dst_desc, put_size, channel_id, DEFER);
              assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
            }

            if constexpr (FORWARD_DISPATCH) {
              // Coalesced prob put across the run. The dest-major source
              // layout `[NUM_OF_NODES][max_tokens][prob_per_token]` makes
              // a contiguous token run for a given dest contiguous in memory.
              constexpr size_t prob_per_token = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
              constexpr size_t prob_size = prob_per_token * sizeof(float);
              size_t local_offset = ((size_t)actual_remote_node_rank * MAX_NUM_OF_TOKENS_PER_RANK + token_start) * prob_size;
              size_t remote_offset = (size_t)token_start * prob_size;
              size_t put_size = (size_t)run_len * prob_size;

              nixlMemViewElem src_desc{nixl_ctx->local_mvh, 1, local_offset};
              nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 1, remote_offset};

              nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
                src_desc, dst_desc, put_size, channel_id, DEFER);
              assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
            }

            if constexpr (std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
              size_t local_offset = (size_t)token_start * (HIDDEN_DIM / 128) * sizeof(float);
              size_t remote_offset = (size_t)token_start * (HIDDEN_DIM / 128) * sizeof(float);
              size_t put_size = (size_t)run_len * (HIDDEN_DIM / 128) * sizeof(float);

              nixlMemViewElem src_desc{nixl_ctx->local_mvh, (size_t)(local_stride - 1), local_offset};
              nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + (remote_stride - 1), remote_offset};

              nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
                src_desc, dst_desc, put_size, channel_id, DEFER);
              assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
            }
          }
        }
      }

      __syncwarp();
      if (total_tokens > 0 && INTER_NODE_GROUP::thread_rank() == 0) {
        const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
        nixlMemViewElem sig{nixl_ctx->remote_signal_mvh, (size_t)remote_idx, flag_offset};
        assert(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(1, sig, channel_id, 0 /* NODELAY: flush all pending */) >= NIXL_SUCCESS);
        atomicAdd((unsigned long long*)&nixl_ctx->local_flag_counters[remote_idx], 1ULL);
      }
    }
  }
}

// NIXL inter-node combine warp function (1 warp per CUDA block).
// Transfers: tokens, probs (BACKWARD_COMBINE).
// Coalesced path: bulk put per buffer. Sparse path: contiguous token runs are
// merged into bulk puts (reduces per-nixlPut overhead: atomic WQE reservation,
// descriptor lookup, doorbell logic). All data puts use nixl_gpu_flags::defer;
// the final atomic signal uses NODELAY to flush everything in one doorbell.
template<typename INTER_NODE_RDMA_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_N2N_warp_group_device_function(
    const int node_rank,
    const int num_of_tokens_per_rank,
    const bool *rdma_to_attn_map,
    struct combine_gpu_nixl_ctx *nixl_ctx,
    SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");

  const size_t remote_stride = nixl_ctx->remote_data_mvh_stride;
  
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int TOTAL_NUM_OF_CHUNKS = (NUM_OF_NODES - 1) * NUM_OF_CHUNKS_PER_RANK;
  
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  uint32_t token_consumer_parity = 0;
  uint64_t (*mbarrier_ptr)[MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK] = nullptr;
  if constexpr(NUM_OF_NODES != 1)
    mbarrier_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;

  for (int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS) {
    const int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    const int chunk_id = i / (NUM_OF_NODES - 1);
    const int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    const int remote_idx = rdma_remote_node_id;
    const int my_node_rank_in_remote = (node_rank < node_id) ? node_rank : (node_rank - 1);
    const int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    const int token_range = min(NUM_OF_TOKENS_PER_CHUNK, num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK);

    while (!cuda::ptx::mbarrier_try_wait_parity(&mbarrier_ptr[rdma_remote_node_id][chunk_id], token_consumer_parity)) {}

    // Count pass: determine how many tokens need RDMA write
    int total_tokens = 0;
    for (int t = INTER_NODE_RDMA_GROUP::thread_rank(); t < NUM_OF_TOKENS_PER_CHUNK; t += INTER_NODE_RDMA_GROUP::size()) {
      const bool need_write = (t < token_range) && rdma_to_attn_map[t + chunk_base_token_idx];
      total_tokens += __popc(__ballot_sync(0xffffffff, need_write));
    }

    if (total_tokens == token_range && token_range > 0) {
      // All tokens in this chunk need write: single coalesced put per buffer
      if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
        unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
        {
          size_t chunk_local_base = (size_t)(rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK
                                             + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * HIDDEN_DIM * sizeof(uint16_t);
          size_t chunk_remote_base = (size_t)(chunk_id * NUM_OF_TOKENS_PER_CHUNK) * HIDDEN_DIM * sizeof(uint16_t);
          size_t chunk_size = (size_t)token_range * HIDDEN_DIM * sizeof(uint16_t);

          nixlMemViewElem src_desc{nixl_ctx->local_mvh, 0, chunk_local_base};
          nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 0, chunk_remote_base};

          nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, chunk_size, channel_id, nixl_gpu_flags::defer);
          assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        }
        if constexpr(BACKWARD_COMBINE) {
          constexpr size_t prob_per_token = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
          size_t chunk_local_base = (size_t)(rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK
                                             + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * prob_per_token * sizeof(float);
          size_t chunk_remote_base = (size_t)(chunk_id * NUM_OF_TOKENS_PER_CHUNK) * prob_per_token * sizeof(float);
          size_t chunk_size = (size_t)token_range * prob_per_token * sizeof(float);

          nixlMemViewElem src_desc{nixl_ctx->local_mvh, 1, chunk_local_base};
          nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 1, chunk_remote_base};

          nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, chunk_size, channel_id, nixl_gpu_flags::defer);
          assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        }
      }
    } else if (total_tokens > 0) {
      // Sparse routing: merge contiguous token runs into bulk puts.
      // Each nixlPut has fixed overhead (atomic WQE reservation + descriptor lookup),
      // so merging N contiguous tokens into 1 put reduces overhead ~N×.
      if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
        unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
        int t = 0;
        while (t < token_range) {
          if (!rdma_to_attn_map[t + chunk_base_token_idx]) { t++; continue; }
          const int run_start = t;
          while (t < token_range && rdma_to_attn_map[t + chunk_base_token_idx]) t++;
          const int run_len = t - run_start;
          const int token_start = run_start + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
          const int local_token_start = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + token_start;
          {
            size_t local_offset = (size_t)local_token_start * HIDDEN_DIM * sizeof(uint16_t);
            size_t remote_offset = (size_t)token_start * HIDDEN_DIM * sizeof(uint16_t);
            size_t put_size = (size_t)run_len * HIDDEN_DIM * sizeof(uint16_t);

            nixlMemViewElem src_desc{nixl_ctx->local_mvh, 0, local_offset};
            nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 0, remote_offset};

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, put_size, channel_id, nixl_gpu_flags::defer);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }
          if constexpr(BACKWARD_COMBINE) {
            constexpr size_t prob_per_token = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
            size_t local_offset = (size_t)local_token_start * prob_per_token * sizeof(float);
            size_t remote_offset = (size_t)token_start * prob_per_token * sizeof(float);
            size_t put_size = (size_t)run_len * prob_per_token * sizeof(float);

            nixlMemViewElem src_desc{nixl_ctx->local_mvh, 1, local_offset};
            nixlMemViewElem dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * remote_stride + 1, remote_offset};

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, put_size, channel_id, nixl_gpu_flags::defer);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }
        }
      }
    }

    __syncwarp();
    if (total_tokens > 0 && INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      const size_t flag_offset = (my_node_rank_in_remote * NUM_OF_CHUNKS_PER_RANK + chunk_id) * sizeof(uint64_t);
      const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
      nixlMemViewElem sig{nixl_ctx->remote_signal_mvh, (size_t)remote_idx, flag_offset};
      assert(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(1, sig, channel_id, 0 /* NODELAY: flush all pending */) >= NIXL_SUCCESS);
      atomicAdd((unsigned long long*)&nixl_ctx->local_flag_counters[remote_idx], 1ULL);
    }
  }

  token_consumer_parity ^= 1;
}
#else
// Device function for inter-node node2node(RDMA) warp for dispatch kernel. There can be only 1 inter-node warp per CUDA block!
template<typename INTER_NODE_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void N2N_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool *attn_to_rdma_map,
                                                      doca_gpu_dev_verbs_qp **d_qps_gpu,
                                                      struct dispatch_memory_region_info_t *mr_info,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  // Load attn_to_rdma_map using LDG.128. Each token will need 1 bool from this map.
  // using attn_to_rdma_map_load_t = uint4;
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int MAX_NUM_OF_CHUNKS_PER_RANK = (MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  constexpr int WQE_NUM_RATIO = 1 + std::is_same<TOKEN_DATA_TYPE, uint8_t>::value + FORWARD_DISPATCH;
  // constexpr int NUN_OF_ATTN_TO_RDMA_MAP_LOAD_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1) / sizeof(attn_to_rdma_map_load_t);
  static_assert(INTER_NODE_GROUP::size() == 32, "INTER_NODE_GROUP should be 1 warp.");
  static_assert(INTER_NODE_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  // static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(attn_to_rdma_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(attn_to_rdma_map_load_t).");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Loading mr_infos to shared memory.
  struct dispatch_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    if (INTER_NODE_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_mr_info_ptr[INTER_NODE_GROUP::thread_rank()] = mr_info[INTER_NODE_GROUP::thread_rank() + block_offset];
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_GROUP::thread_rank()] = 0;
    }
    __syncwarp();
  }
  // For each chunk.
  for (int chunk_idx = blockIdx.x; chunk_idx < NUM_OF_CHUNKS_PER_RANK; chunk_idx += NUM_OF_BLOCKS) {
    int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    // Attn_to_rdma_map cached in shared memory.
    bool *smem_attn_to_rdma_map_ptr = nullptr;
    // Reading one chunk of attn_to_rdma_map into shared memory.
    if constexpr(NUM_OF_NODES != 1) {
      smem_attn_to_rdma_map_ptr = smem_buffer_ptr->attn_to_rdma_map_buffer;
      if (chunk_base_token_idx + token_range > num_of_tokens_per_rank) {
        token_range = num_of_tokens_per_rank - chunk_base_token_idx;
      }
      for (int map_load_idx = INTER_NODE_GROUP::thread_rank();
           map_load_idx < token_range * (NUM_OF_NODES - 1);
           map_load_idx += INTER_NODE_GROUP::size()) {
        smem_attn_to_rdma_map_ptr[map_load_idx] = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + map_load_idx];
      }
      __syncwarp();
    }
    // For each remote.
    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      int rank_in_remote = remote_idx < node_rank ? (node_rank - 1) : node_rank;
      // Queue pair for the current block to the current remote.
      struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[remote_idx + block_offset];
      // Real remote node rank.
      int remote_node_rank = remote_idx < node_rank ? remote_idx : remote_idx + 1;
      // Calculating total num of tokens need to be sent to the current remote.
      int num_of_tokens_need_write = 0;
      for (int token_idx_in_chunk = INTER_NODE_GROUP::thread_rank();
           token_idx_in_chunk < token_range;
           token_idx_in_chunk += INTER_NODE_GROUP::size()) {
        num_of_tokens_need_write += smem_attn_to_rdma_map_ptr[remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
      }
      num_of_tokens_need_write = __reduce_add_sync(0xffffffff, num_of_tokens_need_write);
      int total_write_cnt = num_of_tokens_need_write * WQE_NUM_RATIO + 1;
      // Getting wqe slots.
      uint64_t base_wqe_idx = 0;
      if (INTER_NODE_GROUP::thread_rank() == 0) {
        base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, total_write_cnt);
        smem_inter_node_num_of_write_per_node_ptr[remote_idx] += total_write_cnt;
      }
      base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
      uint64_t curr_wqe_idx = base_wqe_idx;
      // For the current chunk to the current remote.
      for (int token_idx_in_chunk = INTER_NODE_GROUP::thread_rank();
           token_idx_in_chunk < NUM_OF_TOKENS_PER_CHUNK;
           token_idx_in_chunk += INTER_NODE_GROUP::size()) {
        int token_idx = token_idx_in_chunk + chunk_base_token_idx;
        bool need_write = false;
        if (token_idx_in_chunk < token_range) {
          need_write = smem_attn_to_rdma_map_ptr[remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
        }
        uint32_t write_map = __ballot_sync(0xffffffff, need_write);
        uint32_t partial_write_map = ((1 << INTER_NODE_GROUP::thread_rank()) - 1) & write_map;
        int write_cnt = __popc(write_map);
        int write_idx = __popc(partial_write_map);
        if (need_write) {
          // Constructing wqes for tokens.
          uint64_t my_wqe_idx = curr_wqe_idx + write_idx;
          struct doca_gpu_dev_verbs_wqe *token_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
          doca_gpu_dev_verbs_wqe_prepare_write(qp, token_wqe_ptr, my_wqe_idx,
                                                    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                    smem_mr_info_ptr[remote_idx].token_raddr + token_idx * static_cast<int64_t>(HIDDEN_DIM) * sizeof(TOKEN_DATA_TYPE),
                                                    smem_mr_info_ptr[remote_idx].token_rkey,
                                                    smem_mr_info_ptr[remote_idx].token_laddr + token_idx * static_cast<int64_t>(HIDDEN_DIM) * sizeof(TOKEN_DATA_TYPE),
                                                    smem_mr_info_ptr[remote_idx].token_lkey,
                                                    HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
          // Constructing wqes for probs.
          if constexpr(FORWARD_DISPATCH) {
            my_wqe_idx += write_cnt;
            struct doca_gpu_dev_verbs_wqe *prob_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_write(qp, prob_wqe_ptr, my_wqe_idx,
                                                      DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                      smem_mr_info_ptr[remote_idx].prob_raddr + token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].prob_rkey,
                                                      smem_mr_info_ptr[remote_idx].prob_laddr + (token_idx * NUM_OF_NODES + remote_node_rank) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].prob_lkey,
                                                      (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
          }
          // Constructing wqes for scaling factor(Only for FP8 token).
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            my_wqe_idx += write_cnt;
            struct doca_gpu_dev_verbs_wqe *sf_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_write(qp, sf_wqe_ptr, my_wqe_idx,
                                                      DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_raddr + token_idx * (HIDDEN_DIM / 128) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_rkey,
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_laddr + token_idx * (HIDDEN_DIM / 128) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_lkey,
                                                      (HIDDEN_DIM / 128) * sizeof(float));
          }
        }
        curr_wqe_idx += write_cnt * WQE_NUM_RATIO;
        __syncwarp(0xffffffff);
       }
      if (INTER_NODE_GROUP::thread_rank() == 0) {
        // Construct wqe for flag.
        struct doca_gpu_dev_verbs_wqe *flag_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, curr_wqe_idx);
        uint64_t offset_flag_laddr = smem_mr_info_ptr[remote_idx].flag_laddr + remote_idx * MAX_NUM_OF_CHUNKS_PER_RANK * sizeof(uint64_t);
        uint64_t offset_flag_raddr = smem_mr_info_ptr[remote_idx].flag_raddr + rank_in_remote * MAX_NUM_OF_CHUNKS_PER_RANK * sizeof(uint64_t);
        doca_gpu_dev_verbs_wqe_prepare_atomic(qp, flag_wqe_ptr, curr_wqe_idx,
                                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                                   DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                   offset_flag_raddr + chunk_idx * sizeof(uint64_t),
                                                   smem_mr_info_ptr[remote_idx].flag_rkey,
                                                   offset_flag_laddr + chunk_idx * sizeof(uint64_t),
                                                   smem_mr_info_ptr[remote_idx].flag_lkey,
                                                   sizeof(uint64_t), 1, 0);
        // Post send and poll cqs.
        doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, base_wqe_idx, curr_wqe_idx);
        doca_gpu_dev_verbs_submit_db<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                          DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                          DOCA_GPUNETIO_VERBS_QP_SQ>(qp, (curr_wqe_idx + 1));
      }
      __syncwarp();
    }
  }
  if (INTER_NODE_GROUP::thread_rank() < NUM_OF_NODES - 1) {
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[block_offset + INTER_NODE_GROUP::thread_rank()];
    uint32_t wc_num_to_poll = smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_GROUP::thread_rank()];
    if (wc_num_to_poll > 0) {
      int status = doca_gpu_dev_verbs_poll_cq<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                              DOCA_GPUNETIO_VERBS_QP_SQ>(
                                              doca_gpu_dev_verbs_qp_get_cq_sq(qp), wc_num_to_poll);
      assert(status >= 0);
    }
  }
}
#endif // USE_NIXL
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE

// Device function for intra-node G2S warp for dispatch kernel. There can be only 1 intra-node G2S warp per CUDA block!
template<typename INTRA_NODE_G2S_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void G2S_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const uint64_t* expected_flag_value,
                                                      const bool* rdma_to_attn_map,
                                                      const TOKEN_DATA_TYPE* attn_input_token, 
                                                      const float* attn_input_prob,
                                                      const float* attn_input_token_scaling_factor,
                                                      const TOKEN_DATA_TYPE* rdma_inter_node_group_token,
                                                      const float* rdma_inter_node_group_prob,
                                                      const float* rdma_inter_node_group_scaling_factor,
                                                      uint64_t* rdma_inter_node_group_flags,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK; 
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int max_num_of_chunks_per_rank = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int stage = 0;
  uint32_t consumer_parity = 1;

  // Only 1 thread within the G2S warp will be active, other threads will just exit.
  if (elect_sync(~0)) {
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
        int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
        // Check if the chunk of this node is ready to be consumed.
        // The chunks of local node is the attn input buffers, which are always ready to be consumed.
        // The chunks of remote node is the rdma_inter_node_group buffers, which is produced by remote RDMA Write operation. Should poll the flag produced by remote RDMA Atomic FA before consumed.
        if(node_id != node_rank){
          const uint64_t* flag_location = rdma_inter_node_group_flags + (rdma_buffer_tile_id * max_num_of_chunks_per_rank + i);
          uint64_t rdma_flag = 0;
          uint64_t expected = *expected_flag_value;
          do{
            rdma_flag = 0;
            // Need a strong system-scope load to observe external RDMA Atomic result.
            asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                         : "=l"(rdma_flag)
                         : "l"(__cvta_generic_to_global(flag_location))
                         : "memory");
          }while(rdma_flag != expected);
        }
        // Load every token and its properties from Global to Shared. Only load tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
        const TOKEN_DATA_TYPE* token_load_base_addr;
        const float* prob_load_base_addr;
        const float* scaling_factor_load_base_addr;
        // For other node's attn token and properties, read from rdma_inter_node_group buffers.
        // For this node's attn token and properties, read from attn input buffers.
        if(node_id != node_rank){
          int chunk_first_token_id = rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = rdma_inter_node_group_token + chunk_first_token_id * static_cast<int64_t>(HIDDEN_DIM);
          if constexpr(FORWARD_DISPATCH){
            prob_load_base_addr = rdma_inter_node_group_prob + chunk_first_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
            scaling_factor_load_base_addr = rdma_inter_node_group_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }else{
          int chunk_first_token_id = i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = attn_input_token + chunk_first_token_id * static_cast<int64_t>(HIDDEN_DIM);
          if constexpr(FORWARD_DISPATCH){
#ifdef USE_NIXL
            // NIXL: dest-major `attn_input_prob`
            // `[NUM_OF_NODES][MAX_NUM_OF_TOKENS_PER_RANK][prob_per_token]`.
            // Local node's slice starts at `node_rank * MAX_TOKENS`;
            // per-token stride is `prob_per_token`.
            prob_load_base_addr = attn_input_prob + (static_cast<int64_t>(node_rank) * MAX_NUM_OF_TOKENS_PER_RANK + chunk_first_token_id) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
#else
            prob_load_base_addr = attn_input_prob + chunk_first_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
#endif
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
            scaling_factor_load_base_addr = attn_input_token_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }
        //#pragma unroll
        for(int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if(current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            // If a token is needed by this node(i.e. any expert of this node), load the token and its properties to shared memory entry.
            if(token_needed_by_this_node){
              // Wait until shared memory has free entry.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_buffer[stage][1], consumer_parity)){}
              // Issue TMA to load current token and its properties from global to shared memory.
              uint32_t total_tx_size = 0;
              // Load token.
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_token_buffer[stage][0]),
                                       reinterpret_cast<const void*>(token_load_base_addr + (current_token_id * static_cast<int64_t>(HIDDEN_DIM))),
                                       (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)),
                                       &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));

              // Optionally load prob(Only in FW dispatch).
              if constexpr(FORWARD_DISPATCH){
                // rdma_inter_node_group prob buffers and attn prob buffers will have different prob vec length.
                const float* prob_load_token_addr;
                if(node_id != node_rank){
                  prob_load_token_addr = prob_load_base_addr + (current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
                }else{
#ifdef USE_NIXL
                  // Dest-major `attn_input_prob`: per-token stride is
                  // `prob_per_token`; local-node base already offset by
                  // `node_rank * MAX_TOKENS` in `prob_load_base_addr`.
                  prob_load_token_addr = prob_load_base_addr + (current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
#else
                  prob_load_token_addr = prob_load_base_addr + (current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES)) +
                                                               (node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
#endif
                }
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_prob_buffer[stage][0]),
                                         reinterpret_cast<const void*>(prob_load_token_addr),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                         &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

                total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
              }

              // Optionally load scaling factor(Only for FP8 token).
              if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_scaling_factor_buffer[stage][0]),
                                         reinterpret_cast<const void*>(scaling_factor_load_base_addr + (current_token_id * (HIDDEN_DIM / 128))),
                                         (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)),
                                         &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

                total_tx_size += (uint32_t)((HIDDEN_DIM / 128) * sizeof(float));
              }

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0],
                                                   total_tx_size);

              stage += 1;
              if(stage == NUM_OF_STAGES){
                stage = 0;
                consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // Update residue flags.
  int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
  for (int node_id = blockIdx.x; node_id < NUM_OF_NODES - 1; node_id += gridDim.x) {
    uint64_t *residue_flag_base_ptr = rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
    for (int flag_id = INTRA_NODE_G2S_GROUP::thread_rank(); flag_id < residue_flag_count; flag_id += INTRA_NODE_G2S_GROUP::size()) {
      residue_flag_base_ptr[flag_id] = *expected_flag_value;
    }
  }
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE
}

// Device function for intra-node S2G warp group for dispatch kernel.
template<typename INTRA_NODE_S2G_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int NUM_OF_IN_FLIGHT_S2G,
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, // Different from NUM_OF_IN_FLIGHT_S2G, this parameter indicates how many in-flight S2G token entry before notifying the consumer(permute_G2S warp groups).
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void S2G_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool* rdma_to_attn_map,
                                                      const int32_t* sparse_to_dense_map,
                                                      TOKEN_DATA_TYPE* const* remote_expert_output_token,
                                                      float* const* remote_expert_output_prob,
                                                      float* const* remote_expert_output_scaling_factor,
                                                      uint32_t* const* intra_node_expert_output_chunk_flags,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(NUM_OF_IN_FLIGHT_S2G < NUM_OF_STAGES, "NUM_OF_IN_FLIGHT_S2G must smaller than NUM_OF_STAGES.");
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);
  
  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // How many S2G token entry of have been in-flight.
  int in_flight_s2g = 0;
  int stage = 0;
  uint32_t producer_parity = 0;
  // sparse_to_dense map stage for consuming.
  uint32_t sparse_to_dense_map_stage = 0;
  // sparse_to_dense map parity for consuming.
  uint32_t sparse_to_dense_map_parity = 0;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Whether there are S2G TMA operations of a previous chunk's token entry in-flight(unfinished).
  bool outstanding_in_flight_chunk = false;
  // global chunk id in per-rank buffer for previous chunk. Used for updating flags.
  int last_chunk_global_chunk_id;
#endif

  // Only 1 thread per warp within the S2G warp group will be active, other threads will just exit.
  if(elect_sync(~0)){
    // First warp(thread) will load the sparse_to_dense map for the first chunk for this CUDA block if any.
    if(INTRA_NODE_S2G_GROUP::warp_rank() == 0){
      if((int)blockIdx.x < num_of_chunks_per_rank){
        // How many token for this chunk.
        int current_chunk_size;
        if(remainder_chunk_size != 0 && (int)blockIdx.x == num_of_chunks_per_rank - 1){
          current_chunk_size = remainder_chunk_size;
        }else{
          current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }
        // sparse_to_dense map load base addr.
        const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + (int)blockIdx.x * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;
        // Load the sparse_to_dense map for the first chunk.
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                 cuda::ptx::space_global,
                                 reinterpret_cast<void*>(&smem_buffer_ptr->sparse_to_dense_map_buffer[sparse_to_dense_map_stage][0][0]),
                                 reinterpret_cast<const void*>(sparse_to_dense_map_load_base_addr),
                                 (uint32_t)(current_chunk_size * NUM_OF_RANKS_PER_NODE * sizeof(int32_t)),
                                 &smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[sparse_to_dense_map_stage]);

        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                             cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared,
                                             &smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[sparse_to_dense_map_stage],
                                             (uint32_t)(current_chunk_size * NUM_OF_RANKS_PER_NODE * sizeof(int32_t))); 
      }
    }
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        // All S2G warps(threads) need to sync to make sure all of them have finished consuming the sparse_to_dense map for the last chunk before prefetching the sparse_to_dense map for next chunk.
        // Equal to arrive_and_wait. But arrive_and_wait can only used for whole warps.
        uint64_t state_token = cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->S2G_group_mbarrier_buffer);
        while(!cuda::ptx::mbarrier_try_wait(&smem_buffer_ptr->S2G_group_mbarrier_buffer, state_token)){}

        // First warp(thread) will prefetch sparse_to_dense map for next chunk.
        if(INTRA_NODE_S2G_GROUP::warp_rank() == 0){
          // Calculate next chunk id for this CUDA block to prefetch sparse_to_dense map for next chunk.
          int next_chunk_id;
          int next_node_id;
          int next_node_iter = j + 1;
          if(next_node_iter < NUM_OF_NODES){
            next_chunk_id = i;
            next_node_id = node_rank >= next_node_iter ? node_rank - next_node_iter : node_rank + NUM_OF_NODES - next_node_iter;
          }else{
            next_chunk_id = i + NUM_OF_BLOCKS;
            next_node_id = node_rank;
          }
          
          // If next chunk exist, load the sparse_to_dense map for next chunk.
          if(next_chunk_id < num_of_chunks_per_rank){
            // How many token for this chunk.
            int current_chunk_size;
            if(remainder_chunk_size != 0 && next_chunk_id == num_of_chunks_per_rank - 1){
              current_chunk_size = remainder_chunk_size;
            }else{
              current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
            }
            // sparse_to_dense map load base addr.
            const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (next_node_id * num_of_tokens_per_rank + next_chunk_id * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;
            // Load the sparse_to_dense map for the next chunk.
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                     cuda::ptx::space_global,
                                     reinterpret_cast<void*>(&smem_buffer_ptr->sparse_to_dense_map_buffer[sparse_to_dense_map_stage ^ 1][0][0]),
                                     reinterpret_cast<const void*>(sparse_to_dense_map_load_base_addr),
                                     (uint32_t)(current_chunk_size * NUM_OF_RANKS_PER_NODE * sizeof(int32_t)),
                                     &smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[sparse_to_dense_map_stage ^ 1]);

            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                 cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared,
                                                 &smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[sparse_to_dense_map_stage ^ 1],
                                                 (uint32_t)(current_chunk_size * NUM_OF_RANKS_PER_NODE * sizeof(int32_t)));
          }
        }
        
        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // Store every token and its properties from Shared to Global. Only store tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));

        // Wait for sparse_to_dense map ready in smem for current chunk.
        while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[sparse_to_dense_map_stage], sparse_to_dense_map_parity)){}

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
        // How many S2G token entry of current chunk have been in-flight.
        int additional_in_flight_s2g = 0;
#endif
        for(int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
            // When permute fusion is enabled,
            // check whether there is a previous chunk's token entry S2G is in-flight and also current chunk already has NUM_OF_ADDITIONAL_IN_FLIGHT_S2G token entry S2G in-flight.
            // If so, wait for previous chunk's token entry S2G finish and notify the permute_G2S warp groups on all target ranks.
            if(outstanding_in_flight_chunk && (additional_in_flight_s2g == NUM_OF_ADDITIONAL_IN_FLIGHT_S2G)){
              // Wait for previous chunk's token entry S2G finish.
              cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<NUM_OF_ADDITIONAL_IN_FLIGHT_S2G>{});
              // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk 
              // before they can observe the update of the flags.
              // Required for both intra-node (NVLink peer memory) and inter-node communication.
              asm volatile("fence.release.sys;"
                           :
                           : 
                           : "memory");
              // Notify the permute_G2S warp groups of all target ranks in this node.
              // Atomically reduce add 1 to the u32 flag of the last attn token chunk to all target ranks within the current node.
              for(int m = INTRA_NODE_S2G_GROUP::warp_rank(); m < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; m += INTRA_NODE_S2G_GROUP::warp_size()){
                #pragma unroll
                for(int t = 0; t < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; t++){
                  int remote_rank_id = m * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + t;
                  uint32_t* last_chunk_flag_addr = intra_node_expert_output_chunk_flags[remote_rank_id] + last_chunk_global_chunk_id;
                  // Need a strong system-scope red to make sure the target ranks can observe the update of the flag,
                  // Notify last chunk.
                  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                               :
                               : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                               : "memory");
                }
              }
              outstanding_in_flight_chunk = false;
            }
#endif
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if(current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            if(token_needed_by_this_node){
              const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                                (&smem_buffer_ptr->sparse_to_dense_map_buffer[sparse_to_dense_map_stage][k * NUM_OF_TOKENS_PER_LOAD_ITER + n][0]);
              // Wait until token entry within the shared memory has been produced.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0], producer_parity)){}

              // This token entry will be multicast to all ranks within this node which need this token and its properties.
              // The current implementation do the multicast by issue each unicast separately(we call it a unicast group). If NVLS can be used, we should use it here. 
              // Multicast of a src token will be ditributed to multiple S2G threads.
              for(int m = INTRA_NODE_S2G_GROUP::warp_rank(); m < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; m += INTRA_NODE_S2G_GROUP::warp_size()){
                // Load sparse_to_dense_map.
                sparse_to_dense_map_load_t sparse_to_dense_map_data = sparse_to_dense_map_load_addr[m];
                #pragma unroll
                for(int t = 0; t < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; t++){
                  int32_t output_buffer_index = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data) + t);
                  // Only unicast to this rank if it need the current token.
                  if(output_buffer_index != -1){
                    int remote_rank_id = m * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + t;
                    // Store the token from shared to remote global.
                    TOKEN_DATA_TYPE* remote_token_addr = remote_expert_output_token[remote_rank_id] + (output_buffer_index * static_cast<int64_t>(HIDDEN_DIM));
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                             cuda::ptx::space_shared,
                                             reinterpret_cast<void*>(remote_token_addr),
                                             reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_token_buffer[stage][0]),
                                             (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)));

                    // Store the prob from shared to remote global for FW dispatch.
                    if constexpr(FORWARD_DISPATCH){
                      float* remote_prob_addr = remote_expert_output_prob[remote_rank_id] + (output_buffer_index * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_prob_addr),
                                               reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_prob_buffer[stage][0]),
                                               (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)));

                    }

                    // Store the scaling factor from shared to remote global for FP8 tokens.
                    if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
                      float* remote_scaling_factor_addr = remote_expert_output_scaling_factor[remote_rank_id] + (output_buffer_index * (HIDDEN_DIM / 128));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_scaling_factor_addr),
                                               reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_scaling_factor_buffer[stage][0]),
                                               (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)));

                    }
                  }
                }
              }
              // Commit the previous issued S2G TMA instructions for the same shared memory token entry to a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
              // Add 1 more in-flight S2G token entry to the counter.
              in_flight_s2g += 1;
              // If in-flight S2G token entry count has exceeded the expectation, release the 1 oldest token entry for the producer.
              if(in_flight_s2g > NUM_OF_IN_FLIGHT_S2G){
                // Wait for all TMA S2G instructions for the 1 oldest token entry to finish reading the shared memory, so the token entry can be reused by the producer.
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
                // Reduce 1 in-flight S2G token entry from the counter.
                in_flight_s2g -= 1;
                // Notify the producer warp to load next token entry to the oldest token entry as the shared memory can be reused.
                int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ? (stage - NUM_OF_IN_FLIGHT_S2G) : (stage - NUM_OF_IN_FLIGHT_S2G + NUM_OF_STAGES);
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_mbarrier_buffer[notify_stage][1]);
              }
              
              // Goto next token entry in shared memory.
              stage += 1;
              if(stage == NUM_OF_STAGES){
                stage = 0;
                producer_parity ^= 1;
              }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
              // Another token entry's S2G in-flight.
              additional_in_flight_s2g += 1;
#endif
            }
          }
        }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
        // When permute fusion is enabled, we need to notify the permute_G2S warp groups of all target ranks in this node that a chunks is ready to be consumed.
        // Calculate what is the global chunk id of the current chunk within the per-rank buffer.
        int global_chunk_id = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * node_id + num_of_chunks_per_rank * local_rank + i;
        // If the current chunk does not have NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token entry in-flight, which is possible of rdma_to_attn map is really sparse.
        // We need to wait for both previous and current chunks' S2G entry to finish and notify the permute_G2S warp groups.
        if(outstanding_in_flight_chunk){
          // Wait for all previous chunk's(i.e. previous and current chunk) S2G finish.
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk 
          // before they can observe the update of the flags.
          // Required for both intra-node (NVLink peer memory) and inter-node communication.
          asm volatile("fence.release.sys;"
                        :
                        : 
                        : "memory");
          // Notify the permute_G2S warp groups of all target ranks in this node.
          // Atomically reduce add 1 to the u32 flag of this attn token chunk to all target ranks within the current node.
          for(int k = INTRA_NODE_S2G_GROUP::warp_rank(); k < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; k += INTRA_NODE_S2G_GROUP::warp_size()){
            #pragma unroll
            for(int n = 0; n < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; n++){
              int remote_rank_id = k * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + n;
              uint32_t* last_chunk_flag_addr = intra_node_expert_output_chunk_flags[remote_rank_id] + last_chunk_global_chunk_id;
              uint32_t* current_chunk_flag_addr = intra_node_expert_output_chunk_flags[remote_rank_id] + global_chunk_id;
              // Need a strong system-scope red to make sure the target ranks can observe the update of the flag,
              // Notify last chunk.
              asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                           :
                           : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                           : "memory");
              // Notify current chunk.
              asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                           :
                           : "l"(__cvta_generic_to_global(current_chunk_flag_addr)), "n"(1)
                           : "memory");
            }
          }
          outstanding_in_flight_chunk = false;
        }else{ // Otherwise, the current chunks is in-flight.
          outstanding_in_flight_chunk = true;
        }

        // Update last chunk's id.
        last_chunk_global_chunk_id = global_chunk_id;
#endif
        // Before goto next chunk, go to next sparse_to_dense map stage.
        sparse_to_dense_map_stage += 1;
        if(sparse_to_dense_map_stage == 2){
          sparse_to_dense_map_stage = 0;
          sparse_to_dense_map_parity ^= 1;
        }
      }
    }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // When all chunks have been processed, we need to check whether the last chunk is still in-flight.
    // If so, wait for it and notify the permute_G2S warp groups.
    if(outstanding_in_flight_chunk){
      // Wait for the last chunk's S2G finish.
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk 
      // before they can observe the update of the flags.
      // Required for both intra-node (NVLink peer memory) and inter-node communication.
      asm volatile("fence.release.sys;"
                    :
                    : 
                    : "memory");
      // Notify the permute_G2S warp groups of all target ranks in this node.
      // Atomically reduce add 1 to the u32 flag of the last attn token chunk to all target ranks within the current node.
      for(int i = INTRA_NODE_S2G_GROUP::warp_rank(); i < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; i += INTRA_NODE_S2G_GROUP::warp_size()){
        #pragma unroll
        for(int j = 0; j < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; j++){
          int remote_rank_id = i * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + j;
          uint32_t* last_chunk_flag_addr = intra_node_expert_output_chunk_flags[remote_rank_id] + last_chunk_global_chunk_id;
          // Need a strong system-scope red to make sure the target ranks can observe the update of the flag,
          // Notify last chunk.
          asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                        :
                        : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                        : "memory");
        }
      }
    }
#endif
  }
}

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
// Device function for rank-local permute G2S warp for dispatch kernel. There can be only 1 such warp per CUDA block!
template<typename PERMUTE_G2S_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_PERMUTE_BLOCKS,
         int NUM_OF_DISPATCH_BLOCKS, // In current design, each dispatch grid has N dispatch blocks + M permute blocks, with dispatch block come first to minimize code change.
         bool FORWARD_DISPATCH>
inline __device__ void permute_G2S_warp_group_device_function(const int node_rank,
                                                              const int num_of_tokens_per_rank,
                                                              const uint32_t* expected_flag_value,
                                                              const int32_t* dense_chunk_layout,
                                                              const TOKEN_DATA_TYPE* remote_expert_output_token,
                                                              const float* remote_expert_output_prob,
                                                              const float* remote_expert_output_scaling_factor,
                                                              uint32_t* intra_node_expert_output_chunk_flags,
                                                              SMEM_TYPE* smem_buffer_ptr)
{ 
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // How many total chunks.
  const int num_of_total_attn_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  // How many chunks iterations. Including full iters and the remainder iter.
  const int num_of_total_chunk_iters = ((num_of_chunks_per_rank - 1) / NUM_OF_DISPATCH_BLOCKS) + 1;
  // Size of the remainder iter.
  const int remainder_iter_size = num_of_chunks_per_rank % NUM_OF_DISPATCH_BLOCKS;

  int stage = 0;
  uint32_t consumer_parity = 1;

  // Only 1 thread within the permute G2S warp will be active, other threads will just exit if no residue flag need to updated.
  if(elect_sync(~0)){
    // Loop through all data chunk. Data(chunk) parallel between multiple permute CUDA blocks.
    // We flatten the global chunk id of all attn chunks.
    // Need to take the dispatch block's offset into account.
    for(int i = blockIdx.x - NUM_OF_DISPATCH_BLOCKS; i < num_of_total_attn_chunks; i += NUM_OF_PERMUTE_BLOCKS){
      // Calculate which node, rank and chunk does this global chunk id map to.
      // The order of producing attn chunks in the local per-rank buffer(i.e. the remote_expert_output_token buffer) is [0, N-1] chunks for all ranks on local_node(N is NUM_OF_DISPATCH_BLOCKS), 
      // then [0, N-1] chunks for all ranks on local_node - 1, ......, then [0, N-1] chunks for all ranks on local_node + 1, then [N, 2N-1] chunks for all ranks on local_node,
      // ......, then [N, 2N -1] chunks for all ranks on local_node + 1. etc.
      // So the mapping order of global chunk id is first chunk_id, then rank_id and finally node_id, this full iteration called chunk iteration. 
      // Which is ([0, N-1] chunks for rank 0 on local_node, [0, N-1] chunks for rank 1 on local_node,
      // ......, [0, N-1] chunks for rank 0 on local_node - 1, [0, N-1] chunks for rank 1 on local_node - 1, ......,) ([N, 2N-1] chunks for rank 0 on local_node,
      // [N, 2N-1] chunks for rank 1 on local_node, ......, [N, 2N-1] chunks for rank 0 on local_node - 1, [N, 2N-1] chunks for rank 1 on local_node - 1,) etc.
      int current_chunk_iter = i / (NUM_OF_DISPATCH_BLOCKS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
      int local_chunk_id_offset = current_chunk_iter * NUM_OF_DISPATCH_BLOCKS;
      int current_iter_chunks_per_rank;
      if(remainder_iter_size != 0 && current_chunk_iter == num_of_total_chunk_iters - 1){
        current_iter_chunks_per_rank = remainder_iter_size;
      }else{
        current_iter_chunks_per_rank = NUM_OF_DISPATCH_BLOCKS;
      }
      int local_iter_chunk_id = i % (NUM_OF_DISPATCH_BLOCKS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
      int current_iter_chunks_per_node = current_iter_chunks_per_rank * NUM_OF_RANKS_PER_NODE;
      int current_node_linear_id = local_iter_chunk_id / current_iter_chunks_per_node;
      // The node id of current chunk.
      int current_node_id = node_rank >= current_node_linear_id ? node_rank - current_node_linear_id : node_rank + NUM_OF_NODES - current_node_linear_id;
      int local_node_chunk_id = local_iter_chunk_id % current_iter_chunks_per_node;
      // The rank id of the current chunk within its node.
      int current_rank_id = local_node_chunk_id / current_iter_chunks_per_rank;
      // The chunk id of the current chunk within its rank.
      int current_chunk_id = local_node_chunk_id % current_iter_chunks_per_rank + local_chunk_id_offset;

      // Calculate the chunk id of the current chunk within the per-rank buffer(i.e. the remote_expert_output_token buffer) according to the node_id, rank_id and chunk_id.
      int current_global_id = current_node_id * num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE + current_rank_id * num_of_chunks_per_rank + current_chunk_id;

      // Load the chunk layout info for current chunk from dense_chunk_layout map, and calculate the starting token and number of tokens of the current chunk within the local per-rank buffer.
      // Per-rank buffer is a dense buffer, every token within this buffer is needed by this rank(which means every token within this buffer is needed by at least 1 local expert).
      // So, every token within this buffer need to be multicast to at least 1 local expert, which means every token within this buffer need to be loaded into smem FIFO.
      int next_chunk_starting_location_within_expert_output_buffer = dense_chunk_layout[current_global_id];
      int current_chunk_starting_location_within_expert_output_buffer = 0;
      if(current_global_id != 0){
        current_chunk_starting_location_within_expert_output_buffer = dense_chunk_layout[current_global_id - 1];
      }
      int num_of_tokens_for_current_chunk = next_chunk_starting_location_within_expert_output_buffer - current_chunk_starting_location_within_expert_output_buffer;
      
      // Check if the current chunk is ready to be consumed within the local per-rank buffer.
      // The chunk within the local per-rank buffer is produced by S2G warp group of all rank within this node(a.k.a peer ranks). 
      // Should poll the flag produced by peer ranks' atomic reduce add before consuming the chunk.
      const uint32_t* flag_location = intra_node_expert_output_chunk_flags + current_global_id;
      uint32_t intra_node_chunk_flag = 0;
      do{
        intra_node_chunk_flag = 0;
        // Need a strong system-scope load to observe peer ranks' Atomic result.
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                      : "=r"(intra_node_chunk_flag)
                      : "l"(__cvta_generic_to_global(flag_location))
                      : "memory");
      }while(intra_node_chunk_flag != *expected_flag_value);

      const TOKEN_DATA_TYPE* token_load_base_addr = remote_expert_output_token + current_chunk_starting_location_within_expert_output_buffer * static_cast<int64_t>(HIDDEN_DIM);
      const float* prob_load_base_addr;
      const float* scaling_factor_load_base_addr;
      if constexpr(FORWARD_DISPATCH){
        prob_load_base_addr = remote_expert_output_prob + current_chunk_starting_location_within_expert_output_buffer * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
      }
      if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
        scaling_factor_load_base_addr = remote_expert_output_scaling_factor + current_chunk_starting_location_within_expert_output_buffer * (HIDDEN_DIM / 128);
      }

      // Future optimization point: considering that any chunk in per-rank buffer is a contiguous buffer, so theoretically we can do token coalescing in permute/unpermute up to chunk granularity.
      // But you need enough smem capacity to hold that.
      // So we still do a single-token level data transfer here which is the most generic pattern, if we find that this pattern is bottlenecking the overall perf(which is unlikely), 
      // we will need to use token coalescing pattern.
      for(int j = 0; j < num_of_tokens_for_current_chunk; j++){
        // Wait until shared memory has free entry.
        while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->permute_mbarrier_buffer[stage][1], consumer_parity)){}
        // Issue TMA to load current token and its properties from global to shared memory.
        uint32_t total_tx_size = 0;
        // Load token.
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                 cuda::ptx::space_global,
                                 reinterpret_cast<void*>(&smem_buffer_ptr->permute_token_buffer[stage][0]),
                                 reinterpret_cast<const void*>(token_load_base_addr + (j * static_cast<int64_t>(HIDDEN_DIM))),
                                 (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)),
                                 &smem_buffer_ptr->permute_mbarrier_buffer[stage][0]);

        total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));

        // Optionally load prob(Only in FW dispatch).
        // The prob vec in per-rank buffer has been padded to NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE to fit TMA requirement, so we still load the token from per-rank buffer in this format.
        if constexpr(FORWARD_DISPATCH){
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                   cuda::ptx::space_global,
                                   reinterpret_cast<void*>(&smem_buffer_ptr->permute_prob_buffer[stage][0]),
                                   reinterpret_cast<const void*>(prob_load_base_addr + (j * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE))),
                                   (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                   &smem_buffer_ptr->permute_mbarrier_buffer[stage][0]);

          total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
        }

        // Optionally load scaling factor(Only for FP8 token).
        if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                   cuda::ptx::space_global,
                                   reinterpret_cast<void*>(&smem_buffer_ptr->permute_scaling_factor_buffer[stage][0]),
                                   reinterpret_cast<const void*>(scaling_factor_load_base_addr + (j * (HIDDEN_DIM / 128))),
                                   (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)),
                                   &smem_buffer_ptr->permute_mbarrier_buffer[stage][0]);

          total_tx_size += (uint32_t)((HIDDEN_DIM / 128) * sizeof(float));
        }

        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                             cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared,
                                             &smem_buffer_ptr->permute_mbarrier_buffer[stage][0],
                                             total_tx_size);

        stage += 1;
        if(stage == NUM_OF_STAGES){
          stage = 0;
          consumer_parity ^= 1;
        }
      }
    }
  }
  // Update residue flags in intra_node_expert_output_chunk_flags. Write-and-forget operations.
  int residue_flag_count = (MAX_NUM_OF_CHUNKS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) - num_of_total_attn_chunks;
  // The residue flags will be updated by all threads of the permute_G2S warp group of all permute CUDA block. Calculate how many threads to perform this update oeprations.
  constexpr int RESIDUE_FLAG_UPDATE_THREAD_COUNT = PERMUTE_G2S_GROUP::size() * NUM_OF_PERMUTE_BLOCKS;
  int residue_flag_update_thread_id = PERMUTE_G2S_GROUP::thread_rank() + (blockIdx.x - NUM_OF_DISPATCH_BLOCKS) * PERMUTE_G2S_GROUP::size();
  for(int i = residue_flag_update_thread_id; i < residue_flag_count; i += RESIDUE_FLAG_UPDATE_THREAD_COUNT){
    intra_node_expert_output_chunk_flags[num_of_total_attn_chunks + i] = *expected_flag_value;
  }
}

// Device function for intra-node S2G warp group for dispatch kernel.
template<typename PERMUTE_S2G_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int NUM_OF_IN_FLIGHT_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int LOCAL_EXPERTS_PADDING_SIZE,
         int NUM_OF_PERMUTE_BLOCKS,
         int NUM_OF_DISPATCH_BLOCKS, // In current design, each dispatch grid has N dispatch blocks + M permute blocks, with dispatch block come first to minimize code change.
         bool FORWARD_DISPATCH>
inline __device__ void permute_S2G_warp_group_device_function(const int local_rank,
                                                              const int node_rank,
                                                              const int num_of_tokens_per_rank,
                                                              const int32_t* dense_chunk_layout,
                                                              const int32_t* dense_to_expert_map,
                                                              const int32_t* num_of_local_experts_tokens,
                                                              TOKEN_DATA_TYPE* local_expert_output_token,
                                                              float* local_expert_output_prob,
                                                              float* local_expert_output_scaling_factor,
                                                              SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(NUM_OF_IN_FLIGHT_S2G < NUM_OF_STAGES, "NUM_OF_IN_FLIGHT_S2G must smaller than NUM_OF_STAGES.");

  // Load dense_to_expert_map according to the NUM_OF_EXPERTS_PER_RANK.
  using dense_to_expert_map_load_t = Copy_t<NUM_OF_EXPERTS_PER_RANK * sizeof(int32_t)>;
  constexpr int NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_INPUT_TOKEN = (NUM_OF_EXPERTS_PER_RANK * sizeof(int32_t)) / sizeof(dense_to_expert_map_load_t);
  constexpr int NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER = sizeof(dense_to_expert_map_load_t) / sizeof(int32_t);
  
  //const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // How many total chunks.
  const int num_of_total_attn_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  // How many chunks iterations. Including full iters and the remainder iter.
  const int num_of_total_chunk_iters = ((num_of_chunks_per_rank - 1) / NUM_OF_DISPATCH_BLOCKS) + 1;
  // Size of the remainder iter.
  const int remainder_iter_size = num_of_chunks_per_rank % NUM_OF_DISPATCH_BLOCKS;

  // How many S2G token entry of smem FIFO have been in-flight.
  int in_flight_s2g = 0;
  int stage = 0;
  uint32_t producer_parity = 0;

  // Zero-init all padding tokens in local experts' output buffer. Write-and-forget operations.
  // The padding tokens will be init by all threads of the permute_S2G warp group of all permute CUDA block.
  // Padding tokens will be assigned to all permute blocks, and then a padding token will be divided into init unit and assigned to all threads of the permute_S2G warp group of this permute block.
  
  // Calculate the padding token count for each local expert, and their starting location within the local expert's output buffer.
  int32_t num_of_local_experts_padding_tokens[NUM_OF_EXPERTS_PER_RANK];
  int32_t starting_index_of_local_experts_padding_token[NUM_OF_EXPERTS_PER_RANK];
  int32_t token_acc = 0;
  int32_t total_local_experts_padding_tokens = 0;
  #pragma unroll
  for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
    // Number of actual tokens of current local expert.
    int32_t current_expert_tokens = num_of_local_experts_tokens[i];
    // Padding the number of token to padding size. Local experts sum can be >= zero, so need to handle the corner case.
    int num_of_padding_tile = (current_expert_tokens % LOCAL_EXPERTS_PADDING_SIZE == 0) ? (current_expert_tokens / LOCAL_EXPERTS_PADDING_SIZE)
                                                                                        : (current_expert_tokens / LOCAL_EXPERTS_PADDING_SIZE + 1);
    // Number of total tokens(actual + padding) of current expert.
    int32_t current_expert_tokens_with_padding = num_of_padding_tile * LOCAL_EXPERTS_PADDING_SIZE;
    // Number of padding tokens of current expert.
    num_of_local_experts_padding_tokens[i] = current_expert_tokens_with_padding - current_expert_tokens;
    // Starting location(global offset within the local expert buffer) of the first padding token of current local expert. 
    starting_index_of_local_experts_padding_token[i] = token_acc + current_expert_tokens;
    // Accumulate current local expert's total token count to accumulator.
    token_acc += current_expert_tokens_with_padding;
    total_local_experts_padding_tokens += num_of_local_experts_padding_tokens[i];
  }

  // We use uint4(STG.128) as the unit to init padding token.
  static_assert((HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)) % 16 == 0, "The size of each token must be multiple of 16B.");
  static_assert(sizeof(uint4) == 16, "uint4 is not 16Byte?");
  using token_init_t = uint4;
  constexpr int NUM_OF_INIT_ITER_PER_PADDING_TOKEN = (HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)) / sizeof(token_init_t);

  // We use uint4(STG.128) as the unit to init padding scaling vector.
  if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
    static_assert(((HIDDEN_DIM / 128) * sizeof(float)) % 16 == 0, "The size of each scaling vector must be multiple of 16B.");
  }
  using scaling_factor_init_t = uint4;
  constexpr int NUM_OF_INIT_ITER_PER_PADDING_SCALING_FACTOR = ((HIDDEN_DIM / 128) * sizeof(float)) / sizeof(scaling_factor_init_t);

  // We use float(STG.32) as the unit to init padding prob element(each padding token in the local experts' output buffer only has 1 prob element, not a prob vec).
  using prob_init_t = float;

  // Assign Padding tokens to all permute blocks.
  for(int i = blockIdx.x - NUM_OF_DISPATCH_BLOCKS; i < total_local_experts_padding_tokens; i += NUM_OF_PERMUTE_BLOCKS){
    // Calculate the global token id of this padding token within the local experts' output buffer.
    int padding_token_index;
    int32_t lower_bound = 0;
    #pragma unroll
    for(int j = 0; j < NUM_OF_EXPERTS_PER_RANK; j++){
      int32_t upper_bound = lower_bound + num_of_local_experts_padding_tokens[j];
      if(i < upper_bound){
        padding_token_index = starting_index_of_local_experts_padding_token[j] + (i - lower_bound);
        break;
      }
      lower_bound = upper_bound;
    }

    // Divide padding token into init unit and assigned them to all permute_S2G threads.
    token_init_t* padding_token_base_addr = reinterpret_cast<token_init_t*>(local_expert_output_token + padding_token_index * static_cast<int64_t>(HIDDEN_DIM));
    for(int j = PERMUTE_S2G_GROUP::thread_rank(); j < NUM_OF_INIT_ITER_PER_PADDING_TOKEN; j += PERMUTE_S2G_GROUP::size()){
      padding_token_base_addr[j] = make_uint4(0, 0, 0, 0);
    }

    // Divide padding scaling factor into init unit and assigned them to all permute_S2G threads for FP8 tokens.
    if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
      scaling_factor_init_t* padding_scaling_factor_base_addr = reinterpret_cast<scaling_factor_init_t*>(local_expert_output_scaling_factor + padding_token_index * (HIDDEN_DIM / 128));
      for(int j = PERMUTE_S2G_GROUP::thread_rank(); j < NUM_OF_INIT_ITER_PER_PADDING_SCALING_FACTOR; j += PERMUTE_S2G_GROUP::size()){
        padding_scaling_factor_base_addr[j] = make_uint4(0, 0, 0, 0);
      }
    }

    // Prob only has 1 init unit, so let the first permute_S2G thread to init it for FW dispatch.
    if constexpr(FORWARD_DISPATCH){
      prob_init_t* padding_prob_base_addr = reinterpret_cast<prob_init_t*>(local_expert_output_prob + padding_token_index);
      if(PERMUTE_S2G_GROUP::thread_rank() == 0){
        *padding_prob_base_addr = 0.0f;
      }
    }
  }

  // Only 1 thread per warp within the permute S2G warp group will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Loop through all data chunk. Data(chunk) parallel between multiple permute CUDA blocks.
    // We flatten the global chunk id of all attn chunks.
    // Need to take the dispatch block's offset into account.
    for(int i = blockIdx.x - NUM_OF_DISPATCH_BLOCKS; i < num_of_total_attn_chunks; i += NUM_OF_PERMUTE_BLOCKS){
      // Calculate which node, rank and chunk does this global chunk id map to.
      // Permute_S2G consume the tokens as the exact same order of which permute_G2S produce tokens, so the same calculation as permute_G2S warp group.
      int current_chunk_iter = i / (NUM_OF_DISPATCH_BLOCKS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
      int local_chunk_id_offset = current_chunk_iter * NUM_OF_DISPATCH_BLOCKS;
      int current_iter_chunks_per_rank;
      if(remainder_iter_size != 0 && current_chunk_iter == num_of_total_chunk_iters - 1){
        current_iter_chunks_per_rank = remainder_iter_size;
      }else{
        current_iter_chunks_per_rank = NUM_OF_DISPATCH_BLOCKS;
      }
      int local_iter_chunk_id = i % (NUM_OF_DISPATCH_BLOCKS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
      int current_iter_chunks_per_node = current_iter_chunks_per_rank * NUM_OF_RANKS_PER_NODE;
      int current_node_linear_id = local_iter_chunk_id / current_iter_chunks_per_node;
      // The node id of current chunk.
      int current_node_id = node_rank >= current_node_linear_id ? node_rank - current_node_linear_id : node_rank + NUM_OF_NODES - current_node_linear_id;
      int local_node_chunk_id = local_iter_chunk_id % current_iter_chunks_per_node;
      // The rank id of the current chunk within its node.
      int current_rank_id = local_node_chunk_id / current_iter_chunks_per_rank;
      // The chunk id of the current chunk within its rank.
      int current_chunk_id = local_node_chunk_id % current_iter_chunks_per_rank + local_chunk_id_offset;

      // Calculate the chunk id of the current chunk within the per-rank buffer(i.e. the remote_expert_output_token buffer) according to the node_id, rank_id and chunk_id.
      int current_global_id = current_node_id * num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE + current_rank_id * num_of_chunks_per_rank + current_chunk_id;

      // Load the chunk layout info for current chunk from dense_chunk_layout map, and calculate the starting token and number of tokens of the current chunk within the local per-rank buffer.
      // Per-rank buffer is a dense buffer, every token within this buffer is needed by this rank(which means every token within this buffer is needed by at least 1 local expert).
      // So, every token within this buffer need to be multicast to at least 1 local expert, which means every token within this buffer need to be loaded into smem FIFO.
      int next_chunk_starting_location_within_expert_output_buffer = dense_chunk_layout[current_global_id];
      int current_chunk_starting_location_within_expert_output_buffer = 0;
      if(current_global_id != 0){
        current_chunk_starting_location_within_expert_output_buffer = dense_chunk_layout[current_global_id - 1];
      }
      int num_of_tokens_for_current_chunk = next_chunk_starting_location_within_expert_output_buffer - current_chunk_starting_location_within_expert_output_buffer;

      // Base addr to load dense_to_expert_map for this chunk.
      const int32_t* dense_to_expert_map_load_base_addr = dense_to_expert_map + current_chunk_starting_location_within_expert_output_buffer * NUM_OF_EXPERTS_PER_RANK;

      for(int j = 0; j < num_of_tokens_for_current_chunk; j++){
        const dense_to_expert_map_load_t* dense_to_expert_map_load_addr = reinterpret_cast<const dense_to_expert_map_load_t*>(dense_to_expert_map_load_base_addr + j * NUM_OF_EXPERTS_PER_RANK);
        // Wait until token entry within the shared memory has been produced.
        while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->permute_mbarrier_buffer[stage][0], producer_parity)){}

        // This token entry will be multicast to all local experts on local rank which need this token and its properties.
        // The current implementation do the multicast by issue each unicast separately(we call it a unicast group). 
        // Multicast of a src token will be ditributed to multiple permute S2G threads.
        for(int k = PERMUTE_S2G_GROUP::warp_rank(); k < NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_INPUT_TOKEN; k += PERMUTE_S2G_GROUP::warp_size()){
          // Load dense_to_expert_map.
          dense_to_expert_map_load_t dense_to_expert_map_data = dense_to_expert_map_load_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; n++){
            int current_local_expert_id = k * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + n;
            int32_t output_buffer_index = *(reinterpret_cast<int32_t*>(&dense_to_expert_map_data) + n);
            // Only unicast to this local expert if it need the current token.
            if(output_buffer_index != -1){
              // Store the token from shared to local global(local expert output buffers).
              TOKEN_DATA_TYPE* local_expert_token_addr = local_expert_output_token + (output_buffer_index * static_cast<int64_t>(HIDDEN_DIM));
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(local_expert_token_addr),
                                       reinterpret_cast<const void*>(&smem_buffer_ptr->permute_token_buffer[stage][0]),
                                       (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)));

              // Store the prob from shared to local global(local expert output buffers) for FW dispatch.
              if constexpr(FORWARD_DISPATCH){
                float* local_expert_prob_addr = local_expert_output_prob + output_buffer_index;
                // Only need to save 1 prob element back to the local global buffer, so can't use TMA for prob element, use normal LDS+STG instead.
                *local_expert_prob_addr = smem_buffer_ptr->permute_prob_buffer[stage][local_rank * NUM_OF_EXPERTS_PER_RANK + current_local_expert_id];
              }

              // Store the scaling factor from shared to local global(local expert output buffers) for FP8 tokens.
              if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
                float* local_expert_scaling_factor_addr = local_expert_output_scaling_factor + (output_buffer_index * (HIDDEN_DIM / 128));
                cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                         cuda::ptx::space_shared,
                                         reinterpret_cast<void*>(local_expert_scaling_factor_addr),
                                         reinterpret_cast<const void*>(&smem_buffer_ptr->permute_scaling_factor_buffer[stage][0]),
                                         (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)));

              }
            }
          }
        }
        // Commit the previous issued S2G TMA instructions for the same shared memory token entry to a bulk async copy group.
        cuda::ptx::cp_async_bulk_commit_group();
        // Add 1 more in-flight S2G token entry to the counter.
        in_flight_s2g += 1;
        // If in-flight S2G token entry count has exceeded the expectation, release the 1 oldest token entry for the producer.
        if(in_flight_s2g > NUM_OF_IN_FLIGHT_S2G){
          // Wait for all TMA S2G instructions for the 1 oldest token entry to finish reading the shared memory, so the token entry can be reused by the producer.
          cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
          // Reduce 1 in-flight S2G token entry from the counter.
          in_flight_s2g -= 1;
          // Notify the producer warp to load next token entry to the oldest token entry as the shared memory can be reused.
          int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ? (stage - NUM_OF_IN_FLIGHT_S2G) : (stage - NUM_OF_IN_FLIGHT_S2G + NUM_OF_STAGES);
          // mbarrier_arrive will have a default .release semantics to .cta scope(all threads within this permute CUDA block).
          // So itself can already guarantee that any thread within this permute block can observe the completion of 
          // all the normal LDS and STG instructions to the prob elements before the this arrive operations(up till this token entry) and the TMA reading operations for the oldest token entry
          // when they successfully wait for the arrive-on operation of this mbarrier.
          // So, although we don't use TMA instructions for prob elements, mbarrier_arrive can already guarantee the correct behaviour.
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->permute_mbarrier_buffer[notify_stage][1]);
        }
              
        // Goto next token entry in shared memory.
        stage += 1;
        if(stage == NUM_OF_STAGES){
          stage = 0;
          producer_parity ^= 1;
        }
      }
    }
  }
}
#endif

// Device function for intra-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template<typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S, 
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void intra_node_G2S_warp_group_device_function(const int node_rank,
                                                                 const int local_rank,
                                                                 const int num_of_tokens_per_rank, 
                                                                 const uint32_t* expected_flag_value,
                                                                 const bool* rdma_to_attn_map,
                                                                 const int32_t* sparse_to_dense_map, 
                                                                 uint16_t* const* remote_expert_input_token,
                                                                 float* const* remote_expert_input_prob,
                                                                 uint32_t* intra_node_expert_input_chunk_flags,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  // The intra node reduction warp group of each CUDA block produce a chunk at a time.
  // The chunk order is: first produce the same chunk id for all other nodes id, then produce following chunk id.
  // (i.e. chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, node + 2, ... node - 1)
  // The RDMA warp group of a CUDA block will consume the chunk by the same order. So each CUDA block will produce and consume the same set of chunks id.
  // The reason to distribute chunk in this order is that the inter-node reduction will need the same chunk id from all other nodes, so we need to produce and send chunks in this order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to produce for RDMA warps to consume.
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Token stage id and phase.
  int token_stage = 0;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within the intra-node G2S warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Iterate through all chunks assigned to this block.
    for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
      // Which node this chunk will be sent to.
      int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
      // What is the chunk id of this chunk for the node it will be sent to.
      int chunk_id = i / (NUM_OF_NODES - 1);
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
    
      const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;
    
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      // The base offset of the flags to be polled within the local unpermute flag buffer.
      int current_flag_base_offset = node_id * NUM_OF_RANKS_PER_NODE * num_of_chunks_per_rank + chunk_id;
      // Array to hold the status of all src chunks from per-rank buffer(remote_expert_input buffer) for the current (dst) chunk, and init to false.
      // So for each src chunk, we only need to poll the flag once per intra_node_G2S thread(warp) and record its status for later usage.
      bool remote_expert_input_chunk_flag_clear[NUM_OF_RANKS_PER_NODE];
      #pragma unroll
      for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
        remote_expert_input_chunk_flag_clear[j] = false;
      }
#endif
      // Iterate through all dst tokens within this chunk.
      for(int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++){
        rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
        #pragma unroll
        for(int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++){
          int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
          // If the current token is out-of-bound, then just end this load iter.
          if(current_token_id >= current_chunk_size){
            break;
          }
          // Check whether this dst token is needed by this node. If not needed, just skip.
          bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
          // If this dst token is needed by this node, load the sparse_to_dense map and load the src token for this dst token.
          if(token_needed_by_this_node){
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * NUM_OF_RANKS_PER_NODE);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            sparse_to_dense_map_load_t sparse_to_dense_map_data[NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = 0;
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  last_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                }
              }
            }

            // Then issue all G2S TMA for this row.
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
                  // If the current src chunk in the target per-rank buffer is not ready yet, wait for the src chunk ready first.
                  if(remote_expert_input_chunk_flag_clear[current_src_token_id] == false){
                    const uint32_t* flag_location = intra_node_expert_input_chunk_flags + (current_flag_base_offset + current_src_token_id * num_of_chunks_per_rank);
                    uint32_t intra_node_chunk_flag = 0;
                    do{
                      intra_node_chunk_flag = 0;
                      // Need a strong system-scope load to observe peer ranks' Atomic result.
                      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                                   : "=r"(intra_node_chunk_flag)
                                   : "l"(__cvta_generic_to_global(flag_location))
                                   : "memory");
                    }while(intra_node_chunk_flag != *expected_flag_value);
            
                    // Mark the src chunk from this rank is already clear.
                    remote_expert_input_chunk_flag_clear[current_src_token_id] = true;
                  }
#endif
                  // Wait until current token entry within the shared memory has been consumed.
                  while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_token_G2S_buffer[token_stage][0]),
                                           reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * static_cast<int64_t>(HIDDEN_DIM))),
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0]);

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE){
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_prob_G2S_buffer[token_stage][0]),
                                             reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE))),
                                             (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                             &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0]);

                    total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
                  }

                  if(current_src_token_id == last_src_token_id){
                    smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage] = true;
                  }
                  else{
                    smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage] = false;
                  }

                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0],
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if(token_stage == NUM_OF_STAGES_G2S){
                    token_stage = 0;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Device function for intra-node reduction warp group for combine kernel.
template<typename INTRA_NODE_RED_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         bool BACKWARD_COMBINE>
inline __device__ void intra_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 uint16_t* rdma_intra_node_red_token,
                                                                 float* rdma_intra_node_red_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  /*using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);*/

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be multiple of 2.");
  constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
  //static_assert((HIDDEN_DIM / 2) % INTRA_NODE_RED_GROUP::size() == 0, "HIDDEN_DIM / 2 must be multiple of INTRA_NODE_RED_GROUP::size(), we may relax this if it is the problem.");
  constexpr int NUM_OF_ELEMENT_PER_THREAD = ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Processing prob using fp32.
  constexpr int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  //static_assert(INTRA_NODE_RED_GROUP::size() >= NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE, "The size of intra-node reduction warp group must not be smaller than prob size.");

  // The intra node reduction warp group of each CUDA block produce a chunk at a time.
  // The chunk order is: first produce the same chunk id for all other nodes id, then produce following chunk id.
  // (i.e. chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, node + 2, ... node - 1)
  // The RDMA warp group of a CUDA block will consume the chunk by the same order. So each CUDA block will produce and consume the same set of chunks id.
  // The reason to distribute chunk in this order is that the inter-node reduction will need the same chunk id from all other nodes, so we need to produce and send chunks in this order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to produce for RDMA warps to consume.
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Src token stage id and phase.
  int token_stage = 0;
  uint32_t token_producer_parity = 0;

  // Dst token stage id.
  int dst_token_stage = 0;

  // Whether there are S2G TMA operations of a previous chunk's dst token in-flight(unfinished).
  bool outstanding_in_flight_chunk = false;

  // rdma_remote_node_id and chunk_id for previous chunk.
  int last_chunk_id;
  int last_rdma_remote_node_id;

  // Iterate through all chunks assigned to this block.
  for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    // Which node this chunk belongs to in output rdma reduction buffers.
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int rdma_intra_node_red_id = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    // How many rdma_to_attn load iter for this chunk.
    int num_of_routing_info_load_iter_for_current_chunk;
    // How many token for this chunk.
    int current_chunk_size;
    if(remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1){
      num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
      current_chunk_size = remainder_chunk_size;
    }else{
      num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }

    const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                      (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

    uint16_t* rdma_intra_node_red_token_base_ptr = rdma_intra_node_red_token + rdma_intra_node_red_id * static_cast<int64_t>(HIDDEN_DIM);
    float* rdma_intra_node_red_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE){
      rdma_intra_node_red_prob_base_ptr = rdma_intra_node_red_prob + rdma_intra_node_red_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
    }

    // How many dst token entry of current chunk have been in-flight.
    int additional_in_flight_s2g = 0;
    // Iterate through all dst tokens within this chunk.
    for(int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++){
      rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
      #pragma unroll
      for(int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++){
        // Check whether there is a previous chunk's dst token S2G in-flight and also current chunk already has NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token S2G in-flight.
        // If so, wait for previous chunk's S2G finish and notify the RDMA warp groups.
        if(outstanding_in_flight_chunk && (additional_in_flight_s2g == NUM_OF_ADDITIONAL_IN_FLIGHT_S2G)){
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              // Wait for previous chunk's S2G finish.
              cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<NUM_OF_ADDITIONAL_IN_FLIGHT_S2G>{});
              // Notify the rdma warp group.
              if constexpr(NUM_OF_NODES != 1){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
              }
            }
          }
          outstanding_in_flight_chunk = false;
        }
        int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
        // If the current token is out-of-bound, then just end this load iter.
        if(current_token_id >= current_chunk_size){
          break;
        }
        // Check whether this dst token is needed by this node. If not needed, just skip.
        bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
        // If this dst token is needed by this node, which means this dst token will have at least 1 src token within the shread memory.
        // Then, load the src token for this dst token from shared memory and accumulate it to the accumulator.
        if(token_needed_by_this_node){
          // Accumulator for this dst token. Token must be accumulated in FP32.
          float2 acc_token_fp32[NUM_OF_ELEMENT_PER_THREAD];
          // Optional Accumulator for this dst token prob.
          float acc_prob[NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
          // End reduction group flag.
          bool last_src_token = false;
          // Init accumulator.
          #pragma unroll
          for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
            acc_token_fp32[n].x = 0.0f;
            acc_token_fp32[n].y = 0.0f;
          }
          #pragma unroll
          for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
            acc_prob[n] = 0.0f;
          }

          // Continue loading src token for this dst token and reduce them to accumulator until all src token for this dst token have been accumulated.
          do{
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->intra_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->intra_node_prob_G2S_buffer[token_stage][0];
            }

            // Wait until current src token ready in shared memory.
            if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
              int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
              if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
                acc_token_fp32[n].x += src_data_fp32.x;
                acc_token_fp32[n].y += src_data_fp32.y;
              }   
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  float src_data = load_prob_base_ptr[element_id];
                  acc_prob[n] += src_data;
                }
              }
            }

            // Check flag for last src token.
            last_src_token = smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage];

            // Make sure all warp group have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
            if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == NUM_OF_STAGES_G2S){
              token_stage = 0;
              token_producer_parity ^= 1;
            }

          }while(!last_src_token);

          // Base address for current dst token and prob(optional) in shared memory.
          __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->intra_node_token_S2G_buffer[dst_token_stage][0]);
          float* store_prob_base_ptr;
          if constexpr(BACKWARD_COMBINE){
            store_prob_base_ptr = &smem_buffer_ptr->intra_node_prob_S2G_buffer[dst_token_stage][0];
          }

          // Let the TMA thread to wait for previously issued TMA S2G operations finish reading this entry.
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
            }
          }
          // Make sure all threads within the red warp group have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
          
          // Store the token.
          #pragma unroll
          for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
            int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
            if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
              // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
              store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
            }
          }

          // Store the prob(optional).
          if constexpr(BACKWARD_COMBINE){
            #pragma unroll
            for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
              int element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
              if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                store_prob_base_ptr[element_id] = acc_prob[n];
              }
            }
          }

          // Make sure the shared memory stored by current thread is visible by async proxy.
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

          // Make sure all threads within the red warp group have finished storing the current token entry and making it visible to async proxy.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

          // Let the TMA thread to issue S2G TMA operations for current token entry.
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              uint16_t* current_token_addr = rdma_intra_node_red_token_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * static_cast<int64_t>(HIDDEN_DIM);
              // Store the token from shared to global.
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_token_addr),
                                       reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_token_S2G_buffer[dst_token_stage][0]),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

              // Store the prob from shared to global(Optional).
              if constexpr(BACKWARD_COMBINE){
                float* current_prob_addr = rdma_intra_node_red_prob_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
                cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                         cuda::ptx::space_shared,
                                         reinterpret_cast<void*>(current_prob_addr),
                                         reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_prob_S2G_buffer[dst_token_stage][0]),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)));

              }
              // Commit S2G TMA operations for this dst token into a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
            }
          }

          // Goto next dst token entry.
          dst_token_stage += 1;
          if(dst_token_stage == NUM_OF_STAGES_S2G){
            dst_token_stage = 0;
          }

          // Another token entry's S2G in-flight.
          additional_in_flight_s2g += 1;
        }
      }
    }
    // If the current chunk does not have NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token entry in-flight, which is possible of rdma_to_attn map is really sparse.
    // We need to wait for both previous and current chunks' dst token entry S2G to finish and notify the RDMA warp group.
    if(outstanding_in_flight_chunk){
      if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
        if(elect_sync(~0)){
          // Wait for all previous chunk's(i.e. previous and current chunk) S2G finish.
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          // Notify the rdma warp group.
          if constexpr(NUM_OF_NODES != 1){
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[rdma_remote_node_id][chunk_id]);
          }
        }
      }
      outstanding_in_flight_chunk = false;
    }else{ // Otherwise, the current chunks is in-flight.
      outstanding_in_flight_chunk = true;
    }

    // Update last chunk's id.
    last_rdma_remote_node_id = rdma_remote_node_id;
    last_chunk_id = chunk_id;
  }

  // When all chunks have been processed, we need to check whether the last chunk is still in-flight.
  // If so, wait for it and notify RDMA warp group.
  if(outstanding_in_flight_chunk){
    if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
      if(elect_sync(~0)){
        // Wait for the last chunk's S2G finish.
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Notify the rdma warp group.
        if constexpr(NUM_OF_NODES != 1){
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
        }
      }
    }
  }
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifndef USE_NIXL
// Device function for inter-node node2node(RDMA) warp for combine kernel. There can be only 1 inter-node warp per CUDA block!
template<typename INTER_NODE_RDMA_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_N2N_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 doca_gpu_dev_verbs_qp **d_qps_gpu,
                                                                 struct combine_memory_region_info_t *mr_info,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  constexpr int WQE_NUM_RATIO = 1 + BACKWARD_COMBINE;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
  static_assert(INTER_NODE_RDMA_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(rdma_to_attn_map_load_t).");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Mr_infos and rdma_mbarrier_buffer in shared memory.
  struct combine_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  uint64_t (*intra_node_to_rdma_mbarrier_buffer_ptr)[MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK] = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    if (INTER_NODE_RDMA_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_mr_info_ptr[INTER_NODE_RDMA_GROUP::thread_rank()] = mr_info[INTER_NODE_RDMA_GROUP::thread_rank() + block_offset];
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_RDMA_GROUP::thread_rank()] = 0;
    }
    intra_node_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;
  }
  __syncwarp();
  // Total number of chunks to produce for RDMA warps to consume.
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int MAX_NUM_OF_CHUNKS_PER_RANK = (MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int TOTAL_NUM_OF_CHUNKS = (NUM_OF_NODES - 1) * NUM_OF_CHUNKS_PER_RANK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // INTRA_NODE_RED_GROUP should be 1 warp.
  // The inter_node_N2N_warp should process the same chunk as intra_node_red_warp(They belong to the same block.)
  uint32_t token_consumer_parity = 0;
  // Loop for every chunks.
  for(int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS){
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int rank_in_remote = rdma_remote_node_id < node_rank ? (node_rank - 1) : node_rank;
    int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    if (chunk_id * NUM_OF_TOKENS_PER_CHUNK + token_range > num_of_tokens_per_rank) {
      token_range = num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    }
    // Queue pair for the current block to the current remote.
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[rdma_remote_node_id + block_offset];
    // Try wait mbarrier.
    while(!cuda::ptx::mbarrier_try_wait_parity(&intra_node_to_rdma_mbarrier_buffer_ptr[rdma_remote_node_id][chunk_id], token_consumer_parity)){}
    // Calculating total num of tokens of the current chunk need to be sent.
    int num_of_tokens_need_write = 0;
    for (int token_idx_in_chunk = INTER_NODE_RDMA_GROUP::thread_rank();
         token_idx_in_chunk < token_range;
         token_idx_in_chunk += INTER_NODE_RDMA_GROUP::size()) {
      num_of_tokens_need_write += rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
    }
    num_of_tokens_need_write = __reduce_add_sync(0xffffffff, num_of_tokens_need_write);
    int total_write_cnt = num_of_tokens_need_write * WQE_NUM_RATIO + 1;
    // Getting wqe buffer.
    uint64_t base_wqe_idx = 0;
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, total_write_cnt);
      smem_inter_node_num_of_write_per_node_ptr[rdma_remote_node_id] += total_write_cnt;
    }
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
    uint64_t curr_wqe_idx = base_wqe_idx;
    // Processing one chunk.
    for (int token_idx_in_chunk = INTER_NODE_RDMA_GROUP::thread_rank();
         token_idx_in_chunk < NUM_OF_TOKENS_PER_CHUNK;
         token_idx_in_chunk += INTER_NODE_RDMA_GROUP::size()) {
      int token_idx = token_idx_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
      int local_token_idx = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + token_idx;
      bool need_write = false;
      if (token_idx_in_chunk < token_range) {
        need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
      }
      uint32_t write_map = __ballot_sync(0xffffffff, need_write);
      uint32_t partial_write_map = ((1 << INTER_NODE_RDMA_GROUP::thread_rank()) - 1) & write_map;
      int write_cnt = __popc(write_map);
      int write_idx = __popc(partial_write_map);
      if (need_write) {
        // Construct wqes for tokens
        uint64_t my_wqe_idx = curr_wqe_idx + write_idx;
        struct doca_gpu_dev_verbs_wqe *token_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
        doca_gpu_dev_verbs_wqe_prepare_write(qp, token_wqe_ptr, my_wqe_idx,
                                                  DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                  DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_raddr + token_idx * static_cast<int64_t>(HIDDEN_DIM) * sizeof(uint16_t),
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_rkey,
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_laddr + local_token_idx * static_cast<int64_t>(HIDDEN_DIM) * sizeof(uint16_t),
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_lkey,
                                                  HIDDEN_DIM * sizeof(uint16_t));
        if constexpr(BACKWARD_COMBINE) {
          my_wqe_idx += write_cnt;
          struct doca_gpu_dev_verbs_wqe *prob_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
          doca_gpu_dev_verbs_wqe_prepare_write(qp, prob_wqe_ptr, my_wqe_idx,
                                                    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_raddr + token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_rkey,
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_laddr + local_token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_lkey,
                                                    (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
        }
      }
      curr_wqe_idx += write_cnt * WQE_NUM_RATIO;
      __syncwarp();
    }
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      // Construct wqe for flag.
      struct doca_gpu_dev_verbs_wqe *flag_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, curr_wqe_idx);
      uint64_t offset_flag_laddr = smem_mr_info_ptr[rdma_remote_node_id].flag_laddr + rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK * sizeof(uint64_t);
      uint64_t offset_flag_raddr = smem_mr_info_ptr[rdma_remote_node_id].flag_raddr + rank_in_remote * MAX_NUM_OF_CHUNKS_PER_RANK * sizeof(uint64_t);
      doca_gpu_dev_verbs_wqe_prepare_atomic(qp, flag_wqe_ptr, curr_wqe_idx,
                                                 DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                                 DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                 offset_flag_raddr + chunk_id * sizeof(uint64_t),
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_rkey,
                                                 offset_flag_laddr + chunk_id * sizeof(uint64_t),
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_lkey,
                                                 sizeof(uint64_t), 1, 0);
      // Post send and poll cqs.
      doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, base_wqe_idx, curr_wqe_idx);
      doca_gpu_dev_verbs_submit_db<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                        DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                        DOCA_GPUNETIO_VERBS_QP_SQ>(qp, (curr_wqe_idx + 1));
    }
    __syncwarp();
  }
  if (INTER_NODE_RDMA_GROUP::thread_rank() < NUM_OF_NODES - 1) {
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[block_offset + INTER_NODE_RDMA_GROUP::thread_rank()];
    uint32_t wc_num_to_poll = smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_RDMA_GROUP::thread_rank()];
    if (wc_num_to_poll > 0) {
      int status = doca_gpu_dev_verbs_poll_cq<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                              DOCA_GPUNETIO_VERBS_QP_SQ>(
                                              doca_gpu_dev_verbs_qp_get_cq_sq(qp), wc_num_to_poll);
      assert(status >= 0);
    }
  }
  token_consumer_parity ^= 1;
}
#endif // USE_NIXL
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE

// Device function for inter-node G2S warp for combine kernel.
template<typename SMEM_TYPE,
         typename INTER_NODE_G2S_GROUP,
         int NUM_OF_STAGES_G2S, 
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_G2S_warp_group_device_function(const int node_rank,
                                                                 const int local_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const uint32_t* expected_intra_node_expert_input_chunk_flag_value,
                                                                 const uint64_t* expected_flag_value,
                                                                 const bool* rdma_to_attn_map,
                                                                 const bool* attn_to_rdma_map,
                                                                 const int32_t* sparse_to_dense_map, 
                                                                 uint16_t* const* remote_expert_input_token,
                                                                 float* const* remote_expert_input_prob,
                                                                 const uint16_t* rdma_inter_node_group_token,
                                                                 const float* rdma_inter_node_group_prob,
                                                                 uint32_t* intra_node_expert_input_chunk_flags,
                                                                 uint64_t* rdma_inter_node_group_flags,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // The warps from inter-node G2S warp group will be divided into multiple independent pipeline. 
  // Each pipeline can only have 1 warp, so INTER_NODE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK and warp has the same meaning as pipeline in inter-node G2S warp group.
  // Number of pipeline should match inter-node red warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
  // Evenly distribute the inter-node G2S FIFO to every pipeline(warp) within the inter-node G2S warp group.
  // When inter-node G2S warp group only has 1 warp, then the algorith is the same as old version(1 pipeline per CUDA block).
  static_assert(NUM_OF_STAGES_G2S % INTER_NODE_G2S_GROUP::warp_size() == 0, "NUM_OF_STAGES_G2S must be multiple of inter-node G2S warp group warp size.");
  constexpr int NUM_OF_STAGES_G2S_PER_WARP = NUM_OF_STAGES_G2S / INTER_NODE_G2S_GROUP::warp_size();
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks. 
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
  constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;
  
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
  // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
  // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1. 
  // So inter node reduction warp group will consume the src chunk in the same order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int max_num_of_chunks_per_rank = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
  const int total_num_of_chunks = num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Starting and ending index within G2S FIFO for this warp(pipeline).
  const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * INTER_NODE_G2S_GROUP::warp_rank();
  const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * (INTER_NODE_G2S_GROUP::warp_rank() + 1);
  // Token stage id and phase.
  int token_stage = starting_G2S_index;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within each inter-node G2S warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Iterate through all chunks. All chunks will assign to all CUDA block.
    for(int i = 0; i < total_num_of_chunks; i++){
      // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
      int num_of_token_groups_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }

      const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;

      const bool* attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1); 

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      // The base offset of the flags to be polled within the local unpermute flag buffer.
      int current_flag_base_offset = node_rank * NUM_OF_RANKS_PER_NODE * num_of_chunks_per_rank + i;
      // Array to hold the status of all src chunks from per-rank buffer(remote_expert_input buffer) for the current (dst) chunk, and init to false.
      // So for each src chunk, we only need to poll the flag once per inter_node_G2S thread(warp/pipeline) and record its status for later usage.
      bool remote_expert_input_chunk_flag_clear[NUM_OF_RANKS_PER_NODE];
      #pragma unroll
      for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
        remote_expert_input_chunk_flag_clear[j] = false;
      }
#endif
      // Padding from NUM_OF_NODES - 1 to NUM_OF_NODES in case NUM_OF_NODES = 1.
      // We still only use first NUM_OF_NODES - 1 flags, the last flag is the padding and not been used.
      bool rdma_flag_clear[NUM_OF_NODES];
      #pragma unroll
      for(int j = 0; j < NUM_OF_NODES; j++){
        rdma_flag_clear[j] = false;
      }

      // Iterate through all token groups within this chunk which assign to this CUDA block.
      for(int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS){
        // Iterate through all dst(output) tokens within this token group.
        // Assign each dst token to each G2S warp(pipeline) using a round-robin fasion.
        for(int k = INTER_NODE_G2S_GROUP::warp_rank(); k < NUM_OF_TOKENS_PER_GROUP; k += INTER_NODE_G2S_GROUP::warp_size()){
          int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
          // If the current token is out-of-bound, then just end this load iter.
          if(current_token_id >= current_chunk_size){
            break;
          }
          // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
          // Accumulate local tokens first, then rdma tokens.

          // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
          bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
          // If this dst token is needed by this node, load the sparse_to_dense map and load the local src token for this dst token.
          if(token_needed_by_this_node){
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * NUM_OF_RANKS_PER_NODE);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            sparse_to_dense_map_load_t sparse_to_dense_map_data[NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = 0;
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  last_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                }
              }
            }
            // Then issue all G2S TMA for this row.
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
                  // If the current src chunk in the target per-rank buffer is not ready yet, wait for the src chunk ready first.
                  if(remote_expert_input_chunk_flag_clear[current_src_token_id] == false){
                    const uint32_t* flag_location = intra_node_expert_input_chunk_flags + (current_flag_base_offset + current_src_token_id * num_of_chunks_per_rank);
                    uint32_t intra_node_chunk_flag = 0;
                    do{
                      intra_node_chunk_flag = 0;
                      // Need a strong system-scope load to observe peer ranks' Atomic result.
                      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                                   : "=r"(intra_node_chunk_flag)
                                   : "l"(__cvta_generic_to_global(flag_location))
                                   : "memory");
                    }while(intra_node_chunk_flag != *expected_intra_node_expert_input_chunk_flag_value);
            
                    // Mark the src chunk from this rank is already clear.
                    remote_expert_input_chunk_flag_clear[current_src_token_id] = true;
                  }
#endif
                  // Wait until current token entry within the shared memory has been consumed.
                  while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]),
                                           reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * static_cast<int64_t>(HIDDEN_DIM))),
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE){
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0]),
                                             reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE))),
                                             (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                             &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                    total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
                  }

                  if(current_src_token_id == last_src_token_id){
                    smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage] = true;
                  }
                  else{
                    smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage] = false;
                  }

                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0],
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if(token_stage == ending_G2S_index){
                    token_stage = starting_G2S_index;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
          // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
          const bool* attn_to_rdma_map_load_addr = attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
          #pragma unroll
          for (int n = 1; n < NUM_OF_NODES; n++) {
            // The current node been processed. For each chunk id, node_id order is 
            // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
            int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
            // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
            int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
            // Check wether current dst token need src token from this node.
            if(attn_to_rdma_map_load_addr[rdma_buffer_tile_id]){
              // If the current chunk is not ready yet, wait for related rdma inter-node group buffer chunks ready first.
              if(rdma_flag_clear[n - 1] == false){
                const uint64_t* flag_location = rdma_inter_node_group_flags + (rdma_buffer_tile_id * max_num_of_chunks_per_rank + i);
                uint64_t rdma_flag = 0;
                do{
                  rdma_flag = 0;
                  // Need a strong system-scope load to observe external RDMA Atomic result.
                  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                              : "=l"(rdma_flag)
                              : "l"(__cvta_generic_to_global(flag_location))
                              : "memory");
                }while(rdma_flag != *expected_flag_value);
                // Mark the chunk from this node(tile) is already clear.
                rdma_flag_clear[n - 1] = true;
              }
              // Wait until current token entry within the shared memory has been consumed.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}
              // Load the src token from this rdma inter-node group buffer chunk to shared memory entry.
              uint32_t total_tx_size = 0;
              const uint16_t* rdma_inter_node_group_token_load_addr = rdma_inter_node_group_token + 
                                                                      (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + 
                                                                      i * NUM_OF_TOKENS_PER_CHUNK + 
                                                                      j * NUM_OF_TOKENS_PER_GROUP + k) * static_cast<int64_t>(HIDDEN_DIM);
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]),
                                       reinterpret_cast<const void*>(rdma_inter_node_group_token_load_addr),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                       &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

              if constexpr(BACKWARD_COMBINE){
                const float* rdma_inter_node_group_prob_load_addr = rdma_inter_node_group_prob + 
                                                                    (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + 
                                                                    i * NUM_OF_TOKENS_PER_CHUNK + 
                                                                    j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);

                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0]),
                                         reinterpret_cast<const void*>(rdma_inter_node_group_prob_load_addr),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                         &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
              }

              // Inter-node token does not need flag since the red warp group will also read attn_to_rdma_map.

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0],
                                                   total_tx_size);
              // Goto next token entry in shared memory.
              token_stage += 1;
              if(token_stage == ending_G2S_index){
                token_stage = starting_G2S_index;
                token_consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // Update residue flags.
  int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
  for (int node_id = blockIdx.x; node_id < NUM_OF_NODES - 1; node_id += gridDim.x) {
    uint64_t *residue_flag_base_ptr = rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
    for (int flag_id = INTER_NODE_G2S_GROUP::thread_rank(); flag_id < residue_flag_count; flag_id += INTER_NODE_G2S_GROUP::size()) {
      residue_flag_base_ptr[flag_id] = *expected_flag_value;
    }
  }
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE
}

// Device function for inter-node reduction warp group for combine kernel.
template<typename SMEM_TYPE,
         typename INTER_NODE_RED_GROUP,
         int NUM_OF_DATA_PIPELINE_PER_BLOCK,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 const bool* attn_to_rdma_map, 
                                                                 uint16_t* attn_output_token,
                                                                 float* attn_output_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // The warps from inter-node red warp group will be divided into multiple independent pipeline. Each pipeline has INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK warps.
  // Number of pipeline should match inter-node G2S warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
  static_assert(INTER_NODE_RED_GROUP::warp_size() % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "The warp count of inter-node red warp group must be multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_THREADS_PER_PIPELINE = (INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK) * WARP_SIZE;
  // Evenly distribute the inter-node G2S FIFO to every pipeline within the inter-node red warp group.
  // When NUM_OF_DATA_PIPELINE_PER_BLOCK = 1 and INTER_NODE_RED_GROUP::warp_size() = 4, then the algorith is the same as old version(1 pipeline w/ 4 warps per CUDA block).
  static_assert(NUM_OF_STAGES_G2S % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_G2S must be multiple of data pipeline per CUDA block.");
  constexpr int NUM_OF_STAGES_G2S_PER_PIPELINE = NUM_OF_STAGES_G2S / NUM_OF_DATA_PIPELINE_PER_BLOCK;
  // Evenly distribute the inter-node S2G FIFO to every pipeline within the inter-node red warp group.
  static_assert(NUM_OF_STAGES_S2G % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_S2G must be multiple of data pipeline per CUDA block.");
  constexpr int NUM_OF_STAGES_S2G_PER_PIPELINE = NUM_OF_STAGES_S2G / NUM_OF_DATA_PIPELINE_PER_BLOCK;
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks. 
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
  constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;

  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be multiple of 2.");
  constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
  //static_assert((HIDDEN_DIM / 2) % NUM_OF_THREADS_PER_PIPELINE == 0, "HIDDEN_DIM / 2 must be multiple of NUM_OF_THREADS_PER_PIPELINE, we may relax this if it is the problem.");
  constexpr int NUM_OF_ELEMENT_PER_THREAD = ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;
  // Processing prob using fp32.
  constexpr int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;
  //static_assert(INTER_NODE_RED_GROUP::size() >= NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE, "The size of inter-node reduction warp group must not be smaller than prob size.");

  // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
  // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
  // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1. 
  // So inter node reduction warp group will consume the src chunk in the same order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
  const int total_num_of_chunks = num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Pipeline rank and thread/warp rank within the pipeline for this thread.
  const int pipeline_rank = INTER_NODE_RED_GROUP::thread_rank() / NUM_OF_THREADS_PER_PIPELINE;
  const int thread_rank_within_pipeline = INTER_NODE_RED_GROUP::thread_rank() % NUM_OF_THREADS_PER_PIPELINE;
  const int warp_rank_within_pipeline = thread_rank_within_pipeline / WARP_SIZE;
  // Starting and ending index within G2S FIFO for this pipeline.
  const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * pipeline_rank;
  const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * (pipeline_rank + 1);
  // Src token stage id and phase.
  int token_stage = starting_G2S_index;
  uint32_t token_producer_parity = 0;

  // Starting and ending index within S2G FIFO for this pipeline.
  const int starting_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * pipeline_rank;
  const int ending_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * (pipeline_rank + 1);
  // Dst token stage id.
  int dst_token_stage = starting_S2G_index;

  // Iterate through all chunks. All chunks will assign to all CUDA block.
  for(int i = 0; i < total_num_of_chunks; i++){
    // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
    int num_of_token_groups_for_current_chunk;
    // How many token for this chunk.
    int current_chunk_size;
    if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
      num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
      current_chunk_size = remainder_chunk_size;
    }else{
      num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }

    const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
    const bool* attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1);
    uint16_t* attn_output_token_base_ptr = attn_output_token + (i * NUM_OF_TOKENS_PER_CHUNK) * static_cast<int64_t>(HIDDEN_DIM);
    float* attn_output_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE){
      attn_output_prob_base_ptr = attn_output_prob + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
    }
    // Iterate through all token groups within this chunk which assign to this CUDA block.
    for(int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS){
      // Iterate through all dst(output) tokens within this token group.
      // Assign each dst token to each pipeline using a round-robin fasion.
      for(int k = pipeline_rank; k < NUM_OF_TOKENS_PER_GROUP; k += NUM_OF_DATA_PIPELINE_PER_BLOCK){
        int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
        // If the current token is out-of-bound, then just end this load iter.
        if(current_token_id >= current_chunk_size){
          break;
        }
        // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
        // Accumulate local tokens first, then rdma tokens.
        // Accumulator for this dst token. Token must be accumulated in FP32.
        float2 acc_token_fp32[NUM_OF_ELEMENT_PER_THREAD];
        // Optional Accumulator for this dst token prob.
        // Different node's prob need to be gathered together to output.
        // 0 used for local node's prob, [1, NUM_OF_NODES - 1] used for remote node's prob.
        float acc_prob[NUM_OF_NODES][NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
        // Init accumulator.
        #pragma unroll
        for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
          acc_token_fp32[n].x = 0.0f;
          acc_token_fp32[n].y = 0.0f;
        }
        #pragma unroll
        for(int n = 0; n < NUM_OF_NODES; n++){
          #pragma unroll
          for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
            acc_prob[n][m] = 0.0f;
          }
        }

        // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
        bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
        // If this dst token is needed by this node, load the local src token from shared memory and accumulate them.
        if(token_needed_by_this_node){
          // End reduction group flag.
          bool last_local_node_src_token = false;
          
          // Continue loading local src token for this dst token and reduce them to accumulator until all local src token for this dst token have been accumulated.
          do{
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0];
            }

            // Wait until current src token ready in shared memory.
            if(warp_rank_within_pipeline == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
              int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
              if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
                acc_token_fp32[n].x += src_data_fp32.x;
                acc_token_fp32[n].y += src_data_fp32.y;
              }     
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int element_id = thread_rank_within_pipeline + n * NUM_OF_THREADS_PER_PIPELINE;
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  float src_data = load_prob_base_ptr[element_id];
                  acc_prob[0][n] += src_data;
                }
              }
            }

            // Check flag for last src token.
            last_local_node_src_token = smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage];

            // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
            if(warp_rank_within_pipeline == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == ending_G2S_index){
              token_stage = starting_G2S_index;
              token_producer_parity ^= 1;
            }

          }while(!last_local_node_src_token);
        }

        // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
        const bool* attn_to_rdma_map_load_addr = attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
        #pragma unroll
        for(int n = 1; n < NUM_OF_NODES; n++){
          // The current node been processed. For each chunk id, node_id order is 
          // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
          int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
          // The tile id within the rdma buffers(include attn_to_rdma map) for the current node id. Because these rdma buffers only have NUM_OF_NODES - 1 tile or element.
          int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
          // Check wether current dst token need src token from this (remote) node.
          if(attn_to_rdma_map_load_addr[rdma_buffer_tile_id]){
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0];
            }
            // Wait until current src token ready in shared memory.
            if(warp_rank_within_pipeline == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int m = 0; m < NUM_OF_ELEMENT_PER_THREAD; m++){
              int element_id = (m * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
              if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
                acc_token_fp32[m].x += src_data_fp32.x;
                acc_token_fp32[m].y += src_data_fp32.y;
              }     
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
                int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  acc_prob[n][m] = load_prob_base_ptr[element_id];
                }
              }
            }

            // Inter-node token does not need flag.

            // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
            if(warp_rank_within_pipeline == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == ending_G2S_index){
              token_stage = starting_G2S_index;
              token_producer_parity ^= 1;
            }
          }
        }

        // Store the dst token back to share memory. 
        // Because each attn token must have go to TOPK rank in dispatch, so it must have been reduced in combine. So each attn dst token must be written back.
        // Base address for current dst token and prob(optional) in shared memory.
        __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_S2G_buffer[dst_token_stage][0]);
        float* store_prob_base_ptr;
        if constexpr(BACKWARD_COMBINE){
          store_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_S2G_buffer[dst_token_stage][0];
        }

        // Select the TMA thread within the pipeline to wait for previously issued TMA S2G operations finish reading this entry.
        if(warp_rank_within_pipeline == 0){
          if(elect_sync(~0)){
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G_PER_PIPELINE - 1>{});
          }
        }
        // Make sure all threads within the pipeline have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
          
        // Store the token.
        #pragma unroll
        for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
          int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
          if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
            // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
            store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
          }
        }

        // Store the prob(optional).
        if constexpr(BACKWARD_COMBINE){
          #pragma unroll
          for(int n = 0; n < NUM_OF_NODES; n++){
            int attn_prob_output_node_id = (node_rank - n) >= 0 ? node_rank - n : node_rank + NUM_OF_NODES - n;
            int element_base_id = attn_prob_output_node_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
            #pragma unroll
            for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
              int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
              if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                store_prob_base_ptr[element_base_id + element_id] = acc_prob[n][m];
              }
            }
          }
        }

        // Make sure the shared memory stored by current thread is visible by async proxy.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

        // Make sure all threads within the pipeline have finished storing the current token entry and making it visible to async proxy.
        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

        // Select the TMA thread within the pipeline to issue S2G TMA operations for current token entry.
        if(warp_rank_within_pipeline == 0){
          if(elect_sync(~0)){
            uint16_t* current_token_addr = attn_output_token_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * static_cast<int64_t>(HIDDEN_DIM);
            // Store the token from shared to global output.
            cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                     cuda::ptx::space_shared,
                                     reinterpret_cast<void*>(current_token_addr),
                                     reinterpret_cast<const void*>(&smem_buffer_ptr->inter_node_token_S2G_buffer[dst_token_stage][0]),
                                     (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

            // Store the prob from shared to global output.
            if constexpr(BACKWARD_COMBINE){
              float* current_prob_addr = attn_output_prob_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_prob_addr),
                                       reinterpret_cast<const void*>(&smem_buffer_ptr->inter_node_prob_S2G_buffer[dst_token_stage][0]),
                                       (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) * sizeof(float)));

            }
            // Commit S2G TMA operations for this dst token into a bulk async copy group.
            cuda::ptx::cp_async_bulk_commit_group();
          }
        }

        // Goto next dst token entry.
        dst_token_stage += 1;
        if(dst_token_stage == ending_S2G_index){
          dst_token_stage = starting_S2G_index;
        }
      }
    }
  }
  // Because the attn output buffers will only be produced by local combine kernel, not by the combine kernels on other ranks,
  // so we only need to wait for local combine kernel to finish writing all token data back to output buffer before we can exit.
  // Also, a kernel will be considered completed from CUDA stream's perspective if and only if all the threads are exit and all memory operations(including TMA operations)
  // issued by all threads have been completed and made visible to sys scope.
  // So the CUDA stream's kernel boundary implicit synchronization should be enough to sync with all TMA operations issued in the combine kernel.
  // So we can directly exit w/o any explicit synchronization with TMA operations.
}

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
// Device function for unpermute G2S warp for combine kernel.
template<typename SMEM_TYPE,
         typename UNPERMUTE_G2S_GROUP,
         int NUM_OF_STAGES_G2S, 
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_UNPERMUTE_BLOCKS,
         int NUM_OF_COMBINE_BLOCKS, // In current design, each combine grid has N combine blocks + M unpermute blocks, with combine block come first to minimize code change.
         bool BACKWARD_COMBINE>
inline __device__ void unpermute_G2S_warp_group_device_function(const int node_rank,
                                                                const int num_of_tokens_per_rank,
                                                                const uint32_t* expected_flag_value,
                                                                const int32_t* dense_chunk_layout,
                                                                const int32_t* dense_to_expert_map,
                                                                const uint16_t* local_expert_input_token,
                                                                const float* local_expert_input_prob,
                                                                uint32_t* intra_node_expert_input_chunk_flags,
                                                                SMEM_TYPE* smem_buffer_ptr)
{
  // The warps from unpermute G2S warp group will be divided into multiple independent pipeline. 
  // Each pipeline can only have 1 warp, so UNPERMUTE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_UNPERMUTE_BLOCK and warp has the same meaning as pipeline in unpermute G2S warp group.
  // Number of pipeline should match unpermute red warp group, so they can coupled into multiple independent data pipeline within a unpermute CUDA block.
  // Evenly distribute the unpermute G2S FIFO to every pipeline(warp) within the unpermute G2S warp group.
  static_assert(NUM_OF_STAGES_G2S % UNPERMUTE_G2S_GROUP::warp_size() == 0, "NUM_OF_STAGES_G2S must be multiple of unpermute G2S warp group warp size.");
  constexpr int NUM_OF_STAGES_G2S_PER_WARP = NUM_OF_STAGES_G2S / UNPERMUTE_G2S_GROUP::warp_size();

  // Load dense_to_expert_map according to the NUM_OF_EXPERTS_PER_RANK.
  using dense_to_expert_map_load_t = Copy_t<NUM_OF_EXPERTS_PER_RANK * sizeof(int32_t)>;
  constexpr int NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_EXPERTS_PER_RANK * sizeof(int32_t)) / sizeof(dense_to_expert_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(dense_to_expert_map_load_t) / sizeof(int32_t);

  // The unpermute blocks will produce the token chunks within the local rank's per-rank buffer(remote_expert_input buffers on local rank) in this order:
  // chunk 0 for all ranks(0 -> NUM_OF_RANKS_PER_NODE - 1) on node + 1, chunk 0 for all ranks on node + 2, ......, chunk 0 for all ranks on local node,......
  // chunk 1 for all ranks on node + 1, chunk 1 for all ranks on node + 2, ......, chunk 1 for all ranks on local node,......
  // So the chunk order is local_rank_id(0 -> NUM_OF_RANKS_PER_NODE - 1) -> node_id(node + 1 -> local_node) -> chunk_id(0 -> num_of_chunks_per_rank - 1).
  // We assign chunks in the previous order to each unpermute blocks.

  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(per-rank buffer).
  const int total_num_of_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  // Starting and ending index within G2S FIFO for this warp(pipeline).
  const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * UNPERMUTE_G2S_GROUP::warp_rank();
  const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * (UNPERMUTE_G2S_GROUP::warp_rank() + 1);
  // Token stage id and phase.
  int token_stage = starting_G2S_index;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within each unpermute G2S warp will be active, other threads will just exit if no residue flag need to updated.
  if(elect_sync(~0)){
    // Iterate through all chunks. Data(chunk) parallel between multiple unpermute CUDA blocks.
    // We flatten the global chunk id of all attn chunks.
    // Need to take the combine block's offset into account.
    for(int i = blockIdx.x - NUM_OF_COMBINE_BLOCKS; i < total_num_of_chunks; i += NUM_OF_UNPERMUTE_BLOCKS){
      // Calculate which node, rank and chunk does this global chunk id map to.
      // The chunk id of the current chunk within the rank.
      int current_chunk_id = i / (NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
      // The rank id of the current chunk within the node.
      int current_rank_id = i % NUM_OF_RANKS_PER_NODE;
      // The node id of current chunk.
      int current_node_linear_id = (i / NUM_OF_RANKS_PER_NODE) % NUM_OF_NODES;
      int current_node_id = (current_node_linear_id + (node_rank + 1)) % NUM_OF_NODES;

      // Calculate the chunk id of the current chunk within the per-rank buffer(i.e. the remote_expert_input buffers on local rank) according to the node_id, rank_id and chunk_id.
      int current_global_id = current_node_id * num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE + current_rank_id * num_of_chunks_per_rank + current_chunk_id;

      // Load the chunk layout info for current chunk from dense_chunk_layout map, and calculate the starting token and number of tokens of the current chunk within the local per-rank buffer.
      // Per-rank buffer is a dense buffer, every token within this buffer is needed by this rank(which means every token within this buffer is needed by at least 1 local expert).
      // So, every token within this buffer need to be reduce from at least 1 local expert, which means every token within this buffer need to occupy at least 1 smem G2S FIFO entry.
      int next_chunk_starting_location_within_expert_input_buffer = dense_chunk_layout[current_global_id];
      int current_chunk_starting_location_within_expert_input_buffer = 0;
      if(current_global_id != 0){
        current_chunk_starting_location_within_expert_input_buffer = dense_chunk_layout[current_global_id - 1];
      }
      int num_of_tokens_for_current_chunk = next_chunk_starting_location_within_expert_input_buffer - current_chunk_starting_location_within_expert_input_buffer;

      const int32_t* dense_to_expert_map_load_base_addr = dense_to_expert_map + current_chunk_starting_location_within_expert_input_buffer * NUM_OF_EXPERTS_PER_RANK;

      // Iterate through all dst(output) tokens within this chunk in the per-rank buffer by collecting all src tokens from local expert input buffer for each of them.
      // Assign each dst token to each G2S warp(pipeline) using a round-robin fasion.
      for(int j = UNPERMUTE_G2S_GROUP::warp_rank(); j < num_of_tokens_for_current_chunk; j += UNPERMUTE_G2S_GROUP::warp_size()){
        const dense_to_expert_map_load_t* dense_to_expert_map_load_addr = reinterpret_cast<const dense_to_expert_map_load_t*>(dense_to_expert_map_load_base_addr + j * NUM_OF_EXPERTS_PER_RANK);
        // Load dense_to_expert map for this dst token(i.e. a row in dense_to_expert map).
        dense_to_expert_map_load_t dense_to_expert_map_data[NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
        // First load dense_to_expert map and decide the last src token within this row.
        int last_src_token_id = -1;
        #pragma unroll
        for(int k = 0; k < NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; k++){
          dense_to_expert_map_data[k] = dense_to_expert_map_load_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; n++){
            int32_t dense_to_expert_map_value = *(reinterpret_cast<int32_t*>(&dense_to_expert_map_data[k]) + n);
            if(dense_to_expert_map_value != -1){
              last_src_token_id = k * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + n;
            }
          }
        }
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
        // This dst token has no src token. This will only happen when token drop is triggered and all src tokens of this dst token have been dropped.
        // To make unpermute_red_warp_group_device_function work, we need to produce a single dummy src token entry with garbage value for this dst token.
        // The dummy src token entry has all its data(token, prob, expert_id) left unset as garbage value and the end group flag set as true.
        if(last_src_token_id == -1){
          // Wait until current token entry within the shared memory has been consumed.
          while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

          smem_buffer_ptr->unpermute_flag_G2S_buffer[token_stage] = true;
          // Directly mark the producer to consumer mbarrier clear for this src token entry to let unpermute_red_warp_group_device_function consume this dummy src token entry.
          if constexpr(BACKWARD_COMBINE){
            // When BACKWARD_COMBINE is true, arrive twice as the mbarrier will be init to 2. Otherwise, arrive once.
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0]);
          }
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0]);

          // Goto next token entry in shared memory.
          token_stage += 1;
          if(token_stage == ending_G2S_index){
            token_stage = starting_G2S_index;
            token_consumer_parity ^= 1;
          }
        }
        // This dst token has at least 1 src token. This will happen if no token drop happens or not all src tokens have been dropped for this dst token.
        // Just go through normal code path.
        else{
#endif
        // Then issue all G2S TMA/LDGSTS for this row(dst token).
        #pragma unroll
        for(int k = 0; k < NUM_OF_DENSE_TO_EXPERT_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; k++){
          #pragma unroll
          for(int n = 0; n < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; n++){
            int32_t dense_to_expert_map_value = *(reinterpret_cast<int32_t*>(&dense_to_expert_map_data[k]) + n);
            if(dense_to_expert_map_value != -1){
              int current_src_token_id = k * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + n;
              // Wait until current token entry within the shared memory has been consumed.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

              uint32_t total_tx_size = 0;
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(&smem_buffer_ptr->unpermute_token_G2S_buffer[token_stage][0]),
                                       reinterpret_cast<const void*>(local_expert_input_token + (dense_to_expert_map_value * static_cast<int64_t>(HIDDEN_DIM))),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                       &smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0]);

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

              if constexpr(BACKWARD_COMBINE){
                // Store the local expert id of this src token to the G2S entry for unpermute red warp group to use if BW combine.
                smem_buffer_ptr->unpermute_local_expert_id_G2S_buffer[token_stage] = current_src_token_id;
                // Each src token within the local expert input buffer only has 1 prob element(a single float element), so cannot use TMA to transfer the prob element for this src token.
                /*cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->unpermute_prob_G2S_buffer[token_stage]),
                                         reinterpret_cast<const void*>(local_expert_input_prob + dense_to_expert_map_value),
                                         (uint32_t)(sizeof(float)),
                                         &smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0]);

                total_tx_size += (uint32_t)(sizeof(float));*/
                // Use LDGSTS instead of TMA.
                __pipeline_memcpy_async(reinterpret_cast<void*>(&smem_buffer_ptr->unpermute_prob_G2S_buffer[token_stage]),
                                        reinterpret_cast<const void*>(local_expert_input_prob + dense_to_expert_map_value),
                                        sizeof(float));
                // Track the completion of the LDGSTS instruction to the mbarrier.
                // We use the noinc version, so when BACKWARD_COMBINE is true, the mbarrier need to be init to 2 instead 1.
                cuda::ptx::cp_async_mbarrier_arrive_noinc(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0]);
              }

              if(current_src_token_id == last_src_token_id){
                smem_buffer_ptr->unpermute_flag_G2S_buffer[token_stage] = true;
              }
              else{
                smem_buffer_ptr->unpermute_flag_G2S_buffer[token_stage] = false;
              }

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   &smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0],
                                                   total_tx_size);

              // Goto next token entry in shared memory.
              token_stage += 1;
              if(token_stage == ending_G2S_index){
                token_stage = starting_G2S_index;
                token_consumer_parity ^= 1;
              }
            }
          }
        }
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
        }
#endif
      }
    }
  }
  // Update residue flags in intra_node_expert_input_chunk_flags. Write-and-forget operations.
  int residue_flag_count = (MAX_NUM_OF_CHUNKS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) - total_num_of_chunks;
  // The residue flags will be updated by all threads of the unpermute_G2S warp group of all unpermute CUDA block. Calculate how many threads to perform this update oeprations.
  constexpr int RESIDUE_FLAG_UPDATE_THREAD_COUNT = UNPERMUTE_G2S_GROUP::size() * NUM_OF_UNPERMUTE_BLOCKS;
  int residue_flag_update_thread_id = UNPERMUTE_G2S_GROUP::thread_rank() + (blockIdx.x - NUM_OF_COMBINE_BLOCKS) * UNPERMUTE_G2S_GROUP::size();
  for(int i = residue_flag_update_thread_id; i < residue_flag_count; i += RESIDUE_FLAG_UPDATE_THREAD_COUNT){
    intra_node_expert_input_chunk_flags[total_num_of_chunks + i] = *expected_flag_value;
  }
}

// Device function for unpermute reduction warp group for combine kernel.
template<typename SMEM_TYPE,
         typename UNPERMUTE_RED_GROUP,
         int NUM_OF_DATA_PIPELINE_PER_BLOCK,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_UNPERMUTE_BLOCKS,
         int NUM_OF_COMBINE_BLOCKS, // In current design, each combine grid has N combine blocks + M unpermute blocks, with combine block come first to minimize code change.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         bool BACKWARD_COMBINE>
inline __device__ void unpermute_red_warp_group_device_function(const int node_rank,
                                                                const int local_rank,
                                                                const int num_of_tokens_per_rank,
                                                                const int32_t* dense_chunk_layout, 
                                                                uint16_t* remote_expert_input_token,
                                                                float* remote_expert_input_prob,
                                                                uint32_t* const* intra_node_expert_input_chunk_flags,
                                                                SMEM_TYPE* smem_buffer_ptr)
{
  // The warps from unpermute red warp group will be divided into multiple independent pipeline. Each pipeline has UNPERMUTE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK warps.
  // Number of pipeline should match unpermute G2S warp group, so they can coupled into multiple independent data pipeline within a unpermute CUDA block.
  static_assert(UNPERMUTE_RED_GROUP::warp_size() % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "The warp count of unpermute red warp group must be multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_THREADS_PER_PIPELINE = (UNPERMUTE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK) * WARP_SIZE;
  // Evenly distribute the unpermute G2S FIFO to every pipeline within the unpermute red warp group.
  static_assert(NUM_OF_STAGES_G2S % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_G2S must be multiple of data pipeline per unpermute CUDA block.");
  constexpr int NUM_OF_STAGES_G2S_PER_PIPELINE = NUM_OF_STAGES_G2S / NUM_OF_DATA_PIPELINE_PER_BLOCK;
  // Evenly distribute the unpermute S2G FIFO to every pipeline within the unpermute red warp group.
  static_assert(NUM_OF_STAGES_S2G % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_S2G must be multiple of data pipeline per unpermute CUDA block.");
  constexpr int NUM_OF_STAGES_S2G_PER_PIPELINE = NUM_OF_STAGES_S2G / NUM_OF_DATA_PIPELINE_PER_BLOCK;

  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be multiple of 2.");
  constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
  constexpr int NUM_OF_ELEMENT_PER_THREAD = ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;
  // Processing prob using fp32.
  constexpr int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

  // The unpermute blocks will produce the token chunks within the local rank's per-rank buffer(remote_expert_input buffers on local rank) in this order:
  // chunk 0 for all ranks(0 -> NUM_OF_RANKS_PER_NODE - 1) on node + 1, chunk 0 for all ranks on node + 2, ......, chunk 0 for all ranks on local node,......
  // chunk 1 for all ranks on node + 1, chunk 1 for all ranks on node + 2, ......, chunk 1 for all ranks on local node,......
  // So the chunk order is local_rank_id(0 -> NUM_OF_RANKS_PER_NODE - 1) -> node_id(node + 1 -> local_node) -> chunk_id(0 -> num_of_chunks_per_rank - 1).
  // We assign chunks in the previous order to each unpermute blocks.

  // expert id offset of current local rank within the node.
  const int expert_id_offset = local_rank * NUM_OF_EXPERTS_PER_RANK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(per-rank buffer).
  const int total_num_of_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  // Pipeline rank and thread/warp rank within the pipeline for this thread.
  const int pipeline_rank = UNPERMUTE_RED_GROUP::thread_rank() / NUM_OF_THREADS_PER_PIPELINE;
  const int thread_rank_within_pipeline = UNPERMUTE_RED_GROUP::thread_rank() % NUM_OF_THREADS_PER_PIPELINE;
  const int warp_rank_within_pipeline = thread_rank_within_pipeline / WARP_SIZE;
  // Starting and ending index within G2S FIFO for this pipeline.
  const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * pipeline_rank;
  const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * (pipeline_rank + 1);
  // Src token stage id and phase.
  int token_stage = starting_G2S_index;
  uint32_t token_producer_parity = 0;

  // Starting and ending index within S2G FIFO for this pipeline.
  const int starting_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * pipeline_rank;
  const int ending_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * (pipeline_rank + 1);
  // Dst token stage id.
  int dst_token_stage = starting_S2G_index;

  // Whether there are S2G TMA operations of a previous chunk's token entry in-flight(unfinished).
  bool outstanding_in_flight_chunk = false;
  // Flag location within the unpermute flag buffer for previous chunk. Used for updating flags.
  int last_chunk_global_chunk_id;
  // The rank id of the previous chunk within the node. Used for updating flags.
  int last_chunk_rank_id;

  // Iterate through all chunks. Data(chunk) parallel between multiple unpermute CUDA blocks.
  // We flatten the global chunk id of all attn chunks.
  // Need to take the combine block's offset into account.
  for(int i = blockIdx.x - NUM_OF_COMBINE_BLOCKS; i < total_num_of_chunks; i += NUM_OF_UNPERMUTE_BLOCKS){
    // Calculate which node, rank and chunk does this global chunk id map to.
    // The chunk id of the current chunk within the rank.
    int current_chunk_id = i / (NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
    // The rank id of the current chunk within the node.
    int current_rank_id = i % NUM_OF_RANKS_PER_NODE;
    // The node id of current chunk.
    int current_node_linear_id = (i / NUM_OF_RANKS_PER_NODE) % NUM_OF_NODES;
    int current_node_id = (current_node_linear_id + (node_rank + 1)) % NUM_OF_NODES;

    // Calculate the chunk id of the current chunk within the per-rank buffer(i.e. the remote_expert_input buffers on local rank) according to the node_id, rank_id and chunk_id.
    int current_global_id = current_node_id * num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE + current_rank_id * num_of_chunks_per_rank + current_chunk_id;
    // Calculate the flag id to be notified of the current chunk within the unpermute flag buffer according to the node_id, local_rank and chunk_id.
    int current_flag_id = (current_node_id * NUM_OF_RANKS_PER_NODE + local_rank) * num_of_chunks_per_rank + current_chunk_id;

    // Load the chunk layout info for current chunk from dense_chunk_layout map, and calculate the starting token and number of tokens of the current chunk within the local per-rank buffer.
    // Per-rank buffer is a dense buffer, every token within this buffer is needed by this rank(which means every token within this buffer is needed by at least 1 local expert).
    // So, every token within this buffer need to be reduce from at least 1 local expert, which means every token within this buffer need to occupy 1 smem S2G FIFO entry.
    int next_chunk_starting_location_within_expert_input_buffer = dense_chunk_layout[current_global_id];
    int current_chunk_starting_location_within_expert_input_buffer = 0;
    if(current_global_id != 0){
      current_chunk_starting_location_within_expert_input_buffer = dense_chunk_layout[current_global_id - 1];
    }
    int num_of_tokens_for_current_chunk = next_chunk_starting_location_within_expert_input_buffer - current_chunk_starting_location_within_expert_input_buffer;


    uint16_t* remote_expert_input_token_base_ptr = remote_expert_input_token + current_chunk_starting_location_within_expert_input_buffer * static_cast<int64_t>(HIDDEN_DIM);
    float* remote_expert_input_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE){
      remote_expert_input_prob_base_ptr = remote_expert_input_prob + current_chunk_starting_location_within_expert_input_buffer * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
    }

    // How many S2G token entry of current chunk have been in-flight.
    int additional_in_flight_s2g = 0;
    
    // Iterate through all dst(output) tokens within this chunk in the per-rank buffer by reducing all src tokens for each of them.
    // Assign each dst token to each pipeline using a round-robin fasion.
    for(int j = pipeline_rank; j < num_of_tokens_for_current_chunk; j += NUM_OF_DATA_PIPELINE_PER_BLOCK){
      // Check whether there is a previous chunk's token entry S2G is in-flight and also current chunk already has NUM_OF_ADDITIONAL_IN_FLIGHT_S2G token entry S2G in-flight.
      // If so, wait for previous chunk's token entry S2G finish and notify the inter_node_G2S and intra_node_G2S warp groups on all ranks.
      if(outstanding_in_flight_chunk && (additional_in_flight_s2g == NUM_OF_ADDITIONAL_IN_FLIGHT_S2G)){
        if(warp_rank_within_pipeline == 0){
          if(elect_sync(~0)){
            // Wait for previous chunk's token entry S2G finish.
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<NUM_OF_ADDITIONAL_IN_FLIGHT_S2G>{});
            // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk
            // before they can observe the update of the flags.
            // Required for both intra-node (NVLink peer memory) and inter-node communication.
            asm volatile("fence.release.sys;"
                         :
                         :
                         : "memory");
            // Notify the inter_node_G2S and intra_node_G2S warp groups on all ranks in this node.
            // Atomically reduce add 1 to the u32 flag of the last token chunk to target flag buffer.
            // Since each unpermute block will have NUM_OF_DATA_PIPELINE_PER_BLOCK pipeline processing the same chunk, 
            // the expected value of this chunk's flag should atomicAdd NUM_OF_DATA_PIPELINE_PER_BLOCK not 1.
            uint32_t* last_chunk_flag_addr = intra_node_expert_input_chunk_flags[last_chunk_rank_id] + last_chunk_global_chunk_id;
            // Need a strong system-scope red to make sure all ranks can observe the update of the flag,
            // Notify last chunk.
            asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                          :
                          : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                          : "memory");
          }
        }
        outstanding_in_flight_chunk = false;
      }
      // Each dst token need to accumulate src tokens from local rank's local expert buffer.
      // Accumulator for this dst token. Token must be accumulated in FP32.
      float2 acc_token_fp32[NUM_OF_ELEMENT_PER_THREAD];
      // Optional Accumulator for this dst token prob.
      float acc_prob[NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
      // End reduction group flag.
      bool last_src_token = false;
      // Init token accumulator.
      #pragma unroll
      for(int k = 0; k < NUM_OF_ELEMENT_PER_THREAD; k++){
        acc_token_fp32[k].x = 0.0f;
        acc_token_fp32[k].y = 0.0f;
      }
      // Init prob accumulator.
      #pragma unroll
      for(int k = 0; k < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; k++){
        acc_prob[k] = 0.0f;
      }
          
      // Continue loading src token for this dst token and reduce them to accumulator until all src token for this dst token have been accumulated.
      do{
        // Base address for current token and prob(optional) in shared memory.
        __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->unpermute_token_G2S_buffer[token_stage][0]);
        float* load_prob_base_ptr;
        if constexpr(BACKWARD_COMBINE){
          load_prob_base_ptr = &smem_buffer_ptr->unpermute_prob_G2S_buffer[token_stage];
        }

        // Wait until current src token ready in shared memory.
        if(warp_rank_within_pipeline == 0){
          if(elect_sync(~0)){
            while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
          }
        }
        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 1 + pipeline_rank);

        // Accumulate token and prob(optional).
        #pragma unroll
        for(int k = 0; k < NUM_OF_ELEMENT_PER_THREAD; k++){
          int element_id = (k * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
          if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
            __nv_bfloat162 src_data = load_token_base_ptr[element_id];
            float2 src_data_fp32 = __bfloat1622float2(src_data);
            acc_token_fp32[k].x += src_data_fp32.x;
            acc_token_fp32[k].y += src_data_fp32.y;
          }     
        }

        if constexpr(BACKWARD_COMBINE){
          // Load the local expert id of this src token.
          int src_token_local_expert_id = smem_buffer_ptr->unpermute_local_expert_id_G2S_buffer[token_stage];
          #pragma unroll
          for(int k = 0; k < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; k++){
            int element_id = thread_rank_within_pipeline + k * NUM_OF_THREADS_PER_PIPELINE;
            if(element_id == expert_id_offset + src_token_local_expert_id){
              acc_prob[k] = *load_prob_base_ptr;
            }
          }
        }

        // Check flag for last src token.
        last_src_token = smem_buffer_ptr->unpermute_flag_G2S_buffer[token_stage];

        // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
        // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 1 + pipeline_rank);
        if(warp_rank_within_pipeline == 0){
          if(elect_sync(~0)){
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[token_stage][1]);
          }
        }
            
        // Goto next src token entry.
        token_stage += 1;
        if(token_stage == ending_G2S_index){
          token_stage = starting_G2S_index;
          token_producer_parity ^= 1;
        }

      }while(!last_src_token);
      
      // Store the dst token back to share memory. 
      // Base address for current dst token and prob(optional) in shared memory.
      __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->unpermute_token_S2G_buffer[dst_token_stage][0]);
      float* store_prob_base_ptr;
      if constexpr(BACKWARD_COMBINE){
        store_prob_base_ptr = &smem_buffer_ptr->unpermute_prob_S2G_buffer[dst_token_stage][0];
      }

      // Select the TMA thread within the pipeline to wait for previously issued TMA S2G operations finish reading this entry.
      if(warp_rank_within_pipeline == 0){
        if(elect_sync(~0)){
          cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G_PER_PIPELINE - 1>{});
        }
      }
      // Make sure all threads within the pipeline have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
      arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 1 + pipeline_rank);
          
      // Store the token.
      #pragma unroll
      for(int k = 0; k < NUM_OF_ELEMENT_PER_THREAD; k++){
        int element_id = (k * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
        if(element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN){
          // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
          store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[k]);
        }
      }

      // Store the prob(optional).
      if constexpr(BACKWARD_COMBINE){
        #pragma unroll
        for(int k = 0; k < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; k++){
          int element_id = thread_rank_within_pipeline + k * NUM_OF_THREADS_PER_PIPELINE;
          if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
            store_prob_base_ptr[element_id] = acc_prob[k];
          }
        }
      }

      // Make sure the shared memory stored by current thread is visible by async proxy.
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

      // Make sure all threads within the pipeline have finished storing the current token entry and making it visible to async proxy.
      arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 1 + pipeline_rank);

      // Select the TMA thread within the pipeline to issue S2G TMA operations for current token entry.
      if(warp_rank_within_pipeline == 0){
        if(elect_sync(~0)){
          uint16_t* current_token_addr = remote_expert_input_token_base_ptr + j * static_cast<int64_t>(HIDDEN_DIM);
          // Store the token from shared to global output.
          cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                   cuda::ptx::space_shared,
                                   reinterpret_cast<void*>(current_token_addr),
                                   reinterpret_cast<const void*>(&smem_buffer_ptr->unpermute_token_S2G_buffer[dst_token_stage][0]),
                                   (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

          // Store the prob from shared to global output.
          if constexpr(BACKWARD_COMBINE){
            float* current_prob_addr = remote_expert_input_prob_base_ptr + j * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
            cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                     cuda::ptx::space_shared,
                                     reinterpret_cast<void*>(current_prob_addr),
                                     reinterpret_cast<const void*>(&smem_buffer_ptr->unpermute_prob_S2G_buffer[dst_token_stage][0]),
                                     (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)));

          }
          // Commit S2G TMA operations for this dst token into a bulk async copy group.
          cuda::ptx::cp_async_bulk_commit_group();
        }
      }

      // Goto next dst token entry.
      dst_token_stage += 1;
      if(dst_token_stage == ending_S2G_index){
        dst_token_stage = starting_S2G_index;
      }

      // Another token entry's S2G in-flight.
      additional_in_flight_s2g += 1;
    }
    // If the current chunk does not have NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token entry in-flight.
    // We need to wait for both previous and current chunks' S2G entry to finish and notify the inter_node_G2S and intra_node_G2S warp groups.
    if(outstanding_in_flight_chunk){
      if(warp_rank_within_pipeline == 0){
        if(elect_sync(~0)){
          // Wait for all previous chunk's(i.e. previous and current chunk) token entry S2G finish.
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk
          // before they can observe the update of the flags.
          // Required for both intra-node (NVLink peer memory) and inter-node communication.
          asm volatile("fence.release.sys;"
                        :
                        :
                        : "memory");
          // Notify the inter_node_G2S and intra_node_G2S warp groups on all ranks in this node.
          // Atomically reduce add 1 to the u32 flag of the last and current token chunk to target flag buffer.
          // Since each unpermute block will have NUM_OF_DATA_PIPELINE_PER_BLOCK pipeline processing the same chunk,
          // the expected value of this chunk's flag should atomicAdd NUM_OF_DATA_PIPELINE_PER_BLOCK not 1.
          uint32_t* last_chunk_flag_addr = intra_node_expert_input_chunk_flags[last_chunk_rank_id] + last_chunk_global_chunk_id;
          uint32_t* current_chunk_flag_addr = intra_node_expert_input_chunk_flags[current_rank_id] + current_flag_id;
          // Need a strong system-scope red to make sure all ranks can observe the update of the flag,
          // Notify last chunk.
          asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                        :
                        : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                        : "memory");
          // Notify current chunk.
          asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                        :
                        : "l"(__cvta_generic_to_global(current_chunk_flag_addr)), "n"(1)
                        : "memory");
        }
      }
      outstanding_in_flight_chunk = false;
    }else{  // Otherwise, the current chunks is in-flight.
      outstanding_in_flight_chunk = true;
      // Update last chunk's id.
      last_chunk_global_chunk_id = current_flag_id;
      last_chunk_rank_id = current_rank_id;
    }
  }
  // When all chunks have been processed, we need to check whether the last chunk is still in-flight.
  // If so, wait for it and notify the inter_node_G2S and intra_node_G2S warp groups.
  if(outstanding_in_flight_chunk){
    if(warp_rank_within_pipeline == 0){
      if(elect_sync(~0)){
        // Wait for the last chunk's S2G finish.
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Need a system-scope release memory fence to let all target ranks can observe the side effect of TMA writes of this chunk
        // before they can observe the update of the flags.
        // Required for both intra-node (NVLink peer memory) and inter-node communication.
        asm volatile("fence.release.sys;"
                      :
                      :
                      : "memory");
        // Notify the inter_node_G2S and intra_node_G2S warp groups on all ranks in this node.
        // Atomically reduce add 1 to the u32 flag of the last token chunk to target flag buffer.
        // Since each unpermute block will have NUM_OF_DATA_PIPELINE_PER_BLOCK pipeline processing the same chunk,
        // the expected value of this chunk's flag should atomicAdd NUM_OF_DATA_PIPELINE_PER_BLOCK not 1.
        uint32_t* last_chunk_flag_addr = intra_node_expert_input_chunk_flags[last_chunk_rank_id] + last_chunk_global_chunk_id;
        // Need a strong system-scope red to make sure all ranks can observe the update of the flag,
        // Notify last chunk.
        asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                      :
                      : "l"(__cvta_generic_to_global(last_chunk_flag_addr)), "n"(1)
                      : "memory");
      }
    }
  }
}
#endif

template<int NUM_OF_RANKS_PER_NODE>
__launch_bounds__(1, 1)
__global__ void device_sync_kernel(uint32_t* intra_node_remote_flags, uint32_t* expected_flag_value, uint32_t* parity)
{
  __threadfence_system();

  // What's the current parity used for this sync.
  uint32_t flag_parity = *parity;
  uint32_t current_parity_expected_flag_value = expected_flag_value[flag_parity] + NUM_OF_RANKS_PER_NODE;
  // Atomically reduce add 1 to the u32 flag on rank #0 in current NVLink domain. 
  // Need a strong system-scope red to make sure all ranks from current NVLink domain can see the side effect.
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                :
                : "l"(__cvta_generic_to_global(intra_node_remote_flags + flag_parity)), "n"(1)
                : "memory");

  // Polling flag value from the u32 flag on rank #0 in current NVLink domain.
  // Keep polling until reach the expected value.
  uint32_t flag_data = 0;
  do{
      flag_data = 0;
      // Need a strong system-scope load to observe other ranks' Atomic result.
      // But no no memory fence(i.e. .aquired) needed since no memory operation behind this.
      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                    : "=r"(flag_data)
                    : "l"(__cvta_generic_to_global(intra_node_remote_flags + flag_parity))
                    : "memory");
    }while(flag_data != current_parity_expected_flag_value);
  // Save the new expected_flag_value and the parity back to global memory.
  expected_flag_value[flag_parity] = current_parity_expected_flag_value;
  *parity = flag_parity ^ 1;
}

// This kernel will update expected_rdma_flag_value by increasing the expected_rdma_flag_value by EXPECTED_RDMA_FLAG_VALUE_INCREMENT in local device memory.
// When permute fusion is enabled, will also update expected_permute_flag_value or expected_unpermute_flag_value in local device memory
// by increasing the expected_permute_flag_value or expected_unpermute_flag_value by EXPECTED_PERMUTE_UNPERMUTE_FLAG_VALUE_INCREMENT.
template<int NUM_OF_NODES,
         int EXPECTED_RDMA_FLAG_VALUE_INCREMENT,
         int EXPECTED_PERMUTE_UNPERMUTE_FLAG_VALUE_INCREMENT>
__launch_bounds__(1, 1)
__global__ void update_expected_value_kernel(uint64_t* expected_rdma_flag_value, uint32_t* expected_permute_unpermute_flag_value)
{
  if constexpr(NUM_OF_NODES != 1){
    (*expected_rdma_flag_value) += EXPECTED_RDMA_FLAG_VALUE_INCREMENT;
  }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  (*expected_permute_unpermute_flag_value) += EXPECTED_PERMUTE_UNPERMUTE_FLAG_VALUE_INCREMENT;
#endif
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifndef USE_NIXL
// This kernel will atomic add the flags of the sender by 1, and then verify that all senders have added 1.
template<typename T>
__launch_bounds__(1, 1)
__global__ void rdma_sync_kernel(const int num_of_nodes,
                                          const int node_rank,
                                          const uint64_t *expected_flag_value,
                                          uint64_t* rdma_inter_node_group_flags,
                                          doca_gpu_dev_verbs_qp **d_qps_gpu,
                                          T *mr_info) {
  for (int node_idx = 0; node_idx < num_of_nodes - 1; ++node_idx) {
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[node_idx];
    T *curr_mr_info = mr_info + node_idx;
    int rank_in_remote = node_idx < node_rank ? (node_rank - 1) : node_rank;
    uint64_t flag_offset = curr_mr_info->back_sync_barrier_idx;
    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, 1);
    struct doca_gpu_dev_verbs_wqe *flag_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(qp, flag_wqe_ptr, wqe_idx,
                                          DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                          curr_mr_info->flag_raddr + (flag_offset + rank_in_remote) * sizeof(uint64_t),
                                          curr_mr_info->flag_rkey,
                                          curr_mr_info->flag_laddr + (flag_offset + node_idx) * sizeof(uint64_t),
                                          curr_mr_info->flag_lkey,
                                          sizeof(uint64_t), 1, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit_db<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                      DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                      DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                      DOCA_GPUNETIO_VERBS_QP_SQ>(qp, (wqe_idx + 1));
    int status = doca_gpu_dev_verbs_poll_cq<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                            DOCA_GPUNETIO_VERBS_QP_SQ>(
                                            doca_gpu_dev_verbs_qp_get_cq_sq(qp), 1);
    assert(status >= 0);
  }
  for (int node_idx = 0; node_idx < num_of_nodes - 1; ++node_idx) {
    T *curr_mr_info = mr_info + node_idx;
    uint64_t flag_offset = curr_mr_info->back_sync_barrier_idx;
    const uint64_t* flag_location = rdma_inter_node_group_flags + flag_offset + node_idx;
    uint64_t rdma_flag = 0;
    do {
      rdma_flag = 0;
      // Need a strong system-scope load to observe external RDMA Atomic result.
      asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                   : "=l"(rdma_flag)
                   : "l"(__cvta_generic_to_global(flag_location))
                   : "memory");
    } while(rdma_flag != *expected_flag_value);
  }
}
#endif
#endif

template<typename TOKEN_DATA_TYPE, 
         // This type represent inter-node warp group.
         typename INTER_NODE_GROUP, 
         // This type represent intra-node G2S warp group.
         typename INTRA_NODE_G2S_GROUP,
         // This type represent intra-node S2G warp group.
         typename INTRA_NODE_S2G_GROUP,
         // This type represent permute G2S warp group.
         typename PERMUTE_G2S_GROUP,
         // This type represent permute S2G warp group.
         typename PERMUTE_S2G_GROUP,
         // Number of token entry in the shared memory.
         int NUM_OF_STAGES,
         // Number of token entry in the shared memory for permute block.
         int NUM_OF_STAGES_PERMUTE_BLOCK,
         // Number of in-flight S2G token entry in the shared memory, must be smaller than NUM_OF_STAGES.
         int NUM_OF_IN_FLIGHT_S2G,
         // Number of in-flight S2G token entry in the shared memory for permute block, must be smaller than NUM_OF_STAGES_PERMUTE_BLOCK.
         int NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK,
         // The padding size for each local expert.
         int LOCAL_EXPERTS_PADDING_SIZE,
         // Number of additional in-flight S2G entry before notify the permute G2S warp group.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         // Size of each chunk.
         int NUM_OF_TOKENS_PER_CHUNK,
         // Model configuration.
         int HIDDEN_DIM,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         // Number of CUDA block running dispatch kernel.
         int NUM_OF_BLOCKS,
         // Number of CUDA block running permute.
         int NUM_OF_PERMUTE_BLOCKS,
         // Whether the dispatch kernel is used in forward process or backward process.
         bool FORWARD_DISPATCH>
// Each CUDA block of dispatch kernel has 3 warp groups and has the following layout: 
// 1. inter-node warp group(i.e. RDMA N2N warp group, 1 warp, only valid for multinode scenario) 2. intra-node G2S warp group(i.e. NVL G2S warp group, 1 warp). 
// 3. intra-node S2G warp group(i.e. NVL S2G warp group, 2(multinode scenario)-3(single-node scenario) warps). Total 4 warps per CUDA block/SM.
// When permute fusion is enabled, the dispatch kernel will has NUM_OF_BLOCKS dispatch blocks + NUM_OF_PERMUTE_BLOCKS permute blocks(block-specialization enabled), with dispatch blocks come first.
// The dispatch block still follow the previous warp group layout, the permute block has 2 warp groups and has the following layout:
// 1. permute G2S warp group(1 warp). 2. permute S2G warp group(3 warps). Total 4 warps per CUDA block/SM, same as dispatch blocks.
__launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size(), 1)
__global__ void dispatch_kernel(const __grid_constant__ dispatch_kernel_param_t<TOKEN_DATA_TYPE> param)
{
  // Compile-time check.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  static_assert(INTER_NODE_GROUP::size() == 32, "Dispatch kernel only support 1 N2N warp currently.");
#endif
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Dispatch kernel only support 1 G2S warp currently.");
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  static_assert(PERMUTE_G2S_GROUP::size() == 32, "Dispatch kernel only support 1 permute G2S warp currently.");
  static_assert(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() == PERMUTE_G2S_GROUP::size() + PERMUTE_S2G_GROUP::size(), "Dispatch blocks and permute block should have the same size.");
#endif
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  static_assert((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * sizeof(float)) % 16 == 0, "Currently, expert per node must be multiple of 4(So the prob for each token is multiple of 16B) to make TMA work.");
  static_assert((HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)) % 16 == 0, "Currently, the size of token must be multiple of 16B to make TMA work.");
  if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
    // If FP8 token is used, HIDDEN_DIM must be multiple of 128 for scaling factor usage.
    static_assert(HIDDEN_DIM % 128 == 0, "HIDDEN_DIM must be multiple of 128 for scaling factor");
    // If FP8 token is used, HIDDEN_DIM must be multiple of 512 to make scaling factor multiple of 16B to make TMA work.
    static_assert(((HIDDEN_DIM / 128) * sizeof(float)) % 16 == 0, "Currently, scaling factor per token must be multiple of 16B.");
  }


  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
  using cur_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, FORWARD_DISPATCH>;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, also need to declare the type for the smem for permute blocks.
  using cur_permute_block_smem_t = dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES_PERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, FORWARD_DISPATCH>;
#endif
  cur_smem_t* smem_buffer_ptr = reinterpret_cast<cur_smem_t*>(smem_bytes);

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, also need to declare the ptr for the smem for permute block.
  // Different types of blocks will use different ptr.
  cur_permute_block_smem_t* permute_block_smem_buffer_ptr = reinterpret_cast<cur_permute_block_smem_t*>(smem_bytes);
#endif
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // To prevent compiler generate pointless comparison warning.
  int blockIdx_x_int = (int)blockIdx.x;
  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    if(blockIdx_x_int < NUM_OF_BLOCKS){
      // Dispatch blocks.
      for(int i = 0; i < NUM_OF_STAGES; i++){
        // Initialize mbarrier
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][0], 1);
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][1], INTRA_NODE_S2G_GROUP::warp_size());
      }
      // Initialize sparse_to_dense map mbarrier.
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[0], 1);
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[1], 1);
      // Initialize S2G warp group mbarrier.
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->S2G_group_mbarrier_buffer, INTRA_NODE_S2G_GROUP::warp_size());
      // Make mbarriers initialization visible to async proxy(TMA).
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }else if(blockIdx_x_int < NUM_OF_BLOCKS + NUM_OF_PERMUTE_BLOCKS){
      // Permute blocks.
      for(int i = 0; i < NUM_OF_STAGES_PERMUTE_BLOCK; i++){
        // Initialize mbarrier
        cuda::ptx::mbarrier_init(&permute_block_smem_buffer_ptr->permute_mbarrier_buffer[i][0], 1);
        cuda::ptx::mbarrier_init(&permute_block_smem_buffer_ptr->permute_mbarrier_buffer[i][1], PERMUTE_S2G_GROUP::warp_size());
      }
      // Make mbarriers initialization visible to async proxy(TMA).
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }else{
      // Too many blocks, should not goes here.
    }
  }
#else
  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    for(int i = 0; i < NUM_OF_STAGES; i++){
      // Initialize mbarrier
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][0], 1);
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][1], INTRA_NODE_S2G_GROUP::warp_size());
    }
    // Initialize sparse_to_dense map mbarrier.
    cuda::ptx::mbarrier_init(&smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[0], 1);
    cuda::ptx::mbarrier_init(&smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer[1], 1);
    // Initialize S2G warp group mbarrier.
    cuda::ptx::mbarrier_init(&smem_buffer_ptr->S2G_group_mbarrier_buffer, INTRA_NODE_S2G_GROUP::warp_size());
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
#endif

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now blocks can become specialized if permute fusion is enabled.
  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  if(blockIdx_x_int < NUM_OF_BLOCKS){
    // Dispatch blocks.
    if(threadIdx_x_int < INTER_NODE_GROUP::size()){
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
      // Inter-node warps groups.
      if constexpr(NUM_OF_NODES != 1){
#ifdef USE_NIXL
        N2N_warp_group_device_function
        <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
        (param.node_rank, param.num_of_tokens_per_rank, param.attn_to_rdma_map, reinterpret_cast<dispatch_gpu_nixl_ctx*>(param.multinode_ctx_ptr), smem_buffer_ptr);
#else
        N2N_warp_group_device_function
        <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
        (param.node_rank, param.num_of_tokens_per_rank, param.attn_to_rdma_map, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<dispatch_memory_region_info_t*>(param.multinode_aux_ptr), smem_buffer_ptr);
#endif
      }
#endif
    }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
      // Intra-node G2S warp groups.
      G2S_warp_group_device_function
      <INTRA_NODE_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
      MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.node_rank, param.num_of_tokens_per_rank, param.expected_rdma_flag_value, param.rdma_to_attn_map,
      param.attn_input_token, param.attn_input_prob, param.attn_input_token_scaling_factor, param.rdma_inter_node_group_token,
      param.rdma_inter_node_group_prob, param.rdma_inter_node_group_scaling_factor, param.rdma_inter_node_group_flags, smem_buffer_ptr);
    }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()){
      // Intra-node S2G warp groups.
      S2G_warp_group_device_function
      <INTRA_NODE_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_IN_FLIGHT_S2G, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_output_token, param.expert_output_prob,
      param.expert_output_scaling_factor, param.intra_node_expert_output_chunk_flags, smem_buffer_ptr);
    }else{
      // Too many threads, should not goes here.
    }
  }else if(blockIdx_x_int < NUM_OF_BLOCKS + NUM_OF_PERMUTE_BLOCKS){
    // Permute blocks.
    if(threadIdx_x_int < PERMUTE_G2S_GROUP::size()){
      // Permute G2S warp groups.
      permute_G2S_warp_group_device_function
      <PERMUTE_G2S_GROUP, TOKEN_DATA_TYPE, cur_permute_block_smem_t, NUM_OF_STAGES_PERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK,
      NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_PERMUTE_BLOCKS, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.node_rank, param.num_of_tokens_per_rank, param.expected_permute_flag_value, param.dense_chunk_layout, param.expert_output_token[param.local_rank], 
      param.expert_output_prob[param.local_rank], param.expert_output_scaling_factor[param.local_rank], param.intra_node_expert_output_chunk_flags[param.local_rank], permute_block_smem_buffer_ptr);
    }else if(threadIdx_x_int < PERMUTE_G2S_GROUP::size() + PERMUTE_S2G_GROUP::size()){
      // Permute S2G warp groups.
      permute_S2G_warp_group_device_function
      <PERMUTE_S2G_GROUP, TOKEN_DATA_TYPE, cur_permute_block_smem_t, NUM_OF_STAGES_PERMUTE_BLOCK, NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK,
      NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, LOCAL_EXPERTS_PADDING_SIZE, NUM_OF_PERMUTE_BLOCKS, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.dense_chunk_layout, param.dense_to_expert_map, param.num_of_local_experts_tokens, param.local_expert_output_token,
      param.local_expert_output_prob, param.local_expert_output_scaling_factor, permute_block_smem_buffer_ptr);
    }else{
      // Too many threads, should not goes here.
    }
  }else{
    // Too many blocks, should not goes here.
  }
#else
  if(threadIdx_x_int < INTER_NODE_GROUP::size()){
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    // Inter-node warps groups.
    if constexpr(NUM_OF_NODES != 1){
#ifdef USE_NIXL
      // Use NIXL for inter-node communication
      N2N_warp_group_device_function
      <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
       MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.node_rank, param.num_of_tokens_per_rank, param.attn_to_rdma_map, reinterpret_cast<dispatch_gpu_nixl_ctx*>(param.multinode_ctx_ptr), smem_buffer_ptr);
#else
      // Use DOCA for inter-node communication
      N2N_warp_group_device_function
      <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
       MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.node_rank, param.num_of_tokens_per_rank, param.attn_to_rdma_map, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<dispatch_memory_region_info_t*>(param.multinode_aux_ptr), smem_buffer_ptr);
#endif // USE_NIXL
    }
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE
  }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp groups.
    G2S_warp_group_device_function
    <INTRA_NODE_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
     MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.node_rank, param.num_of_tokens_per_rank, param.expected_rdma_flag_value, param.rdma_to_attn_map,
     param.attn_input_token, param.attn_input_prob, param.attn_input_token_scaling_factor, param.rdma_inter_node_group_token,
     param.rdma_inter_node_group_prob, param.rdma_inter_node_group_scaling_factor, param.rdma_inter_node_group_flags, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()){
    // Intra-node S2G warp groups.
    S2G_warp_group_device_function
    <INTRA_NODE_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_IN_FLIGHT_S2G, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_output_token, param.expert_output_prob,
    param.expert_output_scaling_factor, param.intra_node_expert_output_chunk_flags, smem_buffer_ptr);
  }else{
    // Too many threads, should not goes here.
  }
#endif
}

template<// This type represent intra-node reduction warp group.
         typename INTRA_NODE_RED_GROUP, 
         // This type represent inter-node reduction warp group.
         typename INTER_NODE_RED_GROUP, 
         // This type represent intra-node G2S warp group.
         typename INTRA_NODE_G2S_GROUP,
         // This type represent inter-node G2S warp group.
         typename INTER_NODE_G2S_GROUP,
         // This type represent inter-node rdma warp group.
         typename INTER_NODE_RDMA_GROUP,
         // This type represent unpermute G2S warp group.
         typename UNPERMUTE_G2S_GROUP,
         // This type represent unpermute red warp group.
         typename UNPERMUTE_RED_GROUP,
         // Number of independent data pipeline per CUDA block. 
         int NUM_OF_DATA_PIPELINE_PER_BLOCK,
         // Number of token entry in the shared memory for G2S operations.
         int NUM_OF_STAGES_G2S,
         // Number of token entry in the shared memory for S2G operations.
         int NUM_OF_STAGES_S2G,
         // Number of token entry in the shared memory for G2S operations for unpermute block.
         int NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK,
         // Number of token entry in the shared memory for S2G operations for unpermute block.
         int NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK,
         // Number of token per group in the inter-node reduction/G2S warp group.
         int NUM_OF_TOKENS_PER_GROUP,
         // Size of each chunk.
         int NUM_OF_TOKENS_PER_CHUNK,
         // Model configuration.
         int HIDDEN_DIM,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         // Number of CUDA block running combine kernel.
         int NUM_OF_BLOCKS,
         // Number of CUDA block running unpermute.
         int NUM_OF_UNPERMUTE_BLOCKS,
         // Number of fully in-flight S2G in intra-node reduction warp group.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, 
         // Number of fully in-flight S2G in unpermute reduction warp group.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCKS,
         // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
         bool BACKWARD_COMBINE>
// Each CUDA block of combine kernel has 5 warp groups and has the following layout: 
// 1. intra-node reduction warp group(4 warps, only valid for multinode scenario). 2. inter-node reduction warp group(4 warps, 1 pipeline for multinode scenario, 2 pipeline otherwise).
// 3. intra-node G2S warp group(1 warp, only valid for multinode scenario). 4. inter-node G2S warp group(1 warp for multinode scenario, 2 warps otherwise). 5. inter-node N2N rdma warp group(1 warp, only valid for multinode scenario). 
// Total 6(single-node) or 11(multi-node) warps per CUDA block/SM.
// When (un)permute fusion is enabled, the combine kernel will has NUM_OF_BLOCKS combine blocks + NUM_OF_UNPERMUTE_BLOCKS unpermute blocks(block-specialization enabled), with combine blocks come first.
// The combine block still follow the previous warp group layout, the unpermute block has 2 warp groups and has the following layout:
// 1. unpermute G2S warp group(1 warp for multinode scenario, 2 warps otherwise, 1 warp per pipeline). 2. unpermute reduction warp group(4 warps, 1 pipeline for multinode scenario, 2 pipeline otherwise). 
// Total 6(single-node) or 5(multi-node) warps per CUDA block/SM, same as inter-node G2S and reduction warp group.
__launch_bounds__(INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size(), 1)
__global__ void combine_kernel(const __grid_constant__ combine_kernel_param_t param)
{
  // Compile-time check.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
  static_assert(INTER_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
#endif
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  static_assert(UNPERMUTE_G2S_GROUP::size() == INTER_NODE_G2S_GROUP::size(), "unpermute G2S warp groups should have the same layout as inter-node G2S warp groups.");
  static_assert(UNPERMUTE_RED_GROUP::size() == INTER_NODE_RED_GROUP::size(), "unpermute red warp groups should have the same layout as inter-node red warp groups.");
#endif
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  static_assert((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * sizeof(float)) % 16 == 0, "Currently, expert per node must be multiple of 4(So the prob for each token is multiple of 16B) to make TMA work.");
  static_assert((HIDDEN_DIM * sizeof(uint16_t)) % 16 == 0, "Currently, the size of token must be multiple of 16B to make TMA work.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;

  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
  using cur_smem_t = combine_kernel_dynamic_shared_memory_buffer_t
  <NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, BACKWARD_COMBINE>;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, also need to declare the type for the smem for unpermute blocks.
  using cur_unpermute_block_smem_t = combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t
  <NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK, NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, BACKWARD_COMBINE>;
#endif
  cur_smem_t* smem_buffer_ptr = reinterpret_cast<cur_smem_t*>(smem_bytes);
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, also need to declare the ptr for the smem for unpermute block.
  // Different types of blocks will use different ptr.
  cur_unpermute_block_smem_t* unpermute_block_smem_buffer_ptr = reinterpret_cast<cur_unpermute_block_smem_t*>(smem_bytes);
#endif

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // To prevent compiler generate pointless comparison warning.
  int blockIdx_x_int = (int)blockIdx.x;
  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    if(blockIdx_x_int < NUM_OF_BLOCKS){
      // Combine blocks.
      for(int i = 0; i < NUM_OF_STAGES_G2S; i++){
        // Initialize mbarrier
        if constexpr(NUM_OF_NODES != 1){
          cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][0], 1);
          cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][1], 1);
        }
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][0], 1);
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][1], 1);
      }
      if constexpr(NUM_OF_NODES != 1){
        // Initialize mbarrier
        for(int i = 0; i < NUM_OF_NODES - 1; i++){
          for(int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++){
            cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[i][j], 1);
          }
        }
      }
      // Make mbarriers initialization visible to async proxy(TMA).
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }else if(blockIdx_x_int < NUM_OF_BLOCKS + NUM_OF_UNPERMUTE_BLOCKS){
      // Unpermute blocks.
      for(int i = 0; i < NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK; i++){
        // Initialize mbarrier
        if constexpr(BACKWARD_COMBINE){
          // When BACKWARD_COMBINE is true, i.e. we have prob element to copy from G2S, we need to take LDGSTS into account.
          cuda::ptx::mbarrier_init(&unpermute_block_smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[i][0], 2);
        }else{
          // Otherwise, no LDGSTS involved, only generic thread will arrive on.
          cuda::ptx::mbarrier_init(&unpermute_block_smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[i][0], 1);
        }
        cuda::ptx::mbarrier_init(&unpermute_block_smem_buffer_ptr->unpermute_mbarrier_G2S_buffer[i][1], 1);
      }
      // Make mbarriers initialization visible to async proxy(TMA).
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }else{
      // Too many blocks, should not goes here.
    }
  }
#else
  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    for(int i = 0; i < NUM_OF_STAGES_G2S; i++){
      // Initialize mbarrier
      if constexpr(NUM_OF_NODES != 1){
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][0], 1);
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][1], 1);
      }
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][0], 1);
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][1], 1);
    }
    if constexpr(NUM_OF_NODES != 1){
      // Initialize mbarrier
      for(int i = 0; i < NUM_OF_NODES - 1; i++){
        for(int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++){
          cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[i][j], 1);
        }
      }
    }
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
#endif

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now blocks can become specialized if permute fusion is enabled.
  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  if(blockIdx_x_int < NUM_OF_BLOCKS){
    // Combine blocks.
    if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size()){
      // Intra-node reduction warp group.
      if constexpr(NUM_OF_NODES != 1){
        intra_node_red_warp_group_device_function
        <INTRA_NODE_RED_GROUP, cur_smem_t, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK,
        NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>
        (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.rdma_intra_node_red_token, param.rdma_intra_node_red_prob, smem_buffer_ptr);
      }
    }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size()){
      // Inter-node reduction warp group.
      inter_node_red_warp_group_device_function
      <cur_smem_t, INTER_NODE_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK,
      NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.attn_to_rdma_map, param.attn_output_token, param.attn_output_prob, smem_buffer_ptr);
    }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
      // Intra-node G2S warp group.
      if constexpr(NUM_OF_NODES != 1){
        intra_node_G2S_warp_group_device_function
        <cur_smem_t, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
        (param.node_rank, param.local_rank, param.num_of_tokens_per_rank, param.expected_unpermute_flag_value, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_input_token, 
        param.expert_input_prob, param.intra_node_expert_input_chunk_flags[param.local_rank], smem_buffer_ptr);
      }
    }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size()){
      // Inter-node G2S warp group.
      inter_node_G2S_warp_group_device_function
      <cur_smem_t, INTER_NODE_G2S_GROUP, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS,
      NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
      (param.node_rank, param.local_rank, param.num_of_tokens_per_rank, param.expected_unpermute_flag_value, param.expected_rdma_flag_value, param.rdma_to_attn_map, param.attn_to_rdma_map, 
      param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob, param.rdma_inter_node_group_token, param.rdma_inter_node_group_prob, 
      param.intra_node_expert_input_chunk_flags[param.local_rank], param.rdma_inter_node_group_flags, smem_buffer_ptr);
    }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()){
      // Inter-node rdma warp group.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
      if constexpr(NUM_OF_NODES != 1){
#ifdef USE_NIXL
        inter_node_N2N_warp_group_device_function
        <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
        (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, reinterpret_cast<combine_gpu_nixl_ctx*>(param.multinode_ctx_ptr), smem_buffer_ptr);
#else
        inter_node_N2N_warp_group_device_function
        <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
        (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<combine_memory_region_info_t*>(param.multinode_aux_ptr), smem_buffer_ptr);
#endif
      }
#endif
    }else{
      // Too many threads, should not goes here.
    }
  }else if(blockIdx_x_int < NUM_OF_BLOCKS + NUM_OF_UNPERMUTE_BLOCKS){
    // Unpermute blocks.
    if(threadIdx_x_int < UNPERMUTE_RED_GROUP::size()){
      // Unpermute red warp groups.
      unpermute_red_warp_group_device_function
      <cur_unpermute_block_smem_t, UNPERMUTE_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK, NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
      NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_UNPERMUTE_BLOCKS, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.local_rank, param.num_of_tokens_per_rank, param.dense_chunk_layout, param.expert_input_token[param.local_rank], param.expert_input_prob[param.local_rank],
      param.intra_node_expert_input_chunk_flags, unpermute_block_smem_buffer_ptr);
    }else if(threadIdx_x_int < UNPERMUTE_RED_GROUP::size() + UNPERMUTE_G2S_GROUP::size()){
      // Unpermute G2S warp groups.
      unpermute_G2S_warp_group_device_function
      <cur_unpermute_block_smem_t, UNPERMUTE_G2S_GROUP, NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK,
      NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_UNPERMUTE_BLOCKS, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.expected_unpermute_flag_value, param.dense_chunk_layout, param.dense_to_expert_map, param.local_expert_input_token,
      param.local_expert_input_prob, param.intra_node_expert_input_chunk_flags[param.local_rank], unpermute_block_smem_buffer_ptr);
    }else{
      // The combine block maybe larger than the unpermute block, so there maybe some residue threads in the unpermute block left unused, these thread will do nothing and exit.
    }
  }else{
    // Too many blocks, should not goes here.
  }
#else
  if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size()){
    // Intra-node reduction warp group.
    if constexpr(NUM_OF_NODES != 1){
      intra_node_red_warp_group_device_function
      <INTRA_NODE_RED_GROUP, cur_smem_t, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK,
      NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.rdma_intra_node_red_token, param.rdma_intra_node_red_prob, smem_buffer_ptr);
    }
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size()){
    // Inter-node reduction warp group.
    inter_node_red_warp_group_device_function
    <cur_smem_t, INTER_NODE_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK,
    NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.attn_to_rdma_map, param.attn_output_token, param.attn_output_prob, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp group.
    if constexpr(NUM_OF_NODES != 1){
      intra_node_G2S_warp_group_device_function
      <cur_smem_t, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.local_rank, param.num_of_tokens_per_rank, param.expected_unpermute_flag_value, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_input_token, 
      param.expert_input_prob, param.intra_node_expert_input_chunk_flags[param.local_rank], smem_buffer_ptr);
    }
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size()){
    // Inter-node G2S warp group.
    inter_node_G2S_warp_group_device_function
    <cur_smem_t, INTER_NODE_G2S_GROUP, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS,
    NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.node_rank, param.local_rank, param.num_of_tokens_per_rank, param.expected_unpermute_flag_value, param.expected_rdma_flag_value, param.rdma_to_attn_map, param.attn_to_rdma_map, 
    param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob, param.rdma_inter_node_group_token, param.rdma_inter_node_group_prob, 
    param.intra_node_expert_input_chunk_flags[param.local_rank], param.rdma_inter_node_group_flags, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()){
    // Inter-node rdma warp group.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    if constexpr(NUM_OF_NODES != 1){
#ifdef USE_NIXL
      // Use NIXL for inter-node communication
      inter_node_N2N_warp_group_device_function
      <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, reinterpret_cast<combine_gpu_nixl_ctx*>(param.multinode_ctx_ptr), smem_buffer_ptr);
#else
      // Use DOCA for inter-node communication
      inter_node_N2N_warp_group_device_function
      <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<combine_memory_region_info_t*>(param.multinode_aux_ptr), smem_buffer_ptr);
#endif // USE_NIXL
    }
#endif // HYBRID_EP_BUILD_MULTINODE_ENABLE
  }else{
    // Too many threads, should not goes here.
  }
#endif
}

template<int NUM_THREADS_PER_BLOCK,
         int NUM_OF_BLOCKS,
         int LOCAL_EXPERTS_PADDING_SIZE, 
         int NUM_OF_TOKENS_PER_CHUNK, 
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_EXPERTS_PER_RANK>
__launch_bounds__(NUM_THREADS_PER_BLOCK, 1)
__global__ void scan(const bool* input_routing_map, 
                     tmp_state_t* tmp, 
                     tmp_state_t* local_experts_tmp, 
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
                     const int local_experts_tokens_limit, // This size MUST be multiple of LOCAL_EXPERTS_PADDING_SIZE!
                     const int num_of_tokens_per_rank)
{
  // Calculate the warps per block.
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;

  // Calculate total threads count.
  constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;
  
  // Calculate the number of tokens belong to each CUDA block, warp and thread.
  // We assign 1 token(row in routing map) to 1 thread.
  const int num_of_total_attn_tokens = num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  //static_assert(NUM_OF_TOTAL_ATTN_TOKENS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_ATTN_TOKENS must be multiple of NUM_OF_TOTAL_THREADS");
  const int num_of_tokens_per_thread = ((num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
  const int num_of_tokens_per_warp = num_of_tokens_per_thread * WARP_SIZE;
  const int num_of_tokens_per_block = num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // How many total chunks for all ranks.
  const int num_of_total_attn_chunks = num_of_chunks_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
#endif
  // For each token(row in routing map), calculate how many bytes need to be loaded from the routing map and how to load them.
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  constexpr int NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
  using copy_t = Copy_t<NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN % sizeof(copy_t) == 0, "NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN and copy_t mismatch");
  constexpr int ROUTING_MAP_LOAD_ITER = NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN / sizeof(copy_t);

  // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
  constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * NUM_OF_RANKS_PER_NODE;
  using write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN % sizeof(write_t) == 0, "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN and write_t mismatch");
  constexpr int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, calculate how many bytes need to be store to dense_to_expert_map per token.
  constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS = sizeof(int32_t) * NUM_OF_EXPERTS_PER_RANK;
  using local_experts_write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS>;
  static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS % sizeof(local_experts_write_t) == 0, "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS and local_experts_write_t mismatch");
  constexpr int D2E_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN_FOR_LOCAL_EXPERTS / sizeof(local_experts_write_t);
#endif

  // How to convert per-expert routing info to per-rank routing info. We support any number of expert per rank.
  using expert_to_rank_t = Reduce_t<NUM_OF_EXPERTS_PER_RANK>;
  static_assert(NUM_OF_EXPERTS_PER_RANK % sizeof(expert_to_rank_t) == 0, "NUM_OF_EXPERTS_PER_RANK and expert_to_rank_t mismatch");
  constexpr int EXPERTS_TO_RANK_REDUCE_ITER = NUM_OF_EXPERTS_PER_RANK / sizeof(expert_to_rank_t);

  // How to convert per-rank routing info to per-node routing info. We support any number of ranks per node(nvl domain).
  //using rank_to_node_t = Reduce_t<NUM_OF_RANKS_PER_NODE>;
  //static_assert(NUM_OF_RANKS_PER_NODE % sizeof(rank_to_node_t) == 0, "NUM_OF_RANKS_PER_NODE and rank_to_node_t mismatch");
  //constexpr int RANKS_TO_NODE_REDUCE_ITER = NUM_OF_RANKS_PER_NODE / sizeof(rank_to_node_t);

  // How do a warp save per-rank routing info back to shared memory. What's the max number of elements does each thread save back.
  constexpr int NUM_OF_RANKS_PER_THREAD = ((NUM_OF_RANKS_PER_NODE - 1) / WARP_SIZE) + 1;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // How do a warp save local experts' routing info back to shared memory. What's the max number of elements does each thread save back.
  constexpr int NUM_OF_LOCAL_EXPERTS_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK - 1) / WARP_SIZE) + 1;
#endif

  // Sum of per-rank routing info of all warps within the block.
  __shared__ int32_t warp_token_routing_map_sum[NUM_OF_WARPS_PER_BLOCK][NUM_OF_RANKS_PER_NODE];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of local experts' routing info of all warps within the block.
  __shared__ int32_t warp_token_local_experts_routing_map_sum[NUM_OF_WARPS_PER_BLOCK][NUM_OF_EXPERTS_PER_RANK];
#endif
  // Sum of previous blocks' per-rank routing info.
  __shared__ int32_t previous_block_sum[NUM_OF_RANKS_PER_NODE];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of all blocks' local experts' routing info.
  __shared__ int32_t all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
  // Sum of previous blocks' local experts' routing info accumulated with previous local experts' routing info.
  __shared__ int32_t previous_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
#endif

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Init shared memory which are used as accumulator.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    all_block_local_experts_sum[i] = 0;
    previous_block_local_experts_sum[i] = 0;
  }
#endif

  // We assign contiguous tokens called chunk to each CUDA block, each CUDA block get the same size of chunk.
  int block_starting_token = blockIdx.x * num_of_tokens_per_block;
  // warp id and lane id.
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  // We assign contiguous tokens called sub-chunk to each warp within a CUDA block, each warp within a CUDA block get the same size of sub-chunk.
  int warp_starting_token = block_starting_token + warp_id * num_of_tokens_per_warp;
  // Within a sub-chunk, we assign tokens to thread in a interleave pattern. So each thread process a token each time and each warp sum a tile of 32 tokens each time.
  int thread_starting_token = warp_starting_token + lane_id;
  
  // Step 0: Each warp sum the sub-chunk assigned to them and store the sum back to shared memory.
  // All warps within all CTA attend this step.
  // Also, some tokens need per-node info which store to rdma_to_attn_map, also processed here.

  // Sum of per-rank token routing map within a thread.
  int32_t token_routing_map_sum[NUM_OF_RANKS_PER_NODE];
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    token_routing_map_sum[i] = 0;
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // Sum of local experts' token routing map within a thread.
  int32_t token_local_experts_routing_map_sum[NUM_OF_EXPERTS_PER_RANK];
  #pragma unroll
  for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
    token_local_experts_routing_map_sum[i] = 0;
  }
#endif

  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then just end summing tokens assigned to this thread. 
    if(current_token_id >= num_of_total_attn_tokens){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
    // If the token belongs to the inter-node group.
    // We need to calculate the per-node routing info and save back to rdma_to_attn_map.
    bool per_node_routing_info = (current_token_local_rank == local_rank);
    int current_token_rdma_to_attn_map_id = current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;
    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
      *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
    }

    // Convert the routing map to per rank routing info and accumulate to accumulator.
    // Also convert the per rank routing info to per node routing info.
    // When permute fusion is enabled, also accumulate local experts to accumulator.
    bool token_needed_by_this_node = false;
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      bool token_needed_by_this_rank = false;
      #pragma unroll
      for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
        int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
        expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        if(reduction_data != (expert_to_rank_t)0){
          token_needed_by_this_rank = true;
          break;
        }
      }
      if(token_needed_by_this_rank){
        token_routing_map_sum[j] += 1;
        token_needed_by_this_node = true;
      }
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      if(j == local_rank){
        int current_local_expert_id = j * NUM_OF_EXPERTS_PER_RANK;
        #pragma unroll
        for(int k = 0; k < NUM_OF_EXPERTS_PER_RANK; k++){
          token_local_experts_routing_map_sum[k] += (int32_t)(token_routing_map[current_local_expert_id + k]);
        }
      }
#endif
    }

    // Save the per node routing info back to rdma_to_attn_map if needed.
    if(per_node_routing_info){
      rdma_to_attn_map[current_token_rdma_to_attn_map_id] = token_needed_by_this_node;
    }
  }

  // Each warp sum the per-rank routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // Each warp store the sum of per-rank routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      warp_token_routing_map_sum[warp_id][element_id] = token_routing_map_sum[i];
    }
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each warp sum the local experts' routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_local_experts_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_local_experts_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // When permute fusion is enabled, each warp store the sum of local experts' routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      warp_token_local_experts_routing_map_sum[warp_id][element_id] = token_local_experts_routing_map_sum[i];
    }
  }
#endif

  // Sync within a CUDA block to make sure all warps have produced the per-rank sum data to the shared memory before any thread can consume them to produce CUDA block level's sum data.
  // When permute fusion is enabled, also make sure all warps have produced the local experts' sum data to the shared memory.
  __syncthreads();

  // Step 1: Communication between CUDA blocks. Each CUDA block's threads need to produce and store the current block's per-rank sum data to global memory,
  // and load and accumulate previous blocks' per-rank sum data and save the result to shared memory.
  // When permute fusion is enabled, Each CUDA block's threads also need to produce and store the current block's local experts' sum data to global memory,
  // and load and accumulate all & previous blocks' local experts' sum data and save the result to shared memory. This is due to the layout requirement of local experts' output buffer.

  // Each thread within a CUDA block calculate the CUDA block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t rank_acc = 0;
    // Calculate the sum of current rank within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      rank_acc += warp_token_routing_map_sum[j][i];
    }

    // Store the sum of current rank within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &tmp[blockIdx.x * NUM_OF_RANKS_PER_NODE + i];
    tmp_state_t tmp_data{PRIV_SUM, rank_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each thread within a CUDA block calculate the CUDA block level sum for a local expert at a time.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    int32_t local_expert_acc = 0;
    // Calculate the sum of current local expert within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      local_expert_acc += warp_token_local_experts_routing_map_sum[j][i];
    }

    // Store the sum of current local expert within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &local_experts_tmp[blockIdx.x * NUM_OF_EXPERTS_PER_RANK + i];
    tmp_state_t tmp_data{PRIV_SUM, local_expert_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }
#endif

  // Each thread within a CUDA block load previous blocks' block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t previous_block_sum_for_current_rank = 0;
    for(int j = 0; j < blockIdx.x; j++){
      tmp_state_t tmp_data{EMPTY, 0};
      tmp_state_t* tmp_src = &tmp[j * NUM_OF_RANKS_PER_NODE + i];
      do{
          // Load previous blocks' per-rank sum from global memory.
          // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
          uint64_t data = 0;
          asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                        : "=l"(data)
                        : "l"(__cvta_generic_to_global(tmp_src))
                        : "memory");
          tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
      }while(tmp_data.state != PRIV_SUM);
      previous_block_sum_for_current_rank += tmp_data.value;
    }
    previous_block_sum[i] = previous_block_sum_for_current_rank;
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, all threads within a CUDA block load all blocks' block level sum and accumulate to shared memory.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK * NUM_OF_BLOCKS; i += NUM_THREADS_PER_BLOCK){
    // Which block and which local expert is this sum element belongs to.
    int block_index = i / NUM_OF_EXPERTS_PER_RANK;
    int local_expert_index = i % NUM_OF_EXPERTS_PER_RANK;
    // Poll the sum element from global memory.
    tmp_state_t tmp_data{EMPTY, 0};
    tmp_state_t* tmp_src = &local_experts_tmp[i];
    do{
        // Load a block-level local expert's sum from global memory.
        // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
        uint64_t data = 0;
        asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                      : "=l"(data)
                      : "l"(__cvta_generic_to_global(tmp_src))
                      : "memory");
        tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
    }while(tmp_data.state != PRIV_SUM);

    // Atomically add the block-level local expert's sum element to shared memory to produce all blocks' sum and previous blocks' sum.
    atomicAdd_block(&all_block_local_experts_sum[local_expert_index], tmp_data.value);
    if(block_index < (int)blockIdx.x){
      atomicAdd_block(&previous_block_local_experts_sum[local_expert_index], tmp_data.value);
    }
  }
#endif

  // Sync within a CUDA block to make sure all previous blocks' per-rank sum have been produced to the shared memory before any thread can consume them in scan operation.
  // When permute fusion is enabled, also make sure all and previous blocks' local experts' sum have been produced to the shared memory.
  __syncthreads();

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE 
  // Load sum of all blocks' local experts' routing info to produce the accumulation of previous local experts' routing info. 
  int32_t thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
  // Only threads which will participate in accumulation will need to load the data from the shared memory.
  if(threadIdx.x < NUM_OF_EXPERTS_PER_RANK){
    #pragma unroll
    for(int i = 0; i < NUM_OF_EXPERTS_PER_RANK; i++){
      thread_local_all_block_local_experts_sum[i] = all_block_local_experts_sum[i];
    }
  }
  
  // When permute fusion is enabled, all threads within a CUDA block produced sum of previous blocks' local experts' routing info accumulated with previous local experts' routing info.
  for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
    int32_t current_expert_previous_block_sum = previous_block_local_experts_sum[i];
    int32_t previous_experts_acc = 0;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
    int32_t previous_experts_acc_plus_current_expert_valid_tokens;
    int32_t current_expert_valid_tokens;
    #pragma unroll
    for(int j = 0; j < NUM_OF_EXPERTS_PER_RANK; j++){
      if(j < i){
        // local experts sum can be >= zero, so need to handle the corner case.
        int num_of_padding_tile = (thread_local_all_block_local_experts_sum[j] % LOCAL_EXPERTS_PADDING_SIZE == 0) ? (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE)
                                                                                                                  : (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE + 1);
        int32_t local_expert_sum_with_padding = num_of_padding_tile * LOCAL_EXPERTS_PADDING_SIZE;
        previous_experts_acc += local_expert_sum_with_padding;
      }else if(j == i){
        current_expert_valid_tokens = thread_local_all_block_local_experts_sum[j];
        previous_experts_acc_plus_current_expert_valid_tokens = previous_experts_acc + thread_local_all_block_local_experts_sum[j];
      }
    }
#else
    #pragma unroll
    for(int j = 0; j < NUM_OF_EXPERTS_PER_RANK - 1; j++){
      if(j < i){
        // local experts sum can be >= zero, so need to handle the corner case.
        int num_of_padding_tile = (thread_local_all_block_local_experts_sum[j] % LOCAL_EXPERTS_PADDING_SIZE == 0) ? (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE)
                                                                                                                  : (thread_local_all_block_local_experts_sum[j] / LOCAL_EXPERTS_PADDING_SIZE + 1);
        int32_t local_expert_sum_with_padding = num_of_padding_tile * LOCAL_EXPERTS_PADDING_SIZE;
        previous_experts_acc += local_expert_sum_with_padding;
      }
    }
#endif
    previous_block_local_experts_sum[i] = current_expert_previous_block_sum + previous_experts_acc;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
    // First block will need to save all local experts' sum back to output buffer subject to token drop conditions.
    // First block also need to determine whether token drop is triggered.
    if(blockIdx.x == 0){
      int32_t num_of_current_experts_tokens;
      if(local_experts_tokens_limit > previous_experts_acc){
        // If previous local experts have not already fully occupied the local expert buffer, at least some valid tokens from current experts can be stored to the buffer.
        // This code path ONLY work when local_experts_tokens_limit is guarantee to be multiple of LOCAL_EXPERTS_PADDING_SIZE!
        if(local_experts_tokens_limit >= previous_experts_acc_plus_current_expert_valid_tokens){
          // If the local expert buffer's capacity can hold all the valid tokens from current expert, all valid tokens from current expert can be stored to the buffer.
          num_of_current_experts_tokens = current_expert_valid_tokens;
        }else{
          // If the local expert buffer's capacity cannot hold all the valid tokens from current expert, only partial of valid tokens can be strored to the buffer.
          num_of_current_experts_tokens = local_experts_tokens_limit - previous_experts_acc;
        }
      }else{
        // If all tokens from previous local experts(including both valid tokens and padding tokens) already exceed local expert buffer capacity, no more space for current local expert.
        num_of_current_experts_tokens = 0;
      }
      num_of_local_experts_tokens[i] = num_of_current_experts_tokens;

      // The thread which process the last local expert should determine whether token drop is triggered.
      if(i == NUM_OF_EXPERTS_PER_RANK - 1){
        // If all tokens from all local experts(including both valid tokens and padding tokens) do not exceed local expert buffer capacity, token drop is not triggered. Otherwise triggered.
        // We can use the following condition to determine whether token drop is triggered ONLY when local_experts_tokens_limit is guarantee to be multiple of LOCAL_EXPERTS_PADDING_SIZE!
        if(previous_experts_acc_plus_current_expert_valid_tokens <= local_experts_tokens_limit){
          *token_drop_triggered = 0;
        }else{
          *token_drop_triggered = 1;
        }
      }
    }
#endif
  }

  // Sync within a CUDA block to make sure all the final accumulated previous blocks' local experts' routing info have been produced to the shared memory 
  // before any thread can consume them in scan operation.
  __syncthreads();

#ifndef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
  // First block will need to save all local experts' sum back to output buffer.
  if(blockIdx.x == 0){
    for(int i = threadIdx.x; i < NUM_OF_EXPERTS_PER_RANK; i += NUM_THREADS_PER_BLOCK){
      num_of_local_experts_tokens[i] = all_block_local_experts_sum[i];
    }
  }
#endif
#endif

  // Step 2: Each warp scan the sub-chunk assigned to them(the same sub-chunk as step 0) and produce sparse_to_dense_map, local_expert_routing_map and num_of_tokens_for_experts.
  int32_t previous_token_sum[NUM_OF_RANKS_PER_NODE];

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE 
  // When permute fusion is enabled, each warp will also need to scan and produce dense_chunk_layout and dense_to_expert_map.
  int32_t previous_token_local_experts_sum[NUM_OF_EXPERTS_PER_RANK];
#endif

  // Each warp load the previous blocks' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      previous_token_sum[i] = previous_block_sum[element_id];
    }
  }

  // Each warp accumulate the previous warps' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      for(int j = 0; j < warp_id; j++){
        previous_token_sum[i] += warp_token_routing_map_sum[j][element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous per-rank routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = NUM_OF_RANKS_PER_NODE - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_sum[i] = __shfl_sync(~0, previous_token_sum[src_id], src_tid);
  }

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
  // When permute fusion is enabled, each warp load the previous blocks' local experts' sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      previous_token_local_experts_sum[i] = previous_block_local_experts_sum[element_id];
    }
  }

  // When permute fusion is enabled, each warp accumulate the previous warps' local experts' sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_LOCAL_EXPERTS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_EXPERTS_PER_RANK){
      for(int j = 0; j < warp_id; j++){
        previous_token_local_experts_sum[i] += warp_token_local_experts_routing_map_sum[j][element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous local experts' routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = NUM_OF_EXPERTS_PER_RANK - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_local_experts_sum[i] = __shfl_sync(~0, previous_token_local_experts_sum[src_id], src_tid);
  }
#endif

  // Each warp scan all the tiles within its sub-chunk.
  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then mark it as out-of-bound. 
    int token_out_of_bound = 0;
    if(current_token_id >= num_of_total_attn_tokens){
      token_out_of_bound = 1;
    }
    // If the whole tiles are out-of-bound, the warp just finish and exit the scan loop together.
    if(__all_sync(~0, token_out_of_bound) != 0){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // When permute fusion is enabled, calculate chunk-related info.
    bool first_token_of_a_chunk = (current_token_local_id % NUM_OF_TOKENS_PER_CHUNK) == 0;
    int current_token_global_chunk_id = (current_token_node_rank * NUM_OF_RANKS_PER_NODE + current_token_local_rank) * num_of_chunks_per_rank +
                                        (current_token_local_id / NUM_OF_TOKENS_PER_CHUNK);
    // If this token belongs to a valid attn token chunk, and it is the first token of this chunk, then we need to save this token's per-rank ex-scan of local rank to dense_chunk_layout map.
    bool token_needed_by_dense_chunk_layout = first_token_of_a_chunk && current_token_global_chunk_id > 0 && current_token_global_chunk_id < num_of_total_attn_chunks;
#endif

    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token. Only load when the token is not out-of-bound.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    if(token_out_of_bound == 0){
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }
    }
    
    // Convert the routing map to per rank routing info for current token, 
    // then produce the per-rank final exclusive scan within the warp for this tile.
    int32_t final_ex_scan[NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      int32_t temp_scan = 0;
      bool token_needed_by_this_rank = false;
      // Old warp-level scan implementation, using warp shuffle, suitable for general data type, but not fast enough for bool type.
      // If the token is not out-of-bound, check whether this rank need this token.
      /*if(token_out_of_bound == 0){
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          temp_scan = 1;
        }else{
          temp_scan = 0;
        }
      }
      
      // Each warp perform a inclusive scan from all threads(lanes).
      #pragma unroll
      for(int k = 1; k < WARP_SIZE; k *= 2){
        int32_t temp = __shfl_up_sync(~0, temp_scan, k);
        if(lane_id >= k){
          temp_scan += temp;
        }
      }

      // The inclusive scan from last lane is the sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __shfl_sync(~0, temp_scan, WARP_SIZE - 1);

      // Make scan exclusive.
      int32_t exclusive_scan = __shfl_up_sync(~0, temp_scan, 1);
      temp_scan = (lane_id >= 1) ? exclusive_scan : 0;*/

      // New warp-level scan implementation for bool value, using warp vote instead of warp shuffle. Warp vote is way faster than warp shuffle.
      // If the token is not out-of-bound, check whether this rank need this token.
      if(token_out_of_bound == 0){
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
      }

      // Each warp vote to create a bit mask indicating which token is needed by this rank within this tile.
      unsigned vote_result = __ballot_sync(~0, token_needed_by_this_rank);
      // The sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __popc(vote_result);
      // Each warp perform a exclusive scan from all threads(lanes).
      temp_scan = __popc(vote_result << (WARP_SIZE - lane_id));

      // Calculate the final exclusive scan for current token. -1 represent that the current rank does not need the current token. 
      final_ex_scan[j] = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;

#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
      // When permute fusion is enabled, we need to do extra work to local experts of local rank.
      if(j == local_rank){
        int32_t final_local_experts_ex_scan[NUM_OF_EXPERTS_PER_RANK];
        // First calculate ex-scan for this tile for all local experts of the local rank.
        #pragma unroll
        for(int k = 0; k < NUM_OF_EXPERTS_PER_RANK; k++){
          bool token_needed_by_this_local_expert = false;
          if(token_out_of_bound == 0){
            token_needed_by_this_local_expert = token_routing_map[j * NUM_OF_EXPERTS_PER_RANK + k];
          }
          unsigned local_expert_vote_result = __ballot_sync(~0, token_needed_by_this_local_expert);
          int32_t local_expert_temp_sum = __popc(local_expert_vote_result);
          int32_t local_expert_temp_scan = __popc(local_expert_vote_result << (WARP_SIZE - lane_id));
          final_local_experts_ex_scan[k] = token_needed_by_this_local_expert ? previous_token_local_experts_sum[k] + local_expert_temp_scan : -1;
#ifdef HYBRID_EP_BUILD_TOKEN_DROP_ENABLE
          if(final_local_experts_ex_scan[k] >= local_experts_tokens_limit){
            final_local_experts_ex_scan[k] = -1;
          }
#endif
          previous_token_local_experts_sum[k] += local_expert_temp_sum;
        }
        // Then save the ex-scan back to dense_to_expert map if the current token is needed by local rank.
        if(token_needed_by_this_rank){
          local_experts_write_t* dense_to_expert_map_store_base_addr = reinterpret_cast<local_experts_write_t*>(dense_to_expert_map + final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK);
          #pragma unroll
          for(int k = 0; k < D2E_MAP_STORE_ITER; k++){
            dense_to_expert_map_store_base_addr[k] = *(reinterpret_cast<local_experts_write_t*>(final_local_experts_ex_scan) + k);
          }
        }
        // If condition meet, we also need to save current token's local rank's ex-scan to dense_chunk_layout map.
        if(token_needed_by_dense_chunk_layout){
          dense_chunk_layout[current_token_global_chunk_id - 1] = previous_token_sum[j] + temp_scan;
        }
      }
#else
      // Each thread save local routing map for this token of the local rank to local_expert_routing_map if this token is needed by the local rank.
      if(j == local_rank && token_needed_by_this_rank){
        expert_to_rank_t* local_expert_routing_map_store_base_addr = reinterpret_cast<expert_to_rank_t*>(local_expert_routing_map + (final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK));
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          local_expert_routing_map_store_base_addr[k] = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        }
      }
#endif
      // Accumulate the sum to accumulator.
      previous_token_sum[j] += temp_sum;
      // The thread that processing the global last token save the final sum for current rank to num_of_tokens_for_experts.
      if(current_token_id == num_of_total_attn_tokens - 1 && j == local_rank){
        *num_of_tokens_for_experts = previous_token_sum[j];
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
        // When permute fusion is enabled, also need to save the final sum for current rank to the last element of dense_chunk_layout.
        dense_chunk_layout[num_of_total_attn_chunks - 1] = previous_token_sum[j];
#endif
      }
    }

    // Save final exclusive scan of this token back to sparse_to_dense_map if current token is not out-of-bound and is needed. 
    if(token_out_of_bound == 0 && current_token_local_rank == local_rank){
      // sparse_to_dense_map store base addr for current token.
      write_t* sparse_to_dense_map_store_base_addr = reinterpret_cast<write_t*>(sparse_to_dense_map + 
                                                                                (current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) * NUM_OF_RANKS_PER_NODE);
      #pragma unroll
      for(int j = 0; j < S2D_MAP_STORE_ITER; j++){
        sparse_to_dense_map_store_base_addr[j] = *(reinterpret_cast<write_t*>(final_ex_scan) + j);
      }
    }
  }

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // Step 3: When NUM_OF_NODES > 1, we need to produce attn_to_rdma_map.
  // Since each token(row) is fully independent, each token(row) is assigned to each threads in a interleave pattern.
  if constexpr(NUM_OF_NODES != 1){
    const int num_of_total_token_rows = (NUM_OF_NODES - 1) * num_of_tokens_per_rank;
    //static_assert(NUM_OF_TOTAL_TOKEN_ROWS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_TOKEN_ROWS must be multiple of NUM_OF_TOTAL_THREADS.");
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;

    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;

    //#pragma unroll
    for(int i = 0; i < num_of_token_rows_per_thread; i++){
      int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
      // If the current token is out-of-bound, then just end processing token rows assigned to this thread. 
      if(current_token_id >= num_of_total_token_rows){
        break;
      }
      int current_token_attn_to_rdma_map_node_id = current_token_id % (NUM_OF_NODES - 1);
      int current_token_node_id = current_token_attn_to_rdma_map_node_id < node_rank ? current_token_attn_to_rdma_map_node_id : current_token_attn_to_rdma_map_node_id + 1;
      int current_token_local_id = current_token_id / (NUM_OF_NODES - 1);

      const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                                ((node_rank * NUM_OF_RANKS_PER_NODE + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                                                                                (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                                (current_token_node_id * NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

      bool* attn_to_rdma_map_base_addr = attn_to_rdma_map + (current_token_local_id * (NUM_OF_NODES - 1) + current_token_attn_to_rdma_map_node_id);

      // Load the routing map for current token row.
      bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }

      // Convert the routing map to per rank routing info and then to per node routing info.
      bool token_needed_by_this_node = false;
      #pragma unroll
      for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
        bool token_needed_by_this_rank = false;
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          token_needed_by_this_node = true;
          break;
        }
      }

      *attn_to_rdma_map_base_addr = token_needed_by_this_node;
    }
  }
#endif
}

template< 
        // Hidden size of a token.
        int HIDDEN_DIM,
        // The max num of attn tokens output by a rank/GPU. Used by combine API.
        int MAX_NUM_OF_TOKENS_PER_RANK,
        // Number of ranks/GPU per NVLink domain.
        int NUM_OF_RANKS_PER_NODE,
        // Number of total NVLink domain, i.e. the size of RDMA domain.
        int NUM_OF_NODES,
        // Number of experts running on each rank/GPU. Hybrid-ep support multiple experts running on a single rank/GPU.
        int NUM_OF_EXPERTS_PER_RANK>
class hybrid_ep{
public:

  // Ctor, don't need for now.
  /*hybrid_ep(int local_rank, int node_rank, MPI_Comm comm):
    local_rank_(local_rank),
    node_rank_(node_rank),
    comm_(comm) {}*/

  // Dtor, don't need for now.
  //~hybrid_ep() {}

  // Processing metadata. Calculate routing info needed by dispatch and combine operations.
  // input_routing_map: IO: input, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES]. 
  // Routing map which contain global routing info from all tokens to all expert. Allgather is needed before passing the routing map to this API.
  // preprocessing_tmp: IO: output/input, dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_RANKS_PER_NODE].
  // The temp buffer needed by the preprocessing kernel.
  // preprocessing_local_experts_tmp: IO: output/input, dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_EXPERTS_PER_RANK].
  // The temp buffer needed by the preprocessing kernel when the permute fusion is enabled.
  // sparse_to_dense_map: IO: output, dtype: int32_t, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_NODES, NUM_OF_RANKS_PER_NODE].
  // The routing info needed by NVL warps(i.e. intra-node communication warps) during both dispatch and combine operation. Remains the same in a trainning iteration(FW+BP).
  // rdma_to_attn_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK padded to 16 * NUM_OF_NODES]
  // The routing info mainly needed by RDMA warps during the combine operation. Remains the same in a trainning iteration(FW+BP).
  // attn_to_rdma_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES - 1].
  // The routing info mainly needed by RDMA warps during the dispatch operation. Remains the same in a trainning iteration(FW+BP).
  // num_of_tokens_for_experts: IO: output, dtype: int32_t, shape: [1].
  // The total size of expert buffer on this rank(in number of tokens), according to the global routing map. If there are multiple expert on this rank, each token will only appear once.
  // Remains the same in a trainning iteration(FW+BP).
  // local_expert_routing_map: IO: output, dtype: bool, shape: [at least num_of_tokens_for_experts, NUM_OF_EXPERTS_PER_RANK].
  // The per-expert routing info for all tokens within the expert buffer of this rank. It is used by later layer to routing the tokens to different experts on this rank.
  // Valid only when permute fusion is NOT enabled.
  // Remains the same in a trainning iteration(FW+BP).
  // dense_chunk_layout: IO: output, dtype: int32_t, shape: [num_of_chunks_per_rank * num_of_total_ranks].
  // What's the starting location of each attn token chunk within the local rank's per-rank buffer. dense_chunk_layout[i + 1] - dense_chunk_layout[i] means chunk size of (i + 1) chunk.
  // Valid only when permute fusion is enabled.
  // dense_to_expert_map: IO: output, dtype: int32_t, shape: [at least num_of_tokens_for_experts, NUM_OF_EXPERTS_PER_RANK].
  // The index of each token within the per-rank buffer to the local expert buffer. Valid only when permute fusion is enabled.
  // num_of_local_experts_tokens: IO: output, dtype: int32_t, shape: [NUM_OF_EXPERTS_PER_RANK].
  // How many real token per each local expert w/o padding. Valid only when permute fusion is enabled.
  template<// The padding size of each local expert's output.
           int LOCAL_EXPERTS_PADDING_SIZE,
           // The size of token chunk used in preprocessing kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Block size for preprocessing kernel.
           int NUM_THREADS_PER_BLOCK, 
           // Grid size for preprocessing kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS>
  static void metadata_preprocessing(const bool* input_routing_map, 
                                     tmp_state_t* preprocessing_tmp,
                                     tmp_state_t* preprocessing_local_experts_tmp,
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
                                     const int local_experts_tokens_limit, // This size MUST be multiple of LOCAL_EXPERTS_PADDING_SIZE!
                                     const int num_of_tokens_per_rank,
                                     cudaStream_t stream)
  {
    // Gather routing map from all ranks to all ranks.
    // All ranks should have the same global routing map after this communication.
    // It is a synchronous communication.
    /*MPI_CHECK(MPI_Allgather(reinterpret_cast<const void *>(input_routing_map),
                            NUM_OF_TOKENS_PER_RANK * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES), 
                            MPI_BYTE,
                            reinterpret_cast<void *>(global_routing_map_), 
                            NUM_OF_TOKENS_PER_RANK * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES),
                            MPI_BYTE, 
                            comm_));*/

    // Init preprocessing_tmp buffers.
    constexpr size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * NUM_OF_RANKS_PER_NODE * sizeof(tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(preprocessing_tmp, 0, preprocessing_tmp_sz, stream));
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // When permute fusion is enabled, also init preprocessing_tmp buffers for local expert scan.
    constexpr size_t preprocessing_local_experts_tmp_sz = NUM_OF_BLOCKS * NUM_OF_EXPERTS_PER_RANK * sizeof(tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(preprocessing_local_experts_tmp, 0, preprocessing_local_experts_tmp_sz, stream));
#endif

    // Launch the preprocessing kernel to process the global routing map.
    scan<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, LOCAL_EXPERTS_PADDING_SIZE, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
    <<<NUM_OF_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream>>>
    (input_routing_map, preprocessing_tmp, preprocessing_local_experts_tmp, sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts, local_expert_routing_map, 
    dense_chunk_layout, dense_to_expert_map, num_of_local_experts_tokens, token_drop_triggered, node_rank, local_rank, local_experts_tokens_limit, num_of_tokens_per_rank);

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

  // Dispatch tokens or token gradient to expert MLPs.
  template<// Token data type. Only support uint16_t(represent for BF16) and uint8_t(represent for FP8) for now.
           typename TOKEN_DATA_TYPE,
           // Number of token entry in the shared memory.
           int NUM_OF_STAGES,
           // Number of token entry in the shared memory for permute block.
           int NUM_OF_STAGES_PERMUTE_BLOCK,
           // Number of in-flight S2G token entry in the shared memory, must be smaller than NUM_OF_STAGES.
           int NUM_OF_IN_FLIGHT_S2G,
           // Number of in-flight S2G token entry in the shared memory for permute block, must be smaller than NUM_OF_STAGES_PERMUTE_BLOCK.
           int NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK,
           // The padding size for each local expert.
           int LOCAL_EXPERTS_PADDING_SIZE,
           // Number of additional in-flight S2G entry before notify the permute G2S warp group.
           int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
           // The size of token chunk used in dispatch kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Grid size for dispatch kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Number of CUDA block running permute(1:1 block:SM mapping).
           int NUM_OF_PERMUTE_BLOCKS,
           // Whether the dispatch kernel is used in forward process.
           bool FORWARD_DISPATCH,
           // Whether the dispatch kernel need device-side sync before exit. 
           bool DEVICE_SIDE_SYNC>
  static void dispatch(dispatch_kernel_param_t<TOKEN_DATA_TYPE> param, cudaStream_t stream)
  {
    // The warp groups data type for dispatch kernel, must match the warp groups layout required by the dispatch kernel.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    using INTER_NODE_GROUP = warp_group<1, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 1>;
    using INTRA_NODE_S2G_GROUP = warp_group<2, 2>;
#else
    using INTER_NODE_GROUP = warp_group<0, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 0>;
    using INTRA_NODE_S2G_GROUP = warp_group<3, 1>;
#endif
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    using PERMUTE_G2S_GROUP = warp_group<1, 0>;
    using PERMUTE_S2G_GROUP = warp_group<3, 1>;
#else
    using PERMUTE_G2S_GROUP = warp_group<0, 0>;
    using PERMUTE_S2G_GROUP = warp_group<0, 0>;
#endif
    // The shared memory needed by the dispatch kernel.
    using dispatch_kernel_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK,
                                                                                  NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, FORWARD_DISPATCH>;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    using dispatch_kernel_permute_block_smem_t = dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES_PERMUTE_BLOCK, HIDDEN_DIM,  NUM_OF_EXPERTS_PER_RANK,
                                                                                                              NUM_OF_RANKS_PER_NODE, FORWARD_DISPATCH>;
#endif
    // The dispatch kernel to be launched.
    const auto dispatch_kernel_ptr = dispatch_kernel<TOKEN_DATA_TYPE, INTER_NODE_GROUP, INTRA_NODE_G2S_GROUP, INTRA_NODE_S2G_GROUP, PERMUTE_G2S_GROUP, PERMUTE_S2G_GROUP, NUM_OF_STAGES,
                                                     NUM_OF_STAGES_PERMUTE_BLOCK, NUM_OF_IN_FLIGHT_S2G, NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK, LOCAL_EXPERTS_PADDING_SIZE, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                                                     NUM_OF_TOKENS_PER_CHUNK, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, 
                                                     NUM_OF_BLOCKS, NUM_OF_PERMUTE_BLOCKS, FORWARD_DISPATCH>;

    // Configure dynamic shared memory for the dispatch kernel.
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    constexpr int SMEM_SIZE = sizeof(dispatch_kernel_smem_t) > sizeof(dispatch_kernel_permute_block_smem_t) ? sizeof(dispatch_kernel_smem_t) : sizeof(dispatch_kernel_permute_block_smem_t);
#else
    constexpr int SMEM_SIZE = sizeof(dispatch_kernel_smem_t);
#endif
    CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES, 1, 1>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_permute_flag_value);

#ifndef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // If the permute fusion is not enabled, we need to launch device_sync_kernel before AND after the dispatch kernel if DEVICE_SIDE_SYNC is true.
    // If the permute fusion is enabled, only need to launch device_sync_kernel after the dispatch kernel.
    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<NUM_OF_RANKS_PER_NODE><<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value, param.intra_node_flag_parity);
    }
#endif
    // Launch dispatch kernel.
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    static_assert(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() == PERMUTE_G2S_GROUP::size() + PERMUTE_S2G_GROUP::size(), "Dispatch blocks and permute block should have the same size.");
    constexpr int NUM_OF_BLOCKS_TOTAL = NUM_OF_BLOCKS + NUM_OF_PERMUTE_BLOCKS;
#else
    constexpr int NUM_OF_BLOCKS_TOTAL = NUM_OF_BLOCKS;
#endif
    constexpr int BLOCK_DIM = INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size();
    dispatch_kernel_ptr<<<NUM_OF_BLOCKS_TOTAL, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<NUM_OF_RANKS_PER_NODE><<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value, param.intra_node_flag_parity);
    }
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifndef USE_NIXL
    // RDMA sync is needed.
    rdma_sync_kernel<<<1, 1, 0, stream>>>(NUM_OF_NODES, param.node_rank, param.expected_rdma_flag_value,
                                                   param.rdma_inter_node_group_flags, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<dispatch_memory_region_info_t*>(param.multinode_aux_ptr));
#endif
#endif
    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

  // Combine tokens or token gradient from expert MLPs.
  template<// Number of token entry in the shared memory for G2S TMA.
           int NUM_OF_STAGES_G2S,
           // Number of token entry in the shared memory for S2G TMA.
           int NUM_OF_STAGES_S2G,
           // Number of token entry in the shared memory for G2S operations for unpermute block.
           int NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK,
           // Number of token entry in the shared memory for S2G operations for unpermute block.
           int NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK,
           // The size of token chunk used in combine kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Number of token per group in the inter-node reduction/G2S warp group.
           int NUM_OF_TOKENS_PER_GROUP,
           // Grid size for combine kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Number of CUDA block running unpermute(1:1 block:SM mapping).
           int NUM_OF_UNPERMUTE_BLOCKS,
           // Number of fully in-flight S2G in intra-node reduction warp group.
           int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
           // Number of fully in-flight S2G in unpermute reduction warp group.
           int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCKS,
           // Whether the combine kernel is used in backward process.
           bool BACKWARD_COMBINE,
           // Whether the combine kernel need device-side sync before launch.
           bool DEVICE_SIDE_SYNC>
  static void combine(combine_kernel_param_t param, cudaStream_t stream)
  {
    // The warp groups data type for combine kernel, must match the warp groups layout required by the combine kernel.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    using INTRA_NODE_RED_GROUP = warp_group<4, 0>;
    using INTER_NODE_RED_GROUP = warp_group<4, 4>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 8>;
    using INTER_NODE_G2S_GROUP = warp_group<1, 9>;
    using INTER_NODE_RDMA_GROUP = warp_group<1, 10>;
    constexpr int NUM_OF_DATA_PIPELINE_PER_BLOCK = 1;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    using UNPERMUTE_RED_GROUP = warp_group<4, 0>;
    using UNPERMUTE_G2S_GROUP = warp_group<1, 4>;
#else
    using UNPERMUTE_RED_GROUP = warp_group<0, 0>;
    using UNPERMUTE_G2S_GROUP = warp_group<0, 0>;
#endif
#else
    using INTRA_NODE_RED_GROUP = warp_group<0, 0>;
    using INTER_NODE_RED_GROUP = warp_group<4, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<0, 4>;
    using INTER_NODE_G2S_GROUP = warp_group<2, 4>;
    using INTER_NODE_RDMA_GROUP = warp_group<0, 6>;
    constexpr int NUM_OF_DATA_PIPELINE_PER_BLOCK = 2;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    using UNPERMUTE_RED_GROUP = warp_group<4, 0>;
    using UNPERMUTE_G2S_GROUP = warp_group<2, 4>;
#else
    using UNPERMUTE_RED_GROUP = warp_group<0, 0>;
    using UNPERMUTE_G2S_GROUP = warp_group<0, 0>;
#endif
#endif
    static_assert(INTER_NODE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK, "Inter-node G2S warp group pipeline and inter-node red warp group pipeline mismatch.");
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    static_assert(UNPERMUTE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK, "Unpermute G2S warp group pipeline and unpermute red warp group pipeline mismatch.");
#endif

    // The shared memory needed by the combine kernel.
    using combine_kernel_smem_t = combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK,
                                                                                NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, BACKWARD_COMBINE>;
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    using combine_kernel_unpermute_block_smem_t = combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK, NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK, HIDDEN_DIM, 
                                                                                                                NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, BACKWARD_COMBINE>;
#endif
    // The combine kernel to be launched.
    const auto combine_kernel_ptr = combine_kernel<INTRA_NODE_RED_GROUP, INTER_NODE_RED_GROUP, INTRA_NODE_G2S_GROUP, INTER_NODE_G2S_GROUP, INTER_NODE_RDMA_GROUP, UNPERMUTE_G2S_GROUP, 
                                                   UNPERMUTE_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK, 
                                                   NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK, NUM_OF_TOKENS_PER_GROUP, NUM_OF_TOKENS_PER_CHUNK, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK,
                                                   NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_UNPERMUTE_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, 
                                                   NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCKS, BACKWARD_COMBINE>;

    // Configure dynamic shared memory for the combine kernel.
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    constexpr int SMEM_SIZE = sizeof(combine_kernel_smem_t) > sizeof(combine_kernel_unpermute_block_smem_t) ? sizeof(combine_kernel_smem_t) : sizeof(combine_kernel_unpermute_block_smem_t);
#else
    constexpr int SMEM_SIZE = sizeof(combine_kernel_smem_t);
#endif
    CUDA_CHECK(cudaFuncSetAttribute(combine_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES, 1, NUM_OF_DATA_PIPELINE_PER_BLOCK>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_unpermute_flag_value);

#ifndef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    // If the permute fusion is not enabled, we need to launch device_sync_kernel before AND after the combine kernel if DEVICE_SIDE_SYNC is true.
    // If the permute fusion is enabled, only need to launch device_sync_kernel after the combine kernel.
    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<NUM_OF_RANKS_PER_NODE><<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value, param.intra_node_flag_parity);
    }
#endif

    // Launch combine kernel.
#ifdef HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE
    constexpr int NUM_OF_BLOCKS_TOTAL = NUM_OF_BLOCKS + NUM_OF_UNPERMUTE_BLOCKS;
#else
    constexpr int NUM_OF_BLOCKS_TOTAL = NUM_OF_BLOCKS;
#endif
    constexpr int BLOCK_DIM = INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size();
    combine_kernel_ptr<<<NUM_OF_BLOCKS_TOTAL, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<NUM_OF_RANKS_PER_NODE><<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value, param.intra_node_flag_parity);
    }

    // RDMA sync is needed for inter-node scenario.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#ifndef USE_NIXL
    rdma_sync_kernel<<<1, 1, 0, stream>>>(NUM_OF_NODES, param.node_rank, param.expected_rdma_flag_value,
                                                   param.rdma_inter_node_group_flags, reinterpret_cast<doca_gpu_dev_verbs_qp**>(param.multinode_ctx_ptr), reinterpret_cast<combine_memory_region_info_t*>(param.multinode_aux_ptr));
#endif
#endif
    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }



  /*private:
  // Rank within the current node/host.
  int local_rank_; 
  // Rank for the current node/host.
  int node_rank_;

  // MPI Communicator for out-of-bond communication.
  // This is used to gather routing map from all other ranks, so the communicator should contains all ranks.
  MPI_Comm comm_;

  // The global routing map which collected from all other ranks, remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES].
  bool* global_routing_map_;
  // The temp buffer needed by the preprocessing kernel.
  // dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_RANKS_PER_NODE].
  tmp_state_t* preprocessing_tmp_;
  // The routing info needed by NVL warps(i.e. intra-node communication warps) during both dispatch and combine operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: int32_t, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_NODES, NUM_OF_RANKS_PER_NODE].
  int32_t* sparse_to_dense_map_;
  // The routing info mainly needed by RDMA warps during the combine operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK padded to 16 * NUM_OF_NODES].
  bool* rdma_to_attn_map_;
  // The routing info mainly needed by RDMA warps during the dispatch operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES - 1].
  bool* attn_to_rdma_map_;
  // The total size of expert input/output buffer on this rank(in number of tokens), according to the global routing map.
  // If there are multiple expert on this rank, each token will only appear once.
  // Remains the same in a trainning iteration(FW+BP).
  int32_t* num_of_tokens_for_experts_;
  // The per-expert routing info for all tokens within the expert input/output buffer of this rank.
  // It is used by later layer to routing the tokens to different experts on this rank.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [at least num_of_tokens_for_experts_, NUM_OF_EXPERTS_PER_RANK].
  bool* local_expert_routing_map_;*/
};
} // namespace hybrid_ep
