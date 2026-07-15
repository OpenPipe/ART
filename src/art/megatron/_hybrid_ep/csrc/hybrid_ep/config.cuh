// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

#pragma once
#include <tuple>
#include <map>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <optional>
#include "utils.cuh"

// Now we support up to 72(GB200) ranks per node.
// This will be used to initialize the template param_t for communication kernel.
#define MAX_NUM_OF_RANKS_PER_NODE 72

static constexpr int64_t HYBRID_EP_IB_QP_MAX_TX_DEPTH = 65536;
static constexpr int64_t HYBRID_EP_DISPATCH_TX_DEPTH_TOKEN_FACTOR = 3;
static constexpr int64_t HYBRID_EP_DISPATCH_TX_DEPTH_EXTRA = 1;

static inline bool hybrid_ep_token_capacity_is_valid(
    int max_num_of_tokens_per_rank, int num_of_nodes, const char* config_name) {
  if (num_of_nodes <= 1) {
    return true;
  }

  int64_t tx_depth =
      HYBRID_EP_DISPATCH_TX_DEPTH_TOKEN_FACTOR *
          static_cast<int64_t>(max_num_of_tokens_per_rank) +
      HYBRID_EP_DISPATCH_TX_DEPTH_EXTRA;
  if (tx_depth < HYBRID_EP_IB_QP_MAX_TX_DEPTH) {
    return true;
  }

  int max_num_of_tokens_per_rank_supported = static_cast<int>(
      (HYBRID_EP_IB_QP_MAX_TX_DEPTH - 1 - HYBRID_EP_DISPATCH_TX_DEPTH_EXTRA) /
      HYBRID_EP_DISPATCH_TX_DEPTH_TOKEN_FACTOR);
  fprintf(stderr,
          "[Error] Invalid %s: max_num_of_tokens_per_rank=%d exceeds the supported "
          "maximum number of %d tokens in multi-node mode (required IB QP tx depth %ld, "
          "limit < %ld).\n",
          config_name,
          max_num_of_tokens_per_rank,
          max_num_of_tokens_per_rank_supported,
          static_cast<long>(tx_depth),
          static_cast<long>(HYBRID_EP_IB_QP_MAX_TX_DEPTH));
  fflush(stderr);
  return false;
}

static inline int hybrid_ep_pad_num_of_tokens_per_rank(
    int max_num_of_tokens_per_rank, int num_of_tokens_per_chunk_combine_api) {
  if (num_of_tokens_per_chunk_combine_api <= 0) {
    return max_num_of_tokens_per_rank;
  }
  return static_cast<int>(
      (static_cast<int64_t>(max_num_of_tokens_per_rank) +
       num_of_tokens_per_chunk_combine_api - 1) /
      num_of_tokens_per_chunk_combine_api *
      num_of_tokens_per_chunk_combine_api);
}

// Config used for buffer allocation.
struct BufferConfig {
  int hidden_dim;
  int max_num_of_tokens_per_rank;
  int num_of_experts_per_rank;
  int num_of_ranks_per_node;
  int num_of_nodes;
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_blocks_preprocessing_api;
  int num_of_blocks_dispatch_api;
  int num_of_blocks_combine_api;
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_tokens_per_chunk_combine_api;
  /** Number of chunks, used for buffer sizing; grow_to on this triggers reallocate when chunk size shrinks. */
  int num_of_dispatch_chunks;
  int num_of_combine_chunks;

  /*
   *  Validation check
   */
   bool is_valid(){
    bool valid = true;
    if (token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
      valid &= (hidden_dim % 512 == 0); // Make TMA work in scaling factor.
    } else {
      valid &= (hidden_dim % 16 == 0); // Make TMA work.
    }
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    // TMA requires (num_of_tokens_per_chunk * num_of_ranks_per_node * 4) % 16 == 0
    valid &= ((num_of_tokens_per_chunk_dispatch_api * num_of_ranks_per_node) % 4 == 0);
    valid &= hybrid_ep_token_capacity_is_valid(
        max_num_of_tokens_per_rank, num_of_nodes, "BufferConfig");
    if(!valid){
      fprintf(stderr, "[Error] Invalid BufferConfig: hidden_dim=%d, num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_tokens_per_chunk_dispatch_api=%d\n", 
              hidden_dim, num_of_experts_per_rank, num_of_ranks_per_node, num_of_tokens_per_chunk_dispatch_api);
      fflush(stderr);
    }
    return valid;
  }
};

// Config used for hybrid-ep kernel.
struct HybridEpConfigInstance {
  /*
   *  Hybrid-ep Config
   */
  int hidden_dim;
  int max_num_of_tokens_per_rank;
  int num_of_experts_per_rank;
  int num_of_ranks_per_node;
  int num_of_nodes;
  int pad_multiple;

  /*
   *  Metadata-preprocessing API Config
   */
  int num_of_tokens_per_chunk_preprocessing_api;
  int num_of_threads_per_block_preprocessing_api;
  int num_of_blocks_preprocessing_api;

  // In standalone permute kernel. it is the number of CUDA blocks running permute kernel.
  // In fused permute-dispatch kernel. it is the number of CUDA blocks for permute part in the fused kernel.
  int num_of_blocks_permute;
  int num_of_blocks_unpermute;

  /*
   *  Dispatch API Config
   */
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_stages_dispatch_api;
  int num_of_stages_permute_block_dispatch_api;
  int num_of_in_flight_s2g_dispatch_api;
  int num_of_in_flight_s2g_permute_block_dispatch_api;
  int num_of_additional_in_flight_s2g_dispatch_api;
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_blocks_dispatch_api;
  bool forward_dispatch_api;
  bool device_side_sync_dispatch_api = true;

  /*
   *  Combine API Config
   */
  int num_of_stages_g2s_combine_api;
  int num_of_stages_s2g_combine_api;
  int num_of_stages_g2s_unpermute_block;
  int num_of_stages_s2g_unpermute_block;
  int num_of_tokens_per_chunk_combine_api;
  int num_of_tokens_per_group_combine_api;
  int num_of_blocks_combine_api;
  int num_of_additional_in_flight_s2g_combine_api;
  int num_of_additional_in_flight_s2g_unpermute_block_combine_api;
  bool backward_combine_api;
  bool device_side_sync_combine_api = true;

  /*
   *  Validation check
   */
  bool is_valid(bool fuse_permute_dispatch = false){
    bool valid = true;
    if (token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
      valid &= (hidden_dim % 512 == 0); // Make TMA work in scaling factor.
    } else {
      valid &= (hidden_dim % 16 == 0); // Make TMA work.
    }
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    // TMA requires (num_of_tokens_per_chunk * num_of_ranks_per_node * 4) % 16 == 0
    valid &= ((num_of_tokens_per_chunk_dispatch_api * num_of_ranks_per_node) % 4 == 0);
    valid &= hybrid_ep_token_capacity_is_valid(
        max_num_of_tokens_per_rank, num_of_nodes, "HybridEpConfigInstance");
    // In fuse mode, all chunk sizes must be the same
    if (fuse_permute_dispatch) {
      bool chunk_match = (num_of_tokens_per_chunk_dispatch_api == num_of_tokens_per_chunk_combine_api)
                      && (num_of_tokens_per_chunk_dispatch_api == num_of_tokens_per_chunk_preprocessing_api);
      if (!chunk_match) {
        fprintf(stderr, "[Error] Fuse mode requires identical chunk sizes: dispatch=%d, combine=%d, preprocessing=%d\n",
                num_of_tokens_per_chunk_dispatch_api, num_of_tokens_per_chunk_combine_api, num_of_tokens_per_chunk_preprocessing_api);
        fflush(stderr);
      }
      valid &= chunk_match;
    }
    if(!valid){
      fprintf(stderr, "[Error] Invalid HybridEpConfigInstance: hidden_dim=%d, num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_tokens_per_chunk_dispatch_api=%d\n", 
              hidden_dim, num_of_experts_per_rank, num_of_ranks_per_node, num_of_tokens_per_chunk_dispatch_api);
      fflush(stderr);
    }
    return valid;
  }

  bool operator<(const HybridEpConfigInstance& other) const {
    return std::memcmp(this, &other, sizeof(HybridEpConfigInstance)) < 0;
  }
};

static int get_env_int(const char* name, int default_value) {
    const char* val = getenv(name);
    return val ? atoi(val) : default_value;
}

// Helper to simulate C++ struct layout with alignas fields.
// Usage: call add(size, align) for each field in declaration order,
// then call total() to get the final struct size (with trailing padding).
struct SmemLayoutBuilder {
    int64_t offset = 0;
    int max_align = 1;

    void add(int64_t size, int align) {
        // Align current offset to the field's alignment requirement
        offset = ((offset + align - 1) / align) * align;
        offset += size;
        if (align > max_align) max_align = align;
    }

    int64_t total() {
        // Pad to the maximum alignment seen (struct trailing padding)
        return ((offset + max_align - 1) / max_align) * max_align;
    }
};

// Computed shared memory sizes for dispatch, permute_block, combine, and unpermute_block kernels.
struct SmemSizes {
    int64_t dispatch;
    int64_t permute_block;
    int64_t combine;
    int64_t unpermute_block;
};

// Compute the dynamic shared memory sizes for all kernel types in one call.
// Mirrors the struct layouts in hybrid_ep_backend.cuh:
//   - dispatch_kernel_dynamic_shared_memory_buffer_t
//   - dispatch_kernel_permute_block_dynamic_shared_memory_buffer_t
//   - combine_kernel_dynamic_shared_memory_buffer_t
//   - combine_kernel_unpermute_block_dynamic_shared_memory_buffer_t
static SmemSizes compute_smem_sizes(const HybridEpConfigInstance& c) {
    static std::map<HybridEpConfigInstance, SmemSizes> cache;
    auto it = cache.find(c);
    if (it != cache.end()) return it->second;

    SmemSizes result;
    bool is_fp8 = (c.token_data_type == APP_TOKEN_DATA_TYPE::UINT8);
    int token_size = is_fp8 ? 1 : 2;
    bool multinode = (c.num_of_nodes > 1);

    // --- dispatch kernel ---
    {
        SmemLayoutBuilder b;
        b.add((int64_t)c.num_of_stages_dispatch_api * c.hidden_dim * token_size, 128);
        b.add((int64_t)2 * c.num_of_tokens_per_chunk_dispatch_api * c.num_of_ranks_per_node * 4, 128);
        if (c.forward_dispatch_api)
            b.add((int64_t)c.num_of_stages_dispatch_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
        if (is_fp8)
            b.add((int64_t)c.num_of_stages_dispatch_api * (c.hidden_dim / 128) * 4, 16);
        if (multinode)
            b.add((int64_t)c.num_of_tokens_per_chunk_dispatch_api * (c.num_of_nodes - 1), 16);
        b.add((int64_t)c.num_of_stages_dispatch_api * 2 * 8, 8);
        b.add((int64_t)2 * 8, 8);
        b.add((int64_t)8, 8);
        if (multinode)
            b.add((int64_t)(c.num_of_nodes - 1) * 96, 8);
        if (multinode)
            b.add((int64_t)(c.num_of_nodes - 1) * 4, 4);
        result.dispatch = b.total();
    }

    // --- dispatch permute_block kernel ---
    {
        SmemLayoutBuilder b;
        b.add((int64_t)c.num_of_stages_permute_block_dispatch_api * c.hidden_dim * token_size, 128);
        if (c.forward_dispatch_api)
            b.add((int64_t)c.num_of_stages_permute_block_dispatch_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
        if (is_fp8)
            b.add((int64_t)c.num_of_stages_permute_block_dispatch_api * (c.hidden_dim / 128) * 4, 16);
        b.add((int64_t)c.num_of_stages_permute_block_dispatch_api * 2 * 8, 8);
        result.permute_block = b.total();
    }

    // --- combine kernel (always uint16_t, 2 bytes per token) ---
    {
        SmemLayoutBuilder b;
        if (multinode) {
            b.add((int64_t)c.num_of_stages_g2s_combine_api * c.hidden_dim * 2, 128);
            b.add((int64_t)c.num_of_stages_s2g_combine_api * c.hidden_dim * 2, 128);
        }
        b.add((int64_t)c.num_of_stages_g2s_combine_api * c.hidden_dim * 2, 128);
        b.add((int64_t)c.num_of_stages_s2g_combine_api * c.hidden_dim * 2, 128);
        if (c.backward_combine_api) {
            if (multinode) {
                b.add((int64_t)c.num_of_stages_g2s_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
                b.add((int64_t)c.num_of_stages_s2g_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
                b.add((int64_t)c.num_of_stages_g2s_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
                b.add((int64_t)c.num_of_stages_s2g_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * c.num_of_nodes * 4, 16);
            } else {
                b.add((int64_t)c.num_of_stages_g2s_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
                b.add((int64_t)c.num_of_stages_s2g_combine_api * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
            }
        }
        if (multinode)
            b.add((int64_t)c.num_of_stages_g2s_combine_api * 2 * 8, 8);
        b.add((int64_t)c.num_of_stages_g2s_combine_api * 2 * 8, 8);
        if (multinode) {
            b.add((int64_t)(c.num_of_nodes - 1) * (c.max_num_of_tokens_per_rank / c.num_of_tokens_per_chunk_combine_api) * 8, 8);
            b.add((int64_t)(c.num_of_nodes - 1) * 72, 8);
            b.add((int64_t)(c.num_of_nodes - 1) * 4, 4);
        }
        if (multinode)
            b.add((int64_t)c.num_of_stages_g2s_combine_api, 1);
        b.add((int64_t)c.num_of_stages_g2s_combine_api, 1);
        result.combine = b.total();
    }

    // --- combine unpermute_block kernel ---
    {
        SmemLayoutBuilder b;
        b.add((int64_t)c.num_of_stages_g2s_unpermute_block * c.hidden_dim * 2, 128);
        b.add((int64_t)c.num_of_stages_s2g_unpermute_block * c.hidden_dim * 2, 128);
        if (c.backward_combine_api) {
            b.add((int64_t)c.num_of_stages_s2g_unpermute_block * c.num_of_experts_per_rank * c.num_of_ranks_per_node * 4, 16);
            b.add((int64_t)c.num_of_stages_g2s_unpermute_block * 4, 16);
        }
        b.add((int64_t)c.num_of_stages_g2s_unpermute_block * 2 * 8, 8);
        if (c.backward_combine_api)
            b.add((int64_t)c.num_of_stages_g2s_unpermute_block * 4, 4);
        b.add((int64_t)c.num_of_stages_g2s_unpermute_block, 1);
        result.unpermute_block = b.total();
    }

    cache[c] = result;
    return result;
}

class Configurer {
public:
    BufferConfig buffer_config;
    int max_smem_per_block;  // Device max dynamic shared memory per block (optin)

    int sm_count;
    int num_blocks_permute_;
    int num_blocks_unpermute_;

    Configurer(
        int hidden_dim,
        int max_num_of_tokens_per_rank,
        int num_local_experts,
        int num_of_ranks_per_node,
        int num_of_nodes,
        bool use_fp8 = false,
        std::optional<int> num_sms_dispatch_api = std::nullopt,
        std::optional<int> num_sms_combine_api = std::nullopt,
        std::optional<int> num_sms_preprocessing_api = std::nullopt,
        std::optional<int> num_blocks_permute = std::nullopt,
        std::optional<int> num_blocks_unpermute = std::nullopt
    ) {
        // Auto-detect SM count and max shared memory
        cudaDeviceProp props;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        sm_count = props.multiProcessorCount;
        max_smem_per_block = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

        int sms_preprocessing = num_sms_preprocessing_api.value_or(108);
        int sms_dispatch = num_sms_dispatch_api.value_or((num_of_nodes == 1) ? 24 : 8);
        int sms_combine = num_sms_combine_api.value_or((num_of_nodes == 1) ? 24 : 8);
        num_blocks_permute_ = num_blocks_permute.value_or(-1);
        num_blocks_unpermute_ = num_blocks_unpermute.value_or(-1);

        assert(sm_count >= sms_dispatch
            && sm_count >= sms_combine);

        // Fill BufferConfig
        buffer_config.hidden_dim = hidden_dim;
        buffer_config.num_of_experts_per_rank = num_local_experts;
        buffer_config.num_of_ranks_per_node = num_of_ranks_per_node;
        buffer_config.num_of_nodes = num_of_nodes;
        buffer_config.num_of_blocks_dispatch_api = sms_dispatch;
        buffer_config.num_of_blocks_combine_api = sms_combine;
        buffer_config.num_of_blocks_preprocessing_api = sms_preprocessing;
        buffer_config.token_data_type = use_fp8 ? APP_TOKEN_DATA_TYPE::UINT8 : APP_TOKEN_DATA_TYPE::UINT16;
        buffer_config.num_of_tokens_per_chunk_dispatch_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", 64);
        buffer_config.num_of_tokens_per_chunk_combine_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", 64);
        buffer_config.max_num_of_tokens_per_rank = hybrid_ep_pad_num_of_tokens_per_rank(
            std::max(max_num_of_tokens_per_rank, 512),
            buffer_config.num_of_tokens_per_chunk_combine_api);
        buffer_config.num_of_dispatch_chunks = (buffer_config.max_num_of_tokens_per_rank - 1)
            / buffer_config.num_of_tokens_per_chunk_dispatch_api + 1;
        buffer_config.num_of_combine_chunks = (buffer_config.max_num_of_tokens_per_rank - 1)
            / buffer_config.num_of_tokens_per_chunk_combine_api + 1;

        if (!buffer_config.is_valid()) {
            fprintf(stderr, "[Error] Configurer: invalid buffer config. hidden_dim=%d, max_num_of_tokens_per_rank=%d, "
                    "num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_nodes=%d\n",
                    hidden_dim, max_num_of_tokens_per_rank, num_local_experts, num_of_ranks_per_node, num_of_nodes);
            fflush(stderr);
            throw std::runtime_error("The buffer config is not valid.");
        }
    }

    HybridEpConfigInstance get_default_config(bool fuse_permute_dispatch = false) {
        HybridEpConfigInstance config;
        // Defaults from buffer_config (can be overridden per-call)
        config.hidden_dim = buffer_config.hidden_dim;
        config.max_num_of_tokens_per_rank = buffer_config.max_num_of_tokens_per_rank;
        config.num_of_experts_per_rank = buffer_config.num_of_experts_per_rank;
        config.num_of_ranks_per_node = buffer_config.num_of_ranks_per_node;
        config.num_of_nodes = buffer_config.num_of_nodes;

        // Semi-static from buffer_config
        config.num_of_blocks_preprocessing_api = buffer_config.num_of_blocks_preprocessing_api;
        config.num_of_blocks_dispatch_api = buffer_config.num_of_blocks_dispatch_api;
        config.num_of_blocks_combine_api = buffer_config.num_of_blocks_combine_api;
        config.token_data_type = buffer_config.token_data_type;

        // Env-var defaults (runtime chunk sizes use 64, different from buffer's 32)
        config.num_of_threads_per_block_preprocessing_api = get_env_int("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", 256);
        int default_chunk_size = 64;
        config.num_of_tokens_per_chunk_preprocessing_api  = get_env_int("NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API", default_chunk_size);
        config.forward_dispatch_api = true;
        config.device_side_sync_dispatch_api = true;
        config.num_of_stages_dispatch_api = get_env_int("NUM_OF_STAGES_DISPATCH_API", 10);
        config.num_of_stages_permute_block_dispatch_api = get_env_int("NUM_OF_STAGES_PERMUTE_BLOCK_DISPATCH_API", 10);
        config.num_of_in_flight_s2g_dispatch_api = get_env_int("NUM_OF_IN_FLIGHT_S2G_DISPATCH_API", 8);
        config.num_of_in_flight_s2g_permute_block_dispatch_api = get_env_int("NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK_DISPATCH_API", 8);
        config.num_of_additional_in_flight_s2g_dispatch_api = get_env_int("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_DISPATCH_API", 6);
        config.num_of_tokens_per_chunk_dispatch_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", default_chunk_size);

        config.backward_combine_api = true;
        config.device_side_sync_combine_api = true;
        config.num_of_stages_g2s_combine_api = get_env_int("NUM_OF_STAGES_G2S_COMBINE_API",
            buffer_config.num_of_nodes > 1 ? 5 : 10);
        config.num_of_stages_s2g_combine_api = get_env_int("NUM_OF_STAGES_S2G_COMBINE_API", 2);
        config.num_of_stages_g2s_unpermute_block = get_env_int("NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK", 2);
        config.num_of_stages_s2g_unpermute_block = get_env_int("NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK", 2);
        config.num_of_tokens_per_chunk_combine_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", default_chunk_size);
        config.num_of_tokens_per_group_combine_api = get_env_int("NUM_OF_TOKENS_PER_GROUP_COMBINE_API", 4);
        config.num_of_additional_in_flight_s2g_combine_api = get_env_int("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API", 2);
        config.num_of_additional_in_flight_s2g_unpermute_block_combine_api = get_env_int("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCK_COMBINE_API", 2);
        
        config.pad_multiple = 1;

        // If we use the fused permute-dispatch kernel, the number of blocks
        // for the permute part is the same as the number of blocks for the dispatch part.
        if (fuse_permute_dispatch) {
            config.num_of_blocks_permute = min(108, sm_count - config.num_of_blocks_dispatch_api);
            config.num_of_blocks_unpermute = min(108, sm_count - config.num_of_blocks_combine_api);
        }else{
            config.num_of_blocks_permute = sm_count * 16;
            config.num_of_blocks_unpermute = sm_count * 16;
        }

        // Update num_of_blocks_permute and num_of_blocks_unpermute with predefined values
        if (num_blocks_permute_ >= 0) 
            config.num_of_blocks_permute = num_blocks_permute_;
        if (num_blocks_unpermute_ >= 0) 
            config.num_of_blocks_unpermute = num_blocks_unpermute_;
        return config;
    }

    // Adjust template parameters (stages, in-flight counts) so that kernel
    // shared memory fits within the device limit. Reduces stages down to
    // MIN_STAGES, then errors if still too large.
    void adjust_template(HybridEpConfigInstance& config, bool fuse_permute_dispatch = false) {
        config.max_num_of_tokens_per_rank = hybrid_ep_pad_num_of_tokens_per_rank(
            config.max_num_of_tokens_per_rank,
            config.num_of_tokens_per_chunk_combine_api);

        const int max_smem = max_smem_per_block;
        constexpr int MIN_STAGES = 2;

        // Ensure val < limit (in-flight must be strictly less than stages)
        auto clamp_below = [](int& val, int limit) {
            if (val >= limit) val = limit - 1;
        };
        // Snap val down to the nearest multiple of align, but not below floor.
        // combine/unpermute g2s and s2g stages must satisfy:
        //   % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0  (always 2)
        //   % warp_group::warp_size() == 0          (always 2)
        // so align = 2 covers both constraints.
        constexpr int STAGE_ALIGN = 2;
        auto align_down = [](int& val, int align, int floor) {
            val = std::max(floor, (val / align) * align);
        };
        // Effective dispatch smem: in fuse mode, take max of dispatch and permute_block
        auto dispatch_smem = [&]() -> int64_t {
            auto s = compute_smem_sizes(config);
            return fuse_permute_dispatch ? std::max(s.dispatch, s.permute_block) : s.dispatch;
        };
        // Effective combine smem: in fuse mode, take max of combine and unpermute_block
        auto combine_smem = [&]() -> int64_t {
            auto s = compute_smem_sizes(config);
            return fuse_permute_dispatch ? std::max(s.combine, s.unpermute_block) : s.combine;
        };

        // 1. Dispatch: reduce stages
        while (dispatch_smem() > max_smem && config.num_of_stages_dispatch_api > MIN_STAGES)
            config.num_of_stages_dispatch_api--;
        clamp_below(config.num_of_in_flight_s2g_dispatch_api, config.num_of_stages_dispatch_api);
        clamp_below(config.num_of_additional_in_flight_s2g_dispatch_api, config.num_of_stages_dispatch_api);

        // 2. Permute block (fuse permute-dispatch mode only): reduce stages independently
        if (fuse_permute_dispatch) {
            while (compute_smem_sizes(config).permute_block > max_smem
                   && config.num_of_stages_permute_block_dispatch_api > MIN_STAGES)
                config.num_of_stages_permute_block_dispatch_api--;
            clamp_below(config.num_of_in_flight_s2g_permute_block_dispatch_api,
                        config.num_of_stages_permute_block_dispatch_api);
        }

        // 3. Combine: alternately reduce g2s / s2g stages
        bool reduce_g2s = true;
        while (combine_smem() > max_smem
               && (config.num_of_stages_g2s_combine_api > MIN_STAGES
                   || config.num_of_stages_s2g_combine_api > MIN_STAGES)) {
            if (reduce_g2s && config.num_of_stages_g2s_combine_api > MIN_STAGES)
                config.num_of_stages_g2s_combine_api--;
            else if (config.num_of_stages_s2g_combine_api > MIN_STAGES)
                config.num_of_stages_s2g_combine_api--;
            reduce_g2s = !reduce_g2s;
        }
        // Snap to nearest legal multiple (must be divisible by 2)
        align_down(config.num_of_stages_g2s_combine_api, STAGE_ALIGN, MIN_STAGES);
        align_down(config.num_of_stages_s2g_combine_api, STAGE_ALIGN, MIN_STAGES);
        clamp_below(config.num_of_additional_in_flight_s2g_combine_api,
                    config.num_of_stages_s2g_combine_api);

        // 4. Unpermute block (fuse unpermute-combine mode only): alternately reduce g2s / s2g stages
        if (fuse_permute_dispatch) {
            bool reduce_g2s_up = true;
            while (compute_smem_sizes(config).unpermute_block > max_smem
                   && (config.num_of_stages_g2s_unpermute_block > MIN_STAGES
                       || config.num_of_stages_s2g_unpermute_block > MIN_STAGES)) {
                if (reduce_g2s_up && config.num_of_stages_g2s_unpermute_block > MIN_STAGES)
                    config.num_of_stages_g2s_unpermute_block--;
                else if (config.num_of_stages_s2g_unpermute_block > MIN_STAGES)
                    config.num_of_stages_s2g_unpermute_block--;
                reduce_g2s_up = !reduce_g2s_up;
            }
            // Snap to nearest legal multiple (must be divisible by 2)
            align_down(config.num_of_stages_g2s_unpermute_block, STAGE_ALIGN, MIN_STAGES);
            align_down(config.num_of_stages_s2g_unpermute_block, STAGE_ALIGN, MIN_STAGES);
            clamp_below(config.num_of_additional_in_flight_s2g_unpermute_block_combine_api,
                        config.num_of_stages_s2g_unpermute_block);
        }

        // 5. Final validation
        int64_t final_dispatch = dispatch_smem();
        int64_t final_combine = combine_smem();
        if (final_dispatch > max_smem || final_combine > max_smem) {
            fprintf(stderr, "[Error] adjust_template: smem exceeds device limit (%d B)."
                    " dispatch=%ld, combine=%ld\n", max_smem, (long)final_dispatch, (long)final_combine);
            fflush(stderr);
            throw std::runtime_error("Cannot fit kernels into shared memory even with minimum stages.");
        }
        buffer_config.max_num_of_tokens_per_rank = std::max(
            buffer_config.max_num_of_tokens_per_rank,
            config.max_num_of_tokens_per_rank);
    }
};
