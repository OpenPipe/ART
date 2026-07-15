// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "permute.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

template void permute_launcher<uint16_t, float, float>(PermuteArgs args);
template void permute_launcher<uint8_t, float, float>(PermuteArgs args);
template void unpermute_launcher<uint16_t, float>(UnpermuteArgs args);

namespace {

constexpr int kKernelMaxThreads = 1024;
constexpr int kPermuteHiddenVec = 4;
constexpr int kUnpermuteHiddenVec = 4;

int dense_permute_block_threads(int64_t token_count, int launch_blocks) {
  assert(launch_blocks > 0);
  const int64_t token_blocks_32 = (token_count + 31) / 32;
  return 4 * token_blocks_32 >= 5LL * launch_blocks ? 512 : 256;
}

int dense_unpermute_block_threads(int64_t dense_token_count, int launch_blocks) {
  assert(launch_blocks > 0);
  return dense_token_count >= 16LL * launch_blocks ? 512 : 256;
}

__device__ __forceinline__ void store_float4_cg(float4* __restrict__ ptr, float4 value) {
  union {
    float4 f;
    uint4 u;
  } bits;
  bits.f = value;
  asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(bits.u.x), "r"(bits.u.y), "r"(bits.u.z), "r"(bits.u.w)
               : "memory");
}

}  // namespace

__global__ void pad_tokens_per_expert_kernel(const int32_t* src, int64_t* dst,
                                             int num_experts, int pad_multiple) {
  const int i = threadIdx.x;
  if (i < num_experts) {
    int32_t val = src[i];
    if (pad_multiple > 0) {
      val = ((val + pad_multiple - 1) / pad_multiple) * pad_multiple;
    }
    dst[i] = static_cast<int64_t>(val);
  }
}

void pad_tokens_per_expert(const int32_t* src, int64_t* dst, int num_experts,
                           int pad_multiple, cudaStream_t stream) {
  pad_tokens_per_expert_kernel<<<1, num_experts, 0, stream>>>(src, dst, num_experts, pad_multiple);
}

template <typename DType>
__global__ __launch_bounds__(kKernelMaxThreads, 1) void permute_kernel(
    const DType* __restrict__ tokens,
    DType* __restrict__ permuted_tokens,
    const float* __restrict__ scaling_factor,
    float* __restrict__ permuted_scaling_factor,
    const float* __restrict__ probs,
    float* __restrict__ permuted_probs,
    const int* __restrict__ dense_chunk_layout,
    const int* __restrict__ dense_to_expert_map,
    const int* __restrict__ num_local_experts_tokens,
    int num_dense_chunks,
    int pad_multiple,
    int num_local_experts,
    int hidden_size,
    int scales_per_token,
    int local_rank,
    int num_ranks_per_node) {
  // Thread layout: one warp owns one dense token row. Each lane walks the
  // hidden dimension in float4 units, so the token copy is coalesced.
  constexpr int lanes_per_token = 32;
  constexpr int num_eles_per_float4 = sizeof(float4) / sizeof(DType);
  const int tokens_per_block = blockDim.x / lanes_per_token;
  const int lane = threadIdx.x % lanes_per_token;
  const int token_slot = threadIdx.x / lanes_per_token;
  const int num_dense_tokens = num_dense_chunks > 0 ? dense_chunk_layout[num_dense_chunks - 1] : 0;
  const int hidden_size_fp4 = hidden_size / num_eles_per_float4;

  // Token buffers are accessed as float4. The dtype-specific alignment checks
  // in the launcher guarantee this vectorization is valid.
  const float4* __restrict__ tokens_fp4 = reinterpret_cast<const float4*>(tokens);
  float4* __restrict__ permuted_tokens_fp4 = reinterpret_cast<float4*>(permuted_tokens);

  // Per-token scratch stores the active expert output rows for the current
  // dense token. Only up to one warp writes/reads each 32-int row.
  __shared__ int route_smem[kKernelMaxThreads];

  // Grid-stride over real dense tokens. dense_chunk_layout[-1] is produced by
  // the fused scan and excludes padding rows.
  for (int block_start = blockIdx.x * tokens_per_block;
       block_start < num_dense_tokens;
       block_start += tokens_per_block * gridDim.x) {
    const int token_id = block_start + token_slot;
    if (token_id < num_dense_tokens) {
      const int64_t map_base = static_cast<int64_t>(token_id) * num_local_experts;
      const int64_t token_base = static_cast<int64_t>(token_id) * hidden_size_fp4;
      const int warp_lane = threadIdx.x & 31;
      int* active_dest_row = route_smem + token_slot * 32;
      const int64_t prob_base = static_cast<int64_t>(token_id) * num_local_experts *
                                    num_ranks_per_node +
                                local_rank * num_local_experts;

      // Scan one dense_to_expert_map row. Active entries are real expert output
      // rows; -1 means this dense token does not route to that local expert.
      for (int expert_base = 0; expert_base < num_local_experts; expert_base += 32) {
        const int expert = expert_base + warp_lane;
        const int dest = expert < num_local_experts ? dense_to_expert_map[map_base + expert] : -1;
        const unsigned active_mask = __ballot_sync(0xffffffffu, dest >= 0);
        const int active_count = __popc(active_mask);
        if (dest >= 0) {
          const unsigned lane_mask = warp_lane == 0 ? 0u : ((1u << warp_lane) - 1u);
          const int active_offset = __popc(active_mask & lane_mask);
          active_dest_row[active_offset] = dest;
          if (probs != nullptr) {
            permuted_probs[dest] = probs[prob_base + expert];
          }
        }
        __syncwarp();

        // Load this token stripe once, then scatter it to every active expert
        // destination for this token. All token-row addressing uses int64_t so
        // dest * hidden cannot overflow 32-bit arithmetic.
        for (int j = lane; j < hidden_size_fp4; j += lanes_per_token * kPermuteHiddenVec) {
          float4 value[kPermuteHiddenVec];
#pragma unroll
          for (int u = 0; u < kPermuteHiddenVec; ++u) {
            const int hidden_j = j + u * lanes_per_token;
            if (hidden_j < hidden_size_fp4) {
              value[u] = tokens_fp4[token_base + hidden_j];
            }
          }
          for (int active_idx = 0; active_idx < active_count; ++active_idx) {
            const int64_t active_base = static_cast<int64_t>(active_dest_row[active_idx]) *
                                        hidden_size_fp4;
#pragma unroll
            for (int u = 0; u < kPermuteHiddenVec; ++u) {
              const int hidden_j = j + u * lanes_per_token;
              if (hidden_j < hidden_size_fp4) {
                store_float4_cg(permuted_tokens_fp4 + active_base + hidden_j, value[u]);
              }
            }
          }
        }

        // FP8 inputs carry per-token scaling factors. Copy them with float4
        // vectorization when the scale count permits, otherwise use scalar
        // stores so odd scale counts still match baseline semantics.
        if (scaling_factor != nullptr) {
          if ((scales_per_token & 3) == 0) {
            const int scales_fp4 = scales_per_token / 4;
            const float4* __restrict__ scaling_factor_fp4 =
                reinterpret_cast<const float4*>(scaling_factor);
            float4* __restrict__ permuted_scaling_factor_fp4 =
                reinterpret_cast<float4*>(permuted_scaling_factor);
            const int64_t scale_base = static_cast<int64_t>(token_id) * scales_fp4;
            for (int j = lane; j < scales_fp4; j += lanes_per_token) {
              const float4 value = scaling_factor_fp4[scale_base + j];
              for (int active_idx = 0; active_idx < active_count; ++active_idx) {
                const int dest = active_dest_row[active_idx];
                permuted_scaling_factor_fp4[static_cast<int64_t>(dest) * scales_fp4 + j] = value;
              }
            }
          } else {
            const int64_t scale_base = static_cast<int64_t>(token_id) * scales_per_token;
            for (int j = lane; j < scales_per_token; j += lanes_per_token) {
              const float value = scaling_factor[scale_base + j];
              for (int active_idx = 0; active_idx < active_count; ++active_idx) {
                const int dest = active_dest_row[active_idx];
                permuted_scaling_factor[static_cast<int64_t>(dest) * scales_per_token + j] = value;
              }
            }
          }
        }
        __syncwarp();
      }
    }
  }

  // Padding is not present in dense_to_expert_map. Derive padded slots from the
  // per-expert token counts and explicitly zero token/prob/scale outputs.
  if (pad_multiple > 0 && num_local_experts_tokens != nullptr) {
    const int pad_work = num_local_experts * pad_multiple;
    const int pad_warp = threadIdx.x / 32;
    const int pad_lane = threadIdx.x & 31;
    const int pad_warps = blockDim.x / 32;
    for (int pad_entry = blockIdx.x * pad_warps + pad_warp;
         pad_entry < pad_work;
         pad_entry += gridDim.x * pad_warps) {
      int slot = -1;
      if (pad_lane == 0) {
        const int expert = pad_entry / pad_multiple;
        const int pad_idx = pad_entry - expert * pad_multiple;
        if (expert < num_local_experts) {
          const int count = num_local_experts_tokens[expert];
          const int padded = ((count + pad_multiple - 1) / pad_multiple) * pad_multiple;
          if (pad_idx < padded - count) {
            slot = count + pad_idx;
            for (int prev = 0; prev < expert; ++prev) {
              const int prev_count = num_local_experts_tokens[prev];
              slot += ((prev_count + pad_multiple - 1) / pad_multiple) * pad_multiple;
            }
          }
        }
      }
      slot = __shfl_sync(0xffffffffu, slot, 0);
      if (slot >= 0) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = pad_lane; j < hidden_size_fp4; j += 32) {
          permuted_tokens_fp4[static_cast<int64_t>(slot) * hidden_size_fp4 + j] = zero;
        }
        if (permuted_probs != nullptr && pad_lane == 0) {
          permuted_probs[slot] = 0.0f;
        }
        if (permuted_scaling_factor != nullptr) {
          for (int j = pad_lane; j < scales_per_token; j += 32) {
            permuted_scaling_factor[static_cast<int64_t>(slot) * scales_per_token + j] = 0.0f;
          }
        }
      }
    }
  }
}

template <typename DType, typename ProbType, typename ScalarType>
void permute_launcher(PermuteArgs args) {
  assert((std::is_same<DType, uint8_t>::value || std::is_same<DType, uint16_t>::value));
  assert((std::is_same<ProbType, float>::value));
  assert((std::is_same<ScalarType, float>::value));
  if (std::is_same<DType, uint8_t>::value) {
    assert(args.hidden_size % 16 == 0);
  } else if (std::is_same<DType, uint16_t>::value) {
    assert(args.hidden_size % 8 == 0);
  }
  if (args.num_permuted_token == 0) return;
  assert(args.output_tokens_ptr != nullptr);
  assert(args.dense_chunk_layout.dtype() == torch::kInt32);
  assert(args.dense_to_expert_map.dtype() == torch::kInt32);
  assert(args.num_of_local_experts_tokens.dtype() == torch::kInt32);
  assert(args.dense_chunk_layout.is_contiguous());
  assert(args.dense_to_expert_map.is_contiguous());
  assert(args.num_of_local_experts_tokens.is_contiguous());

  const int grid_size = args.num_of_blocks_permute;
  assert(grid_size > 0);
  const int block_threads = dense_permute_block_threads(args.num_permuted_token, grid_size);
  permute_kernel<DType><<<grid_size, block_threads, 0, args.stream>>>(
      reinterpret_cast<const DType*>(args.tokens_ptr),
      reinterpret_cast<DType*>(args.output_tokens_ptr),
      args.use_fp8 ? reinterpret_cast<float*>(args.scaling_factor_ptr) : nullptr,
      args.use_fp8 ? reinterpret_cast<float*>(args.output_scaling_factor_ptr) : nullptr,
      args.with_probs ? reinterpret_cast<float*>(args.probs_ptr) : nullptr,
      args.with_probs ? reinterpret_cast<float*>(args.output_probs_ptr) : nullptr,
      args.dense_chunk_layout.data_ptr<int>(),
      args.dense_to_expert_map.data_ptr<int>(),
      args.num_of_local_experts_tokens.data_ptr<int>(),
      static_cast<int>(args.dense_chunk_layout.numel()),
      args.pad_multiple,
      args.num_of_local_experts,
      args.hidden_size,
      args.scales_per_token,
      args.local_rank,
      args.num_ranks_per_node);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DType>
__global__ __launch_bounds__(kKernelMaxThreads, 1) void unpermute_kernel(
    const DType* __restrict__ permuted_tokens,
    DType* __restrict__ tokens,
    const float* __restrict__ permuted_probs,
    float* __restrict__ probs,
    const int* __restrict__ dense_chunk_layout,
    const int* __restrict__ dense_to_expert_map,
    int num_dense_chunks,
    int num_local_experts,
    int hidden_size,
    int local_rank,
    int num_ranks_per_node) {
  static_assert(std::is_same<DType, __nv_bfloat16>::value, "dense unpermute supports bf16");

  // Thread layout mirrors permute: one warp owns one dense output token row.
  // Each lane accumulates a strided float4 stripe of the hidden dimension.
  constexpr int lanes_per_token = 32;
  constexpr int num_eles_per_float4 = sizeof(float4) / sizeof(DType);
  constexpr int bf16x2_per_float4 = sizeof(float4) / sizeof(__nv_bfloat162);
  const int tokens_per_block = blockDim.x / lanes_per_token;
  const int lane = threadIdx.x % lanes_per_token;
  const int token_slot = threadIdx.x / lanes_per_token;
  const int num_dense_tokens = num_dense_chunks > 0 ? dense_chunk_layout[num_dense_chunks - 1] : 0;
  const int hidden_size_fp4 = hidden_size / num_eles_per_float4;

  // Read expert outputs as float4, accumulate in fp32 bf16x2 pairs, then write
  // bf16 float4 rows back to the dense token order expected by combine.
  const float4* __restrict__ permuted_tokens_fp4 = reinterpret_cast<const float4*>(permuted_tokens);
  float4* __restrict__ tokens_fp4 = reinterpret_cast<float4*>(tokens);

  // One shared-memory row per warp stores compacted active source rows for the
  // current dense token. Source row ids are kept as int32; source * hidden uses
  // int64_t at the load site.
  extern __shared__ int route_smem[];
  const int warp_id = threadIdx.x / 32;
  const int warp_lane = threadIdx.x & 31;
  int* warp_active_routes = route_smem + warp_id * num_local_experts;

  // Grid-stride over real dense tokens. Padding never participates in unpermute
  // because it was only inserted for expert GEMM alignment.
  for (int block_start = blockIdx.x * tokens_per_block;
       block_start < num_dense_tokens;
       block_start += tokens_per_block * gridDim.x) {
    const int token_id = block_start + token_slot;
    if (token_id < num_dense_tokens) {
      const int64_t map_base = static_cast<int64_t>(token_id) * num_local_experts;
      const int64_t token_base = static_cast<int64_t>(token_id) * hidden_size_fp4;
      const int prob_width = num_local_experts * num_ranks_per_node;
      const int64_t prob_base = static_cast<int64_t>(token_id) * prob_width;
      if (permuted_probs != nullptr) {
        // Baseline writes a full prob row, including zeros for non-local ranks
        // and inactive experts. Do that explicitly; output buffers are dirty.
        for (int j = lane; j < prob_width; j += lanes_per_token) {
          probs[prob_base + j] = 0.0f;
        }
        __syncwarp();
      }

      // Compact active expert source rows and scatter their probabilities into
      // the local-rank slice of the dense prob row.
      int active_count = 0;
      for (int expert_base = 0; expert_base < num_local_experts; expert_base += 32) {
        const int expert = expert_base + warp_lane;
        const int source = expert < num_local_experts ? dense_to_expert_map[map_base + expert] : -1;
        const unsigned active_mask = __ballot_sync(0xffffffffu, source >= 0);
        if (source >= 0) {
          const unsigned lane_mask = warp_lane == 0 ? 0u : ((1u << warp_lane) - 1u);
          const int active_offset = __popc(active_mask & lane_mask);
          warp_active_routes[active_count + active_offset] = source;
          if (permuted_probs != nullptr) {
            probs[prob_base + local_rank * num_local_experts + expert] = permuted_probs[source];
          }
        }
        active_count += __popc(active_mask);
      }
      __syncwarp();

      // Reduce all active expert rows for this dense token. Accumulation is
      // fp32 per bf16x2 lane pair, then converted back to bf16 on store.
      for (int j = lane; j < hidden_size_fp4; j += lanes_per_token * kUnpermuteHiddenVec) {
        float4 buffer_fp4[kUnpermuteHiddenVec];
        float2 accumulator[kUnpermuteHiddenVec][bf16x2_per_float4];
#pragma unroll
        for (int u = 0; u < kUnpermuteHiddenVec; ++u) {
#pragma unroll
          for (int k = 0; k < bf16x2_per_float4; ++k) {
            accumulator[u][k].x = 0.0f;
            accumulator[u][k].y = 0.0f;
          }
        }
        for (int route_idx = 0; route_idx < active_count; ++route_idx) {
          const int64_t source_base = static_cast<int64_t>(warp_active_routes[route_idx]) *
                                      hidden_size_fp4;
#pragma unroll
          for (int u = 0; u < kUnpermuteHiddenVec; ++u) {
            const int hidden_j = j + u * lanes_per_token;
            if (hidden_j < hidden_size_fp4) {
              buffer_fp4[u] = permuted_tokens_fp4[source_base + hidden_j];
            }
          }
#pragma unroll
          for (int u = 0; u < kUnpermuteHiddenVec; ++u) {
            const int hidden_j = j + u * lanes_per_token;
            if (hidden_j < hidden_size_fp4) {
              const __nv_bfloat162* buffer_ptr =
                  reinterpret_cast<const __nv_bfloat162*>(&buffer_fp4[u]);
#pragma unroll
              for (int k = 0; k < bf16x2_per_float4; ++k) {
                const float2 value = __bfloat1622float2(buffer_ptr[k]);
                accumulator[u][k].x += value.x;
                accumulator[u][k].y += value.y;
              }
            }
          }
        }
#pragma unroll
        for (int u = 0; u < kUnpermuteHiddenVec; ++u) {
          const int hidden_j = j + u * lanes_per_token;
          if (hidden_j < hidden_size_fp4) {
            __nv_bfloat162* buffer_ptr = reinterpret_cast<__nv_bfloat162*>(&buffer_fp4[u]);
#pragma unroll
            for (int k = 0; k < bf16x2_per_float4; ++k) {
              buffer_ptr[k] = __float22bfloat162_rn(accumulator[u][k]);
            }
            tokens_fp4[token_base + hidden_j] = buffer_fp4[u];
          }
        }
      }
    }
  }
}

template <typename DType, typename ProbType>
void unpermute_launcher(UnpermuteArgs args) {
  assert(args.permuted_tokens.dtype() == torch::kBFloat16);
  if (args.with_probs) {
    assert(args.permuted_probs.has_value());
    assert(args.permuted_probs.value().dtype() == torch::kFloat32);
  }
  assert((std::is_same<DType, uint16_t>::value));
  assert((std::is_same<ProbType, float>::value));
  assert(args.hidden_size % 8 == 0);
  assert(args.dense_chunk_layout.dtype() == torch::kInt32);
  assert(args.dense_to_expert_map.dtype() == torch::kInt32);
  assert(args.dense_chunk_layout.is_contiguous());
  assert(args.dense_to_expert_map.is_contiguous());

  const int grid_size = args.num_of_blocks_unpermute;
  assert(grid_size > 0);
  const int block_threads = dense_unpermute_block_threads(args.permuted_tokens.size(0), grid_size);
  const size_t shared_mem_size =
      static_cast<size_t>(block_threads / 32) * args.num_of_local_experts * sizeof(int);
  unpermute_kernel<__nv_bfloat16>
      <<<grid_size, block_threads, shared_mem_size, args.stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(args.permuted_tokens.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(args.tokens_ptr),
          args.with_probs ? reinterpret_cast<float*>(args.permuted_probs.value().data_ptr()) : nullptr,
          args.with_probs ? reinterpret_cast<float*>(args.probs_ptr) : nullptr,
          args.dense_chunk_layout.data_ptr<int>(),
          args.dense_to_expert_map.data_ptr<int>(),
          static_cast<int>(args.dense_chunk_layout.numel()),
          args.num_of_local_experts,
          args.hidden_size,
          args.local_rank,
          args.num_ranks_per_node);
  CUDA_CHECK(cudaGetLastError());
}
