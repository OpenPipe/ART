// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "allgather.cuh"

#define MAX_BLOCKS 256
#define TIMEOUT 200000000000ull

template<int SHARED_SIZE = 1024>
__global__ void ag_nvl_kernel(
    void** dst_buffers_all_ranks, 
    void* src, 
    int bytes_per_rank, 
    int64_t *iter_id_ptr, // Normal GPU memory
    unsigned long long *flag_nvl_ptr, // Register memory on rank 0 
    unsigned long long *flag_sm_ptr, // Normal GPU memory
    int rank_idx, 
    int rank_num
) {
    int is_last_SM = 0;
    uint4** dst_list_ptr = reinterpret_cast<uint4**>(dst_buffers_all_ranks);
    uint4* src_ptr = reinterpret_cast<uint4*>(src);
    auto iter_id = *iter_id_ptr;
    iter_id ++ ; // increment iter_id

    __shared__ uint4 shared_data[SHARED_SIZE];
    // Compute the data size assigned to each SM
    int uint4_per_rank = bytes_per_rank / sizeof(uint4);  // uint4 = 16 bytes
    int chunk_size = (uint4_per_rank + gridDim.x - 1) / gridDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, uint4_per_rank);

    int loop_time = (chunk_end - chunk_start + SHARED_SIZE - 1) / SHARED_SIZE;
    for(int i = 0; i < loop_time; i++) {
        int start_idx = chunk_start + i * SHARED_SIZE;
        int end_idx = min(start_idx + SHARED_SIZE, chunk_end);

        // Load the data from src to shared_data
        for(int j = threadIdx.x; j < end_idx - start_idx; j += blockDim.x) {
            shared_data[j] = src_ptr[start_idx + j];
        }
        __syncthreads();

        // Copy the data from src to dst
        for(int j = 0; j < rank_num; j++) {
            auto dst_rank = (rank_idx + j) % rank_num;
            auto dst_ptr = dst_list_ptr[dst_rank];
            for(int k = threadIdx.x; k < end_idx - start_idx; k += blockDim.x) {
                auto local_offset = start_idx + k;
                auto dst_offset = local_offset + rank_idx * uint4_per_rank;
                dst_ptr[dst_offset] = shared_data[k];
            }
        }
    }

    __syncthreads();
    __threadfence();

    if(threadIdx.x == 0) {
        unsigned long long value_to_add = blockIdx.x == 0 ? MAX_BLOCKS - gridDim.x + 1 : 1;
        auto old_val_sm_sync = atomicAdd(flag_sm_ptr, value_to_add);  
        is_last_SM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == iter_id * MAX_BLOCKS);
    }

    __threadfence_system();
    if(is_last_SM) {
      // Update the flag_nvl_ptr
      asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;"
                    :
                    : "l"(__cvta_generic_to_global(flag_nvl_ptr)), "n"(1)
                    : "memory");
      *iter_id_ptr = iter_id;
      auto expected = iter_id * rank_num;
      clock_t s = clock64();
      unsigned long long flag_data = 0;

      // Wait for the flag_nvl_ptr to be updated from all ranks in nvl domain
      do{
        asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                      : "=l"(flag_data)
                      : "l"(__cvta_generic_to_global(flag_nvl_ptr))
                      : "memory");
        if (clock64() - s > 2ull * TIMEOUT) {
          printf("HYBRID-EP ALLGATHER TIMEOUT:SM %d [%d]:expecting %llu got %llu\n", blockIdx.x,
                  threadIdx.x, (unsigned long long)expected, flag_data);
          __trap();
        }
      }while(flag_data < expected);
    }
}

void CustomAllgather::launch(torch::Tensor src, int ag_sms, cudaStream_t stream) {
    auto bytes_per_rank = src.numel() * src.element_size();;
    auto rank_num = num_of_ranks_per_node;
    assert(rank_idx >= 0 && rank_idx < rank_num);
    assert(rank_num <= MAX_NUM_OF_RANKS_PER_NODE);
    assert(bytes_per_rank % 16 == 0); // Use LDG.128 / STG.128

    int block_size = 1024;
    ag_nvl_kernel<<<ag_sms, block_size, 0, stream>>>(
        dst_buffers_all_ranks_gpu, 
        src.data_ptr(), 
        bytes_per_rank, 
        iter_id_ptr, 
        flag_nvl_ptr, 
        flag_sm_ptr, 
        rank_idx, 
        rank_num
    );
}

void CustomAllgather::init(pybind11::object process_group, int rank_idx, BufferConfig buffer_config, ExtendedMemoryAllocator* allocator) {
    this->rank_idx = rank_idx;
    this->num_of_ranks_per_node = buffer_config.num_of_ranks_per_node;
    this->num_of_experts_per_rank = buffer_config.num_of_experts_per_rank;
    this->num_of_tokens_per_rank = buffer_config.max_num_of_tokens_per_rank;
    this->num_of_nodes = buffer_config.num_of_nodes;
    this->allocator = allocator;
    this->process_group = process_group;
}

bool CustomAllgather::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    return changed;
}

void CustomAllgather::update_config(BufferConfig config) {
    this->num_of_ranks_per_node = config.num_of_ranks_per_node;
    this->num_of_experts_per_rank = config.num_of_experts_per_rank;
    this->num_of_tokens_per_rank = config.max_num_of_tokens_per_rank;
    this->num_of_nodes = config.num_of_nodes;
}

void CustomAllgather::allocate_buffers() {
    allocate_ag_buffer();
}

void CustomAllgather::allocate_ag_buffer() {
    // Allocate the output buffer
    auto num_of_expert = num_of_experts_per_rank * num_of_ranks_per_node * num_of_nodes;
    auto gathered_elets = num_of_expert * num_of_tokens_per_rank * num_of_ranks_per_node * num_of_nodes;
    auto gathered_bytes = gathered_elets * sizeof(bool);
    allocator->allocate(&dst_buffer, gathered_bytes);

    if(num_of_nodes == 1) {
        // Allocate the nvl sync flag on the rank 0
        if(rank_idx == 0) {
            allocator->allocate((void**)&flag_nvl_ptr, sizeof(unsigned long long));
            CUDA_CHECK(cudaMemset(flag_nvl_ptr, 0, sizeof(unsigned long long)));
        }

        // Allocate the sm sync flag
        CUDA_CHECK(cudaMalloc((void**)&flag_sm_ptr, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(flag_sm_ptr, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&iter_id_ptr, sizeof(int64_t)));
        CUDA_CHECK(cudaMemset(iter_id_ptr, 0, sizeof(int64_t)));

        // Allocate the dst_buffers_all_ranks
        dst_buffers_all_ranks = (void**)malloc(num_of_ranks_per_node * sizeof(void*));
        CUDA_CHECK(cudaMalloc((void**)&dst_buffers_all_ranks_gpu, num_of_ranks_per_node * sizeof(void*)));

        // Get handle of the nvl sync flag
        MemHandle handles[2];
        allocator->get_handle(&handles[0], dst_buffer);
        if (rank_idx == 0) {
            allocator->get_handle(&handles[1], flag_nvl_ptr);
        }
        // Pack handles into tensor
        ag_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                                torch::dtype(torch::kUInt8).device(torch::kCPU));
        memcpy(ag_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

        open_ag_handles();
    }
}

void CustomAllgather::open_ag_handles() {
    if(num_of_nodes > 1 ) return;

    // Use Python's torch.distributed APIs through py::object
    auto torch_distributed = py::module_::import("torch.distributed");    
    // Move tensors to CUDA for communication
    auto ag_handles_cuda = ag_handles.cuda();  
    // Get world size from process group
    int world_size = process_group.attr("size")().cast<int>();
    // Create empty tensors for allgather output
    py::list ag_handles_output_list;
  
    for (int i = 0; i < world_size; i++) {
        ag_handles_output_list.append(torch::empty_like(ag_handles_cuda));
    }
    // Perform allgather using Python API
    torch_distributed.attr("all_gather")(ag_handles_output_list, ag_handles_cuda, process_group);
  
    // Convert back to C++ vectors and move to CPU
    std::vector<torch::Tensor> ag_handles_cpu_tensors;
    for (int i = 0; i < world_size; i++) {
        ag_handles_cpu_tensors.push_back(ag_handles_output_list[i].cast<torch::Tensor>().cpu());
    }
    
    // Open the flag_nvl_ptr handle
    if (rank_idx != 0) {
        MemHandle flag_nvl_ptr_handle;
        // Only rank 0 will allocate memory for this flag
        memcpy(&flag_nvl_ptr_handle, ag_handles_cpu_tensors[0].data_ptr<uint8_t>() + sizeof(MemHandle),
            sizeof(MemHandle));
        allocator->open_handle((void**)(&flag_nvl_ptr), &flag_nvl_ptr_handle);
    }

    // Open the dst_buffers_all_ranks handles
    for (int i = 0; i < num_of_ranks_per_node; i++) {
        MemHandle dst_buffer_handle;
        // Extract the handles from the tensor.
        memcpy(&dst_buffer_handle, ag_handles_cpu_tensors[i].data_ptr<uint8_t>(), sizeof(MemHandle));
        if(i != rank_idx) {
            allocator->open_handle((void**)(&dst_buffers_all_ranks[i]), &dst_buffer_handle);
        } else {
            // For local rank, use direct pointer assignment (more efficient, no IPC overhead)
            dst_buffers_all_ranks[i] = dst_buffer;
        }
    }

    CUDA_CHECK(cudaMemcpy(dst_buffers_all_ranks_gpu, dst_buffers_all_ranks, num_of_ranks_per_node * sizeof(void*), cudaMemcpyHostToDevice));
}

void CustomAllgather::destroy() {
    if(num_of_nodes == 1) {
        if (flag_nvl_ptr != nullptr) {
            if(rank_idx == 0) {
                allocator->free(flag_nvl_ptr);
            } else {
                allocator->close_handle(flag_nvl_ptr);
            }
            flag_nvl_ptr = nullptr;
        }
        
        if (flag_sm_ptr != nullptr) {
            CUDA_CHECK(cudaFree(flag_sm_ptr));
            flag_sm_ptr = nullptr;
        }
        
        if (iter_id_ptr != nullptr) {
            CUDA_CHECK(cudaFree(iter_id_ptr));
            iter_id_ptr = nullptr;
        }
    
        // Close remote memory handles (not locally allocated, just mapped)
        if (dst_buffers_all_ranks != nullptr) {
            for(int i = 0; i < num_of_ranks_per_node; i++) {
                if(i != rank_idx) {
                    allocator->close_handle(dst_buffers_all_ranks[i]);
                }
            }
            free(dst_buffers_all_ranks);
            dst_buffers_all_ranks = nullptr;
        }
        
        // Free the GPU buffer
        if (dst_buffers_all_ranks_gpu != nullptr) {
            CUDA_CHECK(cudaFree(dst_buffers_all_ranks_gpu));
            dst_buffers_all_ranks_gpu = nullptr;
        }
    }
    
    if (dst_buffer != nullptr) {
        allocator->free(dst_buffer);
        dst_buffer = nullptr;
    }
}

void * CustomAllgather::get_output_buffer() {
    return dst_buffer;
}

CustomAllgather::~CustomAllgather() {
    destroy();
}