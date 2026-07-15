// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

#ifndef USE_NIXL

#include <unordered_map>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <iostream>
#include <dlfcn.h>
#include "doca_gpunetio_host.h"
#include "doca_gpunetio_device.h"
#include "infiniband/verbs.h"
#include "infiniband/mlx5dv.h"
#include "coordinator.cuh"
#include "backend/topo_detection.cuh"
#include "backend/ibvcore.h"
#include "utils.cuh"

#define RC  (0)
#define UC  (1)
#define UD  (2)
#define RawEth  (3)
#define XRC (4)
#define DC  (5)
#define SRD (6)

#define GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR 10
#define DOCA_VERBS_DB_UAR_SIZE 8

enum memory_type {
  MEMORY_HOST,
  MEMORY_MMAP,
  MEMORY_CUDA,
  MEMORY_ROCM,
  MEMORY_NEURON,
  MEMORY_DEVMEM,
  MEMORY_HL
};

constexpr int32_t CONNECTION_TYPE = RC;
constexpr int32_t DEF_HOP_LIMIT = 255;
constexpr int32_t DEF_IB_TC = 0;
constexpr int32_t DEF_RX_RDMA = 128;
constexpr int32_t DEF_TX_BW = 512;
constexpr int32_t EQ_NUM = 0;
constexpr int32_t GVERBS_WQ_BUF_LOC = DOCA_GPU_MEM_TYPE_GPU;
constexpr int32_t GVERBS_CQ_BUF_LOC = DOCA_GPU_MEM_TYPE_GPU;
constexpr int32_t GVERBS_USE_ASYNC_STIRE_RELEASE = 0;
constexpr uint8_t IB_PORT = 1;
constexpr int32_t INLINE_SIZE = 0;
constexpr int32_t MTU = 0;
constexpr short PKEY_INDEX = 0;
constexpr uint8_t QP_TIMEOUT = 22;
constexpr uint8_t SL = 0;
constexpr uint8_t TRAFFIC_CLASS = 0;
constexpr enum memory_type MEMORY_TYPE = MEMORY_CUDA;
static const char *portStates[] = {"Nop","Down","Init","Armed","","Active Defer"};

struct doca_gpu_mtable {
    uintptr_t base_addr;
    size_t size_orig;
    uintptr_t align_addr_gpu;
    uintptr_t align_addr_cpu;
    size_t size;
    enum doca_gpu_mem_type mtype;
    void *gdr_mh;
};

struct doca_gpu {
  CUdevice cuda_dev;
  std::unordered_map<uintptr_t, struct doca_gpu_mtable *> *mtable;
  bool support_gdrcopy;
  bool support_dmabuf;
  bool support_wq_gpumem;
  bool support_cq_gpumem;
  bool support_uar_gpumem;
  bool support_async_store_release;
  bool support_bf_uar;
};

struct gverbs_context {
    int pdn;
    union ibv_gid gid;
    struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr;
    struct doca_verbs_qp_attr *qp_attr;
    struct doca_gpu_verbs_qp_hl **qp_hls;
    struct doca_gpu_dev_verbs_qp **d_qps_gpu;
};

struct remote_info {
    int           lid;
    int           qpn;
    int           psn;
    int           gid_index;
    union ibv_gid gid;
    __be32        token_rkey;
    uint64_t      token_vaddr;
    __be32        flag_rkey;
    uint64_t      flag_vaddr;
    __be32        prob_rkey;
    uint64_t      prob_vaddr;
    __be32        scaling_factor_rkey;
    uint64_t      scaling_factor_vaddr;
};

ibv_device *ctx_find_dev(const char *ib_devname);
int get_gpu_handler(struct doca_gpu *handler, struct ibv_context *ib_context, int local_rank);
void setup_qp_init_attr(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                        struct doca_gpu *gpu_handler, struct ibv_pd *ib_pd, int tx_depth);
int create_and_place_qps(struct gverbs_context *g_ctx,
                         struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, int num_qps);
int setup_qp_attr_for_modify(struct ibv_port_attr *port_attr, struct doca_verbs_qp_attr *qp_attr,
                                    struct remote_info *l_info, struct remote_info *r_info,
                                    struct ibv_context *ib_context);
int doca_gpunetio_test_change_qp_state(struct doca_gpu_verbs_qp_hl *qp,
                                       struct doca_verbs_qp_attr *qp_attr, int attr_mask);
int setup_qp_attr_and_set_qp(struct gverbs_context *g_ctx,
                                    struct ibv_context *ib_context,
                                    struct ibv_port_attr *port_attr,
                                    struct remote_info *rem_dest,
                                    struct doca_verbs_qp_attr *qp_attr,
                                    int num_of_blocks, int num_of_nodes,
                                    int node_rank, uint32_t qp_cnt);

class RDMACoordinator : public InterNodeCoordinator {
public:
    RDMACoordinator() = default;
    ~RDMACoordinator() override;
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
    int gid_index = 0;
    int node_rank = -1;
    int local_rank = -1;
    BufferConfig buffer_config;
    pybind11::object process_group;

    struct ibv_context *ib_context = nullptr;
    struct ibv_pd *ib_pd = nullptr;
    struct doca_gpu *gpu_handler = nullptr;
    struct ibv_port_attr port_attr = {};
    int mr_access_flag = -1;
    bool buffer_allocated = false;
    bool rdma_initialized = false;

    struct ibv_mr *attn_input_token_mr = nullptr;
    struct ibv_mr *dispatch_rdma_inter_node_group_token_mr = nullptr;
    struct ibv_mr *attn_input_flags_mr = nullptr;
    struct ibv_mr *dispatch_rdma_inter_node_group_flags_mr = nullptr;
    struct ibv_mr *attn_input_prob_mr = nullptr;
    struct ibv_mr *dispatch_rdma_inter_node_group_prob_mr = nullptr;
    struct ibv_mr *attn_input_token_scaling_factor_mr = nullptr;
    struct ibv_mr *dispatch_rdma_inter_node_group_scaling_factor_mr = nullptr;
    struct remote_info *dispatch_remote_info_vec = nullptr;
    struct dispatch_memory_region_info_t *dispatch_mr_info_h = nullptr;
    struct gverbs_context dispatch_gverbs_ctx;
    struct dispatch_memory_region_info_t *dispatch_mr_info_d = nullptr;

    struct ibv_mr *rdma_intra_node_red_token_mr = nullptr;
    struct ibv_mr *combine_rdma_inter_node_group_token_mr = nullptr;
    struct ibv_mr *rdma_intra_node_red_prob_mr = nullptr;
    struct ibv_mr *combine_rdma_inter_node_group_prob_mr = nullptr;
    struct ibv_mr *attn_output_flags_mr = nullptr;
    struct ibv_mr *combine_rdma_inter_node_group_flags_mr = nullptr;
    struct remote_info *combine_remote_info_vec = nullptr;
    struct combine_memory_region_info_t *combine_mr_info_h = nullptr;
    struct gverbs_context combine_gverbs_ctx;
    struct combine_memory_region_info_t *combine_mr_info_d = nullptr;

    void exchange_remote_rdma_info(remote_info* dst, remote_info *src, int num_of_qps);
};

#endif  // !USE_NIXL
