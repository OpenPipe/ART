// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved
#include "buffer/internode_doca.cuh"
#include <sstream>
#include <cstdlib>
#include <unordered_map>

// Functions realted to get RDMA context.
ibv_device *ctx_find_dev(const char *ib_devname) {
  int num_of_device;
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev = NULL;
  dev_list = ibv_get_device_list(&num_of_device);
  // coverity[uninit_use]
  if (num_of_device <= 0) {
    fprintf(stderr, " Did not detect devices \n");
    fprintf(stderr, " If device exists, check if driver is up\n");
    return NULL;
  }
  for (; (ib_dev = *dev_list); ++dev_list) {
    if (!strcmp(ibv_get_device_name(ib_dev), ib_devname))
      break;
  }
  if (!ib_dev) {
    fprintf(stderr, "IB device %s not found\n", ib_devname);
    return NULL;
  }
  return ib_dev;
}

// Get NIC name with optional manual mapping from environment variables.
// If HYBRID_EP_ENABLE_MANUAL_NIC_MAPPING=1, parse HYBRID_EP_NIC_MAPPING
// Format: "0:mlx5_0:1,1:mlx5_1:1,..." (gpu_id:nic_name:port)
static void get_nic_name(const std::vector<int>& gpu_idx_vec, int local_device_idx, const char** net_name) {
  static thread_local std::string nic_name_storage;
  
  const char* manual_mapping_env = std::getenv("HYBRID_EP_ENABLE_MANUAL_NIC_MAPPING");
  if (manual_mapping_env != nullptr && std::string(manual_mapping_env) == "1") {
    const char* nic_mapping_env = std::getenv("HYBRID_EP_NIC_MAPPING");
    if (nic_mapping_env == nullptr) {
      fprintf(stderr, "[Error] HYBRID_EP_ENABLE_MANUAL_NIC_MAPPING=1 but HYBRID_EP_NIC_MAPPING is not set\n");
      assert(false);
    }
    
    std::unordered_map<int, std::string> device_mapping;
    std::string mapping_str(nic_mapping_env);
    std::stringstream ss(mapping_str);
    std::string entry;
    while (std::getline(ss, entry, ',')) {
      size_t first_colon = entry.find(':');
      if (first_colon == std::string::npos) {
        fprintf(stderr, "[Error] Invalid mapping format '%s' in HYBRID_EP_NIC_MAPPING. Expected format: '<device_id>:<nic_name>'\n", entry.c_str());
        assert(false);
      }
      int device_id = std::stoi(entry.substr(0, first_colon));
      std::string nic_name = entry.substr(first_colon + 1);  // Keep the rest as NIC name (including :1)
      device_mapping[device_id] = nic_name;
    }
    
    auto it = device_mapping.find(local_device_idx);
    if (it == device_mapping.end()) {
      fprintf(stderr, "[Error] Device %d not found in HYBRID_EP_NIC_MAPPING\n", 
              local_device_idx);
      assert(false);
    }
    nic_name_storage = it->second;
    *net_name = nic_name_storage.c_str();
  } else {
    hybrid_ep::get_nic(gpu_idx_vec, local_device_idx, net_name);
  }
}

// Functions related to initialization of gverbs_context.
int get_gpu_handler(struct doca_gpu *handler,
                           struct ibv_context *ib_context, int local_rank) {
  char pciBusId[256];
  int compute_cap_major;
  unsigned int flag = 1;
  cudaError_t cuda_error;
  CUresult cu_error;
  CUdeviceptr dev_ptr = 0UL;
  CUdeviceptr db_gpu_ptr = 0UL;
  CUdeviceptr bf_gpu_ptr = 0UL;
  struct mlx5dv_devx_uar *db_uar = nullptr;
  struct mlx5dv_devx_uar *bf_uar = nullptr;
  // Getting BDF of GPU and setting dev_id.
  cuda_error = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), local_rank);
  assert(cuda_error == cudaSuccess);
  handler->cuda_dev = local_rank;
  // Determining whether GPU supports ASYNC_STORE_RELEASE.
  cuda_error = cudaDeviceGetAttribute(
      &compute_cap_major, cudaDevAttrComputeCapabilityMajor, local_rank);
  assert(cuda_error == cudaSuccess);
  handler->support_async_store_release =
      compute_cap_major >=
      GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR;
  // Determining whether GPU supports DMABUF.
  int support_dmabuf = 0;
  cu_error = cuDeviceGetAttribute(
      &support_dmabuf, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, local_rank);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_dmabuf = support_dmabuf;
  // Determining whether GPU supports wq_gpumem and cq_gpumem.
  cu_error = cuMemAlloc(&dev_ptr, 1 << 11);
  assert(cu_error == CUDA_SUCCESS);
  cu_error =
      cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_ptr);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_wq_gpumem = true;
  handler->support_cq_gpumem = true;
  // Determining whether uar_gpumem is supported.
  db_uar =
      mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
#if CUDA_VERSION >= 12020
  if (!db_uar)
    db_uar = mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_NC);
#endif
  assert(db_uar);
  cu_error = cuMemHostRegister(db_uar->reg_addr, DOCA_VERBS_DB_UAR_SIZE,
                               CU_MEMHOSTREGISTER_DEVICEMAP |
                                   CU_MEMHOSTREGISTER_PORTABLE |
                                   CU_MEMHOSTREGISTER_IOMEMORY);
  assert(cu_error == CUDA_SUCCESS);
  cu_error = cuMemHostGetDevicePointer(&db_gpu_ptr, db_uar->reg_addr, 0);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_uar_gpumem = true;
  // Determining whether bf_uar is supported.
  bf_uar = mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_BF);
  if (bf_uar) {
    cu_error = cuMemHostRegister(bf_uar->reg_addr, DOCA_VERBS_DB_UAR_SIZE,
                                 CU_MEMHOSTREGISTER_DEVICEMAP |
                                     CU_MEMHOSTREGISTER_PORTABLE |
                                     CU_MEMHOSTREGISTER_IOMEMORY);
    assert(cu_error == CUDA_SUCCESS);
    cu_error = cuMemHostGetDevicePointer(&bf_gpu_ptr, bf_uar->reg_addr, 0);
    assert(cu_error == CUDA_SUCCESS);
    handler->support_bf_uar = true;
  } else {
    handler->support_bf_uar = false;
  }
  // Setting support_gdrcopy to false.
  handler->support_gdrcopy = false;
  // Creating mtable.
  try {
    handler->mtable =
        new std::unordered_map<uintptr_t, struct doca_gpu_mtable *>();
  } catch (...) {
    fprintf(stderr, "mtable map allocation failed\n");
    assert(0);
  }
  cuMemFree(dev_ptr);
  if (db_uar) {
    cuMemHostUnregister(db_uar->reg_addr);
    mlx5dv_devx_free_uar(db_uar);
  }
  if (bf_uar) {
    cuMemHostUnregister(bf_uar->reg_addr);
    mlx5dv_devx_free_uar(bf_uar);
  }
  return 0;
}

void setup_qp_init_attr(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                        struct doca_gpu *gpu_handler, struct ibv_pd *ib_pd,
                        int tx_depth) {
  assert(tx_depth > 0 && tx_depth < 65536);
  qp_init_attr->gpu_dev = gpu_handler;
  qp_init_attr->ibpd = ib_pd;
  qp_init_attr->sq_nwqe = tx_depth;
  qp_init_attr->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  qp_init_attr->mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
}

int create_and_place_qps(struct gverbs_context *g_ctx,
                         struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                         int num_qps) {
  int status = 0;
  for (int i = 0; i < num_qps; i++) {
    struct doca_gpu_verbs_qp_hl *qp_hl = NULL;
    status = doca_gpu_verbs_create_qp_hl(qp_init_attr, &qp_hl);
    if (status) {
      fprintf(stderr, "Failed to create %dth QP with status %d\n", i, status);
      assert(0);
    }
    g_ctx->qp_hls[i] = qp_hl;
  }
  return status;
}

int setup_qp_attr_for_modify(struct ibv_port_attr *port_attr, struct doca_verbs_qp_attr *qp_attr, 
  struct remote_info *l_info, struct remote_info *r_info,
  struct ibv_context *ib_context) {
  int status = 0;
  status = doca_verbs_qp_attr_set_dest_qp_num(qp_attr, r_info->qpn);
  assert(status == 0);
  struct doca_verbs_ah_attr *ah = nullptr;
  status = doca_verbs_ah_attr_create(ib_context, &ah);
  assert(status == 0);
  if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    status = doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH);
  } else {
    status = doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IPv4);
  }
  assert(status == 0);
  status = doca_verbs_ah_attr_set_dlid(ah, r_info->lid);
  assert(status == 0);
  status = doca_verbs_ah_attr_set_gid(ah, *((struct doca_verbs_gid *)(&r_info->gid)));
  assert(status == 0);
  status = doca_verbs_ah_attr_set_sl(ah, 0);
  assert(status == 0);
  status = doca_verbs_ah_attr_set_sgid_index(ah, l_info->gid_index);
  assert(status == 0);
  status = doca_verbs_ah_attr_set_hop_limit(ah, DEF_HOP_LIMIT);
  assert(status == 0);
  status = doca_verbs_ah_attr_set_traffic_class(ah, DEF_IB_TC);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_ah_attr(qp_attr, ah);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_pkey_index(qp_attr, PKEY_INDEX);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_rq_psn(qp_attr, 0);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_sq_psn(qp_attr, 0);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_path_mtu(qp_attr, DOCA_VERBS_MTU_SIZE_4K_BYTES);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_min_rnr_timer(qp_attr, 12);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_ack_timeout(qp_attr, 20);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_retry_cnt(qp_attr, 7);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_rnr_retry(qp_attr, 7);
  assert(status == 0);
  return 0;
}


int doca_gpunetio_test_change_qp_state(struct doca_gpu_verbs_qp_hl *qp,
                                       struct doca_verbs_qp_attr *qp_attr,
                                       int attr_mask) {
  int status;
  int init_mask = attr_mask;
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_INIT);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to INIT: %d\n", status);
    return status;
  }
  attr_mask = init_mask;
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to INIT\n");
    return status;
  }
  attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
              DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_AH_ATTR |
              DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER;
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTR);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to RTR: %d\n", status);
    return status;
  }
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to RTR\n");
    return status;
  }
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTS);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to RTS: %d\n", status);
    return status;
  }
  attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
              DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
              DOCA_VERBS_QP_ATTR_RNR_RETRY;
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to RTS\n");
    return status;
  }
  return 0;
}

int setup_qp_attr_and_set_qp(struct gverbs_context *g_ctx, struct ibv_context *ib_context, struct ibv_port_attr *port_attr,
  struct remote_info *rem_dest, struct doca_verbs_qp_attr *qp_attr,
  int num_of_blocks, int num_of_nodes, int node_rank, uint32_t qp_cnt) {
  int attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                  DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PORT_NUM |
                  DOCA_VERBS_QP_ATTR_PKEY_INDEX;
  for (int qp_idx = 0; qp_idx < num_of_blocks; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node = peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int curr_qp_idx = peer_idx + qp_idx * (num_of_nodes - 1);
      int local_idx = curr_qp_idx + node_rank * qp_cnt;
      int rem_idx = actual_node_idx * qp_cnt + qp_idx * (num_of_nodes - 1) + actual_idx_in_node;
      struct remote_info *l_info = &rem_dest[local_idx];
      struct remote_info *r_info = &rem_dest[rem_idx];
      struct doca_gpu_verbs_qp_hl *qp = g_ctx->qp_hls[curr_qp_idx];
      setup_qp_attr_for_modify(port_attr, qp_attr, l_info, r_info, ib_context);
      doca_gpunetio_test_change_qp_state(qp, qp_attr, attr_mask);
    }
  }
  return 0;
}

bool RDMACoordinator::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
  bool changed = false;
  changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
  changed |= grow_to(buf_config.hidden_dim, config.hidden_dim);
  changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
  changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
  changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
  changed |= grow_to(buf_config.num_of_blocks_dispatch_api, config.num_of_blocks_dispatch_api);
  changed |= grow_to(buf_config.num_of_blocks_combine_api, config.num_of_blocks_combine_api);
  if (buf_config.num_of_tokens_per_chunk_dispatch_api != config.num_of_tokens_per_chunk_dispatch_api) {
    changed = true;
    buf_config.num_of_tokens_per_chunk_dispatch_api = config.num_of_tokens_per_chunk_dispatch_api;
  }
  if (buf_config.num_of_tokens_per_chunk_combine_api != config.num_of_tokens_per_chunk_combine_api) {
    changed = true;
    buf_config.num_of_tokens_per_chunk_combine_api = config.num_of_tokens_per_chunk_combine_api;
  }
  return changed;
}

void RDMACoordinator::update_config(BufferConfig config) {
  this->buffer_config = config;
}

void RDMACoordinator::allocate_buffers() {
  allocate_combine_buffers();
  allocate_dispatch_buffers();
}

void RDMACoordinator::init(  
      pybind11::object process_group,
      int node_rank,
      int local_rank, 
      BufferConfig config
  ) {
  this->process_group = process_group;
  this->node_rank = node_rank;
  this->local_rank = local_rank;
  this->buffer_config = config;
  assert(buffer_config.num_of_nodes > 1);

  std::vector<int> gpu_idx_vec;
  // The node in config means the nvlink domain
  // The local device index is the index of the device in the real device list within the physical node. 
  int num_of_local_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_of_local_devices));
  num_of_local_devices = std::min(num_of_local_devices, buffer_config.num_of_ranks_per_node);
  int local_device_idx = local_rank % num_of_local_devices;
  for (int i = 0; i < num_of_local_devices; ++i) {
    gpu_idx_vec.push_back(i);
  }
  // Get name of ibv device.
  const char *net_name = nullptr;
  get_nic_name(gpu_idx_vec, local_device_idx, &net_name);
  // Find ib device and get ibv_context.
  struct ibv_device *ib_dev = ctx_find_dev(net_name);

  ib_context = ibv_open_device(ib_dev);;
  auto transport_type = ib_context->device->transport_type;
  assert(transport_type == IBV_TRANSPORT_IB);
  ibv_query_port(ib_context, IB_PORT, &port_attr);
  uint8_t link_layer = port_attr.link_layer;
  assert(link_layer == IBV_LINK_LAYER_INFINIBAND || link_layer == IBV_LINK_LAYER_ETHERNET);
  hybrid_ep::ncclIbGetGidIndex(ib_context, IB_PORT, &port_attr, &gid_index);

  // Alloc protect domain.
  ib_pd = ibv_alloc_pd(ib_context);
  gpu_handler = (struct doca_gpu *)calloc(1, sizeof(struct doca_gpu));
  get_gpu_handler(gpu_handler, ib_context, local_device_idx);
  mr_access_flag = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                      IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_RELAXED_ORDERING;
  
  rdma_initialized = true;
}

void RDMACoordinator::allocate_dispatch_buffers(){
  dispatch_buffers.data_type = buffer_config.token_data_type;
  size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);

  // Calculate rdma buffers sizes
  auto attn_input_token_elts = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
  auto attn_input_prob_elts = buffer_config.max_num_of_tokens_per_rank 
                            * (buffer_config.num_of_experts_per_rank 
                            * buffer_config.num_of_ranks_per_node 
                            * buffer_config.num_of_nodes);
  auto attn_input_token_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank 
                                            * (buffer_config.hidden_dim / 128);
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank * 
                                          (buffer_config.num_of_nodes - 1) * 
                                          buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank 
                                        * (buffer_config.num_of_nodes - 1) 
                                        * (buffer_config.num_of_experts_per_rank 
                                        * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank * 
                                                    (buffer_config.num_of_nodes - 1) * (buffer_config.hidden_dim / 128);
  size_t rdma_inter_node_group_flags_barrier_idx = (size_t)((buffer_config.max_num_of_tokens_per_rank - 1) /
                                           buffer_config.num_of_tokens_per_chunk_dispatch_api + 1) *
                                          (buffer_config.num_of_nodes - 1);
  size_t rdma_inter_node_group_flags_elts = rdma_inter_node_group_flags_barrier_idx + 2 * (buffer_config.num_of_nodes - 1);
  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_token,
                        attn_input_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_prob,
                        attn_input_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_scaling_factor,
                        attn_input_token_scaling_factor_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_scaling_factor,
                        rdma_inter_node_group_scaling_factor_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.rdma_inter_node_group_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.attn_input_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));

  // Allocate RDMA flags here because it is needed by the device_sync kernel.
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));

  // Allocate memory region
  attn_input_token_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_token,
                        attn_input_token_elts * sizeof_token_data_type, mr_access_flag);
  dispatch_rdma_inter_node_group_token_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type, mr_access_flag);
  attn_input_flags_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  dispatch_rdma_inter_node_group_flags_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  attn_input_prob_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_prob,
                        attn_input_prob_elts * sizeof(float), mr_access_flag);
  dispatch_rdma_inter_node_group_prob_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float), mr_access_flag);
  attn_input_token_scaling_factor_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_scaling_factor,
                        attn_input_token_scaling_factor_elts * sizeof(float), mr_access_flag);
  dispatch_rdma_inter_node_group_scaling_factor_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_scaling_factor,
                        rdma_inter_node_group_scaling_factor_elts * sizeof(float), mr_access_flag);

  // Set dispatch queue pair attributes.
  int num_of_dispatch_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_dispatch_api;
  memset(&dispatch_gverbs_ctx, 0, sizeof(gverbs_context));
  ibv_query_gid(ib_context, IB_PORT, gid_index, &dispatch_gverbs_ctx.gid);
  dispatch_gverbs_ctx.qp_init_attr = (struct doca_gpu_verbs_qp_init_attr_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_init_attr_hl));
  setup_qp_init_attr(dispatch_gverbs_ctx.qp_init_attr, gpu_handler, ib_pd, 3 * buffer_config.max_num_of_tokens_per_rank + 1);
  dispatch_gverbs_ctx.qp_hls = (struct doca_gpu_verbs_qp_hl **)calloc(sizeof(struct doca_gpu_verbs_qp_hl *), num_of_dispatch_qps);
  create_and_place_qps(&dispatch_gverbs_ctx, dispatch_gverbs_ctx.qp_init_attr, num_of_dispatch_qps);
  doca_verbs_qp_attr_create(&dispatch_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_set_port_num(dispatch_gverbs_ctx.qp_attr, IB_PORT);
  doca_verbs_qp_attr_set_allow_remote_write(dispatch_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_read(dispatch_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_atomic(dispatch_gverbs_ctx.qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
  
  // Construct dispatch remote_info
  dispatch_remote_info_vec = static_cast<remote_info *>(calloc(buffer_config.num_of_nodes * num_of_dispatch_qps, sizeof(remote_info)));
  remote_info *my_dispatch_info = static_cast<remote_info *>(calloc(num_of_dispatch_qps, sizeof(remote_info)));
  int token_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
  int prob_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node;
  int scaling_factor_stride = buffer_config.max_num_of_tokens_per_rank * (buffer_config.hidden_dim / 128);
  // For each queue pair to the same remote. 
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_dispatch_api; ++qp_idx) {
    // For each remote.
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      // Fill rkeys and raddrs into remote_info.
      int idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      struct remote_info *curr_info = my_dispatch_info + idx;
      curr_info->lid = port_attr.lid;
      curr_info->qpn = doca_verbs_qp_get_qpn(dispatch_gverbs_ctx.qp_hls[idx]->qp);
      curr_info->gid_index = gid_index;
      memset(&curr_info->gid, 0, sizeof(curr_info->gid));;
      memcpy(curr_info->gid.raw, dispatch_gverbs_ctx.gid.raw, 16);
      curr_info->token_rkey = dispatch_rdma_inter_node_group_token_mr->rkey;
      switch (dispatch_buffers.data_type) {
        case APP_TOKEN_DATA_TYPE::UINT8:
          curr_info->token_vaddr = (uintptr_t)((uint8_t *)dispatch_rdma_inter_node_group_token_mr->addr + peer_idx * token_stride);
          break;
        case APP_TOKEN_DATA_TYPE::UINT16:
          curr_info->token_vaddr = (uintptr_t)((uint16_t *)dispatch_rdma_inter_node_group_token_mr->addr + peer_idx * token_stride);
          break;
      }
      curr_info->flag_rkey = dispatch_rdma_inter_node_group_flags_mr->rkey;
      curr_info->flag_vaddr = (uintptr_t)dispatch_rdma_inter_node_group_flags_mr->addr;
      curr_info->prob_rkey = dispatch_rdma_inter_node_group_prob_mr->rkey;
      curr_info->prob_vaddr = (uintptr_t)((float *)dispatch_rdma_inter_node_group_prob_mr->addr +
                                          peer_idx * prob_stride);
      curr_info->scaling_factor_rkey = dispatch_rdma_inter_node_group_scaling_factor_mr->rkey;
      curr_info->scaling_factor_vaddr = (uintptr_t)((float *)dispatch_rdma_inter_node_group_scaling_factor_mr->addr +
                                                    peer_idx * scaling_factor_stride);
    }
  }
  exchange_remote_rdma_info(dispatch_remote_info_vec, my_dispatch_info, num_of_dispatch_qps);

  // Init queue pairs.
  setup_qp_attr_and_set_qp(&dispatch_gverbs_ctx, ib_context, &port_attr,
    dispatch_remote_info_vec, dispatch_gverbs_ctx.qp_attr,
    buffer_config.num_of_blocks_dispatch_api, buffer_config.num_of_nodes, node_rank, num_of_dispatch_qps);
  // Move queue pairs to GPU.
  doca_gpu_dev_verbs_qp **h_qps_gpu = (doca_gpu_dev_verbs_qp**)calloc(sizeof(*h_qps_gpu), num_of_dispatch_qps);
  for (int idx = 0; idx <  num_of_dispatch_qps; ++idx) {
    doca_gpu_verbs_get_qp_dev(dispatch_gverbs_ctx.qp_hls[idx]->qp_gverbs, &h_qps_gpu[idx]);
  }
  CUDA_CHECK(cudaMalloc(&dispatch_gverbs_ctx.d_qps_gpu, num_of_dispatch_qps * sizeof(doca_gpu_dev_verbs_qp*)));
  CUDA_CHECK(cudaMemcpy(dispatch_gverbs_ctx.d_qps_gpu, h_qps_gpu, num_of_dispatch_qps * sizeof(doca_gpu_dev_verbs_qp*), cudaMemcpyHostToDevice));
  // Move Memory regions to GPU.
  dispatch_mr_info_h = (dispatch_memory_region_info_t *)calloc(sizeof(dispatch_memory_region_info_t), num_of_dispatch_qps);
  CUDA_CHECK(cudaMalloc((void**)&dispatch_mr_info_d, num_of_dispatch_qps * sizeof(dispatch_memory_region_info_t)));
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_dispatch_api; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node = peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int my_idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      int rem_idx = actual_node_idx * num_of_dispatch_qps + qp_idx * (buffer_config.num_of_nodes - 1) + actual_idx_in_node;
      struct dispatch_memory_region_info_t *data = dispatch_mr_info_h + my_idx;
      data->token_laddr = (uint64_t)attn_input_token_mr->addr;
      data->token_lkey = htobe32(attn_input_token_mr->lkey);
      data->token_raddr = dispatch_remote_info_vec[rem_idx].token_vaddr;
      data->token_rkey = htobe32(dispatch_remote_info_vec[rem_idx].token_rkey);
      data->scaling_factor_laddr = (uint64_t)attn_input_token_scaling_factor_mr->addr;
      data->scaling_factor_lkey = htobe32(attn_input_token_scaling_factor_mr->lkey);
      data->scaling_factor_raddr = dispatch_remote_info_vec[rem_idx].scaling_factor_vaddr;
      data->scaling_factor_rkey = htobe32(dispatch_remote_info_vec[rem_idx].scaling_factor_rkey);
      data->flag_laddr = (uint64_t)attn_input_flags_mr->addr;
      data->flag_lkey = htobe32(attn_input_flags_mr->lkey);
      data->flag_raddr = dispatch_remote_info_vec[rem_idx].flag_vaddr;
      data->flag_rkey = htobe32(dispatch_remote_info_vec[rem_idx].flag_rkey);
      data->back_sync_barrier_idx = rdma_inter_node_group_flags_barrier_idx;
      data->prob_laddr = (uint64_t)attn_input_prob_mr->addr;
      data->prob_lkey = htobe32(attn_input_prob_mr->lkey);
      data->prob_raddr = dispatch_remote_info_vec[rem_idx].prob_vaddr;
      data->prob_rkey = htobe32(dispatch_remote_info_vec[rem_idx].prob_rkey);
    }
  }
  CUDA_CHECK(cudaMemcpy(dispatch_mr_info_d, dispatch_mr_info_h, num_of_dispatch_qps * sizeof(dispatch_memory_region_info_t), cudaMemcpyHostToDevice));

  // Set RDMA attributes to dispatch buffers.
  dispatch_buffers.d_qps_gpu = dispatch_gverbs_ctx.d_qps_gpu;
  dispatch_buffers.mr_info = dispatch_mr_info_d;

  // Free temporary resources.
  free(my_dispatch_info);
  free(h_qps_gpu);
  buffer_allocated = true;
}

void RDMACoordinator::allocate_combine_buffers(){
  // Calculate rdma buffers sizes
  auto rdma_intra_node_red_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_intra_node_red_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                       (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                          (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                         (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  size_t combine_rdma_inter_node_group_flags_barrier_idx = (size_t)((buffer_config.max_num_of_tokens_per_rank - 1) /
                                           buffer_config.num_of_tokens_per_chunk_combine_api + 1) *
                                          (buffer_config.num_of_nodes - 1);
  size_t rdma_inter_node_group_flags_elts = combine_rdma_inter_node_group_flags_barrier_idx + 2 * (buffer_config.num_of_nodes - 1);
                                    
  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_token,
                        rdma_intra_node_red_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_prob,
                        rdma_intra_node_red_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.rdma_inter_node_group_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.attn_output_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.attn_output_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));

  // Allocate RDMA flags here because it is needed by the device_sync kernel.
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));

  rdma_intra_node_red_token_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_intra_node_red_token,
                        rdma_intra_node_red_token_elts * sizeof(uint16_t), mr_access_flag);
  combine_rdma_inter_node_group_token_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof(uint16_t), mr_access_flag);
  attn_output_flags_mr = ibv_reg_mr(ib_pd, combine_buffers.attn_output_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  combine_rdma_inter_node_group_flags_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  rdma_intra_node_red_prob_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_intra_node_red_prob,
                        rdma_intra_node_red_prob_elts * sizeof(float), mr_access_flag);
  combine_rdma_inter_node_group_prob_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float), mr_access_flag);

  // Set combine queue pair attributes.
  int num_of_combine_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_combine_api;
  memset(&combine_gverbs_ctx, 0, sizeof(gverbs_context));
  ibv_query_gid(ib_context, IB_PORT, gid_index, &combine_gverbs_ctx.gid);
  combine_gverbs_ctx.qp_init_attr = (struct doca_gpu_verbs_qp_init_attr_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_init_attr_hl));
  setup_qp_init_attr(combine_gverbs_ctx.qp_init_attr, gpu_handler, ib_pd, 2 * buffer_config.max_num_of_tokens_per_rank + 1);
  combine_gverbs_ctx.qp_hls = (struct doca_gpu_verbs_qp_hl **)calloc(sizeof(struct doca_gpu_verbs_qp_hl *), num_of_combine_qps);
  create_and_place_qps(&combine_gverbs_ctx, combine_gverbs_ctx.qp_init_attr, num_of_combine_qps);
  doca_verbs_qp_attr_create(&combine_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_set_port_num(combine_gverbs_ctx.qp_attr, IB_PORT);
  doca_verbs_qp_attr_set_allow_remote_write(combine_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_read(combine_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_atomic(combine_gverbs_ctx.qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);

  // Construct combine remote_info
  combine_remote_info_vec = static_cast<remote_info *>(calloc(buffer_config.num_of_nodes * num_of_combine_qps, sizeof(remote_info)));
  remote_info *my_combine_info = static_cast<remote_info *>(calloc(num_of_combine_qps, sizeof(remote_info)));
  int token_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
  int prob_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node;
  // For each queue pair to the same remote. 
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_combine_api; ++qp_idx) {
    // For each remote.
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      // Fill rkeys and raddrs into remote_info.
      int idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      struct remote_info *curr_info = my_combine_info + idx;
      curr_info->lid = port_attr.lid;
      curr_info->qpn = doca_verbs_qp_get_qpn(combine_gverbs_ctx.qp_hls[idx]->qp);
      curr_info->gid_index = gid_index;
      memset(&curr_info->gid, 0, sizeof(curr_info->gid));;
      memcpy(curr_info->gid.raw, combine_gverbs_ctx.gid.raw, 16);
      curr_info->token_rkey = combine_rdma_inter_node_group_token_mr->rkey;
      curr_info->token_vaddr = (uintptr_t)((uint16_t *)combine_rdma_inter_node_group_token_mr->addr +
                                           peer_idx * token_stride);
      curr_info->flag_rkey = combine_rdma_inter_node_group_flags_mr->rkey;
      curr_info->flag_vaddr = (uintptr_t)combine_rdma_inter_node_group_flags_mr->addr;
      curr_info->prob_rkey = combine_rdma_inter_node_group_prob_mr->rkey;
      curr_info->prob_vaddr = (uintptr_t)((float *)combine_rdma_inter_node_group_prob_mr->addr +
                                          peer_idx * prob_stride);
    }
  }

  exchange_remote_rdma_info(combine_remote_info_vec, my_combine_info, num_of_combine_qps);

  // Init queue pairs.
  setup_qp_attr_and_set_qp(&combine_gverbs_ctx, ib_context, &port_attr,
    combine_remote_info_vec, combine_gverbs_ctx.qp_attr,
    buffer_config.num_of_blocks_combine_api, buffer_config.num_of_nodes, node_rank, num_of_combine_qps);
  // Move queue pairs to GPU.
  doca_gpu_dev_verbs_qp **h_qps_gpu = (doca_gpu_dev_verbs_qp**)calloc(sizeof(*h_qps_gpu), num_of_combine_qps);
  for (int idx = 0; idx <  num_of_combine_qps; ++idx) {
    doca_gpu_verbs_get_qp_dev(combine_gverbs_ctx.qp_hls[idx]->qp_gverbs, &h_qps_gpu[idx]);
  }
  CUDA_CHECK(cudaMalloc(&combine_gverbs_ctx.d_qps_gpu, num_of_combine_qps * sizeof(doca_gpu_dev_verbs_qp*)));
  CUDA_CHECK(cudaMemcpy(combine_gverbs_ctx.d_qps_gpu, h_qps_gpu, num_of_combine_qps * sizeof(doca_gpu_dev_verbs_qp*), cudaMemcpyHostToDevice));
  // Move Memory regions to GPU.
  combine_mr_info_h = (combine_memory_region_info_t *)calloc(sizeof(combine_memory_region_info_t), num_of_combine_qps);
  CUDA_CHECK(cudaMalloc((void**)&combine_mr_info_d, num_of_combine_qps * sizeof(combine_memory_region_info_t)));
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_combine_api; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node = peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int my_idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      int rem_idx = actual_node_idx * num_of_combine_qps + qp_idx * (buffer_config.num_of_nodes - 1) + actual_idx_in_node;
      struct combine_memory_region_info_t *data = combine_mr_info_h + my_idx;
      data->token_laddr = (uint64_t)((uint16_t *)rdma_intra_node_red_token_mr->addr);
      data->token_lkey = htobe32(rdma_intra_node_red_token_mr->lkey);
      data->token_raddr = combine_remote_info_vec[rem_idx].token_vaddr;
      data->token_rkey = htobe32(combine_remote_info_vec[rem_idx].token_rkey);
      data->flag_laddr = (uint64_t)attn_output_flags_mr->addr;
      data->flag_lkey = htobe32(attn_output_flags_mr->lkey);
      data->flag_raddr = combine_remote_info_vec[rem_idx].flag_vaddr;
      data->flag_rkey = htobe32(combine_remote_info_vec[rem_idx].flag_rkey);
      data->back_sync_barrier_idx = combine_rdma_inter_node_group_flags_barrier_idx;
      data->prob_laddr = (uint64_t)rdma_intra_node_red_prob_mr->addr;
      data->prob_lkey = htobe32(rdma_intra_node_red_prob_mr->lkey);
      data->prob_raddr = combine_remote_info_vec[rem_idx].prob_vaddr;
      data->prob_rkey = htobe32(combine_remote_info_vec[rem_idx].prob_rkey);
    }
  }
  CUDA_CHECK(cudaMemcpy(combine_mr_info_d, combine_mr_info_h, num_of_combine_qps * sizeof(combine_memory_region_info_t), cudaMemcpyHostToDevice));

  // Set RDMA attributes to combine buffers.
  combine_buffers.d_qps_gpu = combine_gverbs_ctx.d_qps_gpu;
  combine_buffers.mr_info = combine_mr_info_d;
  // Free temporary resources.
  free(my_combine_info);
  free(h_qps_gpu);
  buffer_allocated = true;
}    

void RDMACoordinator::destroy() {
  CUDA_CHECK(cudaDeviceSynchronize());

  // Close memory regions 
  #define CLOSE_MR(mr)                 \
    do {                               \
      if ((mr) != nullptr) {           \
        ibv_dereg_mr((mr));            \
        (mr) = nullptr;                \
      }                                \
    } while (0)
  // Free misc resources.
  #define FREE_CUDA_MEMORY(ptr)            \
    do {                                   \
      if ((ptr) != nullptr) {              \
        CUDA_CHECK(cudaFree((ptr)));       \
        (ptr) = nullptr;                   \
      }                                    \
    } while (0)
  #define FREE_CPU_MEMORY(ptr)             \
    do {                                   \
      if ((ptr) != nullptr) {              \
        free((ptr));                       \
        (ptr) = nullptr;                   \
      }                                    \
    } while (0)

  CLOSE_MR(attn_input_token_mr);
  CLOSE_MR(dispatch_rdma_inter_node_group_token_mr);
  CLOSE_MR(attn_input_flags_mr);
  CLOSE_MR(dispatch_rdma_inter_node_group_flags_mr);
  CLOSE_MR(attn_input_prob_mr);
  CLOSE_MR(dispatch_rdma_inter_node_group_prob_mr);
  CLOSE_MR(attn_input_token_scaling_factor_mr);
  CLOSE_MR(dispatch_rdma_inter_node_group_scaling_factor_mr);
  CLOSE_MR(rdma_intra_node_red_token_mr);
  CLOSE_MR(combine_rdma_inter_node_group_token_mr);
  CLOSE_MR(rdma_intra_node_red_prob_mr);
  CLOSE_MR(combine_rdma_inter_node_group_prob_mr);
  CLOSE_MR(attn_output_flags_mr);
  CLOSE_MR(combine_rdma_inter_node_group_flags_mr);


  FREE_CPU_MEMORY(dispatch_remote_info_vec);
  FREE_CPU_MEMORY(dispatch_mr_info_h);
  FREE_CUDA_MEMORY(dispatch_gverbs_ctx.d_qps_gpu);
  FREE_CUDA_MEMORY(dispatch_mr_info_d);
  FREE_CPU_MEMORY(combine_remote_info_vec);
  FREE_CPU_MEMORY(combine_mr_info_h);
  FREE_CUDA_MEMORY(combine_gverbs_ctx.d_qps_gpu);
  FREE_CUDA_MEMORY(combine_mr_info_d);

  FREE_CUDA_MEMORY(dispatch_buffers.rdma_inter_node_group_token);
  FREE_CUDA_MEMORY(dispatch_buffers.rdma_inter_node_group_prob);
  FREE_CUDA_MEMORY(dispatch_buffers.rdma_inter_node_group_scaling_factor);
  FREE_CUDA_MEMORY(dispatch_buffers.rdma_inter_node_group_flags);
  FREE_CUDA_MEMORY(dispatch_buffers.attn_input_flags);
  FREE_CUDA_MEMORY(dispatch_buffers.attn_input_token);
  FREE_CUDA_MEMORY(dispatch_buffers.attn_input_prob);
  FREE_CUDA_MEMORY(dispatch_buffers.attn_input_scaling_factor);
  FREE_CUDA_MEMORY(dispatch_buffers.expected_rdma_flag_value);
  FREE_CUDA_MEMORY(combine_buffers.rdma_intra_node_red_token);
  FREE_CUDA_MEMORY(combine_buffers.rdma_intra_node_red_prob);
  FREE_CUDA_MEMORY(combine_buffers.rdma_inter_node_group_token);
  FREE_CUDA_MEMORY(combine_buffers.rdma_inter_node_group_prob);
  FREE_CUDA_MEMORY(combine_buffers.rdma_inter_node_group_flags);
  FREE_CUDA_MEMORY(combine_buffers.attn_output_flags);
  FREE_CUDA_MEMORY(combine_buffers.expected_rdma_flag_value);

  // If we use doca_gpu_verbs_destroy_qp_hl and re-allocate RDMA resources, "part or all of the requested memory range is already mapped" occurs. Do not know why now, so just comment it out.
  int num_of_dispatch_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_dispatch_api;
  int num_of_combine_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_combine_api;
  for (int idx = 0; idx < num_of_dispatch_qps; ++idx) {
    doca_gpu_verbs_destroy_qp_hl(dispatch_gverbs_ctx.qp_hls[idx]);
  }
  for (int idx = 0; idx < num_of_combine_qps; ++idx) {
    doca_gpu_verbs_destroy_qp_hl(combine_gverbs_ctx.qp_hls[idx]);
  }
  FREE_CPU_MEMORY(dispatch_gverbs_ctx.qp_hls);
  FREE_CPU_MEMORY(dispatch_gverbs_ctx.qp_init_attr);
  FREE_CPU_MEMORY(combine_gverbs_ctx.qp_hls);
  FREE_CPU_MEMORY(combine_gverbs_ctx.qp_init_attr);
  doca_verbs_qp_attr_destroy(dispatch_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_destroy(combine_gverbs_ctx.qp_attr);

  buffer_allocated = false;
  #undef CLOSE_MR
  #undef FREE_CUDA_MEMORY
  #undef FREE_CPU_MEMORY
}

void RDMACoordinator::exchange_remote_rdma_info(remote_info* dst, remote_info *src, int num_of_qps) {
  auto torch_distributed = py::module_::import("torch.distributed");
  auto num_bytes = static_cast<int64_t>(num_of_qps) *
                static_cast<int64_t>(sizeof(remote_info));
  torch::Tensor buffer = torch::empty({num_bytes}, at::device(at::kCPU).dtype(at::kByte));
  memcpy(buffer.data_ptr<uint8_t>(), reinterpret_cast<void *>(src), num_of_qps * sizeof(remote_info));
  buffer = buffer.cuda();

  // Get world size from process group
  int world_size = process_group.attr("size")().cast<int>();
  // Create empty tensors for allgather output
  py::list output_list;
  for (int i = 0; i < world_size; i++) {
    output_list.append(torch::empty_like(buffer));
  }

  torch_distributed.attr("all_gather")(output_list, buffer, process_group);

  // Move the gathered remote info to CPU.
  for(int i = local_rank; i < world_size; i += buffer_config.num_of_ranks_per_node) {
    auto tensor = output_list[i].cast<torch::Tensor>().cpu();
    memcpy(dst + num_of_qps * (i / buffer_config.num_of_ranks_per_node), tensor.data_ptr<uint8_t>(), num_of_qps * sizeof(remote_info));
  }
}

RDMACoordinator::~RDMACoordinator() {
  if(buffer_allocated) {
    destroy();
  }
  
  if(rdma_initialized) {
    // Dealloc protect domain.
    ibv_dealloc_pd(ib_pd);
    // Close device.
    ibv_close_device(ib_context);
    delete gpu_handler->mtable;
    if(gpu_handler != nullptr) {
      free(gpu_handler);
      gpu_handler = nullptr;
    }
  }

  rdma_initialized = false;
  buffer_allocated = false;
}