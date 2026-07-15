// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "allocator/allocator.cuh"
#include "hybrid_ep.cuh"
#include "utils.cuh"
#include "config.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HybridEP, efficiently enable the expert-parallel communication in "
              "the Hopper+ architectures";
    m.attr("SM_ARCH") = SM_ARCH;
    
    pybind11::class_<ExtendedMemoryAllocator>(m, "ExtendedMemoryAllocator")
        .def(py::init<>())
        .def("detect_accessible_ranks", &ExtendedMemoryAllocator::detect_accessible_ranks, py::arg("process_group"));
      
    pybind11::enum_<APP_TOKEN_DATA_TYPE>(m, "APP_TOKEN_DATA_TYPE")
        .value("UINT16", APP_TOKEN_DATA_TYPE::UINT16)
        .value("UINT8", APP_TOKEN_DATA_TYPE::UINT8)
        .export_values() // So we can use hybrid_ep_cpp.TYPE instead of the
                         // hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.TYPE
        .def("__str__",
             [](const APP_TOKEN_DATA_TYPE &type) { return type_to_string(type); });
  
    pybind11::class_<BufferConfig>(m, "BufferConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &BufferConfig::hidden_dim)
        .def_readwrite("max_num_of_tokens_per_rank", &BufferConfig::max_num_of_tokens_per_rank)
        .def_readwrite("num_of_experts_per_rank", &BufferConfig::num_of_experts_per_rank)
        .def_readwrite("num_of_ranks_per_node", &BufferConfig::num_of_ranks_per_node)
        .def_readwrite("num_of_nodes", &BufferConfig::num_of_nodes)
        .def_readwrite("token_data_type", &BufferConfig::token_data_type)
        .def_readwrite("num_of_blocks_preprocessing_api", &BufferConfig::num_of_blocks_preprocessing_api)
        .def_readwrite("num_of_blocks_dispatch_api", &BufferConfig::num_of_blocks_dispatch_api)
        .def_readwrite("num_of_blocks_combine_api", &BufferConfig::num_of_blocks_combine_api)
        .def_readwrite("num_of_tokens_per_chunk_dispatch_api", &BufferConfig::num_of_tokens_per_chunk_dispatch_api)
        .def_readwrite("num_of_tokens_per_chunk_combine_api", &BufferConfig::num_of_tokens_per_chunk_combine_api)
        .def_readwrite("num_of_dispatch_chunks", &BufferConfig::num_of_dispatch_chunks)
        .def_readwrite("num_of_combine_chunks", &BufferConfig::num_of_combine_chunks)
        .def("is_valid", &BufferConfig::is_valid)
        .def("__repr__", [](const BufferConfig &config) {
          return "<BufferConfig hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " num_of_experts_per_rank=" + std::to_string(config.num_of_experts_per_rank) +
                 " num_of_ranks_per_node=" + std::to_string(config.num_of_ranks_per_node) +
                 " num_of_nodes=" + std::to_string(config.num_of_nodes) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 " num_of_blocks_preprocessing_api=" + std::to_string(config.num_of_blocks_preprocessing_api) + 
                 " num_of_blocks_dispatch_api=" + std::to_string(config.num_of_blocks_dispatch_api) + 
                 " num_of_blocks_combine_api=" + std::to_string(config.num_of_blocks_combine_api) + 
                 " num_of_tokens_per_chunk_dispatch_api=" + std::to_string(config.num_of_tokens_per_chunk_dispatch_api) + 
                 " num_of_tokens_per_chunk_combine_api=" + std::to_string(config.num_of_tokens_per_chunk_combine_api) + 
                 " num_of_dispatch_chunks=" + std::to_string(config.num_of_dispatch_chunks) +
                 " num_of_combine_chunks=" + std::to_string(config.num_of_combine_chunks) + ">";
        });

    pybind11::class_<HybridEpConfigInstance>(m, "HybridEpConfigInstance")
        .def(py::init<>())
        // Hybrid-ep Config
        .def_readwrite("hidden_dim", &HybridEpConfigInstance::hidden_dim)
        .def_readwrite("max_num_of_tokens_per_rank",
                       &HybridEpConfigInstance::max_num_of_tokens_per_rank)
        .def_readwrite("num_of_experts_per_rank",
                       &HybridEpConfigInstance::num_of_experts_per_rank)
        .def_readwrite("num_of_ranks_per_node",
                       &HybridEpConfigInstance::num_of_ranks_per_node)
        .def_readwrite("num_of_nodes", &HybridEpConfigInstance::num_of_nodes)
        .def_readwrite("pad_multiple", &HybridEpConfigInstance::pad_multiple)
        // Metadata-preprocessing API Config
        .def_readwrite("num_of_tokens_per_chunk_preprocessing_api", &HybridEpConfigInstance::num_of_tokens_per_chunk_preprocessing_api)
        .def_readwrite(
            "num_of_threads_per_block_preprocessing_api",
            &HybridEpConfigInstance::num_of_threads_per_block_preprocessing_api)
        .def_readwrite("num_of_blocks_preprocessing_api",
                       &HybridEpConfigInstance::num_of_blocks_preprocessing_api)
        .def_readwrite("num_of_blocks_permute",
                       &HybridEpConfigInstance::num_of_blocks_permute)
        .def_readwrite("num_of_blocks_unpermute",
                       &HybridEpConfigInstance::num_of_blocks_unpermute)
        // Dispatch API Config
        .def_readwrite("token_data_type", &HybridEpConfigInstance::token_data_type)
        .def_readwrite("num_of_stages_dispatch_api",
                       &HybridEpConfigInstance::num_of_stages_dispatch_api)
        .def_readwrite("num_of_stages_permute_block_dispatch_api",
                       &HybridEpConfigInstance::num_of_stages_permute_block_dispatch_api)
        .def_readwrite("num_of_in_flight_s2g_dispatch_api",
                       &HybridEpConfigInstance::num_of_in_flight_s2g_dispatch_api)
        .def_readwrite("num_of_in_flight_s2g_permute_block_dispatch_api",
                       &HybridEpConfigInstance::num_of_in_flight_s2g_permute_block_dispatch_api)
        .def_readwrite("num_of_additional_in_flight_s2g_dispatch_api",
                       &HybridEpConfigInstance::num_of_additional_in_flight_s2g_dispatch_api)
        .def_readwrite("num_of_tokens_per_chunk_dispatch_api",
                       &HybridEpConfigInstance::num_of_tokens_per_chunk_dispatch_api)
        .def_readwrite("num_of_blocks_dispatch_api",
                       &HybridEpConfigInstance::num_of_blocks_dispatch_api)
        .def_readwrite("forward_dispatch_api",
                       &HybridEpConfigInstance::forward_dispatch_api)
        .def_readwrite("device_side_sync_dispatch_api",
                       &HybridEpConfigInstance::device_side_sync_dispatch_api)
        // Combine API Config
        .def_readwrite("num_of_stages_g2s_combine_api",
                       &HybridEpConfigInstance::num_of_stages_g2s_combine_api)
        .def_readwrite("num_of_stages_s2g_combine_api",
                       &HybridEpConfigInstance::num_of_stages_s2g_combine_api)
        .def_readwrite("num_of_tokens_per_chunk_combine_api",
                       &HybridEpConfigInstance::num_of_tokens_per_chunk_combine_api)
        .def_readwrite("num_of_tokens_per_group_combine_api",
                       &HybridEpConfigInstance::num_of_tokens_per_group_combine_api)
        .def_readwrite("num_of_blocks_combine_api",
                       &HybridEpConfigInstance::num_of_blocks_combine_api)
        .def_readwrite(
            "num_of_additional_in_flight_s2g_combine_api",
            &HybridEpConfigInstance::num_of_additional_in_flight_s2g_combine_api)
        .def_readwrite("backward_combine_api",
                       &HybridEpConfigInstance::backward_combine_api)
        .def_readwrite("device_side_sync_combine_api",
                       &HybridEpConfigInstance::device_side_sync_combine_api)
        .def("is_valid", &HybridEpConfigInstance::is_valid, py::arg("fuse_permute_dispatch") = false)
        .def("__repr__", [](const HybridEpConfigInstance &config) {
          return "<HybridEpConfigInstance hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 ">";
        });

    pybind11::class_<Configurer>(m, "Configurer")
        .def(py::init<int, int, int, int, int, bool,
                      std::optional<int>, std::optional<int>, std::optional<int>,
                      std::optional<int>, std::optional<int>>(),
            py::arg("hidden_dim"),
            py::arg("max_num_of_tokens_per_rank"),
            py::arg("num_local_experts"),
            py::arg("num_of_ranks_per_node"),
            py::arg("num_of_nodes"),
            py::arg("use_fp8") = false,
            py::arg("num_sms_dispatch_api") = std::nullopt,
            py::arg("num_sms_combine_api") = std::nullopt,
            py::arg("num_sms_preprocessing_api") = std::nullopt,
            py::arg("num_blocks_permute") = std::nullopt,
            py::arg("num_blocks_unpermute") = std::nullopt)
        .def_readwrite("buffer_config", &Configurer::buffer_config)
        .def("get_default_config", &Configurer::get_default_config,
            py::arg("fuse_permute_dispatch") = false)
        .def("adjust_template", &Configurer::adjust_template,
            py::arg("config"),
            py::arg("fuse_permute_dispatch") = false);

    pybind11::class_<HandleImpl>(m, "HandleImpl")
        .def(py::init<>())
        .def_readwrite("sparse_to_dense_map", &HandleImpl::sparse_to_dense_map)
        .def_readwrite("rdma_to_attn_map", &HandleImpl::rdma_to_attn_map)
        .def_readwrite("attn_to_rdma_map", &HandleImpl::attn_to_rdma_map)
        .def_readwrite("num_dispatched_tokens_tensor", &HandleImpl::num_dispatched_tokens_tensor)
        .def_readwrite("local_expert_routing_map", &HandleImpl::local_expert_routing_map)
        .def_readwrite("num_of_tokens_per_rank", &HandleImpl::num_of_tokens_per_rank)
        .def_readwrite("config", &HandleImpl::config)
        .def_readwrite("tokens_per_expert", &HandleImpl::tokens_per_expert)
        .def_readwrite("padded_tokens_per_expert", &HandleImpl::padded_tokens_per_expert)
        .def_readwrite("overflow_flag", &HandleImpl::overflow_flag)
        .def_readwrite("num_permuted_tokens", &HandleImpl::num_permuted_tokens)
        .def_readwrite("dense_chunk_layout", &HandleImpl::dense_chunk_layout)
        .def_readwrite("dense_to_expert_map", &HandleImpl::dense_to_expert_map);

    pybind11::class_<HybridEPBuffer>(m, "HybridEPBuffer")
        .def(py::init<py::object, BufferConfig, int, int, int, std::string, std::string, std::string, std::string, bool, bool>(),
            py::arg("process_group"),
            py::arg("config"),
            py::arg("local_rank"),
            py::arg("node_rank"),
            py::arg("group_size"),
            py::arg("base_path"),
            py::arg("cuda_home"),
            py::arg("cccl_include_dir"),
            py::arg("jit_cache_dir"),
            py::arg("use_shared_buffer") = true,
            py::arg("enable_custom_allgather") = true)
        .def("update_buffer", &HybridEPBuffer::update_buffer, py::arg("config"))
        .def("metadata_preprocessing", &HybridEPBuffer::metadata_preprocessing,
             py::kw_only(),
             py::arg("config"),
             py::arg("routing_map"),
             py::arg("num_of_tokens_per_rank"),
             py::arg("num_permuted_tokens") = std::nullopt,
             py::arg("pad_multiple") = std::nullopt,
             py::arg("enable_permute") = false,
             py::arg("fuse_permute_dispatch") = false,
             py::arg("non_blocking") = false)
        .def("dispatch", &HybridEPBuffer::dispatch, py::kw_only(),
             py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("handle"),
             py::arg("with_probs"))
        .def("combine", &HybridEPBuffer::combine, py::kw_only(),
             py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("handle"),
             py::arg("with_probs"))
        .def("dispatch_with_permute", &HybridEPBuffer::dispatch_with_permute, py::kw_only(),
             py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("handle"),
             py::arg("pad_multiple") = std::nullopt,
             py::arg("fuse_permute_dispatch") = false,
             py::arg("non_blocking") = false,
             py::arg("with_probs") = false)
        .def("combine_with_unpermute", &HybridEPBuffer::combine_with_unpermute, py::kw_only(),
             py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("handle"),
             py::arg("pad_multiple") = std::nullopt,
             py::arg("fuse_unpermute_combine") = false,
             py::arg("with_probs") = false);    
    
  }
