# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import torch
import os
import shutil
import hybrid_ep_cpp
import warnings
from math import gcd

from .hybrid_ep_toolchain import runtime_paths

def _aligned_comm_rows(rows: int, ranks_per_node: int) -> int:
    # TMA requires 16-byte transfers; each physical routing row has one int32 per local rank.
    multiple = 4 // gcd(4, ranks_per_node)
    return (rows + multiple - 1) // multiple * multiple

def _pad_comm_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.size(0) == rows:
        return tensor
    return torch.cat((tensor, tensor.new_zeros((rows - tensor.size(0), *tensor.shape[1:]))))

class HybridEPHandle(tuple):
    """Tuple-compatible metadata handle carrying the unpadded token count."""

    def __new__(cls, fields: tuple, logical_num_tokens: int):
        handle = super().__new__(cls, fields)
        handle.logical_num_tokens = logical_num_tokens
        return handle

    def __reduce__(self):
        return type(self), (tuple(self), self.logical_num_tokens)

def indices_to_map(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_of_tokens: int,
    num_of_experts: int,
):
    """
    Map the map to the indices.
    """
    # Generate the routing map and the probs according to the topk_idx and topk_weights.
    assert topk_idx is not None
    routing_map = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.bool
    )
    routing_map = routing_map.scatter(1, topk_idx.to(torch.int64), 1).bool()
    if topk_weights is not None:
        probs = torch.zeros(
            num_of_tokens, num_of_experts, device="cuda", dtype=torch.float32
        )
        probs = probs.scatter(1, topk_idx.to(torch.int64), topk_weights)
    else:
        probs = None
    return routing_map, probs


class HybridEPBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        # Parameters for the hybrid-ep buffer allocation
        hidden_dim: int,
        max_num_of_tokens_per_rank: int,
        num_local_experts: int,
        use_fp8: bool = False,
        # Device-SM occupancy setting
        num_sms_dispatch_api: int = None,
        num_sms_combine_api: int = None,
        num_sms_preprocessing_api: int = None,
        num_blocks_permute: int = None,
        num_blocks_unpermute: int = None,
        # Experimental features
        use_shared_buffer: bool = True,
        enable_custom_allgather: bool = False,
        # Deprecated parameters
        num_of_hybrid_ep_ranks_per_nvlink_domain: int = None,
        use_mnnvl: bool = None
    ):
        self.group = group
        self.rank = self.group.rank()
        self.group_size = self.group.size()

        allocator = hybrid_ep_cpp.ExtendedMemoryAllocator()
        detected_ranks = allocator.detect_accessible_ranks(self.group)
        env_value = os.getenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
        if env_value is not None:
            self.num_of_hybrid_ep_ranks_per_nvlink_domain = int(env_value)
            if self.num_of_hybrid_ep_ranks_per_nvlink_domain != detected_ranks:
                warnings.warn(
                    f"[Warning] NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN={self.num_of_hybrid_ep_ranks_per_nvlink_domain} "
                    f"differs from detected value {detected_ranks}. Using environment variable."
                )
        else:
            self.num_of_hybrid_ep_ranks_per_nvlink_domain = detected_ranks
        
        assert (
            self.group_size % self.num_of_hybrid_ep_ranks_per_nvlink_domain == 0
        ), f"The number of ranks {self.group_size} should be divisible by the number of ranks per node {self.num_of_hybrid_ep_ranks_per_nvlink_domain} at rank={self.rank}."

        # Local rank: the active rank in the nvlink domain.
        self.local_rank = self.rank % self.num_of_hybrid_ep_ranks_per_nvlink_domain
        # Node rank: the active rank between the nvlink domains.
        self.node_rank = self.rank // self.num_of_hybrid_ep_ranks_per_nvlink_domain
        # The number of nodes.
        self.num_of_nodes = self.group_size // self.num_of_hybrid_ep_ranks_per_nvlink_domain
        max_num_of_tokens_per_rank = _aligned_comm_rows(
            max_num_of_tokens_per_rank,
            self.num_of_hybrid_ep_ranks_per_nvlink_domain,
        )
        self._num_of_tokens_per_rank = None
        # Create Configurer: auto-detects SM count, applies SM defaults, fills and validates BufferConfig.
        self.configurer = hybrid_ep_cpp.Configurer(
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_of_tokens_per_rank,
            num_local_experts=num_local_experts,
            num_of_ranks_per_node=self.num_of_hybrid_ep_ranks_per_nvlink_domain,
            num_of_nodes=self.num_of_nodes,
            use_fp8=use_fp8,
            num_sms_dispatch_api=num_sms_dispatch_api,
            num_sms_combine_api=num_sms_combine_api,
            num_sms_preprocessing_api=num_sms_preprocessing_api,
            num_blocks_permute=num_blocks_permute,
            num_blocks_unpermute=num_blocks_unpermute,
        )

        cuda_home, cccl_include_dir, self.jit_cache_path = runtime_paths()

        # Create C++ buffer - this will allocate all buffers during construction
        self.runtime = hybrid_ep_cpp.HybridEPBuffer(
            self.group,
            self.configurer.buffer_config,
            self.local_rank,
            self.node_rank,
            self.group_size,
            os.path.dirname(os.path.abspath(__file__)),
            cuda_home,
            cccl_include_dir,
            self.jit_cache_path,
            use_shared_buffer=use_shared_buffer,
            enable_custom_allgather=enable_custom_allgather,
        )

    def set_num_tokens_per_rank(self, rows: int) -> None:
        rows = _aligned_comm_rows(
            int(rows), self.num_of_hybrid_ep_ranks_per_nvlink_domain
        )
        capacity = int(self.configurer.buffer_config.max_num_of_tokens_per_rank)
        if rows <= 0 or rows > capacity:
            raise RuntimeError(
                f"HybridEP communication rows must be in [1, {capacity}], got {rows}"
            )
        self._num_of_tokens_per_rank = rows

    def _comm_rows(self, logical_rows: int) -> int:
        rows = self._num_of_tokens_per_rank
        if rows is None:
            raise RuntimeError(
                "HybridEP communication rows were not configured before dispatch"
            )
        if logical_rows > rows:
            raise RuntimeError(
                "HybridEP logical rows exceed the configured communication extent: "
                f"{logical_rows} > {rows}"
            )
        return rows

    def empty_jit_cache(self):
        '''
        Clean the cached kernel files.
        '''
        if os.path.exists(self.jit_cache_path):
            shutil.rmtree(self.jit_cache_path)

    def update_template_config(
        self,
        hidden_dim: int = None,
        num_of_tokens_per_rank: int = None,
        num_local_experts: int = None,
        pad_multiple: int = None,
        use_fp8: bool = None,
        fuse_permute_dispatch: bool = False,
        **kwargs,
    ):
        """
        Initialize the HybridEpConfigInstance which used to control the detailed setting of the hybrid-ep kernel.
        In common case, no need to change the default setting.
        """
        # Get a config with all env-var defaults and buffer-level state filled in.
        config = self.configurer.get_default_config(fuse_permute_dispatch)

        # Per-call dynamic overrides
        if hidden_dim is not None:
            config.hidden_dim = hidden_dim
        if num_of_tokens_per_rank is not None:
            # Align num_of_tokens_per_rank up to the nearest multiple of 16.
            num_of_tokens_per_rank = (num_of_tokens_per_rank + 15) // 16 * 16
            config.max_num_of_tokens_per_rank = max(
                num_of_tokens_per_rank,
                self.configurer.buffer_config.max_num_of_tokens_per_rank,
            )
            self.configurer.buffer_config.max_num_of_tokens_per_rank = config.max_num_of_tokens_per_rank
        if num_local_experts is not None:
            config.num_of_experts_per_rank = num_local_experts
        if pad_multiple is not None and pad_multiple > 0:
            config.pad_multiple = pad_multiple
        if use_fp8 is not None:
            config.token_data_type = (
                hybrid_ep_cpp.UINT8 if use_fp8 else hybrid_ep_cpp.UINT16
            )

        # Update the config with the kwargs.
        for key, value in kwargs.items():
            setattr(config, key, value)
        # Auto-tune stages based on current device shared memory limit.
        self.configurer.adjust_template(config, fuse_permute_dispatch)
        assert config.is_valid(fuse_permute_dispatch), "The config is not valid."

        # Use the runtime kernel config to update the buffer.
        self.runtime.update_buffer(config)
        return config

    def dispatch(
        self,
        hidden: torch.Tensor,
        scaling_factor: torch.Tensor = None,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        num_of_experts: int = None,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        num_dispatched_tokens_tensor: torch.Tensor = None,
        num_dispatched_tokens: int = None,
        handle: tuple = None,
    ):
        """
        Dispatch the data to the experts.

        Forward direction:
        dispatch_in_forward -> local_permute -> epxert_mlp -> local_unpermute -> combine_in_forward

        Backward direction:
        combine_in_backward <- local_unpermute -> expert_mlp -> local_permute -> dispatch_in_backward
        """
        hidden = hidden.contiguous()
        if probs is not None:
            probs = probs.contiguous()
        if scaling_factor is not None:
            scaling_factor = scaling_factor.contiguous()
        logical_num_tokens, hidden_dim = hidden.shape

        if routing_map is not None:
            assert routing_map.dtype == torch.bool
            num_of_experts = routing_map.size(-1)
        else:
            # Generate the routing map and the probs according to the topk_idx and topk_weights.
            assert (
                num_of_experts is not None
            ), "The number of experts should be provided on index-based routing."
            if topk_idx is not None:
                routing_map, probs = indices_to_map(
                    topk_idx, topk_weights, logical_num_tokens, num_of_experts
                )

        assert (
            handle is not None or routing_map is not None
        ), "The handle and routing_map should not be both None"
        if handle is None:
            num_of_tokens = self._comm_rows(logical_num_tokens)
            hidden = _pad_comm_rows(hidden, num_of_tokens)
            routing_map = _pad_comm_rows(routing_map, num_of_tokens)
            if probs is not None:
                probs = _pad_comm_rows(probs, num_of_tokens)
            if scaling_factor is not None:
                scaling_factor = _pad_comm_rows(scaling_factor, num_of_tokens)
            config = self.update_template_config(
                hidden_dim=hidden_dim,
                num_of_tokens_per_rank=num_of_tokens,
            )
            handle_impl = self.runtime.metadata_preprocessing(
                config=config,
                routing_map=routing_map,
                num_of_tokens_per_rank=num_of_tokens,
                enable_permute=False,
                non_blocking=False,
            )
        else:
            # Convert legacy tuple to HandleImpl
            handle_impl = hybrid_ep_cpp.HandleImpl()
            (
                handle_impl.sparse_to_dense_map,
                handle_impl.rdma_to_attn_map,
                handle_impl.attn_to_rdma_map,
                handle_impl.num_dispatched_tokens_tensor,
                handle_impl.local_expert_routing_map,
                handle_impl.num_of_tokens_per_rank,
                handle_impl.config,
            ) = handle
            logical_num_tokens = handle.logical_num_tokens
            assert logical_num_tokens == hidden.size(0)
            hidden = _pad_comm_rows(hidden, handle_impl.num_of_tokens_per_rank)
            if probs is not None:
                probs = _pad_comm_rows(probs, handle_impl.num_of_tokens_per_rank)
            if scaling_factor is not None:
                scaling_factor = _pad_comm_rows(scaling_factor, handle_impl.num_of_tokens_per_rank)

        if num_dispatched_tokens is None:
            # Synchronize the stream to make sure the data in the pinned_memory_buffer: num_dispatched_tokens_tensor is ready.
            torch.cuda.current_stream().synchronize()

        dispatched_token, dispatched_probs, dispatched_scaling_factor = (
            self.runtime.dispatch(
                hidden=hidden,
                probs=probs,
                scaling_factor=scaling_factor,
                handle=handle_impl,
                with_probs=probs is not None,
            )
        )

        return (
            dispatched_token,
            dispatched_probs,
            dispatched_scaling_factor,
            HybridEPHandle((
                handle_impl.sparse_to_dense_map,
                handle_impl.rdma_to_attn_map,
                handle_impl.attn_to_rdma_map,
                handle_impl.num_dispatched_tokens_tensor,
                handle_impl.local_expert_routing_map,
                handle_impl.num_of_tokens_per_rank,
                handle_impl.config,
            ), logical_num_tokens),
        )

    def combine(
        self, hidden: torch.Tensor, probs: torch.Tensor = None, handle: tuple = None
    ):
        """
        Combine the data from the experts.
        Do not require preprocessing, but the handle is necessary.
        """
        hidden = hidden.contiguous()
        if probs is not None:
            probs = probs.contiguous()
        assert handle is not None, "The handle is necessary for combine."
        handle_impl = hybrid_ep_cpp.HandleImpl()
        (
            handle_impl.sparse_to_dense_map,
            handle_impl.rdma_to_attn_map,
            handle_impl.attn_to_rdma_map,
            handle_impl.num_dispatched_tokens_tensor,
            handle_impl.local_expert_routing_map,
            handle_impl.num_of_tokens_per_rank,
            handle_impl.config,
        ) = handle
        logical_num_tokens = handle.logical_num_tokens

        combined_token, combined_probs = self.runtime.combine(
            hidden=hidden,
            probs=probs,
            handle=handle_impl,
            with_probs=probs is not None,
        )
        return (
            combined_token[:logical_num_tokens],
            None if combined_probs is None else combined_probs[:logical_num_tokens],
        )

    def dispatch_with_permute(
        self,
        *,
        # Input tensors
        hidden: torch.Tensor,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        num_of_experts_per_rank: int = None,
        num_of_experts: int = None,
        use_fp8: bool = None,
        routing_map: torch.Tensor = None,
        probs: torch.Tensor = None,
        scaling_factor: torch.Tensor = None,
        # Used in the sync-free permute
        num_permuted_tokens: int = None,
        # If we use permute kernel, the output tensor will be permuted. the result can be directly used in the gemm.
        pad_multiple: int = None,
        # The handle means the cached info from the first invocation of the dispatch kernel.
        # The dense-layout handle keeps the full metadata interface:
        # 1. sparse_to_dense_map
        # 2. rdma_to_attn_map
        # 3. attn_to_rdma_map
        # 4. num_dispatched_tokens_tensor
        # 5. local_expert_routing_map
        # 6. dense_chunk_layout
        # 7. dense_to_expert_map
        # 8. tokens_per_expert
        # 9. num_of_tokens_per_rank
        # 10. template_config: HybridEpConfigInstance
        # 11. overflow_flag
        handle: tuple = None,
        # If non_blocking is True, no stream synchronization will be used, the metadata outputs are on the GPU.
        # Otherwise, tokens_per_expert is copied through pinned memory so Python can derive num_permuted_tokens.
        non_blocking: bool = False,
        fuse_permute_dispatch: bool = False,
        # Deprecated parameters
        num_dispatched_tokens: int = None,
        use_host_meta: bool = None,
    ):
        """
        Dispatch the data to the experts with permute.
        """
        if num_dispatched_tokens is not None:
            warnings.warn("The num_dispatched_tokens is deprecated, it will be removed in the future.")
        if use_host_meta is not None:
            warnings.warn("The use_host_meta is deprecated, it will be removed in the future.")
            non_blocking = not use_host_meta

        hidden = hidden.contiguous()
        if probs is not None:
            probs = probs.contiguous()
        if scaling_factor is not None:
            scaling_factor = scaling_factor.contiguous()
        with torch.cuda.nvtx.range("hybrid-ep dispatch with permute phase"):
            logical_num_tokens, hidden_dim = hidden.shape
            if routing_map is not None:
                assert routing_map.dtype == torch.bool
                num_of_experts = routing_map.size(-1)
            else:
                # Generate the routing map and the probs according to the topk_idx and topk_weights.
                if topk_idx is not None:
                    assert (
                        num_of_experts is not None
                    ), "The number of experts should be provided on index-based routing."
                    routing_map, probs = indices_to_map(
                        topk_idx, topk_weights, logical_num_tokens, num_of_experts
                    )
            if non_blocking:
                assert num_permuted_tokens is not None and num_permuted_tokens >= 0, \
                    "The num_permuted_tokens is required for non-blocking mode."
                if pad_multiple is not None and pad_multiple > 0:
                    assert num_permuted_tokens % pad_multiple == 0, \
                        f"num_permuted_tokens ({num_permuted_tokens}) must be a multiple of pad_multiple ({pad_multiple}) in non-blocking mode."

            if handle is None:
                num_of_tokens_per_rank = self._comm_rows(logical_num_tokens)
                hidden = _pad_comm_rows(hidden, num_of_tokens_per_rank)
                routing_map = _pad_comm_rows(routing_map, num_of_tokens_per_rank)
                if probs is not None:
                    probs = _pad_comm_rows(probs, num_of_tokens_per_rank)
                if scaling_factor is not None:
                    scaling_factor = _pad_comm_rows(scaling_factor, num_of_tokens_per_rank)
                assert hidden.size(0) == routing_map.size(
                    0
                ), "The hidden and the routing_map should have the same row number."
                config = self.update_template_config(
                    hidden_dim=hidden_dim,
                    num_of_tokens_per_rank=num_of_tokens_per_rank,
                    num_local_experts=num_of_experts_per_rank,
                    pad_multiple=pad_multiple,
                    use_fp8=use_fp8,
                    fuse_permute_dispatch=fuse_permute_dispatch,
                )
                handle_impl = self.runtime.metadata_preprocessing(
                    config=config,
                    routing_map=routing_map,
                    num_of_tokens_per_rank=num_of_tokens_per_rank,
                    num_permuted_tokens=num_permuted_tokens,
                    pad_multiple=pad_multiple,
                    enable_permute=True,
                    fuse_permute_dispatch=fuse_permute_dispatch,
                    non_blocking=non_blocking,
                )
            else:
                handle_impl = hybrid_ep_cpp.HandleImpl()
                (
                    handle_impl.sparse_to_dense_map,
                    handle_impl.rdma_to_attn_map,
                    handle_impl.attn_to_rdma_map,
                    handle_impl.num_dispatched_tokens_tensor,
                    handle_impl.local_expert_routing_map,
                    handle_impl.dense_chunk_layout,
                    handle_impl.dense_to_expert_map,
                    handle_impl.tokens_per_expert,
                    handle_impl.num_of_tokens_per_rank,
                    handle_impl.config,
                    handle_impl.overflow_flag,
                ) = handle
                logical_num_tokens = handle.logical_num_tokens
                handle_impl.num_permuted_tokens = num_permuted_tokens
                assert logical_num_tokens == hidden.size(0)
                hidden = _pad_comm_rows(hidden, handle_impl.num_of_tokens_per_rank)
                if probs is not None:
                    probs = _pad_comm_rows(probs, handle_impl.num_of_tokens_per_rank)
                if scaling_factor is not None:
                    scaling_factor = _pad_comm_rows(scaling_factor, handle_impl.num_of_tokens_per_rank)

            (
                dispatched_token,
                dispatched_probs,
                dispatched_scaling_factor,
            ) = self.runtime.dispatch_with_permute(
                hidden=hidden,
                probs=probs,
                scaling_factor=scaling_factor,
                handle=handle_impl,
                pad_multiple=pad_multiple,
                fuse_permute_dispatch=fuse_permute_dispatch,
                non_blocking=non_blocking,
                with_probs=probs is not None,
            )
        
        return (
            dispatched_token,
            dispatched_probs,
            dispatched_scaling_factor,
            handle_impl.padded_tokens_per_expert,
            HybridEPHandle((
                handle_impl.sparse_to_dense_map,
                handle_impl.rdma_to_attn_map,
                handle_impl.attn_to_rdma_map,
                handle_impl.num_dispatched_tokens_tensor,
                handle_impl.local_expert_routing_map,
                handle_impl.dense_chunk_layout,
                handle_impl.dense_to_expert_map,
                handle_impl.tokens_per_expert,
                handle_impl.num_of_tokens_per_rank,
                handle_impl.config,
                handle_impl.overflow_flag,
            ), logical_num_tokens),
        )

    def combine_with_unpermute(
        self,
        *,
        # Input tensors
        hidden: torch.Tensor,
        probs: torch.Tensor = None,
        handle: tuple = None,
        pad_multiple: int = None,
        fuse_unpermute_combine: bool = False,
        # Deprecated parameters
        num_dispatched_tokens: int = None,
    ):
        """
        Combine the data from the experts with unpermute.
        Do not require the routing_map, but the handle is necessary.
        """
        if num_dispatched_tokens is not None:
            warnings.warn("The num_dispatched_tokens is deprecated, it will be removed in the future.")

        hidden = hidden.contiguous()
        if probs is not None:
            probs = probs.contiguous()
        with torch.cuda.nvtx.range("hybrid-ep combine with unpermute phase"):
            assert self.configurer is not None, "Please initialize the configurer first."
            assert handle is not None, "The handle is necessary in the combine pass."

            handle_impl = hybrid_ep_cpp.HandleImpl()
            (
                handle_impl.sparse_to_dense_map,
                handle_impl.rdma_to_attn_map,
                handle_impl.attn_to_rdma_map,
                handle_impl.num_dispatched_tokens_tensor,
                handle_impl.local_expert_routing_map,
                handle_impl.dense_chunk_layout,
                handle_impl.dense_to_expert_map,
                handle_impl.tokens_per_expert,
                handle_impl.num_of_tokens_per_rank,
                handle_impl.config,
                handle_impl.overflow_flag,
            ) = handle
            logical_num_tokens = handle.logical_num_tokens
            combined_token, combined_probs = self.runtime.combine_with_unpermute(
                hidden=hidden,
                probs=probs,
                handle=handle_impl,
                pad_multiple=pad_multiple,
                fuse_unpermute_combine=fuse_unpermute_combine,
                with_probs=probs is not None,
            )
        return (
            combined_token[:logical_num_tokens],
            None if combined_probs is None else combined_probs[:logical_num_tokens],
        )
