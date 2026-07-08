"""DSV4-specific monkey patches for the ART-owned vLLM runtime."""

import importlib
from typing import Any


def apply_dsv4_vllm_runtime_patches() -> None:
    patch_layerwise_reload_shadow_attrs()
    patch_dsv4_attn_sink_layerwise_reload()
    patch_dsv4_mhc_pre_fixed_split()
    patch_dsv4_lora_support()
    patch_dsv4_mla_lora_aliases()
    patch_dsv4_fast_path_lora()
    patch_dsv4_triton_moe_topk6_routing()
    patch_lora_linear_base_attr_proxy()
    patch_marlin_lora_swiglu_limit()


def _drop_reload_shadow_attrs(layer: Any, names: Any) -> None:
    for name in names:
        if (
            name in getattr(layer, "__dict__", {})
            and name not in layer._parameters
            and name not in layer._buffers
            and name not in layer._modules
        ):
            delattr(layer, name)


def patch_layerwise_reload_shadow_attrs() -> None:
    """Allow vLLM layerwise reload to restore processed DSV4 MegaMoE params.

    DeepSeek V4 MegaMoE drops loader-side Parameters after transforming them for
    DeepGEMM. Some vLLM builds leave same-name plain attributes behind; PyTorch
    then rejects register_parameter during the next checkpoint-format reload.
    """
    from vllm.model_executor.model_loader.reload import layerwise, meta

    if getattr(meta, "_art_reload_shadow_attrs_patched", False):
        return

    original_restore_layer_on_meta = meta.restore_layer_on_meta
    original_place_kernel_tensors = layerwise._place_kernel_tensors

    def restore_layer_on_meta(layer: Any, info: Any) -> None:
        restore_params, restore_buffers = info.restore_metadata
        _drop_reload_shadow_attrs(layer, tuple(restore_params) + tuple(restore_buffers))
        return original_restore_layer_on_meta(layer, info)

    def _place_kernel_tensors(layer: Any, info: Any) -> None:
        assert info.kernel_tensors is not None
        parameters, buffers = info.kernel_tensors
        _drop_reload_shadow_attrs(layer, tuple(parameters) + tuple(buffers))
        return original_place_kernel_tensors(layer, info)

    restore_layer_on_meta.__art_patched__ = True  # type: ignore[attr-defined]
    _place_kernel_tensors.__art_patched__ = True  # type: ignore[attr-defined]
    meta.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise._place_kernel_tensors = _place_kernel_tensors  # type: ignore[method-assign]
    setattr(meta, "_art_reload_shadow_attrs_patched", True)


def _import_dsv4_model_module() -> Any | None:
    for module_name in (
        "vllm.model_executor.models.deepseek_v4",
        "vllm.models.deepseek_v4.nvidia.model",
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def patch_dsv4_attn_sink_layerwise_reload() -> None:
    """Route DSV4 attention-sink loads through vLLM's layerwise loader.

    Merged-weight transfer uses vLLM checkpoint-format reload. During that path,
    every loadable parameter must be applied through its `weight_loader`; direct
    `copy_` into `attn_sink` bypasses layerwise accounting and finalize restores
    the old kernel tensor. With `load_format=dummy`, that old tensor is the
    initialized sink, not the checkpoint sink.
    """
    dsv4_model = _import_dsv4_model_module()
    if dsv4_model is None:
        return
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    model_cls = getattr(dsv4_model, "DeepseekV4Model", None)
    if model_cls is None:
        return
    original = model_cls.load_weights
    if getattr(original, "__art_patched__", False):
        return

    def load_weights(self: Any, weights: Any) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = dsv4_model.get_tensor_model_parallel_world_size()
        tp_rank = dsv4_model.get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == dsv4_model.torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(dsv4_model.torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        success = param.weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                if "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    padded_weight = loaded_weight.new_full(
                        tuple(param.shape), -float("inf")
                    )
                    padded_weight[: narrow_weight.shape[0]].copy_(narrow_weight)
                    weight_loader = getattr(
                        param, "weight_loader", dsv4_model.default_weight_loader
                    )
                    weight_loader(param, padded_weight)
                    loaded_params.add(name)
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", dsv4_model.default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    load_weights.__art_patched__ = True  # type: ignore[attr-defined]
    model_cls.load_weights = load_weights  # type: ignore[method-assign]


def patch_dsv4_mhc_pre_fixed_split() -> None:
    """Make DSV4 mHC pre reductions invariant to total prefill length.

    vLLM's current DSV4 mHC pre path chooses split-K from ``num_tokens``. The
    same prompt prefix can therefore use a different reduction tree when a
    suffix is appended, producing float32-level post/comb mix drift that becomes
    bf16 output drift and later MoE route changes. Pinning the DSV4 prenorm
    shape to the TileLang kernel default split count keeps the reduction plan
    stable without changing model math.
    """
    try:
        mhc = importlib.import_module("vllm.model_executor.layers.mhc")
    except ImportError:
        return
    original = getattr(mhc, "compute_num_split", None)
    if original is None or getattr(original, "__art_dsv4_fixed_split_patched__", False):
        return

    def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
        if block_k == 64 and k == 16_384:
            return 16
        return original(block_k, k, grid_size)

    compute_num_split.__art_dsv4_fixed_split_patched__ = True  # type: ignore[attr-defined]
    mhc.compute_num_split = compute_num_split


def patch_dsv4_lora_support() -> None:
    """Enable vLLM's existing LoRA manager for ART-served DSV4.

    DSV4 itself does not need a custom LoRA executor here. Once the model
    advertises packed MLA/shared-expert modules and MoE expert children, vLLM
    wraps the same FusedMoE module it already uses for serving. With LoRA
    enabled, vLLM's modular MoE selector picks Marlin, whose expert backend
    supports fused MoE LoRA. Do not point this patch at the FlashInfer TRTLLM
    MXFP4 backend; that backend currently has no LoRA hooks.
    """
    dsv4_model = _import_dsv4_model_module()
    if dsv4_model is None:
        return
    model_cls = getattr(dsv4_model, "DeepseekV4ForCausalLM", None)
    if model_cls is None or getattr(model_cls, "_art_dsv4_lora_patched", False):
        return
    model_cls.supports_lora = True
    model_cls.embedding_modules = {}
    model_cls.packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls.is_3d_moe_weight = False
    model_cls.is_non_gated_moe = False
    model_cls.lora_skip_prefixes = ["mtp", "indexer"]
    model_cls._art_dsv4_lora_patched = True
    _patch_dsv4_lora_manager_indexer_skip(model_cls)


def _patch_dsv4_lora_manager_indexer_skip(model_cls: type) -> None:
    from vllm.lora.model_manager import LoRAModelManager

    original = LoRAModelManager._match_target_modules
    if getattr(original, "__art_dsv4_indexer_skip_patched__", False):
        return

    duplicate_mla_aliases = {"fused_wqa_wkv", "wq_b", "wo_a", "wo_b"}

    def _match_target_modules(self: Any, module_name: str) -> bool:
        if isinstance(self.model, model_cls) and ".indexer." in module_name:
            return False
        if (
            isinstance(self.model, model_cls)
            and ".attn.mla_attn." in module_name
            and module_name.rsplit(".", 1)[-1] in duplicate_mla_aliases
        ):
            return False
        return original(self, module_name)

    _match_target_modules.__art_dsv4_indexer_skip_patched__ = True  # type: ignore[attr-defined]
    LoRAModelManager._match_target_modules = _match_target_modules  # type: ignore[method-assign]


def patch_dsv4_mla_lora_aliases() -> None:
    """Keep DSV4 MLA wrapper references aligned with vLLM LoRA replacements.

    DeepSeek V4 stores direct references to several attention linears inside
    ``mla_attn`` during model construction. vLLM's LoRA manager later replaces
    the canonical modules on ``attn``. Without refreshing the aliases, the DSV4
    custom attention path keeps calling the original unwrapped linears.
    """
    from vllm.lora.model_manager import LoRAModelManager

    original = LoRAModelManager.register_module
    if getattr(original, "__art_dsv4_mla_alias_patched__", False):
        return

    alias_names = {"fused_wqa_wkv", "wq_b", "wo_a", "wo_b"}

    def register_module(self: Any, module_name: str, module: Any) -> Any:
        result = original(self, module_name, module)
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in alias_names:
            return result
        parent_name, _, _ = module_name.rpartition(".")
        if parent_name.endswith(".mla_attn"):
            mla_name = parent_name
        else:
            mla_name = f"{parent_name}.mla_attn"
        try:
            mla_attn = self.model.get_submodule(mla_name)
        except AttributeError:
            return result
        if mla_attn is not None and hasattr(mla_attn, leaf):
            setattr(mla_attn, leaf, module)
        return result

    register_module.__art_dsv4_mla_alias_patched__ = True  # type: ignore[attr-defined]
    LoRAModelManager.register_module = register_module  # type: ignore[method-assign]


def _is_lora_wrapped_linear(module: Any) -> bool:
    return all(
        hasattr(module, name)
        for name in ("lora_a_stacked", "lora_b_stacked", "punica_wrapper")
    )


def _apply_lora_to_existing_linear_output(
    module: Any,
    x: Any,
    output: Any,
) -> Any:
    if not _is_lora_wrapped_linear(module):
        return output
    wrapper = module.punica_wrapper
    if getattr(wrapper, "no_lora", False):
        return output
    if getattr(wrapper, "indices_len", [None])[0] is None:
        return output
    return module._apply_lora_to_output(x, output)


def _register_dsv4_lora_expand_fp32_output_op() -> None:
    if getattr(_register_dsv4_lora_expand_fp32_output_op, "_registered", False):
        return

    import torch
    from vllm import envs
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs, supports_pdl
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op

    @triton.jit
    def _kernel(
        input_ptr,
        lora_b_ptr,
        output_ptr,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        M,
        slice_offset,
        input_stride_m,
        input_stride_k,
        lora_stride_lora,
        lora_stride_out,
        lora_stride_k,
        output_stride_m,
        output_stride_n,
        RANK: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_GDC: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        cta_n_num = tl.cdiv(OUT_WIDTH, BLOCK_N)
        cta_m_num = tl.cdiv(M, BLOCK_M)
        pid_mn = tl.program_id(0)
        pid_m = pid_mn % cta_m_num
        pid_n = (pid_mn // cta_m_num) % cta_n_num
        lora_idx = tl.program_id(1)

        lora_id = tl.load(lora_ids + lora_idx)
        if lora_id == -1:
            return

        lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
        cta_m_offset = pid_m * BLOCK_M
        if cta_m_offset >= lora_m_size:
            return

        cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
        lora_m_start = tl.load(lora_token_start_loc + lora_idx)
        row_ptr = token_indices_sorted_by_lora_ids + lora_m_start + cta_m_offset
        row_offsets = tl.arange(0, BLOCK_M) % cta_m_len
        rows = tl.load(row_ptr + row_offsets)

        offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, RANK, BLOCK_K):
            offs_k = tl.arange(0, BLOCK_K) + k_start
            x = tl.load(
                input_ptr
                + rows[:, None] * input_stride_m
                + offs_k[None, :] * input_stride_k,
                mask=(row_offsets[:, None] < cta_m_len) & (offs_k[None, :] < RANK),
                other=0.0,
            )
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            b = tl.load(
                lora_b_ptr
                + lora_id * lora_stride_lora
                + offs_n[None, :] * lora_stride_out
                + offs_k[:, None] * lora_stride_k,
                mask=(offs_k[:, None] < RANK) & (offs_n[None, :] < OUT_WIDTH),
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(x, b)

        out_cols = slice_offset + offs_n
        out_ptrs = (
            output_ptr
            + rows[:, None] * output_stride_m
            + out_cols[None, :] * output_stride_n
        )
        mask = (row_offsets[:, None] < cta_m_len) & (offs_n[None, :] < OUT_WIDTH)
        old = tl.load(out_ptrs, mask=mask)
        tl.store(out_ptrs, old + accumulator, mask=mask)

    @torch.inference_mode()
    def _impl(
        inputs: torch.Tensor,
        lora_b_weight: torch.Tensor,
        output_tensor: torch.Tensor,
        token_indices_sorted_by_lora_ids: torch.Tensor,
        num_tokens_per_lora: torch.Tensor,
        lora_token_start_loc: torch.Tensor,
        lora_ids: torch.Tensor,
        no_lora_flag_cpu: torch.Tensor,
        num_active_loras: torch.Tensor,
        slice_offset: int,
    ) -> None:
        assert no_lora_flag_cpu.numel() == 1
        if no_lora_flag_cpu.item():
            return
        assert inputs.dtype == torch.float32
        assert output_tensor.dtype == torch.float32
        assert output_tensor.is_contiguous()
        assert lora_b_weight.ndim == 4
        assert lora_b_weight.size(1) == 1
        assert lora_b_weight.is_contiguous()

        m = inputs.size(0)
        out_width = lora_b_weight.size(2)
        rank = lora_b_weight.size(3)
        kernel_config = get_lora_op_configs(
            op_type="expand",
            max_loras=lora_ids.size(0),
            batch=m,
            hidden_size=out_width,
            rank=rank,
            num_slices=1,
            add_inputs=True,
        )
        block_m = kernel_config["block_m"]
        block_n = kernel_config["block_n"]
        block_k = kernel_config["block_k"]
        use_gdc = supports_pdl(inputs.device) and envs.VLLM_LORA_ENABLE_DUAL_STREAM
        grid = (
            triton.cdiv(m, block_m) * triton.cdiv(out_width, block_n),
            num_active_loras.item(),
        )
        _kernel[grid](
            inputs,
            lora_b_weight,
            output_tensor,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            m,
            slice_offset,
            inputs.stride(0),
            inputs.stride(1),
            lora_b_weight.stride(0),
            lora_b_weight.stride(2),
            lora_b_weight.stride(3),
            output_tensor.stride(0),
            output_tensor.stride(1),
            RANK=rank,
            OUT_WIDTH=out_width,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_GDC=use_gdc,
            num_warps=kernel_config["num_warps"],
            num_ctas=kernel_config["num_ctas"],
            num_stages=kernel_config["num_stages"],
            launch_pdl=use_gdc,
        )

    def _fake(
        inputs: torch.Tensor,
        lora_b_weight: torch.Tensor,
        output_tensor: torch.Tensor,
        token_indices_sorted_by_lora_ids: torch.Tensor,
        num_tokens_per_lora: torch.Tensor,
        lora_token_start_loc: torch.Tensor,
        lora_ids: torch.Tensor,
        no_lora_flag_cpu: torch.Tensor,
        num_active_loras: torch.Tensor,
        slice_offset: int,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="art_dsv4_lora_expand_fp32_output",
        op_func=_impl,
        mutates_args=["output_tensor"],
        fake_impl=_fake,
    )
    _register_dsv4_lora_expand_fp32_output_op._registered = True  # type: ignore[attr-defined]


def _apply_dsv4_compressor_lora_to_existing_output(
    module: Any,
    x: Any,
    output: Any,
) -> Any:
    if not _is_active_lora_wrapped_linear(module):
        return output

    import torch
    from vllm.platforms import current_platform

    _register_dsv4_lora_expand_fp32_output_op()
    original_shape = output.shape if output.ndim == 3 else None
    if x.ndim == 3 and output.ndim == 3:
        x = x.flatten(0, 1)
        output = output.flatten(0, 1)
    if x.ndim != 2 or output.ndim != 2:
        raise RuntimeError(
            "DSV4 compressor LoRA expects 2D hidden/output tensors, got "
            f"x={tuple(x.shape)}, output={tuple(output.shape)}."
        )
    if output.dtype != torch.float32:
        raise RuntimeError(
            "DSV4 compressor LoRA must preserve the fp32 compressor projection, "
            f"got output dtype {output.dtype}."
        )
    if getattr(module, "tp_size", 1) != 1:
        raise RuntimeError(
            "DSV4 compressor LoRA expects disable_tp compressor linears."
        )

    wrapper = module.punica_wrapper
    local_rank = module.lora_a_stacked[0].shape[2]
    buffers = torch.empty_strided(
        (len(module.output_slices), x.shape[0], local_rank),
        (x.shape[0] * local_rank, local_rank, 1),
        dtype=torch.float32,
        device=x.device,
    )
    shrunk = wrapper.add_shrink(buffers, x, module.lora_a_stacked, 1.0)
    if not current_platform.can_update_inplace():
        buffers = shrunk

    (
        _,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    ) = wrapper.token_mapping_meta.meta_args(
        x.shape[0], wrapper.lora_config.specialize_active_lora
    )
    offset = 0
    for idx, width in enumerate(module.output_slices):
        torch.ops.vllm.art_dsv4_lora_expand_fp32_output(
            buffers[idx],
            module.lora_b_stacked[idx],
            output,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            no_lora_flag_cpu,
            num_active_loras,
            offset,
        )
        offset += width
    if offset != output.shape[-1]:
        raise RuntimeError(
            "DSV4 compressor LoRA output slice width mismatch: "
            f"slices={module.output_slices}, output={tuple(output.shape)}."
        )
    return output.reshape(original_shape) if original_shape is not None else output


def _is_active_lora_wrapped_linear(module: Any) -> bool:
    if not _is_lora_wrapped_linear(module):
        return False
    wrapper = module.punica_wrapper
    return not getattr(wrapper, "no_lora", False) and (
        getattr(wrapper, "indices_len", [None])[0] is not None
    )


def _register_dsv4_inv_rope_lora_input_op() -> None:
    if getattr(_register_dsv4_inv_rope_lora_input_op, "_registered", False):
        return

    import torch
    from vllm.platforms import current_platform
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op

    @triton.jit
    def _kernel(
        o_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        fp8_ptr,
        scale_ptr,
        lora_input_ptr,
        num_tokens,
        heads_per_group: tl.constexpr,
        o_stride_token,
        o_stride_head,
        cache_stride_pos,
        fp8_stride_group,
        fp8_stride_token,
        scale_stride_group,
        scale_stride_k,
        lora_stride_group,
        lora_stride_token,
        fp8_max: tl.constexpr,
        eps: tl.constexpr,
        QUANT_GROUP_SIZE: tl.constexpr,
        CHUNKS_PER_HEAD: tl.constexpr,
        ROPE_START: tl.constexpr,
        HALF_ROPE: tl.constexpr,
        TMA_ALIGNED_SCALES: tl.constexpr,
    ):
        pid_token = tl.program_id(0).to(tl.int64)
        pid_gh = tl.program_id(1).to(tl.int64)
        g = pid_gh // heads_per_group
        head_in_group = pid_gh % heads_per_group
        qb_start = head_in_group * CHUNKS_PER_HEAD

        if pid_token >= num_tokens:
            if TMA_ALIGNED_SCALES:
                scale_addr = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + head_in_group * scale_stride_k
                )
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
            else:
                block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
                qb_indices = qb_start + block_offsets
                scale_addrs = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + qb_indices * scale_stride_k
                )
                tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
            return

        head_dim: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
        offsets = tl.arange(0, head_dim)
        input_base = o_ptr + pid_token * o_stride_token + pid_gh * o_stride_head
        x = tl.load(input_base + offsets).to(tl.float32)

        rope_abs_start: tl.constexpr = (
            CHUNKS_PER_HEAD - 1
        ) * QUANT_GROUP_SIZE + ROPE_START
        pos = tl.load(positions_ptr + pid_token)
        cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
        is_rope = offsets >= rope_abs_start
        rope_local = offsets - rope_abs_start

        x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
            tl.float32
        )
        cs_idx = tl.maximum(rope_local >> 1, 0)
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
        x_add = x * cos_v + x_partner * sin_v
        x_sub = x * cos_v - x_partner * sin_v
        is_even = (rope_local & 1) == 0
        x = tl.where(is_rope, tl.where(is_even, x_add, x_sub), x)

        group_head_offset = head_in_group * head_dim
        lora_base = (
            lora_input_ptr
            + g * lora_stride_group
            + pid_token * lora_stride_token
            + group_head_offset
        )
        tl.store(lora_base + offsets, x)

        x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scale_raw = block_absmax * (1.0 / fp8_max)
        scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))
        scales_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
            ),
            (head_dim,),
        )
        x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

        fp8_base = (
            fp8_ptr
            + g * fp8_stride_group
            + pid_token * fp8_stride_token
            + qb_start * QUANT_GROUP_SIZE
        )
        tl.store(fp8_base + offsets, x_quant)

        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        if TMA_ALIGNED_SCALES:
            scale_bits = scales.to(tl.int32, bitcast=True)
            ue8m0_bytes = (scale_bits >> 23) & 0xFF
            packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, packed_val)
        else:
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, scales)

    def _impl(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        fp8_buf: torch.Tensor,
        scale_buf: torch.Tensor,
        lora_input: torch.Tensor,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        tma_aligned_t: int,
        num_tokens: int,
    ) -> None:
        grid = (tma_aligned_t, fp8_buf.shape[0] * heads_per_group)
        pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
        _kernel[grid](
            o,
            positions,
            cos_sin_cache,
            fp8_buf,
            scale_buf,
            lora_input,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            fp8_stride_group=fp8_buf.stride(0),
            fp8_stride_token=fp8_buf.stride(1),
            scale_stride_group=scale_buf.stride(0),
            scale_stride_k=scale_buf.stride(2),
            lora_stride_group=lora_input.stride(0),
            lora_stride_token=lora_input.stride(1),
            fp8_max=fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=rope_start,
            HALF_ROPE=half_rope,
            TMA_ALIGNED_SCALES=tma_aligned_scales,
            num_stages=1,
            num_warps=1,
            **pdl_kwargs,
        )

    def _fake(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        fp8_buf: torch.Tensor,
        scale_buf: torch.Tensor,
        lora_input: torch.Tensor,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        tma_aligned_t: int,
        num_tokens: int,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="art_dsv4_inv_rope_fp8_quant_lora_input",
        op_func=_impl,
        mutates_args=["fp8_buf", "scale_buf", "lora_input"],
        fake_impl=_fake,
    )
    _register_dsv4_inv_rope_lora_input_op._registered = True  # type: ignore[attr-defined]


def _dsv4_fused_inv_rope_fp8_quant_with_lora_input(
    dsv4_attn: Any,
    o: Any,
    positions: Any,
    cos_sin_cache: Any,
    *,
    n_groups: int,
    heads_per_group: int,
    lora_dtype: Any,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[Any, Any, Any]:
    import torch
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    fp8_dtype = torch.float8_e4m3fn
    tma_aligned_t = get_tma_aligned_size(num_tokens, 4)
    scale_inner = (
        (num_scale_blocks + 3) // 4 if tma_aligned_scales else num_scale_blocks
    )

    fp8_buf = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )
    lora_input = torch.empty(
        (n_groups, num_tokens, d),
        dtype=lora_dtype,
        device=o.device,
    )
    dsv4_attn.torch.ops.vllm.art_dsv4_inv_rope_fp8_quant_lora_input(
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        lora_input,
        heads_per_group,
        quant_group_size,
        chunks_per_head,
        nope_dim % quant_group_size,
        rope_dim // 2,
        tma_aligned_scales,
        torch.finfo(fp8_dtype).max,
        tma_aligned_t,
        num_tokens,
    )
    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1), lora_input


def _dsv4_wo_a_lora_b_group_cache(
    wo_a: Any,
    *,
    n_local_groups: int,
    out_per_group: int,
) -> tuple[Any, ...]:
    source = wo_a.lora_b_stacked[0]
    key = (
        source.data_ptr(),
        getattr(source, "_version", None),
        n_local_groups,
        out_per_group,
        source.shape[-1],
    )
    if getattr(wo_a, "_art_wo_a_lora_b_group_cache_key", None) != key:
        wo_a._art_wo_a_lora_b_group_cache = tuple(
            source[
                :, :, group * out_per_group : (group + 1) * out_per_group, :
            ].contiguous()
            for group in range(n_local_groups)
        )
        wo_a._art_wo_a_lora_b_group_cache_key = key
    return wo_a._art_wo_a_lora_b_group_cache


def _apply_dsv4_wo_a_lora_fast(
    wo_a: Any,
    z: Any,
    *,
    lora_input: Any,
    n_local_groups: int,
) -> Any:
    if not _is_active_lora_wrapped_linear(wo_a):
        return z

    import torch
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.platforms import current_platform

    wrapper = wo_a.punica_wrapper
    out_per_group = z.shape[-1]
    z_flat = z.view(z.shape[0], n_local_groups * out_per_group)
    group_b = _dsv4_wo_a_lora_b_group_cache(
        wo_a,
        n_local_groups=n_local_groups,
        out_per_group=out_per_group,
    )
    local_rank = wo_a.lora_a_stacked[0].shape[2]
    buffers = torch.empty_strided(
        (n_local_groups, z.shape[0], local_rank),
        (z.shape[0] * local_rank, local_rank, 1),
        dtype=torch.float32,
        device=z.device,
    )
    for group, lora_b in enumerate(group_b):
        buffer = buffers[group : group + 1]
        buffer.zero_()
        shrunk = wrapper.add_shrink(buffer, lora_input[group], wo_a.lora_a_stacked, 1.0)
        if not current_platform.can_update_inplace():
            buffer = shrunk
        buffer = tensor_model_parallel_all_gather(buffer)
        expanded = wrapper.add_expand(
            z_flat,
            buffer,
            (lora_b,),
            (out_per_group,),
            offset_start=group * out_per_group,
            add_inputs=True,
        )
        if not current_platform.can_update_inplace():
            z_flat = expanded
            z = z_flat.view_as(z)
    return z


def _patch_dsv4_compressor_fast_path_lora(attention_cls: Any) -> None:
    if getattr(attention_cls, "_art_compressor_fast_path_lora_patched", False):
        return

    original_attn_gemm_parallel_execute = attention_cls.attn_gemm_parallel_execute

    def attn_gemm_parallel_execute(self: Any, hidden_states: Any) -> tuple[Any, ...]:
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            original_attn_gemm_parallel_execute(self, hidden_states)
        )
        if self.compressor is not None:
            kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.compressor.fused_wkv_wgate,
                hidden_states,
                kv_score,
            )
        if self.indexer is not None:
            indexer_kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.indexer.compressor.fused_wkv_wgate,
                hidden_states,
                indexer_kv_score,
            )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    attn_gemm_parallel_execute.__art_patched__ = True  # type: ignore[attr-defined]
    attention_cls.attn_gemm_parallel_execute = attn_gemm_parallel_execute
    attention_cls._art_compressor_fast_path_lora_patched = True


def _dsv4_deep_gemm_fp8_o_proj_with_lora(
    o_proj_mod: Any,
    o: Any,
    positions: Any,
    cos_sin_cache: Any,
    wo_a: Any,
    wo_b: Any,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> Any:
    wo_a_lora_input = None
    if _is_active_lora_wrapped_linear(wo_a):
        o_fp8, o_scale, wo_a_lora_input = (
            _dsv4_fused_inv_rope_fp8_quant_with_lora_input(
                o_proj_mod,
                o,
                positions,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                lora_dtype=wo_a.lora_a_stacked[0].dtype,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                tma_aligned_scales=tma_aligned_scales,
            )
        )
    else:
        o_fp8, o_scale = o_proj_mod.fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            tma_aligned_scales=tma_aligned_scales,
        )

    z = o_proj_mod.torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=o_proj_mod.torch.bfloat16,
    )
    o_proj_mod.fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
    if wo_a_lora_input is not None:
        z = _apply_dsv4_wo_a_lora_fast(
            wo_a,
            z,
            lora_input=wo_a_lora_input,
            n_local_groups=n_groups,
        )
    return wo_b(z.flatten(1))


def _patch_dsv4_cuda_o_proj_lora(attn_cls: Any, o_proj_mod: Any) -> None:
    if getattr(attn_cls, "_art_wo_a_fast_path_lora_patched", False):
        return

    def _o_proj(self: Any, o: Any, positions: Any) -> Any:
        return _dsv4_deep_gemm_fp8_o_proj_with_lora(
            o_proj_mod,
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    _o_proj.__art_patched__ = True  # type: ignore[attr-defined]
    attn_cls._o_proj = _o_proj
    attn_cls._art_wo_a_fast_path_lora_patched = True


def _patch_current_dsv4_fast_path_lora() -> bool:
    try:
        dsv4_attention = importlib.import_module("vllm.models.deepseek_v4.attention")
    except ModuleNotFoundError:
        return False

    attention_cls = getattr(dsv4_attention, "DeepseekV4Attention", None)
    if attention_cls is None:
        return False

    _patch_dsv4_compressor_fast_path_lora(attention_cls)

    try:
        o_proj_mod = importlib.import_module(
            "vllm.models.deepseek_v4.nvidia.ops.o_proj"
        )
    except ModuleNotFoundError:
        return True

    for module_name, class_name in (
        (
            "vllm.models.deepseek_v4.nvidia.flashmla",
            "DeepseekV4FlashMLAAttention",
        ),
        (
            "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
            "DeepseekV4FlashInferMLAAttention",
        ),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        attn_cls = getattr(module, class_name, None)
        if attn_cls is not None:
            _patch_dsv4_cuda_o_proj_lora(attn_cls, o_proj_mod)
    return True


def patch_dsv4_fast_path_lora() -> None:
    """Apply LoRA deltas on DSV4 paths that read base weights directly.

    vLLM's generic LoRA manager can wrap DSV4 linear modules, but the DSV4
    Flash runtime bypasses some wrapped forwards for performance: compressor
    projections are direct ``hidden @ fused_wkv_wgate.weight.T`` calls, and
    ``wo_a`` is a custom inverse-RoPE/FP8/einsum path. Without this patch vLLM
    accepts and activates these adapter tensors while silently omitting their
    deltas from generation.
    """
    _register_dsv4_inv_rope_lora_input_op()
    _register_dsv4_lora_expand_fp32_output_op()
    if _patch_current_dsv4_fast_path_lora():
        return

    dsv4_attn = importlib.import_module(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    wrapper_cls = getattr(dsv4_attn, "DeepseekV4MultiHeadLatentAttentionWrapper", None)
    if wrapper_cls is None:
        return
    if getattr(wrapper_cls, "_art_fast_path_lora_patched", False):
        return

    original_attn_gemm_parallel_execute = wrapper_cls.attn_gemm_parallel_execute
    original_forward = wrapper_cls.forward

    def attn_gemm_parallel_execute(self: Any, hidden_states: Any) -> tuple[Any, ...]:
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            original_attn_gemm_parallel_execute(self, hidden_states)
        )
        if self.compressor is not None:
            kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.compressor.fused_wkv_wgate,
                hidden_states,
                kv_score,
            )
        if self.indexer is not None:
            indexer_kv_score = _apply_dsv4_compressor_lora_to_existing_output(
                self.indexer.compressor.fused_wkv_wgate,
                hidden_states,
                indexer_kv_score,
            )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def forward(
        self: Any,
        positions: Any,
        hidden_states: Any,
        llama_4_scaling: Any | None = None,
    ) -> Any:
        if dsv4_attn.current_platform.is_rocm():
            return original_forward(self, positions, hidden_states, llama_4_scaling)

        num_tokens = hidden_states.shape[0]
        o_padded = dsv4_attn.torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        dsv4_attn.torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        wo_a_lora_input = None
        if _is_active_lora_wrapped_linear(self.wo_a):
            o_fp8, o_scale, wo_a_lora_input = (
                _dsv4_fused_inv_rope_fp8_quant_with_lora_input(
                    dsv4_attn,
                    o,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    n_groups=self.n_local_groups,
                    heads_per_group=self.n_local_heads // self.n_local_groups,
                    lora_dtype=self.wo_a.lora_a_stacked[0].dtype,
                    nope_dim=self.nope_head_dim,
                    rope_dim=self.rope_head_dim,
                    tma_aligned_scales=self._tma_aligned_scales,
                )
            )
        else:
            o_fp8, o_scale = dsv4_attn.fused_inv_rope_fp8_quant(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                tma_aligned_scales=self._tma_aligned_scales,
            )

        z = dsv4_attn.torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=dsv4_attn.torch.bfloat16,
        )
        dsv4_attn.torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            self.wo_a.weight,
            self.wo_a.weight_scale_inv,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )
        if wo_a_lora_input is not None:
            z = _apply_dsv4_wo_a_lora_fast(
                self.wo_a,
                z,
                lora_input=wo_a_lora_input,
                n_local_groups=self.n_local_groups,
            )
        return self.wo_b(z.flatten(1))

    attn_gemm_parallel_execute.__art_patched__ = True  # type: ignore[attr-defined]
    forward.__art_patched__ = True  # type: ignore[attr-defined]
    wrapper_cls.attn_gemm_parallel_execute = attn_gemm_parallel_execute
    wrapper_cls.forward = forward
    wrapper_cls._art_fast_path_lora_patched = True


def _next_power_of_two(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def patch_dsv4_triton_moe_topk6_routing() -> None:
    """Make vLLM's DSV4 Triton MoE routing sort compile for top-k 6.

    Triton's ``tl.arange`` requires a power-of-two range. Current vLLM's
    DSV4/MXFP4 routing path sorts ``BLOCK_M * num_experts_per_tok`` entries;
    DSV4 uses top-k 6, so the first profile run tries ``tl.arange(0, 192)`` and
    the engine exits before serving starts. Keep the original indexing stride at
    192, but sort over a padded power-of-two vector and mask padded lanes.
    """
    try:
        import torch
        import triton
        import triton.language as tl
        from vllm.third_party.triton_kernels.routing_details._expt_data import (
            _expt_data_compute,
        )
    except ImportError:
        return

    @triton.jit
    def _routing_compute_indx_pow2(
        pid_m,
        GatherIndx,
        ScatterIndx,
        GateScal,
        ExptScal,
        ExptIndx,
        PartialOffs,
        stride_pm,
        stride_pn,
        TokensStart,
        n_tokens,
        BLOCK_M: tl.constexpr,
        N_EXPTS_ACT: tl.constexpr,
        BLOCK_SIZE_PADDED: tl.constexpr,
    ):
        if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
            n_tokens = tl.load(n_tokens)
        n_gates = n_tokens * N_EXPTS_ACT
        block_size: tl.constexpr = N_EXPTS_ACT * BLOCK_M
        tl.static_assert(BLOCK_SIZE_PADDED >= block_size)
        tl.static_assert(BLOCK_SIZE_PADDED <= 32768)

        local_offs = tl.arange(0, BLOCK_SIZE_PADDED)
        valid_local = local_offs < block_size
        offs = pid_m * block_size + local_offs
        expert = tl.load(
            ExptIndx + offs,
            mask=valid_local & (offs < n_gates),
            other=-1,
        ).to(tl.uint32)

        kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
        kv_pairs = tl.sort(kv_pairs, 0)
        expert = kv_pairs >> 16
        offs = pid_m * block_size + (kv_pairs & 0xFFFF)
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add_pow2)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
        gates += tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)

    @triton.jit
    def _keyed_add_pow2(x, y):
        key_mask: tl.constexpr = 0xFFFF0000
        kx = x & key_mask
        ky = y & key_mask
        return tl.where(kx == ky, x + y - kx, y)

    @triton.jit
    def _combined_routing_compute_pow2(
        GatherIndx,
        ScatterIndx,
        GateScal,
        ExptScal,
        ExptIndx,
        PartialOffs,
        stride_pm,
        stride_pn,
        TokensStart,
        n_tokens,
        BLOCK_M: tl.constexpr,
        N_EXPTS_ACT: tl.constexpr,
        Hist,
        MDTileStarts,
        tile_starts_stridem,
        MDTileInfo,
        tile_info_stridem,
        first_tile_dim_log2,
        SIZES: tl.constexpr,
        BLOCK: tl.constexpr,
        blocks2a,
        BLOCK_SIZE_PADDED: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < blocks2a:
            _expt_data_compute(
                Hist,
                MDTileStarts,
                tile_starts_stridem,
                MDTileInfo,
                tile_info_stridem,
                first_tile_dim_log2,
                SIZES,
                BLOCK,
            )
        else:
            pid -= blocks2a
            _routing_compute_indx_pow2(
                pid,
                GatherIndx,
                ScatterIndx,
                GateScal,
                ExptScal,
                ExptIndx,
                PartialOffs,
                stride_pm,
                stride_pn,
                TokensStart,
                n_tokens,
                BLOCK_M,
                N_EXPTS_ACT,
                BLOCK_SIZE_PADDED,
            )

    for module_name in (
        "vllm.third_party.triton_kernels.routing",
        "triton_kernels.routing",
    ):
        try:
            routing = importlib.import_module(module_name)
        except ImportError:
            continue
        original_forward = routing.SortTokens.forward
        if getattr(original_forward, "__art_dsv4_topk6_pow2_patched__", False):
            continue

        def forward(
            ctx: Any,
            expt_scal: Any,
            expt_indx: Any,
            n_expts_tot: int,
            bitmatrix: Any,
            _routing: Any = routing,
        ) -> Any:
            hist_block_m = 32
            indx_offs_block_m = 512
            memset_block = 1024
            cdiv = triton.cdiv

            device = expt_scal.device
            dtype = expt_scal.dtype
            n_tokens_raw, _ = bitmatrix.shape
            n_tokens_pad, n_expts_act = expt_scal.shape
            n_gates_pad = n_tokens_pad * n_expts_act

            hist, partial_hist = bitmatrix.sum(partials_block_size=hist_block_m)
            hist = hist[:n_expts_tot]
            assert hist.dtype == torch.int32
            expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
            combined_indx = torch.empty(
                n_gates_pad * 2, dtype=torch.int32, device=device
            )
            topk_indx = combined_indx[:n_gates_pad]
            gate_indx = combined_indx[n_gates_pad:]
            gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)

            (
                token_offs_combined,
                token_offs_raw,
                token_offs_pad,
                block_pid_map,
                blocks1a,
                blocks2a,
                memset_block_a,
                hist2_block_m,
                block_m_log2_start,
                block_m_num,
            ) = _routing._compute_expt_data_internal(hist, n_expts_tot, n_gates_pad)

            blocks1b = cdiv(n_gates_pad * 2, memset_block) + n_expts_tot + 1
            blocks2b = cdiv(n_tokens_pad, hist_block_m)

            _routing._combined_routing_memset[(blocks1a + blocks1b,)](
                combined_indx,
                n_gates_pad * 2,
                -1,
                memset_block,
                hist,
                expt_offs,
                hist.shape[0],
                n_expts_tot,
                partial_hist,
                partial_hist.shape[0],
                partial_hist.stride(0),
                partial_hist.stride(1),
                token_offs_combined,
                token_offs_combined.stride(0),
                blocks1a,
                block_pid_map,
                block_m_log2_start,
                SIZES=block_m_num,
                BLOCK_A=memset_block_a,
                BLOCK_N=512,
                BLOCK_M=indx_offs_block_m,
            )

            indx_offs = partial_hist
            _combined_routing_compute_pow2[(blocks2a + blocks2b,)](
                topk_indx,
                gate_indx,
                gate_scal,
                expt_scal,
                expt_indx,
                indx_offs,
                indx_offs.stride(0),
                indx_offs.stride(1),
                expt_offs,
                n_tokens_raw,
                hist_block_m,
                n_expts_act,
                hist,
                token_offs_pad,
                token_offs_pad.stride(0),
                block_pid_map,
                block_pid_map.stride(0),
                block_m_log2_start,
                block_m_num,
                hist2_block_m,
                blocks2a,
                BLOCK_SIZE_PADDED=_next_power_of_two(hist_block_m * n_expts_act),
            )

            ctx.n_tokens_raw = n_tokens_raw
            ctx.n_tokens_pad = n_tokens_pad
            ctx.n_expts_act = n_expts_act
            ctx.save_for_backward(gate_indx)
            return (
                hist,
                topk_indx,
                gate_indx,
                gate_scal,
                token_offs_raw,
                token_offs_pad,
                block_pid_map,
            )

        forward.__art_dsv4_topk6_pow2_patched__ = True  # type: ignore[attr-defined]
        forward.__art_original__ = original_forward  # type: ignore[attr-defined]
        routing.SortTokens.forward = staticmethod(forward)
        routing._combined_routing_compute = _combined_routing_compute_pow2


def _base_layer_attr_proxy(name: str) -> property:
    def attr(self: Any) -> Any:
        return getattr(self.base_layer, name)

    return property(attr)


def patch_lora_linear_base_attr_proxy() -> None:
    """Expose DSV4 base metadata through vLLM linear LoRA wrappers.

    DeepSeek V4's output attention path calls a custom FP8 einsum directly and
    reads ``wo_a.weight_scale_inv`` next to ``wo_a.weight``. vLLM's linear LoRA
    wrappers already proxy ``weight`` but not the quant scale. Its router also
    reads dynamic gate metadata from ``self.gate`` after that gate can be LoRA
    wrapped. Keep these tensors owned by the base layer instead of copying or
    re-registering them on every wrapper.
    """
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    if getattr(BaseLinearLayerWithLoRA, "_art_base_attr_proxy_patched", False):
        return

    for name in ("weight_scale_inv", "tid2eid", "e_score_correction_bias"):
        if not hasattr(BaseLinearLayerWithLoRA, name):
            setattr(BaseLinearLayerWithLoRA, name, _base_layer_attr_proxy(name))
    BaseLinearLayerWithLoRA._art_base_attr_proxy_patched = True


def patch_marlin_lora_swiglu_limit() -> None:
    """Keep Marlin MoE LoRA active when DSV4 uses a SwiGLU clamp limit.

    vLLM's Marlin LoRA path injects W13 LoRA inside the activation callback and
    stores that activated cache for W2 LoRA. DSV4 sets ``gemm1_clamp_limit``;
    upstream Marlin bypasses the callback in that case and calls the clamp op
    directly, so W13 LoRA is skipped and W2 LoRA later misses ``cache2``. Route
    the callback through the same clamp op while preserving Marlin execution.
    """
    try:
        marlin_moe = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.fused_marlin_moe"
        )
    except ModuleNotFoundError:
        return

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

    MarlinExperts = marlin_moe.MarlinExperts

    original_apply = MarlinExperts.apply
    if getattr(original_apply, "__art_patched__", False):
        return

    sentinel = object()

    def apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        clamp_limit = getattr(self, "gemm1_clamp_limit", None)
        if getattr(self, "_lora_context", None) is None or clamp_limit is None:
            return original_apply(self, *args, **kwargs)

        original_activation = self.activation
        previous_activation = self.__dict__.get("activation", sentinel)
        previous_clamp_limit = self.gemm1_clamp_limit

        def activation_with_clamp(
            activation: Any,
            output: Any,
            input: Any,
        ) -> None:
            if activation == MoEActivation.SILU:
                swiglu_limit_func(output, input, clamp_limit)
            else:
                original_activation(activation, output, input)

        self.activation = activation_with_clamp
        self.gemm1_clamp_limit = None
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            self.gemm1_clamp_limit = previous_clamp_limit
            if previous_activation is sentinel:
                delattr(self, "activation")
            else:
                self.activation = previous_activation

    apply.__art_patched__ = True  # type: ignore[attr-defined]
    MarlinExperts.apply = apply  # type: ignore[method-assign]
