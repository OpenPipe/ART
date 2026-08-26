"""Correctness patches for vLLM's fused MoE LoRA kernels."""

from typing import Any

import torch


def patch_local_3d_moe_dummy_lora() -> None:
    """Reshape EP-local warmup adapters by their local expert count."""
    from vllm.lora.model_manager import LoRAModelManager

    original_create = LoRAModelManager.create_dummy_lora
    if getattr(original_create, "__art_local_3d_moe_dummy_patched__", False):
        return
    original_stack = LoRAModelManager._stack_moe_lora_weights

    def create_dummy_lora(self: Any, *args: Any, **kwargs: Any) -> Any:
        lora_model = original_create(self, *args, **kwargs)
        lora_model._art_local_3d_moe_lora = True
        return lora_model

    def stack_moe_lora_weights(
        self: Any, lora_model: Any, module: Any, module_name: str
    ) -> Any:
        down = self._get_lora_layer_weights(lora_model, module_name)
        gate_up = self._get_lora_layer_weights(lora_model, module_name + ".base_layer")
        local_experts = module.w13_lora_a_stacked[0].shape[1]
        if (
            not getattr(lora_model, "_art_local_3d_moe_lora", False)
            or local_experts == module.global_num_experts
            or down is None
            or gate_up is None
            or not torch.is_tensor(down.lora_a)
        ):
            return original_stack(self, lora_model, module, module_name)
        gate_up.lora_a = gate_up.lora_a.reshape(
            local_experts, -1, gate_up.lora_a.shape[-1]
        )
        down.lora_a = down.lora_a.reshape(local_experts, -1, down.lora_a.shape[-1])
        gate_up.lora_b = (
            gate_up.lora_b.reshape(gate_up.lora_b.shape[0], -1, local_experts)
            .permute(2, 0, 1)
            .contiguous()
        )
        down.lora_b = (
            down.lora_b.reshape(down.lora_b.shape[0], -1, local_experts)
            .permute(2, 0, 1)
            .contiguous()
        )
        down.lora_a = [gate_up.lora_a.contiguous(), down.lora_a.contiguous()]
        down.lora_b = [gate_up.lora_b, down.lora_b]
        return original_stack(self, lora_model, module, module_name)

    create_dummy_lora.__art_local_3d_moe_dummy_patched__ = True  # type: ignore[attr-defined]
    LoRAModelManager.create_dummy_lora = create_dummy_lora  # type: ignore[method-assign]
    LoRAModelManager._stack_moe_lora_weights = stack_moe_lora_weights  # type: ignore[method-assign]


def patch_small_batch_moe_lora_intermediate_dtype() -> None:
    from vllm.lora.ops.triton_ops import fused_moe_lora_op

    kernel = fused_moe_lora_op._fused_moe_lora_small_batch_kernel.fn
    source = kernel.src
    cast = "            rank_vec = rank_vec.to(out_ptr.dtype.element_ty)\n"
    if cast in source:
        return
    anchor = (
        "            # EXPAND: walk n_tiles_per_program consecutive output-N tiles\n"
    )
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported vLLM small-batch MoE LoRA kernel source")
    kernel._unsafe_update_src(source.replace(anchor, f"{cast}\n{anchor}"))
