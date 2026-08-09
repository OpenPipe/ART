"""Correctness patches for vLLM's fused MoE LoRA kernels."""


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
