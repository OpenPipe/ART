"""Qwen3.5 compatibility patches for the ART-owned vLLM runtime."""

from typing import Any


def patch_trtllm_monolithic_route_capture() -> None:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        TrtLlmBf16Experts,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_apply = TrtLlmBf16Experts.apply
    if not getattr(original_apply, "__art_route_capture_patched__", False):

        def apply(
            self: Any,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            router_logits: torch.Tensor,
            activation: Any,
            global_num_experts: int,
            expert_map: torch.Tensor | None,
            a1q_scale: torch.Tensor | None,
            apply_router_weight_on_input: bool,
            num_expert_group: int | None = None,
            e_score_correction_bias: torch.Tensor | None = None,
            routed_scaling_factor: float | None = None,
            topk_group: int | None = None,
        ) -> torch.Tensor:
            capture = getattr(self, "_art_route_capture", None)
            if capture is None:
                return original_apply(
                    self,
                    hidden_states,
                    w1,
                    w2,
                    router_logits,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    apply_router_weight_on_input,
                    num_expert_group,
                    e_score_correction_bias,
                    routed_scaling_factor,
                    topk_group,
                )

            del expert_map, a1q_scale, apply_router_weight_on_input
            import flashinfer

            assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]
            replay_out = capture[0]
            output = flashinfer.fused_moe.trtllm_bf16_moe(
                routing_logits=router_logits,
                routing_bias=e_score_correction_bias,
                hidden_states=hidden_states,
                gemm1_weights=w1,
                gemm2_weights=w2,
                num_experts=global_num_experts,
                top_k=self.topk,
                n_group=num_expert_group,
                topk_group=topk_group,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=self.ep_rank * self.local_num_experts,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=self.routing_method_type,
                activation_type=activation_to_flashinfer_int(activation),
                routing_replay_out=replay_out,
            )
            capture[1](replay_out[: hidden_states.shape[0]])
            return output

        setattr(apply, "__art_route_capture_patched__", True)
        setattr(apply, "__art_original__", original_apply)
        TrtLlmBf16Experts.apply = apply  # type: ignore[method-assign]

    original_bind = GPUModelRunner._bind_routed_experts_capturer
    if getattr(original_bind, "__art_trtllm_route_capture_patched__", False):
        return

    def bind(self: Any, capturer: Any) -> None:
        original_bind(self, capturer)
        for module in self.compilation_config.static_forward_context.values():
            if not isinstance(module, MoERunner):
                continue
            kernel = module.routed_experts.quant_method.moe_kernel
            if kernel is None or not kernel.is_monolithic:
                continue
            experts = kernel.impl.fused_experts
            if not isinstance(experts, TrtLlmBf16Experts):
                continue
            capture_fn = getattr(module.router, "capture_fn", None)
            if capture_fn is None:
                continue
            experts._art_route_capture = (  # type: ignore[attr-defined]
                torch.empty(
                    (capturer.device_buffer.shape[0], experts.topk),
                    dtype=torch.int16,
                    device=capturer.device_buffer.device,
                ),
                capture_fn,
            )

    setattr(bind, "__art_trtllm_route_capture_patched__", True)
    setattr(bind, "__art_original__", original_bind)
    GPUModelRunner._bind_routed_experts_capturer = bind  # type: ignore[method-assign]


def apply_qwen35_vllm_runtime_patches() -> None:
    patch_trtllm_monolithic_route_capture()
