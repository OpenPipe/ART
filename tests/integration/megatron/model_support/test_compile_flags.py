import torch
from torch._dynamo.testing import CompileCounter

from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
)
from art.megatron.training.compile import _configure_dynamo


class _DynamicProjection(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(2, 4))
        self.b = torch.nn.Parameter(torch.randn(4, width))

    @torch.compiler.disable
    def active_parameters(
        self,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        return self.a, self.b

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        a, b = self.active_parameters()
        return (value @ a) @ b


def test_dynamic_projection_parameters_reuse_compiled_graph() -> None:
    torch._dynamo.reset()
    counter = CompileCounter()
    try:
        with torch._dynamo.config.patch(
            force_parameter_static_shapes=True, recompile_limit=32
        ):
            _configure_dynamo()
            assert not torch._dynamo.config.force_parameter_static_shapes
            compiled = [
                torch.compile(_DynamicProjection(width), backend=counter)
                for width in (8, 4, 16, 32, 12, 20, 24, 28, 36, 40)
            ]
            outputs = [projection(torch.ones(1, 2)) for projection in compiled]
        assert [tuple(output.shape) for output in outputs] == [
            (1, projection.b.shape[1]) for projection in compiled
        ]
        assert counter.frame_count <= 2
        sum(output.sum() for output in outputs).backward()
        assert all(
            projection.a.grad is not None and projection.b.grad is not None
            for projection in compiled
        )
    finally:
        torch._dynamo.reset()


def test_gemma4_wide_global_attention_uses_lower_triton_stage_count() -> None:
    provider = type("Provider", (), {"global_head_dim": 512})()

    assert GEMMA4_DENSE_HANDLER.flex_attention_compile_crash_config(
        provider
    ).triton_num_stages_2_head_dims == (512,)
    assert GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
        provider
    ).triton_num_stages_2_head_dims == (512,)


def test_gemma4_standard_global_attention_keeps_default_triton_stage_count() -> None:
    provider = type("Provider", (), {"global_head_dim": 256})()

    assert (
        GEMMA4_DENSE_HANDLER.flex_attention_compile_crash_config(
            provider
        ).triton_num_stages_2_head_dims
        == ()
    )
    assert (
        GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
            provider
        ).triton_num_stages_2_head_dims
        == ()
    )
