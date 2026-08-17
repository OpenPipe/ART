from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from art.megatron.flex_attn.compiled import _needs_blackwell_wide_head_tile
from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
)
from art.megatron.training import compile as compile_module
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


def test_disabled_training_compile_does_not_change_dynamo_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        disable_compile=False,
        flags=(),
        unconditional_flags=(),
    )
    bundle = SimpleNamespace(
        handler=SimpleNamespace(
            compile_workaround_config=lambda _provider: config,
        )
    )
    monkeypatch.setattr(compile_module, "compile_enabled", lambda: False)
    monkeypatch.setattr(
        compile_module,
        "_configure_dynamo",
        lambda: pytest.fail("disabled compilation must not mutate Dynamo config"),
    )

    assert not compile_module.configure_training_compile(
        model=[], provider=object(), provider_bundle=cast(Any, bundle)
    )


def test_wide_head_tile_workaround_is_blackwell_only(monkeypatch) -> None:
    def selected(major: int) -> bool:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda _device: (major, 0)
        )
        return _needs_blackwell_wide_head_tile(
            backend="TRITON",
            head_dim=512,
            head_dim_v=512,
            triton_num_stages_2_head_dims=(512,),
            device=torch.device("cuda"),
        )

    assert selected(10)
    assert not selected(11)


def test_gemma4_wide_global_attention_uses_lower_triton_stage_count() -> None:
    provider = type(
        "Provider",
        (),
        {
            "global_head_dim": 512,
            "hidden_size": 5376,
            "kv_channels": 256,
            "num_attention_heads": 32,
            "num_layers": 12,
        },
    )()

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
