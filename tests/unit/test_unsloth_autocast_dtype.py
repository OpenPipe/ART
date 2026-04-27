import torch

from art.unsloth.train import _get_dtype_for_autocasting


class _TinyModel(torch.nn.Module):
    def __init__(self, dtype_numels: list[tuple[torch.dtype, int]]) -> None:
        super().__init__()
        self.params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(numel, dtype=dtype))
                for dtype, numel in dtype_numels
            ]
        )


def test_get_dtype_for_autocasting_infers_bfloat16_model_when_env_unset(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    model = _TinyModel(
        [
            (torch.bfloat16, 8),
            (torch.float32, 1),
        ]
    )

    assert _get_dtype_for_autocasting(model) == torch.bfloat16


def test_get_dtype_for_autocasting_keeps_fp16_default_for_fp32_model(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    model = _TinyModel([(torch.float32, 8)])

    assert _get_dtype_for_autocasting(model) == torch.float16


def test_get_dtype_for_autocasting_honors_explicit_fp16(monkeypatch) -> None:
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    model = _TinyModel([(torch.bfloat16, 8)])

    assert _get_dtype_for_autocasting(model) == torch.float16


def test_get_dtype_for_autocasting_honors_explicit_bfloat16(monkeypatch) -> None:
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    model = _TinyModel([(torch.float16, 8)])

    assert _get_dtype_for_autocasting(model) == torch.bfloat16
