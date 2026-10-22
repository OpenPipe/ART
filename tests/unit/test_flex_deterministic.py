"""The standard PyTorch determinism opt-in must select a supported backend."""

import pytest
import torch

compiled = pytest.importorskip("art.megatron.flex_attn.compiled")


@pytest.mark.parametrize("enabled,expected", [(False, "FLASH"), (True, "FLASH")])
def test_determinism_preserves_forward_backend(monkeypatch, enabled, expected):
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: enabled)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(compiled, "_FORCED_FLEX_BACKEND", "FLASH")
    assert (
        compiled.flex_backend_for_head_dims(
            head_dim=128, head_dim_v=128, device=torch.device("cuda")
        )
        == expected
    )


def test_determinism_preserves_flash_mask_geometry(monkeypatch):
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert compiled.flash_sparse_block_size_for_head_dim(
        head_dim=256, head_dim_v=256, device=torch.device("cuda")
    ) == (128, 64)


@pytest.mark.parametrize("deterministic", [False, True])
def test_deterministic_flash_rejects_uncompiled_cache_fallback(
    monkeypatch, deterministic
):
    original = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(compiled, "flex_attention", lambda q, *args, **kwargs: q + 1)
    torch._dynamo.reset()
    try:
        torch.use_deterministic_algorithms(deterministic)
        attention = torch.compile(
            compiled._forced_flex_attention_dense, backend="eager", dynamic=False
        )
        with torch._dynamo.config.patch(recompile_limit=1):
            options = dict(block_mask=None, scale=1.0, enable_gqa=False)
            q = torch.zeros(2)
            assert torch.equal(attention(q, q, q, **options), q + 1)
            q = torch.zeros(3)
            if deterministic:
                with pytest.raises(RuntimeError, match="compiled FLASH"):
                    attention(q, q, q, **options)
            else:
                assert torch.equal(attention(q, q, q, **options), q + 1)
    finally:
        torch._dynamo.reset()
        torch.use_deterministic_algorithms(original, warn_only=warn_only)
