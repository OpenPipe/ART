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
