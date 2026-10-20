import pytest
import torch

from art.megatron.flex_attn.compiled import (
    flash_sparse_block_size_for_head_dim,
    flex_backend_for_head_dims,
)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    "major,backend", [(9, "FLASH"), (10, "TRITON"), (11, "TRITON"), (12, "TRITON")]
)
def test_sparse_training_uses_supported_backend(monkeypatch, head_dim, major, backend):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    device = torch.device("cuda")
    assert (
        flex_backend_for_head_dims(
            head_dim=head_dim, head_dim_v=head_dim, device=device
        )
        == backend
    )
    block = flash_sparse_block_size_for_head_dim(
        head_dim=head_dim, head_dim_v=head_dim, device=device
    )
    assert block == ((128, 64) if major == 9 and head_dim == 256 else (128, 128))
