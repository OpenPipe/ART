import pytest
import torch

from art.megatron.flex_attn import compiled
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


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("major", [9, 10, 12])
def test_wide_head_dispatch_preserves_hopper_and_sm100(
    monkeypatch, sparse, head_dim, major
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    backend = "FLASH" if major == 9 else "TRITON"
    kind = "sparse" if sparse else "dense"
    if sparse:
        actual = compiled.get_sparse_compiled_flex_attention(
            family_key="test",
            backend=backend,
            head_dim=head_dim,
            head_dim_v=head_dim,
            device=torch.device("cuda"),
            triton_num_stages_2_head_dims=(256,),
        )
    else:
        actual = compiled.get_dense_compiled_flex_attention(
            backend=backend,
            head_dim=head_dim,
            head_dim_v=head_dim,
            device=torch.device("cuda"),
            triton_num_stages_2_head_dims=(256,),
        )
    prefix = ""
    if major != 9:
        prefix = "triton_"
        if head_dim == 256:
            prefix = "sm120_wide_head_" if major == 12 else "blackwell_wide_head_"
    assert actual is getattr(compiled, f"{prefix}{kind}_compiled_flex_attention")


def test_cpu_and_unspecified_device_keep_flash():
    for device in (None, torch.device("cpu")):
        assert (
            compiled.flex_backend_for_head_dims(
                head_dim=256, head_dim_v=256, device=device
            )
            == "FLASH"
        )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_blackwell_fallback_does_not_change_h200(monkeypatch, capability):
    for part, expected in ((capability, "TRITON"), ((9, 0), "FLASH")):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: part)
        assert (
            flex_backend_for_head_dims(
                head_dim=128, head_dim_v=256, device=torch.device("cuda")
            )
            == expected
        )


def test_sm120_wide_head_tiles_cover_both_backward_paths(monkeypatch):
    options = compiled._SM120_WIDE_HEAD_FLEX_KERNEL_OPTIONS
    assert options == {
        "BACKEND": "TRITON",
        "num_stages": 1,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 32,
        "bwd_BLOCK_M2": 32,
        "bwd_BLOCK_N2": 32,
    }
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    assert compiled._needs_sm120_wide_head_tile(
        "TRITON", 128, 256, torch.device("cuda")
    )
    assert not compiled._needs_sm120_wide_head_tile(
        "FLASH", 256, 256, torch.device("cuda")
    )
