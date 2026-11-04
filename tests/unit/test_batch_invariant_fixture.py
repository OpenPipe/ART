"""Exercise the granted tiny decoder's actual rotary geometry entirely on CPU."""

from types import SimpleNamespace

import pytest
import torch

from tests.integration.megatron.test_batch_invariant import _configure_tiny_provider


@pytest.mark.parametrize("moe", [False, True])
def test_tiny_rotary_sections_fit_the_reduced_head(moe):
    provider = SimpleNamespace(rotary_percent=0.25, mrope_section=[11, 11, 10])
    _configure_tiny_provider(provider, moe=moe)
    width = int(provider.kv_channels * provider.rotary_percent)
    assert 2 * sum(provider.mrope_section) == width
    assert len(provider.linear_attention_freq) == provider.num_layers


@pytest.mark.parametrize("moe", [False, True])
def test_native_tiny_rotary_preprocess_on_cpu(monkeypatch, moe):
    rotary = pytest.importorskip(
        "megatron.core.models.common.embeddings.rotary_pos_embedding"
    )
    provider = SimpleNamespace(rotary_percent=0.25, mrope_section=[11, 11, 10])
    _configure_tiny_provider(provider, moe=moe)
    # Megatron's constructor chooses CUDA explicitly; only its storage device
    # changes here. The production rotary forward and split run unchanged.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    embedding = rotary.MultimodalRotaryEmbedding(
        provider.kv_channels, provider.rotary_percent
    )
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5)
    actual = embedding(positions, provider.mrope_section)
    assert actual.shape == (5, 1, 1, 8)
    assert torch.equal(
        actual[:4], embedding(positions[..., :4], provider.mrope_section)
    )
