import sys
from types import SimpleNamespace

import pytest
import torch

from art.megatron.gdn import operator


@pytest.mark.parametrize("lengths", [(2, 3), (1, 64, 65), (3, 3)])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_bucket_passes_matching_cpu_boundaries_only_for_packed_fla(
    lengths, output_final_state, monkeypatch
):
    cpu_boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    device_boundaries = cpu_boundaries.clone()
    bucket = SimpleNamespace(
        segment_count=len(lengths),
        real_token_count=sum(lengths),
        length=max(lengths),
        lengths_cpu=torch.tensor(lengths),
        cu_seqlens=device_boundaries,
        cu_seqlens_cpu=cpu_boundaries,
    )
    gdn = SimpleNamespace(
        num_key_heads=1,
        num_value_heads=1,
        tp_size=1,
        key_head_dim=1,
        value_head_dim=1,
        use_qk_l2norm=False,
    )
    qkv = torch.arange(sum(lengths) * 3, dtype=torch.float32).reshape(-1, 3)
    qkv.requires_grad_()
    beta = torch.zeros(sum(lengths), 1)
    recurrent_g = torch.zeros_like(beta)
    conv = torch.zeros(len(lengths), 3, 2)
    initial = torch.zeros(len(lengths), 1, 1, 1)
    monkeypatch.setattr(
        operator,
        "_causal_conv1d_packed_varlen_with_state",
        lambda gdn, qkv, state, boundaries, **kwargs: (qkv, state),
    )
    monkeypatch.setattr(
        operator,
        "_prepare_packed_recurrent_inputs_fused",
        lambda qkv, beta, g, **kwargs: operator._prepare_dense_recurrent_inputs(
            qkv.unsqueeze(0), beta.unsqueeze(0), g.unsqueeze(0), **kwargs
        ),
    )
    calls = []

    def recurrent(query, key, value, **kwargs):
        calls.append(kwargs)
        return value, initial if kwargs["output_final_state"] else None

    monkeypatch.setattr(operator, "_chunk_gated_delta_rule", recurrent)
    result, _, final = operator.run_gdn_bucket(
        bucket,
        (qkv, beta, recurrent_g),
        (conv, initial),
        gdn=gdn,
        output_final_state=output_final_state,
    )
    assert len(calls) == 1
    call = calls[0]
    dense = len(set(lengths)) == 1
    assert call["cu_seqlens"] is (None if dense else device_boundaries)
    assert call.get("cu_seqlens_cpu") is (None if dense else cpu_boundaries)
    assert call["output_final_state"] is output_final_state
    assert final is (initial if output_final_state else None)
    torch.testing.assert_close(result.flatten(), qkv[:, 2], rtol=0, atol=0)
    result.sum().backward()
    expected = torch.zeros_like(qkv)
    expected[:, 2] = 1
    torch.testing.assert_close(qkv.grad, expected, rtol=0, atol=0)


def test_fp32_oracle_accepts_hint_but_uses_independent_device_boundaries(monkeypatch):
    from tests.integration.megatron.model_support.gdn_fp32_reference import (
        _torch_chunk_gated_delta_rule_reference,
    )

    lengths = []

    def recurrent(query, key, value, **kwargs):
        lengths.append(value.shape[1])
        return value, None

    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_5.modeling_qwen3_5",
        SimpleNamespace(torch_chunk_gated_delta_rule=recurrent),
    )
    value = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
    result, final = _torch_chunk_gated_delta_rule_reference(
        value,
        value,
        value,
        g=torch.zeros(1, 5, 1),
        beta=torch.zeros(1, 5, 1),
        cu_seqlens=torch.tensor([0, 2, 5]),
        cu_seqlens_cpu=torch.tensor([0, 4, 5]),
    )
    assert lengths == [2, 3]
    torch.testing.assert_close(result, value, rtol=0, atol=0)
    assert final is None
