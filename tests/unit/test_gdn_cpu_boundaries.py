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
    bucket = operator.GdnSegmentBucketPlan(
        length=max(lengths),
        lengths=torch.tensor(lengths),
        lengths_cpu=torch.tensor(lengths),
        cu_seqlens=device_boundaries,
        cu_seqlens_cpu=cpu_boundaries,
        real_mask=torch.ones(sum(lengths), dtype=torch.bool),
        row_indices=torch.zeros(sum(lengths), dtype=torch.long),
        position_indices=torch.cat([torch.arange(n) for n in lengths]),
        family_indices=torch.arange(len(lengths)),
        real_token_count_static=sum(lengths),
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


@pytest.mark.parametrize("root_closed,tokens", [(False, 64), (True, 64), (True, 0)])
@pytest.mark.parametrize("needs_final_state", [False, True])
def test_tree_cp_final_state_autograd_edges_match_main_unless_root_closed(
    monkeypatch, root_closed, tokens, needs_final_state
):
    from dataclasses import replace

    from art.megatron.gdn.gdn_prefix_tree import (
        GdnSegmentSpec,
        _build_position_bucket_plan,
    )

    bucket = replace(
        _build_position_bucket_plan(
            (GdnSegmentSpec(0, 0, 0, 0, 0, 64, "prefix"),),
            ((0, 64, 0),) if tokens else (),
            sequence_length=64,
            device="cpu",
            allow_empty_shards=root_closed,
        ),
        allow_empty_shards=root_closed,
        needs_final_state=needs_final_state,
        parent_indices=torch.tensor([-1]),
        parent_indices_cpu=torch.tensor([-1]),
    )
    output = torch.ones(tokens, requires_grad=True)
    conv = torch.ones(1, requires_grad=True)
    rec = torch.ones(1, requires_grad=True)
    cache = operator._TreeStateChunkCache(device=output.device)
    monkeypatch.setattr(cache, "parent_states", lambda *args, **kwargs: (conv, rec))
    monkeypatch.setattr(cache, "append", lambda *args: None)
    monkeypatch.setattr(
        operator, "_gather_bucket_streams", lambda *args: (output, output, output)
    )
    monkeypatch.setattr(
        operator, "run_gdn_bucket", lambda *args, **kwargs: (output, conv, rec)
    )
    monkeypatch.setattr(
        operator, "_scatter_bucket_recurrent_output", lambda *args: output
    )
    _, dependency = operator._run_tree_bucket(
        None,
        output,
        output,
        output,
        output,
        cache,
        bucket,
        state_reference=output,
        recurrent_cp=True,
    )
    grads = torch.autograd.grad(dependency, (output, conv, rec), allow_unused=True)
    assert (grads[0] is not None) == (tokens > 0)
    for grad in grads[1:]:
        assert (grad is not None) == (needs_final_state or root_closed)


@pytest.mark.parametrize("root_closed", [False, True])
def test_empty_bucket_native_permission_is_forwarded_only_when_opted_in(
    monkeypatch, root_closed
):
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnSegmentSpec,
        _build_position_bucket_plan,
    )

    bucket = _build_position_bucket_plan(
        (GdnSegmentSpec(0, 0, 0, 0, 0, 64, "prefix"),),
        (),
        sequence_length=64,
        device="cpu",
        allow_empty_shards=True,
    )
    from dataclasses import replace

    bucket = replace(bucket, allow_empty_shards=root_closed)
    gdn = SimpleNamespace(
        num_key_heads=1,
        num_value_heads=1,
        tp_size=1,
        key_head_dim=1,
        value_head_dim=1,
        use_qk_l2norm=True,
    )
    conv, rec = torch.zeros(1, 3, 2), torch.zeros(1, 1, 1, 1)
    monkeypatch.setattr(
        operator,
        "_chain_conv_initial_and_final",
        lambda *args, **kwargs: (conv, conv, None),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Empty peers must skip local convolution and L2 kernels")

    monkeypatch.setattr(operator, "_causal_conv1d_packed_varlen_with_state", forbidden)
    monkeypatch.setattr(operator, "_l2norm", forbidden)
    calls = []

    def native(q, k, v, **kwargs):
        calls.append(kwargs)
        return v, rec

    monkeypatch.setattr(operator, "chunk_gated_delta_rule_native_cp", native)

    def run():
        return operator.run_gdn_bucket(
            bucket,
            (torch.empty(0, 3), torch.empty(0, 1), torch.empty(0, 1)),
            (conv, rec),
            gdn=gdn,
            group=object(),
            recurrent_cp=True,
        )

    if root_closed:
        out, conv_final, rec_final = run()
        assert out.shape == (1, 0, 1, 1)
        assert conv_final is conv and rec_final is rec
        assert calls[0]["allow_empty_shards"] is True
    else:
        with pytest.raises(ValueError, match="non-empty"):
            run()
        assert calls == []
