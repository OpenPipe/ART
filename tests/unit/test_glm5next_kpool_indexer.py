from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from art.megatron.glm5next.kpool_indexer import (
    KPoolIndexer,
    compress_keys,
    index_width,
    pooled_indexer_topk,
)


def _sets(indices: torch.Tensor) -> list[set[int]]:
    return [
        set(row[row >= 0].tolist()) for row in indices.reshape(-1, indices.shape[-1])
    ]


def test_real_shapes_and_frozen_parameters() -> None:
    with torch.device("meta"):
        indexer = KPoolIndexer()
    assert indexer.linear_wq_b.weight.shape == (4096, 1536)
    assert indexer.linear_wk.weight.shape == (128, 4096)
    assert indexer.linear_weights_proj.weight.shape == (32, 4096)
    assert indexer.index_kpool_compress_gate.shape == (128, 4096)
    assert indexer.index_kpool_compress_ape.shape == (4, 128)
    assert all(not p.requires_grad for p in indexer.parameters())
    assert index_width(2048) == 2112


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pool_rounding_matches_hf(dtype: torch.dtype) -> None:
    torch.manual_seed(81)
    keys, gates = [torch.randn(39, 128).to(dtype) for _ in range(2)]
    ape = torch.randn(4, 128).to(dtype)
    starts = torch.arange(9) * 4
    # HF casts probabilities to key dtype before its product and reduction.
    probs = (gates[:36].view(9, 4, 128).float() + ape.float()).softmax(1).to(dtype)
    expected = (probs * keys[:36].view(9, 4, 128)).sum(1)
    torch.testing.assert_close(
        compress_keys(keys, gates, ape, starts), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("lengths", [(3,), (32,), (81,), (7, 33, 41)])
def test_selection_budget_causality_and_tail(lengths: tuple[int, ...]) -> None:
    torch.manual_seed(2)
    length = sum(lengths)
    q = torch.randn(length, 4, 8)
    keys, gates = [torch.randn(length, 8) for _ in range(2)]
    weights, ape = torch.randn(length, 4), torch.randn(4, 8)
    cu = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    got = pooled_indexer_topk(q, keys, weights, gates, ape, cu, topk=16)
    chunked = pooled_indexer_topk(
        q, keys, weights, gates, ape, cu, topk=16, query_chunk_size=3
    )
    assert _sets(got) == _sets(chunked)
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        pools = compress_keys(
            keys, gates, ape, torch.arange((end - start) // 4) * 4 + start
        )
        for query in range(start, end):
            ids = got[0, query]
            selected = ids[ids >= 0].tolist()
            assert len(set(selected)) == len(selected)
            complete = (query - start + 1) // 4
            tail = (query - start + 1) % 4
            assert len(selected) == min(complete, 4) * 4 + tail
            assert all(start <= token <= query for token in selected)
            assert set(range(query - tail + 1, query + 1)) <= set(selected)
            if complete <= 4:
                assert set(selected) == set(range(start, query + 1))
            else:
                # Independent per-head definition of each visible pool score.
                scores = torch.stack(
                    [
                        torch.stack(
                            [
                                weights[query, h] * torch.dot(q[query, h], key).relu()
                                for h in range(4)
                            ]
                        ).sum()
                        for key in pools[:complete]
                    ]
                )
                want = {
                    start + int(p) * 4 + slot
                    for p in scores.topk(4).indices
                    for slot in range(4)
                }
                want.update(range(query - tail + 1, query + 1))
                assert set(selected) == want
    if length > 3:
        local = pooled_indexer_topk(
            q[3:], keys, weights[3:], gates, ape, cu, topk=16, query_start=3
        )
        assert _sets(local) == _sets(got[:, 3:])


@pytest.mark.parametrize("length", [3, 32, 81])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_hf_indexer_selected_sets(length: int, dtype: torch.dtype) -> None:
    hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=4,
        index_head_dim=8,
        qk_rope_head_dim=0,
        index_topk=16,
        q_lora_rank=12,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )
    torch.manual_seed(23)
    reference = hf.Glm5NextTextIndexer(config, 0).to(dtype=dtype)
    reference.index_kpool_compress_gate.data.normal_()
    reference.index_kpool_compress_ape.data.normal_()
    got = KPoolIndexer(
        hidden_size=16, q_lora_rank=12, num_heads=4, head_dim=8, topk=16
    ).to(dtype=dtype)
    state_dict = {
        key.replace("wq_b.", "linear_wq_b.")
        .replace("wk.", "linear_wk.")
        .replace("weights_proj.", "linear_weights_proj."): value
        for key, value in reference.state_dict().items()
    }
    got.load_state_dict(state_dict)
    hidden, latent = (
        torch.randn(1, length, 16).to(dtype),
        torch.randn(1, length, 12).to(dtype),
    )
    expected = reference(hidden, latent, torch.ones(1, length, dtype=torch.bool), None)
    q, keys, weights, gates = got.project(
        hidden.transpose(0, 1), latent.transpose(0, 1)
    )
    indices = pooled_indexer_topk(
        q[0],
        keys[0],
        weights[0],
        gates[0],
        got.index_kpool_compress_ape,
        [0, length],
        topk=16,
    )
    assert _sets(indices) == _sets(expected)
    assert not q.requires_grad and not keys.requires_grad


def test_tied_scores_keep_a_valid_pool_set() -> None:
    q, keys, weights, gates, ape = (
        torch.zeros(32, 2, 8),
        torch.zeros(32, 8),
        torch.ones(32, 2),
        torch.zeros(32, 8),
        torch.zeros(4, 8),
    )
    got = pooled_indexer_topk(q, keys, weights, gates, ape, [0, 32], topk=16)
    ids = got[0, -1]
    valid = ids[ids >= 0]
    assert len(set(valid.tolist())) == 16
    assert bool(((valid >= 0) & (valid < 32)).all())


@pytest.mark.parametrize("topk,pool", [(0, 4), (15, 4), (16, 0)])
def test_reject_invalid_budget(topk: int, pool: int) -> None:
    with pytest.raises(ValueError):
        index_width(topk, pool)
