from types import SimpleNamespace

import pytest
import torch

from art.megatron.gdn import operator


@pytest.mark.parametrize("rows", [(1,), (2, 2, 2), (1, 2, 3), (3, 1, 3, 0), (3, 0, 2)])
@pytest.mark.parametrize("conv_dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_state_selection_reuses_indices_and_preserves_values_gradients_and_views(
    rows, conv_dtype, monkeypatch
) -> None:
    conv = torch.arange(60, dtype=conv_dtype).reshape(5, 3, 4).transpose(1, 2)
    rec = torch.arange(80, dtype=torch.float32).reshape(5, 2, 2, 4).transpose(1, 3)
    chunks = (conv.requires_grad_(), rec.requires_grad_())
    refs = tuple(chunk.detach().clone().requires_grad_() for chunk in chunks)
    cache = operator._TreeStateChunkCache(device=conv.device)
    cache.append_families((0, 1, 2, 3, 4), *chunks)
    indices = []
    original = operator._long_tensor

    def record(values, *, device):
        index = original(values, device=device)
        indices.append(index)
        return index

    monkeypatch.setattr(operator, "_long_tensor", record)
    selected = cache.states_for_families(None, rows, state_reference=conv)
    expected = tuple(chunk[list(rows)] for chunk in refs)
    is_view = len(set(rows)) == 1 or rows == tuple(range(rows[0], rows[0] + len(rows)))
    assert len(indices) == (0 if is_view else 1)
    if indices:
        assert indices[0].dtype == torch.long
        assert indices[0].device == conv.device
    for chunk, ref, actual, wanted in zip(
        chunks, refs, selected, expected, strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert actual.dtype == chunk.dtype and actual.device == chunk.device
        if is_view:
            assert (
                actual.untyped_storage().data_ptr()
                == chunk.untyped_storage().data_ptr()
            )
        weights = (
            torch.arange(actual.numel(), dtype=actual.dtype).reshape(actual.shape) % 3
        )
        (actual * weights).sum().backward()
        (wanted * weights).sum().backward()
        torch.testing.assert_close(chunk.grad, ref.grad, rtol=0, atol=0)


def test_parent_selection_preserves_roots_multiple_chunks_and_duplicate_gradients() -> (
    None
):
    gdn = SimpleNamespace(
        conv_dim_local_tp=2,
        conv_kernel_dim=3,
        num_v_heads_local_tp=2,
        key_head_dim=2,
        value_head_dim=2,
    )
    cache = operator._TreeStateChunkCache(device=torch.device("cpu"))
    families = ((4, 1, 7), (9, 2))
    chunks = [
        (
            torch.arange(len(ids) * 4, dtype=torch.float64)
            .reshape(-1, 2, 2)
            .requires_grad_(),
            torch.arange(len(ids) * 8, dtype=torch.float32)
            .reshape(-1, 2, 2, 2)
            .requires_grad_(),
        )
        for ids in families
    ]
    refs = [tuple(x.detach().clone().requires_grad_() for x in pair) for pair in chunks]
    for ids, pair in zip(families, chunks, strict=True):
        cache.append_families(ids, *pair)
    parents = (7, -1, 2, 4, 9, 7)
    selected = cache._mixed_parent_states(
        gdn, parents, state_reference=chunks[0][0], batch_size=len(parents)
    )
    lookup = {
        family: (i, row)
        for i, ids in enumerate(families)
        for row, family in enumerate(ids)
    }
    for kind, actual in enumerate(selected):
        reference_rows = [
            refs[lookup[parent][0]][kind][lookup[parent][1]]
            if parent >= 0
            else torch.zeros_like(refs[0][kind][0])
            for parent in parents
        ]
        expected = torch.stack(reference_rows)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = (
            torch.arange(actual.numel(), dtype=actual.dtype).reshape(actual.shape) % 3
        )
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for pair, ref in zip(chunks, refs, strict=True):
            torch.testing.assert_close(pair[kind].grad, ref[kind].grad, rtol=0, atol=0)

    # Empty CP send sets still carry differentiable, correctly shaped states.
    empty = cache.states_for_families(gdn, (), state_reference=chunks[0][0])
    assert [tuple(x.shape) for x in empty] == [(0, 2, 2), (0, 2, 2, 2)]
    assert all(x.requires_grad for x in empty)
    for missing in ((3,), (-1,)):
        with pytest.raises(RuntimeError, match="missing parent state"):
            cache.states_for_families(gdn, missing, state_reference=chunks[0][0])
