"""Eager source allocations and admission; no CUDA lifetime/whole-plan bound."""

from types import SimpleNamespace
from typing import cast
import weakref

import pytest
from test_trainer_rank_head_memory import rank, request
from test_trainer_rank_head_recompute import _Head, _patch_local_head
import torch

from art import trainer_rank
from art.trainer_rank import ForwardInput, TrainerRank, _impl


@pytest.mark.parametrize("rows", [4, 5])
def test_no_grad_logits_assignment_has_output_and_three_distinct_cpu_storages(
    monkeypatch, rows
):
    _patch_local_head(monkeypatch)
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 4)
    weight = torch.arange(85, dtype=torch.bfloat16).reshape(17, 5) / 100
    hidden = torch.arange(rows * 5, dtype=torch.bfloat16).reshape(rows, 5) / 100
    model = SimpleNamespace(
        output_layer=_Head(weight),
        vocab_size=17,
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda value: value,
    )
    r = object.__new__(TrainerRank)
    r.runtime = SimpleNamespace(model=[model])
    original_select = torch.Tensor.index_select
    original_setitem = torch.Tensor.__setitem__
    selections, observed = [], []

    def select(self, *args, **kwargs):
        result = original_select(self, *args, **kwargs)
        if self.ndim == 2 and self.shape[1] == 17:
            selections.append((weakref.ref(self), weakref.ref(result)))
        return result

    def assign(self, key, value):
        if self.ndim == 2 and self.shape[1] == 17:
            # Weak references do not extend any allocation across callbacks.
            assert len(selections) == 2
            local, selected = (ref() for ref in selections[0])
            chunk, rhs = (ref() for ref in selections[1])
            assert rhs is value and selected is not None
            assert (
                selected.untyped_storage().data_ptr()
                == chunk.untyped_storage().data_ptr()
            )
            tensors = (self, local, chunk, rhs)
            assert all(t is not None and t.dtype == torch.bfloat16 for t in tensors)
            assert len({t.untyped_storage().data_ptr() for t in tensors}) == 4
            observed.append(sum(t.untyped_storage().nbytes() for t in tensors))
            selections.clear()
        return original_setitem(self, key, value)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    monkeypatch.setattr(torch.Tensor, "__setitem__", assign)
    req = ForwardInput(input_tokens=torch.arange(rows), logits=True, no_grad=True)
    positions = (torch.arange(rows),)
    with torch.no_grad():
        outputs = r._project_head(
            [r._forward_item(req)],
            SimpleNamespace(
                positions_by_item=positions, source_positions_by_item=positions
            ),
            hidden,
        )
    assert observed == [
        (rows + 3 * min(4, rows - start)) * 17 * 2 for start in range(0, rows, 4)
    ]
    torch.testing.assert_close(outputs[0].logits, hidden @ weight.T)


@pytest.mark.parametrize("mode", ["short", "disabled", "error", "strict"])
def test_optional_stats_refusal_keeps_capacity_separate_from_lower_bound(
    monkeypatch, mode
):
    calls = []

    def kernel(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("optional kernel failed")

    # Exercise the real dispatch without a CUDA tensor or a Triton import.
    monkeypatch.setattr(
        trainer_rank,
        "topk",
        SimpleNamespace(local_logsumexp_stats=kernel),
        raising=False,
    )
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", "64")
    monkeypatch.setenv(
        "ART_TRAINER_RANK_TRITON_TOPK",
        "0" if mode == "disabled" else "strict" if mode == "strict" else "1",
    )
    rows = 63 if mode == "short" else 64
    probe = cast(torch.Tensor, SimpleNamespace(is_cuda=True, shape=(rows, 17)))
    if mode == "strict":
        with pytest.raises(RuntimeError, match="optional kernel failed"):
            _impl._try_triton_stats("local_logsumexp_stats", probe)
    else:
        assert _impl._try_triton_stats("local_logsumexp_stats", probe) is None
    assert len(calls) == int(mode in {"error", "strict"})
    r = rank()
    dense = rows * 248320 * 2
    for grad in (False, True):
        req = [request(rows, grad=grad)]
        assert r._group_head_workspace_bytes(rows, req, grad_enabled=grad) == 7 * dense
        assert (
            r._group_head_workspace_bytes(
                rows, req, grad_enabled=grad, lower_bound=True
            )
            == (3 if grad else 1) * dense
        )


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("rows", [1, 63])
def test_eager_exp_boundary_has_four_distinct_cpu_dense_storages(
    monkeypatch, grad, rows
):
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda x: x)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda x: x)
    logits = torch.linspace(-2, 2, rows * 17, dtype=torch.bfloat16).reshape(rows, 17)
    logits.requires_grad_(grad)
    original_float, original_exp = torch.Tensor.float, torch.exp
    converted = []
    observed = []

    def as_float(self, *args, **kwargs):
        result = original_float(self, *args, **kwargs)
        if self is logits:
            converted.append(weakref.ref(result))
        return result

    def exp(subtraction):
        result = original_exp(subtraction)
        # Only weak references survive between callbacks; do not manufacture
        # overlap by retaining the converted tensor ourselves.
        tensors = (logits, converted[0](), subtraction, result)
        assert all(tensor is not None for tensor in tensors)
        assert [tensor.dtype for tensor in tensors] == [
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.float32,
        ]
        assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == 4
        observed.append(sum(tensor.untyped_storage().nbytes() for tensor in tensors))
        return result

    monkeypatch.setattr(torch.Tensor, "float", as_float)
    monkeypatch.setattr(torch, "exp", exp)
    with torch.set_grad_enabled(grad):
        actual = _impl._vocab_parallel_log_z(logits)
    assert observed == [7 * rows * 17 * 2]
    torch.testing.assert_close(actual, torch.logsumexp(original_float(logits), dim=-1))
    if grad:
        actual.sum().backward()
        assert logits.grad is not None and logits.grad.isfinite().all()


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("label", [1, -100, None])
def test_group_stats_cover_logits_chunks_before_short_target_tail(
    monkeypatch, grad, label
):
    _patch_local_head(monkeypatch)
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 4)
    original = _impl._vocab_parallel_log_z
    calls = []

    def log_z(logits):
        calls.append(tuple(logits.shape))
        return original(logits)

    monkeypatch.setattr(_impl, "_vocab_parallel_log_z", log_z)
    model = SimpleNamespace(
        output_layer=_Head(torch.arange(85, dtype=torch.float32).reshape(17, 5) / 100),
        vocab_size=17,
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda value: value,
    )
    r = object.__new__(TrainerRank)
    r.runtime = SimpleNamespace(model=[model])
    requests = [
        ForwardInput(input_tokens=torch.arange(13), logits=True, no_grad=not grad)
    ]
    positions = (torch.arange(13),)
    if label is not None:
        requests.append(
            ForwardInput(
                input_tokens=torch.tensor([12]),
                target_tokens=torch.tensor([label]),
                no_grad=not grad,
            )
        )
        positions += (torch.tensor([12]),)
    with torch.set_grad_enabled(grad):
        outputs = r._project_head(
            [r._forward_item(item) for item in requests],
            SimpleNamespace(
                positions_by_item=positions,
                source_positions_by_item=tuple(torch.arange(len(p)) for p in positions),
            ),
            torch.arange(65, dtype=torch.float32).reshape(13, 5) / 100,
        )
    assert calls == ([] if label is None else [(4, 17)] * 3 + [(1, 17)])
    assert outputs[0].logits.shape == (13, 17)
    if label == -100:
        assert outputs[1].target_logprobs.item() == 0
