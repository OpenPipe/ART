"""Ignored labels execute target backward on globally projected rows."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_trainer_rank_head_memory import rank, request
from test_trainer_rank_head_recompute import _Head
import torch

from art.trainer_rank import ForwardInput, TrainerRank, _impl


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_ignored_rows_reactivated_by_same_item_output_keep_backward_floor(extra):
    r = rank()
    item = replace(request(128, grad=True, ignored=True), **extra)
    assert (
        r._plan_head_workspace_bytes(r._plan_flat_forward([item]))
        == 3 * 128 * 248320 * 2
    )
    assert r._estimate_flat_forward([item], exact=True)[-1] == 3 * 128 * 248320 * 2
    assert r._estimate_flat_forward([item])[-1] == 3 * 128 * 248320 * 2
    assert r._head_target_chunk_rows([item], lower_bound=True) == 0


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_ignored_cross_request_overlap_and_validity_have_separate_roles(extra):
    r = rank()
    labelled = request(513, grad=True, ignored=True)
    raw = ForwardInput(input_tokens=torch.arange(512), no_grad=False, **extra)
    positions = (torch.arange(513), torch.arange(512))
    req = [labelled, raw]
    assert r._head_projection_rows(req, positions=positions) == 512
    assert r._head_target_chunk_rows(req, positions=positions) == 512
    assert r._head_target_chunk_rows(req) == 512
    assert r._head_target_chunk_rows(req, lower_bound=True) == 0
    disjoint = (torch.arange(513) + 1000, torch.arange(512))
    assert r._head_target_chunk_rows(req, positions=disjoint) == 0
    valid_tail = replace(
        labelled, target_tokens=torch.cat((torch.full((512,), -100), torch.tensor([1])))
    )
    assert r._head_target_chunk_rows([valid_tail, raw], positions=positions) == 512
    assert r._head_target_chunk_rows([valid_tail, raw], lower_bound=True) == 1
    assert r._head_projection_rows([labelled]) == 0
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([labelled])) == 0


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_actual_ignored_mixed_backward_keeps_zero_dense_index_graph(monkeypatch, extra):
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 4)
    monkeypatch.setattr(_impl, "_language_model", lambda model: model)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda x: x)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda x: x)
    monkeypatch.setattr(_impl, "_vocab_range", lambda logits: (0, logits.shape[-1]))
    monkeypatch.setattr(
        TrainerRank, "_gather_tensor_parallel_logits", lambda self, x: x
    )
    monkeypatch.setattr(
        _impl,
        "_vocab_parallel_topk_from_local",
        lambda values, tokens, *, k, log_z, vocab_start: _impl.TopK(
            values[:, :k] - log_z[:, None], tokens[:, :k]
        ),
    )
    original = _impl._vocab_parallel_target_logprobs
    dense_backward = []
    normalizer_backward = []

    def target_path(logits, labels, log_z, *, row_offsets):
        assert (labels == -100).all()
        log_z.register_hook(
            lambda grad: normalizer_backward.append(
                (tuple(grad.shape), bool((grad == 0).all()))
            )
        )
        output = original(logits, labels, log_z, row_offsets=row_offsets)
        queue = [output.grad_fn]
        seen = set()
        while queue:
            node = queue.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            if type(node).__name__.startswith("IndexBackward"):
                node.register_hook(
                    lambda inputs, outputs: dense_backward.append(
                        (tuple(inputs[0].shape), bool((inputs[0] == 0).all()))
                    )
                )
            queue.extend(next_node for next_node, _ in node.next_functions)
        return output

    monkeypatch.setattr(_impl, "_vocab_parallel_target_logprobs", target_path)
    generator = torch.Generator().manual_seed(79)
    hidden = torch.randn(8, 5, generator=generator, requires_grad=True)
    weight = torch.randn(17, 5, generator=generator)
    model = SimpleNamespace(
        output_layer=_Head(weight),
        vocab_size=17,
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda x: x,
    )
    trainer = object.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(model=[model])
    # The ignored-only item projects no row itself. The independent raw/top-k
    # request reaches its positions and therefore executes the target branch.
    labelled = ForwardInput(
        input_tokens=torch.tensor([0, 7]),
        target_tokens=torch.tensor([-100, -100]),
        no_grad=False,
    )
    raw = ForwardInput(input_tokens=torch.arange(8), no_grad=False, **extra)
    positions = (torch.tensor([0, 7]), torch.arange(8))
    outputs = trainer._project_head(
        [trainer._forward_item(r) for r in [labelled, raw]],
        SimpleNamespace(
            positions_by_item=positions,
            source_positions_by_item=(torch.arange(2), torch.arange(8)),
        ),
        hidden,
    )
    loss = outputs[0].target_logprobs.sum()
    assert loss.requires_grad and float(loss.detach()) == 0
    loss.backward()
    assert hidden.grad is not None and bool((hidden.grad == 0).all())
    assert model.output_layer.weight.grad is not None and bool(
        (model.output_layer.weight.grad == 0).all()
    )
    assert dense_backward and all(
        shape == (4, 17) and zero for shape, zero in dense_backward
    )
    assert normalizer_backward and all(zero for _, zero in normalizer_backward)
