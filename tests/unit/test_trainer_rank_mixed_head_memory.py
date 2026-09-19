"""Mixed head admission components; CPU autograd, never a full CUDA bound."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
from test_trainer_rank_head_memory import rank, request
from test_trainer_rank_head_recompute import _Head
import torch

from art.trainer_rank import ForwardInput, TrainerRank, _impl
from art.trainer_rank._impl import Unset


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_adding_output_cannot_erase_existing_target_admission_floor(extra):
    r = rank()
    target = request(128, grad=True)
    added = ForwardInput(input_tokens=torch.tensor([20000]), no_grad=False, **extra)
    before = r._plan_cost(r._plan_flat_forward([target])).required
    plan = r._plan_flat_forward([target, added])
    after = r._plan_cost(plan).required
    print({"extra": extra, "before": before, "after": after})
    assert after >= before
    assert r._plan_head_workspace_bytes(plan) == 3 * 129 * 248320 * 2
    r._available_memory_bytes = lambda: before - 1
    assert not r._memory_check(plan).fits


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_sparse_target_prices_its_full_mixed_chunk_and_short_tail(extra):
    r = rank()
    target = replace(request(1, grad=True), input_tokens=torch.tensor([9999]))
    raw = ForwardInput(input_tokens=torch.arange(512), no_grad=False, **extra)
    req = [target, raw]
    full = (torch.tensor([0]), torch.arange(1, 513))
    tail = (torch.tensor([512]), torch.arange(512))
    assert r._head_target_chunk_rows(req, positions=full) == 512
    assert r._head_target_chunk_rows(req, positions=tail) == 1
    assert r._head_target_chunk_rows(req, lower_bound=True) == 1
    assert r._head_target_chunk_rows(req) == 512
    dense = 512 * 248320 * 2
    assert (
        r._group_head_workspace_bytes(512, req, grad_enabled=True, positions=full)
        == 3 * dense
    )
    assert (
        r._group_head_workspace_bytes(512, req, grad_enabled=True, positions=tail)
        == dense
    )


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_shared_multilabel_union_matches_actual_and_split_bounds(extra):
    r = rank()
    a = replace(
        request(4, grad=True),
        target_tokens=torch.tensor(
            [[-100, 1], [-100, -100], [-100, -100], [-100, -100]]
        ),
    )
    b = replace(
        a,
        target_tokens=torch.tensor(
            [[-100, -100], [-100, -100], [2, -100], [-100, -100]]
        ),
    )
    raw = ForwardInput(input_tokens=torch.arange(4), no_grad=False, **extra)
    req = [a, a, b, b, raw]
    for minimal in (False, True):
        plan = r._plan_flat_forward(req, memory_minimal=minimal)
        exact = r._estimate_flat_forward(req, exact=True, memory_minimal=minimal)
        assert exact[-1] == r._plan_head_workspace_bytes(plan)
        lower = r._estimate_flat_forward(req, memory_minimal=True)[-1]
        upper = r._estimate_flat_forward(req)[-1]
        assert lower <= exact[-1] <= upper
        assert (
            r._split_chunk_lower_cost(
                req, tuple(x.input_tokens for x in req), checkpoint=Unset
            ).required
            <= r._plan_cost(plan).required
        )
    assert (
        r._plan_head_workspace_bytes(r._plan_flat_forward(req, memory_minimal=True))
        == 3 * 4 * 248320 * 2
    )


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_ignored_device_labels_and_no_target_keep_distinct_guards(extra):
    r = rank()
    ignored = replace(request(128, grad=True, ignored=True), **extra)
    dense = 128 * 248320 * 2
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([ignored])) == 3 * dense
    no_target = replace(ignored, target_tokens=None)
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([no_target])) == dense
    device = replace(
        ignored, target_tokens=torch.empty(128, device="meta", dtype=torch.long)
    )
    assert r._head_target_chunk_rows([device], lower_bound=True) == 0
    assert r._head_target_chunk_rows([device]) == 128
    assert r._head_target_chunk_rows([device], positions=(torch.arange(128),)) == 128
    assert (
        r._head_target_chunk_rows(
            [device], positions=(torch.empty(128, device="meta", dtype=torch.long),)
        )
        == 128
    )


@pytest.mark.parametrize("mutation", ["no_grad", "custom_scale", "mup", "head_hook"])
def test_mixed_path_preserves_source_scaling_and_gradient_guards(mutation):
    r = rank()
    item = replace(request(128, grad=True), logits=True)
    model = r.runtime.model[0]
    if mutation == "no_grad":
        item = replace(item, no_grad=True)
    elif mutation == "custom_scale":
        model._scale_logits = lambda logits: logits
    elif mutation == "mup":
        model.config.use_mup = True
    else:
        model.output_layer.register_forward_hook(lambda *args: None)
    expected = 0 if mutation == "head_hook" else 128 * 248320 * 2
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([item])) == expected


@pytest.mark.parametrize(
    "extra",
    [{"logits": True}, {"top_k": 2}, {"top_k": 12}, {"logits": True, "top_k": 2}],
)
def test_actual_mixed_projection_preserves_target_outputs_and_backward(
    monkeypatch, extra
):
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

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
    calls = []

    def target_path(logits, labels, log_z, *, row_offsets):
        calls.append((tuple(logits.shape), labels.tolist(), row_offsets.tolist()))
        return original(logits, labels, log_z, row_offsets=row_offsets)

    monkeypatch.setattr(_impl, "_vocab_parallel_target_logprobs", target_path)
    generator = torch.Generator().manual_seed(97)
    hidden = torch.randn(13, 5, generator=generator, dtype=torch.float64)
    weights = torch.randn(17, 5, generator=generator, dtype=torch.float64)
    target_positions = torch.tensor([0, 2, 12])
    labels = torch.tensor([[1, -100], [2, 3], [16, -100]])
    target = ForwardInput(
        input_tokens=torch.tensor([0, 2, 12]), target_tokens=labels, no_grad=False
    )
    added = ForwardInput(input_tokens=torch.arange(13), no_grad=False, **extra)

    def run(mixed):
        model = SimpleNamespace(
            output_layer=_Head(weights),
            vocab_size=17,
            config=SimpleNamespace(use_mup=False, padded_vocab_size=17),
            share_embeddings_and_output_weights=False,
        )
        model._scale_logits = MethodType(LanguageModule._scale_logits, model)
        trainer = object.__new__(TrainerRank)
        trainer.runtime = SimpleNamespace(model=[model])
        x = hidden.clone().requires_grad_()
        requests = [target, added] if mixed else [target]
        positions = (
            (target_positions, torch.arange(13)) if mixed else (target_positions,)
        )
        outputs = trainer._project_head(
            [trainer._forward_item(r) for r in requests],
            SimpleNamespace(
                positions_by_item=positions,
                source_positions_by_item=tuple(torch.arange(len(p)) for p in positions),
            ),
            x,
        )
        loss = -outputs[0].target_logprobs.sum() / int((labels != -100).sum())
        loss.backward()
        if mixed:
            expected_logits = hidden @ weights.T
            if extra.get("logits"):
                torch.testing.assert_close(outputs[1].logits, expected_logits)
            if "top_k" in extra:
                expected_values, expected_tokens = torch.topk(
                    expected_logits.float().log_softmax(-1), k=extra["top_k"], dim=-1
                )
                torch.testing.assert_close(outputs[1].top_k.logprobs, expected_values)
                torch.testing.assert_close(outputs[1].top_k.tokens, expected_tokens)
        return (
            outputs[0].target_logprobs.detach(),
            x.grad,
            model.output_layer.weight.grad,
        )

    before = run(False)
    calls.clear()
    after = run(True)
    for a, b in zip(after, before, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert any(shape == (4, 17) and len(rows) < 4 for shape, _, rows in calls)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in after)
