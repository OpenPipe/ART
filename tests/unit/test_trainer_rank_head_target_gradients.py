"""Target and top-k log-prob gradients join the head's softmax gradient in FP32.

Rounded to BF16 separately, the target's -1 and the softmax's p cancel to
zero once p rounds to one (p >= 1 - 2**-9, an NLL below 1.95e-3).
"""

import math
from types import SimpleNamespace

import pytest
from test_trainer_rank_head_recompute import _patch_local_head
import torch
import torch.distributed as dist
from trainer_rank_test_support import process_group, spawn_and_join

from art.trainer_rank import ForwardInput, TrainerRank, _impl

ROWS, VOCAB = 40, 300


def _logits(rows, vocab, *, seed, labels_per_row=1):
    """BF16 logits whose first label per row has an NLL from 1e-6 to 1e-1."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=generator) * 2
    labels = torch.randint(0, vocab, (rows, labels_per_row), generator=generator)
    row = torch.arange(rows)
    others = logits.index_put((row, labels[:, 0]), torch.tensor(-math.inf))
    probability = torch.exp(-torch.logspace(-6, -1, rows))
    logits[row, labels[:, 0]] = others.logsumexp(-1) + torch.log(
        probability / (1 - probability)
    )
    return logits.bfloat16(), labels


def _reference(logits, terms):
    """FP32 cross-entropy gradient: one upcast before the gather and the softmax."""
    full = logits.float().requires_grad_()
    log_probs = full.log_softmax(-1)
    loss = torch.zeros(())
    for labels, weights in terms:
        selected = log_probs.gather(-1, labels.clamp(min=0))
        loss = loss - (weights * selected.masked_fill(labels == -100, 0.0)).sum()
    loss.backward()
    assert full.grad is not None
    return log_probs.detach(), full.grad


def _assert_fp32_combined(gradient, reference, weights):
    # One BF16 rounding of an FP32 gradient; the absolute term is the FP32
    # floor of 1 - p for the most confident rows.
    gradient = gradient.float()
    atol = 2**-21 * float(weights.abs().max())
    torch.testing.assert_close(gradient, reference, rtol=2**-7, atol=atol)
    # The row gradient sums to zero, up to rounding of each entry.
    row_sums = gradient.sum(-1).abs()
    assert (row_sums <= 2**-7 * gradient.abs().sum(-1) + atol).all()


def _run_head(monkeypatch, logits, requests, positions, *, path, tp_rank=None):
    """Project ``logits`` rows (as the head's local logits) for ``requests``;
    with ``tp_rank``, ``logits`` is that rank's shard of a TP2 vocabulary."""
    _patch_local_head(monkeypatch)
    monkeypatch.setattr(
        TrainerRank,
        "_local_logits_from_hidden_rows",
        lambda self, model, hidden, output_weight: hidden,
    )
    if tp_rank is not None:
        local = int(logits.shape[-1])
        monkeypatch.setattr(
            _impl,
            "_vocab_range",
            lambda value: (tp_rank * local, (tp_rank + 1) * local),
        )
        monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", _all_reduce_max)
        monkeypatch.setattr(
            _impl, "_all_reduce_tensor_parallel_sum", _AllReduceSum.apply
        )

    def unexpected(*args, **kwargs):
        raise AssertionError(f"the {path} path took another statistics path")

    if path == "fallback":
        monkeypatch.setattr(_impl, "_triton_stats_enabled", lambda cuda, rows: False)
        monkeypatch.setattr(_impl, "_eager_local_logsumexp_stats", unexpected)
    elif path == "bounded":
        # An attempted kernel that fails: the bounded eager statistics run.
        monkeypatch.setattr(_impl, "_triton_stats_enabled", lambda cuda, rows: True)
        monkeypatch.setattr(_impl, "_try_triton_stats", lambda *a, **k: None)
        monkeypatch.setattr(_impl, "_vocab_parallel_log_z", unexpected)
    else:
        monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 64)
        monkeypatch.setattr(_impl, "_eager_local_logsumexp_stats", unexpected)
        monkeypatch.setattr(_impl, "_vocab_parallel_log_z", unexpected)
    model = SimpleNamespace(
        vocab_size=int(logits.shape[-1]),
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda value: value,
    )
    trainer = object.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(model=[model])
    hidden = logits.clone().requires_grad_()
    outputs = trainer._project_head(
        [trainer._forward_item(request) for request in requests],
        SimpleNamespace(
            positions_by_item=positions,
            source_positions_by_item=tuple(torch.arange(len(p)) for p in positions),
        ),
        hidden,
    )
    return outputs, hidden


def _target_case(layout, rows, vocab, *, seed, device="cpu"):
    """Requests, positions and (labels, weights) terms for one label layout."""
    logits, labels = _logits(rows, vocab, seed=seed, labels_per_row=2)
    generator = torch.Generator().manual_seed(seed + 1)
    first = labels[:, :1].clone()
    first[::9] = -100
    if layout == "single":
        terms = [(first, torch.ones(rows, 1))]
    elif layout == "multi":
        # Two labels per row (as distillation targets), some ignored.
        multi = labels.clone()
        multi[1::4, 1] = -100
        multi[::9, 0] = -100
        terms = [(multi, torch.rand(rows, 2, generator=generator) + 0.5)]
    else:
        # Two requests on the same rows: both targets land on each row.
        terms = [
            (first, torch.ones(rows, 1)),
            (labels[:, 1:], torch.randn(rows, 1, generator=generator)),
        ]
    requests = [
        ForwardInput(
            input_tokens=torch.zeros(rows, dtype=torch.long, device=device),
            target_tokens=(
                term_labels if layout == "multi" else term_labels.squeeze(1)
            ).to(device),
        )
        for term_labels, _ in terms
    ]
    positions = tuple(torch.arange(rows, device=device) for _ in terms)
    return logits, requests, positions, terms


def _topk_weights(rows, k, *, seed):
    """Random weights; even rows weight only the top-1 logprob, whose p - 1
    cancels as a target's does."""
    weights = torch.rand(rows, k, generator=torch.Generator().manual_seed(seed))
    weights[::2] = 0.0
    weights[::2, 0] = 1.0
    return weights


def _loss(outputs, terms):
    return -sum(
        (
            weights.to(output.target_logprobs.device).reshape(
                output.target_logprobs.shape
            )
            * output.target_logprobs
        ).sum()
        for output, (_, weights) in zip(outputs, terms, strict=True)
    )


def test_split_bf16_gradients_lose_the_target_term_in_the_dead_zone():
    """The pre-fix formulation on these logits: two BF16 gradients, so the
    target's p - 1 is exactly zero wherever p rounds to one in BF16."""
    logits, labels = _logits(ROWS, VOCAB, seed=0)
    row, target = torch.arange(ROWS), labels[:, 0]
    split = logits.clone().requires_grad_()
    log_probs = split[row, target].float() - split.float().logsumexp(-1)
    (-log_probs.sum()).backward()
    assert split.grad is not None
    probability = logits.float().softmax(-1)[row, target]
    dead = probability >= 1 - 2**-9
    assert int(dead.sum()) >= ROWS // 4
    assert (split.grad[row, target][dead] == 0).all()
    _, reference = _reference(logits, [(labels[:, :1], torch.ones(ROWS, 1))])
    assert (reference[row, target][dead] < 0).all()


@pytest.mark.parametrize("layout", ["single", "multi", "shared"])
@pytest.mark.parametrize("path", ["fallback", "bounded"])
def test_target_gradients_match_fp32_cross_entropy(monkeypatch, path, layout):
    logits, requests, positions, terms = _target_case(layout, ROWS, VOCAB, seed=3)
    outputs, hidden = _run_head(monkeypatch, logits, requests, positions, path=path)
    log_probs, reference = _reference(logits, terms)
    for output, (labels, _) in zip(outputs, terms, strict=True):
        expected = log_probs.gather(-1, labels.clamp(min=0))
        expected = expected.masked_fill(labels == -100, 0.0)
        torch.testing.assert_close(
            output.target_logprobs.reshape(labels.shape), expected
        )
    _loss(outputs, terms).backward()
    assert hidden.grad is not None and hidden.grad.dtype == torch.bfloat16
    labels = terms[0][0][:, 0]
    valid = labels != -100
    target_gradient = hidden.grad[torch.arange(ROWS), labels.clamp(min=0)][valid]
    assert (target_gradient != 0).all()
    weights = torch.cat([weights for _, weights in terms], dim=-1)
    _assert_fp32_combined(hidden.grad, reference, weights)


@pytest.mark.parametrize("path", ["fallback", "bounded"])
def test_topk_gradients_match_fp32_log_softmax(monkeypatch, path):
    logits, _ = _logits(ROWS, VOCAB, seed=5)
    request = ForwardInput(
        input_tokens=torch.zeros(ROWS, dtype=torch.long), top_k=12, no_grad=False
    )
    outputs, hidden = _run_head(
        monkeypatch, logits, [request], (torch.arange(ROWS),), path=path
    )
    top_k = outputs[0].top_k
    assert top_k is not None
    expected = logits.float().log_softmax(-1).topk(12, dim=-1)
    torch.testing.assert_close(top_k.logprobs, expected.values)
    weights = _topk_weights(ROWS, 12, seed=6)
    (-(weights * top_k.logprobs).sum()).backward()
    assert hidden.grad is not None
    _, reference = _reference(logits, [(top_k.tokens, weights)])
    _assert_fp32_combined(hidden.grad, reference, weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton needs CUDA")
@pytest.mark.parametrize("top_k", [None, 4, 12])
def test_triton_statistics_combine_target_gradients_in_fp32(monkeypatch, top_k):
    """Fused top-k (k <= 10), logsumexp, and logsumexp with selected top-k:
    64-row chunks over a vocabulary of two kernel blocks."""
    monkeypatch.delenv("ART_TRAINER_RANK_TRITON_TOPK", raising=False)
    monkeypatch.delenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", raising=False)
    rows, vocab = 128, 5_000
    logits, requests, positions, terms = _target_case(
        "shared", rows, vocab, seed=7, device="cuda"
    )
    if top_k is not None:
        requests[0] = ForwardInput(
            input_tokens=requests[0].input_tokens,
            target_tokens=requests[0].target_tokens,
            top_k=top_k,
        )
    kernels = []
    original = _impl._try_triton_stats

    def kernel(name, local_logits, **kwargs):
        result = original(name, local_logits, **kwargs)
        assert result is not None, f"{name} failed"
        kernels.append(name)
        return result

    monkeypatch.setattr(_impl, "_try_triton_stats", kernel)
    outputs, hidden = _run_head(
        monkeypatch, logits.cuda(), requests, positions, path="triton"
    )
    fused = top_k is not None and top_k <= 10
    assert kernels == ["local_topk_stats" if fused else "local_logsumexp_stats"] * 2
    loss = _loss(outputs, terms)
    weights = torch.cat([weights for _, weights in terms], dim=-1)
    if top_k is not None:
        top = outputs[0].top_k
        assert top is not None
        top_weights = _topk_weights(rows, top_k, seed=8)
        loss = loss - (top_weights.cuda() * top.logprobs).sum()
        terms = [*terms, (top.tokens.cpu(), top_weights)]
        weights = torch.cat((weights, top_weights), dim=-1)
    loss.backward()
    assert hidden.grad is not None
    _, reference = _reference(logits, terms)
    _assert_fp32_combined(hidden.grad.cpu(), reference, weights)


class _AllReduceSum(torch.autograd.Function):
    """Megatron's reduce-from-tensor-parallel region: sum, identity backward."""

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.clone()
        dist.all_reduce(output)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0]


def _all_reduce_max(tensor):
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.MAX)
    return output


def test_tensor_parallel_shards_add_only_owned_target_gradients(tmp_path):
    spawn_and_join(
        _tensor_parallel_worker,
        (f"file://{tmp_path / 'tp'}",),
        timeout=120,
        failure="tensor-parallel head workers did not finish",
    )


def _tensor_parallel_worker(rank, rendezvous):
    with process_group(rank, rendezvous, world_size=2, timeout=90):
        local = VOCAB // 2
        logits, requests, positions, terms = _target_case("shared", ROWS, VOCAB, seed=9)
        # Both ranks own some rows' targets; the 2-label rows straddle shards.
        assert (terms[0][0] < local).any() and (terms[0][0] >= local).any()
        shard = logits[:, rank * local : (rank + 1) * local]
        _, reference = _reference(logits, terms)
        weights = torch.cat([weights for _, weights in terms], dim=-1)
        for path in ("fallback", "bounded"):
            with pytest.MonkeyPatch.context() as monkeypatch:
                outputs, hidden = _run_head(
                    monkeypatch, shard, requests, positions, path=path, tp_rank=rank
                )
                _loss(outputs, terms).backward()
            assert hidden.grad is not None
            shards = [torch.empty_like(hidden.grad) for _ in range(2)]
            dist.all_gather(shards, hidden.grad.contiguous())
            _assert_fp32_combined(torch.cat(shards, dim=-1), reference, weights)
