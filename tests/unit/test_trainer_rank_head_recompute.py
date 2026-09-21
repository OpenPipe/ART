from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import sys
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import ForwardInput, TrainerRank, _impl


class _Head(torch.nn.Module):
    sequence_parallel = False

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def forward(self, hidden, *, weight=None, runtime_gather_output=None):
        return hidden @ (self.weight if weight is None else weight).T, None


def _direct(function, *args, use_reentrant=False, **kwargs):
    return function(*args, **kwargs)


def _patch_local_head(monkeypatch):
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 16)
    monkeypatch.setattr(_impl, "_language_model", lambda model: model)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda value: value)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda value: value)
    monkeypatch.setattr(_impl, "_vocab_range", lambda logits: (0, logits.shape[-1]))
    monkeypatch.setattr(
        _impl,
        "_vocab_parallel_topk_from_local",
        lambda values, tokens, *, k, log_z, vocab_start: _impl.TopK(
            values[:, :k] - log_z[:, None], tokens[:, :k]
        ),
    )
    monkeypatch.setattr(
        TrainerRank, "_gather_tensor_parallel_logits", lambda self, value: value
    )


@pytest.mark.parametrize("top_k", (None, 3, 12))
@pytest.mark.parametrize("include_logits", (False, True))
@pytest.mark.parametrize("multi_target", (False, True))
def test_recomputed_head_preserves_outputs_and_arbitrary_loss_gradients(
    monkeypatch, top_k, include_logits, multi_target
):
    _patch_local_head(monkeypatch)
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(65, 7, generator=generator) / 3
    hidden = torch.randn(41, 7, generator=generator)
    positions = (torch.arange(25), torch.arange(10, 41))
    requests = []
    for rows in positions:
        shape = (len(rows), 2) if multi_target else (len(rows),)
        labels = torch.randint(0, 65, shape, generator=generator)
        labels[::7] = -100
        requests.append(
            ForwardInput(
                input_tokens=torch.ones(len(rows), dtype=torch.long),
                target_tokens=labels,
                top_k=top_k,
                logits=include_logits,
                hidden_states=True,
            )
        )

    def run(recompute):
        model = SimpleNamespace(
            output_layer=_Head(weight),
            vocab_size=65,
            share_embeddings_and_output_weights=False,
            _scale_logits=lambda value: value,
        )
        trainer = object.__new__(TrainerRank)
        trainer.runtime = SimpleNamespace(model=[model])
        value = hidden.clone().requires_grad_()
        saved = []

        def pack(tensor):
            saved.append(
                (
                    tuple(tensor.shape),
                    tensor.untyped_storage().data_ptr(),
                    tensor.untyped_storage().nbytes(),
                )
            )
            return tensor

        with (
            nullcontext()
            if recompute
            else patch("torch.utils.checkpoint.checkpoint", _direct)
        ):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs = trainer._project_head(
                    [trainer._forward_item(request) for request in requests],
                    SimpleNamespace(
                        positions_by_item=positions,
                        source_positions_by_item=tuple(
                            torch.arange(len(rows)) for rows in positions
                        ),
                    ),
                    value,
                )
            tensors = [output.target_logprobs for output in outputs]
            tensors += [output.hidden_states for output in outputs]
            tensors += [
                output.top_k.logprobs for output in outputs if output.top_k is not None
            ]
            tensors += [
                output.logits for output in outputs if output.logits is not None
            ]
            loss = sum(
                tensor.sin().sum() + tensor.square().mean() for tensor in tensors
            )
            loss += tensors[0].mean() * tensors[1].mean()
            loss.backward()
        return (
            [tensor.detach() for tensor in tensors],
            [output.top_k.tokens for output in outputs if output.top_k is not None],
            value.grad,
            model.output_layer.weight.grad,
            saved,
        )

    baseline = run(False)
    recomputed = run(True)
    for actual, expected in zip(recomputed[:4], baseline[:4], strict=True):
        torch.testing.assert_close(actual, expected)
    # Full-vocabulary tensors saved by normalization must disappear. Explicit
    # logits outputs remain part of the public result and are outside this claim.
    assert any(shape[-1:] == (65,) and len(shape) == 2 for shape, _, _ in baseline[4])
    assert not any(
        shape[-1:] == (65,) and len(shape) == 2 for shape, _, _ in recomputed[4]
    )


@pytest.mark.parametrize("cp_size,dp_size", ((2, 1), (4, 1), (2, 2)))
def test_context_parallel_outputs_match_full_sequence(cp_size, dp_size, tmp_path):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _context_parallel_worker,
        args=(cp_size, dp_size, f"file://{tmp_path / 'cp'}", "gloo"),
        nprocs=cp_size * dp_size,
        join=True,
    )


@pytest.mark.parametrize("cp_size", (2, 4))
def test_context_parallel_outputs_cuda(cp_size, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(f"requires {cp_size} CUDA devices")
    pytest.importorskip("megatron.core")
    mp.spawn(
        _context_parallel_worker,
        args=(cp_size, 1, f"file://{tmp_path / 'cp'}", "nccl"),
        nprocs=cp_size,
        join=True,
    )


def _context_parallel_worker(rank, cp_size, dp_size, init_method, backend):
    from megatron.core import parallel_state as ps

    device = torch.device("cpu" if backend == "gloo" else f"cuda:{rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=cp_size * dp_size,
        timeout=timedelta(seconds=90),
    )
    try:
        cp_groups = [
            dist.new_group(list(range(dp * cp_size, (dp + 1) * cp_size)))
            for dp in range(dp_size)
        ]
        dp_groups = [
            dist.new_group(list(range(cp, cp_size * dp_size, cp_size)))
            for cp in range(cp_size)
        ]
        cp_rank, dp_rank = rank % cp_size, rank // cp_size
        cp_group, dp_group = cp_groups[dp_rank], dp_groups[cp_rank]
        with pytest.MonkeyPatch.context() as monkeypatch:
            _patch_local_head(monkeypatch)
            monkeypatch.setattr(ps, "get_context_parallel_world_size", lambda: cp_size)
            monkeypatch.setattr(
                ps,
                "get_data_parallel_group",
                lambda *, with_context_parallel: (
                    dist.group.WORLD if with_context_parallel else dp_group
                ),
            )
            monkeypatch.setattr(ps, "get_tensor_model_parallel_group", lambda **_: None)
            for mode in (
                "hidden",
                "targets",
                "all",
                "logits",
                "head_only",
                "frozen",
                "no_grad",
            ):
                _check_context_parallel_case(
                    cp_rank, dp_rank, cp_size, dp_size, cp_group, device, mode
                )
    finally:
        dist.destroy_process_group()


def _check_context_parallel_case(
    cp_rank, dp_rank, cp_size, dp_size, cp_group, device, mode
):
    tokens = [torch.tensor(row) for row in ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 8, 9])]
    packed = prefix_tree_pack(tokens, max_depth=1)
    generator = torch.Generator(device=device).manual_seed(911)
    features = torch.randn(9, 3, generator=generator, device=device) + dp_rank / 5
    decoder_weight = torch.randn(3, 5, generator=generator, device=device)
    head_weight = torch.randn(11, 5, generator=generator, device=device)
    probe_weight = torch.randn(5, 2, generator=generator, device=device)
    requests = []
    for index, row in enumerate(tokens):
        labels = torch.stack((row, row.roll(1)), dim=1)
        labels[::2] = -100
        if index == 1:
            labels.fill_(-100)
        requests.append(
            ForwardInput(
                input_tokens=row,
                target_tokens=labels if mode not in ("hidden", "logits") else None,
                hidden_states=mode not in ("targets", "logits"),
                logits=mode not in ("hidden", "targets"),
                top_k=3 if mode not in ("hidden", "targets", "logits") else None,
            )
        )
    # Unequal, reversed ownership with a shared prefix; CP4 rank 3 is empty.
    owners = torch.tensor([0, 1, 1, 2, 0, 1, 1, 2, 0]) % cp_size
    rows = torch.nonzero(owners == cp_rank).flatten().flip(0)
    dispatched = torch.cat((rows, torch.tensor([-1])))
    pairs = [
        _impl._local_position_pairs(dispatched[None], positions)
        for positions in packed.positions_by_sequence
    ]

    def run(local):
        decoder = torch.nn.Parameter(
            decoder_weight.clone(), requires_grad=mode not in ("frozen", "head_only")
        )
        head = _Head(head_weight)
        head.weight.requires_grad_(mode != "frozen")
        probe = torch.nn.Parameter(probe_weight.clone(), requires_grad=mode != "frozen")
        trainer = object.__new__(TrainerRank)
        trainer._skipped_forward_waves = {}
        trainer.runtime = SimpleNamespace(
            model=[
                SimpleNamespace(
                    output_layer=head,
                    vocab_size=11,
                    share_embeddings_and_output_weights=False,
                    _scale_logits=lambda value: value,
                )
            ]
        )
        trainer._tag_custom_parameters((probe,))
        with torch.set_grad_enabled(mode != "no_grad"):
            hidden = features @ decoder
            if local:
                hidden = torch.cat((hidden[rows], hidden.new_zeros((1, 5))))
            trainer._decoder_hidden = lambda _: hidden
            trainer._gather_sequence_parallel_hidden = lambda value: value
            outputs = trainer._forward_packed(
                [trainer._forward_item(request) for request in requests],
                SimpleNamespace(
                    positions_by_item=(
                        tuple(pair[0] for pair in pairs)
                        if local
                        else packed.positions_by_sequence
                    ),
                    source_positions_by_item=(
                        tuple(pair[1] for pair in pairs)
                        if local
                        else tuple(torch.arange(len(row)) for row in tokens)
                    ),
                    context_parallel_group=cp_group if local else None,
                ),
            )
            tensors, terms = [], []
            for output, row in zip(outputs, tokens, strict=True):
                assert not hasattr(output, "positions")
                values = [output.target_logprobs, output.logits, output.hidden_states]
                if output.top_k is not None:
                    values.append(output.top_k.logprobs)
                    tensors.append(output.top_k.tokens)
                for value in values:
                    if value is not None:
                        assert value.shape[0] == len(row)
                        tensors.append(value)
                        terms.append(value.mean().square() + value.square().mean())
                if output.hidden_states is not None:
                    # The original failing caller expression needs no CP mapping.
                    selected = output.hidden_states[1:][(row[1:] % 2 == 1).to(device)]
                    terms.append((selected @ probe).square().mean())
            loss = torch.stack(terms).sum()
            if mode not in ("frozen", "no_grad"):
                loss.backward()
        return trainer, tensors, loss.detach(), (decoder, head.weight, probe)

    reference = run(False)
    actual = run(True)
    for value, expected in zip(actual[1], reference[1], strict=True):
        torch.testing.assert_close(value, expected)
        assert value.requires_grad == expected.requires_grad
    torch.testing.assert_close(actual[2], reference[2])
    expected_loss = reference[2].clone()
    dist.all_reduce(expected_loss)
    expected_loss /= cp_size
    trainer = actual[0]
    trainer.dp_reduce(actual[2])
    torch.testing.assert_close(actual[2], expected_loss)
    count = torch.tensor(sum(len(row) for row in tokens), device=device)
    trainer.dp_reduce(count)
    assert count.item() == dp_size * sum(len(row) for row in tokens)
    if mode in ("frozen", "no_grad"):
        return
    reduced = trainer._reduce_dynamic_grads(actual[3], scale_grads=0.5)
    for grad, param in zip(reduced, reference[3], strict=True):
        expected = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        dist.all_reduce(expected)
        expected *= 0.5 / cp_size
        torch.testing.assert_close(grad, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize(
    "mode", ["target", "logits", "topk", "target_logits", "target_topk_logits"]
)
def test_prior_chunk_references_end_before_next_stats(monkeypatch, grad, mode):
    _patch_local_head(monkeypatch)
    monkeypatch.setattr(_impl, "_HEAD_CHUNK_TOKENS", 4)
    original_local = TrainerRank._local_logits_from_hidden_rows
    original_exp = torch.exp
    has_target, has_logits, has_topk = (
        "target" in mode,
        "logits" in mode,
        "topk" in mode,
    )

    refs, previous_live, observations = [], [], []

    def local(self, model, hidden, **kwargs):
        if len(refs) == 1:
            previous_live.append(refs[0]() is not None)
        value = original_local(self, model, hidden, **kwargs)
        refs.append(weakref.ref(value))
        return value

    def exp(subtraction):
        value = original_exp(subtraction)
        if len(refs) == 2 and not observations:
            # Inspect only this boundary, after allocation. Do not retain
            # a frame or a tensor between chunks to manufacture overlap.
            frame = sys._getframe().f_back
            while frame.f_code.co_name != "_project_vocab_parallel":
                frame = frame.f_back
            prior = [
                frame.f_locals.get(name)
                for name in ("local_logits", "chunk_logits", "selected_logits")
            ]
            prior = [tensor for tensor in prior if isinstance(tensor, torch.Tensor)]
            assert not prior
            assert refs[0]() is None
            observations.append(True)
        return value

    model = SimpleNamespace(
        output_layer=_Head(torch.arange(85, dtype=torch.bfloat16).reshape(17, 5) / 100),
        vocab_size=17,
        share_embeddings_and_output_weights=False,
        _scale_logits=lambda value: value,
    )
    r = object.__new__(TrainerRank)
    r.runtime = SimpleNamespace(model=[model])
    x = (torch.arange(40, dtype=torch.bfloat16).reshape(8, 5) / 100).requires_grad_(
        grad
    )
    item = ForwardInput(
        input_tokens=torch.arange(8),
        target_tokens=torch.arange(8) if has_target else None,
        top_k=2 if has_topk else None,
        logits=has_logits,
        no_grad=not grad,
    )
    with monkeypatch.context() as patch:
        patch.setattr(TrainerRank, "_local_logits_from_hidden_rows", local)
        patch.setattr(torch, "exp", exp)
        with torch.set_grad_enabled(grad):
            result = r._project_head(
                [r._forward_item(item)],
                SimpleNamespace(
                    positions_by_item=(torch.arange(8),),
                    source_positions_by_item=(torch.arange(8),),
                ),
                x,
            )[0]
        assert previous_live == [False]
        assert observations == ([True] if has_target or has_topk else [])
        outputs = [
            v
            for v in (
                result.target_logprobs,
                result.logits,
                result.top_k.logprobs if result.top_k else None,
            )
            if v is not None
        ]
        if grad:
            sum(v.float().sum() for v in outputs).backward()
            assert x.grad is not None and x.grad.isfinite().all()
            assert model.output_layer.weight.grad is not None
            assert model.output_layer.weight.grad.isfinite().all()
    assert not torch.cuda.is_initialized()
