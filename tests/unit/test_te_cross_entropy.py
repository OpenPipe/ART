import sys
import textwrap
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron import te_cross_entropy


@pytest.fixture
def kernel(monkeypatch):
    module = ModuleType("transformer_engine.common.triton.cross_entropy")
    source = (
        "prefix\n" + te_cross_entropy._GRADIENT + te_cross_entropy._TARGET + "suffix"
    )
    kernel = SimpleNamespace(src=source, device_caches={0: "compiled original"})
    updates = []

    def update_source(source):
        updates.append(source)
        kernel.src = source

    kernel._unsafe_update_src = update_source
    kernel.updates = updates
    setattr(module, "cross_entropy_kernel", kernel)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return kernel


def test_patch_preserves_kernel_identity_and_invalidates_compiled_source(kernel):
    te_cross_entropy.patch_te_cross_entropy()
    te_cross_entropy.patch_te_cross_entropy()
    assert len(kernel.updates) == 1
    assert kernel.device_caches == {}
    assert kernel.src.startswith("prefix\n") and kernel.src.endswith("suffix")
    assert "X_y = tl.load" not in kernel.src


@pytest.mark.parametrize("source", ["changed upstream", te_cross_entropy._GRADIENT * 2])
def test_patch_rejects_unknown_source_without_modifying_kernel(kernel, source):
    kernel.src = source
    with pytest.raises(RuntimeError, match="Unsupported Transformer Engine"):
        te_cross_entropy.patch_te_cross_entropy()
    assert kernel.updates == []
    assert kernel.device_caches == {0: "compiled original"}


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduce_loss", [False, True])
def test_patched_gradient_combines_target_before_rounding(
    kernel, rank, smoothing, reduce_loss
):
    te_cross_entropy.patch_te_cross_entropy()
    gradient = kernel.src.removeprefix("prefix\n").removesuffix("suffix")
    probability = torch.tensor([0.999, 0.001], dtype=torch.float32)
    namespace: dict[str, Any] = dict(
        tl=SimpleNamespace(exp=torch.exp, where=torch.where),
        X_block=probability.log(),
        m=0.0,
        d=1.0,
        eps=smoothing / 4,
        X_offsets=torch.arange(2),
        rank=rank,
        n_cols=2,
        y=2,
        label_smoothing=smoothing,
        reduce_loss=reduce_loss,
        n_non_ignore=3,
    )
    exec(textwrap.dedent(gradient), namespace)
    expected = probability - smoothing / 4
    if rank == 1:
        expected[0] -= 1 - smoothing
    if reduce_loss:
        expected /= 3
    actual = namespace["X_block"].bfloat16()
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    if rank == 1 and smoothing == 0:
        assert actual[0] != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE cross entropy needs CUDA")
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduce_loss", [False, True])
def test_te_target_gradients_match_fp64(smoothing, reduce_loss):
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    te_cross_entropy.patch_te_cross_entropy()
    rows, vocab = 128, 4099
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(rows, vocab, generator=generator, dtype=torch.float64) * 2
    targets = torch.randint(vocab, (rows,), generator=generator)
    row_ids = torch.arange(rows)
    probabilities = 1 - torch.logspace(-1, -5, rows, dtype=torch.float64)
    logits[row_ids, targets] = -torch.inf
    logits[row_ids, targets] = torch.logsumexp(logits, -1) + torch.log(
        probabilities / (1 - probabilities)
    )
    logits = logits.bfloat16().cuda()
    targets = targets.cuda()
    if not reduce_loss:
        targets[-1] = -100
    reference = logits.double().requires_grad_()
    reference_loss = torch.nn.functional.cross_entropy(
        reference, targets, label_smoothing=smoothing, reduction="none"
    )
    if reduce_loss:
        reference_loss = reference_loss.mean()
    weights = (
        torch.ones_like(reference_loss)
        if reduce_loss
        else torch.linspace(-2, 2, rows, device="cuda", dtype=torch.float64)
    )
    (reference_loss * weights).sum().backward()
    candidate = logits.clone().requires_grad_()
    loss = parallel_cross_entropy(
        candidate[None], targets[None], smoothing, reduce_loss
    )
    (loss * weights).sum().backward()
    assert candidate.grad is not None and reference.grad is not None
    assert loss.dtype == torch.float32 and candidate.grad.dtype == torch.bfloat16
    torch.testing.assert_close(
        loss.flatten().double(), reference_loss.flatten(), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        candidate.grad.double(), reference.grad, atol=2e-7, rtol=0.025
    )
    if not reduce_loss:
        assert torch.count_nonzero(candidate.grad[-1]) == 0
    if smoothing == 0:
        assert (candidate.grad[row_ids[:-1], targets[:-1]] != 0).all()
