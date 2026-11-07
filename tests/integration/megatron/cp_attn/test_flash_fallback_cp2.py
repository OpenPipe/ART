"""Two-rank FLASH contracts through real CP plans, transport and backward."""

from contextlib import nullcontext
from datetime import timedelta
import math
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from tests.unit.trainer_rank_test_support import spawn_and_join

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or torch.cuda.get_device_capability()[0] != 9,
        reason="two SM90 CUDA devices required for compiled FLASH CP2",
    ),
]


class _WithLse(torch.autograd.Function):
    """Expose CP's existing LSE tape to a test-only external cotangent.

    The public operator returns only output. Reuse its real forward/backward;
    add the extra LSE cotangent at the existing merge-tape boundary.
    """

    @staticmethod
    def forward(ctx, q, k, v, sink, state, scale):
        from art.megatron.context_parallel import executor as cp

        original = cp._run_context_parallel_forward_recorded
        captured = []

        def record(**kwargs):
            result = original(**kwargs)
            captured.append(result[2])
            return result

        with patch.object(cp, "_run_context_parallel_forward_recorded", record):
            out = cp.ArtContextParallelFn.forward(
                ctx, q, k, v, sink, state, scale, True, True, None, ()
            )
        return out, captured[0].detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        from art.megatron.context_parallel import executor as cp

        grad_out, grad_lse = grad_outputs
        original = cp._merge_stage_output_grads_from_tape

        def add_cotangent(*, grad_lse_flat=None, **kwargs):
            combined = grad_lse if grad_lse_flat is None else grad_lse_flat + grad_lse
            return original(grad_lse_flat=combined, **kwargs)

        with patch.object(cp, "_merge_stage_output_grads_from_tape", add_cotangent):
            gradients = cp.ArtContextParallelFn.backward(ctx, grad_out)
        return gradients[:6]


def _state(rank, roots, dim):
    from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
    from art.megatron.context_parallel.runtime import _build_runtime_plan
    from art.megatron.context_parallel.types import (
        ArtContextParallelState,
        ContextParallelConfig,
        CpBlockMaskVariant,
        ParallelTopology,
    )
    from art.megatron.flex_attn.compiled import flash_sparse_block_size_for_head_dim

    device = torch.device("cuda", rank)
    groups = torch.cat(
        [torch.full((length,), root) for root, length in enumerate(roots)]
    )
    block_size = flash_sparse_block_size_for_head_dim(
        head_dim=dim, head_dim_v=dim, device=device
    )
    config = ContextParallelConfig(attention_sparse_block_size=block_size)
    spec = build_prefix_tree_attention_spec(
        group_ids=groups[None], parent_ids=groups[None]
    )
    plan = _build_runtime_plan(
        spec,
        topology=ParallelTopology(cp=2),
        config=config,
        original_seq_len=groups.numel(),
    )[rank]
    state = ArtContextParallelState(
        rank_plan=plan,
        cp_group=dist.group.WORLD,
        config=config,
        group_ids=groups,
        parent_ids=groups,
        input_pos=torch.arange(groups.numel()),
        block_mask_variants=(CpBlockMaskVariant(block_size=block_size),),
    )
    from art.megatron.context_parallel.executor import (
        prepare_context_parallel_execution_state,
    )

    prepare_context_parallel_execution_state(state=state, device=device)
    indices = torch.tensor(
        [
            i
            for start, end, _ in plan.token_layout_index.ownership_ranges_by_rank[rank]
            for i in range(start, end)
        ],
        device=device,
        dtype=torch.long,
    )
    return state, indices, groups.to(device)


def _reference(values, groups, scale):
    q, k, v, sink = values
    query = q[:, 0].transpose(0, 1)
    key = k[:, 0].transpose(0, 1).repeat_interleave(2, 0)
    value = v[:, 0].transpose(0, 1).repeat_interleave(2, 0)
    scores = (query @ key.transpose(-1, -2)) * scale
    positions = torch.arange(groups.numel(), device=q.device)
    visible = groups[:, None].eq(groups[None, :]) & (
        positions[:, None] >= positions[None, :]
    )
    scores = scores.masked_fill(~visible, -torch.inf)
    lse = scores.logsumexp(-1)
    with_sink = torch.cat(
        [scores, sink[:, None, None].expand(-1, scores.shape[1], 1)], -1
    )
    out = with_sink.softmax(-1)[..., :-1] @ value
    return out.transpose(0, 1)[:, None], lse


def _objective(out, lse):
    # Fixed global denominators: summing both rank objectives gives the full
    # reference loss, including K/V contributions from remote query owners.
    return out.float().square().sum() / (256 * 4 * out.shape[-1]) + 0.1 * lse.sum() / (
        256 * 4
    )


def _stage_probe(original, records, **kwargs):
    from art.megatron.context_parallel import executor as cp

    out, lse = original(**kwargs)
    q, k = kwargs["q_stage"], kwargs["k_stage"]
    stage, state = kwargs["stage_plan"], kwargs["state"]
    # Executor transports head-major stages; use its real prepared sparse mask.
    block_size = cp._stage_sparse_block_size(q, kwargs["v_stage"])
    execution = cp._resolve_stage_execution_spec(
        stage_plan=stage, state=state, block_size=block_size
    )
    mask = cp._get_prepared_stage_block_mask(
        stage_plan=stage,
        state=state,
        device=q.device,
        execution_spec=execution,
        block_size=block_size,
        sliding_window=None,
    )
    qi = torch.arange(q.shape[1], device=q.device)[:, None]
    ki = torch.arange(k.shape[1], device=q.device)[None, :]
    visible = mask.mask_mod(
        torch.tensor(0, device=q.device), torch.tensor(0, device=q.device), qi, ki
    )
    scores = (
        q.detach().double()
        @ k.detach().double().repeat_interleave(2, 0).transpose(-1, -2)
    ) * kwargs["scale"]
    expected_lse = scores.masked_fill(~visible, -torch.inf).logsumexp(-1)
    records.append((lse.detach(), expected_lse, stage.is_local_stage))
    return out, lse


def _measure(values, state, scale, recompute):
    inputs = [value.detach().clone().requires_grad_() for value in values]
    fn = lambda *xs: _WithLse.apply(*xs, state, scale)
    out, lse = checkpoint(fn, *inputs, use_reentrant=True) if recompute else fn(*inputs)
    lse.retain_grad()
    loss = _objective(out, lse)
    loss.backward()
    gradients = [value.grad.detach().clone() for value in inputs]
    # Sink is replicated; compare its summed gradient with the global reference.
    dist.all_reduce(gradients[3])
    return out.detach(), lse.detach(), gradients, loss.detach(), lse.grad.detach()


def _worker(rank, rendezvous, dim, deterministic, fallback):
    from art.megatron.context_parallel import executor as cp
    from art.megatron.flex_attn import compiled

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    original_policy = torch.are_deterministic_algorithms_enabled()
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_float32_matmul_precision("highest")
        torch._dynamo.reset()
        validations = []
        for roots in [(128, 128), (256,)]:
            # The requested independent roots may be entirely local at CP2.
            # A second, single-root case guarantees a genuine remote merge.
            state, indices, groups = _state(rank, roots, dim)
            generator = torch.Generator().manual_seed(902)
            global_values = [
                (torch.randn(256, 1, heads, dim, generator=generator) * 0.2).to(
                    device=indices.device, dtype=torch.bfloat16
                )
                for heads in (4, 2, 2)
            ]
            sink = torch.linspace(-0.4, 0.3, 4, device=indices.device)
            reference_inputs = [
                x.detach().double().requires_grad_() for x in (*global_values, sink)
            ]
            expected_out, expected_lse = _reference(reference_inputs, groups, dim**-0.5)
            expected_loss = _objective(expected_out, expected_lse)
            expected_grads = torch.autograd.grad(expected_loss, reference_inputs)
            local_values = [
                value.index_select(0, indices) for value in global_values
            ] + [sink]
            stage_records = []
            original_stage = cp._run_stage_attention

            def probe(**kwargs):
                return _stage_probe(original_stage, stage_records, **kwargs)

            counters["art"]["flex_flash_eager_fallback"] = 0
            with patch.object(cp, "_run_stage_attention", probe):
                warm = _measure(local_values, state, dim**-0.5, False)
                warm_count = counters["art"]["flex_flash_eager_fallback"]
                counters["art"]["flex_flash_eager_fallback"] = 0
                raw = compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
                scope = (
                    patch.object(cp, "sparse_compiled_flex_attention", raw)
                    if fallback and rank == 0
                    else nullcontext()
                )
                with scope:
                    measured = _measure(local_values, state, dim**-0.5, True)
                fallback_count = counters["art"]["flex_flash_eager_fallback"]
            target = (
                expected_out.detach().index_select(0, indices),
                expected_lse.detach().index_select(1, indices),
                [g.detach().index_select(0, indices) for g in expected_grads[:3]]
                + [expected_grads[3].detach()],
            )
            # Defer assertions until every model collective and final vote finish.
            validations.append(
                (
                    warm,
                    measured,
                    target,
                    stage_records,
                    warm_count,
                    fallback_count,
                    len(roots),
                )
            )
        completed = torch.ones((), device=rank)
        dist.all_reduce(completed)
        assert completed.item() == 2
        for (
            warm,
            measured,
            target,
            records,
            warm_count,
            fallback_count,
            nroots,
        ) in validations:
            assert warm_count == 0
            assert (
                (fallback_count > 0)
                if fallback and rank == 0
                else (fallback_count == 0)
            )
            if nroots == 1:
                # Only rank 1 needs the remote half for causal attention.
                assert any(not local for _, _, local in records) == (rank == 1)
            for actual_lse, wanted_lse, _ in records:
                torch.testing.assert_close(
                    actual_lse.double(), wanted_lse, atol=2e-3, rtol=2e-3
                )
            for out, lse, gradients, loss, dlse in (warm, measured):
                assert dlse.isfinite().all() and dlse.abs().max() > 0
                assert loss.isfinite()
                # BF16 stage outputs round separately. LSE stays FP32, so its
                # much tighter bound detects the incorrect logarithm base.
                torch.testing.assert_close(
                    out.double(), target[0], atol=8e-3, rtol=6e-2
                )
                torch.testing.assert_close(
                    lse.double(), target[1], atol=2e-3, rtol=2e-3
                )
                for actual, wanted in zip(gradients, target[2], strict=True):
                    assert actual.isfinite().all()
                    relative = (
                        actual.double() - wanted
                    ).norm() / wanted.norm().clamp_min(1e-12)
                    assert relative <= 0.02, f"BF16 gradient rel-L2={relative.item()}"
            if not fallback:
                torch.testing.assert_close(warm[0], measured[0], atol=0, rtol=0)
                torch.testing.assert_close(warm[1], measured[1], atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(original_policy)
        torch.set_float32_matmul_precision(original_precision)
        dist.destroy_process_group()


# Stock non-deterministic compiled FLASH backward returns uncorrelated BF16
# gradients at head_dim 64 before any eager fallback runs (warm pass); see
# https://github.com/OpenPipe/ART/issues/1179. Strict: an XPASS must remove this.
_FLASH_CP2_CASES = [
    pytest.param(
        False,
        64,
        id="False-64",
        marks=pytest.mark.xfail(
            strict=True,
            raises=torch.multiprocessing.ProcessRaisedException,
            reason=(
                "non-deterministic compiled FLASH backward at head_dim 64 "
                "(OpenPipe/ART#1179)"
            ),
        ),
    ),
    pytest.param(False, 256, id="False-256"),
    pytest.param(True, 64, id="True-64"),
    pytest.param(True, 256, id="True-256"),
]


@pytest.mark.parametrize(("deterministic", "dim"), _FLASH_CP2_CASES)
def test_compiled_flash_cp2_lse_and_gradients(dim, deterministic, tmp_path):
    spawn_and_join(
        _worker,
        (f"file://{tmp_path}/compiled", dim, deterministic, False),
        timeout=600,
        failure="Compiled FLASH CP2 validation did not complete",
    )


@pytest.mark.parametrize(("deterministic", "dim"), _FLASH_CP2_CASES)
def test_mixed_flash_fallback_cp2_lse_and_gradients(dim, deterministic, tmp_path):
    spawn_and_join(
        _worker,
        (f"file://{tmp_path}/fallback", dim, deterministic, True),
        timeout=600,
        failure="Mixed cache-hit/fallback FLASH CP2 did not complete",
    )
