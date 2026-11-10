"""CPU/gloo attention oracle with controlled asynchronous fetch completion.

Only the attention kernel and, for the exhaustive sweep, repeated transport are
substituted. Production planning, ready-batch selection, merge tape and backward
execute unchanged. Set ART_CP_EXHAUSTIVE_READINESS=1 for all ordered partitions.
"""

from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel import executor as ex
from art.megatron.context_parallel import runtime as rt
from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.comm import KvFetchWork
from art.megatron.context_parallel.types import (
    ArtContextParallelState,
    ContextParallelConfig,
    ParallelTopology,
    TokenRange,
)


def _attention(q, k, v, mask):
    k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
    v = v.repeat_interleave(q.shape[-3] // v.shape[-3], dim=-3)
    logits = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    logits = logits.masked_fill(~mask, float("-inf"))
    nonempty = mask.any(-1, keepdim=True)
    # Do not differentiate logsumexp(-inf, ... -inf): empty rows have zero VJP.
    safe = torch.where(nonempty, logits, torch.zeros_like(logits))
    probs = safe.softmax(-1) * nonempty
    lse = torch.where(nonempty.squeeze(-1), safe.logsumexp(-1), float("-inf"))
    return probs @ v, lse


class ReferenceKernel:
    def __init__(self):
        self.lengths = (0, 0)
        self.masks = {}
        self.empty_rows = 0

    def run(self, q, k, v, *, block_mask, scale, enable_gqa):
        nq, nk = self.lengths
        key = (id(block_mask), nq, nk)
        if key not in self.masks:
            mask = block_mask.mask_mod(
                torch.tensor(0),
                torch.tensor(0),
                torch.arange(nq)[:, None],
                torch.arange(nk)[None, :],
            )
            self.masks[key] = mask
            self.empty_rows += int((~mask.any(-1)).sum())
        output, lse = _attention(
            q[..., :nq, :], k[..., :nk, :], v[..., :nk, :], self.masks[key]
        )
        return (
            torch.nn.functional.pad(output, (0, 0, 0, q.shape[-2] - nq)),
            torch.nn.functional.pad(lse, (0, q.shape[-2] - nq), value=float("-inf")),
        )


@contextmanager
def _reference_kernel():
    original = ex._run_stage_attention
    kernel = ReferenceKernel()

    def run(**kw):
        kernel.lengths = (kw["q_stage"].shape[1], kw["k_stage"].shape[1])
        return original(**kw)

    with (
        patch.object(ex, "FlexAttentionKernel", lambda **kw: kernel),
        patch.object(ex, "_run_stage_attention", run),
    ):
        yield kernel


def _indices(ranges):
    return [i for r in ranges if r is not None for i in range(r.start, r.end)]


def _plans(layout_name):
    spec = build_prefix_tree_attention_spec(
        group_ids=torch.zeros(1, 16, dtype=torch.long),
        parent_ids=torch.zeros(1, 16, dtype=torch.long),
    )
    if layout_name == "planner":
        return spec, rt._build_runtime_plan(
            spec,
            topology=ParallelTopology(cp=8),
            config=ContextParallelConfig(),
            original_seq_len=16,
        )
    chunks = tuple(TokenRange(i, i + 1) for i in range(16))
    owners = tuple(range(8)) + tuple(reversed(range(8)))
    layout = rt._build_runtime_token_layout_index(
        chunk_ranges=chunks, owners=owners, cp_size=8
    )
    # Eight globally ordered exchange waves; each rank computes seven remote
    # stages plus its local stage. Empty peer splits still enter every wave.
    plans = tuple(
        rt._build_rank_runtime_plan(
            row_spec=spec.rows[0],
            chunk_ranges=chunks,
            owners=owners,
            wave_assignment=owners,
            token_layout_index=layout,
            cp_size=8,
            original_seq_len=16,
            target_rank=rank,
            block_size=128,
        )
        for rank in range(8)
    )
    return spec, plans


def _inputs():
    generator = torch.Generator().manual_seed(417)
    q = torch.randn(16, 1, 16, 256, generator=generator) * 0.1
    k = torch.randn(16, 1, 2, 256, generator=generator) * 0.1
    v = torch.randn(16, 1, 2, 256, generator=generator) * 0.1
    go = torch.randn(q.shape, generator=generator) * 0.1
    return q, k, v, go


def _oracle(q, k, v, grad):
    leaves = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    out, _ = _attention(
        *[x.permute(1, 2, 0, 3) for x in leaves],
        torch.ones(16, 16, dtype=torch.bool).tril(),
    )
    out = out.permute(2, 0, 1, 3)
    grads = torch.autograd.grad(out, leaves, grad)
    return out.detach(), tuple(x.detach() for x in grads)


class Completion:
    def __init__(self, batches):
        self.batches = [set(batch) for batch in batches]
        self.done = set()
        self.cursor = 0
        self.observed = []

    def ready(self, index):
        return index in self.done or (
            self.cursor < len(self.batches) and index in self.batches[self.cursor]
        )

    def wait(self, index):
        self.done.add(index)
        if self.cursor < len(self.batches) and self.batches[self.cursor] <= self.done:
            self.cursor += 1


class Handle:
    def __init__(self, completion, index):
        self.completion, self.index = completion, index

    def is_completed(self):
        return self.completion.ready(self.index)

    def wait(self):
        self.completion.wait(self.index)


def _partitions(items):
    """Every productive ready batch, with plan order inside a simultaneous batch."""
    if not items:
        yield ()
        return
    first, *rest = items
    for groups in _partitions(rest):
        for pos in range(len(groups) + 1):
            yield groups[:pos] + ((first,),) + groups[pos:]
        for pos in range(len(groups)):
            yield (
                groups[:pos]
                + (tuple(sorted((first, *groups[pos]))),)
                + groups[pos + 1 :]
            )


class CachedTransport:
    def __init__(self, templates, plan):
        self.templates, self.plan = templates, plan
        self.completion = None
        self.remote = []

    def launch_kv_fetch(self, **kw):
        index = int(kw["label"].split(".stage")[1].split(".")[0])
        original = self.templates[index]
        return KvFetchWork(
            original.packed_buffer,
            original.recv_splits,
            Handle(self.completion, index),
            output_layout=original.output_layout,
        )

    def launch_dkv_reduce(self, **kw):
        index = int(kw["label"].split(".stage")[1].split(".")[0])
        self.remote.append(
            (
                self.plan.stage_plans[index],
                kw["dk_remote"].detach().clone(),
                kw["dv_remote"].detach().clone(),
            )
        )
        return SimpleNamespace(wait_post_process=lambda: None)


def _execute(state, q, k, v, go):
    args = dict(
        query=q,
        key=k,
        value=v,
        state=state,
        scale=1 / 16,
        enable_gqa=True,
        compile_enabled=False,
        sliding_window=None,
        triton_num_stages_2_head_dims=(),
        softmax_offset=None,
    )
    out, accum, lse, records = ex._run_context_parallel_forward_recorded(**args)
    dq, dk, dv, _ = ex._run_context_parallel_backward(
        # Each replay receives the same upstream gradient, with private storage.
        grad_output=go.clone(),
        **args,
        replay_records=records,
        replay_accum_out=accum,
        replay_accum_lse=lse,
    )
    return out, (dq, dk, dv)


def _close(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("layout_name", ["planner", "zigzag"])
def test_cp8_readiness_orders_match_reference(tmp_path, layout_name):
    mp.start_processes(
        _worker,
        args=(f"file://{tmp_path / 'dist'}", str(tmp_path), layout_name),
        nprocs=8,
        join=True,
        start_method="spawn",
    )


def _worker(rank, init_method, output, layout_name):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=8,
        timeout=timedelta(minutes=10),
    )
    try:
        spec, plans = _plans(layout_name)
        plan = plans[rank]
        indices = _indices(plan.local_row_ranges)
        assert indices == (
            [rank, 15 - rank] if layout_name == "zigzag" else [2 * rank, 2 * rank + 1]
        )
        fullq, fullk, fullv, fullgo = _inputs()
        q, k, v = [x[indices].detach().requires_grad_() for x in (fullq, fullk, fullv)]
        assert q.shape == (2, 1, 16, 256) and k.shape == v.shape == (2, 1, 2, 256)
        go = fullgo[indices]
        state = ArtContextParallelState(
            plan,
            dist.group.WORLD,
            ContextParallelConfig(),
            torch.zeros(16, dtype=torch.long),
            torch.zeros(16, dtype=torch.long),
            torch.arange(16),
        )
        ex.prepare_context_parallel_execution_state(
            state=state, device=torch.device("cpu")
        )
        expected_out, expected_global = _oracle(fullq, fullk, fullv, fullgo)
        active = tuple(
            s.stage_index
            for s in plan.stage_plans
            if not s.is_local_stage and s.q_len and s.k_len and s.slices
        )
        assert len(active) == (7 if layout_name == "zigzag" else int(rank > 0))
        templates = {}
        native = ex._COMMUNICATOR
        original = native.launch_kv_fetch

        def capture(**kw):
            work = original(**kw)
            templates[int(kw["label"].split(".stage")[1].split(".")[0])] = work
            return work

        with (
            _reference_kernel() as kernel,
            patch.object(native, "launch_kv_fetch", capture),
        ):
            out, grads = _execute(state, q, k, v, go)
        _close(out, expected_out[indices])
        for grad, target in zip(grads, expected_global):
            _close(grad, target[indices])
        partial_go = torch.zeros_like(fullgo)
        partial_go[indices] = go
        _, expected_partials = _oracle(fullq, fullk, fullv, partial_go)
        transport = CachedTransport(templates, plan)
        batches = (
            _partitions(active)
            if os.environ.get("ART_CP_EXHAUSTIVE_READINESS") == "1"
            else [
                tuple((i,) for i in active),
                tuple((i,) for i in reversed(active)),
                (active,),
                (active[::2], active[1::2]),
            ]
        )
        canonical = None
        count = 0
        original_ready = ex._ready_remote_stage_batch

        def ready(**kw):
            result = original_ready(**kw)
            transport.completion.observed.append(tuple(s.stage_index for s in result))
            return result

        with (
            _reference_kernel() as kernel,
            patch.object(ex, "_COMMUNICATOR", transport),
            patch.object(ex, "_ready_remote_stage_batch", ready),
        ):
            for batch in batches:
                transport.completion = Completion(batch)
                transport.remote = []
                out, grads = _execute(state, q, k, v, go)
                _close(out, expected_out[indices])
                _close(grads[0], expected_partials[0][indices])
                assert transport.completion.observed == [
                    tuple(sorted(group)) for group in batch
                ]
                dks = torch.zeros_like(fullk)
                dvs = torch.zeros_like(fullv)
                dks[indices] += grads[1]
                dvs[indices] += grads[2]
                for stage, dk, dv in transport.remote:
                    where = _indices(stage.global_k_ranges)
                    if where:
                        dks[where] += dk.permute(1, 0, 2)[:, None]
                        dvs[where] += dv.permute(1, 0, 2)[:, None]
                for actual, wanted in zip((dks, dvs), expected_partials[1:]):
                    _close(actual, wanted)
                current = (out.detach(), grads[0].detach(), dks, dvs)
                if canonical is None:
                    canonical = tuple(x.clone() for x in current)
                else:
                    for actual, wanted in zip(current, canonical):
                        _close(actual, wanted)
                count += 1
                if count % 1000 == 0:
                    print(f"rank={rank} schedules={count}", flush=True)
        if layout_name == "zigzag" and rank == 0:
            assert kernel.empty_rows > 0
        assert canonical is not None
        for actual, wanted in zip(canonical[2:], expected_global[1:]):
            dist.all_reduce(actual)
            _close(actual, wanted)
        assert not torch.cuda.is_initialized()
        (Path(output) / f"rank-{rank}.txt").write_text(
            f"schedules={count} empty_rows={kernel.empty_rows}\n"
        )
    finally:
        dist.destroy_process_group()


def test_readiness_partition_coverage():
    partitions = list(_partitions(tuple(range(7))))
    assert len(partitions) == len(set(partitions)) == 47293
    assert {len(p) for p in partitions} == set(range(1, 8))
    assert all(
        sorted(i for group in p for i in group) == list(range(7)) for p in partitions
    )


def test_ready_remote_waits_when_none_complete():
    pending = [SimpleNamespace(stage_index=i) for i in range(3)]
    complete = set()
    waited = []

    class Work:
        def __init__(self, index):
            self.index = index

        def is_completed(self):
            return self.index in complete

        def wait(self):
            waited.append(self.index)
            complete.update((0, 2))

    result = ex._ready_remote_stage_batch(
        pending_stages=pending,
        fetch_works_by_stage_index={i: Work(i) for i in range(3)},
    )
    assert waited == [0]
    assert [s.stage_index for s in result] == [0, 2]
