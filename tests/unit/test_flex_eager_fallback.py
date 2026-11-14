"""CPU math HOP, real CP merge autograd, and mixed-rank cache fallback."""

import ast
from contextlib import nullcontext
from datetime import timedelta
import importlib
import json
import logging
import math
from pathlib import Path
import signal
import time
from typing import Any, cast

import pytest
import torch
from torch._dynamo.utils import counters
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask
from torch.utils.checkpoint import checkpoint

from art.megatron.flex_attn import compiled

if __package__:
    from .trainer_rank_test_support import spawn_and_join
else:
    from trainer_rank_test_support import spawn_and_join


def _cp_functions():
    # CP's module imports optional Megatron annotations. Execute its unchanged
    # autograd/merge definitions so these CPU tests need neither Megatron nor TE.
    source = Path(compiled.__file__).parents[1] / "context_parallel/executor.py"
    names = {
        "_safe_logaddexp",
        "_safe_exp_diff",
        "_stage_merge_values",
        "_stage_merge_backward_values_triton",
        "_stage_merge_backward_values",
        "_StageMergeFn",
        "_apply_softmax_offset",
        "_softmax_offset_view",
        "_softmax_offset_backward_values",
    }
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == names
    namespace: dict[str, Any] = {"torch": torch, "cast": cast}
    unit = ast.parse("from __future__ import annotations")
    unit.body.extend(nodes)
    exec(compile(unit, str(source), "exec"), namespace)
    return namespace


CP = _cp_functions()


@pytest.fixture(autouse=True)
def _cpu_math(monkeypatch):
    # The public API rejects CPU backward before reaching its math HOP, whose
    # autograd is device-independent. Bypass only that device validation.
    flex = importlib.import_module("torch.nn.attention.flex_attention")
    monkeypatch.setattr(flex, "_validate_device", lambda *args: None)
    original, warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    counters["art"]["flex_flash_eager_fallback"] = 0
    counters["art"]["flex_flash_eager_fallback_overflow"] = 0
    monkeypatch.setattr(compiled, "_FLASH_EAGER_FALLBACK_WARNINGS", 0, raising=False)
    monkeypatch.setattr(
        compiled, "_FORCED_FLEX_EXECUTION_STATE", {"required": False, "fallback": False}
    )
    if hasattr(compiled, "_FLASH_EAGER_FALLBACK_KEYS"):
        compiled._FLASH_EAGER_FALLBACK_KEYS.clear()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(original, warn_only=warn)
        torch.set_num_threads(threads)


def _inputs(dtype, gqa, length=5):
    generator = torch.Generator().manual_seed(103)
    heads = 4 if gqa else 2
    return [
        x.requires_grad_()
        for x in (
            torch.randn(1, heads, length, 8, generator=generator).to(dtype) * 0.4,
            torch.randn(1, 2, 8, 8, generator=generator).to(dtype) * 0.4,
            torch.randn(1, 2, 8, 8, generator=generator).to(dtype) * 0.4,
            torch.linspace(-0.4, 0.3, heads),
        )
    ]


def _allowed(q, k):
    # Globally empty query row 0, and stages that are empty on other rows.
    return (q > 0) & (k <= q + 1)


def _partial(entry, backend, q, k, v, offset):
    def mask(b, h, query, key):
        return _allowed(query, key + offset)

    block_mask = create_block_mask(
        mask, B=1, H=None, Q_LEN=q.shape[2], KV_LEN=k.shape[2], device="cpu"
    )
    options = dict(block_mask=block_mask, scale=0.7, enable_gqa=True)
    if entry == "sparse":
        out, lse = compiled._sparse_flex_attention_with_options({"BACKEND": backend})(
            q, k, v, **options
        )
    else:
        fn = (
            compiled._forced_flex_attention_dense
            if entry == "forced_dense"
            else compiled._flex_attention_with_options({"BACKEND": backend})
        )
        out, aux = fn(q, k, v, return_aux=AuxRequest(lse=True), **options)
        lse = aux.lse
        assert lse is not None
    return out[0], compiled.normalize_flex_lse(lse, backend=backend)[0]


def _merged(entry, backend, sink, q, k, v, offset):
    parts = [
        _partial(entry, backend, q, k[:, :, a:b], v[:, :, a:b], a)
        for a, b in [(0, 3), (3, 8)]
    ]
    out, lse = CP["_StageMergeFn"].apply(*parts[0], *parts[1])
    if sink:
        out = CP["_apply_softmax_offset"](out, lse, offset)
    return out, lse


def _reference(sink, q, k, v, offset):
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * 0.7
    query = torch.arange(q.shape[2])[:, None]
    key = torch.arange(k.shape[2])[None, :]
    allowed = _allowed(query, key)[None, None]
    scores = scores.masked_fill(~allowed, -torch.inf)
    valid = allowed.any(-1)
    safe_scores = torch.where(valid[..., None], scores, 0.0)
    lse = safe_scores.logsumexp(-1).masked_fill(~valid, -torch.inf)
    if sink:
        scores = torch.cat(
            [scores, offset[None, :, None, None].expand(1, -1, q.shape[2], 1)], dim=-1
        )
        probs = scores.softmax(-1)[..., :-1]
    else:
        probs = safe_scores.softmax(-1).masked_fill(~valid[..., None], 0.0)
    return (probs.to(q.dtype) @ v)[0], lse[0]


def _loss(out, lse):
    return (
        out.float().square().sum() + 0.13 * lse.masked_fill(~lse.isfinite(), 0.0).sum()
    )


def _check(actual, expected, dtype):
    atol, rtol = (8e-3, 6e-2) if dtype == torch.bfloat16 else (1e-6, 1e-5)
    torch.testing.assert_close(
        actual, expected, check_dtype=False, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("entry", ["sparse", "dense", "forced_dense"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("gqa,sink", [(False, False), (True, True)])
def test_eager_flash_cp_merge_matches_full_attention(
    entry, dtype, deterministic, gqa, sink
):
    torch.use_deterministic_algorithms(deterministic)
    inputs = _inputs(dtype, gqa)
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    expected, expected_lse = _reference(sink, *reference)
    expected_grads = torch.autograd.grad(
        _loss(expected, expected_lse), reference, allow_unused=True
    )
    backends = ["FLASH"] if entry == "forced_dense" else ["FLASH", "TRITON"]
    for backend in backends:
        out, lse = _merged(entry, backend, sink, *inputs)
        actual_grads = torch.autograd.grad(_loss(out, lse), inputs, allow_unused=True)
        _check(out, expected, dtype)
        _check(lse, expected_lse, torch.float32)
        for actual, wanted in zip(actual_grads, expected_grads, strict=True):
            if wanted is None:
                assert actual is None
            else:
                assert actual is not None and actual.isfinite().all()
                _check(actual, wanted, dtype)
        if sink:
            # Exercise CP's explicit sink cotangent path as well as ordinary
            # autograd through the offset formula, including dLSE propagation.
            merged, merged_lse = _merged(entry, backend, False, *inputs)
            offset_out = CP["_apply_softmax_offset"](merged, merged_lse, inputs[3])
            grad_out, grad_lse, grad_offset = CP["_softmax_offset_backward_values"](
                grad_output=2 * offset_out,
                accum_out=merged,
                accum_lse=merged_lse,
                softmax_offset=inputs[3],
            )
            manual = torch.autograd.grad(
                (merged, merged_lse),
                inputs[:3],
                grad_outputs=(grad_out, grad_lse + 0.13 * merged_lse.isfinite()),
            )
            for actual, wanted in zip(
                (*manual, grad_offset), expected_grads, strict=True
            ):
                _check(actual, wanted, dtype)


@pytest.mark.parametrize("entry", ["sparse", "forced_dense"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_eager_flash_checkpoint_replays_merge_and_gradients(
    entry, dtype, reentrant, deterministic
):
    torch.use_deterministic_algorithms(deterministic)
    inputs = _inputs(dtype, True)
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    out, lse = checkpoint(
        lambda *xs: _merged(entry, "FLASH", True, *xs), *inputs, use_reentrant=reentrant
    )
    after_forward = counters["art"]["flex_flash_eager_fallback"]
    _loss(out, lse).backward()
    assert counters["art"]["flex_flash_eager_fallback"] > after_forward
    expected, expected_lse = _reference(True, *reference)
    _loss(expected, expected_lse).backward()
    _check(out, expected, dtype)
    _check(lse, expected_lse, torch.float32)
    for actual, wanted in zip(inputs, reference, strict=True):
        assert actual.grad is not None and wanted.grad is not None
        _check(actual.grad, wanted.grad, dtype)


def test_eager_flash_warning_and_counter_are_ranked_and_low_volume(monkeypatch, caplog):
    monkeypatch.setenv("RANK", "7")
    q, k, v, _ = _inputs(torch.float32, False)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            _partial("sparse", "FLASH", q, k, v, 0)
    events = [
        json.loads(row.message.split("ART_FLEX_ATTENTION_EVENT ", 1)[1])
        for row in caplog.records
        if "ART_FLEX_ATTENTION_EVENT " in row.message
    ]
    assert [event["key_count"] for event in events] == [1, 2, 3, 4, 5]
    assert all(
        event["rank"] == "7"
        and event["frame"] == "sparse_flex_attention"
        and event["shapes"][0] == list(q.shape)
        for event in events
    )
    assert counters["art"]["flex_flash_eager_fallback"] == 5


def test_fallback_telemetry_growth_is_bounded_and_lru(caplog):
    with caplog.at_level(logging.WARNING):
        for length in range(1, 1001):
            q = torch.empty(1, 1, length, 8, device="meta")
            assert compiled._flash_eager_fallback("FLASH", "growth", q, q, q)
        table = compiled._FLASH_EAGER_FALLBACK_KEYS
        assert len(table) == 64
        assert counters["art"]["flex_flash_eager_fallback"] == 1000
        assert counters["art"]["flex_flash_eager_fallback_overflow"] == 936
        first, second = list(table)[:2]
        q = torch.empty(1, 1, 937, 8, device="meta")
        compiled._flash_eager_fallback("FLASH", "growth", q, q, q)
        q = torch.empty(1, 1, 1001, 8, device="meta")
        compiled._flash_eager_fallback("FLASH", "growth", q, q, q)
    assert first in table and second not in table and len(table) == 64
    assert counters["art"]["flex_flash_eager_fallback"] == 1002
    assert counters["art"]["flex_flash_eager_fallback_overflow"] == 937
    assert len(caplog.records) == 9


def test_fallback_warning_budget_is_process_wide(caplog):
    q = torch.empty(1, 1, 2, 8, device="meta")
    with caplog.at_level(logging.WARNING):
        for _ in range(2005):
            compiled._flash_eager_fallback("FLASH", "budget", q, q, q)
    events = [
        json.loads(row.message.split("ART_FLEX_ATTENTION_EVENT ", 1)[1])
        for row in caplog.records
    ]
    assert [event["count"] for event in events] == [*range(1, 9), 1000, 2000]
    assert counters["art"]["flex_flash_eager_fallback"] == 2005
    assert counters["art"]["flex_flash_eager_fallback_overflow"] == 0
    assert len(compiled._FLASH_EAGER_FALLBACK_KEYS) == 1


@pytest.mark.parametrize("entry", ["forced_dense", "sparse", "dense"])
@pytest.mark.parametrize("qualified", [False, True])
def test_flash_wrappers_have_no_graph_breaks(entry, qualified):
    torch._dynamo.reset()
    q = torch.ones(1, 1, 2, 8, requires_grad=True)
    k = torch.ones(1, 1, 1, 8, requires_grad=True)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=1, device="cpu"
    )
    if entry == "sparse":
        fn = compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        extra = {}
    else:
        fn = (
            compiled._forced_flex_attention_dense
            if entry == "forced_dense"
            else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
        )
        extra = {"return_aux": AuxRequest(lse=True)}
    if qualified:
        assert (
            compiled.compiled_flex_attention_family(
                torch.compile(fn), device=q.device, head_dim=8
            )
            is not None
        )
    try:
        with (
            compiled.require_compiled_flex_attention(torch.compile(fn))
            if qualified
            else nullcontext()
        ):
            explanation = torch._dynamo.explain(fn)(
                q, k, k, block_mask=mask, scale=0.125, enable_gqa=False, **extra
            )
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0, explanation.break_reasons
    finally:
        torch._dynamo.reset()


class _FlashContract(torch.fx.Interpreter):
    """CPU contract backend: real math, compiled FLASH's natural HOP LSE."""

    def call_function(self, target, args, kwargs):
        result = super().call_function(target, args, kwargs)
        if target is torch.ops.higher_order.flex_attention:
            out, lse, maximum = result
            return out, lse * math.log(2), maximum * math.log(2)
        return result


def test_resumed_sparse_hop_fallback_rechecks_execution_mode(monkeypatch):
    torch._dynamo.reset()
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=128, KV_LEN=128, device="cpu"
    )
    original = mask.as_tuple

    def as_tuple():
        torch._dynamo.graph_break()
        return original()

    monkeypatch.setattr(mask, "as_tuple", as_tuple)
    fn = torch.compile(
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"}),
        backend=lambda graph, _: _FlashContract(graph).run,
        dynamic=False,
    )
    q = torch.ones(1, 1, 2, 8, requires_grad=True)
    k = torch.ones(1, 1, 1, 8, requires_grad=True)
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            for scale, expected in [(0.125, 1.0), (0.25, 2.0)]:
                out, lse = fn(q, k, k, block_mask=mask, scale=scale, enable_gqa=False)
                natural = compiled.normalize_flex_lse(lse, backend="FLASH")
                torch.testing.assert_close(natural, torch.full_like(natural, expected))
                grad = torch.autograd.grad(natural.sum(), q)[0]
                torch.testing.assert_close(grad, torch.full_like(q, scale))
        assert counters["art"]["flex_flash_eager_fallback"] > 0
    finally:
        torch._dynamo.reset()


def test_actual_sparse_hop_cache_fallback_keeps_lse_units():
    torch._dynamo.reset()
    attention = torch.compile(
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"}),
        backend="eager",
        dynamic=False,
    )
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=128, KV_LEN=128, device="cpu"
    )
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            for length in [2, 3]:
                q = torch.ones(1, 1, length, 8, requires_grad=True)
                k = torch.ones(1, 1, 1, 8, requires_grad=True)
                v = torch.ones_like(k, requires_grad=True)
                out, raw_lse = attention(
                    q, k, v, block_mask=mask, scale=0.125, enable_gqa=False
                )
                if length == 3:
                    lse = compiled.normalize_flex_lse(raw_lse, backend="FLASH")
                    torch.testing.assert_close(lse, torch.ones_like(lse))
                    grad_q, grad_k, grad_v = torch.autograd.grad(
                        out.sum() + lse.sum(), (q, k, v)
                    )
                    torch.testing.assert_close(grad_q, torch.full_like(q, 0.125))
                    torch.testing.assert_close(grad_k, torch.full_like(k, 3 * 0.125))
                    torch.testing.assert_close(grad_v, torch.full_like(v, 3.0))
        assert counters["art"]["flex_flash_eager_fallback"] > 0
    finally:
        torch._dynamo.reset()


def _cache_rank_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        cp = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=10))
        # As in the collective-order probe, only attention arithmetic is a stub;
        # Dynamo cache hit/fallback and the CP/status collectives are real.
        setattr(compiled, "flex_attention", lambda q, *args, **kwargs: q + 1)
        counters["art"]["flex_flash_eager_fallback"] = 0
        attention = torch.compile(
            compiled._forced_flex_attention_dense, backend="eager", dynamic=False
        )
        with torch._dynamo.config.patch(recompile_limit=1):
            options: dict[str, Any] = dict(block_mask=None, scale=1.0, enable_gqa=False)
            q = torch.zeros(2)
            attention(q, q, q, **options)
            dist.barrier()
            q = torch.zeros(3 if rank == 0 else 2)
            torch.testing.assert_close(attention(q, q, q, **options), q + 1)
            received = torch.empty(2)
            dist.all_to_all_single(received, torch.full((2,), float(rank)), group=cp)
            torch.testing.assert_close(received, torch.tensor([0.0, 1.0]))
            status = torch.tensor(0)
            dist.all_reduce(status)
            assert status.item() == 0
            assert counters["art"]["flex_flash_eager_fallback"] == (
                1 if rank == 0 else 0
            )
    finally:
        dist.destroy_process_group()


def test_cache_hit_and_fallback_ranks_complete_collectives(tmp_path):
    spawn_and_join(
        _cache_rank_worker,
        (f"file://{tmp_path}/rendezvous",),
        timeout=45,
        failure="Cache hit and fallback ranks diverged in model collectives",
    )


@pytest.mark.parametrize(
    "entry,resumed",
    [("sparse", False), ("dense", False), ("forced_dense", False), ("sparse", True)],
)
def test_qualified_flash_refuses_cache_exhaustion_before_eager_math(
    monkeypatch, entry, resumed
):
    torch._dynamo.reset()
    monkeypatch.setattr(
        compiled, "_FORCED_FLEX_EXECUTION_STATE", {"required": False, "fallback": False}
    )
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=128, KV_LEN=128, device="cpu"
    )
    if resumed:
        as_tuple = mask.as_tuple

        def graph_breaking_tuple():
            torch._dynamo.graph_break()
            return as_tuple()

        monkeypatch.setattr(mask, "as_tuple", graph_breaking_tuple)
    original = (
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        if entry == "sparse"
        else compiled._forced_flex_attention_dense
        if entry == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    attention = torch.compile(
        original, backend=lambda graph, _: _FlashContract(graph).run, dynamic=False
    )
    q = torch.ones(1, 1, 2, 8, requires_grad=True)
    k = torch.ones(1, 1, 1, 8, requires_grad=True)
    options = dict(
        block_mask=mask if entry == "sparse" else None, scale=0.125, enable_gqa=False
    )
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            attention(q, k, k, **options)
            # Qualify after warming, so the fallback cannot use captured state.
            assert (
                compiled.compiled_flex_attention_family(
                    attention, device=q.device, head_dim=8
                )
                is not None
            )
            if resumed:
                options["scale"] = 0.25
            else:
                q = torch.ones(1, 1, 3, 8, requires_grad=True)

            def forbidden_math(*args, **kwargs):
                pytest.fail("qualified FLASH reached quadratic eager math")

            hop = importlib.import_module("torch._higher_order_ops.flex_attention")
            monkeypatch.setattr(hop, "_math_attention_inner", forbidden_math)
            with compiled.require_compiled_flex_attention(attention):
                with pytest.raises(RuntimeError, match="refusing eager FLASH fallback"):
                    attention(q, k, k, **options)
        assert (
            compiled.compiled_flex_attention_family(
                attention, device=q.device, head_dim=8
            )
            is None
        )
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("entry", ["sparse", "dense", "forced_dense"])
def test_earlier_eager_flash_cannot_be_relabelled_as_compiled(monkeypatch, entry):
    monkeypatch.setattr(
        compiled, "_FORCED_FLEX_EXECUTION_STATE", {"required": False, "fallback": False}
    )
    original = (
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        if entry == "sparse"
        else compiled._forced_flex_attention_dense
        if entry == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    q = torch.ones(1, 1, 2, 8)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=2, device="cpu"
    )
    options = dict(block_mask=mask, scale=0.125, enable_gqa=False)
    original(q, q, q, **options)
    attention = torch.compile(original, backend="eager")
    assert (
        compiled.compiled_flex_attention_family(attention, device=q.device, head_dim=8)
        is None
    )
    # Rejected qualification cannot change ordinary eager serving.
    original(q, q, q, **options)
    assert not getattr(
        original, "_art_flex_execution_state", compiled._FORCED_FLEX_EXECUTION_STATE
    )["required"]


@pytest.mark.parametrize("forced", [False, True])
def test_qualified_dense_flash_cannot_resume_inside_public_api(monkeypatch, forced):
    torch._dynamo.reset()
    original = (
        compiled._forced_flex_attention_dense
        if forced
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=1, device="cpu"
    )
    as_tuple = mask.as_tuple

    def graph_breaking_tuple():
        torch._dynamo.graph_break()
        return as_tuple()

    monkeypatch.setattr(mask, "as_tuple", graph_breaking_tuple)
    attention = torch.compile(
        original, backend=lambda graph, _: _FlashContract(graph).run, dynamic=False
    )
    assert (
        compiled.compiled_flex_attention_family(
            attention, device=torch.device("cpu"), head_dim=8
        )
        is not None
    )
    q = torch.ones(1, 1, 2, 8, requires_grad=True)
    k = torch.ones(1, 1, 1, 8, requires_grad=True)

    def forbidden_math(*args, **kwargs):
        pytest.fail("qualified dense FLASH resumed into eager math")

    hop = importlib.import_module("torch._higher_order_ops.flex_attention")
    monkeypatch.setattr(hop, "_math_attention_inner", forbidden_math)
    try:
        with compiled.require_compiled_flex_attention(attention):
            with pytest.raises(RuntimeError, match="graph_break"):
                attention(q, k, k, block_mask=mask, scale=0.125, enable_gqa=False)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("entry", ["sparse", "dense", "forced_dense"])
def test_identity_query_and_requirement_have_separate_lifetimes(entry):
    original = (
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        if entry == "sparse"
        else compiled._forced_flex_attention_dense
        if entry == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    attention = torch.compile(original, backend="eager")
    state = compiled._flex_execution_state(attention)
    assert state is not None
    # A missing head dimension rejects qualification without arming execution.
    assert (
        compiled.compiled_flex_attention_family(attention, device=torch.device("cpu"))
        is None
    )
    assert not state["required"]
    assert (
        compiled.compiled_flex_attention_family(
            attention, device=torch.device("cpu"), head_dim=8
        )
        is not None
    )
    assert not state["required"]
    with pytest.raises(ValueError, match="caller failed"):
        with compiled.require_compiled_flex_attention(attention):
            assert state["required"]
            with compiled.require_compiled_flex_attention(attention):
                assert state["required"]
            assert state["required"]
            raise ValueError("caller failed")
    assert not state["required"]
    q = torch.ones(1, 1, 2, 8)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=2, device="cpu"
    )
    original(q, q, q, block_mask=mask, scale=0.125, enable_gqa=False)
    assert state["fallback"]
    assert (
        compiled.compiled_flex_attention_family(attention, device=q.device, head_dim=8)
        is None
    )
    assert not state["required"]
    original(q, q, q, block_mask=mask, scale=0.125, enable_gqa=False)


def _qualified_cache_rank_worker(rank, rendezvous, directory, entry, peer_entered):
    torch.set_num_threads(1)
    torch._dynamo.reset()
    # A long model timeout makes a timeout-based refusal fail this test.
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    cp = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=60))
    from art.trainer_rank import _calibration_protocol as protocol

    class Rank:
        _forward_replica_group = None

        def _forward_collective_device(self):
            return torch.device("cpu")

        def _recovery_reduce(self, values, *, _phase, _buffer, **kwargs):
            return protocol.vote(self, values, _phase, _buffer)

    progress = Path(directory) / f"rank{rank}.json"

    def dense_math(q, *args, **kwargs):
        if not torch.compiler.is_compiling():
            progress.write_text(json.dumps({"eager_math": True}))
            raise AssertionError("qualified execution entered eager math")
        return q + 1

    def sparse_math(q, k, v, *args):
        out = dense_math(q)
        return out, q.sum(-1), q.sum(-1)

    compiled.flex_attention = dense_math
    compiled.flex_attention_hop = sparse_math
    original = (
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        if entry == "sparse"
        else compiled._forced_flex_attention_dense
        if entry == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    attention = torch.compile(original, backend="eager", dynamic=False)
    mask = (
        create_block_mask(
            lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=128, KV_LEN=128, device="cpu"
        )
        if entry == "sparse"
        else None
    )
    options = dict(block_mask=mask, scale=1.0, enable_gqa=False)
    assert (
        compiled.compiled_flex_attention_family(
            attention, device=torch.device("cpu"), head_dim=8
        )
        is not None
    )
    q = torch.zeros(1, 1, 2, 8)
    with (
        compiled.require_compiled_flex_attention(attention, cohort_supervised=True),
        torch._dynamo.config.patch(recompile_limit=1),
    ):
        attention(q, q, q, **options)
        dist.barrier()
        progress.write_text(json.dumps({"qualified": True}))
        try:
            with protocol.Driver(Rank()):
                if rank == 0:
                    assert peer_entered.wait(
                        10
                    ), "peer never entered model communication"
                q = torch.zeros(1, 1, 3 if rank == 0 else 2, 8)
                attention(q, q, q, **options)
                progress.write_text(
                    json.dumps({"attention": "returned", "model_collective": "entered"})
                )
                received = torch.empty(2)
                peer_entered.set()
                dist.all_to_all_single(
                    received, torch.full((2,), float(rank)), group=cp
                )
                progress.write_text(json.dumps({"model_collective": "completed"}))
        except BaseException as error:
            progress.write_text(
                json.dumps({"error": str(error), "cause": str(error.__cause__)})
            )
            if rank != 1:
                raise
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("entry", ["sparse", "dense", "forced_dense"])
def test_qualified_mixed_cache_failure_terminates_rank_and_unblocks_peer(
    tmp_path, entry
):
    # PyTorch's supervisor is the CPU stand-in for Monarch's ProcMesh teardown:
    # a dead rank must terminate the cohort even if a peer is stuck in Gloo.
    peer_entered = mp.get_context("spawn").Event()
    processes = mp.spawn(
        _qualified_cache_rank_worker,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path), entry, peer_entered),
        nprocs=2,
        join=False,
    )
    started = time.monotonic()
    try:
        with pytest.raises(mp.ProcessExitedException) as failure:
            while not processes.join(timeout=20, grace_period=1):
                assert time.monotonic() - started < 25, "qualified cohort hung"
        assert failure.value.exit_code == 1
        assert failure.value.error_index == 0
        assert time.monotonic() - started < 25  # Model/control timeout is 60s.
        assert all(not process.is_alive() for process in processes.processes)
        assert processes.processes[0].exitcode == 1
        assert processes.processes[1].exitcode in (0, -signal.SIGTERM, -signal.SIGKILL)
        failed = json.loads((tmp_path / "rank0.json").read_text())
        peer = json.loads((tmp_path / "rank1.json").read_text())
        assert failed == {"qualified": True}, failed
        if "error" in peer:
            text = (peer["error"] + peer["cause"]).lower()
            assert "closed" in text or "reset" in text, peer
            assert "timed out" not in text and "timeout" not in text, peer
        else:
            assert peer == {
                "attention": "returned",
                "model_collective": "entered",
            }, peer
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.kill()
            process.join()
