"""CPU ownership selection exercises maintained planner/evaluator bodies."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch


def planner_namespace():
    root = Path(__file__).parents[2] / "src/art"
    namespace = dict(
        __name__=__name__, dataclass=dataclass, replace=replace, np=np, torch=torch
    )
    for name, selected in (
        ("megatron/context_parallel/types.py", {"ContextParallelConfig", "TokenRange"}),
        ("megatron/context_parallel/runtime.py", None),
    ):
        source = root / name
        definitions = [
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and (selected is None or n.name in selected)
        ]
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.ImportFrom(
                                module="__future__",
                                names=[ast.alias(name="annotations")],
                                level=0,
                            ),
                            *definitions,
                        ],
                        type_ignores=[],
                    )
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
    return namespace


def balanced_price():
    rt = planner_namespace()
    config = rt["ContextParallelConfig"](
        planner_local_pair_ms=1,
        planner_remote_pair_ms=1,
        planner_local_backward_pair_ms=1,
        planner_remote_backward_pair_ms=1,
        planner_stage_overhead_ms=0,
        planner_remote_stage_host_ms=0,
        planner_comm_stage_overhead_ms=0,
        planner_interval_overhead_ms=0,
        planner_merge_q_token_ms=0,
        planner_fetch_token_ms=0,
        planner_reduce_token_ms=0,
        planner_remote_stage_underfill_ms=0,
    )
    ranges = tuple(rt["TokenRange"](i * 32768, (i + 1) * 32768) for i in range(8))
    # Continuous causal area; exact finite-token counts differ by O(1/N).
    pairs = [
        [32768**2 if j < i else 32768**2 / 2 if j == i else 0 for j in range(8)]
        for i in range(8)
    ]
    program = rt["_pair_program"](pairs, chunk_ranges=ranges)
    return rt["_evaluate_plan"](
        program=program,
        owners=tuple(range(8)),
        wave_assignment=(0,) * 8,
        cp_size=8,
        config=config,
    )


def test_balanced_last_rank_attention_is_priced_at_1_875_times_mean():
    priced = balanced_price()
    assert priced["rank_scores"][-1] / np.mean(priced["rank_scores"]) == pytest.approx(
        1.875
    )
    assert priced["score"] == priced["rank_scores"][-1]


@dataclass(frozen=True)
class Group:
    packed: object
    cp_config: object = None


@dataclass(frozen=True)
class Plan:
    groups: tuple
    signature: object


def select(monkeypatch, available, *, original_ms=100, balanced_ms=150, required=None):
    from art.trainer_rank import _micro_batch_planner as planner

    original = SimpleNamespace(
        chunk_owners=(0,) * 75
        + tuple(r for r in range(1, 8) for _ in range(25 if r == 1 else 26))
    )
    balanced = SimpleNamespace(
        chunk_owners=tuple(r for r in range(8) for _ in range(32))
    )

    def candidates(**kwargs):
        return ((original, original_ms), (balanced, balanced_ms))

    monkeypatch.setitem(
        sys.modules,
        "art.megatron.context_parallel.runtime",
        SimpleNamespace(context_parallel_ownership_candidates=candidates),
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.training.microbatches",
        SimpleNamespace(_context_parallel_config_for_provider=lambda *a: None),
    )
    monkeypatch.setattr(planner._impl, "_pad_packed_batch", lambda batch, **kw: batch)
    checks = []

    def check(plan):
        owners = plan.groups[0].cp_config.chunk_owners
        rows = max(owners.count(r) for r in range(8)) * 1024
        demand = required(rows) if required else rows * 1000
        checks.append((rows, demand, available))
        return SimpleNamespace(
            fits=demand <= available, estimated_required_bytes=demand
        )

    rank = SimpleNamespace(
        runtime=SimpleNamespace(provider=None, model_support_handler=None),
        device="cpu",
        _num_layers=64,
        _topology=lambda: SimpleNamespace(tp=1, cp=8),
        _planning_status=lambda *a: nullcontext(),
        _memory_check=check,
    )
    batch = SimpleNamespace(
        tokens=torch.empty(1, 262144, device="meta"), group_ids=None, parent_ids=None
    )
    plan = Plan((Group(batch),), SimpleNamespace(topology=(1, 1, 8, 1)))
    result = cast(Any, planner._select_cp_ownership)(rank, plan)
    return result.groups[0].cp_config, checks


def test_original_refusal_becomes_balanced_fit_at_same_safeguards(monkeypatch):
    config, checks = select(monkeypatch, 60000000)
    assert checks == [(76800, 76800000, 60000000), (32768, 32768000, 60000000)]
    assert config.chunk_owners.count(7) == 32


def test_original_fits_and_is_cheaper_remains_unchanged(monkeypatch):
    config, checks = select(monkeypatch, 100000000)
    assert checks[0][1] < checks[0][2]
    assert config.chunk_owners.count(0) == 75


def test_balanced_is_chosen_on_time_even_when_original_fits(monkeypatch):
    config, _ = select(monkeypatch, 100000000, balanced_ms=80)
    assert config.chunk_owners.count(0) == 32


def test_neither_fits_still_returns_a_refusal(monkeypatch):
    _, checks = select(monkeypatch, 100)
    assert all(demand > available for _, demand, available in checks)


def test_concrete_ownership_is_validated_and_repriced():
    rt = planner_namespace()
    rt["_search_generic_chunk_assignment"] = lambda **kw: ((0, 0, 0, 1), (0,) * 4, {})
    rt["_evaluate_plan"] = lambda **kw: {"selected": kw["owners"]}
    rt["_pair_program"] = lambda *a, **kw: None
    kwargs = dict(chunk_ranges=(None,) * 4, pair_matrix=[], q_weights=[], cp_size=2)
    config = rt["ContextParallelConfig"](chunk_owners=(0, 0, 1, 1))
    owners, _, price = rt["_search_chunk_assignment"](**kwargs, config=config)
    assert owners == price["selected"] == config.chunk_owners
    for invalid in ((0,), (0, 0, 1, 2)):
        with pytest.raises(ValueError, match="inventory"):
            rt["_search_chunk_assignment"](
                **kwargs, config=replace(config, chunk_owners=invalid)
            )


def test_admission_and_execution_receive_the_same_concrete_ownership(monkeypatch):
    from art.trainer_rank import _impl as tr
    from art.trainer_rank import _micro_batch_planner as planner

    config, _ = select(monkeypatch, 60000000)
    batch = SimpleNamespace(tokens=torch.empty(1, 262144, device="meta"))
    group = SimpleNamespace(packed=batch, grad_enabled=True, cp_config=config)
    plan = SimpleNamespace(
        groups=(group,),
        signature=SimpleNamespace(topology=(1, 1, 8, 1), grad_enabled=True),
    )
    seen = []
    rank = SimpleNamespace(
        _topology=lambda: SimpleNamespace(cp=8, tp=1),
        _topology_key=lambda: (1, 1, 8, 1),
        _physical_tokens=lambda n: n,
        _recompute_granularity=None,
        _geometry=SimpleNamespace(moe_experts=0),
    )

    def rows(batch, **kwargs):
        seen.append(kwargs["config"])
        return 32768

    rank._max_rank_model_tokens = rows
    rank._prepare_context_parallel_forward = rows
    assert cast(Any, planner._plan_group_rows)(rank, plan) == ((32768, True),)
    assert cast(Any, planner._plan_retained_tokens)(rank, plan) == 32768
    assert (
        cast(Any, tr.TrainerRank._prepare_packed_forward)(rank, batch, config=config)
        == 32768
    )
    assert seen == [config, config, config]
