"""Physical ownership ceiling, with no Megatron/CUDA runtime dependency."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from art.megatron.context_parallel._ownership import capped_owners


def test_262k_oversized_rank_uses_even_candidate():
    lengths = [1024] * 256
    original = tuple(
        [0] * 75
        + [1] * 25
        + [2] * 26
        + [3] * 26
        + [4] * 26
        + [5] * 26
        + [6] * 26
        + [7] * 26
    )
    balanced = tuple(i // 32 for i in range(256))
    assert max(original.count(i) * 1024 for i in range(8)) == 76800
    result = capped_owners(original, balanced, lengths, 8, 32768)
    assert result == balanced
    assert [result.count(i) * 1024 for i in range(8)] == [32768] * 8


@pytest.mark.parametrize("limit", [32768, 65536])
def test_already_feasible_plan_is_preserved(limit):
    owners = (0, 0, 1, 1)
    assert capped_owners(owners, (0, 1, 0, 1), [8192] * 4, 2, limit) is owners


@pytest.mark.parametrize("limit", [0, -1, True, 3.2])
def test_invalid_or_impossible_ceiling_is_refused(limit):
    with pytest.raises(ValueError):
        capped_owners((0, 1), (0, 1), [10, 10], 2, limit)


def test_no_candidate_fits_does_not_invent_smaller_rows():
    with pytest.raises(ValueError, match="Neither"):
        capped_owners((0, 0, 1), (0, 1, 1), [10] * 3, 2, 15)


def test_real_search_dispatch_reprices_selected_ownership():
    # Execute the maintained dispatch body with scalar planning dependencies.
    # This is not a model/runtime reconstruction or a GPU correctness test.
    source = Path(__file__).parents[2] / "src/art/megatron/context_parallel/runtime.py"
    tree = ast.parse(source.read_text())
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_search_chunk_assignment"
    )
    body.returns = None
    for arg in (*body.args.args, *body.args.kwonlyargs):
        arg.annotation = None
    original, balanced = (0, 0, 0, 1), (0, 0, 1, 1)
    prices = []

    def evaluate(**kwargs):
        prices.append(kwargs["owners"])
        return {"selected": kwargs["owners"]}

    namespace = dict(
        __package__="art.megatron.context_parallel",
        _search_generic_chunk_assignment=lambda **kw: (
            original,
            (0,) * 4,
            {"old": True},
        ),
        _contiguous_chunk_assignment=lambda **kw: balanced,
        _pair_program=lambda *a, **kw: "scalar program",
        _evaluate_plan=evaluate,
    )
    exec(
        compile(ast.Module(body=[body], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    search = namespace["_search_chunk_assignment"]
    kwargs = dict(
        chunk_ranges=[SimpleNamespace(size=lambda: 10)] * 4,
        pair_matrix=[],
        q_weights=[],
        cp_size=2,
    )
    assert (
        search(
            **kwargs,
            config=SimpleNamespace(
                planner_max_owned_tokens=None, workload_profile=None
            ),
        )[0]
        == original
    )
    assert not prices
    assert (
        search(
            **kwargs,
            config=SimpleNamespace(planner_max_owned_tokens=20, workload_profile=None),
        )[0]
        == balanced
    )
    assert prices == [balanced]
