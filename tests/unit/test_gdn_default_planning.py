"""CPU planner and bucket snapshots recorded from ART main at e18c570."""

from dataclasses import fields, is_dataclass
import hashlib
import json

from pydantic import BaseModel
import pytest
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn import gdn_prefix_tree as gdn

# Includes every decision field, rank-local bucket tensor, and exchange plan.
_MAIN_SHA256 = {
    (2, 1024): "66481c294647cf755b7e1d6e1b4b351a21042ea61dda8a2ade464757fb626e7c",
    (4, 1024): "68d2bda9fbf05ad670cb584bd34575a69433fabc0dbc045dd7ab28c0970ed687",
    (2, 32768): "11bad73a73eef9ede86d71f0d79611cdea8ef2801817897e7b7b322a9f7c8f65",
    (4, 32768): "2fd6ed2fcba90e8ee8e3f5c851e4c433f722e5432d6d35f00622120b4ff4eb90",
}


def _metadata(value):
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, BaseModel):
        value = {name: getattr(value, name) for name in type(value).model_fields}
    elif is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    return value


def _default_plan_metadata(cp_size, prefix_length, planner=gdn):
    groups, parents = [], []
    for length, group, parent in (
        (prefix_length, 0, 0),
        (512, 1, 0),
        (512, 2, 0),
        (512, 3, 3),
        (512, 4, 3),
    ):
        groups.extend([group] * length)
        parents.extend([parent] * length)
    spec = planner.parse_gdn_prefix_tree_segments(
        torch.tensor([groups]), torch.tensor([parents])
    )
    ranges = tuple(
        ((len(groups) * r // cp_size, len(groups) * (r + 1) // cp_size, 0),)
        for r in range(cp_size)
    )
    layout = TokenLayoutIndex(ranges, tuple(b - a for ((a, b, _),) in ranges))
    decision = planner.build_gdn_global_execution_decision(
        spec,
        cp_size=cp_size,
        attention_token_layout_index=layout,
    )
    plans = tuple(
        planner.materialize_gdn_rank_execution_plan(
            spec, decision, device="cpu", cp_rank=rank
        )
        for rank in range(cp_size)
    )
    return decision, plans


@pytest.mark.parametrize("cp_size,prefix_length", _MAIN_SHA256)
def test_default_cp_decisions_and_buckets_match_main(cp_size, prefix_length):
    decision, plans = _default_plan_metadata(cp_size, prefix_length)
    assert any(decision.chained_nodes) == (prefix_length >= 32768)
    snapshot = json.dumps(
        _metadata((decision, plans)), sort_keys=True, separators=(",", ":")
    )
    assert (
        hashlib.sha256(snapshot.encode()).hexdigest()
        == _MAIN_SHA256[cp_size, prefix_length]
    )
