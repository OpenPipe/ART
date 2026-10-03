"""CPU commitments distinguish real packing changes without reading GPU values."""

from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import (
    _FlatForwardPlan,
    _ForwardGroupPlan,
    _MemorySignature,
    _SplitForwardPlan,
)
from art.trainer_rank._prefix_tree_planner import build_canonical_prefix_tree


def plan(rows, groups=None, *, tp=1):
    groups = (tuple(range(len(rows))),) if groups is None else groups
    packed_groups = []
    for indices in groups:
        tensors = tuple(torch.tensor(rows[i]) for i in indices)
        tree = build_canonical_prefix_tree(tensors)
        packed_groups.append(
            _ForwardGroupPlan(
                None,
                False,
                tuple(indices),
                (),
                prefix_tree_pack(tensors, max_depth=0),
                input_row_fingerprints=tuple(
                    zip(tree.sequence_lengths, tree.row_fingerprints, strict=True)
                ),
            )
        )
    return _FlatForwardPlan(
        request_count=len(rows),
        output_metadata=tuple((None, True) for _ in rows),
        groups=tuple(packed_groups),
        packed_tokens=sum(
            ((g.packed.tokens.numel() + tp - 1) // tp) * tp for g in packed_groups
        ),
        logical_tokens=sum(map(len, rows)),
        output_bytes=0,
        signature=_MemorySignature(
            (1, tp, 1, 1),
            (2, None),
            len(groups),
            ("target",),
            False,
            tuple(False for _ in groups),
        ),
    )


def fingerprints(value):
    return TrainerRank._packing_fingerprints(value)


def test_same_aggregate_geometry_different_boundaries_and_membership():
    first = plan(((10001, 2, 3), (10002, 4, 5, 6, 7)))
    second = plan(((10001, 2, 3, 4), (10002, 5, 6, 7)))
    assert first.packed_tokens == second.packed_tokens == 8
    assert (
        len(first.groups[0].packed.segments)
        == len(second.groups[0].packed.segments)
        == 2
    )
    assert (
        fingerprints(first)["packing_plan_sha256"]
        != fingerprints(second)["packing_plan_sha256"]
    )

    rows = ((10001, 1), (10002, 2), (10003, 3), (10004, 4))
    a, b = plan(rows, ((0, 1), (2, 3))), plan(rows, ((0, 2), (1, 3)))
    assert [g.packed.tokens.numel() for g in a.groups] == [
        g.packed.tokens.numel() for g in b.groups
    ]
    assert (
        fingerprints(a)["packing_plan_sha256"] != fingerprints(b)["packing_plan_sha256"]
    )
    assert (
        fingerprints(a)["input_tokens_sha256"] == fingerprints(b)["input_tokens_sha256"]
    )


def test_input_commitment_is_independent_of_split_and_execution_order():
    rows = ((10001, 1), (10002, 2), (10003, 3), (10004, 4))
    mappings = ((1, 3), (0, 2))
    children = tuple(plan(tuple(rows[i] for i in indices)) for indices in mappings)
    split = _SplitForwardPlan(children, mappings, 4)
    reversed_split = _SplitForwardPlan(children[::-1], mappings[::-1], 4)
    expected = build_canonical_prefix_tree(rows).content_fingerprint
    for candidate in (plan(rows), split, reversed_split):
        assert fingerprints(candidate)["input_tokens_sha256"] == expected
    assert (
        fingerprints(split)["packing_plan_sha256"]
        != fingerprints(reversed_split)["packing_plan_sha256"]
    )
    assert fingerprints(split)["subforward_packing_plan_sha256"] == tuple(
        fingerprints(child)["packing_plan_sha256"] for child in children
    )


def test_input_content_does_not_change_geometry_or_compile_dedup():
    a, b = plan(((987654321, 2, 3),)), plan(((987654322, 2, 3),))
    assert (
        fingerprints(a)["packing_plan_sha256"] == fingerprints(b)["packing_plan_sha256"]
    )
    assert (
        fingerprints(a)["input_tokens_sha256"] != fingerprints(b)["input_tokens_sha256"]
    )
    assert TrainerRank._telemetry_plan_signature(
        a
    ) == TrainerRank._telemetry_plan_signature(b)
    assert (
        fingerprints(a)["packing_plan_sha256"]
        != fingerprints(plan(((987654321, 2, 3),), tp=2))["packing_plan_sha256"]
    )
    assert "987654321" not in json.dumps(TrainerRank._telemetry_signature(a))


def test_inactive_request_is_not_misreported_as_complete_input_identity():
    active = plan(((10001, 2),))
    with_inactive = replace(
        active, request_count=2, output_metadata=((None, True), (None, True))
    )
    assert fingerprints(active)["input_tokens_sha256"] is not None
    assert fingerprints(with_inactive)["input_tokens_sha256"] is None


def test_snapshot_and_event_reuse_hashes_without_tensor_readback(
    monkeypatch: pytest.MonkeyPatch,
):
    selected = plan(((987654321, 2, 3), (987654322, 4, 5)))
    expected = fingerprints(selected)
    rng_before = torch.random.get_rng_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("telemetry must not read tensors or query CUDA")

    for name in ("cpu", "numpy", "tolist", "item"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    for name in ("synchronize", "current_device", "memory_allocated", "is_available"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    rank = TrainerRank.__new__(TrainerRank)
    rank._layout_cache_lock = threading.Lock()
    rank._planning_seconds_accum = 0.0
    rank._speculative_planning_seconds = 0.0
    rank._snapshot_planning_telemetry(
        selected, SimpleNamespace(estimated_required_bytes=0, available_bytes=0)
    )
    for key, value in expected.items():
        assert (
            rank.last_forward_telemetry()[key]
            == TrainerRank._telemetry_signature(selected)[key]
            == value
        )
    assert torch.equal(rng_before, torch.random.get_rng_state())
