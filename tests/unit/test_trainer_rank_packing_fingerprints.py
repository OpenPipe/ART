"""CPU commitments distinguish real packing changes without reading GPU values."""

from dataclasses import replace
import hashlib
import json
import threading

import pytest
import torch

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import (
    _FlatForwardPlan,
    _ForwardGroupPlan,
    _MemoryCheck,
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


def _golden_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _flat_golden_digest(tp, padded_tokens):
    # Literal geometry for rows of lengths 3 and 2; do not derive the oracle
    # from plan metadata or the production fingerprint implementation.
    return _golden_digest(
        (
            "art.prefix-pack/v1",
            (1, tp, 1, 1),
            2,
            (
                (
                    (0, 1),
                    False,
                    5,
                    padded_tokens,
                    (((0,), 0, 3, 0, 1, 1), ((1,), 0, 2, 3, 2, 2)),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    ("tp", "padded_tokens", "golden"),
    (
        (1, 5, "818fce443b1d8e52151f98483ac87b7d941f87d8bc874949930b0ac41b040d6d"),
        (2, 6, "5380518577fc6e1cf9d17b6624aefee7321cac3b69f8dcdef8f632a439a8e7f0"),
        (4, 8, "3f88443236fb1d924c0537154d1af301e3eb50495af81cbf910bc70c053b768b"),
    ),
)
def test_flat_packing_fingerprint_golden_digest(tp, padded_tokens, golden):
    expected = _flat_golden_digest(tp, padded_tokens)
    assert expected == golden
    actual = fingerprints(plan(((10001, 2, 3), (10002, 4)), tp=tp))
    assert actual["packing_fingerprint_schema"] == "art.prefix-pack/v1"
    assert actual["packing_plan_sha256"] == expected
    assert actual["subforward_packing_plan_sha256"] == (expected,)
    if padded_tokens != 5:
        # Keep topology and raw length fixed: the padded length itself matters.
        assert actual["packing_plan_sha256"] != _flat_golden_digest(tp, 5)


def test_split_packing_fingerprint_golden_digest():
    children = (
        plan(((10001, 2, 3), (10002, 4)), tp=2),
        plan(((10003, 5, 6),), tp=2),
    )
    mappings = ((2, 0), (1,))
    child_digests = (
        _flat_golden_digest(2, 6),
        _golden_digest(
            (
                "art.prefix-pack/v1",
                (1, 2, 1, 1),
                1,
                (((0,), False, 3, 4, (((0,), 0, 3, 0, 1, 1),)),),
            )
        ),
    )
    expected = _golden_digest(
        (
            "art.prefix-pack-split/v1",
            3,
            tuple(zip(mappings, child_digests, strict=True)),
        )
    )
    assert (
        expected == "7ca6236996a91feeacd5accff5c7fe231b697396b92bfb061383ec4fabe9334b"
    )
    selected = _SplitForwardPlan(children, mappings, 3)
    actual = fingerprints(selected)
    assert actual["packing_fingerprint_schema"] == "art.prefix-pack/v1"
    assert actual["packing_plan_sha256"] == expected
    assert actual["subforward_packing_plan_sha256"] == child_digests
    assert (
        fingerprints(replace(selected, request_count=4))["packing_plan_sha256"]
        != actual["packing_plan_sha256"]
    )


@pytest.mark.parametrize("field", ("grad_enabled", "request_count", "packed_start"))
def test_packing_fingerprint_discriminates_individual_fields(field):
    selected = plan(((10001, 2, 3), (10002, 4)), tp=2)
    group = selected.groups[0]
    # Vary only the named metadata field to isolate its hash contribution.
    if field == "grad_enabled":
        changed = replace(selected, groups=(replace(group, grad_enabled=True),))
    elif field == "request_count":
        changed = replace(selected, request_count=3)
    else:
        first, second = group.packed.segments
        packed = replace(
            group.packed, segments=(first, replace(second, packed_start=4))
        )
        changed = replace(selected, groups=(replace(group, packed=packed),))
    assert (
        fingerprints(changed)["packing_plan_sha256"]
        != fingerprints(selected)["packing_plan_sha256"]
    )


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
    rank._snapshot_planning_telemetry(selected, _MemoryCheck(0, 0, True))
    for key, value in expected.items():
        assert (
            rank.last_forward_telemetry()[key]
            == TrainerRank._telemetry_signature(selected)[key]
            == value
        )
    assert torch.equal(rng_before, torch.random.get_rng_state())
