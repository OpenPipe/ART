from __future__ import annotations

import pickle
from typing import NamedTuple

import pytest

from art.trajectories import Trajectory, compact_memory


class Holder(NamedTuple):
    value: object


class Opaque:
    def __init__(self, value):
        self.value = value


class TaggedTuple(tuple):
    pass


def fresh(s):
    return s.encode().decode()


@pytest.mark.parametrize("kind", ["namedtuple", "opaque", "subclass_attribute"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_immutable_alias_through_opaque_container(kind, reverse, use_pickle):
    canonical = "shared immutable provider metadata value"
    inner = (fresh(canonical), ["mutable state"])
    if kind == "namedtuple":
        holder = Holder(inner)
    elif kind == "opaque":
        holder = Opaque(inner)
    else:
        holder = TaggedTuple(("outer",))
        holder.value = inner
    pairs = [("holder", holder), ("inner", inner)]
    if reverse:
        pairs.reverse()
    trajectory = Trajectory(metadata={"seed": canonical, **dict(pairs)})
    result = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )
    for graph in (trajectory, result):
        assert graph.metadata["holder"].value is graph.metadata["inner"]
        graph.metadata["inner"][1].append("mutation")
        assert graph.metadata["holder"].value[1][-1] == "mutation"
        assert type(graph.metadata["holder"]) is type(holder)
    assert trajectory.metadata["inner"] is inner


@pytest.mark.parametrize("bridge", ["list", "dict"])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_tuple_backedge_keeps_exact_cycle(bridge, use_pickle):
    canonical = "shared immutable provider metadata value"
    mutable = [] if bridge == "list" else {}
    inner = (fresh(canonical), mutable)
    if bridge == "list":
        mutable.append(inner)
    else:
        mutable["back"] = inner
    trajectory = Trajectory(
        metadata={"seed": canonical, "inner": inner, "mutable": mutable}
    )
    result = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )
    for graph in (trajectory, result):
        actual = graph.metadata["inner"]
        edge = actual[1][0] if bridge == "list" else actual[1]["back"]
        assert edge is actual
        assert actual[1] is graph.metadata["mutable"]
    assert trajectory.metadata["inner"] is inner


@pytest.mark.parametrize("use_pickle", [False, True])
def test_frozenset_alias_through_opaque_container(use_pickle):
    canonical = "shared immutable provider metadata value"
    inner = frozenset([fresh(canonical)])
    trajectory = Trajectory(
        metadata={"seed": canonical, "opaque": Opaque(inner), "inner": inner}
    )
    result = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )
    for graph in (trajectory, result):
        assert graph.metadata["opaque"].value is graph.metadata["inner"]
    assert trajectory.metadata["inner"] is inner


def test_mutable_descendants_still_intern_with_two_tuple_backedges():
    canonical = "shared immutable provider metadata value"
    a, b = [], {}
    t1, t2 = (a,), (b,)
    a.extend([fresh(canonical), t2])
    b.update(value=fresh(canonical), back=t1)
    trajectory = Trajectory(metadata={"seed": canonical, "first": t1, "second": t2})
    for _ in range(2):
        compact_memory(trajectory)
        assert trajectory.metadata["first"] is t1
        assert trajectory.metadata["second"] is t2
        assert a[1] is t2 and b["back"] is t1
        assert a[0] is canonical and b["value"] is canonical
        a[0] = fresh(canonical)
        b["value"] = fresh(canonical)


@pytest.mark.parametrize("kind", [tuple, frozenset])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_immutable_alias_used_as_mapping_key_and_set_member(kind, use_pickle):
    canonical = "shared immutable provider metadata value"
    inner = kind([fresh(canonical)])
    mapping = {inner: "retained value"}
    trajectory = Trajectory(
        metadata={
            "seed": canonical,
            "mapping": mapping,
            "set": {inner},
            "inner": inner,
        }
    )
    result = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )
    for graph in (trajectory, result):
        actual = graph.metadata["inner"]
        assert next(iter(graph.metadata["mapping"])) is actual
        assert next(iter(graph.metadata["set"])) is actual
        assert graph.metadata["mapping"][actual] == "retained value"
    assert trajectory.metadata["inner"] is inner
