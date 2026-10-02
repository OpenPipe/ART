import pickle
from typing import NamedTuple

import pytest

import art
from art.trajectories import compact_memory


class Pair(NamedTuple):
    label: str
    count: int


class TaggedTuple(tuple):
    tag: list[str]


class TaggedFrozen(frozenset):
    tag: list[str]


class NoRebuildDict(dict):
    tag: list[str]

    def clear(self):
        raise AssertionError("custom mapping must not be rebuilt")


class OpaqueKey:
    calls = 0

    def __hash__(self):
        self.calls += 1
        assert self.calls <= 1, "unexpected extra hash callback"
        return hash("opaque key fixture")


@pytest.mark.parametrize("kind", [Pair, TaggedTuple, TaggedFrozen])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_immutable_subclasses_keep_type_state_and_aliases(kind, use_pickle):
    state = ["retained state"]
    value = Pair("label", 3) if kind is Pair else kind(["label", "other"])
    if kind is not Pair:
        value.tag = state
    trajectory = art.Trajectory(
        metadata={"first": value, "second": value, "state": state}
    )

    restored = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )

    assert trajectory.metadata["first"] is value
    for graph in (trajectory, restored):
        actual = graph.metadata["first"]
        assert type(actual) is kind
        assert actual is graph.metadata["second"]
        assert actual == value
        if kind is Pair:
            assert actual.label == "label" and actual.count == 3
        else:
            assert actual.tag is graph.metadata["state"]


@pytest.mark.parametrize("use_pickle", [False, True])
def test_key_compaction_preserves_order_values_and_cycles(use_pickle):
    canonical = "shared public dictionary key"
    key = canonical.encode().decode()
    assert key is not canonical
    mapping = {key: canonical.encode().decode(), "later": 2}
    mapping["self"] = mapping
    trajectory = art.Trajectory(
        metadata={"seed": canonical, "first": mapping, "second": mapping}
    )
    before = list(mapping)

    restored = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )

    for graph in (trajectory, restored):
        actual = graph.metadata["first"]
        assert list(actual) == before
        assert actual is graph.metadata["second"]
        assert actual["self"] is actual
        assert next(iter(actual)) is graph.metadata["seed"]
        assert actual[canonical] is graph.metadata["seed"]
        assert actual["later"] == 2


def test_custom_mapping_is_opaque_and_keeps_state():
    canonical = "shared public dictionary key"
    key = canonical.encode().decode()
    mapping = NoRebuildDict({key: canonical.encode().decode(), "later": 2})
    mapping.tag = ["retained state"]
    trajectory = art.Trajectory(metadata={"seed": canonical, "mapping": mapping})
    before = list(mapping)

    compact_memory(trajectory)

    assert trajectory.metadata["mapping"] is mapping
    assert list(mapping) == before
    assert next(iter(mapping)) is key
    assert mapping[key] == canonical
    assert mapping[key] is not canonical
    assert mapping.tag == ["retained state"]


def test_mixed_mapping_avoids_hash_callbacks_and_preserves_direct_values():
    canonical = "shared public dictionary key"
    key = canonical.encode().decode()
    opaque = OpaqueKey()
    mapping = {key: canonical.encode().decode(), opaque: 2}
    opaque.calls = 0
    trajectory = art.Trajectory(metadata={"seed": canonical, "mapping": mapping})
    before = list(mapping)

    compact_memory(trajectory)

    assert list(mapping) == before
    assert next(iter(mapping)) is key
    assert mapping[key] == canonical
    assert mapping[key] is not canonical
    assert opaque.calls == 0


def test_plain_immutable_containers_keep_identity():
    canonical = "shared public immutable value"
    shared_tuple = (canonical.encode().decode(),)
    trajectory = art.Trajectory(
        metadata={
            "seed": canonical,
            "tuple": shared_tuple,
            "alias": shared_tuple,
            "frozen": frozenset([canonical.encode().decode()]),
        }
    )

    compact_memory(trajectory)

    assert type(trajectory.metadata["tuple"]) is tuple
    assert type(trajectory.metadata["frozen"]) is frozenset
    assert trajectory.metadata["tuple"] is trajectory.metadata["alias"]
    assert trajectory.metadata["tuple"] is shared_tuple
    assert trajectory.metadata["tuple"][0] == canonical
    assert trajectory.metadata["tuple"][0] is not canonical
    assert next(iter(trajectory.metadata["frozen"])) == canonical
