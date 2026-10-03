import gc
import pickle
from typing import ClassVar
import weakref

from pydantic import BaseModel, ConfigDict
import pytest

from art.trajectories import Trajectory, compact_memory


class IdentityLabel(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    calls: ClassVar[int] = 0
    forbid: ClassVar[bool] = False

    def __hash__(self):
        type(self).calls += 1
        if type(self).forbid:
            raise AssertionError("compaction called user hash")
        return id(self.label)

    def __eq__(self, other):
        if type(self).forbid:
            raise AssertionError("compaction called user equality")
        return self is other


class NestedIdentity(IdentityLabel):
    child: Trajectory

    def __hash__(self):
        super().__hash__()
        return id(self.child.metadata["label"])


class HiddenList(list):
    def __iter__(self):
        raise AssertionError("compaction called subclass iterator")


class HiddenIdentity(IdentityLabel):
    children: object

    def __hash__(self):
        super().__hash__()
        assert isinstance(self.children, HiddenList)
        return id(self.children[0].metadata["label"])


def fresh(value):
    return value.encode().decode()


def container_for(kind, member):
    if kind == "set":
        return {member}
    if kind == "frozenset":
        return frozenset([member])
    if kind == "tuple_key":
        return {(member,): "retained"}
    return {member: "retained"}


def contains(kind, container, member):
    return ((member,) if kind == "tuple_key" else member) in container


@pytest.mark.parametrize("kind", ["set", "frozenset", "dict", "tuple_key"])
@pytest.mark.parametrize("alias_first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_hash_sensitive_members_and_outside_aliases(
    kind, alias_first, nested, use_pickle
):
    canonical = "shared hash-sensitive model label"
    child = Trajectory(metadata={"label": fresh(canonical)})
    member = (
        NestedIdentity(label=fresh(canonical), child=child)
        if nested
        else IdentityLabel(label=fresh(canonical))
    )
    container = container_for(kind, member)
    aliases = {"member": member, "child": child, "state": child.metadata}
    metadata = (
        {"seed": canonical, **aliases, "container": container}
        if alias_first
        else {"seed": canonical, "container": container, **aliases}
    )
    trajectory = Trajectory(metadata=metadata)
    original_label = member.label
    original_child_label = child.metadata["label"]
    original_hash = hash(member)
    calls = type(member).calls
    try:
        type(member).forbid = True
        compact_memory(trajectory)
    finally:
        type(member).forbid = False
    assert type(member).calls == calls
    assert member.label is original_label
    assert child.metadata["label"] is original_child_label
    assert hash(member) == original_hash
    assert contains(kind, container, member)
    assert trajectory.metadata["state"] is child.metadata
    if use_pickle:
        result = pickle.loads(pickle.dumps(trajectory))
        # Nested ART reducers must not restart interning after the outer pass.
        assert child.metadata["label"] is original_child_label
        assert contains(kind, container, member)
        assert contains(kind, result.metadata["container"], result.metadata["member"])
        if nested:
            assert result.metadata["member"].child is result.metadata["child"]
        assert result.metadata["state"] is result.metadata["child"].metadata


@pytest.mark.parametrize("alias_first", [False, True])
def test_hash_dependency_inside_opaque_subclass_is_not_reinterned_by_pickle(
    alias_first,
):
    canonical = "shared nested opaque hash dependency"
    child = Trajectory(metadata={"label": fresh(canonical)})
    member = HiddenIdentity(label=fresh(canonical), children=HiddenList([child]))
    members = {member}
    aliases = {"child": child, "state": child.metadata}
    graph = (
        {"seed": canonical, **aliases, "members": members}
        if alias_first
        else {"seed": canonical, "members": members, **aliases}
    )
    trajectory = Trajectory(metadata=graph)
    label = child.metadata["label"]
    compact_memory(trajectory)
    assert child.metadata["label"] is label
    assert member in members
    # Native pickle also uses the subclass iterator. Test implicit ART compaction
    # without asking pickle to serialize the deliberately unpicklable subclass.
    child.__reduce_ex__(5)
    assert child.metadata["label"] is label
    assert member in members


def test_ordinary_graph_still_interns_and_preserves_alias_cycles():
    canonical = "shared ordinary graph label"
    values = [fresh(canonical)]
    graph = {"seed": canonical, "values": values, "alias": values}
    values.append(graph)
    compact_memory(graph)
    assert values[0] is canonical
    assert graph["values"] is graph["alias"] is values
    assert values[1] is graph


class FlagSensitiveTrajectory(Trajectory):
    def __hash__(self):
        return hash(bool(getattr(self, "_art_pickle_strings_interned", False)))


@pytest.mark.parametrize("kind", ["set", "frozenset", "dict", "tuple_key"])
def test_internal_bookkeeping_does_not_mutate_hash_sensitive_model(kind):
    member = FlagSensitiveTrajectory()
    container = container_for(kind, member)
    outer = Trajectory(metadata={"member": member, "container": container})
    compact_memory(outer)
    assert not hasattr(member, "_art_pickle_strings_interned")
    assert contains(kind, container, member)
    restored = pickle.loads(pickle.dumps(outer))
    assert not hasattr(member, "_art_pickle_strings_interned")
    assert contains(kind, container, member)
    assert contains(kind, restored.metadata["container"], restored.metadata["member"])


def test_explicit_fallback_does_not_retain_models():
    child = Trajectory()
    outer = Trajectory(metadata={"member": IdentityLabel(label="x"), "child": child})
    compact_memory(outer)
    refs = weakref.ref(outer), weakref.ref(child)
    del child, outer
    gc.collect()
    assert all(ref() is None for ref in refs)
