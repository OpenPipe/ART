import pickle

import pytest
from pydantic import BaseModel, ConfigDict

from art.trajectories import Trajectory, compact_memory


class HashOnce:
    def __init__(self):
        self.calls = 0

    def __hash__(self):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("opaque member refuses rehash")
        return 17


class AuditedDict(dict):
    writes = 0

    def __setitem__(self, key, value):
        self.writes += 1
        super().__setitem__(key, value)


class AuditedList(list):
    writes = 0

    def __setitem__(self, key, value):
        self.writes += 1
        super().__setitem__(key, value)


class AuditedSet(set):
    writes = 0

    def clear(self):
        self.writes += 1
        super().clear()


class NoIterationList(list):
    def __iter__(self):
        raise RuntimeError("custom iterator must not run")


def fresh(value):
    return value.encode().decode()


@pytest.mark.parametrize("mixed", [False, True])
def test_opaque_set_member_is_not_rehashed_or_lost(mixed):
    member = HashOnce()
    values = {member}
    if mixed:
        values.add("retained string")
    before = list(values)
    trajectory = Trajectory(metadata={"values": values, "alias": values})

    compact_memory(trajectory)

    assert trajectory.metadata["values"] is trajectory.metadata["alias"] is values
    assert len(values) == len(before)
    assert any(item is member for item in values)
    assert member.calls == 1


@pytest.mark.parametrize("kind", [AuditedDict, AuditedList, AuditedSet])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_container_subclass_does_not_run_mutators_or_change_state(kind, use_pickle):
    value = (
        kind({"item": fresh("shared value")})
        if kind is AuditedDict
        else kind([fresh("shared value")])
    )
    value.writes = 0
    before = pickle.dumps(value)
    trajectory = Trajectory(
        metadata={"seed": "shared value", "first": value, "alias": value}
    )

    if use_pickle:
        restored = pickle.loads(pickle.dumps(trajectory))
        assert restored.metadata["first"].writes == 0
        assert restored.metadata["first"] is restored.metadata["alias"]
    else:
        compact_memory(trajectory)

    assert value.writes == 0
    assert pickle.dumps(value) == before
    assert trajectory.metadata["first"] is trajectory.metadata["alias"] is value


def test_opaque_list_iterator_is_not_called():
    value = NoIterationList([1, 2, 3])
    assert compact_memory(value) is value


def test_mixed_mapping_never_rehashes_opaque_keys_and_visits_mutable_values():
    member = HashOnce()
    canonical = "shared dictionary descendant"
    child = [fresh(canonical)]
    value = {member: child, "direct": fresh(canonical)}
    trajectory = Trajectory(
        metadata={"seed": canonical, "first": value, "alias": value}
    )
    before = list(value)

    compact_memory(trajectory)

    assert list(value) == before
    assert member.calls == 1
    assert child[0] is canonical
    assert value["direct"] == canonical and value["direct"] is not canonical
    assert trajectory.metadata["first"] is trajectory.metadata["alias"] is value


def test_string_set_preserves_members_and_identity():
    canonical = "shared exact string set value"
    value = {fresh(canonical)}
    trajectory = Trajectory(
        metadata={"seed": canonical, "values": value, "alias": value}
    )
    compact_memory(trajectory)
    assert next(iter(value)) == canonical
    assert next(iter(value)) is not canonical
    assert trajectory.metadata["values"] is trajectory.metadata["alias"] is value


def test_opaque_subclass_cycle_stays_connected_to_supported_graph():
    canonical = "shared alias cycle string"
    child = [fresh(canonical)]
    value = AuditedDict(child=child)
    child.append(value)
    value.writes = 0
    trajectory = Trajectory(
        metadata={"seed": canonical, "opaque": value, "child": child}
    )
    for _ in range(2):
        compact_memory(trajectory)
        assert child[1] is value and value["child"] is child
        assert child[0] is canonical
        assert value.writes == 0


@pytest.mark.parametrize("strings", [False, True])
def test_set_table_order_and_serialized_bytes_are_unchanged(strings):
    values: set[str | int] = {str(i) if strings else i for i in range(200)}
    retained = list(values)[::37]
    for value in list(values):
        if value not in retained:
            values.remove(value)
    trajectory = Trajectory(metadata={"values": values})
    before_order = list(values)
    before_json = trajectory.model_dump_json()
    before_pickle = pickle.dumps(values)
    compact_memory(trajectory)
    assert list(values) == before_order
    assert trajectory.model_dump_json() == before_json
    assert pickle.dumps(values) == before_pickle
    pickle.dumps(trajectory)
    assert list(values) == before_order
    assert trajectory.model_dump_json() == before_json


class FrozenLabel(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str


@pytest.mark.parametrize("kind", [set, frozenset])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_hashable_model_descendants_preserve_member_aliases(kind, use_pickle):
    canonical = "shared frozen model label"
    member = FrozenLabel(label=fresh(canonical))
    values = kind([member])
    trajectory = Trajectory(
        metadata={"seed": canonical, "values": values, "member": member}
    )
    before_hash = hash(member)
    result = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )
    for graph in (trajectory, result):
        actual = graph.metadata["member"]
        assert next(iter(graph.metadata["values"])) is actual
        assert actual.label is graph.metadata["seed"]
        assert hash(actual) == before_hash
    assert next(iter(values)) is member
