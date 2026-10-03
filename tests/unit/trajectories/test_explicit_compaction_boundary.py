from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import ClassVar

from pydantic import BaseModel, ConfigDict
import pytest

from art.trajectories import Trajectory, TrajectoryGroup, _serialization, compact_memory


class OutsideOwner(BaseModel):
    model_config = ConfigDict(frozen=True)
    child: Trajectory
    forbidden: ClassVar[bool] = False

    def __hash__(self) -> int:
        if self.forbidden:
            raise AssertionError("unexpected user hash")
        return id(self.child.metadata["label"])


class ModulelessOpaque:
    pass


setattr(ModulelessOpaque, "__module__", None)


@dataclass
class ModulelessData:
    label: str


setattr(ModulelessData, "__module__", None)


@pytest.mark.parametrize("raw_alias", [False, True])
@pytest.mark.parametrize("protect_first", [False, True])
@pytest.mark.parametrize("container", [set, dict])
def test_pickle_cannot_mutate_another_owners_hash_state(
    raw_alias, protect_first, container
):
    seed = "outside owner string identity must remain unchanged"
    child = Trajectory(metadata={"label": seed.encode().decode()})
    owner = OutsideOwner(child=child)
    members = {owner} if container is set else {owner: "retained"}
    if protect_first:
        compact_memory(Trajectory(metadata={"members": members, "child": child}))
    before = child.metadata["label"]
    later = Trajectory(
        metadata={"seed": seed, "alias": child.metadata if raw_alias else child}
    )
    try:
        OutsideOwner.forbidden = True
        pickle.dumps(later)
    finally:
        OutsideOwner.forbidden = False
    assert child.metadata["label"] is before
    assert owner in members
    assert next(iter(members)) is owner


@pytest.mark.parametrize("mode", ["bare", "mapping", "pickle"])
@pytest.mark.parametrize("kind", [ModulelessOpaque, ModulelessData])
def test_moduleless_objects_are_opaque(mode, kind):
    value = kind("preserved") if kind is ModulelessData else kind()
    if mode == "bare":
        assert compact_memory(value) is value
    elif mode == "mapping":
        graph = {"opaque": value, "state": ["preserved"]}
        assert compact_memory(graph) is graph
        assert graph["opaque"] is value
    else:
        assert type(pickle.loads(pickle.dumps(value))) is kind
        result = pickle.loads(pickle.dumps(Trajectory(metadata={"value": value})))
        assert type(result.metadata["value"]) is kind


@pytest.mark.parametrize("depth", [8, 16, 32, 64])
def test_private_art_chain_pickle_does_not_rescan_graph(depth, monkeypatch):
    root = current = TrajectoryGroup()
    for _ in range(depth - 1):
        child = TrajectoryGroup()
        current._prepared_training_batch = child
        current = child

    def forbidden(*args, **kwargs):
        raise AssertionError("implicit graph preflight")

    monkeypatch.setattr(_serialization, "_intern_strings", forbidden)
    restored = pickle.loads(pickle.dumps(root))
    count = 1
    while restored._prepared_training_batch is not None:
        restored = restored._prepared_training_batch
        count += 1
    assert count == depth


def test_explicit_repeat_compacts_new_strings_and_keeps_cycles():
    text = "caller explicitly owns this mutable graph"
    items: list[object] = [text.encode().decode()]
    graph = {"seed": text, "items": items, "alias": items}
    items.append(graph)
    for _ in range(2):
        compact_memory(graph)
        assert items[0] is text
        assert items[1] is graph and graph["alias"] is items
        items[0] = text.encode().decode()
