from enum import StrEnum
import pickle

import pytest

import art
from art.trajectories import compact_memory


class State(StrEnum):
    VALUE = "equal provider metadata value"


class TaggedString(str):
    tag: list[str]


@pytest.mark.parametrize("string_type", [State, TaggedString])
@pytest.mark.parametrize("subclass_first", [False, True])
@pytest.mark.parametrize("as_keys", [False, True])
@pytest.mark.parametrize("use_pickle", [False, True])
def test_interning_preserves_string_subclasses(
    string_type: type[str], subclass_first: bool, as_keys: bool, use_pickle: bool
) -> None:
    plain = State.VALUE.value.encode().decode()
    rich = string_type(plain)
    if isinstance(rich, TaggedString):
        rich.tag = ["retained state"]
    ordered = [rich, plain] if subclass_first else [plain, rich]
    items = [{value: value} for value in ordered] if as_keys else ordered
    trajectory = art.Trajectory(metadata={"items": items, "alias": rich})

    restored = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )

    for graph in (trajectory, restored):
        actual = graph.metadata["items"]
        values = [next(iter(item)) for item in actual] if as_keys else actual
        actual_rich, actual_plain = values if subclass_first else reversed(values)
        assert type(actual_plain) is str
        assert type(actual_rich) is string_type
        assert actual_plain == actual_rich == plain
        assert actual_plain is not actual_rich
        assert graph.metadata["alias"] is actual_rich
        if isinstance(actual_rich, TaggedString):
            assert actual_rich.tag == ["retained state"]
        else:
            assert actual_rich is State.VALUE
        if as_keys:
            assert all(next(iter(item)) is next(iter(item.values())) for item in actual)
    assert trajectory.metadata["alias"] is rich


@pytest.mark.parametrize("use_pickle", [False, True])
def test_equal_subclasses_keep_distinct_state_while_plain_strings_share(
    use_pickle: bool,
) -> None:
    first, second = TaggedString(State.VALUE.value), TaggedString(State.VALUE.value)
    first.tag, second.tag = ["first"], ["second"]
    plain = [State.VALUE.value.encode().decode() for _ in range(2)]
    assert plain[0] is not plain[1]
    trajectory = art.Trajectory(
        metadata={"rich": [first, second, first], "plain": plain}
    )

    restored = (
        pickle.loads(pickle.dumps(trajectory))
        if use_pickle
        else compact_memory(trajectory)
    )

    for graph in (trajectory, restored):
        rich, shared = graph.metadata["rich"], graph.metadata["plain"]
        assert rich[0] is rich[2]
        assert rich[0] is not rich[1]
        assert rich[0].tag == ["first"]
        assert rich[1].tag == ["second"]
        assert all(type(value) is TaggedString for value in rich)
        assert type(shared[0]) is str
        assert shared[0] is shared[1]
