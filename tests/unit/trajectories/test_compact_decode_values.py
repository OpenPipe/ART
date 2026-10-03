from __future__ import annotations

import copy
import random

import pytest

from art.trajectories import Trajectory, compact_dump, compact_validate
from art.trajectories._compact import _decode_value


class Integer(int):
    pass


class Floating(float):
    pass


class String(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        0,
        -1,
        2**200,
        0.0,
        -0.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        Integer(3),
        Floating(4.5),
        String("literal"),
        "$0",
        "ordinary",
    ],
)
def test_compact_scalar_decode_preserves_identity(value: object) -> None:
    assert _decode_value(value, {}) is value


def test_compact_nested_numeric_decode_preserves_values_and_container_ownership() -> (
    None
):
    rng = random.Random(148)
    scalars = [None, False, True, 0, -1, 2**200, 0.0, -0.0, 1.25, "$0", "text"]

    def tree(depth: int) -> object:
        if not depth or rng.random() < 0.45:
            return rng.choice(scalars)
        if rng.random() < 0.5:
            return [tree(depth - 1) for _ in range(rng.randrange(6))]
        return {f"key{i}": tree(depth - 1) for i in range(rng.randrange(6))}

    for _ in range(150):
        value = tree(5)
        original = copy.deepcopy(value)
        decoded = _decode_value(value, {})
        assert decoded == original
        assert value == original
        if isinstance(value, (list, dict)):
            assert decoded is not value

    shared = {"numbers": [False, True, None, 1, 2.5]}
    value = [shared, shared]
    first = _decode_value(value, {})
    second = _decode_value(value, {})
    assert isinstance(first, list) and isinstance(second, list)
    assert first[0] is not first[1] and first[0] is not second[0]
    first_item = first[0]
    assert isinstance(first_item, dict)
    numbers = first_item["numbers"]
    assert isinstance(numbers, list)
    numbers.append(7)
    assert second == value
    assert first[1] == shared


@pytest.mark.parametrize("value", [b"invalid", (1, 2), {1}, 1j, {1: "bad key"}])
def test_compact_decode_still_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ValueError):
        _decode_value(value, {})


def test_compact_validation_keeps_repeated_decodes_independent() -> None:
    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "public question"},
            {"role": "assistant", "content": "public response"},
        ],
        metadata={"numbers": [None, False, True, 1, 2.5]},
    )
    payload = compact_dump(trajectory)
    original = copy.deepcopy(payload)
    first = compact_validate(payload, type=Trajectory)
    second = compact_validate(payload, type=Trajectory)
    assert first.model_dump() == second.model_dump() == trajectory.model_dump()
    first.metadata["numbers"].append(6)
    assert payload == original
    assert second.model_dump() == trajectory.model_dump()
    assert compact_dump(second) == original
