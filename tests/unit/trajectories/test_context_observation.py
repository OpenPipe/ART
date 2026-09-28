from collections import UserDict
from typing import Any, cast

import pytest

from art.trajectories._tokenize import _tokenization_context


class ObservedDict(dict):
    reads: int

    def items(self):
        self.reads += 1
        return super().items()


@pytest.mark.parametrize("mapping_type", [dict, UserDict, ObservedDict])
def test_context_preserves_mapping_order_aliases_and_fresh_mutable_reads(mapping_type):
    child = ["before"]
    mapping = mapping_type({"first": child, "second": child})
    if isinstance(mapping, ObservedDict):
        mapping.reads = 0
    original = cast(
        tuple[type, tuple[Any, Any]], _tokenization_context([mapping, mapping])
    )
    first, second = original[1]
    assert first is second
    assert first[0] is mapping_type
    assert [entry[0] for entry in first[1]] == [(str, "first"), (str, "second")]
    assert first[1][0][1] is first[1][1][1]
    if isinstance(mapping, ObservedDict):
        assert mapping.reads == 1
    child[0] = "after"
    assert _tokenization_context([mapping, mapping]) != original
    child[0] = "before"
    assert _tokenization_context([mapping, mapping]) == original
    assert mapping["first"] is mapping["second"] is child


def test_context_rejects_stale_identity_entry_without_borrowing_its_value():
    value = {"request": [1, True, "1"]}
    observed: dict[int, tuple[object, object]] = {id(value): (object(), "stale")}
    assert _tokenization_context(value, _observed=observed) == _tokenization_context(
        value
    )
    assert observed[id(value)][0] is value


@pytest.mark.parametrize(
    "left,right", [(1, True), (0.0, -0.0), ("x", b"x"), (None, "None")]
)
def test_context_retains_typed_scalar_distinctions(left, right):
    assert _tokenization_context({"value": left}) != _tokenization_context(
        {"value": right}
    )


def test_context_preserves_custom_mapping_exception_identity():
    failure = RuntimeError("mapping observation failed")

    class Broken(dict):
        def items(self):
            raise failure

    with pytest.raises(RuntimeError) as caught:
        _tokenization_context(Broken(value=1))
    assert caught.value is failure
