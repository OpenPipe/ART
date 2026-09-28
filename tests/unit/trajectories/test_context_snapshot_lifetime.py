from collections import UserDict
import gc
from typing import cast
import weakref

import pytest

from art.trajectories._tokenize import (
    _tokenization_context,
    _tokenization_context_validator,
)


class _Options(dict):
    opaque: object


class _Slots(dict):
    __slots__ = ("revision", "__weakref__")


@pytest.fixture
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("kind", [_Options, _Slots, UserDict])
def test_completed_snapshot_releases_input_without_cyclic_gc(kind, no_cyclic_gc):
    references = []
    snapshots = []
    for _ in range(32):
        value = kind(key=["stable"])
        value.revision = ["before"]
        references.append(weakref.ref(value))
        snapshots.append(
            cast(
                tuple[type, tuple[object, object]],
                _tokenization_context([value, value]),
            )
        )
        del value
    assert all(reference() is None for reference in references)
    assert all(value[1][0] is value[1][1] for value in snapshots)
    assert all(value == snapshots[0] for value in snapshots)


def test_caller_memo_retains_ownership_until_caller_clears_it(no_cyclic_gc):
    value = _Options(key=["stable"])
    reference = weakref.ref(value)
    observed = {}
    snapshot = _tokenization_context(value, _observed=observed)
    identity = id(value)
    assert observed[identity] == (value, snapshot)
    del value
    assert reference() is not None
    assert observed[identity][1] is snapshot
    observed.clear()
    assert reference() is None
    assert snapshot[1][0][0] == (str, "key")


@pytest.mark.parametrize("partial", [False, True])
def test_failed_snapshot_releases_input_after_error_leaves_scope(partial, no_cyclic_gc):
    value = _Options(key=["stable"])
    if not partial:
        value.opaque = object()
    reference = weakref.ref(value)

    def fail():
        try:
            _tokenization_context([value, object()] if partial else value)
        except TypeError:
            return
        pytest.fail("Opaque context unexpectedly accepted")

    fail()
    del value
    assert reference() is None


def test_validator_owns_input_only_until_validator_is_released(no_cyclic_gc):
    value = _Options(key=["stable"])
    reference = weakref.ref(value)
    validate = _tokenization_context_validator(value)
    validate(True)
    del value
    assert reference() is not None
    del validate
    assert reference() is None
