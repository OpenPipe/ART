"""Canonical report bytes and bounded rejection of large token inventories."""

import json
import random

import pytest

from art.trainer_rank import _planner_misses as reports


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"empty": [[], (), {}]},
        {"tokens": list(range(4097)), "targets": [-1] * 1025},
        {"tokens": [-(1 << 63), (1 << 63) - 1, 1 << 63, -(1 << 63) - 1]},
        {"tokens": [1 << 2000, -(1 << 2000)]},
        {"values": [True, False, None, 1.25, -0.0, 1e-100]},
        {"text": ['"\\\n\t', "é漢字😀", "\ud800", "\udfff"]},
        {"nest": [{"b": tuple(range(1100)), "a": [3, "x", None]}]},
        {3: "number keys", 1: "legacy fallback"},
    ],
)
def test_identical_canonical_bytes_and_exact_limit(value):
    expected = canonical(value)
    assert reports._encode(value, limit=len(expected)) == expected
    with pytest.raises(reports._ReportTooLarge):
        reports._encode(value, limit=len(expected) - 1)


def test_shared_containers_are_not_cycles():
    shared = {"tokens": list(range(2049))}
    value = {"requests": [shared, shared], "layout": shared}
    assert reports._encode(value) == canonical(value)


@pytest.mark.parametrize("kind", ["list", "dict", "indirect"])
def test_cycles_refuse(kind):
    value = {}
    if kind == "list":
        sequence = []
        sequence.append(sequence)
        value["cycle"] = sequence
    else:
        value["cycle"] = value if kind == "dict" else ([], value)
    with pytest.raises(ValueError, match="Circular reference"):
        reports._encode(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_refuse(value):
    with pytest.raises(ValueError):
        reports._encode({"tokens": [*range(1024), value]})


@pytest.mark.parametrize("value", [object(), {("invalid",): 1}, {1: 1, "key": 2}])
def test_serialization_errors_refuse(value):
    with pytest.raises(TypeError):
        reports._encode({"replay": value})


def test_subclasses_keep_standard_encoder_semantics():
    class Number(int):
        def __repr__(self):
            raise AssertionError("JSON must use the integer representation")

    class Sequence(list):
        pass

    class Mapping(dict):
        pass

    value = {"subclasses": Sequence([Number(123), Mapping({"x": [Number(456)]})])}
    assert reports._encode(value) == canonical(value)


def test_sorted_mapping_snapshots_values_before_nested_iterator_mutates_them():
    def record():
        root = {}

        class MutatingList(list):
            def __iter__(self):
                root["z"] = 2
                return super().__iter__()

        root.update(a=MutatingList([3]), z=1)
        return root

    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = ("".join(encoder.iterencode(record())) + "\n").encode()
    assert reports._encode(record()) == expected


@pytest.mark.parametrize("operation", ["replace", "append", "shrink"])
@pytest.mark.parametrize("prefix", [0, 1024])
def test_nested_iterator_mutations_preserve_native_sequence_iteration(
    operation, prefix
):
    def record():
        outer = [0] * prefix

        class MutatingList(list):
            def __iter__(self):
                if operation == "replace":
                    outer[prefix + 1] = 99
                elif operation == "append":
                    outer.append(99)
                else:
                    del outer[prefix + 1 :]
                return super().__iter__()

        outer.extend([MutatingList([0]), 1, 2])
        return {"values": outer}

    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = ("".join(encoder.iterencode(record())) + "\n").encode()
    assert reports._encode(record()) == expected


def test_fast_chunks_are_bounded_and_oversize_stops_before_later_errors(monkeypatch):
    original = json.JSONEncoder.encode
    blocks = []

    def bounded(self, value):
        if type(value) in (list, tuple):
            assert len(value) <= 1024
            assert all(
                type(item) is int and -(1 << 63) <= item < 1 << 63 for item in value
            )
            blocks.append(len(value))
        return original(self, value)

    monkeypatch.setattr(json.JSONEncoder, "encode", bounded)
    value = {"tokens": [0] * 100_000 + [object()]}
    with pytest.raises(reports._ReportTooLarge):
        reports._encode(value, limit=64)
    assert blocks == [1024]
    blocks.clear()
    value = {"tokens": [-(1 << 63)] * 2500}
    expected = json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(value)
    expected = ("".join(expected) + "\n").encode()
    assert reports._encode(value) == expected
    assert blocks == [1024, 1024, 452]


def test_deep_input_refuses_without_persistence():
    value = []
    for _ in range(2000):
        value = [value]
    with pytest.raises(RecursionError):
        reports._encode({"deep": value})


def test_varied_nested_payloads_match_standard_encoder():
    randomizer = random.Random(741)

    def value(depth):
        if not depth:
            return randomizer.choice([None, True, -0.0, "\\😀\n", -(1 << 63), 12345])
        choice = randomizer.randrange(4)
        if choice == 0:
            return {str(i): value(depth - 1) for i in range(randomizer.randrange(6))}
        if choice == 1:
            return [value(depth - 1) for _ in range(randomizer.randrange(6))]
        if choice == 2:
            return tuple(
                randomizer.randrange(-100000, 100000)
                for _ in range(randomizer.randrange(2050))
            )
        return value(0)

    for _ in range(100):
        record = {"replay": value(3)}
        assert reports._encode(record) == canonical(record)
