"""Generated reports keep the external transport contract without re-encoding."""

import json

import pytest
from test_planner_retention_budget import emit, ledger, limits

from art.trainer_rank import _planner_misses as reports


class UnorderedString(str):
    def __lt__(self, other):
        return False


class DuplicateItems(dict):
    def items(self):
        return [("same", 1), ("same", 2)]


@pytest.mark.parametrize(
    "payload,retained",
    [
        ({"a": [1, 2.0, False, None]}, True),
        ({"a": (1, 2, 3)}, True),
        ({1: "one", 2: "two"}, True),
        ({2: "two", 10: "ten"}, False),
        ({False: 1, True: 2}, True),
        ({None: 1}, True),
        ({1.0: 1, 2.5: 2}, True),
        ({"\ud800\udc00": 1, "\ue000": 2}, False),
        ({"\ud800\udc00": 1, "\U00010000": 2}, False),
        ({UnorderedString("z"): 1, UnorderedString("a"): 2}, False),
        (DuplicateItems(seed=True), False),
        ({"a": "\ud800\udc00/\n\u0000\udfff\u2028"}, True),
        ({"a": [0.0, -0.0, 5e-324, -5e-324, 1.7976931348623157e308]}, True),
    ],
)
def test_factory_outputs_keep_public_canonical_acceptance(
    tmp_path, monkeypatch, payload, retained
):
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    encoded = []
    encode = reports._encode

    def capture(record, **kwargs):
        raw = encode(record, **kwargs)
        encoded.append(raw)
        return raw

    monkeypatch.setattr(reports, "_encode", capture)
    bound = limits(tmp_path)
    reporter = reports.Reporter(5)
    with reports.report_retention_scope(bound):
        path = emit(reporter, replay_factory=lambda: {"nested": payload})
    assert (path is not None) == retained
    # The first encoding is the actual producer output, before any validation.
    raw = encoded[0]
    if retained:
        reports.validate_report(raw)
        assert path.read_bytes() == raw
        assert sum(x[1] for x in ledger(bound)["charges"].values()) == len(raw)
        assert ledger(bound)["omitted"] == reporter.failures == 0
    else:
        with pytest.raises(ValueError):
            reports.validate_report(raw)
        assert not list(bound.spool_dir.glob("*.json"))
        assert reporter.failures == 1


@pytest.mark.parametrize("payload", [{1: 1, "1": 2}, {"a": float("nan")}])
def test_factory_encoding_failure_still_retains_original_fallback(
    tmp_path, monkeypatch, payload
):
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    path = emit(
        reports.Reporter(5, spool_dir=tmp_path / "reports"),
        replay_factory=lambda: payload,
        oom=True,
        observed_peak_bytes=None,
        partial_peak_bytes=99,
    )
    record = reports.validate_report(path.read_bytes())
    assert record["oom"] and record["partial_peak_bytes"] == 99
    assert record["replay"] is None and not record["replay_complete"]
    assert record["incomplete_reasons"][0].startswith("replay unavailable: ")


def test_generated_report_encodes_and_parses_once(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    calls = {"encode": 0, "decode": 0}
    encode, decode = reports._encode, json.loads

    def counted_encode(*args, **kwargs):
        calls["encode"] += 1
        return encode(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        calls["decode"] += 1
        return decode(*args, **kwargs)

    monkeypatch.setattr(reports, "_encode", counted_encode)
    monkeypatch.setattr(json, "loads", counted_decode)
    path = emit(reports.Reporter(5, spool_dir=tmp_path / "reports"))
    assert path is not None
    assert calls == {"encode": 1, "decode": 1}
    # Public delivery remains a separate trust boundary, even for the same bytes.
    reports.persist_report(path.read_bytes(), tmp_path / "external")
    assert calls == {"encode": 2, "decode": 2}


@pytest.mark.parametrize(
    "mutation", ["space", "newline", "escape", "number", "nested_nan"]
)
def test_external_transport_still_requires_full_canonical_validation(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    raw = emit(reports.Reporter(5, spool_dir=tmp_path / "seed")).read_bytes()
    if mutation == "space":
        raw = b" " + raw
    elif mutation == "newline":
        raw = raw[:-1]
    elif mutation == "escape":
        raw = raw.replace(b'"forward"', b'"for\\u0077ard"')
    elif mutation == "number":
        raw = raw.replace(b'"threshold_pct":5', b'"threshold_pct":5e0')
    else:
        raw = raw.replace(b'"replay":{', b'"replay":{"arbitrary":NaN,')
    for function in (
        lambda: reports.validate_report(raw),
        lambda: reports.persist_report(raw, tmp_path / "external"),
    ):
        with pytest.raises(ValueError):
            function()
    assert not (tmp_path / "external").exists()


@pytest.mark.parametrize("name", ["validate_report", "persist_report"])
def test_generated_shortcut_is_not_a_public_flag(tmp_path, name):
    function = getattr(reports, name)
    args = (b"{}\n", tmp_path) if name == "persist_report" else (b"{}\n",)
    with pytest.raises(TypeError):
        function(*args, generated=True)
