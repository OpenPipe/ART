"""The calibration harness steers every rank's next forward from world-wide
compile telemetry.

Issue #840: the warm-up loop stopped when the *local* rank's forward was
compile-free. One CP rank whose local shapes still recompiled ran an extra
warm-up of the previous layout while its peers moved on to the next one, so the
context-parallel all-to-alls paired different layouts and deadlocked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_DRIVER = (
    Path(__file__).resolve().parents[2] / "dev" / "trainer_rank_landing_acceptance.py"
)
_spec = importlib.util.spec_from_file_location(
    "trainer_rank_landing_acceptance", _DRIVER
)
assert _spec is not None and _spec.loader is not None
driver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = driver
_spec.loader.exec_module(driver)


class _Watch:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)

    def take(self) -> list[str]:
        statuses, self._statuses = self._statuses, []
        return statuses


def test_world_compile_statuses_merges_every_rank(monkeypatch) -> None:
    # Rank 3 recompiled while ranks 0-2 were compile-free: the warm-up must
    # continue on all four ranks.
    per_rank = [["none"], ["none"], ["none"], ["recompile"]]
    gathered: list[list[str]] = []

    def fake_gather(value, group=None):
        gathered.append(value)
        return per_rank

    monkeypatch.setattr(driver, "_gather_objects", fake_gather)
    statuses = driver._world_compile_statuses(_Watch(["none"]))
    assert gathered == [["none"]], "the local statuses must be gathered"
    assert statuses == ["none", "recompile"]
    assert not driver._warmup_complete(attempt=1, statuses=statuses)


def test_warmup_complete_needs_min_warmups_and_no_compile_anywhere() -> None:
    assert not driver._warmup_complete(0, ["none"])
    assert driver._warmup_complete(1, ["none"])
    assert not driver._warmup_complete(1, [])
    assert not driver._warmup_complete(7, ["none", "recompile"])
    assert driver._merge_rank_statuses([["recompile", "none"], ["none"]]) == [
        "none",
        "recompile",
    ]


def test_every_recorded_compile_status_is_world_wide() -> None:
    source = _DRIVER.read_text()
    assert '"compile_statuses": watch.take()' not in source
    assert source.count('"compile_statuses": _world_compile_statuses(watch)') == 2


class _Tokenizer:
    def __init__(self, size: int, salt: str = "") -> None:
        self._size = size
        self._salt = salt

    def __len__(self) -> int:
        return self._size

    def decode(self, ids: list[int]) -> str:
        return self._salt + " ".join(str(i) for i in ids)


def test_corpus_tokenizer_check_requires_the_same_tokenizer(monkeypatch) -> None:
    import transformers

    corpus = {
        "tokenizer_model": "Qwen/Qwen3-0.6B",
        "groups": [{"histories": [{"tokens": list(range(1, 300))}]}],
    }
    tokenizers = {
        "Qwen/Qwen3-0.6B": _Tokenizer(151_669),
        "Qwen/Qwen3-8B": _Tokenizer(151_669),
        "Qwen/Qwen3.5-4B": _Tokenizer(248_320, salt="other:"),
    }
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, name, **kwargs: tokenizers[name]),
    )
    assert driver._check_corpus_tokenizer(corpus, "Qwen/Qwen3-8B") == {
        "corpus_tokenizer": "Qwen/Qwen3-0.6B",
        "vocabulary": 151_669,
    }
    with pytest.raises(SystemExit):
        driver._check_corpus_tokenizer(corpus, "Qwen/Qwen3.5-4B")


def test_qwen3_ellavox_cells_use_the_qwen3_corpus() -> None:
    assert driver.CALIBRATION_CORPUS_BY_CELL == {
        "cal-ellavox": "qwen35",
        "cal-ellavox-qwen3": "qwen3",
    }
    assert set(driver.ELLAVOX_CORPORA) == {"qwen35", "qwen3"}
    for path, digest in driver.ELLAVOX_CORPORA.values():
        assert path.name.startswith("_trainer_rank_ellavox_") and len(digest) == 64
