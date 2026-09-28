from dataclasses import dataclass
import gc
from typing import Any
import weakref

import pytest
from test_tokenize import _chat_exchange

from art import trajectories as tr
from art.trajectories import _serialization, _tokenize


class Options(dict):
    pass


@pytest.fixture
def no_cyclic_gc():
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def _success(route: str, hold: str):
    value = Options(key=["stable"])
    reference = weakref.ref(value)
    exchange = _chat_exchange([1], [2])
    exchange.request["metadata"] = {"shared_a": value, "shared_b": value}
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    if route == "history":
        history = trajectory.histories()[0]
        assert isinstance(history, tr.ChatCompletionsHistory)
        result = history.tokenize()
        tokenized = result
    elif route == "trajectory":
        result = trajectory.tokenize()
        tokenized = result
    elif route == "trace":
        result = _tokenize._tokenize_trajectory_with_trace(trajectory)
        tokenized = result[0].histories[0]
    else:
        result = tr.TrajectoryGroup([trajectory]).tokenize()
        tokenized = result.trajectories[0]
    assert tokenized.tokens == [1, 2]
    history = tokenized.history
    assert isinstance(history, tr.ChatCompletionsHistory)
    sources = [source for source in history.message_sources if source is not None]
    assert sources and all(source.exchange is exchange for source in sources)
    assert exchange.request["metadata"]["shared_a"] is value
    assert exchange.request["metadata"]["shared_b"] is value
    holders = [trajectory] if hold == "input" else [result] if hold == "output" else []
    return reference, holders


@pytest.mark.parametrize("route", ["history", "trajectory", "trace", "group"])
@pytest.mark.parametrize("hold", ["none", "input", "output"])
def test_success_releases_sources_when_callers_release_them(no_cyclic_gc, route, hold):
    reference, holders = _success(route, hold)
    assert (reference() is not None) == (hold != "none")
    holders.clear()
    assert reference() is None


def test_repeated_success_does_not_accumulate_source_graphs(no_cyclic_gc):
    references = [_success("trajectory", "none")[0] for _ in range(32)]
    assert all(reference() is None for reference in references)


class ObservationError(RuntimeError):
    pass


@dataclass
class CallbackSidecar:
    child: Any
    events: list[str]
    failure: bool = False

    def __getattribute__(self, name):
        if name == "child":
            object.__getattribute__(self, "events").append("child")
            if object.__getattribute__(self, "failure"):
                raise ObservationError("sidecar observation failed")
        return object.__getattribute__(self, name)


def _callback_trial(fail: bool, retain_error: bool):
    value = Options(key=["stable"])
    reference = weakref.ref(value)
    exchange = _chat_exchange([1], [2])
    exchange.request["metadata"] = {"shared": value}
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    events = []
    sidecar = CallbackSidecar(exchange, events, fail)
    holders = []
    error_reference = None
    try:
        _serialization._rebind_history_sources(sidecar, trajectory)
    except ObservationError as error:
        assert fail
        assert str(error) == "sidecar observation failed"
        error_reference = weakref.ref(error)
        if retain_error:
            holders.append(error)
    else:
        assert not fail
        assert object.__getattribute__(sidecar, "child") is exchange
    return reference, error_reference, holders, events


@pytest.mark.parametrize(
    "fail,retain_error", [(False, False), (True, False), (True, True)]
)
def test_rebinding_preserves_callbacks_and_error_ownership(
    no_cyclic_gc, fail, retain_error
):
    reference, error_reference, holders, events = _callback_trial(fail, retain_error)
    assert events == ["child"]
    assert (reference() is not None) == retain_error
    if error_reference is not None:
        assert (error_reference() is not None) == retain_error
    holders.clear()
    assert reference() is None
    if error_reference is not None:
        assert error_reference() is None


def test_finalizer_can_reenter_source_rebinding(no_cyclic_gc):
    references = []
    events = []
    errors = []

    class Payload(dict):
        kind: str

        def __del__(self):
            try:
                events.append(self.kind)
                if self.kind == "outer":
                    _serialization._rebind_history_sources([], make("nested"))
            except BaseException as error:
                errors.append(type(error).__name__)

    def make(kind: str) -> tr.Trajectory:
        value = Payload(key=["stable"])
        value.kind = kind
        references.append(weakref.ref(value))
        exchange = _chat_exchange([1], [2])
        exchange.request["metadata"] = {"shared": value}
        return tr.Trajectory(
            exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
        )

    _serialization._rebind_history_sources([], make("outer"))

    assert events == ["outer", "nested"]
    assert errors == []
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@dataclass
class CopiedSource:
    exchange: tr.ChatCompletionsExchange


def _actual_rebind(mode: str):
    canonical = _chat_exchange([1], [2])
    canonical.request["metadata"] = {"tracked": Options(key=["stable"])}
    copied = canonical.model_copy(deep=True)
    canonical_ref = weakref.ref(canonical)
    copied_ref = weakref.ref(copied)
    sidecar = CopiedSource(copied)
    target = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[canonical])
    )
    if mode == "position":
        source = tr.Trajectory(
            exchanges=tr.TrajectoryExchanges(chat_completions=[copied])
        )
        _serialization._rebind_history_sources(
            sidecar, target, source_trajectory=source
        )
    elif mode == "unfixed":
        _serialization._rebind_history_sources(sidecar)
    else:
        if mode == "nan":
            canonical.request["temperature"] = float("nan")
            copied.request["temperature"] = float("nan")
            assert canonical != copied
        _serialization._rebind_history_sources(sidecar, target)
    assert sidecar.exchange is (copied if mode == "unfixed" else canonical)
    return canonical_ref, copied_ref, sidecar


@pytest.mark.parametrize("mode", ["position", "equal", "nan", "unfixed"])
def test_actual_rebinding_releases_detached_and_canonical_sources(no_cyclic_gc, mode):
    canonical, copied, sidecar = _actual_rebind(mode)
    if mode == "unfixed":
        assert canonical() is None
        assert sidecar.exchange is copied()
    else:
        assert copied() is None
        assert sidecar.exchange is canonical()
    del sidecar
    assert canonical() is None
    assert copied() is None
