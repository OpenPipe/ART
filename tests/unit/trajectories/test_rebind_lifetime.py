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
