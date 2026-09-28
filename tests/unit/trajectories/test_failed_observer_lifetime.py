import gc
from typing import Any, cast
import weakref

import pytest
from test_tokenize import _character_template_history, _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize


@pytest.mark.parametrize(
    "route", ["native", "boundary", "render", "history_factory", "sampled_factory"]
)
@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
def test_failed_tokenization_releases_observer_inputs(route, error_type):
    enabled = gc.isenabled()
    gc.disable()
    try:

        def fail():
            armed = (
                [True]
                if route in ("native", "history_factory", "sampled_factory")
                else []
            )

            class Options(dict):
                def items(self):
                    if armed:
                        armed.pop()
                        raise error_type("public observer failure")
                    return dict.items(self)

            options = Options(key="stable")
            reference = weakref.ref(options)
            try:
                if route == "history_factory":
                    _tokenize._tokenization_context_validator(options)
                elif route == "sampled_factory":
                    exchange = _chat_exchange([1], [2])
                    exchange.request["metadata"] = cast(Any, {"options": options})
                    key = _tokenize._exchange_sampled_source_key(exchange)
                    _tokenize._sampled_source_validator({key: exchange})
                elif route == "boundary":
                    history, tokenizer, _ = _character_template_history()
                    source = history.message_sources[3]
                    assert source is not None
                    source.exchange.request["metadata"] = cast(
                        Any, {"options": options}
                    )
                    decode = tokenizer.decode

                    def arm(tokens, **kwargs):
                        armed.append(True)
                        return decode(tokens, **kwargs)

                    setattr(tokenizer, "decode", arm)
                    history.tokenize(tokenizer=tokenizer)
                else:
                    exchange = _chat_exchange([1], [2])
                    value = tr.Trajectory(
                        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
                    )
                    if route == "native":
                        exchange.request["metadata"] = cast(Any, {"options": options})
                        value.tokenize()
                    else:

                        class Tokenizer:
                            def apply_chat_template(self, messages, **kwargs):
                                armed.append(True)
                                return [1, 2]

                            def __call__(self, text, **kwargs):
                                return [2 if text == "answer" else 1]

                        value.tokenize(
                            tokenizer=cast(Any, Tokenizer()),
                            chat_template="custom",
                            chat_template_kwargs={"options": options},
                        )
            except error_type:
                return reference
            pytest.fail("observer unexpectedly accepted")

        reference = fail()
        assert reference() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_explicit_private_owner_keeps_failure_sticky_until_operation_ends():
    enabled = gc.isenabled()
    gc.disable()
    try:

        def operation():
            armed = []

            class Options(dict):
                def items(self):
                    if armed:
                        armed.pop()
                        raise TypeError("one-shot")
                    return dict.items(self)

            options = Options(key="stable")
            reference = weakref.ref(options)
            validate = _tokenize._tokenization_context_validator(options)
            armed.append(True)
            try:
                validate(True)
            except TypeError as first:
                with pytest.raises(TypeError) as second:
                    validate(True)
                assert second.value is first
                del second
            else:
                pytest.fail("Observer did not fail")
            return reference

        reference = _tokenize._release_context_failures(operation)()
        assert reference() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_cleanup_finalizer_reentry_has_an_independent_failure_owner():
    enabled = gc.isenabled()
    gc.disable()
    references = []
    scopes = []
    try:

        def inner():
            class Options(dict):
                def items(self):
                    raise TypeError("reentrant observer")

            options = Options()
            references.append(weakref.ref(options))
            scopes.append(_tokenize._FAILURE_SCOPE.get())
            _tokenize._tokenization_context_validator(options)

        class ReenteringError(TypeError):
            def __del__(self):
                try:
                    _tokenize._release_context_failures(inner)()
                except TypeError:
                    pass

        def outer():
            class Options(dict):
                def items(self):
                    raise ReenteringError("outer observer")

            scopes.append(_tokenize._FAILURE_SCOPE.get())
            try:
                _tokenize._tokenization_context_validator(Options())
            except TypeError:
                pass

        _tokenize._release_context_failures(outer)()
        assert len(scopes) == 2 and scopes[0] is not scopes[1]
        assert all(
            scope is not None and not scope.active and not scope.states
            for scope in scopes
        )
        assert len(references) == 1 and references[0]() is None
        assert _tokenize._FAILURE_SCOPE.get() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()
