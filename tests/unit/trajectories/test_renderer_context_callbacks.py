from __future__ import annotations

from pickle import PickleBuffer
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize("mutate", [False, True])
def test_render_context_mutation_cannot_be_restored_by_later_encoder(mutate):
    exchange = _chat_exchange([1], [2])
    exchange.request["messages"] = [{"role": "user", "content": "q"}]
    choice = exchange.response.choices[0]
    assert choice.model_extra is not None
    choice.model_extra.pop("prompt_token_ids")
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    before = value.model_dump_json()
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append("render")
            if mutate:
                exchange.request["messages"][0]["content"] = "changed"
                messages[0]["content"] = "changed"
            return [3 if mutate else 1, 2]

        def __call__(self, text, **kwargs):
            calls.append(text)
            if mutate and text == "changed":
                exchange.request["messages"][0]["content"] = "q"
            return [{"q": 1, "changed": 3, "answer": 2}[text]]

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            actual = value.tokenize(
                tokenizer=cast(Any, Tokenizer()), chat_template="custom"
            )
            assert actual.tokens == [3, 2]
            assert actual.logprobs[-1] == -0.2
            assert actual.flags[-1] & tr.TokenFlag.SAMPLED
            assert value.model_dump_json() == before
    else:
        actual = value.tokenize(
            tokenizer=cast(Any, Tokenizer()), chat_template="custom"
        )
        assert actual.tokens == [1, 2]
        assert actual.logprobs[-1] == -0.2
        assert actual.flags[-1] & tr.TokenFlag.SAMPLED
        assert value.model_dump_json() == before
    assert "render" in calls


@pytest.mark.parametrize("mutate", [False, True])
def test_optional_segment_encoder_cannot_swallow_mutated_projection(mutate):
    from test_tokenize import _repeated_text_rerender_history, _RepeatedTextTokenizer

    error = ValueError("encoder capability unavailable")
    calls = []

    class Tokenizer(_RepeatedTextTokenizer):
        working = None

        def apply_chat_template(self, messages, **kwargs):
            if self.working is None:
                self.working = messages
            return super().apply_chat_template(messages, **kwargs)

        def __call__(self, text, **kwargs):
            if "<u>" in text and not kwargs.get("return_offsets_mapping"):
                calls.append(text)
                if mutate:
                    assert self.working is not None
                    self.working[0]["content"] = "changed"
                    raise error
            return super().__call__(text, **kwargs)

    history = _repeated_text_rerender_history(2)
    if mutate:
        with pytest.raises(ValueError) as raised:
            history.tokenize(tokenizer=Tokenizer())
        assert raised.value is error
    else:
        actual = history.tokenize(tokenizer=Tokenizer())
        assert actual.tokens and any(
            flag & tr.TokenFlag.SAMPLED for flag in actual.flags
        )
    assert calls


@pytest.mark.parametrize("api", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("mutate", [False, True])
def test_stop_decision_binds_logprobs_before_encoding(api, mutate):
    from test_tokenize import _CharacterTemplateTokenizer

    from art.trajectories import _tokenize as module

    calls = []
    exchange = _chat_exchange([1], [2, 9])
    choice = exchange.response.choices[0]
    assert choice.model_extra is not None
    assert choice.logprobs is not None and choice.logprobs.content
    choice.model_extra["stop_reason"] = "§"
    rows = choice.logprobs.content
    original = rows[0].logprob

    class Tokenizer(_CharacterTemplateTokenizer):
        def __call__(self, text, **kwargs):
            if text == "§":
                calls.append(True)
                if mutate and len(calls) == 1:
                    rows[0].logprob = -99.0
            return super().__call__(text, **kwargs)

    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    tokenizer = Tokenizer()

    def invoke():
        if api == "history":
            return value.chat_completions_history().tokenize(tokenizer=tokenizer)
        if api == "trace":
            return module._tokenize_trajectory_with_trace(value, tokenizer=tokenizer)[0]
        return value.tokenize(multi_history=True, tokenizer=tokenizer)

    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            invoke()
        assert rows[0].logprob == -99.0
    else:
        actual = invoke()
        history = actual if api == "history" else actual.histories[0]
        assert history.tokens == [1, 2, 9]
        assert history.logprobs[1] == original
    assert calls


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (1, True),
        (1, 1.0),
        (-0.0, 0.0),
        ("😀", "\ud83d\ude00"),
        (b"a", bytearray(b"a")),
        (b"a", PickleBuffer(b"a")),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    ],
)
@pytest.mark.parametrize("mutate", [False, True])
def test_renderer_context_keeps_types_order_and_unicode(before, after, mutate):
    options = {"value": before}
    exchange = _chat_exchange([1], [2])
    exchange.request["messages"] = [{"role": "user", "content": "q"}]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(True)
            assert kwargs["options"] is options
            if mutate:
                options["value"] = after
            return [3 if mutate else 1, 2]

        def __call__(self, text, **kwargs):
            # A later callback must not hide a changed rendering argument.
            if mutate:
                options["value"] = before
            return [2 if text == "answer" else 3 if mutate else 1]

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            value.tokenize(
                tokenizer=cast(Any, Tokenizer()),
                chat_template="custom",
                chat_template_kwargs={"options": options},
            )
    else:
        actual = value.tokenize(
            tokenizer=cast(Any, Tokenizer()),
            chat_template="custom",
            chat_template_kwargs={"options": options},
        )
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
    assert calls


@pytest.mark.parametrize("lookup", ["property", "custom"])
@pytest.mark.parametrize("mutate", [False, True])
def test_tokenizer_attribute_callbacks_keep_projection_guard(lookup, mutate):
    from test_tokenize import _repeated_text_rerender_history, _RepeatedTextTokenizer

    calls = []

    class Tokenizer(_RepeatedTextTokenizer):
        working = None

        def apply_chat_template(self, messages, **kwargs):
            if self.working is None:
                self.working = messages
            return super().apply_chat_template(messages, **kwargs)

        def observe(self):
            if self.working is not None:
                calls.append(True)
                if mutate:
                    self.working[0]["content"] = "changed"
            return None

    if lookup == "property":
        setattr(Tokenizer, "eos_token_id", property(lambda self: self.observe()))
    else:

        def read(self, name):
            if name == "eos_token_id":
                return self.observe()
            return object.__getattribute__(self, name)

        Tokenizer.__getattribute__ = read
    history = _repeated_text_rerender_history(2)
    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            history.tokenize(tokenizer=Tokenizer())
    else:
        actual = history.tokenize(tokenizer=Tokenizer())
        assert actual.tokens and any(
            flag & tr.TokenFlag.SAMPLED for flag in actual.flags
        )
    assert calls


@pytest.mark.parametrize("shape", ["subclass", "cycle"])
@pytest.mark.parametrize("mutate", [False, True])
def test_render_context_rich_fallback_never_invokes_reducers(shape, mutate):
    class Text(str):
        state: int

        def __reduce_ex__(self, protocol):
            raise AssertionError("render guard must not call custom reducers")

    text = Text("stable")
    text.state = 1
    options: dict[str, Any] = {"value": text if shape == "subclass" else []}
    exchange = _chat_exchange([1], [2])
    exchange.request["messages"] = [{"role": "user", "content": "q"}]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(True)
            if mutate:
                if shape == "subclass":
                    text.state = 2
                else:
                    options["value"].append(options["value"])
            return [3 if mutate else 1, 2]

        def __call__(self, encoded, **kwargs):
            text.state = 1
            if shape == "cycle":
                options["value"].clear()
            return [2 if encoded == "answer" else 3 if mutate else 1]

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            value.tokenize(
                tokenizer=cast(Any, Tokenizer()),
                chat_template="custom",
                chat_template_kwargs={"options": options},
            )
    else:
        actual = value.tokenize(
            tokenizer=cast(Any, Tokenizer()),
            chat_template="custom",
            chat_template_kwargs={"options": options},
        )
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
    assert calls


@pytest.mark.parametrize("mutate", [False, True])
def test_render_guard_accepts_equal_copies_but_checks_their_values(mutate):
    shared = {"value": 1}
    options = [shared, shared]
    exchange = _chat_exchange([1], [2])
    exchange.request["messages"] = [{"role": "user", "content": "q"}]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(True)
            options[1] = {"value": 2 if mutate else 1}
            return [3 if mutate else 1, 2]

        def __call__(self, text, **kwargs):
            options[1] = shared
            return [2 if text == "answer" else 3 if mutate else 1]

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            value.tokenize(
                tokenizer=cast(Any, Tokenizer()),
                chat_template="custom",
                chat_template_kwargs={"options": options},
            )
    else:
        actual = value.tokenize(
            tokenizer=cast(Any, Tokenizer()),
            chat_template="custom",
            chat_template_kwargs={"options": options},
        )
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
    assert calls


@pytest.mark.parametrize("mutate", [False, True])
def test_visible_evidence_encoder_cannot_restore_working_projection(mutate):
    from openai.types.responses import Response
    from test_tokenize import _multi_output_responses_chat_history

    projected = _multi_output_responses_chat_history()
    source = next(source for source in projected.message_sources if source is not None)
    exchange = source.exchange
    assert isinstance(exchange, tr.ResponsesExchange)
    data = exchange.response.model_dump(mode="python")
    for item, text, lp in zip(
        data["output"], ["first", "second"], [-0.1, -0.2], strict=True
    ):
        item["content"][0]["logprobs"] = [
            {
                "token": text,
                "logprob": lp,
                "bytes": list(text.encode()),
                "top_logprobs": [],
            }
        ]
    exchange.response = Response.model_validate(data)
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    error = ValueError("offsets unavailable")
    calls = []

    class Tokenizer:
        working = None

        def apply_chat_template(self, messages, **kwargs):
            if self.working is None:
                self.working = messages
            return [token for message in messages for token in self(message["content"])]

        def __call__(self, text, **kwargs):
            token = {"turn 0": 1, "first": 20, "second": 30}[text]
            if kwargs.get("return_offsets_mapping"):
                calls.append(text)
                if mutate and text == "first":
                    assert self.working is not None
                    self.working[0]["content"] = "changed"
                    raise error
                return {"input_ids": [token], "offset_mapping": [(0, len(text))]}
            if self.working is not None:
                self.working[0]["content"] = "turn 0"
            return [token]

    if mutate:
        with pytest.raises(ValueError) as raised:
            value.tokenize(tokenizer=Tokenizer(), chat_template="custom")
        assert raised.value is error
    else:
        actual = value.tokenize(tokenizer=Tokenizer(), chat_template="custom")
        assert actual.tokens == [1, 20, 30]
        assert actual.logprobs[1:] == [-0.1, -0.2]
    assert "first" in calls


def test_render_guard_does_not_certify_unsupported_callback_arguments():
    from art.trajectories._tokenize import _RenderContextGuard

    class Opaque:
        __slots__ = ()

    calls = []
    guard = _RenderContextGuard(lambda: {"stable": True})
    with pytest.raises(TypeError, match="Unsupported mutable tokenization context"):
        guard.call(lambda value: calls.append(value), Opaque())
    assert not calls


def test_trace_unwrap_does_not_probe_custom_tokenizer_class():
    from art.trajectories._tokenize import _TraceBuilder, _sampled_source_key

    class Tokenizer:
        @property
        def __class__(self):
            raise AssertionError("wrapper admission must not call a custom attribute")

    exchange = _chat_exchange([1], [2])
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    tokenized = value.tokenize()
    builder = _TraceBuilder(track_sources=False)
    tokenizer = cast(Any, Tokenizer())
    source = value.chat_completions_history().message_sources[-1]
    assert source is not None
    key = _sampled_source_key(source)
    keys = [key if flag & tr.TokenFlag.SAMPLED else None for flag in tokenized.flags]
    builder.set(tokenized, keys, {key: source}, tokenizer=tokenizer)
    assert builder.tokenizer is tokenizer
