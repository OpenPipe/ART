from copy import deepcopy
from typing import Any

import pytest
from test_recorded_boundaries import _boundary_render, _recorded_boundaries
from test_tokenize import _character_template_history

from art.trajectories import _tokenize as module


@pytest.mark.parametrize("outcome", ["success", "optional-decline", "fatal"])
def test_boundary_callback_source_change_cannot_reach_assembly_or_fallback(
    outcome, monkeypatch: pytest.MonkeyPatch
):
    history, tokenizer, _ = _character_template_history()
    source = history.message_sources[3]
    assert source is not None and isinstance(
        source.exchange, module.ChatCompletionsExchange
    )
    logprobs = source.exchange.response.choices[0].logprobs
    assert logprobs is not None and logprobs.content
    entries = logprobs.content
    original = tokenizer.decode
    failure = RuntimeError("public callback failure")

    def mutate(*args: Any, **kwargs: Any):
        entries[0].logprob -= 1
        if outcome == "optional-decline":
            raise ValueError("optional body decode unsupported")
        if outcome == "fatal":
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", mutate)
    with pytest.raises(RuntimeError if outcome == "fatal" else ValueError) as caught:
        _recorded_boundaries(history, tokenizer, _boundary_render(tokenizer))
    if outcome == "fatal":
        assert caught.value is failure
    else:
        assert "Sampled source changed" in str(caught.value)


@pytest.mark.parametrize("change", ["message", "tools", "kwargs", "source-list"])
@pytest.mark.parametrize("outcome", ["success", "optional-decline", "fatal"])
def test_boundary_rechecks_current_context_and_source_ownership(
    change, outcome, monkeypatch: pytest.MonkeyPatch
):
    history, tokenizer, _ = _character_template_history()
    if change == "tools":
        tools: list[Any] = [
            {"type": "function", "function": {"name": "lookup", "parameters": {}}}
        ]
        history.tools = tools
        for source in history.message_sources:
            if source is not None:
                assert isinstance(source.exchange, module.ChatCompletionsExchange)
                source.exchange.request["tools"] = tools
    if change == "kwargs":
        settings = {"public_setting": {"value": 0}}
        history.chat_template_kwargs = settings
        for source in history.message_sources:
            if source is not None:
                source.exchange.request["chat_template_kwargs"] = settings
    render = tokenizer.apply_chat_template
    failure = RuntimeError("public rendering failure")
    fired = False

    def mutate(messages, **kwargs):
        nonlocal fired
        value = render(messages, **kwargs)
        if not fired:
            fired = True
            if change == "message":
                history.messages[0]["content"] = "changed after evidence admission"
            elif change == "tools":
                kwargs["tools"][0]["function"]["name"] = "changed"
            elif change == "kwargs":
                kwargs["public_setting"]["value"] = 1
            else:
                copied = deepcopy(history.message_sources[-1])
                assert copied is not None and isinstance(
                    copied.exchange, module.ChatCompletionsExchange
                )
                logprobs = copied.exchange.response.choices[0].logprobs
                assert logprobs is not None and logprobs.content
                logprobs.content[0].logprob -= 1
                history.message_sources[-1] = copied
            if outcome == "optional-decline":
                raise NotImplementedError("public optional capability unavailable")
            if outcome == "fatal":
                raise failure
        return value

    monkeypatch.setattr(tokenizer, "apply_chat_template", mutate)
    with pytest.raises(RuntimeError if outcome == "fatal" else ValueError) as caught:
        history.tokenize(tokenizer=tokenizer)
    assert fired
    if outcome == "fatal":
        assert caught.value is failure
    else:
        assert "Sampled source changed" in str(caught.value)


@pytest.mark.parametrize("context_kind", ["opaque", "cycle", "metaclass"])
def test_unprovable_boundary_context_declines_before_any_callback(context_kind):
    history, tokenizer, _ = _character_template_history()
    equality_calls = []

    class Meta(type):
        def __eq__(cls, other):
            equality_calls.append(other)
            return True

    class Masquerading(metaclass=Meta):
        pass

    context: Any = object()
    if context_kind == "cycle":
        context = []
        context.append(context)
    elif context_kind == "metaclass":
        context = Masquerading()
    history.chat_template_kwargs = {"public_setting": context}

    def forbidden(*args, **kwargs):
        pytest.fail("an unproved boundary must decline before rendering")

    assert _recorded_boundaries(history, tokenizer, forbidden) is None
    assert not equality_calls
    if isinstance(context, list):
        context.clear()


@pytest.mark.parametrize("behavior", ["unchanged", "mutated", "fatal"])
def test_boundary_guard_tracks_consumed_stop_evidence(monkeypatch, behavior):
    from test_native_terminal import ProjectedToolTokenizer, projected_history

    import art.trajectories as tr

    trajectory, tokenizer, records = projected_history(
        finish="stop", sampled_eos=True, earlier_length=True
    )
    history = trajectory.chat_completions_history()
    source = history.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, module.ChatCompletionsExchange
    )
    final = source.exchange.response.choices[0]
    assert final.model_extra is not None
    final.model_extra["stop_reason"] = "§"
    extra = final.model_extra
    encode = ProjectedToolTokenizer.__call__
    calls = []
    fatal = RuntimeError("public stop encoder failure")

    def observed(self, text, **kwargs):
        result = encode(self, text, **kwargs)
        if text == "§":
            calls.append(extra["stop_reason"])
            if len(calls) == 2 and behavior != "unchanged":
                extra["stop_reason"] = "unmatched stop string"
                if behavior == "fatal":
                    raise fatal
        return result

    def render(selected_messages, *, add_generation_prompt):
        return tokenizer.apply_chat_template(
            selected_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    monkeypatch.setattr(ProjectedToolTokenizer, "__call__", observed)
    try:
        value = _recorded_boundaries(history, tokenizer, render)
    except RuntimeError as error:
        assert behavior == "fatal" and error is fatal
    except ValueError as error:
        assert behavior == "mutated" and "Sampled source changed" in str(error)
    else:
        assert behavior == "unchanged" and value is not None
        assert value.tokens == records[-1][0] + records[-1][1]
        assert value.flags[-1] & tr.TokenFlag.STOP
    assert len(calls) == 2
