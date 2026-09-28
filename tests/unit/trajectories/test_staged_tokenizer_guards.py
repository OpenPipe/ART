from __future__ import annotations

import gc
from typing import Any, cast
import weakref

from openai.types.chat import ChatCompletion
import pytest
from test_tokenize import _BoundaryTokenizer, _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize


@pytest.mark.parametrize("field", ["reasoning", "refusal"])
@pytest.mark.parametrize("mutate", [False, True])
def test_renderer_guard_tracks_accepted_message_replacement(field, mutate):
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["logprobs"] = None
    choice["message"] = {"role": "assistant", "content": "answer", field: "extra"}
    exchange.response = ChatCompletion.model_validate(data)
    history = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    before = history.model_dump_json()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(
                map(
                    ord,
                    "".join(
                        (message.get("reasoning_content") or "")
                        + (message.get("content") or "")
                        for message in messages
                    ),
                )
            )

        def __call__(self, text, **kwargs):
            if mutate:
                state.messages[-1]["content"] = "changed inside encoder"
            return list(map(ord, text))

    state = _tokenize._ChatViewTokenizer(
        history,
        base_model=None,
        tokenizer=cast(Any, Tokenizer()),
        chat_template=None,
        chat_template_kwargs=None,
    )
    try:
        original_messages = state.messages
        state._render_messages()
        assert state.messages is not original_messages
        assert field not in state.messages[-1]
        if mutate:
            with pytest.raises(ValueError, match="[Cc]ontext changed"):
                state._part_ids("probe")
        else:
            assert state._part_ids("probe") == list(map(ord, "probe"))
        assert history.model_dump_json() == before
    finally:
        del state.prefix_render_cache


@pytest.mark.parametrize("fail", [False, True])
def test_staged_tokenizer_releases_callback_cycle_without_gc(monkeypatch, fail):
    references = []
    original_init = _tokenize._ChatViewTokenizer.__init__

    def observe(self, *args, **kwargs):
        references.append(weakref.ref(self))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(_tokenize._ChatViewTokenizer, "__init__", observe)
    exchange = _chat_exchange([1], [2, 9])
    exchange.response.choices[0].model_extra.pop("prompt_token_ids")
    history = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer(_BoundaryTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            if fail:
                raise ValueError("renderer unavailable")
            return super().apply_chat_template(*args, **kwargs)

    def invoke():
        return _tokenize._tokenize_chat_view(
            history,
            base_model=None,
            tokenizer=Tokenizer(("user", [8]), ("assistant", [2, 9])),
            chat_template=None,
            chat_template_kwargs=None,
            _projection_matches=True,
        )

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        if fail:
            with pytest.raises(ValueError, match="renderer unavailable"):
                invoke()
        else:
            assert invoke().tokens
        assert len(references) == 1
        assert references[0]() is None
    finally:
        if was_enabled:
            gc.enable()
