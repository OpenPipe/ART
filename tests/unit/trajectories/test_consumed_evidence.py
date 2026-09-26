from __future__ import annotations

from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange, _message_exchange

import art.trajectories as tr


def trajectory(*exchanges):
    return tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=list(exchanges))
    )


@pytest.mark.parametrize("changed_header", [False, True])
def test_request_role_requires_full_original_prompt(changed_header):
    class Tokenizer(_CharacterTemplateTokenizer):
        def apply_chat_template(
            self, messages, *, tokenize=True, add_generation_prompt, **kwargs
        ):
            text = "".join(m["role"] + ":" + m["content"] + "§" for m in messages)
            if add_generation_prompt:
                text += "assistant:"
            return self._encode(text) if tokenize else text

    tokenizer = Tokenizer()
    messages = [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if changed_header:
        text = text.replace("assistant:", "user_____:", 1)
    prompt = tokenizer._encode(text)
    exchange = _chat_exchange(prompt, tokenizer._encode("answer§"))
    exchange.request["messages"] = cast(list[ChatCompletionMessageParam], messages)
    value = trajectory(exchange)
    if changed_header:
        with pytest.raises(ValueError, match="[Rr]equest roles|assistant boundaries"):
            value.tokenize(tokenizer=tokenizer)
    else:
        actual = value.tokenize(tokenizer=tokenizer)
        assert actual.tokens[: len(prompt)] == prompt
        assert actual.flags[len("assistant:")] & tr.TokenFlag.ASSISTANT


@pytest.mark.parametrize("mutate", [False, True])
def test_exchange_fallback_binds_native_logprobs_before_prompt_render(mutate):
    exchange = _message_exchange(
        cast(
            Any,
            {
                "model": "test/model",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
        token_ids=[2],
        logprobs=[-0.2],
    )
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(True)
            if mutate:
                assert exchange.response.model_extra is not None
                exchange.response.model_extra["logprobs"][0] = -9
            return [1]

    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            value.tokenize(tokenizer=cast(Any, Tokenizer()))
    else:
        actual = value.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
    assert calls


@pytest.mark.parametrize("mutate", [False, True])
def test_completed_rendered_logprobs_keep_consumed_validator(mutate):
    first = _chat_exchange([1], [2])
    first.request["messages"] = [{"role": "user", "content": "first"}]
    choice = first.response.choices[0]
    assert (
        choice.model_extra is not None
        and choice.logprobs is not None
        and choice.logprobs.content is not None
    )
    choice.model_extra.pop("token_ids")
    choice.message.content = "a"
    choice.logprobs.content[0].token = "a"
    choice.logprobs.content[0].bytes = list(b"a")
    second = _chat_exchange([3], [4], offset=1)
    second.request["messages"] = [{"role": "user", "content": "second"}]
    seen = []

    class Tokenizer(_CharacterTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if messages and messages[0]["content"] == "second":
                seen.append(True)
                if mutate:
                    assert choice.logprobs is not None and choice.logprobs.content
                    choice.logprobs.content[0].logprob = -9
            return super().apply_chat_template(messages, **kwargs)

    value = trajectory(first, second)
    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            value.tokenize(
                tokenizer=Tokenizer(), multi_history=True, chat_template="explicit"
            )
    else:
        actual = value.tokenize(
            tokenizer=Tokenizer(), multi_history=True, chat_template="explicit"
        )
        assert len(actual.histories) == 2
        assert -0.2 in actual.histories[0].logprobs
        assert not any(f & tr.TokenFlag.SAMPLED for f in actual.histories[0].flags)
    assert seen


@pytest.mark.parametrize("field", ["tools", "chat_template", "chat_template_kwargs"])
@pytest.mark.parametrize("mutate", [False, True])
def test_generic_evidence_reads_check_consumed_render_settings(field, mutate):
    from test_evidence_reuse import extras

    second = None
    calls = []

    class Tokenizer(_CharacterTemplateTokenizer):
        eos_token_id = None
        all_special_tokens = []

        def convert_tokens_to_ids(self, token):
            return None

        def __call__(self, text, **kwargs):
            if text in ("FIRST", "SECOND"):
                calls.append(text)
                assert second is not None
                if mutate and text == "FIRST":
                    second.request[field] = {
                        "tools": [{"type": "function", "function": {"name": "edited"}}],
                        "chat_template": "edited",
                        "chat_template_kwargs": {"preserve_thinking": False},
                    }[field]
                elif mutate:
                    # A final-only check could miss this restoring callback.
                    second.request.pop(field, None)
                return {"input_ids": [99]}
            return super().__call__(text, **kwargs)

    tokenizer = Tokenizer()
    prompt = tokenizer._encode("turn 0")
    first = _chat_exchange(prompt, [20])
    second = _chat_exchange([*prompt, 20, *tokenizer._encode("turn 1")], [21], offset=1)
    extras(first)["stop_reason"] = "FIRST"
    extras(second)["stop_reason"] = "SECOND"
    value = trajectory(first, second)
    before = value.model_dump_json()
    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            value.tokenize(tokenizer=tokenizer, chat_template="explicit override")
        assert "FIRST" in calls and "SECOND" not in calls
    else:
        actual = value.tokenize(tokenizer=tokenizer, chat_template="explicit override")
        assert [
            token
            for token, flag in zip(actual.tokens, actual.flags, strict=True)
            if flag & tr.TokenFlag.SAMPLED
        ] == [20, 21]
        assert value.model_dump_json() == before
