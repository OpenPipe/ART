from typing import Any

from openai.types.chat import ChatCompletion
import pytest
from test_native_terminal import projected_history

import art.trajectories as tr


def _case():
    trajectory, tokenizer, _ = projected_history(
        finish="stop", sampled_eos=True, earlier_length=False
    )
    exchange = trajectory.exchanges.chat_completions[0]
    exchange.request["messages"].insert(
        1, {"role": "assistant", "content": "historical context"}
    )
    prompt = tokenizer.apply_chat_template(
        exchange.request["messages"], add_generation_prompt=True
    )
    payload = exchange.response.model_dump(mode="python")
    payload["prompt_token_ids"] = prompt
    payload["choices"][0]["prompt_token_ids"] = prompt
    exchange.response = ChatCompletion.model_validate(payload)
    assert not hasattr(tokenizer, "chat_template")
    return trajectory, tokenizer


@pytest.mark.parametrize("operation", ["render", "encode"])
@pytest.mark.parametrize("error_kind", [ValueError, RuntimeError])
def test_missing_metadata_preserves_actual_callback_error(
    monkeypatch: pytest.MonkeyPatch, operation: str, error_kind: type[Exception]
) -> None:
    trajectory, tokenizer = _case()
    failure = error_kind("Opaque renderer failed")

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise failure

    if operation == "render":
        monkeypatch.setattr(tokenizer, "apply_chat_template", failed)
    else:
        monkeypatch.setattr(type(tokenizer), "__call__", failed)
    with pytest.raises(error_kind) as caught:
        trajectory.tokenize(tokenizer=tokenizer)
    assert caught.value is failure


@pytest.mark.parametrize("mutate", [False, True])
def test_missing_metadata_admission_is_revalidated_after_decline(
    monkeypatch: pytest.MonkeyPatch, mutate: bool
) -> None:
    trajectory, tokenizer = _case()
    expected = trajectory.tokenize(tokenizer=tokenizer)
    prefix_length = next(
        index
        for index, flag in enumerate(expected.flags)
        if flag & tr.TokenFlag.SAMPLED
    )
    encode = type(tokenizer).__call__
    calls = []

    def no_offsets(self: Any, text: str, **kwargs: Any) -> dict[str, object]:
        result = encode(self, text, **kwargs)
        result.pop("offset_mapping", None)
        if mutate:
            self.chat_template = None
        calls.append(True)
        return result

    monkeypatch.setattr(type(tokenizer), "__call__", no_offsets)
    if mutate:
        with pytest.raises(ValueError, match="Sampled source changed"):
            trajectory.tokenize(tokenizer=tokenizer)
    else:
        result = trajectory.tokenize(tokenizer=tokenizer)
        assert result.tokens == expected.tokens
        assert result.flags[:prefix_length] == [tr.TokenFlag.EXACT] * prefix_length
        assert result.flags[prefix_length:] == expected.flags[prefix_length:]
        assert result.tokens[-1] == tokenizer.eos_token_id
        assert result.flags[-1] & tr.TokenFlag.STOP
    assert calls == [True]
