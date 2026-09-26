from __future__ import annotations

import copy
import math
from typing import Any

from openai.types.responses import Response
import pytest
from test_tokenize import _response_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("changed_generation", [0, 2])
def test_copied_responses_stop_cannot_return_stale_prior_logprobs(
    monkeypatch: pytest.MonkeyPatch, mutate: bool, changed_generation: int
) -> None:
    exchange = _response_exchange("three-generations", 2, prompt_token_ids=[1])
    payload = exchange.response.model_dump(mode="python")
    message = payload["output"][0]
    payload["output"] = []
    for index in range(3):
        item = copy.deepcopy(message)
        item["id"] = f"public-message-{index}"
        item["content"][0]["text"] = f"answer{index}"
        payload["output"].append(item)
    payload["token_generations"] = [
        {
            "prompt_token_ids": prompt,
            "output_tokens": [
                {"token_id": token, "logprob": -token / 10} for token in output
            ],
            "output_indices": [index],
        }
        for index, (prompt, output) in enumerate(
            [([1], [2]), ([1, 2, 3], [4, 5]), ([1, 2, 3, 5, 6], [7])]
        )
    ]
    exchange.response = Response.model_validate(payload)
    trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    before = trajectory.model_dump_json()

    class Tokenizer:
        armed = False
        copied_callback = False

        def convert_tokens_to_ids(self, token: str) -> None:
            del token
            if self.armed:
                self.armed = False
                self.copied_callback = True
                if mutate:
                    assert exchange.response.model_extra is not None
                    exchange.response.model_extra["token_generations"][
                        changed_generation
                    ]["output_tokens"][0]["logprob"] = -9.5

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("complete native records must not render")

    tokenizer: Any = Tokenizer()
    native = module._tokenize_exact_responses_history

    def observe(history: Any, **kwargs: Any) -> Any:
        # Arm only after preflight and entry to the history containing generation
        # 2. Its first converter callback is the copied generation-1 STOP probe.
        # Generation 2's prompt has already been read as a suffix witness, so
        # that record is consumed too. Preserve real callbacks without ordinals.
        tokenizer.armed = any(
            source is not None and source.generation_index == 2
            for source in history.input_sources
        )
        return native(history, **kwargs)

    monkeypatch.setattr(module, "_tokenize_exact_responses_history", observe)
    if mutate:
        with pytest.raises(ValueError, match="Sampled source changed"):
            trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
        assert tokenizer.copied_callback
        assert exchange.response.model_extra is not None
        assert (
            exchange.response.model_extra["token_generations"][changed_generation][
                "output_tokens"
            ][0]["logprob"]
            == -9.5
        )
    else:
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
        assert tokenizer.copied_callback
        assert [history.tokens for history in result.histories] == [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 5, 6, 7],
        ]
        original, copied = result.histories
        assert original.logprobs[1] == copied.logprobs[1] == -0.2
        assert original.logprobs[3:] == [-0.4, -0.5]
        assert copied.logprobs[-1] == -0.7
        assert math.isnan(copied.logprobs[3])
        assert copied.flags[3] == (
            tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT
        )
        assert trajectory.model_dump_json() == before
