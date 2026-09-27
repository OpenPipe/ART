from __future__ import annotations

import hashlib
import math
from typing import Any, cast

from openai.types.responses import Response
import pytest
from test_tokenize import _response_exchange

import art.trajectories as tr


def _run_case(mode: str, behavior: str) -> dict[str, Any]:
    exchange = _response_exchange("public-selection-disposition", 2)
    data = exchange.response.model_dump(mode="python")
    data.pop("token_generations")
    data["output"][0]["content"][0]["logprobs"] = [
        {
            "token": "answer",
            "logprob": -0.2,
            "bytes": list(b"answer"),
            "top_logprobs": [],
        }
    ]
    exchange.response = Response.model_validate(data)
    original = exchange.model_dump_json()
    calls = []

    class Tokenizer:
        chat_template = {"default": "public template"}
        retained = None

        def get_chat_template(self, **kwargs):
            before = self.retained["content"] if self.retained is not None else None
            if self.retained is not None and mode != "none":
                self.retained["content"] = "changed before completion"
                if mode == "restored_before_consumer":
                    self.retained["content"] = "turn 0"
            calls.append(
                {
                    "call": "select",
                    "before": before,
                    "after": self.retained["content"]
                    if self.retained is not None
                    else None,
                }
            )
            return "public template"

        def apply_chat_template(self, messages, **kwargs):
            completed = any(message["role"] == "assistant" for message in messages)
            consumed = messages[0]["content"]
            changed = consumed != "turn 0"
            prompt = 101 if completed and changed and behavior == "prefix" else 99
            output = 3 if changed and behavior == "output" else 2
            tokens = [prompt, output] if completed else [prompt]
            calls.append(
                {
                    "call": "render",
                    "completed": completed,
                    "consumed": consumed,
                    "same_retained_object": self.retained is messages[0],
                    "returned": tokens,
                }
            )
            if not completed:
                self.retained = messages[0]
            elif mode == "consumed_then_restored":
                messages[0]["content"] = "turn 0"
            return tokens

        def __call__(self, text, **kwargs):
            calls.append(
                {
                    "call": "encode",
                    "text": text,
                    "offsets": bool(kwargs.get("return_offsets_mapping")),
                }
            )
            assert text == "answer"
            return (
                {"input_ids": [2], "offset_mapping": [(0, len(text))]}
                if kwargs.get("return_offsets_mapping")
                else [2]
            )

    tokenizer = Tokenizer()
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    row = {
        "mode": mode,
        "behavior": behavior,
        "native_prompt_ids": None,
        "native_output_ids": None,
        "native_token_generations": None,
        "visible_response_text": "answer",
        "visible_response_logprob": -0.2,
    }
    try:
        result = value.tokenize(tokenizer=cast(Any, tokenizer))
        row["result"] = {
            "tokens": list(result.tokens),
            "logprobs": [None if math.isnan(x) else x for x in result.logprobs],
            "flags": [int(x) for x in result.flags],
            "first_masks": {
                flag.name: tr.first_occurrence_masks([result], where=flag)
                for flag in (
                    tr.TokenFlag.SAMPLED,
                    tr.TokenFlag.OUTPUT,
                    tr.TokenFlag.ASSISTANT,
                    tr.TokenFlag.STOP,
                )
            },
        }
    except Exception as exc:
        row["error"] = {"class": type(exc).__name__, "message": str(exc)}
    row.update(
        calls=calls,
        original_exchange_unchanged=exchange.model_dump_json() == original,
        original_exchange_sha256=hashlib.sha256(original.encode()).hexdigest(),
        retained_final=tokenizer.retained["content"]
        if tokenizer.retained is not None
        else None,
    )
    return row


@pytest.mark.parametrize("behavior", ["same", "prefix", "output"])
@pytest.mark.parametrize(
    "mode", ["none", "restored_before_consumer", "consumed_then_restored", "lasting"]
)
def test_responses_selection_does_not_change_reused_projection(
    mode: str, behavior: str
) -> None:
    row = _run_case(mode, behavior)
    assert row["original_exchange_unchanged"]
    rendered = [call for call in row["calls"] if call["call"] == "render"]
    assert rendered[0]["consumed"] == "turn 0"
    if mode in ("consumed_then_restored", "lasting"):
        assert row["error"]["class"] == "ValueError"
        assert "context changed" in row["error"]["message"]
        # The changed projection never reaches the completion renderer.
        assert len(rendered) == 1
    else:
        assert len(rendered) == 2
        assert rendered[1]["consumed"] == "turn 0"
        assert rendered[1]["same_retained_object"]
        assert row["result"]["tokens"] == [99, 2]
        assert row["result"]["logprobs"] == [None, -0.2]
        assert row["result"]["flags"] == [0, 20]
        assert row["result"]["first_masks"] == {
            "SAMPLED": [[False, False]],
            "OUTPUT": [[False, True]],
            "ASSISTANT": [[False, True]],
            "STOP": [[False, False]],
        }
        assert row["retained_final"] == "turn 0"
