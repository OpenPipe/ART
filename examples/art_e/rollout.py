from __future__ import annotations

import asyncio
import json
import os
import random

from dotenv import load_dotenv
import openai
import requests
from scenarios import (
    SCENARIOS,
    EmailScenario,
    parse_json_command,
    read_email,
    score_answer,
    search_emails,
)

import art

load_dotenv()

MAX_TURNS = 6

SYSTEM_PROMPT = """You are ART-E, an email research agent.

You need to answer the user's question by searching and reading their inbox.

Use exactly one command per assistant message:

<search>{"keywords":["keyword"],"sent_before":"YYYY-MM-DD"}</search>
<read>{"message_id":"message-id"}</read>
<answer>{"answer":"final answer","reference_message_ids":["message-id"]}</answer>

Rules:
- Search before answering unless the answer is already present in the context.
- Read a message before citing it.
- Cite only message IDs that support your answer.
- Keep final answers concise and factual.
"""


def tool_message(payload: object) -> dict[str, str]:
    return {
        "role": "user",
        "content": "Tool result:\n" + json.dumps(payload, indent=2, sort_keys=True),
    }


@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(
    model: art.Model,
    scenario: EmailScenario,
    step: int = 0,
    is_validation: bool = False,
    verbose: bool = False,
) -> art.Trajectory:
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Inbox: {scenario.inbox_address}\n"
                    f"Today: {scenario.query_date}\n"
                    f"Question: {scenario.question}"
                ),
            },
        ],
        metadata={
            "scenario_id": scenario.id,
            "step": step,
            "validation": is_validation,
        },
        reward=0,
    )

    client = model.openai_client()
    searched = False
    read_message_ids: set[str] = set()

    for _ in range(MAX_TURNS):
        completion = await client.chat.completions.create(
            max_completion_tokens=256,
            messages=trajectory.messages(),
            model=model.get_inference_name(),
        )
        choice = completion.choices[0]
        content = choice.message.content or ""
        trajectory.messages_and_choices.append(choice)

        if verbose:
            print(content)

        if search_payload := parse_json_command(content, "search"):
            keywords = search_payload.get("keywords", [])
            if not isinstance(keywords, list):
                trajectory.reward = -0.25
                trajectory.metrics["invalid_command"] = 1
                break
            sent_before = search_payload.get("sent_before", scenario.query_date)
            results = search_emails(
                scenario,
                [str(keyword) for keyword in keywords],
                sent_before=str(sent_before) if sent_before else None,
            )
            searched = True
            trajectory.messages_and_choices.append(
                tool_message({"search_results": results})
            )
            continue

        if read_payload := parse_json_command(content, "read"):
            message_id = read_payload.get("message_id")
            if not isinstance(message_id, str):
                trajectory.reward = -0.25
                trajectory.metrics["invalid_command"] = 1
                break
            email = read_email(scenario, message_id)
            if email is None:
                trajectory.messages_and_choices.append(
                    tool_message({"error": f"Message not found: {message_id}"})
                )
            else:
                read_message_ids.add(message_id)
                trajectory.messages_and_choices.append(tool_message({"email": email}))
            continue

        if answer_payload := parse_json_command(content, "answer"):
            answer = answer_payload.get("answer", "")
            reference_message_ids = answer_payload.get("reference_message_ids", [])
            if not isinstance(answer, str) or not isinstance(
                reference_message_ids, list
            ):
                trajectory.reward = -0.25
                trajectory.metrics["invalid_command"] = 1
                break

            references = [str(message_id) for message_id in reference_message_ids]
            reward, metrics = score_answer(scenario, answer, references)
            unread_citation = any(
                message_id not in read_message_ids for message_id in references
            )
            if unread_citation:
                reward *= 0.5
            trajectory.reward = reward
            trajectory.metrics.update(metrics)
            trajectory.metrics["searched"] = float(searched)
            trajectory.metrics["unread_citation"] = float(unread_citation)
            break

        trajectory.messages_and_choices.append(
            tool_message(
                {
                    "error": (
                        "Invalid command. Use one of <search>, <read>, or "
                        "<answer> with a JSON payload."
                    )
                }
            )
        )
    else:
        trajectory.reward = -0.1
        trajectory.metrics["ran_out_of_turns"] = 1

    return trajectory


if __name__ == "__main__":
    random.seed(42)

    smoke_test_model = art.Model(
        name="gpt-4o-mini",
        project="art-e",
        inference_model_name="openai/gpt-4o-mini",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    async def main() -> None:
        trajectory = await rollout(
            smoke_test_model,
            random.choice(SCENARIOS),
            is_validation=True,
            verbose=True,
        )
        print("reward:", trajectory.reward)
        print("metrics:", trajectory.metrics)

    asyncio.run(main())
