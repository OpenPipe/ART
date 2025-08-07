import json
import logging
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Dict

import litellm
from art_e.data.types_enron import SyntheticQuery
from art_e.email_search_tools import (
    Email,
)
from art_e.email_search_tools import (
    read_email as read_email_impl,
)
from art_e.email_search_tools import (
    search_emails as search_emails_impl,
)
from art_e.project_types import ProjectPolicyConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm import acompletion
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel, Field, ValidationError, validate_call
from rich import print
from tenacity import retry, stop_after_attempt

import art
from art import Trajectory
from art.utils.litellm import convert_litellm_choice_to_openai

litellm.cache = Cache(type=LiteLLMCacheType.DISK)
litellm.drop_params = True
logging.getLogger("weave.trace.op").setLevel(logging.WARNING)


@dataclass
class FinalRubric:
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {k: int(v) for k, v in asdict(self).items()}
        metrics["failed_format_validation"] = int(
            self.bad_tool_call_name
            or self.bad_tool_call_args
            or self.cant_parse_tool_call
        )
        return metrics


def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: FinalRubric, traj: Trajectory
) -> float:
    # As an ablation, let's try the simplest possible reward function: just give
    # 1 point for a correct answer, and 0 for anything else. Otherwise, we'll do something
    # more complex.
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Note: make sure all possible partial rewards always sum to less than 0.5.
    partial_rewards = 0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting error: reward will be -2 to -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # No formatting error, but wrong answer: reward will be -1 to 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # Returned no answer at all: reward will be 0 to 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # Answer is correct: reward will be 1 to 2
    if rubric.answer_correct:
        # Partial credit calculation is different for correct answers.

        reward = 1
        reward += 0.3 if rubric.sources_correct else 0

        # Extra credit for not including extra sources.
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

        # Extra credit for being faster (taking fewer turns).
        reward += 0.1 * (1 - rubric.num_turns / policy_config.max_turns)
        return reward

    traj.logs.append(f"Rubric: {rubric}")
    traj.logs.append("Rubric not handled properly")
    raise ValueError("Rubric is not handled properly")


# Pydantic model for the answer-checking judge response
class CorrectnessJudgeResponse(BaseModel):
    """LLM-judge output for answer correctness.

    thinking: free-form rationale for the judgement (not used in scoring but
    helpful for debugging).
    accept: True if the AI answer contains every essential fact from the
    reference answer with no contradictions.
    """

    thinking: str = Field(description="Explanation of the reasoning process.")
    accept: bool = Field(description="Whether the AI answer should be accepted.")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(
    answer: str, query: SyntheticQuery
) -> CorrectnessJudgeResponse:
    """Use an LLM to decide whether *answer* matches *query.answer*.

    Returns a structured ``AnswerJudgeResponse`` with a free-form *thinking*
    string (useful for debugging) and a boolean *accept* flag used for
    scoring.
    """

    system_prompt = textwrap.dedent(
        """You are given a question, the reference answer (labelled **Reference answer**), and an answer generated by an AI assistant (labelled **AI answer**).

Follow these steps to decide whether the AI answer should be accepted:
1. Identify EXACTLY what information the **question** is asking for (e.g. who, what, when, where, why, how, quantity, etc.).
2. From the **Reference answer**, extract ONLY the facts that are required to directly satisfy the information need identified in step 1. Treat all other facts as non-essential context.
3. Verify that every essential fact from step 2 appears in the **AI answer** with the same meaning. Differences in wording, order, or additional non-conflicting details are allowed.
4. If any essential fact is missing or contradicted in the **AI answer**, then *accept* must be **false**. Otherwise *accept* must be **true**.

Important: Do NOT penalise the **AI answer** for omitting non-essential facts that appear in the **Reference answer**. The answer should only be rejected for errors or omissions in the information explicitly requested by the question.

Return your judgement as **pure JSON** (no markdown) with this exact schema:
{
  "thinking": string,  // Brief explanation of your reasoning.
  "accept": boolean   // true if the AI answer should be accepted.
}
"""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query.question}\n"
                f"Reference answer: {query.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = await acompletion(
        model="openai/gpt-4.1",
        messages=messages,
        caching=True,
        response_format=CorrectnessJudgeResponse,
    )

    first_choice = response.choices[0]  # type: ignore[attr-defined]
    raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        # If parsing fails, fall back to 'accept': False
        return CorrectnessJudgeResponse(
            thinking=f"Parse error: {e}\nRaw: {raw_content}", accept=False
        )


class ProjectTrajectory(Trajectory):
    scenario: SyntheticQuery
    generated_answer: str | None = None


def clean_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in message.items() if v is not None}


@retry(stop=stop_after_attempt(3))
async def rollout(
    model: art.Model[ProjectPolicyConfig],
    scenario: SyntheticQuery,
) -> ProjectTrajectory:
    rubric = FinalRubric()
    traj = ProjectTrajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id},
        scenario=scenario,
    )

    system_prompt = textwrap.dedent(
        f"""\
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {model.config.max_turns} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """
    )

    if model.config.include_qwen3_nothink:
        system_prompt += "\n/no_think"

    async def search_emails(keywords: list[str]) -> list[dict]:
        """
        Search the user's email inbox for emails that match the given keywords.

        Args:
            keywords (list[str]): A list of keywords to search for in the user's email inbox.

        Returns:
            list[dict]: A list of matching messages and snippets.
        """
        resp = search_emails_impl(
            inbox=scenario.inbox_address,
            sent_before=scenario.query_date,
            keywords=keywords,
        )

        for r in resp:
            if r.message_id == scenario.message_ids[0]:
                rubric.ever_found_right_email = True
        return [asdict(r) for r in resp]

    async def read_email(message_id: str) -> Email | dict:
        """
        Read the content of an email from the user's email inbox. Returns the email content.
        """
        email_content = read_email_impl(message_id)

        if message_id == scenario.message_ids[0]:
            rubric.ever_read_right_email = True
        if email_content is None:
            return {"error": "Email not found"}
        else:
            return email_content.model_dump()

    async def return_final_answer(answer: str, sources: list[str]):
        """
        This function is used to return the final answer to the user's query.
        It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

        Args:
            answer: (str) the answer to the user's query. If you cannot find the answer, you should return "I don't know" with an empty list of sources.
            sources: (list[str]) a list of message ids that are relevant to the query. Usually there will be only one. If you cannot find the answer, you should return an empty list.

        Returns:
            (str) the final answer to the user's query
        """

        rubric.attempted_answer = True
        traj.generated_answer = answer

        if answer == "I don't know":
            rubric.returned_i_dont_know = True
        else:
            async with traj.track_duration("determine_if_answer_is_correct"):
                judge_response = await judge_correctness(answer, scenario)
                traj.log(f"Correctness judge response: {judge_response}")
                rubric.answer_correct = judge_response.accept
            rubric.sources_correct = scenario.message_ids[0] in sources

    tools = [
        search_emails,
        read_email,
        return_final_answer,
    ]

    traj.tools = [convert_to_openai_tool(t) for t in tools]  # type: ignore

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]
    llm_response: ModelResponse | None = None

    while not rubric.attempted_answer:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        litellm_params = model.litellm_completion_params()
        if scenario.split == "test":
            litellm_params["temperature"] = 0

        async with traj.track_duration("llm_completion"):
            llm_response = await acompletion(
                # **litellm_params,
                **model.litellm_completion_params(),
                messages=traj.messages(),
                caching=not model.trainable,
                max_completion_tokens=model.config.max_tokens,
                tools=traj.tools,
                tool_choice=None if model.trainable else "required",
            )  # type: ignore

        assert isinstance(llm_response, ModelResponse)
        rubric.prompt_tokens += llm_response.usage.prompt_tokens  # type: ignore
        rubric.completion_tokens += llm_response.usage.completion_tokens  # type: ignore
        if len(llm_response.choices) == 0:
            raise ValueError(f"No choices in response: {llm_response}")
        choice = llm_response.choices[0]
        assert isinstance(choice, Choices)

        traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))  # type: ignore

        if choice.message.tool_calls is None:
            rubric.bad_tool_call_name = True
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": "You did not call any tools. This is not allowed.",
                }
            )
            break

        for tool_call in choice.message.tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
                assert isinstance(tool_args, dict)
            except Exception:
                rubric.bad_tool_call_args = True
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error parsing arguments for {tool_call.function.name}. Cannot continue.",
                    }
                )
                break

            for tool_fn in tools:
                if tool_fn.__name__ == tool_call.function.name:
                    try:
                        validated_fn = validate_call(tool_fn)
                        result = await validated_fn(**tool_args)
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            }
                        )
                    except ValidationError as e:
                        rubric.bad_tool_call_args = True
                        traj.logs.append(
                            f"Invalid args for {tool_call.function.name}: {e}"
                        )
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Invalid args for {tool_call.function.name}: {e}",
                            }
                        )
                        break
                    break
            else:
                rubric.bad_tool_call_name = True
                break

        # If we encountered an invalid tool name or arguments, we cannot
        # respond to the tool call properly. Break out of the main loop to
        # avoid sending an incomplete tool interaction back to the LLM, which
        # would cause the "assistant message with 'tool_calls' must be
        # followed by tool messages" error.
        if rubric.bad_tool_call_name or rubric.bad_tool_call_args:
            break

    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()

    traj.finish()
    return traj


if __name__ == "__main__":
    import asyncio

    import yaml
    from art_e.data.query_iterators import load_synthetic_queries
    from dotenv import load_dotenv

    load_dotenv()

    scenario = load_synthetic_queries(split="test", limit=1)[0]

    traj = asyncio.run(
        rollout(
            art.Model(
                name="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                project="email_agent",
                inference_model_name="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                config=ProjectPolicyConfig(),
            ),
            scenario,
        )
    )
    print(yaml.dump(traj.for_logging()))
    print(f"Question: {scenario.question}")
    print(f"Expected answer: {scenario.answer}")
    print(f"Generated answer: {traj.generated_answer}")
