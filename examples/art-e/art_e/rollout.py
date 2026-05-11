from textwrap import dedent
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import weave

import art
from art.langgraph import init_chat_model

from .email_tools import LocalInbox
from .scoring import grade_answer
from .types import FinalAnswer, Scenario


class EmailScenario(BaseModel):
    step: int = 0
    scenario: Scenario


class EmailTrajectory(art.Trajectory):
    final_answer: FinalAnswer | None = None


@weave.op
async def rollout(model: art.Model, email_scenario: EmailScenario) -> EmailTrajectory:
    scenario = email_scenario.scenario
    inbox = LocalInbox(scenario.messages)
    final_answer: FinalAnswer | None = None

    traj = EmailTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metrics={
            "answer_match": False,
            "cited_expected_message": False,
        },
        metadata={
            "scenario_id": scenario.id,
            "step": email_scenario.step,
        },
        scenario=email_scenario,
    )

    @tool
    def search_inbox_tool(keywords: list[str]) -> list[dict[str, str]]:
        """Search the user's inbox for messages matching keyword terms."""
        return [
            message.preview()
            for message in inbox.search(
                scenario.inbox_address, keywords, scenario.query_date
            )
        ]

    @tool
    def read_email_tool(message_id: str) -> dict[str, str] | None:
        """Read one email by message ID."""
        message = inbox.read(message_id)
        return message.model_dump() if message else None

    @tool
    def return_final_answer_tool(
        answer: str, source_ids: list[str]
    ) -> dict[str, object]:
        """Return the final answer and source email IDs."""
        nonlocal final_answer
        final_answer = FinalAnswer(answer=answer, source_ids=source_ids)
        return final_answer.model_dump()

    system_prompt = dedent(
        f"""
        You are an email research agent. Use the tools to answer the user's question.
        Inbox: {scenario.inbox_address}
        Today's date: {scenario.query_date}

        Search before reading. Read messages before answering. When you know the answer,
        call return_final_answer_tool with a concise answer and the source message IDs.
        """
    ).strip()

    try:
        chat_model = init_chat_model(model.get_inference_name(), temperature=1.0)
        agent = create_react_agent(
            chat_model,
            [search_inbox_tool, read_email_tool, return_final_answer_tool],
        )
        await agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config={
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 10,
            },
        )
    except Exception as exc:
        traj.log(f"rollout failed: {exc}")

    if final_answer:
        grade = grade_answer(scenario, final_answer.answer, final_answer.source_ids)
        traj.final_answer = final_answer
        traj.reward = grade.reward
        traj.metrics["answer_match"] = grade.answer_match
        traj.metrics["cited_expected_message"] = grade.cited_expected_message

    return traj
