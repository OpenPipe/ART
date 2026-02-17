"""MCP agent rollout implementation.

This module provides a rollout function for running MCP agents with scenarios.
Based on the art-e rollout.py structure.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from servers.python.mcp_alphavantage.server_params import server_params
import weave

import art

from .checks import check_successful
from .utils import get_content_text

load_dotenv()

logging.getLogger("weave.trace.op").setLevel(logging.WARNING)


weave.init("mcp-agent-training")


@dataclass
class McpScenario:
    """A scenario for MCP agent evaluation."""

    task_description: str
    server_params: StdioServerParameters
    max_turns: int = 10


@weave.op()
async def rollout(
    model: art.Model,
    scenario: McpScenario,
    debug: bool = False,
) -> art.Trajectory:
    """Run an MCP agent rollout with server parameters.

    Args:
        model: The ART model to use for the agent
        scenario: The MCP scenario to run (must include server_params)

    Returns:
        Trajectory containing the results of the rollout
    """
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={
            "task_completed": False,
            "success": False,
            "ran_out_of_turns": False,
        },
        scenario=scenario,
    )

    # Initialize system prompt
    system_prompt = f"""You are an MCP (Model Context Protocol) agent.\n\nYou have access to MCP tools through the server. Use them to complete your task.\n\nWhen you believe you have completed the task, call the 'complete_task' function with a summary of what you accomplished. You have a total of {scenario.max_turns} turns to complete the task. Only use tool calls, do not write any content. After you have completed the task, call the 'complete_task' function with a summary of what you accomplished. Call complete_task by itself, not in conjunction with any other tools."""

    # Connect to MCP server using stdio
    try:
        async with stdio_client(scenario.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Get available tools from the server
                tools_result = await session.list_tools()

                # Convert to OpenAI format
                tool_schemas = []
                for tool in tools_result.tools:
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"MCP tool: {tool.name}",
                            "parameters": tool.inputSchema
                            or {"type": "object", "properties": {}},
                        },
                    }
                    tool_schemas.append(tool_schema)

                if debug:
                    available_tools = [
                        tool["function"]["name"] for tool in tool_schemas
                    ]
                    print(f"Available MCP tools: {available_tools}")

                # Add completion tool schema
                tool_schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "complete_task",
                            "description": "Complete the task with a summary",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "summary": {
                                        "type": "string",
                                        "description": "Summary of accomplishments",
                                    }
                                },
                                "required": ["summary"],
                            },
                        },
                    }
                )

                traj.tools = tool_schemas

                # Initialize conversation
                traj.messages_and_choices = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Please complete this task: {scenario.task_description}",
                    },
                ]

                if debug:
                    print(traj.messages())

                num_turns = 0
                task_completed = False

                # Main interaction loop
                while num_turns < scenario.max_turns and not task_completed:
                    num_turns += 1

                    try:
                        # Get LLM response
                        async with traj.track_duration("llm_completion"):
                            client = AsyncOpenAI(
                                api_key=model.inference_api_key,
                                base_url=model.inference_base_url,
                            )

                            response = await client.chat.completions.create(
                                model=model.inference_model_name
                                if model.inference_model_name
                                else model.name,
                                messages=traj.messages(),
                                temperature=1.0,
                                tools=tool_schemas,
                                max_completion_tokens=8000,
                            )

                        choice = response.choices[0]

                        if debug:
                            print(f"Choice: {choice.message}")

                        traj.messages_and_choices.append(choice)

                        # Handle tool calls
                        if choice.message.tool_calls:
                            for tool_call in choice.message.tool_calls:
                                try:
                                    tool_args = json.loads(tool_call.function.arguments)

                                    if tool_call.function.name == "complete_task":
                                        traj.metrics["task_completed"] = True
                                        traj.log(
                                            f"Task completion attempted with summary: {tool_args['summary']}"
                                        )
                                        try:
                                            traj.metrics[
                                                "success"
                                            ] = await check_successful(traj)
                                        except Exception as e:
                                            print(f"Error checking success: {e}")
                                            traj.log(f"Error checking success: {e}")

                                        task_completed = True
                                    else:
                                        # Call MCP tool through session
                                        result = await session.call_tool(
                                            tool_call.function.name, tool_args
                                        )

                                        content_text = get_content_text(result)

                                        if len(content_text) > 20000:
                                            print(
                                                f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                            )
                                            print(f"Args: {tool_args}")
                                            # print first and last 1000 characters
                                            print(content_text[:1000])
                                            print(content_text[-1000:])
                                            raise Exception(
                                                f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                            )

                                        # Add tool response
                                        traj.messages_and_choices.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "content": content_text,
                                            }
                                        )

                                    if debug:
                                        print(f"Tool call result: {content_text}")

                                except Exception as e:
                                    traj.log(f"Tool call error: {e}")

                                    # Add error response
                                    traj.messages_and_choices.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "content": f"Error: {str(e)}",
                                        }
                                    )
                        else:
                            # No tool calls, just continue conversation
                            break

                    except Exception as e:
                        traj.log(f"Error in turn {num_turns}: {e}")
                        break

    except Exception as e:
        traj.log(f"MCP server error: {e}")
    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True

    traj.metrics["num_turns"] = num_turns

    if debug:
        for message in traj.messages_and_choices:
            print("\n")
            print(message)
            print("\n")

    return traj.finish()


async def test_rollout():
    model = art.Model(
        name="o4-mini",
        project="mcp-agent-training",
        inference_model_name="o4-mini",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1",
    )

    scenario = McpScenario(
        task_description="Use search_symbol to find emerging biotech companies by searching with keywords 'biotech' and review the results.",
        server_params=server_params,
    )

    traj = await rollout(model, scenario, debug=True)
    print(traj)


async def main():
    """Run test scenario."""
    print("=== Testing Python MCP AlphaVantage Server ===")
    await test_rollout()


if __name__ == "__main__":
    asyncio.run(main())
