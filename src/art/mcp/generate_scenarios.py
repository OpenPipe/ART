"""Scenario generation for MCP tools."""

import json
import time
from typing import Any, Dict, List, Optional

import openai

from art.mcp.types import GeneratedScenarioCollection, MCPResource, MCPTool
from art.utils.logging import _C, dim, err, info, ok, step


def preview_scenarios(scenarios: List[Dict[str, Any]], n: int = 5):
    """Preview generated scenarios."""
    n = min(n, len(scenarios))
    for i in range(n):
        s = scenarios[i]
        task_preview = s["task"][:120].strip()
        ellipsis = "&" if len(s["task"]) > 120 else ""
        difficulty = s.get("difficulty", "N/A")
        dim(
            f"   {i + 1}. {task_preview}{ellipsis}  "
            f"{_C.GRAY}(difficulty {difficulty}/5){_C.RESET}"
        )


async def generate_scenarios(
    tools: List[MCPTool] | List[Dict[str, Any]],
    resources: List[MCPResource] | List[Dict[str, Any]] = [],
    num_scenarios: int = 24,
    show_preview: bool = True,
    custom_instructions: Optional[str] = None,
    generator_model: str = "openai/gpt-4.1-mini",
    generator_api_key: Optional[str] = None,
    generator_base_url: str = "https://openrouter.ai/api/v1",
) -> GeneratedScenarioCollection:
    """
    Generate scenarios for MCP tools.

    Args:
        tools: List of Tool objects or list of tool dictionaries
        resources: Optional list of Resource objects or list of resource dictionaries
        num_scenarios: Number of scenarios to generate (default: 24)
        show_preview: Whether to show a preview of generated scenarios (default: True)
        custom_instructions: Optional custom instructions for scenario generation
        generator_model: Model to use for generation (default: "openai/gpt-4.1-mini")
        generator_api_key: API key for the generator model. If None, will use OPENROUTER_API_KEY env var
        generator_base_url: Base URL for the API (default: OpenRouter)

    Returns:
        GeneratedScenarioCollection containing the generated scenarios
    """
    import os

    t0 = time.perf_counter()

    # Handle API key
    if generator_api_key is None:
        generator_api_key = os.getenv("OPENROUTER_API_KEY")
        if not generator_api_key:
            raise ValueError(
                "generator_api_key is required or OPENROUTER_API_KEY env var must be set"
            )

    # Validate that we have at least tools or resources
    if not tools and not resources:
        raise ValueError("At least one tool or resource must be provided")

    ok(f"Using model: {generator_model}")

    # Convert tools to dictionaries
    if isinstance(tools, list) and tools and isinstance(tools[0], MCPTool):
        tools_info = [tool.to_dict() for tool in tools]  # type: ignore
    else:
        # Assume it's already a list of dictionaries
        tools_info = [
            {
                "name": tool.get("name", "")  # ty:ignore[no-matching-overload]
                if isinstance(tool, dict)
                else getattr(tool, "name", ""),
                "description": tool.get("description", "")  # ty:ignore[no-matching-overload]
                if isinstance(tool, dict)
                else getattr(tool, "description", ""),
                "parameters": tool.get("parameters", {})  # ty:ignore[no-matching-overload]
                if isinstance(tool, dict)
                else getattr(tool, "parameters", {}),
            }
            for tool in tools
        ]

    # Convert resources to dictionaries
    if resources is None:
        resources_info = []
    elif (
        isinstance(resources, list)
        and resources
        and isinstance(resources[0], MCPResource)
    ):
        resources_info = [resource.to_dict() for resource in resources]  # type: ignore
    else:
        # Assume it's already a list of dictionaries
        resources_info = resources or []

    info(f"Available: {len(tools_info)} tool(s), {len(resources_info)} resource(s).")

    step("Preparing prompt & JSON schema &")
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2)
        if resources_info
        else "No resources available"
    )

    prompt = f"""You are an expert at creating realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers.

Given the following available tools and resources from an MCP server, generate {num_scenarios} diverse, realistic scenarios that a user might want to accomplish using these tools.

AVAILABLE TOOLS:
{tools_description}

AVAILABLE RESOURCES:
{resources_description}

Requirements for scenarios:
1. Each scenario should be a task that can be accomplished using the available tools
2. Scenarios should vary in complexity - some simple (1-2 tool calls), some complex (multiple tool calls)
3. Scenarios should cover different use cases and tool combinations (though the task should not specify which tools to use)
4. Each scenario should be realistic - something a real user might actually want to do
5. Assign a difficulty rating from 1 (easy, single tool call) to 5 (hard, complex multi-step analysis)
6. The task should always include generating a summary of the work done and a thorough analysis and report of the results

You must respond with a JSON object containing a "scenarios" array of exactly {num_scenarios} objects. Each object must have:
- "task": string describing the scenario
- "difficulty": integer from 1-5 representing complexity
"""

    if custom_instructions:
        prompt += f"\n\nPay close attention to the following instructions when generating scenarios:\n\n{custom_instructions}"

    response_schema = {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "difficulty"],
                    "additionalProperties": False,
                },
                "minItems": num_scenarios,
                "maxItems": num_scenarios,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }

    step(f"Calling model: {_C.BOLD}{generator_model}{_C.RESET} &")
    client_openai = openai.OpenAI(
        api_key=generator_api_key,
        base_url=generator_base_url,
    )

    t1 = time.perf_counter()
    response = client_openai.chat.completions.create(
        model=generator_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )
    dt = time.perf_counter() - t1
    ok(f"Model responded in {dt:.2f}s.")

    content = response.choices[0].message.content
    if content is None:
        err("Model response content is None.")
        raise ValueError("Model response content is None")
    info(f"Raw content length: {len(content)} chars.")

    # Parse JSON
    try:
        result = json.loads(content)
    except Exception as e:
        err("Failed to parse JSON from model response.")
        dim(f"   Exception: {e}")
        dim("   First 500 chars of response content:")
        dim(content[:500] if content else "No content")
        raise

    # Extract scenarios
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    # Validate count
    if len(scenarios) != num_scenarios:
        err(f"Expected {num_scenarios} scenarios, got {len(scenarios)}.")
        raise ValueError(f"Expected {num_scenarios} scenarios, got {len(scenarios)}")

    ok(f"Parsed {len(scenarios)} scenario(s) successfully.")

    # Convert to ScenarioCollection
    scenario_collection = GeneratedScenarioCollection.from_dicts(scenarios)

    # Show difficulty distribution and preview using the collection methods
    scenario_collection.print_difficulty_distribution()

    if show_preview:
        scenario_collection.preview(n=min(5, num_scenarios))

    total_time = time.perf_counter() - t0
    ok(f"Generated {len(scenario_collection)} scenarios in {total_time:.2f}s total.")

    return scenario_collection
