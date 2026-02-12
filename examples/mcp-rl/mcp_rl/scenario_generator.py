"""Generate scenarios for MCP agent evaluation using OpenAI's o3 model."""

import argparse
import asyncio
import importlib.util
import json
import os
import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client
import openai

load_dotenv()


async def generate_scenarios(
    server_params_path: str,
    num_scenarios: int = 24,
    scenarios_dir: str = None,
) -> List[Dict[str, Any]]:
    """Generate scenarios for MCP agent evaluation using OpenAI's o3 model.

    Args:
        server_params_path: Relative path to the server_params file (e.g., "servers/python/mcp_alphavantage/server_params.py")
        num_scenarios: Number of scenarios to generate (default: 24)
        scenarios_dir: Directory to save scenarios (default: "scenarios" in same dir as server_params)

    Returns:
        List of scenario objects with 'task' and 'difficulty' fields
    """
    # Load server_params from the given path
    spec = importlib.util.spec_from_file_location("server_params", server_params_path)
    server_params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_params_module)
    server_params = server_params_module.server_params
    # Connect to MCP server to get available tools and resources
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()
            tools_info = []
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
                tools_info.append(tool_info)

            # Get available resources
            try:
                resources_result = await session.list_resources()
                resources_info = []
                for resource in resources_result.resources:
                    resource_info = {
                        "uri": str(resource.uri),
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType,
                    }
                    resources_info.append(resource_info)
            except Exception:
                # Some servers might not have resources
                resources_info = []

    # Prepare the prompt for o3
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

Example:
{{
  "scenarios": [
    {{"task": "Get the current stock price for Apple (AAPL)", "difficulty": 1}},
    {{"task": "Compare the 30-day SMA with current price for Tesla and determine if it's above or below the moving average and generate a thorough analysis and report", "difficulty": 2}}
  ]
}}"""

    # Call OpenAI's model with structured JSON output
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define the JSON schema for the response
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

    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    result = json.loads(content)

    # Extract scenarios from the response
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        # If the response is just an array
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    # Validate we got exactly the right number
    if len(scenarios) != num_scenarios:
        raise ValueError(f"Expected {num_scenarios} scenarios, got {len(scenarios)}")

    # Determine scenarios directory - default to "scenarios" in same dir as server_params
    if scenarios_dir is None:
        server_params_dir = os.path.dirname(server_params_path)
        scenarios_dir = os.path.join(server_params_dir, "scenarios")

    # Create scenarios directory if it doesn't exist
    os.makedirs(scenarios_dir, exist_ok=True)

    # Shuffle scenarios randomly
    random.shuffle(scenarios)

    # Split into train (2/3) and val (1/3)
    train_size = int(len(scenarios) * 2 / 3)
    train_scenarios = scenarios[:train_size]
    val_scenarios = scenarios[train_size:]

    # Save train scenarios
    train_file = os.path.join(scenarios_dir, "train.jsonl")
    with open(train_file, "w") as f:
        for scenario in train_scenarios:
            f.write(json.dumps(scenario) + "\n")

    # Save val scenarios
    val_file = os.path.join(scenarios_dir, "val.jsonl")
    with open(val_file, "w") as f:
        for scenario in val_scenarios:
            f.write(json.dumps(scenario) + "\n")

    print(f"Saved {len(train_scenarios)} training scenarios to {train_file}")
    print(f"Saved {len(val_scenarios)} validation scenarios to {val_file}")

    return scenarios


async def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate scenarios for MCP agent evaluation using OpenAI's o3 model."
    )
    parser.add_argument(
        "server_params_path",
        help="Path to the server_params.py file (e.g., 'servers/python/mcp_alphavantage/server_params.py')",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=24,
        help="Number of scenarios to generate (default: 24)",
    )
    parser.add_argument(
        "--scenarios-dir",
        default=None,
        help="Directory to save scenarios (default: scenarios/ in same directory as server_params)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only show minimal output")

    args = parser.parse_args()

    # Validate that the server_params file exists
    if not os.path.exists(args.server_params_path):
        print(f"Error: Server params file not found: {args.server_params_path}")
        return 1

    # Determine scenarios directory path - default to scenarios/ in same directory as server_params
    scenarios_dir = args.scenarios_dir
    if scenarios_dir is None:
        server_params_dir = os.path.dirname(args.server_params_path)
        scenarios_dir = os.path.join(server_params_dir, "scenarios")

    if not args.quiet:
        print(
            f"Generating {args.num_scenarios} scenarios using {args.server_params_path}..."
        )
        print(f"Output will be saved to: {scenarios_dir}/")

    try:
        scenarios = await generate_scenarios(
            args.server_params_path,
            num_scenarios=args.num_scenarios,
            scenarios_dir=scenarios_dir,
        )

        if not args.quiet:
            print(f"\nGenerated {len(scenarios)} scenarios:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"{i}. Task: {scenario['task']}")
                print(f"   Difficulty: {scenario['difficulty']}/5")

        return 0

    except Exception as e:
        print(f"Error generating scenarios: {e}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
