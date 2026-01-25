"""Scenario generation for MCP tools using local MCP server for JSON formatting."""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from art.mcp.types import GeneratedScenarioCollection, MCPResource, MCPTool
from art.utils.logging import _C, dim, err, info, ok, step


async def generate_scenarios(
    tools: List[MCPTool] | List[Dict[str, Any]],
    resources: List[MCPResource] | List[Dict[str, Any]] = [],
    num_scenarios: int = 24,
    show_preview: bool = True,
    custom_instructions: Optional[str] = None,
    generator_model: str = "openai/gpt-4.1-mini",
    generator_api_key: Optional[str] = None,
    generator_base_url: str = "https://openrouter.ai/api/v1",
    mcp_server_command: str = "python",
    mcp_server_args: Optional[List[str]] = None,
) -> GeneratedScenarioCollection:
    """
    Generate scenarios for MCP tools using an MCP server for JSON formatting.

    Args:
        tools: List of Tool objects or list of tool dictionaries
        resources: Optional list of Resource objects or list of resource dictionaries
        num_scenarios: Number of scenarios to generate (default: 24)
        show_preview: Whether to show a preview of generated scenarios (default: True)
        custom_instructions: Optional custom instructions for scenario generation
        generator_model: Model to use for generation (default: "openai/gpt-4.1-mini")
        generator_api_key: API key for the generator model
        generator_base_url: Base URL for the API (default: OpenRouter)
        mcp_server_command: Command to start MCP server (default: "python")
        mcp_server_args: Args for MCP server (default: None, will use bundled format_server)

    Returns:
        GeneratedScenarioCollection containing the generated scenarios
    """
    if mcp_server_args is None:
        mcp_server_args = ["format_server.py"]  # Will be replaced with bundled version

    t0 = time.perf_counter()

    # Handle API key
    if generator_api_key is None:
        generator_api_key = os.getenv("OPENROUTER_API_KEY")
        if not generator_api_key:
            raise ValueError(
                "generator_api_key is required or OPENROUTER_API_KEY env var must be set"
            )

    # Validate inputs
    if not tools and not resources:
        raise ValueError("At least one tool or resource must be provided")

    ok(f"Using model: {generator_model}")

    # Convert tools to dictionaries
    if tools and hasattr(tools[0], 'to_dict'):
        tools_info = [tool.to_dict() for tool in tools]
    else:
        tools_info = [
            {
                "name": tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", ""),
                "description": tool.get("description", "") if isinstance(tool, dict) else getattr(tool, "description", ""),
                "parameters": tool.get("parameters", {}) if isinstance(tool, dict) else getattr(tool, "parameters", {}),
            }
            for tool in tools
        ]

    # Convert resources to dictionaries
    if resources and hasattr(resources[0], 'to_dict'):
        resources_info = [resource.to_dict() for resource in resources]
    else:
        resources_info = resources or []
    
    # Ensure all values are JSON-serializable (convert AnyUrl, etc.)
    def make_serializable(obj):
        """Convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj
    
    resources_info = [make_serializable(r) for r in resources_info]

    info(f"Available: {len(tools_info)} tool(s), {len(resources_info)} resource(s).")

    step("Preparing prompt for scenario generation")
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2) if resources_info else "No resources available"
    )

    # Simple prompt that asks for plain text output
    prompt = f"""Generate {num_scenarios} diverse, realistic scenarios for testing AI agents with these MCP tools and resources.

AVAILABLE TOOLS:
{tools_description}

AVAILABLE RESOURCES:
{resources_description}

Requirements:
1. Each scenario should use the available tools
2. Vary complexity from simple (1-2 tool calls) to complex (multiple tool calls)
3. Cover different use cases and tool combinations
4. Make scenarios realistic - what real users would actually want to do
5. Rate difficulty from 1 (easy, single tool) to 5 (hard, complex multi-step)
6. Tasks should include generating summaries and thorough analysis/reports

{f"CUSTOM INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

For each scenario, provide:
- A task description (what the user wants to accomplish)
- A difficulty rating (1-5)

Format each scenario as:
SCENARIO N:
Task: [description]
Difficulty: [1-5]

Generate exactly {num_scenarios} scenarios."""

    step(f"Calling model: {_C.BOLD}{generator_model}{_C.RESET}")
    client = openai.OpenAI(api_key=generator_api_key, base_url=generator_base_url)

    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model=generator_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
    )
    dt = time.perf_counter() - t1
    ok(f"Model responded in {dt:.2f}s.")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model response content is None")

    info(f"Raw content length: {len(content)} chars.")

    # Parse plain text response
    step("Parsing model output")
    scenarios_raw = _parse_plain_text_scenarios(content)
    
    if len(scenarios_raw) != num_scenarios:
        dim(f"   Warning: Expected {num_scenarios} scenarios, got {len(scenarios_raw)}.")

    # Use MCP server to format into proper JSON
    step("Connecting to MCP server for JSON formatting")
    
    # If no custom command provided, use the bundled format_server
    if mcp_server_command == "python" and mcp_server_args == ["format_server.py"]:
        import art.mcp.format_server
        server_script = art.mcp.format_server.__file__
        server_params = StdioServerParameters(
            command=mcp_server_command,
            args=[server_script],
        )
    else:
        server_params = StdioServerParameters(
            command=mcp_server_command,
            args=mcp_server_args,
        )
    
    formatted_scenarios = []
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Get available tools
            tools_response = await session.list_tools()
            
            if not tools_response.tools:
                raise ValueError("MCP server has no tools available")
            
            format_tool = tools_response.tools[0]  # Use first tool
            ok(f"Using MCP tool: {format_tool.name}")
            
            # Format each scenario through MCP
            for i, scenario in enumerate(scenarios_raw):
                result = await session.call_tool(
                    format_tool.name,
                    arguments={
                        "task": scenario["task"],
                        "difficulty": scenario["difficulty"],
                    }
                )
                
                # Extract text content
                if result.content and hasattr(result.content[0], 'text'):
                    formatted_scenarios.append(json.loads(result.content[0].text))
                
                if (i + 1) % 5 == 0:
                    info(f"Formatted {i + 1}/{len(scenarios_raw)} scenarios")

    ok(f"Formatted {len(formatted_scenarios)} scenarios via MCP server.")

    # Create collection
    scenario_collection = GeneratedScenarioCollection.from_dicts(formatted_scenarios)
    scenario_collection.print_difficulty_distribution()

    if show_preview:
        scenario_collection.preview(n=min(5, num_scenarios))

    total_time = time.perf_counter() - t0
    ok(f"Generated {len(scenario_collection)} scenarios in {total_time:.2f}s total.")

    return scenario_collection


def _parse_plain_text_scenarios(content: str) -> List[Dict[str, Any]]:
    """Parse plain text scenarios from model output."""
    scenarios = []
    lines = content.strip().split("\n")
    
    current_scenario = {}
    for line in lines:
        line = line.strip()
        
        if line.startswith("Task:") or line.startswith("task:"):
            current_scenario["task"] = line.split(":", 1)[1].strip()
        elif line.startswith("Difficulty:") or line.startswith("difficulty:"):
            try:
                diff = int(line.split(":", 1)[1].strip().split()[0])
                current_scenario["difficulty"] = max(1, min(5, diff))
            except (ValueError, IndexError):
                current_scenario["difficulty"] = 3
            
            # Scenario complete
            if current_scenario.get("task"):
                scenarios.append(current_scenario)
                current_scenario = {}
    
    # Handle last scenario if needed
    if current_scenario.get("task") and current_scenario.get("difficulty"):
        scenarios.append(current_scenario)
    
    return scenarios
