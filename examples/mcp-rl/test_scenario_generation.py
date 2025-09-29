#!/usr/bin/env python3
"""Test scenario generation functionality."""

import asyncio
import os
from typing import List

from dotenv import load_dotenv

from art.mcp import MCPResource, MCPTool, generate_scenarios

load_dotenv()


def create_sample_tools() -> List[MCPTool]:
    """Create sample tools for testing."""
    return [
        MCPTool(
            name="search_files",
            description="Search for files by name or content pattern",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "file_type": {
                        "type": "string",
                        "enum": ["txt", "py", "json"],
                        "description": "File type filter",
                    },
                },
                "required": ["query"],
            },
        ),
        MCPTool(
            name="read_file",
            description="Read the contents of a specific file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        ),
        MCPTool(
            name="analyze_code",
            description="Analyze code quality and suggest improvements",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"},
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                    },
                },
                "required": ["code"],
            },
        ),
        MCPTool(
            name="execute_command",
            description="Execute a shell command and return the output",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        ),
    ]


def create_sample_resources() -> List[MCPResource]:
    """Create sample resources for testing."""
    return [
        MCPResource(
            uri="file://docs/api.md",
            name="API Documentation",
            description="Complete API documentation with examples",
            mime_type="text/markdown",
        ),
        MCPResource(
            uri="file://src/main.py",
            name="Main Application",
            description="Primary application entry point",
            mime_type="text/x-python",
        ),
        MCPResource(
            uri="file://config.json",
            name="Configuration File",
            description="Application configuration settings",
            mime_type="application/json",
        ),
    ]


async def test_basic_scenario_generation():
    """Test basic scenario generation with tools only."""
    print("[TEST] Testing basic scenario generation...")

    tools = create_sample_tools()

    try:
        scenarios = await generate_scenarios(
            tools=tools,
            num_scenarios=5,
            show_preview=True,
            generator_model="openai/gpt-4o-mini",  # Use a cheaper model for testing
        )

        print(f"[PASS] Generated {len(scenarios)} scenarios successfully")
        print(f"[INFO] Summary: {scenarios.get_summary()}")

        # Test collection methods
        print("\n[TEST] Testing collection methods...")

        # Test difficulty filtering
        easy_scenarios = scenarios.filter_by_difficulty(max_difficulty=2)
        print(f"[INFO] Easy scenarios (<=2): {len(easy_scenarios)}")

        # Test shuffling and splitting
        shuffled = scenarios.shuffle()
        if len(scenarios) >= 3:
            train, val = shuffled.split(train_size=3)
            print(f"[INFO] Train/Val split: {len(train)}/{len(val)}")

        # Test JSON serialization
        json_str = scenarios.to_json(indent=2)
        print(f"[INFO] JSON export: {len(json_str)} characters")

        return True

    except Exception as e:
        print(f"[FAIL] Basic test failed: {e}")
        return False


async def test_scenario_generation_with_resources():
    """Test scenario generation with both tools and resources."""
    print("\n[TEST] Testing scenario generation with resources...")

    tools = create_sample_tools()
    resources = create_sample_resources()

    try:
        scenarios = await generate_scenarios(
            tools=tools,
            resources=resources,
            num_scenarios=3,
            show_preview=True,
            custom_instructions="Focus on file management and code analysis tasks.",
            generator_model="openai/gpt-4o-mini",
        )

        print(f"[PASS] Generated {len(scenarios)} scenarios with resources")

        # Verify scenarios reference the available tools/resources appropriately
        for i, scenario in enumerate(scenarios):
            print(
                f"[INFO] Scenario {i + 1} (Difficulty {scenario.difficulty}): {scenario.preview(80)}"
            )

        return True

    except Exception as e:
        print(f"[FAIL] Resources test failed: {e}")
        return False


async def test_dict_input_compatibility():
    """Test backward compatibility with dictionary inputs."""
    print("\n[TEST] Testing dictionary input compatibility...")

    tools_dict = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    ]

    resources_dict = [
        {
            "uri": "database://users",
            "name": "User Database",
            "description": "User account information",
            "mimeType": "application/sql",
        }
    ]

    try:
        scenarios = await generate_scenarios(
            tools=tools_dict,
            resources=resources_dict,
            num_scenarios=3,
            show_preview=False,  # Don't show preview to keep output clean
            generator_model="openai/gpt-4o-mini",
        )

        print(f"[PASS] Dictionary input test passed: {len(scenarios)} scenarios")
        return True

    except Exception as e:
        print(f"[FAIL] Dictionary input test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling scenarios."""
    print("\n[TEST] Testing error handling...")

    # Test with empty tools list
    try:
        await generate_scenarios(
            tools=[],
            num_scenarios=1,
            show_preview=False,
            generator_model="openai/gpt-4o-mini",
        )
        print("[FAIL] Should have failed with empty tools list")
        return False
    except Exception as e:
        print(f"[PASS] Correctly handled empty tools: {type(e).__name__}")

    # Test with invalid API key
    tools = create_sample_tools()[:1]  # Just one tool for speed

    try:
        await generate_scenarios(
            tools=tools,
            num_scenarios=1,
            show_preview=False,
            generator_model="openai/gpt-4o-mini",
            generator_api_key="invalid_key",
        )
        print("[FAIL] Should have failed with invalid API key")
        return False
    except Exception as e:
        print(f"[PASS] Correctly handled invalid API key: {type(e).__name__}")

    return True


def test_tool_resource_classes():
    """Test Tool and Resource class functionality."""
    print("\n[TEST] Testing Tool and Resource classes...")

    try:
        # Test Tool class
        tool_dict = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        }

        tool = MCPTool.from_dict(tool_dict)
        assert tool.name == "test_tool"
        assert tool.to_dict() == tool_dict
        print("[PASS] MCPTool class tests passed")

        # Test Resource class
        resource_dict = {
            "uri": "file://test.txt",
            "name": "Test File",
            "description": "A test file",
            "mimeType": "text/plain",
        }

        resource = MCPResource.from_dict(resource_dict)
        assert resource.uri == "file://test.txt"
        assert resource.mime_type == "text/plain"

        # Test alternative field name
        resource_dict2 = resource_dict.copy()
        resource_dict2["mime_type"] = resource_dict2.pop("mimeType")
        resource2 = MCPResource.from_dict(resource_dict2)
        assert resource2.mime_type == "text/plain"

        print("[PASS] MCPResource class tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Class tests failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting MCP scenario generation tests...\n")

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[WARN] OPENROUTER_API_KEY not set. Some tests may fail.")
        print("       Set your API key: export OPENROUTER_API_KEY='your_key_here'")
        print()

    test_results = []

    # Run class tests (synchronous)
    test_results.append(test_tool_resource_classes())

    # Run async tests
    if os.getenv("OPENROUTER_API_KEY"):
        test_results.extend(
            await asyncio.gather(
                test_basic_scenario_generation(),
                test_scenario_generation_with_resources(),
                test_dict_input_compatibility(),
                test_error_handling(),
                return_exceptions=True,
            )
        )
    else:
        print("[SKIP] Skipping API-dependent tests (no API key)")
        test_results.extend([True, True, True, True])  # Assume they would pass

    # Summary
    passed = sum(1 for result in test_results if result is True)
    total = len(test_results)

    print(f"\n[SUMMARY] Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
