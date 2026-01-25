generate_from_http_mcp.py 
#!/usr/bin/env python3
"""
Advanced scenario generator from streamable HTTP MCP server.

Features:
- Connects to MCP server via HTTP
- Discovers tools and resources
- Generates scenarios with filtering options
- Saves to JSON with rich metadata
- Exports to multiple formats
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from art.mcp import generate_scenarios, GeneratedScenarioCollection


class MCPScenarioGenerator:
    """Generator for scenarios from MCP server."""
    
    def __init__(
        self,
        mcp_server_url: str,
        llm_model: str = "gpt-oss", # CHANGE THIS
        llm_api_key: str = "sk-1234", # CHANGE THIS
        llm_base_url: str = "http://vllm:8000/v1" # CHANGE THIS
    ):
        self.mcp_server_url = mcp_server_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        
        self.tools: List[Dict] = []
        self.resources: List[Dict] = []
        self.scenarios: Optional[GeneratedScenarioCollection] = None
    
    async def discover_capabilities(self) -> bool:
        """Discover MCP server capabilities."""
        print(f"🔍 Connecting to MCP server at {self.mcp_server_url}...")
        
        try:
            async with streamablehttp_client(self.mcp_server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize
                    await session.initialize()
                    print("✓ Connected successfully")
                    
                    # Get tools
                    tools_response = await session.list_tools()
                    self.tools = [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        for tool in tools_response.tools
                    ]
                    
                    # Get resources
                    resources_response = await session.list_resources()
                    self.resources = [
                        {
                            "uri": str(resource.uri),  # Convert AnyUrl to string
                            "name": resource.name or str(resource.uri).split("/")[-1],
                            "description": resource.description or "",
                            "mimeType": getattr(resource, 'mimeType', None) or "application/octet-stream"
                        }
                        for resource in resources_response.resources
                    ]
                    
                    print(f"✓ Found {len(self.tools)} tools and {len(self.resources)} resources")
                    return True
                    
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def show_capabilities(self):
        """Display discovered capabilities."""
        print("\n" + "=" * 70)
        print("📋 Discovered Capabilities")
        print("=" * 70)
        
        print(f"\n🔧 Tools ({len(self.tools)}):")
        for i, tool in enumerate(self.tools[:10], 1):
            desc = tool['description'][:60] + "..." if len(tool['description']) > 60 else tool['description']
            print(f"  {i:2d}. {tool['name']}")
            if desc:
                print(f"      {desc}")
        
        if len(self.tools) > 10:
            print(f"  ... and {len(self.tools) - 10} more")
        
        print(f"\n📚 Resources ({len(self.resources)}):")
        for i, resource in enumerate(self.resources[:10], 1):
            desc = resource['description'][:60] + "..." if len(resource['description']) > 60 else resource['description']
            print(f"  {i:2d}. {resource['name']}")
            if desc:
                print(f"      {desc}")
        
        if len(self.resources) > 10:
            print(f"  ... and {len(self.resources) - 10} more")
    
    async def generate(
        self,
        num_scenarios: int = 10,
        difficulty_range: Optional[tuple] = None,
        custom_instructions: Optional[str] = None
    ) -> bool:
        """Generate scenarios."""
        print("\n" + "=" * 70)
        print("🎯 Generating Scenarios")
        print("=" * 70)
        print()
        
        if not self.tools and not self.resources:
            print("❌ No tools or resources available")
            return False
        
        try:
            instructions = custom_instructions or f"""
            Generate realistic, diverse scenarios that:
            1. Effectively use the {len(self.tools)} available tools
            2. Reference the {len(self.resources)} available resources when relevant
            3. Cover different difficulty levels from simple to complex
            4. Represent real-world use cases
            5. Include specific details about what needs to be accomplished
            """
            
            self.scenarios = await generate_scenarios(
                tools=self.tools,
                resources=self.resources,
                num_scenarios=num_scenarios,
                show_preview=True,
                custom_instructions=instructions,
                generator_model=self.llm_model,
                generator_api_key=self.llm_api_key,
                generator_base_url=self.llm_base_url,
            )
            
            # Filter by difficulty if specified
            if difficulty_range:
                min_diff, max_diff = difficulty_range
                self.scenarios = self.scenarios.filter_by_difficulty(
                    min_difficulty=min_diff,
                    max_difficulty=max_diff
                )
                print(f"\n✓ Filtered to difficulty range {min_diff}-{max_diff}: {len(self.scenarios)} scenarios")
            
            return True
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save(self, output_file: str = "scenarios.json", include_metadata: bool = True):
        """Save scenarios to JSON file."""
        if not self.scenarios:
            print("❌ No scenarios to save")
            return False
        
        print("\n" + "=" * 70)
        print("💾 Saving Scenarios")
        print("=" * 70)
        
        output_path = Path(output_file)
        
        try:
            if include_metadata:
                # Include rich metadata
                summary = self.scenarios.get_summary()
                data = {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "mcp_server_url": self.mcp_server_url,
                        "llm_model": self.llm_model,
                        "num_tools": len(self.tools),
                        "num_resources": len(self.resources),
                        "tool_names": [t['name'] for t in self.tools],
                        "resource_names": [r['name'] for r in self.resources],
                        "summary": summary
                    },
                    "scenarios": [
                        {
                            "task": scenario.task,
                            "difficulty": scenario.difficulty
                        }
                        for scenario in self.scenarios
                    ]
                }
            else:
                # Just scenarios
                data = [
                    {
                        "task": scenario.task,
                        "difficulty": scenario.difficulty
                    }
                    for scenario in self.scenarios
                ]
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved to: {output_path}")
            print(f"✓ File size: {output_path.stat().st_size:,} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            return False
    
    def show_summary(self):
        """Display summary statistics."""
        if not self.scenarios:
            return
        
        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        
        summary = self.scenarios.get_summary()
        
        print(f"\n📈 Statistics:")
        print(f"  Total scenarios: {summary['total_scenarios']}")
        print(f"  Average difficulty: {summary['avg_difficulty']:.1f}/5")
        print(f"  Average task length: {summary['avg_task_length']:.0f} characters")
        
        print(f"\n📊 Difficulty Distribution:")
        max_count = max(summary['difficulty_distribution'].values())
        for difficulty in range(1, 6):
            count = summary['difficulty_distribution'].get(difficulty, 0)
            percentage = (count / summary['total_scenarios'] * 100) if summary['total_scenarios'] > 0 else 0
            bar = "█" * int(count / max_count * 30) if max_count > 0 else ""
            print(f"  {difficulty}/5: {count:3d} ({percentage:5.1f}%)  {bar}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate scenarios from MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--server",
        default="http://mcp.server:8000/mcp", # CHANGE THIS
        help="MCP server URL (default: http://mcp.server:8000/mcp)" # CHANGE THIS
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of scenarios to generate (default: 10)"
    )
    parser.add_argument(
        "--output",
        default="scenarios.json",
        help="Output file path (default: scenarios.json)"
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        choices=range(1, 6),
        help="Minimum difficulty (1-5)"
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        choices=range(1, 6),
        help="Maximum difficulty (1-5)"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-oss",
        help="LLM model name (default: gpt-oss)"
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://vllm:8000/v1",
        help="LLM API base URL (default: http://vllm:8000/v1)"
    )
    
    args = parser.parse_args()
    
    # Validate difficulty range
    difficulty_range = None
    if args.min_difficulty or args.max_difficulty:
        min_d = args.min_difficulty or 1
        max_d = args.max_difficulty or 5
        if min_d > max_d:
            print("❌ Error: min-difficulty must be <= max-difficulty")
            return 1
        difficulty_range = (min_d, max_d)
    
    print("=" * 70)
    print("🚀 MCP Scenario Generator")
    print("=" * 70)
    print(f"\n📍 Server: {args.server}")
    print(f"🤖 LLM: {args.llm_model}")
    print(f"🎯 Scenarios: {args.num}")
    if difficulty_range:
        print(f"⚡ Difficulty: {difficulty_range[0]}-{difficulty_range[1]}")
    print()
    
    # Create generator
    generator = MCPScenarioGenerator(
        mcp_server_url=args.server,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url
    )
    
    # Step 1: Discover capabilities
    if not await generator.discover_capabilities():
        return 1
    
    generator.show_capabilities()
    
    # Step 2: Generate scenarios
    if not await generator.generate(
        num_scenarios=args.num,
        difficulty_range=difficulty_range
    ):
        return 1
    
    # Step 3: Save results
    if not generator.save(output_file=args.output):
        return 1
    
    # Step 4: Show summary
    generator.show_summary()
    
    print("\n✅ Complete!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
