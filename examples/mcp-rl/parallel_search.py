# /// script
# requires-python = ">=3.12"
# dependencies = ["openpipe-art", "mcp>=1.11,<2", "python-dotenv"]
# [tool.uv.sources]
# openpipe-art = { path = "../..", editable = true }
# ///
"""Discover and call Parallel tools through the MCP-RL HTTP connection."""

import argparse
import asyncio
import uuid

from mcp import ClientSession
from mcp_rl.transport import connect_mcp_transport, parallel_search_params
from mcp_rl.utils import get_content_text

from art.mcp import MCPTool


async def main(query: str, url: str):
    async with connect_mcp_transport(parallel_search_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            # These schemas can also be passed to art.mcp.generate_scenarios.
            art_tools = [
                MCPTool(t.name, t.description or "", t.inputSchema) for t in tools.tools
            ]
            print("Available tools:", ", ".join(t.name for t in art_tools))
            session_id = str(uuid.uuid4())
            for name, arguments in (
                ("web_search", {"objective": query, "search_queries": [query]}),
                ("web_fetch", {"urls": [url], "objective": query}),
            ):
                result = await session.call_tool(
                    name, {**arguments, "session_id": session_id}
                )
                if result.isError:
                    raise RuntimeError(get_content_text(result))
                print(f"\n{name}:\n{get_content_text(result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="ART agent reinforcement trainer")
    parser.add_argument("--url", default="https://art.openpipe.ai")
    args = parser.parse_args()
    asyncio.run(main(args.query, args.url))
