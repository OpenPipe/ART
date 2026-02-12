import asyncio
from typing import Any, Dict, Optional

import aiohttp
import click
from mcp.server.lowlevel import Server
import mcp.types as types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class BalldontlieClient:
    """Client for interacting with Balldontlie NBA API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.balldontlie.io/v1"

    async def fetch_data(self, endpoint: str, **params) -> Dict[str, Any]:
        """Fetch data from Balldontlie API"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/{endpoint}", params=params, headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if "error" in data:
                    raise Exception(f"Balldontlie API Error: {data['error']}")

                return data


@click.command()
@click.option("--api-key", help="Balldontlie API key", envvar="BALLDONTLIE_API_KEY")
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(api_key: Optional[str], port: int, transport: str) -> int:
    if not api_key:
        click.echo(
            "Error: Balldontlie API key is required. Set BALLDONTLIE_API_KEY environment variable or use --api-key option."
        )
        return 1

    app = Server("mcp-balldontlie")
    client = BalldontlieClient(api_key)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_teams",
                description="Get all NBA teams or a specific team by ID (Free tier)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "team_id": {
                            "type": "integer",
                            "description": "Optional specific team ID",
                        },
                        "division": {
                            "type": "string",
                            "description": "Filter by division",
                        },
                        "conference": {
                            "type": "string",
                            "description": "Filter by conference (East/West)",
                        },
                    },
                },
            ),
            types.Tool(
                name="get_players",
                description="Get NBA players with optional filters (Free tier)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "player_id": {
                            "type": "integer",
                            "description": "Optional specific player ID",
                        },
                        "search": {
                            "type": "string",
                            "description": "Search players by name",
                        },
                        "first_name": {
                            "type": "string",
                            "description": "Filter by first name",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "Filter by last name",
                        },
                        "team_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Filter by team IDs",
                        },
                        "player_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Filter by specific player IDs",
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Cursor for pagination",
                        },
                        "per_page": {
                            "type": "integer",
                            "description": "Results per page (max 100)",
                            "default": 25,
                        },
                    },
                },
            ),
            types.Tool(
                name="get_games",
                description="Get NBA games with optional filters (Free tier)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "game_id": {
                            "type": "integer",
                            "description": "Optional specific game ID",
                        },
                        "dates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by dates (YYYY-MM-DD format)",
                        },
                        "seasons": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Filter by seasons",
                        },
                        "team_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Filter by team IDs",
                        },
                        "postseason": {
                            "type": "boolean",
                            "description": "Include postseason games",
                            "default": False,
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date for date range (YYYY-MM-DD format)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date for date range (YYYY-MM-DD format)",
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Cursor for pagination",
                        },
                        "per_page": {
                            "type": "integer",
                            "description": "Results per page (max 100)",
                            "default": 25,
                        },
                    },
                },
            ),
        ]

    @app.call_tool()
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, Exception)
        ),
    )
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "get_teams":
                if "team_id" in arguments:
                    endpoint = f"teams/{arguments['team_id']}"
                    params = {}
                else:
                    endpoint = "teams"
                    params = {
                        k: v
                        for k, v in arguments.items()
                        if k in ["division", "conference"]
                    }

                data = await client.fetch_data(endpoint, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"NBA Teams:\n{_format_json(data)}",
                    )
                ]

            elif name == "get_players":
                if "player_id" in arguments:
                    endpoint = f"players/{arguments['player_id']}"
                    params = {}
                else:
                    endpoint = "players"
                    params = {
                        k: v
                        for k, v in arguments.items()
                        if k
                        in [
                            "cursor",
                            "per_page",
                            "search",
                            "first_name",
                            "last_name",
                            "team_ids",
                            "player_ids",
                        ]
                    }

                data = await client.fetch_data(endpoint, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"NBA Players:\n{_format_json(data)}",
                    )
                ]

            elif name == "get_games":
                if "game_id" in arguments:
                    endpoint = f"games/{arguments['game_id']}"
                    params = {}
                else:
                    endpoint = "games"
                    params = {
                        k: v
                        for k, v in arguments.items()
                        if k
                        in [
                            "cursor",
                            "per_page",
                            "dates",
                            "seasons",
                            "team_ids",
                            "postseason",
                            "start_date",
                            "end_date",
                        ]
                    }

                data = await client.fetch_data(endpoint, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"NBA Games:\n{_format_json(data)}",
                    )
                ]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        asyncio.run(arun())

    return 0


def _format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for display"""
    import json

    return json.dumps(data, indent=2)


if __name__ == "__main__":
    main()
