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


class AlphaVantageClient:
    """Client for interacting with Alpha Vantage API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_data(self, function: str, **params) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage API"""
        query_params = {
            "function": function,
            "apikey": self.api_key,
            "datatype": "json",
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=query_params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if "Error Message" in data:
                    raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")

                if (
                    "Thank you for using Alpha Vantage! Please contact premium@alphavantage.co if you are targeting a higher API call volume."
                    in data
                ):
                    raise Exception(
                        "Alpha Vantage API Error: Thank you for using Alpha Vantage! Please contact premium@alphavantage.co if you are targeting a higher API call volume."
                    )

                return data


@click.command()
@click.option("--api-key", help="Alpha Vantage API key", envvar="ALPHAVANTAGE_API_KEY")
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
            "Error: Alpha Vantage API key is required. Set ALPHAVANTAGE_API_KEY environment variable or use --api-key option."
        )
        return 1

    app = Server("mcp-alphavantage")
    client = AlphaVantageClient(api_key)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_stock_quote",
                description="Get real-time stock quote for a symbol",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.Tool(
                name="get_time_series_daily",
                description="Get daily time series data for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        },
                        "outputsize": {
                            "type": "string",
                            "description": "Output size: compact (latest 100 data points)",
                            "enum": ["compact"],
                            "default": "compact",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            types.Tool(
                name="search_symbol",
                description="Search for stock symbols by keywords",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "Keywords to search for (e.g., company name)",
                        }
                    },
                    "required": ["keywords"],
                },
            ),
            types.Tool(
                name="get_company_overview",
                description="Get fundamental data and company overview",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.Tool(
                name="get_sma",
                description="Get Simple Moving Average (SMA) technical indicator",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        },
                        "interval": {
                            "type": "string",
                            "description": "Time interval",
                            "enum": [
                                "1min",
                                "5min",
                                "15min",
                                "30min",
                                "60min",
                                "daily",
                                "weekly",
                                "monthly",
                            ],
                            "default": "daily",
                        },
                        "time_period": {
                            "type": "integer",
                            "description": "Number of data points for SMA calculation",
                            "default": 30,
                        },
                        "series_type": {
                            "type": "string",
                            "description": "Price type to use for calculation",
                            "enum": ["close", "open", "high", "low"],
                            "default": "close",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            types.Tool(
                name="get_rsi",
                description="Get Relative Strength Index (RSI) technical indicator",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        },
                        "interval": {
                            "type": "string",
                            "description": "Time interval",
                            "enum": [
                                "daily",
                                "weekly",
                                "monthly",
                            ],
                            "default": "daily",
                        },
                        "time_period": {
                            "type": "integer",
                            "description": "Number of data points for RSI calculation",
                            "default": 14,
                        },
                        "series_type": {
                            "type": "string",
                            "description": "Price type to use for calculation",
                            "enum": ["close", "open", "high", "low"],
                            "default": "close",
                        },
                    },
                    "required": ["symbol"],
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
            if name == "get_stock_quote":
                data = await client.fetch_data(
                    "GLOBAL_QUOTE", symbol=arguments["symbol"]
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"Stock Quote for {arguments['symbol']}:\n{_format_json(data)}",
                    )
                ]

            elif name == "get_time_series_daily":
                outputsize = arguments.get("outputsize", "compact")
                data = await client.fetch_data(
                    "TIME_SERIES_DAILY",
                    symbol=arguments["symbol"],
                    outputsize=outputsize,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"Daily Time Series for {arguments['symbol']}:\n{_format_json(data)}",
                    )
                ]

            elif name == "search_symbol":
                data = await client.fetch_data(
                    "SYMBOL_SEARCH", keywords=arguments["keywords"]
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"Symbol Search Results for '{arguments['keywords']}':\n{_format_json(data)}",
                    )
                ]

            elif name == "get_company_overview":
                data = await client.fetch_data("OVERVIEW", symbol=arguments["symbol"])
                return [
                    types.TextContent(
                        type="text",
                        text=f"Company Overview for {arguments['symbol']}:\n{_format_json(data)}",
                    )
                ]

            elif name == "get_sma":
                data = await client.fetch_data(
                    "SMA",
                    symbol=arguments["symbol"],
                    interval=arguments.get("interval", "daily"),
                    time_period=arguments.get("time_period", 30),
                    series_type=arguments.get("series_type", "close"),
                )
                tech_analysis_key = "Technical Analysis: SMA"
                time_period = arguments.get("time_period", 30)
                # Alpha Vantage returns a dict keyed by timestamp; convert to list to slice
                data[tech_analysis_key] = dict(
                    list(data[tech_analysis_key].items())[:time_period]
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"SMA for {arguments['symbol']}:\n{_format_json(data)}",
                    )
                ]

            elif name == "get_rsi":
                data = await client.fetch_data(
                    "RSI",
                    symbol=arguments["symbol"],
                    interval=arguments.get("interval", "daily"),
                    time_period=arguments.get("time_period", 14),
                    series_type=arguments.get("series_type", "close"),
                )
                tech_analysis_key = "Technical Analysis: RSI"
                time_period = arguments.get("time_period", 14)
                # Alpha Vantage returns a dict keyed by timestamp; convert to list to slice
                data[tech_analysis_key] = dict(
                    list(data[tech_analysis_key].items())[:time_period]
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"RSI for {arguments['symbol']}:\n{_format_json(data)}",
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
