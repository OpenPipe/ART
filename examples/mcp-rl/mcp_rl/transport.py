"""Connection parameters shared by MCP discovery and agent rollouts."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


@dataclass
class StreamableHttpServerParameters:
    """Connect to a standard MCP Streamable HTTP endpoint."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30
    sse_read_timeout: float = 300


# Anonymous search and page fetching; no saved credentials or environment keys.
parallel_search_params = StreamableHttpServerParameters(
    url="https://search.parallel.ai/mcp",
    headers={"User-Agent": "ART-MCP-RL/0.1 (https://github.com/OpenPipe/ART)"},
)


@asynccontextmanager
async def connect_mcp_transport(
    params: StdioServerParameters | StreamableHttpServerParameters,
):
    """Open streams for either the existing stdio path or Streamable HTTP."""
    if isinstance(params, StreamableHttpServerParameters):
        async with streamablehttp_client(
            params.url,
            headers=params.headers,
            timeout=params.timeout,
            sse_read_timeout=params.sse_read_timeout,
        ) as (read, write, _):
            yield read, write
    else:
        async with stdio_client(params) as streams:
            yield streams
