"""ART MCP package."""

from .mcp_server import AlphaMcpServer, McpServer

__all__ = ["rollout", "McpScenario", "McpServer", "AlphaMcpServer"]


def __getattr__(name: str):
    # Tool-only HTTP clients should not initialize rollout tracing or model services.
    if name in {"McpScenario", "rollout"}:
        from .rollout import McpScenario, rollout

        globals().update(McpScenario=McpScenario, rollout=rollout)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
