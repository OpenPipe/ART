"""MCP Server wrapper class for integration with ART."""

from abc import ABC, abstractmethod
import json
import ssl
from typing import Any, Dict, List, Optional

import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpServer(ABC):
    """Abstract base class for all MCP server implementations."""

    def __init__(self):
        """Initialize MCP server."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the MCP server connection."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the MCP server connection."""
        pass

    @abstractmethod
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server.

        Returns:
            List of tool schemas in OpenAI format
        """
        pass

    @abstractmethod
    async def apply_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class LocalMcpServer(McpServer):
    """MCP server that runs locally via stdio communication."""

    def __init__(self):
        """Initialize local MCP server wrapper."""
        super().__init__()
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._stdio_context = None
        self._session_context = None

    @abstractmethod
    async def init_server(self) -> StdioServerParameters:
        """Initialize server configuration.

        Returns:
            StdioServerParameters for the specific server implementation
        """
        pass

    async def start(self) -> None:
        """Start the MCP server and establish connection."""
        server_params = await self.init_server()

        # Start stdio client
        self._stdio_context = stdio_client(server_params)
        self._read, self._write = await self._stdio_context.__aenter__()

        # Start session
        self._session_context = ClientSession(self._read, self._write)
        self._session = await self._session_context.__aenter__()

        # Initialize session
        await self._session.initialize()

    async def stop(self) -> None:
        """Stop the MCP server and clean up connections."""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            self._session = None

        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None
            self._read = None
            self._write = None

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in OpenAI format.

        Returns:
            List of tool schemas in OpenAI format
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        # Get available tools from MCP server
        tools_result = await self._session.list_tools()

        # Convert to OpenAI format
        tool_schemas = []
        for tool in tools_result.tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"MCP tool: {tool.name}",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
            tool_schemas.append(tool_schema)

        return tool_schemas

    async def apply_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Apply a tool with given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string

        Raises:
            RuntimeError: If server not started
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        result = await self._session.call_tool(tool_name, arguments)
        # Extract text content from MCP result
        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list):
                # Handle list of content items
                content_text = ""
                for item in result.content:
                    if hasattr(item, "text"):
                        content_text += item.text
                    else:
                        content_text += str(item)
            else:
                content_text = (
                    result.content.text
                    if hasattr(result.content, "text")
                    else str(result.content)
                )
        else:
            content_text = str(result)

        return content_text


class RemoteMcpServer(McpServer):
    """MCP server that communicates with a hosted API server via HTTP."""

    def __init__(self, api_endpoint: str):
        """Initialize API-based MCP server.

        Args:
            api_endpoint: Base URL of the hosted MCP server API
        """
        super().__init__()
        self.api_endpoint = api_endpoint.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Start the API session."""
        # Create SSL context that doesn't verify certificates (for development)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self._session = aiohttp.ClientSession(connector=connector)

    async def stop(self) -> None:
        """Stop the API session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the API server.

        Returns:
            List of tool schemas in OpenAI format
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        async with self._session.get(f"{self.api_endpoint}/tools") as response:
            response.raise_for_status()
            tools_data = await response.json()

            # Convert to OpenAI format if needed
            tool_schemas = []
            for tool in tools_data.get("tools", []):
                if "function" in tool:
                    # Already in OpenAI format
                    tool_schemas.append(tool)
                else:
                    # Convert from MCP format to OpenAI format
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get(
                                "description", f"API tool: {tool.get('name', '')}"
                            ),
                            "parameters": tool.get(
                                "inputSchema", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                    tool_schemas.append(tool_schema)

            return tool_schemas

    async def apply_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool via the API server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string

        Raises:
            RuntimeError: If server not started
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        payload = {"tool_name": tool_name, "arguments": arguments}

        async with self._session.post(
            f"{self.api_endpoint}/execute",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            result_data = await response.json()

            # Extract result text
            if "result" in result_data:
                if isinstance(result_data["result"], str):
                    return result_data["result"]
                elif isinstance(result_data["result"], dict):
                    # Handle structured result
                    if "content" in result_data["result"]:
                        content = result_data["result"]["content"]
                        if isinstance(content, list):
                            # Handle list of content items
                            content_text = ""
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    content_text += item["text"]
                                else:
                                    content_text += str(item)
                            return content_text
                        elif isinstance(content, dict) and "text" in content:
                            return content["text"]
                        else:
                            return str(content)
                    else:
                        return json.dumps(result_data["result"])
                else:
                    return str(result_data["result"])
            else:
                return json.dumps(result_data)


class AlphaMcpServer(LocalMcpServer):
    """AlphaVantage MCP server implementation."""

    def __init__(self, api_key: str = "demo"):
        """Initialize AlphaVantage MCP server.

        Args:
            api_key: AlphaVantage API key (defaults to 'demo')
        """
        super().__init__()
        self.api_key = api_key

    async def init_server(self) -> StdioServerParameters:
        """Initialize AlphaVantage server configuration.

        Returns:
            StdioServerParameters for AlphaVantage server
        """
        return StdioServerParameters(
            command="python",
            args=[
                "servers/python/mcp_alphavantage/server.py",
                "--api-key",
                self.api_key,
            ],
            env={"ALPHAVANTAGE_API_KEY": self.api_key},
        )
