"""Simple MCP server for formatting scenario data into JSON via stdio."""

import json
import sys
from typing import Any, Dict


def send_response(id: Any, result: Dict[str, Any]) -> None:
    """Send JSON-RPC response."""
    response = {"jsonrpc": "2.0", "id": id, "result": result}
    print(json.dumps(response), flush=True)


def send_error(id: Any, code: int, message: str) -> None:
    """Send JSON-RPC error."""
    response = {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
    print(json.dumps(response), flush=True)


def handle_request(request: Dict[str, Any]) -> None:
    """Handle MCP protocol request."""
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        send_response(
            req_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "scenario-formatter", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        )

    elif method == "notifications/initialized":
        # Client acknowledges initialization - no response needed
        pass

    elif method == "tools/list":
        send_response(
            req_id,
            {
                "tools": [
                    {
                        "name": "format_scenario",
                        "description": "Format a scenario into proper JSON structure",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "The task description"},
                                "difficulty": {
                                    "type": "integer",
                                    "description": "Difficulty rating from 1-5",
                                },
                            },
                            "required": ["task", "difficulty"],
                        },
                    }
                ]
            },
        )

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "format_scenario":
            # Format and validate the scenario
            formatted = {
                "task": str(args.get("task", "")).strip(),
                "difficulty": max(1, min(5, int(args.get("difficulty", 3)))),
            }

            send_response(
                req_id,
                {"content": [{"type": "text", "text": json.dumps(formatted, indent=2)}]},
            )
        else:
            send_error(req_id, -32601, f"Unknown tool: {tool_name}")

    elif method and method.startswith("notifications/"):
        # Handle other notifications silently
        pass

    else:
        if req_id:  # Only send error if there's an ID to respond to
            send_error(req_id, -32601, f"Unknown method: {method}")


def main():
    """Main server loop."""
    buffer = ""
    for line in sys.stdin:
        buffer += line
        try:
            request = json.loads(buffer)
            buffer = ""
            handle_request(request)
        except json.JSONDecodeError:
            # Not complete JSON yet, keep buffering
            continue


if __name__ == "__main__":
    main()
