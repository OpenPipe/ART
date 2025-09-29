from art.mcp.types import MCPTool

complete_task_tool = MCPTool(
    name="complete_task",
    description="Complete a task",
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of accomplishments",
            }
        },
        "required": ["summary"],
    },
)
