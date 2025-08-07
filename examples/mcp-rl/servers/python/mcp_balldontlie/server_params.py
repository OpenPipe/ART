from mcp import StdioServerParameters
import os
from dotenv import load_dotenv

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=[
        "servers/python/mcp_balldontlie/server.py",
        "--api-key",
        os.getenv("BALLDONTLIE_API_KEY", ""),
    ],
    env={"BALLDONTLIE_API_KEY": os.getenv("BALLDONTLIE_API_KEY")},
)
