import os

from dotenv import load_dotenv
from mcp import StdioServerParameters

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
