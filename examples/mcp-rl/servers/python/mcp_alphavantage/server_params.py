import os

from dotenv import load_dotenv
from mcp import StdioServerParameters

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=[
        "servers/python/mcp_alphavantage/server.py",
        "--api-key",
        os.getenv("ALPHAVANTAGE_API_KEY", "demo"),
    ],
    env={"ALPHAVANTAGE_API_KEY": os.getenv("ALPHAVANTAGE_API_KEY")},
)
