import os

from dotenv import load_dotenv
from mcp import StdioServerParameters

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=[
        "servers/python/mcp_googlemaps/server.py",
        "--api-key",
        os.getenv("GOOGLE_MAPS_API_KEY", ""),
    ],
    env={"GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY")},
)
