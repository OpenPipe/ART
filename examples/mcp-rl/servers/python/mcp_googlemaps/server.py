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


class GoogleMapsClient:
    """Client for interacting with Google Maps APIs"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.geocoding_base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.places_base_url = "https://maps.googleapis.com/maps/api/place"

    async def geocode(self, address: str, **params) -> Dict[str, Any]:
        """Geocode an address to coordinates"""
        query_params = {
            "address": address,
            "key": self.api_key,
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.geocoding_base_url, params=query_params
            ) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") != "OK":
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data

    async def reverse_geocode(self, lat: float, lng: float, **params) -> Dict[str, Any]:
        """Reverse geocode coordinates to address"""
        query_params = {
            "latlng": f"{lat},{lng}",
            "key": self.api_key,
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.geocoding_base_url, params=query_params
            ) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") != "OK":
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data

    async def places_nearby_search(
        self, location: str, radius: int, place_type: Optional[str] = None, **params
    ) -> Dict[str, Any]:
        """Search for nearby places"""
        url = f"{self.places_base_url}/nearbysearch/json"
        query_params = {
            "location": location,
            "radius": radius,
            "key": self.api_key,
            **params,
        }

        if place_type:
            query_params["type"] = place_type

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=query_params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data

    async def places_text_search(self, query: str, **params) -> Dict[str, Any]:
        """Search for places by text query"""
        url = f"{self.places_base_url}/textsearch/json"
        query_params = {
            "query": query,
            "key": self.api_key,
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=query_params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data

    async def place_details(self, place_id: str, **params) -> Dict[str, Any]:
        """Get details for a specific place"""
        url = f"{self.places_base_url}/details/json"
        query_params = {
            "place_id": place_id,
            "key": self.api_key,
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=query_params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") != "OK":
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data

    async def place_autocomplete(self, input_text: str, **params) -> Dict[str, Any]:
        """Get place autocomplete predictions"""
        url = f"{self.places_base_url}/autocomplete/json"
        query_params = {
            "input": input_text,
            "key": self.api_key,
            **params,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=query_params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                data = await response.json()

                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    raise Exception(
                        f"Google Maps API Error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"
                    )

                return data


@click.command()
@click.option("--api-key", help="Google Maps API key", envvar="GOOGLE_MAPS_API_KEY")
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
            "Error: Google Maps API key is required. Set GOOGLE_MAPS_API_KEY environment variable or use --api-key option."
        )
        return 1

    app = Server("mcp-googlemaps")
    client = GoogleMapsClient(api_key)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="geocode",
                description="Convert an address to geographic coordinates (latitude/longitude)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "The address to geocode (e.g., '1600 Amphitheatre Parkway, Mountain View, CA')",
                        },
                        "region": {
                            "type": "string",
                            "description": "Optional region code to bias results (e.g., 'us', 'uk')",
                        },
                        "bounds": {
                            "type": "string",
                            "description": "Optional bounding box to bias results (southwest_lat,southwest_lng|northeast_lat,northeast_lng)",
                        },
                    },
                    "required": ["address"],
                },
            ),
            types.Tool(
                name="reverse_geocode",
                description="Convert geographic coordinates to an address",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude coordinate",
                        },
                        "lng": {
                            "type": "number",
                            "description": "Longitude coordinate",
                        },
                        "result_type": {
                            "type": "string",
                            "description": "Optional filter for result types (e.g., 'street_address', 'postal_code')",
                        },
                    },
                    "required": ["lat", "lng"],
                },
            ),
            types.Tool(
                name="places_nearby_search",
                description="Search for places near a location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location as 'latitude,longitude' (e.g., '37.4219,-122.0841')",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Search radius in meters (max 50000)",
                            "maximum": 50000,
                        },
                        "type": {
                            "type": "string",
                            "description": "Place type to search for (e.g., 'restaurant', 'gas_station', 'hospital')",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to match against place names and types",
                        },
                    },
                    "required": ["location", "radius"],
                },
            ),
            types.Tool(
                name="places_text_search",
                description="Search for places using a text query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text search query (e.g., 'pizza near Times Square')",
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location bias as 'latitude,longitude'",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Optional search radius in meters when location is provided",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="place_details",
                description="Get detailed information about a specific place",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "place_id": {
                            "type": "string",
                            "description": "Place ID from a search result",
                        },
                        "fields": {
                            "type": "string",
                            "description": "Comma-separated list of fields to return (e.g., 'name,rating,formatted_phone_number')",
                        },
                    },
                    "required": ["place_id"],
                },
            ),
            types.Tool(
                name="place_autocomplete",
                description="Get place predictions for autocomplete functionality",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input text for autocomplete predictions",
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location bias as 'latitude,longitude'",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Optional search radius in meters when location is provided",
                        },
                        "types": {
                            "type": "string",
                            "description": "Optional type restrictions (e.g., 'establishment', 'geocode')",
                        },
                    },
                    "required": ["input"],
                },
            ),
        ]

    @app.call_tool()
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, Exception)
        ),
    )
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "geocode":
                address = arguments["address"]
                params = {}
                if "region" in arguments:
                    params["region"] = arguments["region"]
                if "bounds" in arguments:
                    params["bounds"] = arguments["bounds"]

                data = await client.geocode(address, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Geocoding results for '{address}':\n{_format_json(data)}",
                    )
                ]

            elif name == "reverse_geocode":
                lat = arguments["lat"]
                lng = arguments["lng"]
                params = {}
                if "result_type" in arguments:
                    params["result_type"] = arguments["result_type"]

                data = await client.reverse_geocode(lat, lng, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Reverse geocoding results for ({lat}, {lng}):\n{_format_json(data)}",
                    )
                ]

            elif name == "places_nearby_search":
                location = arguments["location"]
                radius = arguments["radius"]
                params = {}
                if "type" in arguments:
                    params["type"] = arguments["type"]
                if "keyword" in arguments:
                    params["keyword"] = arguments["keyword"]

                data = await client.places_nearby_search(location, radius, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Nearby places search results:\n{_format_json(data)}",
                    )
                ]

            elif name == "places_text_search":
                query = arguments["query"]
                params = {}
                if "location" in arguments:
                    params["location"] = arguments["location"]
                if "radius" in arguments:
                    params["radius"] = arguments["radius"]

                data = await client.places_text_search(query, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Text search results for '{query}':\n{_format_json(data)}",
                    )
                ]

            elif name == "place_details":
                place_id = arguments["place_id"]
                params = {}
                if "fields" in arguments:
                    params["fields"] = arguments["fields"]

                data = await client.place_details(place_id, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Place details for {place_id}:\n{_format_json(data)}",
                    )
                ]

            elif name == "place_autocomplete":
                input_text = arguments["input"]
                params = {}
                if "location" in arguments:
                    params["location"] = arguments["location"]
                if "radius" in arguments:
                    params["radius"] = arguments["radius"]
                if "types" in arguments:
                    params["types"] = arguments["types"]

                data = await client.place_autocomplete(input_text, **params)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Autocomplete predictions for '{input_text}':\n{_format_json(data)}",
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
