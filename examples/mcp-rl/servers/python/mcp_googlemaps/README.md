# MCP Google Maps Python Server

A Python implementation of the MCP server for Google Maps APIs, providing access to geocoding and places services.

## Features

- **Geocoding**: Convert addresses to coordinates and vice versa
- **Places Search**: Find nearby places and search by text
- **Place Details**: Get detailed information about specific places
- **Autocomplete**: Get place predictions for search functionality

## Free Tier Compatibility

This server is designed to work within Google Maps API free tier limits:
- 10,000 free API calls per month per API (as of 2025)
- Includes Geocoding API and Places API endpoints
- No billing required for basic usage under free limits

## Setup

1. Get a Google Maps API key:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the following APIs:
     - Geocoding API
     - Places API
   - Create credentials (API key)
   - Optionally restrict the API key to specific APIs and referrers

2. Set the environment variable:
   ```bash
   export GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

## Usage

### Command Line
```bash
python server.py --api-key YOUR_API_KEY
```

### With Environment Variable
```bash
export GOOGLE_MAPS_API_KEY=your_api_key
python server.py
```

### Available Tools

#### Geocoding
- `geocode`: Convert an address to latitude/longitude coordinates
- `reverse_geocode`: Convert coordinates to a human-readable address

#### Places
- `places_nearby_search`: Search for places near a specific location
- `places_text_search`: Search for places using a text query
- `place_details`: Get detailed information about a specific place
- `place_autocomplete`: Get place predictions for autocomplete functionality

## Transport Options

- `stdio` (default): Standard input/output transport
- `sse`: Server-sent events over HTTP

## Example Usage

### Geocoding an Address
```json
{
  "tool": "geocode",
  "arguments": {
    "address": "1600 Amphitheatre Parkway, Mountain View, CA"
  }
}
```

### Finding Nearby Restaurants
```json
{
  "tool": "places_nearby_search",
  "arguments": {
    "location": "37.4219,-122.0841",
    "radius": 1000,
    "type": "restaurant"
  }
}
```

### Text Search for Places
```json
{
  "tool": "places_text_search",
  "arguments": {
    "query": "pizza near Times Square New York"
  }
}
```

## API Rate Limits

To stay within free tier limits:
- Monitor your usage in Google Cloud Console
- Consider implementing client-side caching
- Use appropriate search radii to avoid excessive API calls
- Set up billing alerts to avoid unexpected charges

## Installation

```bash
pip install -e .
```

Or using uv:
```bash
uv pip install -e .
```