# Balldontlie NBA Stats MCP Server

An MCP server that provides access to NBA statistics through the Balldontlie API, focused on free tier endpoints.

## Features (Free Tier Only)

- **Teams**: Get NBA team information with conference/division filters
- **Players**: Search and filter NBA players with comprehensive parameters
- **Games**: Retrieve game data with date/season/team filters and date ranges

## Setup

1. Get a free API key from [Balldontlie](https://www.balldontlie.io/)
2. Set your API key as an environment variable:
   ```bash
   export BALLDONTLIE_API_KEY=your_api_key_here
   ```

## Available Tools (Free Tier)

### get_teams
Get all NBA teams or a specific team by ID with optional conference/division filters.
- Parameters: `team_id`, `division`, `conference`

### get_players  
Search NBA players with comprehensive filtering options.
- Parameters: `player_id`, `search`, `first_name`, `last_name`, `team_ids`, `player_ids`, `cursor`, `per_page`

### get_games
Retrieve NBA games with extensive filtering capabilities.
- Parameters: `game_id`, `dates`, `seasons`, `team_ids`, `postseason`, `start_date`, `end_date`, `cursor`, `per_page`

## Free Tier Limitations

- Rate limit: 5 requests per minute
- Access only to basic teams, players, and games endpoints
- No access to advanced statistics, box scores, player injuries, or other premium features

## Data Coverage

- Historical data from 1979 to present
- Basic team, player, and game information
- Regular season and postseason game data