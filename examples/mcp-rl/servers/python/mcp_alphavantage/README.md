# MCP AlphaVantage Python Server

A Python implementation of the MCP server for Alpha Vantage financial data API.

## Features

- Real-time stock quotes
- Daily time series data
- Symbol search
- Company overview/fundamentals
- Technical indicators (SMA, RSI)

## Setup

1. Get an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Set the environment variable:
   ```bash
   export ALPHAVANTAGE_API_KEY=your_api_key_here
   ```

## Usage

### Command Line
```bash
python server.py --api-key YOUR_API_KEY
```

### With Environment Variable
```bash
export ALPHAVANTAGE_API_KEY=your_api_key
python server.py
```

### Available Tools

- `get_stock_quote`: Get real-time stock quote
- `get_time_series_daily`: Get daily stock data
- `search_symbol`: Search for stock symbols
- `get_company_overview`: Get company fundamentals
- `get_sma`: Simple Moving Average indicator
- `get_rsi`: Relative Strength Index indicator

## Transport Options

- `stdio` (default): Standard input/output transport
- `sse`: Server-sent events over HTTP