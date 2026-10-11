# MCP-RL examples

## Parallel web search

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search and page fetching without an account or Parallel API key.
Anonymous search uses Fast mode and has lower rate limits, so start with small
batches rather than a large training run.

From this directory, run the connection example with Python 3.12 and `uv`:

```bash
uv run parallel_search.py --query "ART agent reinforcement trainer" --url https://art.openpipe.ai
```

The script installs its declared client dependencies, discovers the server's
schemas as ART `MCPTool` objects, and prints real search and fetch results. It does
not run model inference or training.

To use the same connection in the existing agent rollout, supply the opt-in
parameters instead of `StdioServerParameters`:

```python
from mcp_rl.rollout import McpScenario, rollout
from mcp_rl.transport import parallel_search_params

scenario = McpScenario(
    task_description="Find ART's MCP training documentation and summarize the prerequisites.",
    server_params=parallel_search_params,
)
trajectory = await rollout(model, scenario)
```

Use your configured ART model for `model`. The existing rollout uses model
inference credentials and an OpenAI key for its completion judge; these are
separate from the anonymous MCP connection. This does not change the existing
stdio scenarios, server choices, or training CLI defaults.

Other standard Streamable HTTP servers can use
`StreamableHttpServerParameters(url=..., headers=...)`, with optional `timeout`
and `sse_read_timeout` settings. The older `RemoteMcpServer` wrapper is for a
custom REST API, not standard MCP HTTP endpoints.
