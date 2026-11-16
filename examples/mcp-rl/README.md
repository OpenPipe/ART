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

The HTTP connection uses the MCP 1.x SDK (`mcp>=1.11,<2`), declared in the script.
Use that SDK version in your rollout environment too.

To use the same connection in the existing agent rollout, supply the opt-in
parameters instead of `StdioServerParameters`:

```python
from mcp_rl.rollout import McpScenario, rollout
from mcp_transport import parallel_search_params

scenario = McpScenario(
    task_description="Find ART's MCP training documentation and summarize the prerequisites.",
    server_params=parallel_search_params,
    max_tool_result_chars=40000,
)
trajectory = await rollout(model, scenario)
```

Use your configured ART model for `model`. The optional response-size limit
allows room for Parallel's excerpts and JSON formatting; existing scenarios keep
their 20,000-character default. The existing rollout uses model
inference credentials and an OpenAI key for its completion judge; these are
separate from the anonymous MCP connection. This does not change the existing
stdio scenarios, server choices, or training CLI defaults.

Other standard Streamable HTTP servers can use
`StreamableHttpServerParameters(url=..., headers=...)`, with optional `timeout`
and `sse_read_timeout` settings. The older `RemoteMcpServer` wrapper is for a
custom REST API, not standard MCP HTTP endpoints.
