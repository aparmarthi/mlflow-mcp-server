# MLflow MCP Server

> Give Claude direct read+write access to your MLflow experiment history — no UI, no copy-paste, no context switching.

---

## The Problem

ML engineers constantly interrupt their workflow to answer questions like:

- *"Which of my last 12 runs had the best PR-AUC at threshold 0.4?"*
- *"What hyperparameters did my champion LightGBM use?"*
- *"Have I tried learning rates below 0.01?"*

Answering each question means opening the MLflow UI, filtering, copying results, pasting into chat, then asking. Three steps per query, dozens of times per experiment cycle.

## The Solution

An [MCP](https://modelcontextprotocol.io) server that connects Claude directly to MLflow. Ask in plain English — Claude queries your experiment history and answers in one shot.

```
You: "Compare my top 3 churn model runs on PR-AUC and precision@5000. Which should I deploy?"

Claude: [queries MLflow, returns ranked comparison with hyperparameters and recommendation]
```

---

## Tools

| Tool | Type | What it does |
|---|---|---|
| `list_experiments` | Read | List all tracked experiments |
| `get_run` | Read | Full detail on a run — params, metrics, tags, timing |
| `search_runs` | Read | Filter runs by metric thresholds, params, tags; sort by any metric |
| `create_experiment` | Write | Create a new experiment |
| `log_run` | Write | Record a completed run with params, metrics, and tags in one call |
| `set_run_tag` | Write | Tag a run (e.g. `stage=champion`, `stage=deprecated`) |

**Example queries:**
- *"Show me all runs where ROC-AUC > 0.95 and PR-AUC > 0.90, sorted by PR-AUC"*
- *"What hyperparameter ranges have I explored for LightGBM? What's missing?"*
- *"Tag run abc123 as champion"*
- *"Log a new run for my random forest with these results: ..."*

---

## Architecture

```
Claude Desktop
     │
     │  MCP (stdio)
     ▼
 server.py  ──►  mlflow Python client  ──►  ./mlruns (local)
 FastMCP                                     or remote tracking server
```

- **Transport:** stdio for Claude Desktop (default), SSE for Docker/hosted deployments
- **No intermediary:** the `mlflow` Python client talks directly to wherever `MLFLOW_TRACKING_URI` points
- **Stateless:** each tool call is independent; no session management needed

**Stack:** Python 3.11, [FastMCP](https://github.com/jlowin/fastmcp), MLflow 2.14+, python-dotenv

---

## Quick Start

**1. Install**
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**2. Connect Claude Desktop**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "/path/to/mlflow-mcp-server/.venv/bin/python",
      "args": ["/path/to/mlflow-mcp-server/server.py"],
      "env": {
        "MLFLOW_TRACKING_URI": "/path/to/your/mlruns"
      }
    }
  }
}
```

**3. Restart Claude Desktop** and ask: *"List my MLflow experiments."*

---

## Docker

```bash
cp .env.example .env   # set MLFLOW_TRACKING_URI
docker compose up
```

The container mounts `./mlruns` so experiment data persists across restarts.

---

## Deploy to Render / Railway (SSE mode)

1. Push this repo to GitHub
2. Create a new Web Service pointing at the repo
3. Set env vars: `MLFLOW_TRACKING_URI`, `MCP_TRANSPORT=sse`
4. Add the public `/sse` URL to your Claude Desktop config:

```json
{
  "mcpServers": {
    "mlflow": {
      "url": "https://your-app.onrender.com/sse"
    }
  }
}
```

---

## Smoke Test

Verifies all 6 tools against a local fixture without needing Claude:

```bash
python scripts/smoke_test.py
```

Expected output: `All smoke tests passed.`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking server URI — local path or `http://` remote |
| `MCP_TRANSPORT` | `stdio` | `stdio` for Claude Desktop, `sse` for Docker/hosted |
| `MCP_HOST` | `0.0.0.0` | Bind host (SSE mode only) |
| `MCP_PORT` | `8080` | Port (SSE mode only) |

---

## Design Decisions

**Why stdio + SSE dual transport?** Claude Desktop requires stdio (it spawns the process). Docker and hosted deployments need SSE (HTTP-based). Supporting both with a single env var keeps the codebase simple while covering all deployment targets.

**Why return errors as dicts instead of raising exceptions?** MCP tools surface errors to Claude as tool results. Returning `{"error": "..."}` lets Claude relay the problem to the user clearly rather than treating it as a silent tool failure.

**Why no auth in v1?** This server is designed for local and trusted-network use. Adding an API key middleware layer is straightforward and documented as the first v2 addition.

---

## Roadmap

- [ ] API key authentication for public hosting
- [ ] MLflow Model Registry tools (registered models, stage transitions)
- [ ] Artifact listing and download
- [ ] Run deletion with confirmation guard
