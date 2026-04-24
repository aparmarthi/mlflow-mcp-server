# mlflow-mcp-server

MCP server that gives Claude read+write access to MLflow experiment tracking via SSE transport.

## What it does

Six tools:
- `list_experiments` — list all experiments
- `get_run` — full detail on a single run (params, metrics, tags)
- `search_runs` — filter runs by metric thresholds, status, params
- `create_experiment` — create a new experiment
- `log_run` — log a completed run with params, metrics, tags in one call
- `set_run_tag` — tag a run (e.g. mark as "champion")

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # set MLFLOW_TRACKING_URI
python server.py        # listens on localhost:8080
```

## Connect Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mlflow": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

Restart Claude Desktop. Ask: *"List my MLflow experiments."*

## Docker

```bash
cp .env.example .env   # set MLFLOW_TRACKING_URI
docker compose up
```

## Deploy to Render / Railway

1. Push this repo to GitHub
2. Create a new Web Service pointing at the repo
3. Set `MLFLOW_TRACKING_URI` env var to your remote tracking server URL
4. Add the public `/sse` URL to your Claude Desktop config

## Smoke test

```bash
python scripts/smoke_test.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking server URI |
| `MCP_HOST` | `0.0.0.0` | Server bind host |
| `MCP_PORT` | `8080` | Server port |
