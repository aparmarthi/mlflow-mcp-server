from __future__ import annotations

import os
from typing import Any

import mlflow
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("mlflow")


def _client(tracking_uri: str | None) -> mlflow.MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    return mlflow.MlflowClient(tracking_uri=uri)


@mcp.tool()
def list_experiments(tracking_uri: str | None = None) -> list[dict[str, Any]]:
    """List all MLflow experiments."""
    try:
        client = _client(tracking_uri)
        exps = client.search_experiments()
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "artifact_location": e.artifact_location,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in exps
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def get_run(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Get full details of a single MLflow run by run_id."""
    try:
        client = _client(tracking_uri)
        run = client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
    except Exception as exc:
        return {"error": str(exc), "run_id": run_id}


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
