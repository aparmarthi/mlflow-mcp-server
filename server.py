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


@mcp.tool()
def search_runs(
    experiment_names: list[str],
    filter_string: str = "",
    order_by: list[str] | None = None,
    max_results: int = 20,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search runs across experiments.

    filter_string examples:
      "metrics.roc_auc > 0.95"
      "params.model_type = 'lgbm' and metrics.pr_auc > 0.90"
      "tags.stage = 'champion'"

    order_by examples:
      ["metrics.roc_auc DESC"]
      ["start_time DESC"]
    """
    try:
        client = _client(tracking_uri)
        exp_ids = []
        for name in experiment_names:
            exp = client.get_experiment_by_name(name)
            if exp:
                exp_ids.append(exp.experiment_id)
        if not exp_ids:
            return [{"error": f"No experiments found for names: {experiment_names}"}]

        runs = client.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results,
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "params": dict(r.data.params),
                "metrics": dict(r.data.metrics),
                "tags": dict(r.data.tags),
            }
            for r in runs
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def create_experiment(
    name: str,
    artifact_location: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Create a new MLflow experiment. Returns the new experiment_id."""
    try:
        client = _client(tracking_uri)
        experiment_id = client.create_experiment(
            name=name,
            artifact_location=artifact_location,
        )
        return {"experiment_id": experiment_id, "name": name}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def log_run(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """
    Create a completed MLflow run and log params, metrics, and tags in one call.

    Useful for logging results from a training script after the fact,
    or letting Claude record an experimental result.
    """
    try:
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            if metrics:
                numeric = {k: float(v) for k, v in metrics.items() if v is not None}
                if numeric:
                    mlflow.log_metrics(numeric)
            if tags:
                mlflow.set_tags(tags)

            return {
                "run_id": run.info.run_id,
                "experiment_name": experiment_name,
                "run_name": run_name,
            }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def set_run_tag(
    run_id: str,
    key: str,
    value: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Set a tag on an existing run. Useful for marking champion, deprecated, etc."""
    try:
        client = _client(tracking_uri)
        client.set_tag(run_id, key, value)
        return {"run_id": run_id, "key": key, "value": value}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
