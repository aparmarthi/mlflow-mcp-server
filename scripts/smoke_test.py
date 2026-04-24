"""
Smoke test: seeds a local mlruns fixture and calls all six tools directly.
Run from repo root: python scripts/smoke_test.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURE_URI = "/tmp/mlflow_smoke_fixture"

# Clean up any leftover fixture from a previous run so create_experiment
# doesn't collide with an already-existing "smoke_new_exp".
if os.path.exists(FIXTURE_URI):
    shutil.rmtree(FIXTURE_URI)

# Seed fixture with one experiment and two runs
import mlflow

mlflow.set_tracking_uri(FIXTURE_URI)
mlflow.set_experiment("smoke_test_experiment")

with mlflow.start_run(run_name="run_a") as r1:
    mlflow.log_params({"model_type": "lgbm", "n_estimators": "100"})
    mlflow.log_metrics({"roc_auc": 0.91, "pr_auc": 0.85})

with mlflow.start_run(run_name="run_b") as r2:
    mlflow.log_params({"model_type": "xgb", "n_estimators": "200"})
    mlflow.log_metrics({"roc_auc": 0.93, "pr_auc": 0.88})

run_a_id = r1.info.run_id
run_b_id = r2.info.run_id

# Import tools
from server import list_experiments, get_run, search_runs, create_experiment, log_run, set_run_tag

TRACKING = FIXTURE_URI

print("--- list_experiments ---")
result = list_experiments(tracking_uri=TRACKING)
assert len(result) >= 1 and "error" not in result[0], f"FAIL: {result}"
print(json.dumps(result, indent=2))

print("--- get_run ---")
result = get_run(run_id=run_a_id, tracking_uri=TRACKING)
assert "error" not in result and result["run_name"] == "run_a", f"FAIL: {result}"
print(json.dumps(result, indent=2))

print("--- search_runs ---")
result = search_runs(
    experiment_names=["smoke_test_experiment"],
    filter_string="metrics.roc_auc > 0.90",
    order_by=["metrics.roc_auc DESC"],
    tracking_uri=TRACKING,
)
assert len(result) >= 1 and "error" not in result[0], f"FAIL: {result}"
print(json.dumps(result, indent=2))

print("--- create_experiment ---")
result = create_experiment(name="smoke_new_exp", tracking_uri=TRACKING)
assert "experiment_id" in result, f"FAIL: {result}"
print(json.dumps(result, indent=2))

print("--- log_run ---")
result = log_run(
    experiment_name="smoke_test_experiment",
    run_name="run_c",
    params={"model_type": "rf"},
    metrics={"roc_auc": 0.89, "pr_auc": 0.80},
    tags={"stage": "candidate"},
    tracking_uri=TRACKING,
)
assert "run_id" in result, f"FAIL: {result}"
run_c_id = result["run_id"]
print(json.dumps(result, indent=2))

print("--- set_run_tag ---")
result = set_run_tag(run_id=run_c_id, key="stage", value="champion", tracking_uri=TRACKING)
assert "error" not in result, f"FAIL: {result}"
print(json.dumps(result, indent=2))

print("\nAll smoke tests passed.")
