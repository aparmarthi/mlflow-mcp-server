# PRD: MLflow MCP Server

## Problem

ML engineers track experiments in MLflow but context-switch constantly to query them — opening the UI, filtering runs, copying metrics into chat, then asking for analysis. Each query is a 3-step interruption: navigate, copy, ask. Over a 12-run hyperparameter sweep, this overhead compounds and breaks flow.

**Who:** ML engineers and AI practitioners who use MLflow for experiment tracking and LLM assistants (Claude) for analysis and decision support.

**Why now:** The Model Context Protocol (MCP) standardizes how AI assistants connect to external tools. MLflow has no official MCP server. This fills that gap with a minimal, portable implementation.

## Solution

A FastMCP server that gives Claude direct read+write access to MLflow. The engineer stays in the conversation — no context switching to the UI.

**Happy path:** "Which of my LightGBM runs had the best PR-AUC at threshold 0.4, and what were its hyperparameters?" → Claude queries MLflow and answers directly.

## Success Metrics

| Metric | Target |
|---|---|
| North Star | Experiment queries answered without leaving the chat |
| Tool response latency | < 500ms for local MLflow |
| Smoke test pass rate | 100% (all 6 tools) |
| Zero context-switch queries | Any run comparison, hyperparameter lookup, or champion tagging done in chat |

## Tools

| Tool | Type | Purpose |
|---|---|---|
| `list_experiments` | Read | Discover what's being tracked |
| `get_run` | Read | Full detail on a single run |
| `search_runs` | Read | Filter by metrics, params, tags across experiments |
| `create_experiment` | Write | Start a new experiment from chat |
| `log_run` | Write | Record results after the fact |
| `set_run_tag` | Write | Mark champion/deprecated/candidate runs |

## Trade-offs

**Optimized for:** Simplicity and portability. Single file, no database, no auth layer.

**Sacrificed:** Security (no API key protection — must be added before public hosting), streaming artifact access, run deletion.

**Transport choice:** Supports both stdio (Claude Desktop) and SSE (Docker, hosted). Stdio is the default since Claude Desktop is the primary interface; SSE is opt-in via `MCP_TRANSPORT=sse`.

**Why FastMCP over raw MCP SDK:** FastMCP reduces protocol boilerplate to zero — tools are plain Python functions. The tradeoff is a dependency on a third-party library, acceptable here because the MCP protocol itself is stable.

## Scaling Path

| Scale | Approach |
|---|---|
| 1 user, local | `python server.py` + stdio |
| Team, shared tracking server | Docker + `MLFLOW_TRACKING_URI` pointing at shared MLflow |
| Hosted service | Deploy to Render/Railway, add API key auth, expose SSE endpoint |

## Out of Scope (v1)

- Authentication / API key protection
- MLflow Model Registry (registered models, stage transitions)
- Artifact download or streaming
- Run deletion
