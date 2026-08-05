#!/usr/bin/env python3
"""Convert the cached MCP-Atlas parquet into a local tasks CSV for run_eval.py.

The parquet (ScaleAI/MCP-Atlas, 500 rows) has columns:
    TASK, ENABLED_TOOLS, PROMPT, GTFA_CLAIMS, TRAJECTORY
run_eval.py --input needs TASK/PROMPT/ENABLED_TOOLS; scoring --groundtruth-file
needs TASK/PROMPT/GTFA_CLAIMS. We emit all columns so one CSV serves both.

Task subsetting:
  --only-default-servers   keep tasks whose *available* tools (ENABLED_TOOLS) are all
                           served by the 20 no-API-key servers (smoke, no keys).
  --exclude-servers a,b,c  drop tasks whose *gold trajectory* (TRAJECTORY) uses any of
                           the named servers. (ENABLED_TOOLS contains distractor tools,
                           so filtering must use the gold trajectory, not ENABLED_TOOLS.)
  --envfactory-subset      EnvFactory paper Appendix F setup: exclude the 6 servers
                           {mongodb, oxylabs, brave-search, wikipedia, slack,
                           google-workspace}. Yields exactly 291/500 tasks. Pair with
                           ENABLED_SERVERS = the other 30 servers in mcp-atlas/.env
                           (see setup_env.sh MCP_SERVERS_MODE=envfactory).

Usage:
    python prepare_tasks.py --out tasks.csv [--num N] [--only-default-servers]
    python prepare_tasks.py --out tasks.csv --envfactory-subset
    python prepare_tasks.py --out tasks.csv --exclude-servers mongodb,slack
"""
import argparse
import json

import pandas as pd

PARQUET = "/data1/lvnuoyan/dataset/agent/mcp-atlas/hf_dataset/MCP-Atlas.parquet"

# The 20 servers enabled by default (no API key) -- from mcp-atlas env.template.
DEFAULT_SERVERS = [
    "arxiv", "calculator", "cli-mcp-server", "clinicaltrialsgov-mcp-server",
    "context7", "ddg-search", "desktop-commander", "fetch", "filesystem", "git",
    "mcp-code-executor", "mcp-server-code-runner", "memory", "met-museum",
    "open-library", "osm-mcp-server", "pubmed", "weather", "whois", "wikipedia",
]

# All 36 MCP-Atlas servers (mcp_server_template.json).
ALL_SERVERS = [
    "airtable", "alchemy", "arxiv", "brave-search", "calculator", "cli-mcp-server",
    "clinicaltrialsgov-mcp-server", "context7", "ddg-search", "desktop-commander",
    "e2b-server", "exa", "fetch", "filesystem", "git", "github", "google-maps",
    "google-workspace", "lara-translate", "mcp-code-executor", "mcp-server-code-runner",
    "memory", "met-museum", "mongodb", "national-parks", "notion", "open-library",
    "osm-mcp-server", "oxylabs", "pubmed", "slack", "twelvedata", "weather",
    "weather-data", "whois", "wikipedia",
]

# EnvFactory paper (Appendix F) excludes these 6 -> 30 servers, 291 tasks.
ENVFACTORY_EXCLUDED = ["mongodb", "oxylabs", "brave-search", "wikipedia", "slack",
                       "google-workspace"]


def _belongs(tool, server) -> bool:
    return isinstance(tool, str) and (tool == server or tool.startswith(server + "_"))


def tool_is_default(tool) -> bool:
    return any(_belongs(tool, s) for s in DEFAULT_SERVERS)


def task_needs_only_default(enabled_tools_json: str) -> bool:
    try:
        tools = json.loads(enabled_tools_json)
    except Exception:
        return False
    return bool(tools) and all(tool_is_default(t) for t in tools)


def _trajectory_tools(trajectory_json: str):
    """Tool names actually used in the gold trajectory."""
    names = set()
    try:
        arr = json.loads(trajectory_json)
    except Exception:
        return names
    for msg in arr:
        for tc in (msg.get("tool_calls") or []):
            fn = (tc.get("function") or {}).get("name")
            if fn:
                names.add(fn)
    return names


def traj_uses_any(trajectory_json: str, servers) -> bool:
    tools = _trajectory_tools(trajectory_json)
    return any(_belongs(t, s) for t in tools for s in servers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=None, help="keep only first N rows")
    ap.add_argument("--only-default-servers", action="store_true",
                    help="keep only tasks runnable with the 20 no-key servers")
    ap.add_argument("--exclude-servers", type=str, default=None,
                    help="comma-separated servers; drop tasks whose gold trajectory uses any")
    ap.add_argument("--envfactory-subset", action="store_true",
                    help="EnvFactory paper subset (exclude 6 servers -> 291 tasks)")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    total = len(df)

    excluded = set()
    if args.envfactory_subset:
        excluded |= set(ENVFACTORY_EXCLUDED)
    if args.exclude_servers:
        excluded |= {s.strip() for s in args.exclude_servers.split(",") if s.strip()}
    excluded = sorted(excluded) or None

    if args.only_default_servers:
        df = df[df["ENABLED_TOOLS"].apply(task_needs_only_default)].reset_index(drop=True)
    if excluded:
        df = df[~df["TRAJECTORY"].apply(lambda tj: traj_uses_any(tj, excluded))].reset_index(drop=True)
    if args.num is not None:
        df = df.head(args.num)

    df.to_csv(args.out, index=False)
    print(f"[prepare_tasks] wrote {len(df)}/{total} tasks -> {args.out} "
          f"(only_default={args.only_default_servers}, "
          f"excluded={excluded}, num={args.num})")


if __name__ == "__main__":
    main()
