"""Unit tests for MCP-Atlas prepare_tasks filtering + parquet conversion.

Run in the `factory` env:
    cd /home/lvnuoyan/EnvFactory/scripts/eval/mcp_atlas
    python -m pytest tests/test_prepare_tasks.py -v
"""
import importlib.util
import os

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "prepare_tasks", os.path.join(HERE, "prepare_tasks.py")
)
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)


def test_tool_is_default():
    assert pt.tool_is_default("filesystem_read_text_file")
    assert pt.tool_is_default("ddg-search_search")            # hyphenated server
    assert pt.tool_is_default("mcp-code-executor_execute_code")
    assert not pt.tool_is_default("github_search_repositories")  # key-gated
    assert not pt.tool_is_default("slack_post_message")
    assert not pt.tool_is_default(123)                        # non-str is safe


def test_task_needs_only_default():
    import json
    ok = json.dumps(["filesystem_read_text_file", "calculator_add"])
    bad = json.dumps(["filesystem_read_text_file", "github_get_repository"])
    assert pt.task_needs_only_default(ok)
    assert not pt.task_needs_only_default(bad)
    assert not pt.task_needs_only_default("[]")               # empty -> False
    assert not pt.task_needs_only_default("not json")


@pytest.mark.skipif(not os.path.exists(pt.PARQUET), reason="parquet not cached")
def test_prepare_writes_subset(tmp_path):
    out = tmp_path / "tasks.csv"
    import sys
    argv = sys.argv
    sys.argv = ["prepare_tasks.py", "--out", str(out), "--num", "2",
                "--only-default-servers"]
    try:
        pt.main()
    finally:
        sys.argv = argv
    df = pd.read_csv(out)
    assert len(df) <= 2
    # every kept task uses only default servers
    assert df["ENABLED_TOOLS"].apply(pt.task_needs_only_default).all()
    # columns needed by run_eval (--input) AND scoring (--groundtruth-file)
    for col in ["TASK", "PROMPT", "ENABLED_TOOLS", "GTFA_CLAIMS"]:
        assert col in df.columns


# --- EnvFactory paper subset (Appendix F: 30 servers / 291 tasks) ---
def test_envfactory_excluded_list():
    assert set(pt.ENVFACTORY_EXCLUDED) == {
        "mongodb", "oxylabs", "brave-search", "wikipedia", "slack", "google-workspace"
    }
    # 36 total servers, 6 excluded -> 30 enabled
    assert len(pt.ALL_SERVERS) == 36
    enabled = [s for s in pt.ALL_SERVERS if s not in pt.ENVFACTORY_EXCLUDED]
    assert len(enabled) == 30


@pytest.mark.skipif(not os.path.exists(pt.PARQUET), reason="parquet not cached")
def test_envfactory_subset_is_291(tmp_path):
    out = tmp_path / "subset.csv"
    import sys
    argv = sys.argv
    sys.argv = ["prepare_tasks.py", "--out", str(out), "--envfactory-subset"]
    try:
        pt.main()
    finally:
        sys.argv = argv
    df = pd.read_csv(out)
    # Paper Appendix F: 291 of 500 tasks.
    assert len(df) == 291
    # No kept task's gold trajectory uses an excluded server.
    assert not df["TRAJECTORY"].apply(
        lambda tj: pt.traj_uses_any(tj, pt.ENVFACTORY_EXCLUDED)
    ).any()


def test_traj_uses_any_prefix_matching():
    import json
    tj = json.dumps([{"tool_calls": [{"function": {"name": "slack_post_message"}}]}])
    assert pt.traj_uses_any(tj, ["slack"])
    assert not pt.traj_uses_any(tj, ["github"])
    # ENABLED_TOOLS filtering must NOT be used for the subset (distractors inflate it);
    # only the gold trajectory counts.
