import json

import pytest

from qwen_agentworld.tools.graph_map import graph_complexity_for, graph_nodes_for, known_models


def _write_map(tmp_path, data):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


BASIC = {
    "default": {"min_nodes": 3, "max_nodes": 3},
    "models": {
        "Qwen3-8B": {"min_nodes": 4, "max_nodes": 5, "families": {"mcp_notes": {"min_nodes": 6, "max_nodes": 6}}},
    },
}


def test_shipped_map_has_the_measured_qwen3_row():
    # The 2026-08-07 screening is the reason gc moved off 3; a silent revert to
    # the default would restore the 9%-band-yield regime without anything failing.
    assert graph_nodes_for("Qwen3-8B") == (4, 4)
    assert "Qwen3-8B" in known_models()


def test_family_override_beats_model_beats_default(tmp_path):
    path = _write_map(tmp_path, BASIC)
    assert graph_nodes_for("Qwen3-8B", "mcp_notes", path) == (6, 6)
    assert graph_nodes_for("Qwen3-8B", "mcp_terminal", path) == (4, 5)
    assert graph_nodes_for("some-other-model", "mcp_notes", path) == (3, 3)
    assert graph_nodes_for(None, None, path) == (3, 3)


def test_model_id_matches_by_basename_and_case(tmp_path):
    path = _write_map(tmp_path, BASIC)
    assert graph_nodes_for("/root/autodl-tmp/models/Qwen3-8B", None, path) == (4, 5)
    assert graph_nodes_for("qwen3-8b", None, path) == (4, 5)


def test_unknown_model_warns_rather_than_silently_defaulting(tmp_path, caplog):
    path = _write_map(tmp_path, BASIC)
    with caplog.at_level("WARNING"):
        graph_nodes_for("brand-new-model", None, path)
    assert "brand-new-model" in caplog.text


def test_graph_complexity_takes_the_low_end(tmp_path):
    # A range straddling two buckets would split one generation run across two
    # pools that then get screened and compared as if they were one.
    path = _write_map(tmp_path, BASIC)
    assert graph_complexity_for("Qwen3-8B", "mcp_terminal", path) == 4


def test_invalid_range_raises(tmp_path):
    path = _write_map(tmp_path, {"default": {"min_nodes": 5, "max_nodes": 2}, "models": {}})
    with pytest.raises(ValueError):
        graph_nodes_for(None, None, path)
