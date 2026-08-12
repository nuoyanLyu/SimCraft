"""Tests for reading BFCL run artifacts.

The layout being assumed (score file = summary line + one line per *failure*)
is an observation about the harness, not a contract it promises. So the reader
cross-checks its own arithmetic against the accuracy BFCL printed, and these
tests pin that check: a misread file must fail loudly, because every per-entry
verdict derived from it would otherwise be fiction that still prints a p-value.
"""

import json

import pytest

from qwen_agentworld.study.bfcl import (
    BfclArtifactError,
    coverage_warning,
    load_arm,
    paired_entries,
)


def make_arm_dir(root, *, key="qwen3-8b-agentworld", category="simple_python",
                 ran_ids, failed_ids, accuracy=None, version="v4"):
    result_dir = root / "result" / key
    score_dir = root / "score" / key
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / f"BFCL_{version}_{category}_result.json").write_text(
        "\n".join(json.dumps({"id": i, "result": "..."}) for i in ran_ids)
    )
    if accuracy is None:
        accuracy = (len(ran_ids) - len(failed_ids)) / len(ran_ids)
    lines = [json.dumps({
        "accuracy": accuracy,
        "correct_count": len(ran_ids) - len(failed_ids),
        "total_count": len(ran_ids),
    })]
    lines += [json.dumps({"id": i, "error": ["wrong"]}) for i in failed_ids]
    (score_dir / f"BFCL_{version}_{category}_score.json").write_text("\n".join(lines))
    return root


# ---------------------------------------------------------------- read --


def test_per_entry_correctness_is_the_complement_of_the_failure_list(tmp_path):
    make_arm_dir(tmp_path, ran_ids=["a", "b", "c", "d"], failed_ids=["b", "d"])
    arm = load_arm(tmp_path, label="baseline", registry_key="qwen3-8b-agentworld",
                   category="simple_python")
    assert arm.passed_ids == {"a", "c"}
    assert arm.accuracy == pytest.approx(0.5)
    assert arm.passed("a") and not arm.passed("b")


def test_a_score_file_that_disagrees_with_its_own_summary_raises(tmp_path):
    """If the recomputed accuracy does not match the printed one, the assumed
    file layout does not hold for this harness version."""
    make_arm_dir(tmp_path, ran_ids=["a", "b", "c", "d"], failed_ids=["b"], accuracy=0.25)
    with pytest.raises(BfclArtifactError, match="disagrees"):
        load_arm(tmp_path, label="x", registry_key="qwen3-8b-agentworld", category="simple_python")


def test_a_missing_arm_directory_raises_rather_than_scoring_zero(tmp_path):
    with pytest.raises(BfclArtifactError, match="does not exist"):
        load_arm(tmp_path, label="x", registry_key="nope", category="simple_python")


def test_an_unevaluated_arm_raises_rather_than_reporting_no_failures(tmp_path):
    """An empty score file means `bfcl evaluate` did not run. Read naively that
    is an arm with zero failures, i.e. a perfect score."""
    make_arm_dir(tmp_path, ran_ids=["a"], failed_ids=[])
    key = "qwen3-8b-agentworld"
    (tmp_path / "score" / key / "BFCL_v4_simple_python_score.json").write_text("")
    with pytest.raises(BfclArtifactError, match="evaluate step did not run"):
        load_arm(tmp_path, label="x", registry_key=key, category="simple_python")


def test_an_ambiguous_category_name_raises(tmp_path):
    """'simple' matches simple_python and live_simple_python; silently
    averaging them would report a number nobody asked for."""
    make_arm_dir(tmp_path, ran_ids=["a"], failed_ids=[], category="simple_python")
    make_arm_dir(tmp_path, ran_ids=["b"], failed_ids=[], category="live_simple_python")
    with pytest.raises(BfclArtifactError, match="matched several"):
        load_arm(tmp_path, label="x", registry_key="qwen3-8b-agentworld", category="simple")


def test_the_dataset_version_in_the_filename_is_not_hardcoded(tmp_path):
    """A harness upgrade v4 -> v5 must not read as 'file missing'."""
    make_arm_dir(tmp_path, ran_ids=["a", "b"], failed_ids=["b"], version="v5")
    arm = load_arm(tmp_path, label="x", registry_key="qwen3-8b-agentworld",
                   category="simple_python")
    assert arm.accuracy == pytest.approx(0.5)


# -------------------------------------------------------------- paired --


def test_pairing_is_restricted_to_entries_both_arms_ran(tmp_path):
    """A generate step that dropped an entry in one arm only would otherwise
    contribute a phantom win to whichever arm still has it."""
    base_root, evolved_root = tmp_path / "base", tmp_path / "evolved"
    make_arm_dir(base_root, ran_ids=["a", "b", "c"], failed_ids=["c"])
    make_arm_dir(evolved_root, ran_ids=["a", "b"], failed_ids=[])
    key = "qwen3-8b-agentworld"
    base = load_arm(base_root, label="baseline", registry_key=key, category="simple_python")
    evolved = load_arm(evolved_root, label="evolved", registry_key=key, category="simple_python")

    paired = paired_entries(base, evolved)
    assert [eid for eid, _, _ in paired] == ["a", "b"]
    assert coverage_warning(base, evolved) is not None
    assert "3" in coverage_warning(base, evolved) and "2" in coverage_warning(base, evolved)


def test_matching_arms_produce_no_coverage_warning(tmp_path):
    base_root, evolved_root = tmp_path / "base", tmp_path / "evolved"
    make_arm_dir(base_root, ran_ids=["a", "b"], failed_ids=["b"])
    make_arm_dir(evolved_root, ran_ids=["a", "b"], failed_ids=[])
    key = "qwen3-8b-agentworld"
    base = load_arm(base_root, label="baseline", registry_key=key, category="simple_python")
    evolved = load_arm(evolved_root, label="evolved", registry_key=key, category="simple_python")
    assert coverage_warning(base, evolved) is None
    assert paired_entries(base, evolved) == [("a", True, True), ("b", False, True)]
