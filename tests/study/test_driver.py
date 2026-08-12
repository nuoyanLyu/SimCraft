"""Tests for the study driver's pure helpers.

The stages themselves need two vLLM servers and a teacher relay, so what is
testable here is the wiring: which stages run, which artifacts get read, and
whether the two BFCL arms are kept apart. That last one matters more than it
looks — the BFCL harness keys results by model name, so two arms sharing a run
root silently overwrite each other and the study ends up comparing an arm
against itself.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    """Import the driver by path: `scripts/` is not a package, and the module
    inserts the repo root on sys.path at import time."""
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "verify_playbook_effect", REPO_ROOT / "scripts" / "verify_playbook_effect.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def write_iteration(directory, n, *, before, after, task_ids=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"iteration_{n}.json").write_text(json.dumps({
        "iteration": n,
        "playbook_version_before": before,
        "playbook_version_after": after,
        "tasks": [{"task": {"task_id": t}} for t in task_ids],
        "playbook_after": {"version": after, "entries": []},
    }))


# --------------------------------------------------------------- stages --


def test_stages_accept_names_and_numbers_interchangeably():
    assert driver.parse_stages("2,3") == {"indomain", "bfcl"}
    assert driver.parse_stages("indomain,bfcl") == {"indomain", "bfcl"}
    assert driver.parse_stages("1,2,3,4") == set(driver.STAGES)


def test_stage_parsing_tolerates_whitespace_and_blanks():
    assert driver.parse_stages(" 4 , ") == {"report"}


def test_an_unknown_stage_raises_rather_than_silently_running_nothing():
    with pytest.raises(ValueError, match="unknown stage"):
        driver.parse_stages("evolveee")
    with pytest.raises(ValueError, match="out of range"):
        driver.parse_stages("9")


# -------------------------------------------------------------- evolve --


def test_playbook_edits_counts_only_iterations_that_changed_the_playbook(tmp_path):
    write_iteration(tmp_path, 1, before=1, after=2)
    write_iteration(tmp_path, 2, before=2, after=2)  # proposed nothing
    write_iteration(tmp_path, 3, before=2, after=4)
    assert driver.count_playbook_edits(tmp_path) == 2


def test_a_run_that_never_edited_the_playbook_counts_zero(tmp_path):
    """Feeds the precondition that voids the study; a silent 0 here would be
    reported as 'the playbook does not help'."""
    write_iteration(tmp_path, 1, before=1, after=1)
    write_iteration(tmp_path, 2, before=1, after=1)
    assert driver.count_playbook_edits(tmp_path) == 0


def test_train_task_ids_gathers_every_task_the_playbook_saw(tmp_path):
    write_iteration(tmp_path, 1, before=1, after=2, task_ids=["t1", "t2"])
    write_iteration(tmp_path, 2, before=2, after=3, task_ids=["t2", "t3"])
    assert driver.train_task_ids(tmp_path) == {"t1", "t2", "t3"}


def test_eval_task_ids_come_off_the_baseline_arm():
    results = {"checkpoints": {"baseline": {"per_task": {"e1": [True], "e2": [False]}}}}
    assert driver.eval_task_ids(results) == {"e1", "e2"}


# ----------------------------------------------------------- arm lookup --


def test_an_arm_is_found_under_a_collapsed_label():
    """`load_checkpoints` joins content-identical arms into 'mid+final', which
    an exact lookup would miss — and that is exactly the case worth reporting."""
    results = {"checkpoints": {"baseline": {}, "mid+final": {"n_entries": 3}}}
    label, payload = driver.pick_arm(results, prefer="final")
    assert label == "mid+final" and payload["n_entries"] == 3


def test_an_absent_arm_returns_none_rather_than_raising():
    assert driver.pick_arm({"checkpoints": {"baseline": {}}}, prefer="final") is None


def test_a_substring_match_does_not_count_as_an_arm():
    """'final' must not be found inside 'semifinal'."""
    assert driver.pick_arm({"checkpoints": {"semifinal": {}}}, prefer="final") is None


# ------------------------------------------------------------ bfcl env --


def test_each_bfcl_arm_gets_its_own_run_root(tmp_path):
    base = driver.bfcl_env(arm="baseline", run_root=tmp_path / "baseline",
                           playbook_file=None, base_env={})
    evolved = driver.bfcl_env(arm="evolved", run_root=tmp_path / "evolved",
                              playbook_file=tmp_path / "iteration_4.json", base_env={})
    assert base["BFCL_RUN_ROOT"] != evolved["BFCL_RUN_ROOT"]


def test_the_baseline_arm_carries_no_playbook():
    """Unset, not empty-string: run_bfcl.sh branches on `-n "${PLAYBOOK:-}"`,
    so an empty string is still the baseline but an inherited stale value from
    the caller's environment would not be."""
    env = driver.bfcl_env(arm="baseline", run_root=Path("/tmp/x"), playbook_file=None,
                          base_env={"PLAYBOOK": "leftover_from_a_previous_run.json"})
    assert "PLAYBOOK" not in env


def test_the_evolved_arm_points_at_the_checkpoint():
    env = driver.bfcl_env(arm="evolved", run_root=Path("/tmp/x"),
                          playbook_file=Path("/runs/iteration_6.json"), base_env={})
    assert env["PLAYBOOK"] == "/runs/iteration_6.json"


def test_both_arms_share_the_base_system_prompt_setting():
    """The two arms must be byte-identical apart from the playbook text, or the
    measured delta is confounded by the prompt difference."""
    common = {"EF_PROMPT": "1", "TEMPERATURE": "0.7"}
    base = driver.bfcl_env(arm="baseline", run_root=Path("/a"), playbook_file=None, base_env=common)
    evolved = driver.bfcl_env(arm="evolved", run_root=Path("/b"),
                              playbook_file=Path("/p.json"), base_env=common)
    differing = {k for k in set(base) | set(evolved) if base.get(k) != evolved.get(k)}
    assert differing == {"BFCL_RUN_ROOT", "PLAYBOOK"}


# ---------------------------------------------------------------- cli --


def test_the_parser_requires_an_out_dir():
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args([])


def test_skip_bfcl_removes_only_the_benchmark_stage(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(driver, "stage_evolve", lambda a, o: ran.append("evolve"))
    monkeypatch.setattr(driver, "stage_indomain", lambda a, o: ran.append("indomain"))
    monkeypatch.setattr(driver, "stage_bfcl", lambda a, o: ran.append("bfcl"))
    monkeypatch.setattr(driver, "stage_report", lambda a, o: ran.append("report") or 0)
    driver.main(["--out-dir", str(tmp_path), "--skip-bfcl"])
    assert ran == ["evolve", "indomain", "report"]


def test_a_void_study_exits_nonzero(tmp_path, monkeypatch):
    """So an orchestration script cannot mistake an infrastructure failure for
    a completed experiment."""
    monkeypatch.setattr(driver, "stage_report", lambda a, o: 2)
    assert driver.main(["--out-dir", str(tmp_path), "--stages", "4"]) == 2
