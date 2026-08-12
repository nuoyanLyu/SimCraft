"""Tests for checkpoint loading.

The property under protection: a checkpoint this loader cannot honestly read
must raise, never return an empty playbook. An empty return makes the evolved
arm identical to the baseline arm, and the study then reports "no gain" from a
comparison it never actually ran.
"""

import json

import pytest

from qwen_agentworld.core.schemas import Playbook, PlaybookEntry
from qwen_agentworld.study.checkpoints import (
    Checkpoint,
    StalePlaybookFormatError,
    collect_checkpoints,
    dedupe_checkpoints,
    empty_playbook,
    iteration_files,
    load_playbook,
    playbook_from_dict,
)


def write_iteration(directory, n, entries, *, version=None):
    playbook = Playbook(
        version=version if version is not None else n + 1,
        entries=[PlaybookEntry(tag="t", content=c) for c in entries],
    )
    path = directory / f"iteration_{n}.json"
    path.write_text(json.dumps({"iteration": n, "playbook_after": playbook.model_dump(mode="json")}))
    return path


# --------------------------------------------------------------- format --


def test_a_pre_entry_checkpoint_raises_instead_of_loading_empty():
    stale = {"version": 5, "modules": {"schema_grounding": {"content": "old text"}}}
    with pytest.raises(StalePlaybookFormatError):
        playbook_from_dict(stale, source="iteration_4.json")


def test_the_error_names_the_file_so_it_can_be_acted_on():
    with pytest.raises(StalePlaybookFormatError, match="iteration_4.json"):
        playbook_from_dict({"modules": {}, "version": 2}, source="iteration_4.json")


def test_an_iteration_record_and_a_bare_playbook_both_load():
    playbook = Playbook(entries=[PlaybookEntry(tag="a", content="x")])
    raw = playbook.model_dump(mode="json")
    assert len(playbook_from_dict(raw).entries) == 1
    assert len(playbook_from_dict({"playbook_after": raw}).entries) == 1


def test_the_baseline_arm_is_an_empty_playbook_by_construction():
    assert empty_playbook().entries == []
    assert empty_playbook().version == 1


# ------------------------------------------------------------ ordering --


def test_iterations_are_ordered_numerically_not_lexically(tmp_path):
    """`iteration_10` sorts before `iteration_2` as a string, which would label
    the wrong checkpoint 'final' on any run of ten or more iterations."""
    for n in (1, 2, 10):
        write_iteration(tmp_path, n, [f"lesson {n}"])
    assert [p.name for p in iteration_files(tmp_path)] == [
        "iteration_1.json", "iteration_2.json", "iteration_10.json",
    ]


def test_collect_takes_the_last_iteration_as_final(tmp_path):
    for n in (1, 2, 3, 4):
        write_iteration(tmp_path, n, [f"lesson {i}" for i in range(n)])
    labels = {c.label: c for c in collect_checkpoints(tmp_path)}
    assert set(labels) == {"baseline", "mid", "final"}
    assert len(labels["final"].playbook.entries) == 4
    assert labels["baseline"].playbook.entries == []


def test_a_short_run_gets_no_midpoint(tmp_path):
    """With two iterations 'mid' and 'final' are adjacent; the comparison
    between them would say nothing about accumulation."""
    for n in (1, 2):
        write_iteration(tmp_path, n, ["a"])
    assert [c.label for c in collect_checkpoints(tmp_path)] == ["baseline", "final"]


def test_collect_on_an_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_checkpoints(tmp_path)


def test_load_playbook_reads_an_iteration_file(tmp_path):
    path = write_iteration(tmp_path, 1, ["only lesson"])
    assert [e.content for e in load_playbook(path).entries] == ["only lesson"]


# --------------------------------------------------------------- dedupe --


def test_arms_with_identical_entry_text_collapse_into_one():
    """Two arms telling the agent the same thing are one experiment; running
    both spends GPU twice and invites reading the gap as a trend."""
    playbook = Playbook(entries=[PlaybookEntry(tag="t", content="same lesson")])
    kept = dedupe_checkpoints([
        Checkpoint("mid", playbook, "a"),
        Checkpoint("final", playbook.model_copy(update={"version": 9}), "b"),
    ])
    assert len(kept) == 1
    assert kept[0].label == "mid+final"


def test_a_version_bump_alone_does_not_make_a_new_arm():
    """A rolled-back edit bumps the version without changing a word the agent
    reads, so version is the wrong identity to compare on."""
    entries = [PlaybookEntry(tag="t", content="unchanged text")]
    a = Checkpoint("mid", Playbook(version=2, entries=entries), "a")
    b = Checkpoint("final", Playbook(version=7, entries=entries), "b")
    assert a.fingerprint == b.fingerprint


def test_genuinely_different_arms_survive_dedupe():
    a = Checkpoint("mid", Playbook(entries=[PlaybookEntry(tag="t", content="lesson one")]), "a")
    b = Checkpoint("final", Playbook(entries=[PlaybookEntry(tag="t", content="lesson two")]), "b")
    assert len(dedupe_checkpoints([a, b])) == 2
