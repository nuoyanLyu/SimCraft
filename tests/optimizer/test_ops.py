"""Tests for incremental playbook edits.

The property these exist to protect is the one the old design could not have:
recording a new lesson must not delete the ones already there.
"""

from qwen_agentworld.core.schemas import Diagnosis, EntryStats, Playbook, PlaybookEntry
from qwen_agentworld.optimizer.ops import (
    PlaybookOp,
    apply_credit,
    apply_ops,
    find_duplicate,
    render_entries,
    similarity,
)


def entry(entry_id, content, tag="t", helpful=0, harmful=0) -> PlaybookEntry:
    return PlaybookEntry(
        entry_id=entry_id, tag=tag, content=content, stats=EntryStats(helpful=helpful, harmful=harmful)
    )


def pb(*entries, version=1) -> Playbook:
    return Playbook(version=version, entries=list(entries))


# ------------------------------------------------------------------- add --


def test_add_appends_without_touching_existing_entries():
    """The whole point of the redesign. Under the old whole-module rewrite, the
    only way to record this lesson was to re-emit the module, which under a word
    budget meant deleting something else."""
    before = pb(entry("e1", "check required parameters before calling"))
    after, report = apply_ops(
        before, [PlaybookOp(op="add", tag="error-recovery", content="re-read the error before retrying")]
    )
    assert [e.content for e in after.entries] == [
        "check required parameters before calling",
        "re-read the error before retrying",
    ]
    assert len(report.added) == 1


def test_a_batch_can_record_several_lessons_from_one_trajectory():
    """A trajectory that failed for two distinct reasons teaches two entries.
    `most_implicated_category` used to collapse it to a single majority vote."""
    after, report = apply_ops(
        pb(),
        [
            PlaybookOp(op="add", tag="a", content="first distinct lesson about parameters"),
            PlaybookOp(op="add", tag="b", content="second unrelated lesson about ordering"),
        ],
    )
    assert len(after.entries) == 2
    assert after.tags() == ["a", "b"]
    assert report.n_changes == 2


def test_tags_are_free_form_and_need_no_prior_registration():
    after, _ = apply_ops(pb(), [PlaybookOp(op="add", tag="Stale Identifier Reuse", content="a new lesson")])
    assert after.entries[0].tag == "stale-identifier-reuse"


def test_an_add_with_no_tag_still_lands():
    after, _ = apply_ops(pb(), [PlaybookOp(op="add", content="a lesson with no label given")])
    assert after.entries[0].tag == "general"


def test_an_empty_add_is_rejected_not_stored():
    after, report = apply_ops(pb(), [PlaybookOp(op="add", content="   ")])
    assert after.entries == []
    assert report.rejected


# ------------------------------------------------------------ duplicates --


def test_a_near_verbatim_restatement_is_suppressed():
    before = pb(entry("e1", "always confirm required parameters are present before calling a tool"))
    after, report = apply_ops(
        before, [PlaybookOp(op="add", content="always confirm required parameters are present before calling tool")]
    )
    assert len(after.entries) == 1
    assert report.suppressed_duplicates == ["e1"]


def test_a_genuinely_different_lesson_is_not_mistaken_for_a_duplicate():
    """Suppression is tuned to be conservative: losing a real lesson silently is
    far worse than tolerating a near-duplicate the teacher can merge later."""
    before = pb(entry("e1", "always confirm required parameters are present before calling a tool"))
    after, report = apply_ops(
        before, [PlaybookOp(op="add", content="verify the write landed by reading the state back afterwards")]
    )
    assert len(after.entries) == 2
    assert report.suppressed_duplicates == []


def test_similarity_is_symmetric_and_bounded():
    assert similarity("alpha beta gamma", "alpha beta gamma") == 1.0
    assert similarity("alpha beta", "delta epsilon") == 0.0
    assert similarity("", "anything") == 0.0


def test_find_duplicate_returns_none_on_an_empty_playbook():
    assert find_duplicate(pb(), "any content at all") is None


# ---------------------------------------------------------------- update --


def test_update_preserves_the_entry_id_so_credit_survives_an_edit():
    """Credit belongs to the *lesson*; a sharpened wording is the same lesson.
    Regenerating the id would reset helpful/harmful on every edit."""
    before = pb(entry("e1", "check parameters", helpful=4))
    after, _ = apply_ops(before, [PlaybookOp(op="update", entry_ids=["e1"], content="check every required parameter")])
    updated = after.by_id("e1")
    assert updated.content == "check every required parameter"
    assert updated.version == 2
    assert updated.stats.helpful == 4


def test_update_of_an_unknown_id_costs_one_edit_not_the_batch():
    before = pb(entry("e1", "keep me"))
    after, report = apply_ops(
        before,
        [
            PlaybookOp(op="update", entry_ids=["ghost"], content="nope"),
            PlaybookOp(op="add", content="a valid new lesson survives the bad op"),
        ],
    )
    assert len(after.entries) == 2
    assert report.rejected and len(report.added) == 1


# ----------------------------------------------------------------- merge --


def test_merge_folds_entries_together_and_sums_their_credit():
    before = pb(
        entry("e1", "verify writes", helpful=2),
        entry("e2", "read state back after writing", helpful=3, harmful=1),
    )
    after, report = apply_ops(
        before, [PlaybookOp(op="merge", entry_ids=["e1", "e2"], content="read the state back to verify every write")]
    )
    assert len(after.entries) == 1
    merged = after.entries[0]
    assert merged.stats.helpful == 5 and merged.stats.harmful == 1
    assert set(merged.provenance) == {"e1", "e2"}
    assert report.merged


def test_merge_needs_at_least_two_real_entries():
    before = pb(entry("e1", "alone"))
    after, report = apply_ops(before, [PlaybookOp(op="merge", entry_ids=["e1", "ghost"], content="x")])
    assert len(after.entries) == 1 and report.rejected


# ---------------------------------------------------------------- remove --


def test_remove_deletes_exactly_the_named_entry():
    before = pb(entry("e1", "keep"), entry("e2", "drop"))
    after, report = apply_ops(before, [PlaybookOp(op="remove", entry_ids=["e2"])])
    assert [e.entry_id for e in after.entries] == ["e1"]
    assert report.removed == ["e2"]


def test_deletion_is_only_ever_explicit():
    """Nothing but `merge` and `remove` can lose an entry, and both must name
    what they consume — a deletion can never be a side effect of an edit."""
    before = pb(entry("e1", "prior lesson one"), entry("e2", "prior lesson two"))
    after, _ = apply_ops(
        before,
        [
            PlaybookOp(op="add", content="a brand new and quite different lesson"),
            PlaybookOp(op="update", entry_ids=["e1"], content="prior lesson one, sharpened"),
        ],
    )
    assert {e.entry_id for e in before.entries} <= {e.entry_id for e in after.entries}


# -------------------------------------------------------------- versions --


def test_a_no_op_batch_returns_the_original_playbook_unchanged():
    before = pb(entry("e1", "unchanged"), version=7)
    after, report = apply_ops(before, [])
    assert after is before
    assert report.n_changes == 0


def test_a_changed_playbook_bumps_version_and_drops_its_stale_validation():
    before = Playbook(version=7, entries=[], validation_utility=0.5)
    after, _ = apply_ops(before, [PlaybookOp(op="add", content="a new lesson worth recording")])
    assert after.version == 8
    assert after.validation_utility is None


def test_apply_ops_does_not_mutate_the_input_playbook():
    before = pb(entry("e1", "original wording"))
    apply_ops(before, [PlaybookOp(op="update", entry_ids=["e1"], content="rewritten wording")])
    assert before.entries[0].content == "original wording"


# ---------------------------------------------------------------- credit --


def test_credit_is_recorded_per_entry():
    before = pb(entry("e1", "helped"), entry("e2", "hurt"), entry("e3", "uninvolved"))
    after = apply_credit(
        before,
        Diagnosis(
            task_id="t",
            overall_verdict="partial",
            summary="",
            helpful_entry_ids=["e1"],
            harmful_entry_ids=["e2"],
        ),
    )
    assert after.by_id("e1").stats.utility == 1
    assert after.by_id("e2").stats.utility == -1
    assert after.by_id("e3").stats.utility == 0


def test_credit_ignores_ids_the_teacher_invented():
    before = pb(entry("e1", "real"))
    after = apply_credit(
        before,
        Diagnosis(task_id="t", overall_verdict="success", summary="", helpful_entry_ids=["hallucinated"]),
    )
    assert after.by_id("e1").stats.utility == 0


def test_a_diagnosis_with_no_credit_returns_the_playbook_untouched():
    before = pb(entry("e1", "x"))
    assert apply_credit(before, Diagnosis(task_id="t", overall_verdict="success", summary="")) is before


# --------------------------------------------------------------- render --


def test_render_groups_by_tag_and_hides_ids_by_default():
    playbook = pb(entry("e1", "first", tag="alpha"), entry("e2", "second", tag="beta"))
    rendered = render_entries(playbook)
    assert "[alpha]\n- first" in rendered
    assert "[beta]\n- second" in rendered
    assert "e1" not in rendered


def test_render_can_expose_ids_and_stats_for_the_teacher():
    playbook = pb(entry("e1", "first", helpful=2, harmful=1))
    rendered = render_entries(playbook, with_ids=True, with_stats=True)
    assert "(e1)" in rendered and "helpful=2" in rendered and "harmful=1" in rendered


def test_render_of_an_empty_playbook_is_empty():
    assert render_entries(pb()) == ""
