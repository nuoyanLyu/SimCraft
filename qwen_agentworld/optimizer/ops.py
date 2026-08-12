"""Incremental edit operations on a playbook, and how they are applied.

This is the mechanism that replaces whole-module rewriting. Before, an
iteration produced one new version of one module's full text: the only way to
record a new lesson was to re-emit the module, so every update risked losing
what was already there, and under a word budget it *reliably* lost it. The
playbook could not accumulate — measured, it made one edit across an entire
run and the module oscillated in length rather than growing in coverage.

Here an iteration produces a small list of ops instead:

* ``add``    — a lesson the playbook does not yet contain. The default. Prior
               entries are untouched, so learning is monotone by construction.
* ``update`` — sharpen one existing entry, in place, keeping its id lineage.
* ``merge``  — fold 2+ entries that turned out to say the same thing into one.
* ``remove`` — retire an entry that later evidence contradicts.

Only ``merge`` and ``remove`` can lose content, and both must name the entries
they consume, so a deletion is always an explicit, attributable decision
rather than a side effect of rewriting.

There is no size cap (a deliberate choice: capping would evict entries for
being late rather than for being wrong). Growth is bounded by suppressing
near-verbatim re-additions and by the teacher's own merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from qwen_agentworld.core.schemas import (
    Diagnosis,
    EntryStats,
    Playbook,
    PlaybookEntry,
    normalize_tag,
)

# Jaccard overlap above which a proposed `add` is treated as a restatement of
# an entry that already exists. Set high on purpose: the cost of suppressing a
# genuinely new lesson (it is lost, silently) is much worse than the cost of
# keeping a near-duplicate (the playbook is slightly redundant, and the
# teacher can still merge the two later). Only near-verbatim repeats are cut.
DUPLICATE_JACCARD_THRESHOLD = 0.7


class PlaybookOp(BaseModel):
    """One edit. ``entry_ids`` names the entries the op consumes: empty for
    ``add``, exactly one for ``update``/``remove``, two or more for ``merge``.
    """

    op: Literal["add", "update", "merge", "remove"]
    content: str | None = None
    tag: str | None = None
    entry_ids: list[str] = Field(default_factory=list)


@dataclass
class OpsReport:
    """What actually happened, for logging and for the per-iteration metrics.

    "How many edits did this run make" was previously unanswerable without
    diffing playbook texts by hand; it is the headline diagnostic for whether
    the loop is learning at all, so the apply step reports it directly.
    """

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    suppressed_duplicates: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def n_changes(self) -> int:
        return len(self.added) + len(self.updated) + len(self.merged) + len(self.removed)


def _content_tokens(content: str) -> set[str]:
    words = "".join(ch if ch.isalnum() else " " for ch in content.lower()).split()
    return {w for w in words if len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words. Deliberately lexical, not semantic:
    duplicate detection runs on every add, and paying an LLM call for it would
    make the cheapest op the most expensive one.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_duplicate(playbook: Playbook, content: str) -> PlaybookEntry | None:
    best, best_score = None, 0.0
    for entry in playbook.entries:
        score = similarity(entry.content, content)
        if score > best_score:
            best, best_score = entry, score
    return best if best_score >= DUPLICATE_JACCARD_THRESHOLD else None


def apply_ops(playbook: Playbook, ops: list[PlaybookOp]) -> tuple[Playbook, OpsReport]:
    """Apply `ops` to `playbook`, returning the next version and a report.

    Ops are applied in order against the accumulating result, so a `merge` can
    consume an entry an earlier op in the same batch just added. Any op naming
    an entry_id that does not exist is rejected and recorded rather than
    raising: a single hallucinated id should cost one edit, not the iteration.
    """
    entries = [e.model_copy(deep=True) for e in playbook.entries]
    next_version = playbook.version + 1
    report = OpsReport()

    def index_of(entry_id: str) -> int | None:
        return next((i for i, e in enumerate(entries) if e.entry_id == entry_id), None)

    for op in ops:
        working = playbook.model_copy(update={"entries": entries})

        if op.op == "add":
            if not (op.content or "").strip():
                report.rejected.append("add with empty content")
                continue
            duplicate = find_duplicate(working, op.content)
            if duplicate is not None:
                report.suppressed_duplicates.append(duplicate.entry_id)
                continue
            entry = PlaybookEntry(
                tag=normalize_tag(op.tag or "general"),
                content=op.content.strip(),
                created_at_playbook_version=next_version,
                updated_at_playbook_version=next_version,
            )
            entries.append(entry)
            report.added.append(entry.entry_id)

        elif op.op == "update":
            if len(op.entry_ids) != 1 or not (op.content or "").strip():
                report.rejected.append(f"malformed update {op.entry_ids}")
                continue
            idx = index_of(op.entry_ids[0])
            if idx is None:
                report.rejected.append(f"update of unknown entry {op.entry_ids[0]}")
                continue
            old = entries[idx]
            # The id is preserved, not regenerated: accumulated helpful/harmful
            # counts belong to the *lesson*, and a sharpened wording of it is
            # the same lesson. Regenerating would reset credit on every edit.
            entries[idx] = old.model_copy(
                update={
                    "content": op.content.strip(),
                    "tag": normalize_tag(op.tag) if op.tag else old.tag,
                    "version": old.version + 1,
                    "updated_at_playbook_version": next_version,
                }
            )
            report.updated.append(old.entry_id)

        elif op.op == "merge":
            if len(op.entry_ids) < 2 or not (op.content or "").strip():
                report.rejected.append(f"malformed merge {op.entry_ids}")
                continue
            indices = [i for i in (index_of(eid) for eid in op.entry_ids) if i is not None]
            if len(indices) < 2:
                report.rejected.append(f"merge naming unknown entries {op.entry_ids}")
                continue
            consumed = [entries[i] for i in indices]
            merged = PlaybookEntry(
                tag=normalize_tag(op.tag) if op.tag else consumed[0].tag,
                content=op.content.strip(),
                version=max(e.version for e in consumed) + 1,
                provenance=[e.entry_id for e in consumed],
                # Credit is summed rather than reset: the merged entry is the
                # continuation of every lesson it absorbed, and starting it at
                # zero would make merging look like unlearning.
                stats=_sum_stats(consumed),
                created_at_playbook_version=min(e.created_at_playbook_version for e in consumed),
                updated_at_playbook_version=next_version,
            )
            insert_at = min(indices)
            entries = [e for i, e in enumerate(entries) if i not in set(indices)]
            entries.insert(insert_at, merged)
            report.merged.append(merged.entry_id)
            report.removed.extend(e.entry_id for e in consumed)

        elif op.op == "remove":
            if len(op.entry_ids) != 1:
                report.rejected.append(f"malformed remove {op.entry_ids}")
                continue
            idx = index_of(op.entry_ids[0])
            if idx is None:
                report.rejected.append(f"remove of unknown entry {op.entry_ids[0]}")
                continue
            report.removed.append(entries.pop(idx).entry_id)

    if report.n_changes == 0:
        return playbook, report

    return (
        playbook.model_copy(
            update={
                "version": next_version,
                "entries": entries,
                "validation_utility": None,  # a new text has not been validated yet
            }
        ),
        report,
    )


def _sum_stats(consumed: list[PlaybookEntry]) -> EntryStats:
    return EntryStats(
        helpful=sum(e.stats.helpful for e in consumed),
        harmful=sum(e.stats.harmful for e in consumed),
    )


def apply_credit(playbook: Playbook, diagnosis: Diagnosis) -> Playbook:
    """Record which entries the diagnosis found helpful or harmful.

    Without this, "which guidance is actually earning its place" is
    unanswerable and every entry looks identical to selection and to merging.
    Ids the diagnosis invented are ignored rather than raising.
    """
    helpful, harmful = set(diagnosis.helpful_entry_ids), set(diagnosis.harmful_entry_ids)
    if not helpful and not harmful:
        return playbook
    entries = []
    for entry in playbook.entries:
        stats = entry.stats.model_copy(
            update={
                "helpful": entry.stats.helpful + (1 if entry.entry_id in helpful else 0),
                "harmful": entry.stats.harmful + (1 if entry.entry_id in harmful else 0),
            }
        )
        entries.append(entry.model_copy(update={"stats": stats}))
    return playbook.model_copy(update={"entries": entries})


def render_entries(playbook: Playbook, *, with_ids: bool = False, with_stats: bool = False) -> str:
    """Group entries by tag for display. Tags order by first appearance, so a
    reader sees the playbook's own history rather than an imposed taxonomy.
    """
    if not playbook.entries:
        return ""
    blocks = []
    for tag in playbook.tags():
        lines = []
        for entry in playbook.entries_by_tag(tag):
            prefix = f"({entry.entry_id}) " if with_ids else ""
            suffix = (
                f"  [helpful={entry.stats.helpful} harmful={entry.stats.harmful}]" if with_stats else ""
            )
            lines.append(f"- {prefix}{entry.content}{suffix}")
        blocks.append(f"[{tag}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)
