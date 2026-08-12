"""Turn observations into the Pareto scores that the store and the optimizer rank on.

`ParetoScores` rode along on every module since Stage 6 and nothing in
production ever wrote a non-zero value into it, so every consumer degenerated
quietly — `PlaybookOptimizer.select` found no candidate dominating any other,
`merge` always kept its left argument, and `PlaybookStore._update_frontier`
appended unconditionally, because `dominates` cannot be true when every axis
ties. Two things were wrong: nobody supplied the observed axes, and the scores
lived on the wrong object. A rollout exercises the *whole* playbook, so a
per-module score could only ever be the same number copied across modules,
which is precisely the tie that flattened the frontier. The axes now live on
`Playbook`, where they are measured; per-entry credit lives on `EntryStats`,
where it can actually differ between entries.

`compactness` also changed meaning. It used to be a per-module length penalty
and it was load-bearing in the worst way: it was the pressure that made every
mutation delete prior guidance to stay under budget. With an unbounded entry
list the playbook is allowed to grow, so compactness is a *soft preference*
between candidate playbooks of comparable merit (fewer, tighter entries beat a
sprawl that says the same thing), not a limit anything has to satisfy.
"""

from __future__ import annotations

from qwen_agentworld.core.schemas import ParetoScores, Playbook, PlaybookEntry

# Roughly the length at which an entry stops being a rule and starts being a
# document the agent must skim. Per *entry*, not per playbook.
DEFAULT_ENTRY_WORD_BUDGET = 40


def compactness_score(playbook: Playbook, entry_word_budget: int = DEFAULT_ENTRY_WORD_BUDGET) -> float:
    """Mean per-entry tightness: 1.0 when every entry is short, falling to 0.0
    as entries approach `entry_word_budget` words each.

    Deliberately *not* a function of entry count. Penalising count would make a
    playbook that learned twenty things score worse than one that learned two,
    which is the opposite of what this loop is for; the length of each
    individual rule is the thing worth pressure.
    """
    if entry_word_budget <= 0:
        raise ValueError(f"entry_word_budget must be positive, got {entry_word_budget}")
    if not playbook.entries:
        return 1.0
    per_entry = [
        max(0.0, 1.0 - len(entry.content.split()) / entry_word_budget) for entry in playbook.entries
    ]
    return sum(per_entry) / len(per_entry)


def entry_utility(entry: PlaybookEntry) -> int:
    """How much this entry has earned its place: helpful minus harmful."""
    return entry.stats.utility


def score_playbook(
    playbook: Playbook,
    *,
    task_coverage: float | None = None,
    audit_acceptance: float | None = None,
    entry_word_budget: int = DEFAULT_ENTRY_WORD_BUDGET,
) -> Playbook:
    """Return `playbook` with its Pareto scores refreshed.

    `compactness` is always recomputed from the current entries. The two
    observation-driven axes keep their previous value when the caller has no
    new measurement, so a playbook is never silently reset to 0.0 by a caller
    that only wanted to update its length.
    """
    previous = playbook.pareto_scores
    return playbook.model_copy(
        update={
            "pareto_scores": ParetoScores(
                task_coverage=previous.task_coverage if task_coverage is None else task_coverage,
                audit_acceptance=(
                    previous.audit_acceptance if audit_acceptance is None else audit_acceptance
                ),
                compactness=compactness_score(playbook, entry_word_budget),
            )
        }
    )
