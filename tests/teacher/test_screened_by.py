"""A screened pass rate is a statement about an agent, not about a task.

These pin down that the bank refuses to launder an old agent's measurement as
the current one's. Without that, an improving agent keeps being served the
tasks it has already outgrown -- the band still says 0.4 because nobody
re-measured -- and the curriculum quietly stops being a curriculum.
"""

from qwen_agentworld.core.schemas import Playbook, PlaybookEntry
from qwen_agentworld.playbook_store.store import fingerprint
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, TaskBank

from tests.teacher.test_task_bank import make_task


def _pb(content, version=1):
    return Playbook(
        version=version,
        entries=[PlaybookEntry(tag="precondition-check", content=content, version=version)],
    )


def test_fingerprint_follows_content_not_version():
    # A rollback bumps the version without changing a word of what the agent
    # is told, so measurements taken either side of it are still comparable.
    assert fingerprint(_pb("check first", 1)) == fingerprint(_pb("check first", 7))
    assert fingerprint(_pb("check first")) != fingerprint(_pb("check twice"))
    # Same playbook served by a different model is a different agent.
    assert fingerprint(_pb("check first"), "Qwen3-8B") != fingerprint(_pb("check first"), "Qwen3-32B")


def test_draw_ignores_a_rate_measured_against_another_agent(tmp_path):
    bank = TaskBank(tmp_path)
    task = make_task(prompt="mid")
    bank.save(task, split=SPLIT_EVAL)
    bank.set_baseline_pass_rate(task.task_id, 0.5, screened_by="agent_v1")

    in_band = dict(band=(0.3, 0.7), require_screened=True)
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, **in_band)) == 1
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, screened_by="agent_v1", **in_band)) == 1
    # The evolved agent has no measurement for this task, so it must not inherit one.
    assert bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, screened_by="agent_v2", **in_band) == []


def test_a_stale_rate_reads_as_unscreened_not_as_out_of_band(tmp_path):
    """The distinction matters: unscreened tasks are re-measurable, out-of-band
    ones are discarded. A stale rate must land in the first category."""
    bank = TaskBank(tmp_path)
    task = make_task(prompt="ceiling")
    bank.save(task, split=SPLIT_EVAL)
    bank.set_baseline_pass_rate(task.task_id, 1.0, screened_by="agent_v1")

    # Out of band for the agent that measured it...
    assert bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, band=(0.2, 0.6),
                     screened_by="agent_v1") == []
    # ...but merely unmeasured for the next one, so a non-strict draw keeps it.
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, band=(0.2, 0.6),
                         screened_by="agent_v2")) == 1


def test_rate_without_a_screened_by_belongs_to_no_agent(tmp_path):
    """Entries written before this field existed. Guessing that they belong to
    whoever is asking is exactly the silent-staleness bug being prevented."""
    bank = TaskBank(tmp_path)
    task = make_task(prompt="legacy")
    bank.save(task, split=SPLIT_EVAL)
    bank.set_baseline_pass_rate(task.task_id, 0.5)  # no screened_by

    args = dict(band=(0.3, 0.7), require_screened=True)
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, **args)) == 1
    assert bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, screened_by="agent_v1", **args) == []
