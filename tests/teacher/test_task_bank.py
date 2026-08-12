import json
import time

import pytest

from qwen_agentworld.core.schemas import CheckerSpec, DifficultyMeta, Task, TaskGraph, TaskGraphNode
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, SPLIT_TRAIN, TaskBank


def make_task(family="mcp_notes", n_nodes=3, prompt="Do the thing."):
    nodes = [
        TaskGraphNode(node_id=f"n{i}", tool_name="write_note", depends_on=([f"n{i-1}"] if i > 1 else []))
        for i in range(1, n_nodes + 1)
    ]
    return Task(
        tool_family=family,
        task_graph=TaskGraph(nodes=nodes),
        natural_language_prompt=prompt,
        initial_state={"notes": []},
        checker=CheckerSpec(executable_predicate="len(state['notes']) == 1"),
        difficulty_meta=DifficultyMeta(graph_complexity=n_nodes),
    )


def test_save_then_draw_roundtrip(tmp_path):
    bank = TaskBank(tmp_path)
    t = make_task()
    bank.save(t, split=SPLIT_EVAL, origin="unit-test")
    drawn = bank.draw("mcp_notes", 3, n=5, split=SPLIT_EVAL)
    assert [d.task_id for d in drawn] == [t.task_id]
    assert drawn[0].natural_language_prompt == "Do the thing."


def test_draw_will_not_cross_the_train_eval_split(tmp_path):
    """The A/B only means anything if the eval set was never seen by the evolve
    run. Once tasks are pooled and reused that separation stops being automatic,
    so the bank enforces it rather than trusting the caller.
    """
    bank = TaskBank(tmp_path)
    train_task = make_task(prompt="train one")
    bank.save(train_task, split=SPLIT_TRAIN)
    assert bank.draw("mcp_notes", 3, n=5, split=SPLIT_EVAL) == []
    assert len(bank.draw("mcp_notes", 3, n=5, split=SPLIT_TRAIN)) == 1


def test_save_rejects_an_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        TaskBank(tmp_path).save(make_task(), split="holdout")


def test_each_task_is_on_disk_before_the_next_is_generated(tmp_path):
    """A batch used to be written only after it finished, so a crash at task 39
    of 40 discarded every teacher call already paid for.
    """
    bank = TaskBank(tmp_path)
    first = make_task(prompt="first")
    bank.save(first, split=SPLIT_EVAL)
    # Simulate the crash: nothing else is ever saved.
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL)) == 1


def test_band_filter_uses_screened_pass_rate(tmp_path):
    bank = TaskBank(tmp_path)
    easy, mid, hard = make_task(prompt="easy"), make_task(prompt="mid"), make_task(prompt="hard")
    for t in (easy, mid, hard):
        bank.save(t, split=SPLIT_EVAL)
    bank.set_baseline_pass_rate(easy.task_id, 1.0)
    bank.set_baseline_pass_rate(mid.task_id, 0.5)
    bank.set_baseline_pass_rate(hard.task_id, 0.0)
    drawn = bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, band=(0.3, 0.7), require_screened=True)
    assert [d.natural_language_prompt for d in drawn] == ["mid"]


def test_unscreened_tasks_survive_a_band_filter_unless_required(tmp_path):
    bank = TaskBank(tmp_path)
    t = make_task(prompt="unmeasured")
    bank.save(t, split=SPLIT_EVAL)
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, band=(0.3, 0.7))) == 1
    assert bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL, band=(0.3, 0.7), require_screened=True) == []


def test_prune_by_age_keeps_recent(tmp_path):
    bank = TaskBank(tmp_path)
    old, new = make_task(prompt="old"), make_task(prompt="new")
    old_path = bank.save(old, split=SPLIT_EVAL)
    bank.save(new, split=SPLIT_EVAL)
    payload = json.loads(old_path.read_text())
    payload["meta"]["created_at"] = time.time() - 40 * 86400
    old_path.write_text(json.dumps(payload))

    removed = bank.prune(max_age_days=30)
    assert removed == [old_path]
    assert [t.natural_language_prompt for t in bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL)] == ["new"]


def test_prune_dry_run_deletes_nothing(tmp_path):
    bank = TaskBank(tmp_path)
    bank.save(make_task(), split=SPLIT_EVAL)
    removed = bank.prune(max_per_bucket=0, dry_run=True)
    assert len(removed) == 1
    assert len(bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL)) == 1


def test_prune_caps_bucket_size_keeping_newest(tmp_path):
    bank = TaskBank(tmp_path)
    paths = []
    for i in range(3):
        t = make_task(prompt=f"t{i}")
        path = bank.save(t, split=SPLIT_EVAL)
        payload = json.loads(path.read_text())
        payload["meta"]["created_at"] = 1000.0 + i  # t2 newest
        path.write_text(json.dumps(payload))
        paths.append(path)
    bank.prune(max_per_bucket=1)
    survivors = [t.natural_language_prompt for t in bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL)]
    assert survivors == ["t2"]


def test_unreadable_entry_is_skipped_by_draw_and_removed_by_prune(tmp_path):
    bank = TaskBank(tmp_path)
    bank.save(make_task(prompt="good"), split=SPLIT_EVAL)
    junk = tmp_path / "mcp_notes" / "gc3" / "task_broken.json"
    junk.write_text("{not json")
    assert [t.natural_language_prompt for t in bank.draw("mcp_notes", 3, n=10, split=SPLIT_EVAL)] == ["good"]
    assert junk in bank.prune(max_age_days=99999)
    assert not junk.exists()
