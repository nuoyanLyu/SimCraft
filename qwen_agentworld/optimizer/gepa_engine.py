"""GEPA-style engine: reflective edits to a playbook, expressed as a small
batch of incremental operations rather than a rewrite of one module.

"GEPA" here means the core idea this repo leans on for U1 — use natural-
language reflection on failures to evolve a text artifact — not the full
published population-based Pareto search; that variant is future work.

What changed, and why it matters more than it sounds: the engine used to pick
one of five fixed categories (`most_implicated_category`) and re-emit that
module's entire text under a word budget. Since the rewrite was whole and the
budget was binding, the model was explicitly told to "fold the new lesson into
an existing sentence and drop guidance that no longer earns its place" — every
edit therefore overwrote prior learning, and the playbook could not accumulate
across iterations even in principle. Now the engine emits `add` / `update` /
`merge` / `remove` ops (optimizer/ops.py). `add` is the default and touches
nothing else, so learning is monotone unless the teacher deliberately retires
something.
"""

from __future__ import annotations

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Diagnosis, Playbook
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.optimizer.ops import PlaybookOp, apply_credit, apply_ops, render_entries
from qwen_agentworld.optimizer.scoring import DEFAULT_ENTRY_WORD_BUDGET, score_playbook

_EDIT_SYSTEM_PROMPT = (
    "You maintain a reusable, domain-agnostic tool-use playbook: a growing list of short, "
    "self-contained rules learned from observed failures. You are given the current entries and "
    "a diagnosis of one trajectory, and you propose a small batch of edits.\n"
    "Operations:\n"
    '  {"op": "add", "tag": "<short-kebab-case-label>", "content": "<one rule>"} — a lesson the '
    "playbook does not yet contain. This is the usual case.\n"
    '  {"op": "update", "entry_ids": ["<id>"], "content": "<revised rule>"} — sharpen one '
    "existing entry that was close but insufficient.\n"
    '  {"op": "merge", "entry_ids": ["<id>", "<id>"], "content": "<combined rule>"} — two or '
    "more entries turned out to say the same thing.\n"
    '  {"op": "remove", "entry_ids": ["<id>"]} — evidence contradicts this entry.\n'
    "Rules for the content you write: one actionable idea per entry, stated as general strategy "
    "(e.g. 'confirm every required parameter is present before calling a tool', never 'call "
    "search_docs with a query parameter'); never mention specific tool names, endpoints, or task "
    "answers. Choose the `tag` yourself — reuse one already present when it fits, invent one when "
    "it does not; there is no fixed list.\n"
    "Do not restate an entry that already exists, and do not delete an entry merely because this "
    "trajectory did not need it — entries are cheap and the list has no size limit. "
    "Propose no edits at all (an empty list) when the diagnosis teaches nothing reusable.\n"
    'Reply with a single JSON object: {"ops": [ ... ]}.'
)

_MAX_MUTATION_ATTEMPTS = 3


def build_entry_length_rule(word_budget: int) -> str:
    """Bound the length of a single *entry*, not of the playbook.

    The old budget was per-module and the playbook had a fixed number of
    modules, so it was really a cap on total learned content and it forced
    deletion once reached. Here the list may grow without limit; what needs
    bounding is only that one entry stays a rule rather than becoming an essay,
    since an entry is the unit that gets credited, merged and retired.
    """
    return (
        f" Keep each entry under roughly {word_budget} words — one rule, not a paragraph. If a "
        "lesson needs more than that, it is really two entries."
    )


def parse_ops(content: str) -> list[PlaybookOp]:
    """Parse the teacher's reply into ops, skipping individual malformed ones.

    One unparseable op should not discard the others in the same batch: the
    batch is a set of independent edits, not a transaction.
    """
    payload = extract_json_object(content)
    raw_ops = payload["ops"]
    if not isinstance(raw_ops, list):
        raise TypeError(f"'ops' must be a list, got {type(raw_ops).__name__}")
    ops = []
    for raw in raw_ops:
        try:
            ops.append(PlaybookOp.model_validate(raw))
        except Exception:  # noqa: BLE001 — one bad op costs one edit, not the batch
            continue
    return ops


def _chat_for_ops(teacher: LLMClient, messages: list[dict], max_tokens: int) -> list[PlaybookOp]:
    """Retries on an empty or malformed (e.g. truncated mid-string) reply,
    same failure mode `task_generator.instantiate_nl_and_state` and
    `checker_synth.synthesize_checker` already guard against.
    """
    last_error: Exception | None = None
    for _ in range(_MAX_MUTATION_ATTEMPTS):
        result = teacher.chat(messages=messages, max_tokens=max_tokens)
        try:
            return parse_ops(result.content or "")
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "That reply was empty or not valid JSON. Reply again with the JSON "
                    "object described in the system prompt.",
                }
            )
    raise last_error


def observed_tags(diagnosis: Diagnosis) -> list[str]:
    """Tags this diagnosis named, in order, deduplicated.

    Replaces `most_implicated_category`, which collapsed a whole trajectory to
    the single most-voted bucket and discarded every other lesson in it. A
    trajectory that failed for two distinct reasons can now teach two entries.
    """
    tags: list[str] = []
    for sd in diagnosis.step_diagnoses:
        if sd.suggested_tag and sd.suggested_tag not in tags:
            tags.append(sd.suggested_tag)
    return tags


def build_edit_prompt(current: Playbook, diagnosis: Diagnosis) -> str:
    feedback = "\n".join(
        f"- step {sd.step_id} ({sd.verdict}, tag: {sd.suggested_tag or 'unlabelled'}): {sd.feedback}"
        for sd in diagnosis.step_diagnoses
        if sd.verdict != "correct"
    )
    entries = render_entries(current, with_ids=True, with_stats=True) or "(empty — no entries yet)"
    return (
        f"Current playbook entries:\n{entries}\n\n"
        f"Failure feedback from this trajectory:\n{feedback or '(no step-level faults reported)'}\n\n"
        f"Overall trajectory verdict: {diagnosis.overall_verdict}. Summary: {diagnosis.summary}\n\n"
        "Propose the edits this trajectory justifies."
    )


class GEPAEngine(PlaybookOptimizer):
    def __init__(self, teacher: LLMClient, entry_word_budget: int = DEFAULT_ENTRY_WORD_BUDGET) -> None:
        self._teacher = teacher
        self._entry_word_budget = entry_word_budget

    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        # Credit lands even when no edit follows: "the agent used this entry and
        # still failed" is information about the entry, and discarding it on
        # iterations that happen to produce no ops would throw most of it away.
        credited = apply_credit(current, diagnosis)

        if diagnosis.overall_verdict == "success" and not observed_tags(diagnosis):
            return [credited] if credited != current else []

        ops = _chat_for_ops(
            self._teacher,
            messages=[
                {
                    "role": "system",
                    "content": _EDIT_SYSTEM_PROMPT + build_entry_length_rule(self._entry_word_budget),
                },
                {"role": "user", "content": build_edit_prompt(credited, diagnosis)},
            ],
            max_tokens=1500,
        )
        candidate, report = apply_ops(credited, ops)
        if report.n_changes == 0:
            return [credited] if credited != current else []
        return [score_playbook(candidate, entry_word_budget=self._entry_word_budget)]
