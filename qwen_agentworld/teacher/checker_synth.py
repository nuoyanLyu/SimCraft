"""Checker synthesis (research plan constraint: checkers read canonical_state
only — never a natural-language reference answer, since an NL answer is
exactly what a hallucinating simulator or a lazy agent could pattern-match
against instead of actually completing the task).

Claude proposes `executable_predicate` as a restricted Python boolean
expression over a single `state` variable. We never trust that it actually
stayed restricted — `safe_predicate.validate_predicate_ast` statically vets
it, and `audit_no_nl_leak` additionally rejects any string literal inside it
that looks like prose rather than a short canonical value (e.g. an enum-like
status string is fine; a full sentence explaining the answer is not).
"""

from __future__ import annotations

import ast
import json
import time

from qwen_agentworld.core.schemas import CheckerSpec, TaskGraph, ToolSpec
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.teacher.safe_predicate import UnsafePredicateError, is_trivial_tautology, validate_predicate_ast

_MAX_LITERAL_CHARS = 80
_MAX_LITERAL_WORDS = 6
_MAX_SYNTHESIS_ATTEMPTS = 4

_CHECKER_SYSTEM_PROMPT = (
    "You write a machine-checkable post-condition for a tool-use task, given only the tool "
    "graph and the starting state (no transcript, no reference solution). "
    "Write a single Python boolean expression, as a JSON string, that evaluates against a "
    "variable named `state` (the canonical final state dict after execution). You may only use: "
    "comparisons, boolean operators (and/or/not), `state[...]` / `state.get(...)` / other dict "
    "methods, membership tests (in), slicing (state['x'][:2]), any/all over comprehensions, and "
    "the builtins len/abs/min/max/sorted/round. Do NOT use isinstance, type checks, or any other "
    "function call — only the listed builtins and dict/list methods. "
    "Do not write natural language anywhere — the only strings allowed are short literal values "
    "you compare against, like state['status'] == 'completed'. Never describe the answer in "
    "prose and never reference a transcript. "
    "The predicate must genuinely test the task outcome: never write a tautology that is true "
    "for every state (e.g. `x == 'a' or x != 'a'`) or a constant like `True`. "
    "MOST tasks leave a durable observable change, so an `executable_predicate` over the final "
    "`state` is enough. But if this task is inherently reversible — the correct final observable "
    "state can be IDENTICAL to the initial state (e.g. create an item then delete it) — an "
    "end-state predicate cannot distinguish real work-then-revert from doing nothing. In that "
    "case set step_wise_diagnostics=true and ALSO provide `step_wise_predicate`: a boolean "
    "expression over a variable named `states` (an ordered list; states[0] is the initial state "
    "and each later element is the canonical state after one executed step). Assert that some "
    "intermediate state proves the work happened AND the final state states[-1] shows the "
    "expected end condition — e.g. any(n['id'] == 'x' for s in states for n in s['notes']) and "
    "not any(n['id'] == 'x' for n in states[-1]['notes']). Same construct/allowlist rules as "
    "executable_predicate; `states` replaces `state`. When step_wise_diagnostics=false, omit "
    "step_wise_predicate (or set it to null). "
    'Reply with a single JSON object: {"executable_predicate": "<expr>", "step_wise_diagnostics": '
    "<true|false>, "
    '"step_wise_predicate": "<expr over states>"|null}.'
)


class CheckerAuditError(ValueError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def audit_no_nl_leak(predicate: str, root_var: str = "state") -> list[str]:
    """Statically vet `predicate`: must be a safe expression (safe_predicate)
    over `root_var` (`state` for the end-state checker, `states` for the
    step-wise one), must not be a trivial tautology, and must not contain
    string literals long/wordy enough to be a leaked natural-language
    reference answer rather than a canonical value.
    """
    try:
        tree = validate_predicate_ast(predicate, root_names=frozenset({root_var}))
    except UnsafePredicateError as exc:
        return [str(exc)]

    violations: list[str] = []
    if is_trivial_tautology(predicate):
        violations.append(
            "predicate is a tautology (true for every state) and cannot verify the task"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > _MAX_LITERAL_CHARS or len(node.value.split()) > _MAX_LITERAL_WORDS:
                violations.append(f"string literal looks like a natural-language leak: {node.value!r}")
    return violations


def _build_checker_prompt(graph: TaskGraph, tools: list[ToolSpec], initial_state: dict) -> str:
    tool_names = [t.name for t in tools]
    graph_summary = [
        {"node_id": n.node_id, "tool_name": n.tool_name, "depends_on": n.depends_on} for n in graph.nodes
    ]
    return (
        f"Tool graph:\n{json.dumps(graph_summary, indent=2)}\n\n"
        f"Tools involved: {tool_names}\n\n"
        f"Initial state:\n{json.dumps(initial_state, indent=2)}\n\n"
        "Produce the JSON object described in the system prompt."
    )


def synthesize_checker(
    teacher: LLMClient,
    graph: TaskGraph,
    tools: list[ToolSpec],
    initial_state: dict,
    max_attempts: int = _MAX_SYNTHESIS_ATTEMPTS,
) -> CheckerSpec:
    """Ask `teacher` for a predicate, retrying with the audit's own violation
    message fed back as feedback when it's rejected. Live testing showed the
    first draft fails the audit a majority of the time (isinstance, slicing,
    or genuine NL leaks) — this is normal model variance, not a sign the
    predicate needs a human to intervene, so a bounded retry loop resolves it
    without weakening the audit itself.
    """
    messages = [
        {"role": "system", "content": _CHECKER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_checker_prompt(graph, tools, initial_state)},
    ]
    violations: list[str] = []
    for attempt in range(1, max_attempts + 1):
        result = teacher.chat(messages=messages, max_tokens=500)
        content = result.content or ""
        try:
            payload = extract_json_object(content)
            predicate = payload["executable_predicate"]
        except (ValueError, KeyError, TypeError) as exc:
            # Empty / malformed reply, or JSON missing the expected field: feed
            # the problem back and retry within the same bounded loop, mirroring
            # the empty-content retry guards on instantiate_nl_and_state /
            # reflection.diagnose / the optimizer engines. Live run (real
            # Claude teacher, graph_complexity=3) hit an empty content here and
            # crashed the whole iteration because only audit violations were
            # being retried, not parse failures.
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was empty or not a valid JSON object with an "
                        "\"executable_predicate\" string field. Reply again with exactly the "
                        "JSON object described in the system prompt and nothing else."
                    ),
                }
            )
            violations = [f"malformed teacher response: {exc}"]
            # Empty 200s from the Claude relay come in short bursts when the
            # loop fires many calls back-to-back (observed live at
            # graph_complexity=3 with several tasks). An immediate re-ask stays
            # inside the same throttle window and also comes back empty, which
            # is how all attempts got exhausted; a brief backoff lets the burst
            # clear before we retry.
            time.sleep(2.0 * attempt)
            continue

        step_wise = bool(payload.get("step_wise_diagnostics", False))
        step_predicate = payload.get("step_wise_predicate") or None
        violations = audit_no_nl_leak(predicate)
        if step_wise:
            # A reversible task that opts into step-wise scoring must supply a
            # `states` predicate — otherwise judge_checker silently falls back
            # to the end-state predicate that step-wise mode exists to avoid.
            if not step_predicate:
                violations.append(
                    "step_wise_diagnostics is true but step_wise_predicate is missing; "
                    "provide a boolean expression over the `states` list"
                )
            else:
                violations.extend(audit_no_nl_leak(step_predicate, root_var="states"))
        if not violations:
            return CheckerSpec(
                executable_predicate=predicate,
                step_wise_diagnostics=step_wise,
                step_wise_predicate=step_predicate if step_wise else None,
            )

        messages.append({"role": "assistant", "content": result.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "That predicate was rejected by the safety audit: "
                    f"{'; '.join(violations)}. Reply again with a corrected JSON object using "
                    "only the allowed constructs."
                ),
            }
        )

    raise CheckerAuditError(violations)
