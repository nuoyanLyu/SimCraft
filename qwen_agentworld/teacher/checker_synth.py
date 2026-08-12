"""Checker synthesis (research plan constraint: checkers read canonical_state
only — never a natural-language reference answer, since an NL answer is
exactly what a hallucinating simulator or a lazy agent could pattern-match
against instead of actually completing the task).

That constraint is about what the predicate *evaluates*, not about what the
teacher *sees while writing it*. The teacher is shown the task instruction:
without it, the tool graph alone under-determines the task (`tag_note ->
update_note` says nothing about which note, which tag, or what content), so the
teacher invents values that disagree with the instruction and the task becomes
unpassable. Measured 2026-07-28: 4 of 12 eval tasks were unpassable this way,
and they were exactly the 4 that never passed in any A/B arm. The predicate
still reads `state` only, and `audit_no_nl_leak` still rejects any prose that
tries to ride into the predicate as a string literal.

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
import logging
import time

from qwen_agentworld.core.schemas import CheckerSpec, TaskGraph, ToolSpec
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.teacher.safe_predicate import (
    UnsafePredicateError,
    evaluate_predicate,
    is_trivial_tautology,
    validate_predicate_ast,
)
from qwen_agentworld.tools.state_schema import StateSchema

logger = logging.getLogger(__name__)

_MAX_LITERAL_CHARS = 80
_MAX_LITERAL_WORDS = 6
_MAX_SYNTHESIS_ATTEMPTS = 4

_CHECKER_SYSTEM_PROMPT = (
    "You write a machine-checkable post-condition for a tool-use task, given the task "
    "instruction, the tool graph and the starting state (no transcript, no reference "
    "solution). "
    "Your predicate must be satisfied by a state that correctly follows the instruction, "
    "and must not demand anything the instruction does not ask for. Check the concrete "
    "values named in the instruction (titles, tags, contents) rather than inventing your "
    "own, and leave untouched entities exactly as the starting state has them. "
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
    "NEVER assert the character length of a text field (no `len(note['content']) == 47`). You "
    "cannot count characters reliably, and a miscount by one makes the task impossible to pass. "
    "To require text to be unchanged, compare it to the exact string from the initial state "
    "shown below, which you can copy verbatim; to require new text, compare against what the "
    "instruction specifies. `len()` is for collections you can see enumerated (how many notes, "
    "how many tags), never for strings. "
    "Only constrain what the instruction actually determines: assert the changes it asks for, and "
    "assert that everything it does not mention is preserved — but never invent a specific value "
    "for a field the instruction leaves untouched and the initial state does not already fix. "
    "An `executable_predicate` over the final `state` is enough ONLY when every step the "
    "instruction requires leaves a trace in that final state. Check each required step in turn: "
    "if ANY of them is invisible at the end, an end-state predicate cannot tell a correct run "
    "from one that skipped that step, and you MUST set step_wise_diagnostics=true. Steps that go "
    "invisible include: work that is undone later (create an item then delete it); an attribute "
    "set on something later destroyed (tag a note, then delete that note); an intermediate value "
    "overwritten by a later step (create with a placeholder, then update it). Read-only steps "
    "(search, list, show) are covered by `state['_action_log']`: an ordered list of the tool "
    "calls actually executed, each {'tool': <name>, 'arguments': {...}}, appended by the "
    "environment. Use it ONLY for a step that has no other witness in the state -- assert real "
    "effects (records created, files written, fields changed) whenever the step has one, or the "
    "task degenerates into checking that a fixed sequence of calls was replayed. A read-only "
    "step is checked like any(c['tool'] == 'web_search' for c in state['_action_log']), "
    "optionally constraining an argument the instruction fixes. In the remaining cases set "
    "case set step_wise_diagnostics=true and ALSO provide `step_wise_predicate`: a boolean "
    "expression over a variable named `states` (an ordered list; states[0] is the initial state "
    "and each later element is the canonical state after one executed step). Assert that some "
    "intermediate state proves the work happened AND the final state states[-1] shows the "
    "expected end condition — e.g. any(n['id'] == 'x' for s in states for n in s['notes']) and "
    "not any(n['id'] == 'x' for n in states[-1]['notes']). Same construct/allowlist rules as "
    "executable_predicate; `states` replaces `state`. NEVER assert on how many states there are "
    "(no `len(states) == 3`) and never index a middle state positionally (no `states[1]`): the "
    "agent may legitimately take a different number of steps, and a correct solution must not "
    "fail because of it. Quantify over states instead — `any(... for s in states)` — and use "
    "states[0] and states[-1] only for the initial and final conditions. When "
    "step_wise_diagnostics=false, omit "
    "step_wise_predicate (or set it to null). "
    'Reply with a single JSON object: {"executable_predicate": "<expr>", "step_wise_diagnostics": '
    "<true|false>, "
    '"step_wise_predicate": "<expr over states>"|null}.'
)


class CheckerAuditError(ValueError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def collect_canonical_strings(*sources: object) -> set[str]:
    """Every string appearing anywhere in `sources` (initial state, instruction).

    Used to tell a copied canonical value apart from invented prose: the teacher
    can only reproduce these verbatim, so a long literal that matches one is a
    value it was shown, not a reference answer it wrote.
    """
    found: set[str] = set()
    stack = list(sources)
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
    return found


def audit_no_nl_leak(
    predicate: str,
    root_var: str = "state",
    canonical_strings: set[str] | None = None,
) -> list[str]:
    """Statically vet `predicate`: must be a safe expression (safe_predicate)
    over `root_var` (`state` for the end-state checker, `states` for the
    step-wise one), must not be a trivial tautology, and must not contain
    string literals long/wordy enough to be a leaked natural-language
    reference answer rather than a canonical value.

    A long literal that appears verbatim in `canonical_strings` (the initial
    state and the instruction) is a copied value, not a leak, and is allowed —
    checkers are supposed to compare against exact canonical text, and the
    alternative the teacher reaches for otherwise (asserting a character count)
    is what made tasks unpassable.
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
    allowed = canonical_strings or set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            too_long = (
                len(node.value) > _MAX_LITERAL_CHARS
                or len(node.value.split()) > _MAX_LITERAL_WORDS
            )
            if too_long and node.value not in allowed:
                violations.append(f"string literal looks like a natural-language leak: {node.value!r}")
    return violations


# Bookkeeping keys the environment itself writes into every state, so a
# predicate may subscript them even though no family schema declares them.
_HARNESS_KEYS = frozenset({"_action_log", "tool", "arguments"})


class _SubscriptCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.tool_names: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            self.keys.add(node.slice.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # `state.get('notes', [])` reads a key just as `state['notes']` does.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self.keys.add(first.value)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # `c['tool'] == 'search_notes'` -- the right-hand side must be a real tool.
        left = node.left
        if (
            isinstance(left, ast.Subscript)
            and isinstance(left.slice, ast.Constant)
            and left.slice.value == "tool"
        ):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    self.tool_names.add(comparator.value)
        self.generic_visit(node)


def audit_schema_conformance(
    predicate: str,
    schema: StateSchema,
    tools: list[ToolSpec],
) -> list[str]:
    """Reject a predicate that reads a key this domain's state does not have.

    This is the audit that would have caught the 7 defective checkers in the
    240-task bank (`note['id']` five times, an `arguments['note_title']` where
    the tool declares `title`). Such a predicate does not evaluate to False --
    it raises KeyError, and `judge_checker` scores that as "task not passed",
    so a correctly solved task is recorded as a persistent failure and drags
    down every pass rate derived from it.

    Only *reads* are checked, and only against a declared schema: the legal
    vocabulary is the schema's own keys and fields, the tool parameter names
    (for `_action_log` entries' `arguments`), and the harness keys. That is
    strictly narrower than the bank-wide union of observed fields, which is
    what made the same check as a post-hoc script a lower bound only.
    """
    try:
        tree = ast.parse(predicate, mode="eval")
    except SyntaxError as exc:
        return [f"predicate does not parse: {exc}"]

    collector = _SubscriptCollector()
    collector.visit(tree)

    parameter_names: set[str] = set()
    for tool in tools:
        parameter_names.update(((tool.function.parameters or {}).get("properties") or {}).keys())
    allowed = schema.allowed_keys() | parameter_names | _HARNESS_KEYS
    known_tools = {tool.name for tool in tools}

    violations = []
    for key in sorted(collector.keys - allowed):
        violations.append(
            f"the predicate reads {key!r}, which this domain's state does not have; "
            f"the only readable keys are {sorted(allowed)}"
        )
    for name in sorted(collector.tool_names - known_tools):
        violations.append(
            f"the predicate expects a call to tool {name!r}, which does not exist; "
            f"the available tools are {sorted(known_tools)}"
        )
    return violations


def audit_not_satisfied_initially(predicate: str, initial_state: dict) -> list[str]:
    """Reject an end-state predicate that already holds before the agent acts.

    Such a checker scores the task passed for doing nothing, so it measures
    nothing. Only meaningful for end-state checkers: a step-wise checker exists
    precisely because its task's final state can equal its initial state.

    A predicate that raises here (a key the initial state lacks, say) is simply
    not satisfied initially, which is the outcome we want.
    """
    try:
        satisfied = evaluate_predicate(predicate, initial_state)
    except Exception:  # noqa: BLE001 — any failure means "not satisfied initially"
        return []
    if satisfied:
        return [
            "predicate is already true of the initial state, so the task passes without "
            "the agent doing anything; assert the change the instruction asks for"
        ]
    return []


def _build_checker_prompt(
    graph: TaskGraph,
    tools: list[ToolSpec],
    initial_state: dict,
    nl_prompt: str,
    schema: StateSchema | None = None,
) -> str:
    tool_names = [t.name for t in tools]
    graph_summary = [
        {"node_id": n.node_id, "tool_name": n.tool_name, "depends_on": n.depends_on} for n in graph.nodes
    ]
    # The schema goes first and the concrete state second: the predicate has to
    # hold for every state the agent could legally reach, not just this one, so
    # the shape is the binding constraint and the instance is only an example
    # of it. Shown the instance alone, the teacher generalizes from whatever
    # fields this particular state happens to contain.
    schema_block = f"{schema.describe()}\n\n" if schema is not None else ""
    return (
        f"Task instruction given to the agent:\n{nl_prompt}\n\n"
        f"{schema_block}"
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
    *,
    nl_prompt: str,
    schema: StateSchema | None = None,
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
        {
            "role": "user",
            "content": _build_checker_prompt(graph, tools, initial_state, nl_prompt, schema),
        },
    ]
    violations: list[str] = []
    for attempt in range(1, max_attempts + 1):
        result = teacher.chat(messages=messages, max_tokens=500)
        content = result.content or ""
        try:
            payload = extract_json_object(content)
            predicate = payload["executable_predicate"]
            if not isinstance(predicate, str) or not predicate.strip():
                # `null` / "" passes the KeyError guard but is not a predicate;
                # without this it reaches compile() and the TypeError escapes
                # the retry loop, discarding the task instead of re-asking.
                raise ValueError(
                    f"executable_predicate must be a non-empty string, got {predicate!r}"
                )
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
        canonical = collect_canonical_strings(initial_state, nl_prompt)
        violations = audit_no_nl_leak(predicate, canonical_strings=canonical)
        if schema is not None:
            schema_violations = audit_schema_conformance(predicate, schema, tools)
            if schema_violations:
                logger.info("checker violated the %s schema, re-asking: %s",
                            schema.family, "; ".join(schema_violations[:2]))
            violations.extend(schema_violations)
        if not step_wise:
            violations.extend(audit_not_satisfied_initially(predicate, initial_state))
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
                violations.extend(
                    audit_no_nl_leak(step_predicate, root_var="states", canonical_strings=canonical)
                )
                if schema is not None:
                    violations.extend(audit_schema_conformance(step_predicate, schema, tools))
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
