"""Agent rollout against the simulated environment (D2: the simulator is the
"practice gym" the agent tries and fails in — its raw output is never trusted
on its own; `evidence_gate` scores every step before anything downstream
treats it as real).

Two separate LLMClient roles drive one rollout: `agent` picks tool calls
given the task + the current playbook injected as guidance; `simulator`
predicts the next canonical_state given the prior state and the tool call
the agent made. Neither is assumed live here — both are passed in, so tests
and Stage-4-blocked callers can inject mocks.
"""

from __future__ import annotations

import json

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Playbook, Step, Task, ToolCall, ToolSpec, Trajectory
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.ops import render_entries
from qwen_agentworld.tools.state_schema import ObjectSpec, StateSchema, get_schema, neutral_value

_SIMULATOR_SYSTEM_PROMPT = (
    "You simulate the effect of one tool call on an environment's canonical state. "
    "Given the current state and a tool call (name + arguments), predict the resulting state. "
    'Reply with a single JSON object: {"next_state": {...}}. Do not explain your reasoning.'
)


class SimulatorReplyError(RuntimeError):
    """The simulator answered, but not with a state this run can step on."""


def _simulator_system_prompt(schema: StateSchema | None) -> str:
    """The simulator prompt, with the domain's state schema appended.

    Without it the simulator has only the current state to infer the shape
    from, and when that state's collection is empty it has nothing at all: on
    two 2026-07-29 tasks starting from `notes: []` it answered `tag_note` by
    writing `"tag": "client"` instead of appending to `"tags": [...]`, which
    no amount of downstream repair can undo, because nothing in the data says
    which of the two spellings was meant.
    """
    if schema is None:
        return _SIMULATOR_SYSTEM_PROMPT
    return (
        f"{_SIMULATOR_SYSTEM_PROMPT}\n\n{schema.describe()}\n\n"
        "The next_state you produce must match this schema exactly, whatever the "
        "current state happens to look like."
    )


def _build_playbook_context(playbook: Playbook | None) -> str:
    """Render the playbook for the *agent*, grouped by tag.

    Entry ids and helpful/harmful counts are deliberately withheld here even
    though the teacher is shown both: they are bookkeeping about the playbook,
    and telling the agent that a rule has failed before invites it to decide
    which of its own guidance to ignore.
    """
    if playbook is None or not playbook.entries:
        return ""
    return "Guidance from prior experience:\n" + render_entries(playbook)


def _build_agent_system_prompt(task: Task, playbook: Playbook | None) -> str:
    parts = [
        "You are completing a tool-use task. Call tools step by step; when the task is fully "
        "done, stop calling tools and reply with a short confirmation instead.",
    ]
    context = _build_playbook_context(playbook)
    if context:
        parts.append(context)
    return "\n\n".join(parts)


def _parse_tool_arguments(raw: str | None) -> dict:
    """Best-effort parse of a tool call's JSON arguments.

    A model occasionally emits malformed JSON (a truncated string, a trailing
    token, a missing delimiter). One bad tool call should degrade to empty
    arguments — recorded, scoreable, still wrong if the task needed them — not
    crash the entire rollout and discard every other step.
    """
    if not raw:
        return {}
    try:
        parsed = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


ACTION_LOG_KEY = "_action_log"


def _domain_state(state: dict) -> dict:
    """The state minus the action log: what the simulator is asked to predict.

    The log is bookkeeping we own, so showing it to the simulator only spends
    tokens and gives it something to hallucinate edits into.
    """
    if not isinstance(state, dict):
        return state
    return {k: v for k, v in state.items() if k != ACTION_LOG_KEY}


_IDENTITY_KEYS = ("id", "title", "name", "key", "path")


def _empty_like(value: object) -> object:
    """The neutral value of `value`'s type, used to fill a field the simulator
    omitted on an object that has no counterpart in the prior state.
    """
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    if isinstance(value, str):
        return ""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    return None


def _identity(obj: dict) -> tuple[str, object] | None:
    for key in _IDENTITY_KEYS:
        value = obj.get(key)
        if isinstance(value, (str, int, float, bool)):
            return (key, value)
    return None


def _field_defaults(objects: list) -> dict:
    """Union of the field names seen across sibling objects, each mapped to a
    neutral value of the type it was observed to hold.
    """
    defaults: dict = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if defaults.get(key) is None:
                defaults[key] = _empty_like(value)
    return defaults


def _complete_list(prior_list: list, predicted_list: list, spec: ObjectSpec | None = None) -> list:
    defaults = _field_defaults([*prior_list, *predicted_list])
    if spec is not None:
        # The declaration outranks what the neighbouring objects happen to
        # show: a field every sibling also omits is exactly the case the
        # sibling-union inference is blind to, and it is the case that
        # produced the two unrepairable `tag`/`tags` failures.
        for name, token in spec.fields.items():
            defaults.setdefault(name, neutral_value(token))
    if not defaults:
        return predicted_list

    prior_by_identity = {}
    for obj in prior_list:
        if isinstance(obj, dict):
            identity = _identity(obj)
            if identity is not None:
                prior_by_identity[identity] = obj

    completed = []
    for obj in predicted_list:
        if not isinstance(obj, dict):
            completed.append(obj)
            continue
        prior_obj = prior_by_identity.get(_identity(obj)) if _identity(obj) else None
        filled = dict(obj)
        for key, default in defaults.items():
            if key not in filled:
                # An object that already existed keeps whatever it had: the
                # tool call the simulator was asked about did not mention this
                # field, so its prior value is a better guess than a blank.
                filled[key] = prior_obj.get(key, default) if isinstance(prior_obj, dict) else default
        completed.append(filled)
    return completed


def complete_fields(
    prior_state: dict, predicted_state: dict, schema: StateSchema | None = None
) -> dict:
    """Restore fields the simulator silently dropped from its predicted state.

    The simulator is a language model asked to re-emit a whole JSON state, and
    it routinely omits a field it considers irrelevant to the tool call — most
    often on an object it just created (a new note without `tags`). Checkers
    are LLM-synthesized code that subscripts fields directly
    (`'brainstorm' in note['tags'] for note in state['notes']`), so a single
    omission raises KeyError inside the predicate and `judge_checker` scores
    the whole task as not-passed. Measured on the 2026-07-29 runs this decided
    roughly a quarter of all judgments, which contaminates the difficulty
    band, the A/B result and the validation signal at once.

    The completion is deterministic and conservative — it only adds fields
    back, never objects. A note the simulator deleted stays deleted; a note it
    kept but stripped of `tags` gets its prior `tags` back; a note it created
    without `tags` gets the neutral value its siblings' `tags` have. That
    keeps "the agent failed the task" distinguishable from "the simulator
    forgot a key", which is the only distinction the checker cannot make for
    itself.

    With a `schema` the completion stops depending on the data being
    self-describing. Inference from siblings can only restore a field some
    other object still carries, so it is silent in exactly the case that
    caused the two unrepairable 2026-07-29 failures: a collection that starts
    empty, where the first object created is also the only one. The
    declaration covers that case, and it also fixes the *whole collection*
    being absent from a state that never had it.
    """
    if not isinstance(prior_state, dict) or not isinstance(predicted_state, dict):
        return predicted_state

    completed = dict(predicted_state)
    for key, prior_value in prior_state.items():
        if key == ACTION_LOG_KEY:
            continue
        if key not in completed:
            # A whole top-level collection is never dropped on purpose: a tool
            # that empties one produces `[]`, not a missing key.
            completed[key] = prior_value
        elif isinstance(prior_value, list) and isinstance(completed[key], list):
            completed[key] = _complete_list(
                prior_value, completed[key], schema.object_spec_for(key) if schema else None
            )
        elif isinstance(prior_value, dict) and isinstance(completed[key], dict):
            completed[key] = complete_fields(prior_value, completed[key])

    if schema is not None:
        # Declared keys the prior state never had cannot be reached by the loop
        # above, and a key the simulator invented outright is left in place —
        # dropping it here would hide a real simulator defect from the checker.
        for key, entry in schema.entries.items():
            if key in completed and entry.kind == "list" and isinstance(completed[key], list):
                completed[key] = _complete_list([], completed[key], entry.item)
        completed = schema.conform_state(completed)
    return completed


def record_action(prior_state: dict, next_state: dict, tool_call: ToolCall) -> dict:
    """Append this tool call to `next_state`'s action log.

    Deterministic on purpose: the entry is the call the agent actually made,
    which we already know exactly. This is what lets a checker verify a
    read-only step (a search, a listing) that leaves no other trace -- the
    step's only observable consequence is that it happened.
    """
    if not isinstance(next_state, dict):
        # A malformed simulator reply already degrades gracefully downstream;
        # don't turn it into an AttributeError here.
        return next_state
    prior_log = prior_state.get(ACTION_LOG_KEY) if isinstance(prior_state, dict) else None
    log = list(prior_log or [])
    log.append({"tool": tool_call.tool_name, "arguments": tool_call.arguments})
    recorded = {k: v for k, v in next_state.items() if k != ACTION_LOG_KEY}
    recorded[ACTION_LOG_KEY] = log
    return recorded


def _unwrap_next_state(
    payload: dict, state: dict, schema: StateSchema | None, raw: str | None
) -> dict:
    """The predicted state, whether or not the simulator wrapped it.

    The wrapper is asked for in the system prompt, but the model drops it often
    enough to matter: it answers with the bare state, `{"notes": [...]}`. Both
    shapes carry exactly the same information, so rejecting the unwrapped one
    costs a whole task. Worse, `run_iteration`'s per-task guard swallows the
    failure, so a run where the model happened to prefer the bare shape reports
    every task skipped and zero playbook edits -- numerically identical to "the
    playbook does not help".

    A bare reply is recognised by its keys overlapping the state it was asked to
    transform (or the schema's declared keys). That is deliberately narrow: a
    refusal or an explanation shares no keys with the state and still raises.
    """
    inner = payload.get("next_state")
    if isinstance(inner, dict):
        return inner

    known = set(_domain_state(state)) if isinstance(state, dict) else set()
    if schema is not None:
        known |= schema.top_level_keys()
    if known & set(payload):
        return payload

    raise SimulatorReplyError(
        f"simulator reply is neither a wrapped nor a bare state "
        f"(keys: {sorted(payload)}, expected some of: {sorted(known)}); "
        f"raw reply: {(raw or '')[:800]!r}"
    )


def simulate_next_state(
    simulator: LLMClient, state: dict, tool_call: ToolCall, schema: StateSchema | None = None
) -> dict:
    prompt = (
        f"Current state:\n{json.dumps(_domain_state(state), indent=2)}\n\n"
        f"Tool call: {tool_call.tool_name}({json.dumps(tool_call.arguments)})\n\n"
        "Produce the JSON object described in the system prompt."
    )
    result = simulator.chat(
        messages=[
            {"role": "system", "content": _simulator_system_prompt(schema)},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )
    payload = extract_json_object(result.content or "")
    return complete_fields(state, _unwrap_next_state(payload, state, schema, result.content), schema)


def rollout(
    agent: LLMClient,
    simulator: LLMClient,
    task: Task,
    tools: list[ToolSpec],
    playbook: Playbook | None = None,
    max_steps: int = 10,
) -> tuple[Trajectory, dict]:
    """Runs the agent against the simulator for up to `max_steps` tool calls.

    `tools` are the ToolSpec objects for `task.tool_family` (callers look
    these up via `tools/registry.ToolRegistry`; this module doesn't own tool
    lookup, only the rollout loop).

    Returns the raw `Trajectory` (steps have `evidence=None, accepted=False`
    — scoring is `evidence_gate`'s job, not this module's) and the final
    canonical state reached, for the checker to evaluate.
    """
    schema = get_schema(task.tool_family)
    tools_wire = [t.to_wire() for t in tools]
    messages: list[dict] = [
        {"role": "system", "content": _build_agent_system_prompt(task, playbook)},
        {"role": "user", "content": task.natural_language_prompt},
    ]
    state = dict(task.initial_state)
    trajectory = Trajectory(task_id=task.task_id, playbook_version=str(playbook.version) if playbook else "none")

    for step_idx in range(max_steps):
        result = agent.chat(messages=messages, tools=tools_wire or None)
        if not result.tool_calls:
            break

        # Every tool call must carry a stable, non-null id. vLLM's OpenAI
        # server matches each `tool` response back to its `assistant` tool_call
        # by id; some tool-call parsers emit a null id, which then makes the
        # *next* agent turn raise KeyError('id') when the server reloads the
        # conversation. Synthesize one when missing, and reuse the exact same
        # id for the paired tool response so the link stays well-formed.
        call_ids = [tc.id or f"call_{step_idx}_{j}" for j, tc in enumerate(result.tool_calls)]

        messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for cid, tc in zip(call_ids, result.tool_calls)
                ],
            }
        )
        for cid, tc in zip(call_ids, result.tool_calls):
            arguments = _parse_tool_arguments(tc.arguments)
            tool_call = ToolCall(tool_name=tc.name, arguments=arguments)
            prior_state = state
            next_state = record_action(
                prior_state,
                simulate_next_state(simulator, prior_state, tool_call, schema),
                tool_call,
            )
            trajectory.steps.append(
                Step(
                    tool_call=tool_call,
                    simulator_raw_output={"prior_state": prior_state, "next_state": next_state},
                )
            )
            state = next_state
            messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(next_state)})

    return trajectory, state
