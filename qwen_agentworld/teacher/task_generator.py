"""Task generation, task-graph-first (research plan §"任务生成的具体约束"):
sample a controllable tool-call graph *before* asking Claude for any natural
language, so difficulty/coverage are controlled by us, not by what Claude
feels like generating.

Claude's only job here is to flesh out a graph we already committed to into
a natural_language_prompt + initial_state — it does not choose the tool
sequence, and it never sees family-B tools (caller's responsibility: only
pass in tools from the training family).
"""

from __future__ import annotations

import json
import logging
import random

from qwen_agentworld.core.schemas import DifficultyMeta, Task, TaskGraph, TaskGraphNode, ToolSpec
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.teacher.checker_synth import synthesize_checker
from qwen_agentworld.simulator_gym.env import ACTION_LOG_KEY
from qwen_agentworld.tools.families import non_destructive
from qwen_agentworld.tools.state_schema import StateSchema, get_schema

logger = logging.getLogger(__name__)

_MAX_INSTANTIATION_ATTEMPTS = 3


class SchemaViolation(ValueError):
    """The teacher's `initial_state` never conformed to the family schema.

    Raised rather than silently repaired: `conform_state` can add a missing
    field, but it cannot invent the note the task's instruction refers to, and
    a task whose state is wrong in that way is unpassable no matter what the
    agent does. `_generate_tasks_resiliently` already drops a task that raises
    here, which is the correct outcome — one wasted teacher call is much
    cheaper than a permanently-failing task sitting in the bank.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


_SIMULATOR_DOMAIN_CONTEXT = (
    "The simulator that will predict this task's state transitions is trained and evaluated on "
    "agentic tool-use trajectories across seven domains: MCP (an agent calling external "
    "tool/resource APIs — file stores, notes, calendars, CRMs — on a user's behalf), Search "
    "(issuing queries and synthesizing results to satisfy an information need), SWE (editing, "
    "debugging, or testing a codebase to fix an issue or ship a feature), Terminal (running shell "
    "commands to accomplish a system or file-management goal), Android (operating a mobile app's "
    "UI to complete a task), Web (navigating and interacting with a website to complete a task), "
    "and OS (manipulating desktop/OS-level state — files, settings, apps — to complete a task). "
    "Write the task the way a real user would phrase a concrete, goal-directed request to an AI "
    "agent in whichever of these domains the given tools belong to — not an abstract description "
    "of a tool-call sequence. Staying in this style is what keeps the simulator's predictions "
    "reliable."
)

_INSTANTIATION_SYSTEM_PROMPT = (
    "You turn an abstract tool-call graph into a concrete, realistic natural-language task "
    "and a plausible starting environment state, for a fixed set of tools. "
    "You do not choose which tools are called or in what order — that graph is given to you. "
    + _SIMULATOR_DOMAIN_CONTEXT + " "
    "Do not include a solution, reference answer, or hints about how to verify success in the "
    "task text; that is handled separately. "
    "Prefer tasks that leave a durable, observable change in the final environment state (a "
    "change still visible once execution finishes). Avoid net-zero tasks whose correct final "
    "state is identical to the initial state — e.g. create an item then delete it, or set a "
    "value then restore it — because success on those cannot be verified from the end state "
    "alone. If the given tool graph makes a reversible task unavoidable, still phrase a "
    "concrete goal, but leave at least one durable trace (a log entry, a status field, a "
    "renamed/annotated item) so the outcome remains checkable. "
    "Never write a task that requires destroying data the starting state already contains — no "
    "deleting records, removing files, or wiping a field's existing contents. Tasks add, "
    "annotate, reorganize, or update; the destructive tools exist for the agent to decline. "
    "Reply with a single JSON object with exactly two keys: "
    '"natural_language_prompt" (string) and "initial_state" (a JSON object).'
)


def sample_task_graph(
    available_tools: list[ToolSpec],
    min_nodes: int = 2,
    max_nodes: int = 4,
    rng: random.Random | None = None,
) -> TaskGraph:
    """Sample a controllable tool-graph: a linear chain over distinct tools,
    drawn from `available_tools`, with node count in [min_nodes, max_nodes].

    The linear chain is a *scoped decision*, not a placeholder. Branch /
    fail-recovery sampling was on the roadmap as the lever that would make
    tasks harder; the 2026-07-29 screening removed its premise. Across 83
    screened gc=3 tasks the pass rate spanned 0.0-1.0 and no static property
    of a task predicted where it landed (chain length among them, best
    correlation r=-0.21, p=0.053 over six features), and a gc=4 sample came
    out *easier* than gc=3. Graph shape is therefore not a difficulty dial,
    and a richer shape sampler buys variety, not curriculum. Difficulty is
    controlled solely by selecting on measured pass rate
    (`OrchestratorConfig.difficulty_band`).

    So: node count and topology stay fixed here, and adding branches is out
    of scope until something other than difficulty motivates it.
    """
    rng = rng or random.Random()
    # Destructive tools are dropped before sampling, not vetoed afterwards: a
    # graph containing `delete_record` cannot be instantiated under the "never
    # destroy existing data" rule, so it would just burn teacher calls.
    available_tools = non_destructive(available_tools)
    if not available_tools:
        raise ValueError("cannot sample a task graph with zero available tools")

    n = rng.randint(min_nodes, max(min_nodes, min(max_nodes, len(available_tools))))
    chosen = rng.sample(available_tools, k=min(n, len(available_tools)))
    nodes = []
    for i, tool in enumerate(chosen):
        depends_on = [nodes[i - 1].node_id] if i > 0 else []
        nodes.append(TaskGraphNode(node_id=f"n{i+1}", tool_name=tool.name, depends_on=depends_on))
    return TaskGraph(nodes=nodes)


def _build_instantiation_prompt(
    graph: TaskGraph, tools: list[ToolSpec], schema: StateSchema | None = None
) -> str:
    tool_summaries = [
        {"name": t.name, "description": t.function.description, "parameters": t.function.parameters}
        for t in tools
    ]
    graph_summary = [
        {"node_id": n.node_id, "tool_name": n.tool_name, "depends_on": n.depends_on} for n in graph.nodes
    ]
    schema_block = f"{schema.describe()}\n\n" if schema is not None else ""
    return (
        f"Tool graph (execute in this order, respecting depends_on):\n{json.dumps(graph_summary, indent=2)}\n\n"
        f"Available tool definitions:\n{json.dumps(tool_summaries, indent=2)}\n\n"
        f"{schema_block}"
        "Produce the JSON object described in the system prompt."
    )


def instantiate_nl_and_state(
    teacher: LLMClient,
    graph: TaskGraph,
    tools: list[ToolSpec],
    max_attempts: int = _MAX_INSTANTIATION_ATTEMPTS,
    schema: StateSchema | None = None,
) -> tuple[str, dict]:
    """Retries on empty or malformed replies: live testing showed the relay
    occasionally returns empty content for no discernible reason, the same
    failure mode `checker_synth.synthesize_checker` already retries on.

    With a `schema`, a third failure mode is retried too, and this one is not
    transport noise: a state that does not conform to the domain's declared
    shape. It used to pass straight through, and a state missing a field
    (28% of the bank's note objects have no `tags`) is what makes an
    otherwise-correct checker raise KeyError instead of returning False. The
    violations are fed back verbatim as feedback, because they are already
    phrased as the edit to make.
    """
    messages = [
        {"role": "system", "content": _INSTANTIATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_instantiation_prompt(graph, tools, schema)},
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        result = teacher.chat(messages=messages, max_tokens=800)
        try:
            payload = extract_json_object(result.content or "")
            nl_prompt, initial_state = payload["natural_language_prompt"], payload["initial_state"]
        except (ValueError, KeyError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "That reply was empty or not valid JSON. Reply again with the JSON "
                    "object described in the system prompt.",
                }
            )
            continue

        if schema is None:
            return nl_prompt, initial_state
        violations = schema.validate_state(initial_state, ignore_keys=frozenset({ACTION_LOG_KEY}))
        if not violations:
            return nl_prompt, initial_state

        # Logged, not silent: the retry rate is the measurement of how often
        # the teacher would have banked a malformed state, and it is the only
        # place that number is observable once the task itself comes out clean.
        logger.info("initial_state violated the %s schema, re-asking: %s",
                    schema.family, "; ".join(violations[:3]))
        last_error = SchemaViolation(violations)
        messages.append({"role": "assistant", "content": result.content or ""})
        messages.append(
            {
                "role": "user",
                "content": (
                    "That initial_state does not match the canonical state schema: "
                    + "; ".join(violations)
                    + ". Reply again with the same task and a corrected initial_state."
                ),
            }
        )
    raise last_error


def generate_task(
    teacher: LLMClient,
    tools: list[ToolSpec],
    tool_family: str,
    min_nodes: int = 2,
    max_nodes: int = 4,
    rng: random.Random | None = None,
) -> Task:
    """Assembles a full `Task`: graph-first sampling, then NL/state
    instantiation, then checker synthesis — the three teacher calls
    `orchestrator/loop.py` needs per task, wired in the required order
    (checker synthesis needs the graph and initial_state already fixed).
    """
    schema = get_schema(tool_family)
    graph = sample_task_graph(tools, min_nodes=min_nodes, max_nodes=max_nodes, rng=rng)
    nl_prompt, initial_state = instantiate_nl_and_state(teacher, graph, tools, schema=schema)
    # Seed the action log so a checker that quantifies over it sees an empty
    # list in states[0] rather than tripping a KeyError (scored as not-passed).
    initial_state.setdefault(ACTION_LOG_KEY, [])
    checker = synthesize_checker(
        teacher, graph, tools, initial_state, nl_prompt=nl_prompt, schema=schema
    )
    return Task(
        tool_family=tool_family,
        task_graph=graph,
        natural_language_prompt=nl_prompt,
        initial_state=initial_state,
        checker=checker,
        difficulty_meta=DifficultyMeta(graph_complexity=len(graph.nodes)),
    )
