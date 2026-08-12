"""Is the agent steerable through the playbook channel at all?

The sabotage probe was not a fair test: "never call any tool" contradicts the
base system prompt's "call tools step by step", so losing that fight says
nothing about ordinary guidance. This uses an instruction that *cooperates*
with the base prompt and is still unmistakable in the trace: always open with
list_notes. If the agent will not do even that, no playbook can move the pass
rate and every A/B on this harness is measuring noise.

Measured 2026-07-29 (Qwen3-8B agent, 3 gc=3 tasks): the baseline opened with
list_notes 0/3 times and `list_first` 3/3, so the channel works. Two caveats
worth keeping: `verify_after` reproduced the baseline trace exactly, so
steerability is per-instruction rather than guaranteed; and an earlier probe
that told the agent "never call any tool" changed nothing, which is not
evidence of an unsteerable agent -- it contradicts the base system prompt's
"call tools step by step" and simply loses. Steering procedure is also not the
same as moving the pass rate; that is what the A/B is for.
"""
import sys
from collections import Counter

sys.path.insert(0, "/root/SimCraft")
sys.path.insert(0, "/root/SimCraft/scripts")

from qwen_agentworld.core.schemas import Playbook, PlaybookEntry
from qwen_agentworld.llm_clients.agent_qwen3 import AgentClient
from qwen_agentworld.llm_clients.simulator_qwen_aw import SimulatorClient
from qwen_agentworld.simulator_gym.env import rollout
from qwen_agentworld.teacher.task_bank import TaskBank

import live_smoke_real_sim as smoke


def pb(content: str, tag: str = "precondition-check") -> Playbook:
    return Playbook(version=2, entries=[PlaybookEntry(entry_id="probe", tag=tag, content=content)])


ARMS = {
    "none": None,
    "list_first": pb(
        "Before making any change, first call list_notes to see what already exists. "
        "Always begin with list_notes."
    ),
    "verify_after": pb(
        "After every change you make, call search_notes to confirm the change landed "
        "before moving on.",
        "postcondition-verification",
    ),
}

tasks = TaskBank().draw(smoke.TOOL_FAMILY, 3, 3, split="train")
agent, simulator = AgentClient(model="Qwen3-8B"), SimulatorClient()

for label, playbook in ARMS.items():
    first_tools, all_tools = Counter(), Counter()
    for task in tasks:
        traj, _ = rollout(agent, simulator, task, smoke.NOTES_TOOLS, playbook=playbook)
        names = [s.tool_call.tool_name for s in traj.steps]
        if names:
            first_tools[names[0]] += 1
        all_tools.update(names)
    print(f"[{label:13}] first call: {dict(first_tools)}")
    print(f"{'':16}all calls: {dict(all_tools)}")
