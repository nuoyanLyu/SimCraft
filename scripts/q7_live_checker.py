"""Q7 live validation: does the updated checker_synth prompt make the real
Teacher (Claude via AUTODL relay) emit a step_wise_predicate for an inherently
reversible task graph (create-then-delete)?"""
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.core.schemas import TaskGraph, TaskGraphNode, ToolFunctionSpec, ToolSpec
from qwen_agentworld.teacher.checker_synth import synthesize_checker
from qwen_agentworld.teacher.task_generator import instantiate_nl_and_state


def tool(name, desc, params):
    return ToolSpec(function=ToolFunctionSpec(name=name, description=desc, parameters=params), family="mcp_A")

tools = [
    tool("create_note", "Create a new note, returns its id", {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}}),
    tool("delete_note", "Delete a note by id", {"type": "object", "properties": {"id": {"type": "string"}}}),
]
# reversible graph: create then delete -> final observable state == initial
graph = TaskGraph(nodes=[
    TaskGraphNode(node_id="n1", tool_name="create_note"),
    TaskGraphNode(node_id="n2", tool_name="delete_note", depends_on=["n1"]),
])

teacher = TeacherClient()
nl, init = instantiate_nl_and_state(teacher, graph, tools)
print("NL PROMPT:", nl)
print("INIT STATE:", init)
checker = synthesize_checker(teacher, graph, tools, init)
print("\nstep_wise_diagnostics:", checker.step_wise_diagnostics)
print("executable_predicate :", checker.executable_predicate)
print("step_wise_predicate  :", checker.step_wise_predicate)
