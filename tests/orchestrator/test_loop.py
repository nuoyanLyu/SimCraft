import json
from unittest.mock import MagicMock

from qwen_agentworld.capability_probe.prober import RollingPassRateTracker
from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Diagnosis,
    Playbook,
    Step,
    StepDiagnosis,
    Task,
    TaskGraph,
    TaskGraphNode,
    ToolCall,
    ToolFunctionSpec,
    ToolSpec,
    Trajectory,
)
from qwen_agentworld.evidence_gate.gate import EvidenceGate
from qwen_agentworld.llm_clients.base import ChatResult, ToolCallResult
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.orchestrator.loop import (
    OrchestratorConfig,
    run_iteration,
    score_trajectory,
    stop_criterion,
    trajectory_evidence_accepted,
)
from qwen_agentworld.playbook_store.store import PlaybookStore


def tool(name, family="mcp_A"):
    return ToolSpec(
        function=ToolFunctionSpec(name=name, description=f"{name} tool", parameters={"type": "object"}),
        family=family,
    )


def playbook_with_utility(v):
    return Playbook(validation_utility=v)


# --------------------------------------------------------------------------- #
# score_trajectory
# --------------------------------------------------------------------------- #


def test_score_trajectory_accepts_step_with_consistent_simulator_output():
    gate = EvidenceGate()
    simulator = MagicMock()
    simulator.chat.return_value = ChatResult(content=json.dumps({"next_state": {"a": 1}}))
    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="search_docs", arguments={}),
                simulator_raw_output={"prior_state": {}, "next_state": {"a": 1}},
            )
        ],
    )

    score_trajectory(gate, simulator, trajectory, n_agreement_samples=3)

    step = trajectory.steps[0]
    assert step.accepted is True
    assert step.evidence.agreement_score == 1.0
    assert simulator.chat.call_count == 2  # 2 extra agreement samples drawn beyond the original


def test_score_trajectory_wires_counterfactual_probe_when_state_has_an_untouched_key():
    gate = EvidenceGate()
    simulator = MagicMock()
    # 2 extra agreement samples, then 1 counterfactual re-run under perturbation
    simulator.chat.side_effect = [
        ChatResult(content=json.dumps({"next_state": {"a": 1, "b": "same"}})),
        ChatResult(content=json.dumps({"next_state": {"a": 1, "b": "same"}})),
        ChatResult(content=json.dumps({"next_state": {"a": 1, "b": "same"}})),
    ]
    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="bump_a", arguments={}),
                simulator_raw_output={"prior_state": {"a": 0, "b": "same"}, "next_state": {"a": 1, "b": "same"}},
            )
        ],
    )

    score_trajectory(gate, simulator, trajectory, n_agreement_samples=3)

    step = trajectory.steps[0]
    assert simulator.chat.call_count == 3  # 2 agreement + 1 counterfactual probe
    assert step.evidence.counterfactual_pass is True
    assert step.accepted is True


def test_score_trajectory_rejects_step_when_counterfactual_probe_shifts_a_touched_field():
    gate = EvidenceGate()
    simulator = MagicMock()
    simulator.chat.side_effect = [
        ChatResult(content=json.dumps({"next_state": {"a": 1, "b": "same"}})),
        ChatResult(content=json.dumps({"next_state": {"a": 1, "b": "same"}})),
        ChatResult(content=json.dumps({"next_state": {"a": 99, "b": "same"}})),  # "a" drifted under an irrelevant perturbation
    ]
    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="bump_a", arguments={}),
                simulator_raw_output={"prior_state": {"a": 0, "b": "same"}, "next_state": {"a": 1, "b": "same"}},
            )
        ],
    )

    score_trajectory(gate, simulator, trajectory, n_agreement_samples=3)

    step = trajectory.steps[0]
    assert step.evidence.counterfactual_pass is False
    assert step.accepted is False


def test_score_trajectory_calls_adjudicator_only_when_gate_flags_ambiguous():
    gate = MagicMock()
    ambiguous_evidence = MagicMock(confidence=0.5)
    resolved_evidence = MagicMock(confidence=1.0)
    gate.score.side_effect = [ambiguous_evidence, resolved_evidence]
    gate.needs_adjudication.return_value = True
    gate.accept.return_value = True

    simulator = MagicMock()
    adjudicator = MagicMock()
    adjudicator.chat.return_value = ChatResult(content="ACCEPT looks physically plausible")

    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="search_docs", arguments={}),
                simulator_raw_output={"prior_state": {}, "next_state": {"a": 1}},
            )
        ],
    )

    score_trajectory(gate, simulator, trajectory, n_agreement_samples=1, adjudicator=adjudicator)

    adjudicator.chat.assert_called_once()
    assert gate.score.call_count == 2
    assert trajectory.steps[0].evidence is resolved_evidence
    assert trajectory.steps[0].accepted is True


def test_score_trajectory_skips_adjudication_without_an_adjudicator():
    gate = MagicMock()
    evidence = MagicMock(confidence=0.5)
    gate.score.return_value = evidence
    gate.needs_adjudication.return_value = True
    gate.accept.return_value = False

    simulator = MagicMock()
    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="search_docs", arguments={}),
                simulator_raw_output={"prior_state": {}, "next_state": {"a": 1}},
            )
        ],
    )

    score_trajectory(gate, simulator, trajectory, n_agreement_samples=1, adjudicator=None)

    assert gate.score.call_count == 1
    assert trajectory.steps[0].evidence is evidence


# --------------------------------------------------------------------------- #
# trajectory_evidence_accepted
# --------------------------------------------------------------------------- #


def test_trajectory_evidence_accepted_requires_every_step():
    trajectory = Trajectory(
        task_id="t",
        playbook_version="v1",
        steps=[
            Step(tool_call=ToolCall(tool_name="x", arguments={}), accepted=True),
            Step(tool_call=ToolCall(tool_name="y", arguments={}), accepted=False),
        ],
    )
    assert trajectory_evidence_accepted(trajectory) is False


def test_trajectory_evidence_accepted_false_for_empty_trajectory():
    assert trajectory_evidence_accepted(Trajectory(task_id="t", playbook_version="v1", steps=[])) is False


# --------------------------------------------------------------------------- #
# stop_criterion (U7)
# --------------------------------------------------------------------------- #


def test_stop_criterion_false_with_insufficient_history():
    assert stop_criterion([playbook_with_utility(0.5)], window=3) is False


def test_stop_criterion_true_when_no_improvement_for_window():
    history = [playbook_with_utility(v) for v in [0.5, 0.6, 0.6, 0.6, 0.6]]
    assert stop_criterion(history, window=3) is True


def test_stop_criterion_false_when_still_improving():
    history = [playbook_with_utility(v) for v in [0.5, 0.6, 0.7, 0.8, 0.9]]
    assert stop_criterion(history, window=3) is False


def test_stop_criterion_ignores_unevaluated_playbooks():
    history = [
        Playbook(validation_utility=None),
        playbook_with_utility(0.5),
        Playbook(validation_utility=None),
        playbook_with_utility(0.6),
        playbook_with_utility(0.6),
        playbook_with_utility(0.6),
        playbook_with_utility(0.6),
    ]
    assert stop_criterion(history, window=3) is True


# --------------------------------------------------------------------------- #
# run_iteration: full closed loop, every external role mocked end-to-end
# --------------------------------------------------------------------------- #


class _RecordingOptimizer(PlaybookOptimizer):
    """Trivial real (non-abstract) engine: bumps the version whenever the
    diagnosis implicates anything, so the test can assert the store actually
    advanced without depending on GEPA/TextGrad's own (separately tested)
    prompt/parsing logic.
    """

    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        if not diagnosis.step_diagnoses:
            return []
        return [Playbook(version=current.version + 1, modules=dict(current.modules))]


def test_run_iteration_wires_the_full_closed_loop_with_every_role_mocked():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        # 1) task_generator.instantiate_nl_and_state
        ChatResult(content=json.dumps({"natural_language_prompt": "find X", "initial_state": {"docs": []}})),
        # 2) checker_synth.synthesize_checker
        ChatResult(content=json.dumps({"executable_predicate": "state['docs'] == ['X']", "step_wise_diagnostics": False})),
        # 3) reflection.diagnose
        ChatResult(
            content=json.dumps(
                {
                    "overall_verdict": "success",
                    "summary": "completed the task",
                    "steps": [
                        {
                            "step_id": "s1",
                            "verdict": "correct",
                            "feedback": "found the doc",
                            "suggested_category": "schema_grounding",
                        }
                    ],
                }
            )
        ),
    ]

    agent = MagicMock()
    agent.chat.side_effect = [
        ChatResult(content=None, tool_calls=[ToolCallResult(id="call_1", name="search_docs", arguments="{}")]),
        ChatResult(content="done", tool_calls=[]),
    ]

    simulator = MagicMock()
    simulator.chat.return_value = ChatResult(content=json.dumps({"next_state": {"docs": ["X"]}}))

    playbook_store = PlaybookStore()
    playbook_store.seed(Playbook(version=1))

    tracker = RollingPassRateTracker()
    gate = EvidenceGate()
    config = OrchestratorConfig(tool_family="mcp_A", tasks_per_iteration=1)

    tasks = run_iteration(
        teacher=teacher,
        agent=agent,
        simulator=simulator,
        tools=[tool("search_docs")],
        playbook_store=playbook_store,
        optimizer=_RecordingOptimizer(),
        gate=gate,
        tracker=tracker,
        config=config,
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task.tool_family == "mcp_A"

    # checker passed (final simulated state == ['X']) and got recorded into the tracker
    results = tracker.all_results()
    assert len(results) == 1
    assert results[0].tool_family == "mcp_A"
    assert results[0].pass_rate == 1.0

    # evidence was accepted (consistent simulator output) -> diagnosis ran -> optimizer proposed
    # a candidate -> playbook_store advanced past the seeded version
    assert playbook_store.current.version == 2
    assert len(playbook_store.history) == 2


def test_run_iteration_clamps_graph_complexity_to_tool_pool_size(monkeypatch):
    captured_kwargs = []

    def fake_generate_task(teacher, tools, tool_family, min_nodes, max_nodes):
        captured_kwargs.append({"min_nodes": min_nodes, "max_nodes": max_nodes})
        return Task(
            tool_family=tool_family,
            task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name=tools[0].function.name)]),
            natural_language_prompt="find X",
            initial_state={},
            checker=CheckerSpec(executable_predicate="True"),
            difficulty_meta=DifficultyMeta(graph_complexity=len(tools)),
        )

    monkeypatch.setattr("qwen_agentworld.orchestrator.loop.generate_task", fake_generate_task)

    agent = MagicMock()
    agent.chat.return_value = ChatResult(content="done", tool_calls=[])
    simulator = MagicMock()
    teacher = MagicMock()

    playbook_store = PlaybookStore()
    playbook_store.seed(Playbook(version=1))
    tracker = RollingPassRateTracker()
    gate = EvidenceGate()
    # tool pool has only 1 tool, but the curriculum wants graph_complexity=4
    config = OrchestratorConfig(
        tool_family="mcp_A", tasks_per_iteration=1, graph_complexity=4, max_graph_complexity=4
    )

    run_iteration(
        teacher=teacher,
        agent=agent,
        simulator=simulator,
        tools=[tool("only_tool")],
        playbook_store=playbook_store,
        optimizer=_RecordingOptimizer(),
        gate=gate,
        tracker=tracker,
        config=config,
    )

    # requested complexity was clamped to the tool pool size (1), not the
    # curriculum's raw target (4), so task_generator never has to silently
    # clamp on its own and the tracker's bucket matches what was requested
    assert captured_kwargs == [{"min_nodes": 1, "max_nodes": 1}]


def _always_fail_generate_task(*args, **kwargs):
    raise RuntimeError("simulated persistent empty-content generation failure")


def test_run_iteration_skips_tasks_that_keep_failing_generation(monkeypatch):
    monkeypatch.setattr(
        "qwen_agentworld.orchestrator.loop.generate_task", _always_fail_generate_task
    )
    playbook_store = PlaybookStore()
    playbook_store.seed(Playbook(version=1))
    tracker = RollingPassRateTracker()
    config = OrchestratorConfig(tool_family="mcp_A", tasks_per_iteration=2, task_generation_attempts=2)

    # Must not raise: every task fails generation, all are skipped, iteration survives.
    tasks = run_iteration(
        teacher=MagicMock(),
        agent=MagicMock(),
        simulator=MagicMock(),
        tools=[tool("search_docs")],
        playbook_store=playbook_store,
        optimizer=_RecordingOptimizer(),
        gate=EvidenceGate(),
        tracker=tracker,
        config=config,
    )
    assert tasks == []
    assert tracker.all_results() == []
    assert playbook_store.current.version == 1  # untouched


def test_run_iteration_isolates_a_task_whose_processing_raises(monkeypatch):
    def fake_generate_task(teacher, tools, tool_family, min_nodes, max_nodes):
        return Task(
            tool_family=tool_family,
            task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name=tools[0].function.name)]),
            natural_language_prompt="do X",
            initial_state={"docs": []},
            checker=CheckerSpec(executable_predicate="state.get('docs') == ['X']"),
            difficulty_meta=DifficultyMeta(graph_complexity=1),
        )

    monkeypatch.setattr("qwen_agentworld.orchestrator.loop.generate_task", fake_generate_task)

    agent = MagicMock()
    agent.chat.side_effect = RuntimeError("rollout blew up")  # processing fails for every task

    playbook_store = PlaybookStore()
    playbook_store.seed(Playbook(version=1))
    tracker = RollingPassRateTracker()
    config = OrchestratorConfig(tool_family="mcp_A", tasks_per_iteration=2)

    # Two tasks generate fine but both fail during processing; run_iteration must
    # swallow each failure and still return the generated tasks without raising.
    tasks = run_iteration(
        teacher=MagicMock(),
        agent=agent,
        simulator=MagicMock(),
        tools=[tool("search_docs")],
        playbook_store=playbook_store,
        optimizer=_RecordingOptimizer(),
        gate=EvidenceGate(),
        tracker=tracker,
        config=config,
    )
    assert len(tasks) == 2
    assert tracker.all_results() == []  # nothing recorded, both skipped
    assert playbook_store.current.version == 1
