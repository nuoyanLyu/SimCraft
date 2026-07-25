"""Temporary Simulator stand-in: same Claude Sonnet 5 backend as Teacher,
via the AUTODL relay, used only because the real Simulator's weights are
still downloading. Delete this module (and switch callers to a real
vLLM-served `LLMClient`) once the real Simulator is servable — nothing else
in the codebase should depend on this class's existence.

Epistemic caveat: while Teacher and Simulator share one backend, evidence-gate
agreement sampling (`evidence_gate`'s multi-sample consistency check) is not
a meaningful signal — the same model answering its own question consistently
tells you it's self-consistent, not that its predicted state transitions are
correct. Counterfactual/adjudication checks are similarly weaker for the same
reason. Don't read gate-passing evidence from this configuration as
validation of `evidence_gate`'s real-world behavior; it only validates the
gate's *mechanics* (that it fires, scores, and gates as coded), not its
value against an independent, unreliable Simulator.

Reuses `TeacherClient` outright rather than reimplementing anything: it's
the identical backend/relay, including the AUTODL system-role-folding fix.
"""

from __future__ import annotations

from qwen_agentworld.llm_clients.teacher_claude import DEFAULT_TEACHER_MODEL, TeacherClient


class TemporarySimulatorClient(TeacherClient):
    def __init__(self, model: str = DEFAULT_TEACHER_MODEL, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
