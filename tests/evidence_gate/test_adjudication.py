from unittest.mock import MagicMock

from qwen_agentworld.evidence_gate.adjudication import adjudicate
from qwen_agentworld.llm_clients.base import ChatResult


def make_fake_teacher(reply: str) -> MagicMock:
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content=reply)
    return teacher


def test_adjudicate_accepts_on_accept_reply():
    teacher = make_fake_teacher("ACCEPT looks physically plausible")
    assert adjudicate(teacher, {"a": 1}, "search_docs", {"query": "x"}, {"status": "ok"}) is True


def test_adjudicate_rejects_on_reject_reply():
    teacher = make_fake_teacher("REJECT contradicts prior state")
    assert adjudicate(teacher, {"a": 1}, "search_docs", {"query": "x"}, {"status": "ok"}) is False


def test_adjudicate_defaults_to_reject_on_unparseable_reply():
    teacher = make_fake_teacher("uh, maybe?")
    assert adjudicate(teacher, {"a": 1}, "search_docs", {"query": "x"}, {"status": "ok"}) is False
