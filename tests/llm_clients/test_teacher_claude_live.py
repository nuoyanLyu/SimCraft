"""Real smoke test against the AUTODL Claude relay. Excluded by default (see
conftest.py); run explicitly with `pytest -m live` when you want to confirm
the relay is actually reachable and the model name is still valid.
"""

import pytest

from qwen_agentworld.llm_clients.teacher_claude import TeacherClient


@pytest.mark.live
def test_teacher_client_replies():
    client = TeacherClient(max_retries=1)
    result = client.chat(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    assert result.content is not None
    assert "OK" in result.content
