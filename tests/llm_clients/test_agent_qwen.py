from types import SimpleNamespace
from unittest.mock import MagicMock

from qwen_agentworld.llm_clients.agent_qwen import DEFAULT_AGENT_MODEL, AgentClient


def _fake_response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_client_reads_url_and_model_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    client = AgentClient(max_retries=1, client=MagicMock())
    assert client.model == DEFAULT_AGENT_MODEL


def test_agent_client_chat_is_not_folded(monkeypatch):
    monkeypatch.setenv("AGENT_URL", "http://localhost:8000/v1")
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.side_effect = [_fake_response()]
    client = AgentClient(max_retries=1, client=fake_openai_client)

    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    client.chat(messages=messages)

    # unlike TeacherClient, no system-role folding: vLLM is assumed to
    # honor the system role normally
    sent_messages = client._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages == messages
