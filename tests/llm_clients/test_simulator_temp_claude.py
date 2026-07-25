from types import SimpleNamespace
from unittest.mock import MagicMock

from qwen_agentworld.llm_clients.simulator_temp_claude import TemporarySimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import DEFAULT_TEACHER_MODEL


def _fake_response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_temporary_simulator_client_defaults_to_teacher_model():
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.side_effect = [_fake_response()]
    client = TemporarySimulatorClient(max_retries=1, client=fake_openai_client)
    assert client.model == DEFAULT_TEACHER_MODEL


def test_temporary_simulator_client_folds_system_like_teacher_client():
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.side_effect = [_fake_response()]
    client = TemporarySimulatorClient(max_retries=1, client=fake_openai_client)

    client.chat(messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])

    sent_messages = client._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": "SYS\n\nUSER"}]
