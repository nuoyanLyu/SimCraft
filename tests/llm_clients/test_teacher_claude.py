from types import SimpleNamespace
from unittest.mock import MagicMock

from qwen_agentworld.llm_clients.teacher_claude import TeacherClient, _fold_system_into_first_user


def _fake_response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(create_side_effect=None) -> TeacherClient:
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.side_effect = create_side_effect or [_fake_response()]
    return TeacherClient(max_retries=1, client=fake_openai_client)


def test_fold_system_into_first_user_merges_and_drops_system_role():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    folded = _fold_system_into_first_user(messages)
    assert folded == [{"role": "user", "content": "SYS\n\nUSER"}]


def test_fold_system_into_first_user_prepends_when_no_user_message():
    messages = [{"role": "system", "content": "SYS"}]
    folded = _fold_system_into_first_user(messages)
    assert folded == [{"role": "user", "content": "SYS"}]


def test_fold_system_into_first_user_is_noop_without_system_role():
    messages = [{"role": "user", "content": "USER"}]
    assert _fold_system_into_first_user(messages) == messages


def test_teacher_chat_sends_folded_messages_to_backend():
    client = make_client()
    client.chat(messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
    sent_messages = client._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": "SYS\n\nUSER"}]
