from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError

from qwen_agentworld.llm_clients.base import LLMClient


def _fake_response(content="hello", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_tool_call(id_="call_1", name="search_docs", arguments='{"query": "x"}'):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def make_client(create_side_effect) -> LLMClient:
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.side_effect = create_side_effect
    return LLMClient(
        base_url="http://unused",
        api_key="unused",
        model="test-model",
        max_retries=3,
        retry_backoff_seconds=0.0,
        client=fake_openai_client,
    )


def test_chat_returns_content_and_parsed_tool_calls():
    client = make_client([_fake_response(content=None, tool_calls=[_fake_tool_call()])])
    result = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_docs"
    assert result.tool_calls[0].arguments == '{"query": "x"}'


def test_chat_retries_transient_errors_then_succeeds():
    conn_error = APIConnectionError(request=httpx.Request("POST", "http://test"))
    client = make_client([conn_error, conn_error, _fake_response(content="ok")])
    result = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert client._client.chat.completions.create.call_count == 3


def test_chat_raises_after_exhausting_retries():
    conn_error = APIConnectionError(request=httpx.Request("POST", "http://test"))
    client = make_client([conn_error, conn_error, conn_error])
    with pytest.raises(APIConnectionError):
        client.chat(messages=[{"role": "user", "content": "hi"}])
    assert client._client.chat.completions.create.call_count == 3
