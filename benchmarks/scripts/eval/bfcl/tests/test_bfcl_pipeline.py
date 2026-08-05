"""Tests for the BFCL EnvFactory evaluation pipeline.

Run in the `bfcl` conda env:
    cd /home/lvnuoyan/EnvFactory/scripts/eval/bfcl
    python register_models.py            # apply registration first
    pytest tests/test_bfcl_pipeline.py -v

Endpoint tests auto-skip if no vLLM server is reachable, so the pure-logic
tests (parsing / registration) always run.
"""
import os
import sys

import pytest

ENDPOINT = os.getenv("ENDPOINT", "http://localhost:8100/v1")
SERVED_NAME = os.getenv("SERVED_NAME", "envfactory-eval")
REGISTRY_KEY = os.getenv("REGISTRY_KEY", "envfactory-rl-4b")


# ---------------------------------------------------------------------------
# 1) Registration: EnvFactory model is wired to QwenFCHandler
# ---------------------------------------------------------------------------
def test_model_registered():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

    assert REGISTRY_KEY in MODEL_CONFIG_MAPPING, "run register_models.py first"
    assert REGISTRY_KEY in SUPPORTED_MODELS
    cfg = MODEL_CONFIG_MAPPING[REGISTRY_KEY]
    assert cfg.model_handler.__name__ == "EnvFactoryQwenFCHandler"
    assert cfg.model_name == SERVED_NAME  # must match vLLM --served-model-name
    assert cfg.is_fc_model is True


# ---------------------------------------------------------------------------
# 2) Tool-call parsing: BFCL handler parses EnvFactory-format output
#    (<tool_call>{"name":..,"arguments":..}</tool_call>) correctly.
# ---------------------------------------------------------------------------
def _handler():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    cfg = MODEL_CONFIG_MAPPING[REGISTRY_KEY]
    return cfg.model_handler(
        model_name=cfg.model_name,
        temperature=0.7,
        registry_name=REGISTRY_KEY,
        is_fc_model=cfg.is_fc_model,
    )


def test_decode_single_tool_call():
    h = _handler()
    raw = (
        "<think>\nThe user wants the BTC price.\n</think>\n"
        "<tool_call>\n{\"name\": \"CryptoPrice-get_crypto_price\", "
        "\"arguments\": {\"symbol\": \"BTC\"}}\n</tool_call>"
    )
    decoded = h.decode_ast(raw, "Python", has_tool_call_tag=True)
    assert decoded == [{"CryptoPrice-get_crypto_price": {"symbol": "BTC"}}]


def test_decode_multiple_tool_calls():
    h = _handler()
    raw = (
        "<tool_call>\n{\"name\": \"a.b\", \"arguments\": {\"x\": 1}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"c.d\", \"arguments\": {\"y\": \"z\"}}\n</tool_call>"
    )
    decoded = h.decode_ast(raw, "Python", has_tool_call_tag=True)
    assert decoded == [{"a.b": {"x": 1}}, {"c.d": {"y": "z"}}]


def test_decode_execute_format():
    h = _handler()
    raw = "<tool_call>\n{\"name\": \"foo.bar\", \"arguments\": {\"a\": 1, \"b\": \"two\"}}\n</tool_call>"
    calls = h.decode_execute(raw, has_tool_call_tag=True)
    assert calls == ["foo.bar(a=1,b=\x27two\x27)"]


# ---------------------------------------------------------------------------
# 3) Endpoint smoke: vLLM server is up and generates text (skips if down)
# ---------------------------------------------------------------------------
def _endpoint_up():
    import requests

    try:
        r = requests.get(f"{ENDPOINT}/models", timeout=3)
        return r.status_code == 200 and SERVED_NAME in r.text
    except Exception:
        return False


@pytest.mark.skipif(not _endpoint_up(), reason="vLLM endpoint not reachable")
def test_endpoint_completion():
    from openai import OpenAI

    client = OpenAI(base_url=ENDPOINT, api_key="EMPTY")
    resp = client.completions.create(
        model=SERVED_NAME,
        prompt="<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=16,
        temperature=0,
    )
    assert resp.choices[0].text is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
