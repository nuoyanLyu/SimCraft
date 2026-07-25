"""Shared JSON-from-chat-completion parsing, used by every module that asks
an LLM for a single structured JSON object as its reply.
"""

from __future__ import annotations

import json


def extract_json_object(text: str) -> dict:
    text = text.strip()
    # tolerate ```json ... ``` fencing, which chat models reach for even when told not to
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
