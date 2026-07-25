"""Environment/config loading, centralized so no module reads .env directly.

Every external endpoint (Teacher relay, Simulator serving, Agent serving)
is resolved through here, so swapping a provider (U1/U2/U6 in
data/design-decisions.md) never means hunting through business logic for
hardcoded values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(f"required environment variable '{key}' is not set (checked {_REPO_ROOT / '.env'})")
    return value
