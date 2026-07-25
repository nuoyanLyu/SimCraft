"""Layer 2: agreement across multiple independent simulator samples for the
*same* (state, tool_call) pair. Low agreement is a hallucination signal even
before we know what the "right" answer is.
"""

from __future__ import annotations

import difflib
import json
from typing import Any


def _normalize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def pairwise_similarity(a: Any, b: Any) -> float:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def compute_agreement(samples: list[Any]) -> float:
    """Mean pairwise similarity across N>=2 independent simulator samples.

    Returns 1.0 for a single sample (nothing to disagree with — callers that
    always draw >=2 samples should treat a lone sample as a separate signal,
    not blindly trust this value) and 0.0 for an empty list.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return 1.0
    pairs = [
        pairwise_similarity(samples[i], samples[j])
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    return sum(pairs) / len(pairs)
