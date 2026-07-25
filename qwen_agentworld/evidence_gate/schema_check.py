"""Layer 1 of the evidence gate: is the simulator's raw output even
well-formed relative to the tool's declared response schema?

This is deliberately the cheapest, most mechanical check — it should reject
malformed/garbage output before spending agreement or counterfactual-replay
budget on it.
"""

from __future__ import annotations

import json

import jsonschema
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError


def parse_candidate_output(raw_output: str | dict) -> tuple[dict | None, str | None]:
    """Best-effort JSON parse. Returns (parsed, error) — exactly one is None."""
    if isinstance(raw_output, dict):
        return raw_output, None
    try:
        return json.loads(raw_output), None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, str(exc)


def validate_schema(candidate_output: str | dict, response_schema: dict | None) -> tuple[bool, list[str]]:
    """Validate a simulator response against a JSON-schema response contract.

    If ``response_schema`` is None (tool didn't declare an output schema),
    this only checks that the output parses as JSON — see
    data/code-architecture-plan.md's note that ToolSpec currently only
    contracts *inputs*; per-tool output schemas are a Stage-3 task-generation
    concern (checker/tool metadata), not something this layer invents.
    """
    parsed, parse_error = parse_candidate_output(candidate_output)
    if parsed is None:
        return False, [f"output is not valid JSON: {parse_error}"]

    if response_schema is None:
        return True, []

    errors = [e.message for e in jsonschema.Draft202012Validator(response_schema).iter_errors(parsed)]
    return (len(errors) == 0), errors


def is_schema_valid(candidate_output: str | dict, response_schema: dict | None) -> bool:
    ok, _ = validate_schema(candidate_output, response_schema)
    return ok
