"""Layer 3: counterfactual replay. Perturb a field the checker doesn't care
about and re-run the simulator; the invariant (post-condition-relevant)
fields of the output should not change. If they do, the simulator's output
for this step is not tracking causal structure — reject it.
"""

from __future__ import annotations

import copy
from typing import Any

_MISSING = object()


def get_path(obj: Any, dotted_path: str) -> Any:
    node = obj
    for part in dotted_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def counterfactual_pass(
    original_output: dict,
    perturbed_output: dict,
    invariant_fields: list[str],
) -> bool:
    """True iff every ``invariant_fields`` path is identical between the two
    outputs. An empty ``invariant_fields`` list trivially passes (nothing to
    check) — callers should treat that as "inconclusive", not "verified".
    """
    return all(
        get_path(original_output, field) == get_path(perturbed_output, field)
        for field in invariant_fields
    )


def _mutate_value(value: Any) -> Any:
    """Structurally-safe perturbation: changes `value` without changing its
    type, so a re-serialized state stays schema-shaped for the simulator.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " (counterfactual probe)"
    if isinstance(value, list):
        return [*value, "__counterfactual_probe__"]
    if isinstance(value, dict):
        return {**value, "__counterfactual_probe__": True}
    return "__counterfactual_probe__"


def build_counterfactual_probe(
    prior_state: dict, next_state: dict
) -> tuple[dict, list[str]] | None:
    """Builds a perturbed copy of `prior_state` plus the top-level keys this
    tool call actually changed (`invariant_fields`), for Layer 3.

    We don't know which state fields are causally relevant to a given tool
    call without domain-specific schema knowledge, so instead of guessing we
    read it off the simulator's own (untampered) prior_state/next_state pair:
    keys that differ between the two are what this call touched, and are
    exactly the fields that should stay identical if we perturb something the
    call *didn't* touch and re-run the simulator. Keys that didn't change are
    safe to perturb.

    Returns `None` when there's no safe key to perturb — e.g. `prior_state`
    has a single top-level key, or the tool call touched every key — in which
    case callers should treat this step as "inconclusive" for the
    counterfactual leg, same as omitting counterfactual_output/invariant_fields
    entirely.
    """
    if not isinstance(prior_state, dict) or not isinstance(next_state, dict):
        return None

    ordered_keys = list(prior_state) + [k for k in next_state if k not in prior_state]
    touched = [k for k in ordered_keys if prior_state.get(k, _MISSING) != next_state.get(k, _MISSING)]
    untouched = [k for k in prior_state if k not in touched]
    if not touched or not untouched:
        return None

    perturb_key = untouched[0]
    perturbed_state = copy.deepcopy(prior_state)
    perturbed_state[perturb_key] = _mutate_value(perturbed_state[perturb_key])
    return perturbed_state, touched
