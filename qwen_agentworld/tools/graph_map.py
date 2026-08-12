"""Per-model tool-graph size, read from `tool_graph_map.json`.

`sample_task_graph`'s node count used to be a literal in every call site
(`--graph-complexity 3` on four scripts, `OrchestratorConfig.graph_complexity
= 2`), which was fine while there was one agent. It stops being fine as soon
as a second model is screened: the right starting size is a property of the
*agent*, not of the pipeline, because a stronger agent clears short chains and
leaves the difficulty band empty.

The 2026-08-07 screening is the measurement behind that. Task-internal features
predict nothing (`predicate_len` r=-0.024, `initial_state_size` r=-0.019,
`n_clauses` r=+0.028 over 152 tasks), but node count moves the distribution:
band yield went 9% at gc=3 to 28% at gc=4 for Qwen3-8B. So node count is not a
per-task difficulty dial and this module does not pretend it is — it is the
per-model offset that decides where screening starts paying off.

Keeping it in JSON rather than a Python dict is deliberate: adding a model
after a screening run is then a data edit, and the file can carry the evidence
for each number next to the number.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAP_PATH = Path(__file__).with_name("tool_graph_map.json")


@functools.lru_cache(maxsize=8)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize(model: str) -> str:
    """`/root/autodl-tmp/models/Qwen3-8B` and `qwen3-8b` are the same row.

    Serving scripts, CLI flags and `.env` disagree about whether a model is
    named by path or by `--served-model-name`, and a lookup that missed on that
    difference would silently fall back to the default — i.e. generate at the
    wrong size and only show up as a bad band yield hours later.
    """
    return model.strip().rstrip("/").rsplit("/", 1)[-1].lower()


def graph_nodes_for(
    agent_model: str | None,
    tool_family: str | None = None,
    map_path: str | Path | None = None,
) -> tuple[int, int]:
    """`(min_nodes, max_nodes)` to sample a task graph at for this agent.

    Falls back to the file's `default` for an unknown model, with a warning:
    an unmeasured model is a legitimate state (you have to generate *something*
    before you can screen it), but silently using another model's number is how
    a run ends up unexplainable.
    """
    data = _load(str(map_path or _MAP_PATH))
    entry = dict(data["default"])

    if agent_model:
        by_name = {_normalize(k): v for k, v in data.get("models", {}).items()}
        model_entry = by_name.get(_normalize(agent_model))
        if model_entry is None:
            logger.warning(
                "no tool_graph_map entry for agent model %r; using default %s. "
                "Screen this model and add a row to %s.",
                agent_model, entry, _MAP_PATH.name,
            )
        else:
            entry.update({k: v for k, v in model_entry.items() if k in ("min_nodes", "max_nodes")})
            family_entry = (model_entry.get("families") or {}).get(tool_family or "")
            if family_entry:
                entry.update({k: v for k, v in family_entry.items() if k in ("min_nodes", "max_nodes")})

    lo, hi = int(entry["min_nodes"]), int(entry["max_nodes"])
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid node range ({lo}, {hi}) for model {agent_model!r} in {_MAP_PATH}")
    return lo, hi


def graph_complexity_for(
    agent_model: str | None,
    tool_family: str | None = None,
    map_path: str | Path | None = None,
) -> int:
    """The single node count to bank under, for call sites that take one int.

    `min_nodes` rather than a midpoint: the bank buckets by node count, and a
    range that straddles two buckets would split one generation run across two
    pools that then get screened and compared as if they were one.
    """
    return graph_nodes_for(agent_model, tool_family, map_path)[0]


def known_models(map_path: str | Path | None = None) -> list[str]:
    return sorted(_load(str(map_path or _MAP_PATH)).get("models", {}))
