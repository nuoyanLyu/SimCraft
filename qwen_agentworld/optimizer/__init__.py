"""U1 (optimizer engine choice) is settled: TextGrad is the default, chosen
as the simplest engine to start from. GEPA stays available behind the same
`PlaybookOptimizer` interface for later comparison; TF-GRPO was dropped
entirely (see removal note in `data/check_architecture.md`) since the
checker already supplies an absolute per-trajectory reward, so there is no
group of trajectories to normalize against.
"""

from __future__ import annotations

from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine

DEFAULT_ENGINE = "textgrad"

_ENGINES = {
    "textgrad": TextGradEngine,
    "gepa": GEPAEngine,
}


def build_optimizer(teacher: LLMClient, engine: str = DEFAULT_ENGINE) -> PlaybookOptimizer:
    try:
        engine_cls = _ENGINES[engine]
    except KeyError:
        raise ValueError(f"unknown optimizer engine {engine!r}; choose one of {sorted(_ENGINES)}") from None
    return engine_cls(teacher)
