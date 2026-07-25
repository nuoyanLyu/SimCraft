from unittest.mock import MagicMock

import pytest

from qwen_agentworld.optimizer import DEFAULT_ENGINE, build_optimizer
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine


def test_default_engine_is_textgrad():
    assert DEFAULT_ENGINE == "textgrad"


def test_build_optimizer_defaults_to_textgrad():
    optimizer = build_optimizer(MagicMock())
    assert isinstance(optimizer, TextGradEngine)


def test_build_optimizer_can_select_gepa():
    optimizer = build_optimizer(MagicMock(), engine="gepa")
    assert isinstance(optimizer, GEPAEngine)


def test_build_optimizer_rejects_unknown_engine():
    with pytest.raises(ValueError):
        build_optimizer(MagicMock(), engine="tf_grpo")
