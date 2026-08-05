#!/usr/bin/env python3
"""Idempotently register the EnvFactory model into the BFCL harness, with a
configurable system prompt and (documented) tool-handling mode.

Handler: EnvFactoryQwenFCHandler subclasses BFCL's QwenFCHandler. QwenFCHandler's
hard-coded chat template already matches EnvFactory's training format exactly
(tools in <tools>, output <tool_call>{...}</tool_call>, <tool_response>, <think>)
-- i.e. flat single-step Hermes function-calling, NOT MCP list_tools/call_tool.
Our subclass adds one knob:

  EF_SYSTEM_PROMPT   (env)  If set, injected as the system message so the model
                           sees EnvFactory's exact training system prompt. Unset
                           (default) -> stock BFCL behavior (official-harness run).
  EF_SYSTEM_PROMPT_MODE     "prepend" (default) keeps any existing system content
                           and prepends EnvFactory's; "replace" swaps it.

`model_name` is what BFCL sends as the OpenAI `model=` param -> must equal vLLM
--served-model-name. Serve either checkpoint (step20 / released EnvFactory-4B)
under the same served name to evaluate it.

Run in the benchmark venv (see benchmarks/env.autodl.sh):
    python register_models.py          # apply / re-apply (idempotent)
    python register_models.py --check   # verify only
"""
import argparse
import importlib.util
import os
import re
import sys

MARKER = "ENVFACTORY_EVAL_MODELS"
BEGIN = f"# === {MARKER} (auto-injected by EnvFactory scripts/eval, idempotent) ==="
END = f"# === END {MARKER} ==="

# registry key -> (vLLM --served-model-name, display name). Driven by the same
# env vars run_bfcl.sh exports so the harness can point at any served model;
# the defaults reproduce the original EnvFactory 4B setup from the 79 box.
_REGISTRY_KEY = os.getenv("REGISTRY_KEY", "envfactory-rl-4b")
_SERVED_NAME = os.getenv("SERVED_NAME", "envfactory-eval")
ENVFACTORY_MODELS = {
    _REGISTRY_KEY: (_SERVED_NAME, os.getenv("MODEL_DISPLAY", f"{_SERVED_NAME} (FC, local vLLM)")),
}

MODEL_CONFIG_BLOCK = '''
import os as _ef_os
from overrides import override as _ef_override
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler as _EF_Base


class EnvFactoryQwenFCHandler(_EF_Base):
    """QwenFC handler with a configurable EnvFactory system prompt.

    Tool handling is the stock QwenFC Hermes single-step format (matches
    EnvFactory training). Set EF_SYSTEM_PROMPT to inject EnvFactory's training
    system prompt; leave unset for stock BFCL behavior.
    """

    @_ef_override
    def _format_prompt(self, messages, function):
        sp = _ef_os.getenv("EF_SYSTEM_PROMPT", "").strip()
        if sp:
            mode = _ef_os.getenv("EF_SYSTEM_PROMPT_MODE", "prepend").strip()
            messages = list(messages)
            has_sys = bool(messages) and messages[0].get("role") == "system"
            if mode == "replace":
                body = messages[1:] if has_sys else messages
                messages = [{"role": "system", "content": sp}] + list(body)
            else:  # prepend, keeping any existing system content
                if has_sys:
                    merged = sp + "\\n\\n" + str(messages[0].get("content", ""))
                    messages = [{"role": "system", "content": merged}] + list(messages[1:])
                else:
                    messages = [{"role": "system", "content": sp}] + list(messages)
        return super()._format_prompt(messages, function)

'''


def _locate(fname):
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: bfcl_eval not importable; use $BENCH_PY (benchmarks/env.autodl.sh).")
    path = os.path.join(spec.submodule_search_locations[0], "constants", fname)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return path


def _strip_block(content):
    """Remove any previously injected marker block (old or new comment wording),
    so re-applying picks up handler edits without duplicating registrations."""
    pattern = re.compile(
        r"\n*# === " + re.escape(MARKER) + r".*?# === END " + re.escape(MARKER) + r" ===\n?",
        re.DOTALL,
    )
    return pattern.sub("", content)


def _model_config_block():
    lines = [BEGIN, MODEL_CONFIG_BLOCK.strip()]
    for key, (served, display) in ENVFACTORY_MODELS.items():
        lines.append(
            "MODEL_CONFIG_MAPPING[%r] = ModelConfig(\n"
            "    model_name=%r,\n"
            "    display_name=%r,\n"
            '    url="local-vllm",\n'
            '    org="EnvFactory",\n'
            '    license="apache-2.0",\n'
            "    model_handler=EnvFactoryQwenFCHandler,\n"
            "    input_price=None,\n"
            "    output_price=None,\n"
            "    is_fc_model=True,\n"
            "    underscore_to_dot=False,\n"
            ")" % (key, served, display)
        )
    lines.append(END + "\n")
    return "\n\n" + "\n".join(lines)


def _supported_block():
    keys = ", ".join(repr(k) for k in ENVFACTORY_MODELS)
    return f"\n\n{BEGIN}\nSUPPORTED_MODELS = list(SUPPORTED_MODELS) + [{keys}]\n{END}\n"


def apply():
    for fname, block in (
        ("model_config.py", _model_config_block()),
        ("supported_models.py", _supported_block()),
    ):
        path = _locate(fname)
        content = _strip_block(open(path).read()).rstrip() + "\n"
        with open(path, "w") as f:
            f.write(content + block)
        print(f"[patched] {fname}")
    verify()


def verify():
    for mod in list(sys.modules):
        if mod.startswith("bfcl_eval.constants"):
            del sys.modules[mod]
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

    for k in ENVFACTORY_MODELS:
        if k not in MODEL_CONFIG_MAPPING:
            sys.exit(f"ERROR: {k} not in MODEL_CONFIG_MAPPING")
        if k not in SUPPORTED_MODELS:
            sys.exit(f"ERROR: {k} not in SUPPORTED_MODELS")
        cfg = MODEL_CONFIG_MAPPING[k]
        assert cfg.model_handler.__name__ == "EnvFactoryQwenFCHandler", cfg.model_handler
        assert cfg.model_name == ENVFACTORY_MODELS[k][0]
    print(f"[ok] registered + verified: {list(ENVFACTORY_MODELS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, do not patch")
    args = ap.parse_args()
    verify() if args.check else apply()
