"""Canonical EnvFactory eval config -- single source of truth for the harnesses.

Copied verbatim from the training/rollout code so eval can faithfully reproduce
the training-time setup:
  - system prompt:  verl/EnvFactory/configs/utils.py  ASSISTANT_SYSTEM_PROMPT
  - tool handling:  verl/EnvFactory/utils/tool_parser.py  Qwen3ToolParser
                    -> flat single-step Hermes function-calling. Tools are rendered
                    into the <tools> block of the system prompt via the Qwen3 chat
                    template (apply_chat_template(tools=...)); the model emits
                    <tool_call>{"name","arguments"}</tool_call>. There is NO
                    model-facing list_tools/call_tool two-step: MCP list_tools is
                    only used server-side to fetch schemas.
  - hyperparameters: paper Appendix F -- temperature 0.7 for thinking models
                    (0 for non-thinking), tensor-parallel 2. Other sampling params
                    follow the Qwen3 generation_config (top_p 0.95, top_k 20).

Usage:
    python envfactory_config.py --print system_prompt
    python envfactory_config.py --print temperature
"""

# --- system prompt (verbatim from verl/EnvFactory/configs/utils.py) ---
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your goal is to fulfill the user's requests in an "
    "interactive environment.\n"
    "At each step, you will receive either the user's request/reply or the tool call "
    "results.\n"
    "- Choose the appropriate tool from the available set and provide complete, valid "
    "parameters.\n"
    "- Strictly follow the format `<server_name>-<tool_name>` when calling a tool.\n"
    "- When you believe the task is completed, provide a direct, concise response to "
    "the user's original request.\n"
)

# --- tool handling ---
# The only mode EnvFactory trained/rolled out with. Kept as a named constant so eval
# scripts can be explicit (and future modes, if ever needed, can be added here).
TOOL_MODE = "hermes_single_step"  # tools in <tools>, output <tool_call>; no MCP 2-step

# --- sampling hyperparameters (paper Appendix F + Qwen3 generation_config) ---
# Thinking models (Qwen3-*) use temperature 0.7; non-thinking use 0.
TEMPERATURE_THINKING = 0.7
TEMPERATURE_NON_THINKING = 0.0
TOP_P = 0.95
TOP_K = 20
TENSOR_PARALLEL = 2  # paper default; a throughput knob (does not change model quality)

_VALUES = {
    "system_prompt": ASSISTANT_SYSTEM_PROMPT,
    "tool_mode": TOOL_MODE,
    "temperature": TEMPERATURE_THINKING,
    "temperature_non_thinking": TEMPERATURE_NON_THINKING,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "tensor_parallel": TENSOR_PARALLEL,
}

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="key", required=True, choices=list(_VALUES))
    args = ap.parse_args()
    # print without a trailing newline so shells can capture it cleanly
    import sys

    sys.stdout.write(str(_VALUES[args.key]))
