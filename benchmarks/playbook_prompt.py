"""Render an evolved playbook into the system prompt the benchmark harnesses inject.

This is the knob the whole ID-capability experiment turns on: run a benchmark
once with an empty playbook and once with an evolved one, and the delta is the
playbook's contribution.

It deliberately reuses `simulator_gym.env._build_playbook_context` rather than
re-implementing the formatting. If eval rendered the playbook even slightly
differently from training, any measured gain (or loss) would be confounded by
the prompt difference instead of the playbook content.

Usage:
    python playbook_prompt.py --iteration smoke_test_results/<run>/iteration_4.json
    python playbook_prompt.py --empty          # baseline arm, prints nothing
    python playbook_prompt.py --iteration ... --describe   # human summary to stderr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.core.schemas import Playbook
from qwen_agentworld.simulator_gym.env import _build_playbook_context
from qwen_agentworld.study.checkpoints import empty_playbook, playbook_from_dict


def load_playbook(iteration_file: str | None) -> Playbook:
    if iteration_file is None:
        return empty_playbook()
    data = json.loads(Path(iteration_file).read_text())
    # An iteration record stores the post-optimization snapshot; a bare playbook
    # dump is also accepted so a hand-made playbook can be tested. A checkpoint
    # in the pre-entry format raises instead of rendering nothing, which would
    # turn the playbook arm into a second copy of the baseline.
    return playbook_from_dict(data, source=iteration_file)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--iteration", help="path to iteration_N.json (or a raw playbook JSON)")
    src.add_argument("--empty", action="store_true", help="baseline arm: empty playbook")
    ap.add_argument("--describe", action="store_true", help="print a summary to stderr")
    args = ap.parse_args()

    pb = load_playbook(None if args.empty else args.iteration)
    if args.describe:
        print(
            f"[playbook] v{pb.version}, {len(pb.entries)} entries, "
            f"{pb.word_count} words, tags: {', '.join(pb.tags()) or '-'}",
            file=sys.stderr,
        )
    # No trailing newline: shells capture this with $(...) into an env var.
    sys.stdout.write(_build_playbook_context(pb))


if __name__ == "__main__":
    main()
