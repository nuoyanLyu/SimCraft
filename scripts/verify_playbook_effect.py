"""End-to-end verification that the self-evolved playbook actually helps.

Runs the whole claim, start to finish, and ends with one sentence:

    stage 1  evolve   teacher generates tasks -> agent rolls out in the
                      simulator -> checker judges -> teacher diagnoses ->
                      optimizer emits playbook ops -> playbook accumulates.
                      Artifacts: <out>/evolve/iteration_*.json
    stage 2  indomain fixed held-out eval set, agent held fixed, the ONLY
                      variable is which playbook checkpoint is injected.
                      Artifacts: <out>/indomain/results.json
    stage 3  bfcl     the same two prompts on a real external benchmark the
                      playbook never saw. Artifacts: <out>/bfcl/{baseline,evolved}/
    stage 4  report   preconditions, both axes, one verdict.
                      Artifacts: <out>/report.json

Each stage is skippable and resumable (`--stages 3,4`), because stages 1-3 cost
hours of GPU and stage 4 costs nothing — the decision rule should be re-runnable
against artifacts already on disk without paying for them twice.

The verdict logic lives in `qwen_agentworld/study/` and is unit-tested there.
This file is the plumbing: it decides *what to run*, not *what it means*.

Usage (from the repo root, both vLLM servers up):
    python scripts/verify_playbook_effect.py --out-dir studies/run_0806
    python scripts/verify_playbook_effect.py --out-dir studies/run_0806 --stages 4
    python scripts/verify_playbook_effect.py --out-dir studies/run_0806 --skip-bfcl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD
from qwen_agentworld.playbook_store.store import fingerprint
from qwen_agentworld.study import bfcl as bfcl_io
from qwen_agentworld.study.checkpoints import empty_playbook, iteration_files, load_playbook
from qwen_agentworld.study.verdict import (
    bfcl_axis,
    build_report,
    check_preconditions,
    format_report,
    in_domain_axis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_playbook_effect")

STAGES = ("evolve", "indomain", "bfcl", "report")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/study/test_driver.py)
# --------------------------------------------------------------------------- #


def parse_stages(spec: str) -> set[str]:
    """`--stages` accepts names or 1-based numbers: "2,3" == "indomain,bfcl"."""
    wanted = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token) - 1
            if not 0 <= idx < len(STAGES):
                raise ValueError(f"stage {token} out of range 1..{len(STAGES)}")
            wanted.add(STAGES[idx])
        elif token in STAGES:
            wanted.add(token)
        else:
            raise ValueError(f"unknown stage '{raw}'; expected one of {STAGES} or 1..{len(STAGES)}")
    return wanted


def count_playbook_edits(evolve_dir: str | Path) -> int:
    """How many iterations actually changed the playbook.

    The precondition this feeds exists because of the 2026-07-29 run: four
    iterations, one edit, and a null A/B that read as "the playbook does not
    help" when the real finding was "the loop barely learned anything". Zero
    here voids the study rather than producing a number.
    """
    edits = 0
    for path in iteration_files(evolve_dir):
        record = json.loads(Path(path).read_text())
        before = record.get("playbook_version_before")
        after = record.get("playbook_version_after")
        if before is not None and after is not None and after > before:
            edits += 1
    return edits


def train_task_ids(evolve_dir: str | Path) -> set[str]:
    """Every task the playbook was evolved on."""
    ids: set[str] = set()
    for path in iteration_files(evolve_dir):
        record = json.loads(Path(path).read_text())
        for entry in record.get("tasks", []):
            task_id = (entry.get("task") or {}).get("task_id")
            if task_id:
                ids.add(task_id)
    return ids


def eval_task_ids(results: dict) -> set[str]:
    """Task ids the in-domain A/B actually scored, from its results.json."""
    baseline = (results.get("checkpoints") or {}).get("baseline") or {}
    return set((baseline.get("per_task") or {}).keys())


def pick_arm(results: dict, *, prefer: str) -> tuple[str, dict] | None:
    """Find an arm in an A/B results file whose label contains `prefer`.

    Labels are not always the bare word: `load_checkpoints` collapses
    content-identical checkpoints into a joined label like `mid+final`, so an
    exact lookup would miss the very case that most needs reporting.
    """
    checkpoints = results.get("checkpoints") or {}
    if prefer in checkpoints:
        return prefer, checkpoints[prefer]
    for label, payload in checkpoints.items():
        if prefer in label.split("+"):
            return label, payload
    return None


def bfcl_env(
    *,
    arm: str,
    run_root: Path,
    playbook_file: Path | None,
    base_env: dict | None = None,
) -> dict:
    """Environment for one `run_bfcl.sh` invocation.

    Each arm gets its own `BFCL_RUN_ROOT`: the harness keys results by model
    name, so two arms sharing a root overwrite each other and the second arm
    would silently be compared against itself.
    """
    env = dict(os.environ if base_env is None else base_env)
    env["BFCL_RUN_ROOT"] = str(run_root)
    if playbook_file is None:
        env.pop("PLAYBOOK", None)  # unset == baseline arm
    else:
        env["PLAYBOOK"] = str(playbook_file)
    env.setdefault("EF_PROMPT", "1")
    return env


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_evolve(args, out: Path) -> None:
    import live_smoke_real_sim as evolve

    evolve_dir = out / "evolve"
    argv = [
        "--iterations", str(args.iterations),
        "--tasks-per-iteration", str(args.tasks_per_iteration),
        "--judge-mode", args.judge_mode,
        "--judge-threshold", str(args.judge_threshold),
        "--validation-tasks", str(args.validation_tasks),
        "--validation-reps", str(args.validation_reps),
        "--bank-dir", args.bank_dir,
        "--output-dir", str(evolve_dir),
    ]
    if args.graph_complexity is not None:
        argv += ["--graph-complexity", str(args.graph_complexity)]
    if args.agent_model:
        argv += ["--agent-model", args.agent_model]
    if args.bank_train:
        argv += ["--bank-train", "--screened-train"]
    if args.validation_tasks > 0:
        argv.append("--screened-val")
    logger.info("stage evolve: live_smoke_real_sim %s", " ".join(argv))
    evolve.run(evolve.build_parser().parse_args(argv))


def stage_indomain(args, out: Path) -> None:
    import ab_test

    evolve_dir = out / "evolve"
    files = iteration_files(evolve_dir)
    if not files:
        raise SystemExit(f"stage indomain: no iteration_*.json under {evolve_dir}; run the evolve stage first")

    # Mid-point only when there are enough iterations for it to mean something.
    # With two iterations "mid" and "final" are adjacent and the comparison
    # says nothing about accumulation.
    mid = files[len(files) // 2 - 1].name if len(files) >= 3 else files[-1].name
    argv = [
        "--reps", str(args.reps),
        "--n-tasks", str(args.eval_tasks),
        "--judge-mode", args.judge_mode,
        "--judge-threshold", str(args.judge_threshold),
        "--workers", str(args.workers),
        "--bank-dir", args.bank_dir,
        "--bigrun-dir", str(evolve_dir),
        "--mid-iteration", mid,
        "--final-iteration", files[-1].name,
        "--out-dir", str(out / "indomain"),
    ]
    if args.graph_complexity is not None:
        argv += ["--graph-complexity", str(args.graph_complexity)]
    if args.agent_model:
        argv += ["--agent-model", args.agent_model]
    if args.reuse_eval_tasks:
        argv += ["--reuse-tasks", args.reuse_eval_tasks]
    else:
        argv.append("--screened-only")
    logger.info("stage indomain: ab_test %s", " ".join(argv))
    ab_test.run(ab_test.build_parser().parse_args(argv))


def stage_bfcl(args, out: Path) -> None:
    files = iteration_files(out / "evolve")
    if not files:
        raise SystemExit("stage bfcl: no evolved checkpoint to inject; run the evolve stage first")
    final = files[-1]

    script = REPO_ROOT / "benchmarks/scripts/eval/bfcl/run_bfcl.sh"
    if not script.exists():
        raise SystemExit(f"stage bfcl: {script} not found (the benchmarks tree is gitignored and machine-local)")

    for arm, playbook_file in (("baseline", None), ("evolved", final)):
        run_root = out / "bfcl" / arm
        run_root.mkdir(parents=True, exist_ok=True)
        env = bfcl_env(arm=arm, run_root=run_root, playbook_file=playbook_file)
        if args.bfcl_smoke_n:
            env["SMOKE_N"] = str(args.bfcl_smoke_n)
        logger.info("stage bfcl: arm=%s category=%s root=%s", arm, args.bfcl_category, run_root)
        subprocess.run(["bash", str(script), args.bfcl_category], env=env, check=True, cwd=REPO_ROOT)


def stage_report(args, out: Path) -> int:
    results_path = out / "indomain" / "results.json"
    if not results_path.exists():
        raise SystemExit(f"stage report: {results_path} missing; run the indomain stage first")
    results = json.loads(results_path.read_text())

    baseline_arm = pick_arm(results, prefer="baseline")
    final_arm = pick_arm(results, prefer="final")
    if baseline_arm is None or final_arm is None:
        raise SystemExit(
            f"stage report: {results_path} has arms {list((results.get('checkpoints') or {}).keys())}; "
            "need both a baseline and a final arm"
        )
    baseline_label, baseline = baseline_arm
    final_label, final = final_arm
    if baseline_label == final_label:
        # `load_checkpoints` collapses arms with identical playbook text, so
        # this is the "both arms are the same experiment" case arriving late.
        raise SystemExit(
            f"stage report: baseline and final collapsed into one arm ('{baseline_label}') — "
            "the evolved playbook is textually identical to the empty one"
        )

    in_domain = in_domain_axis(baseline["per_task"], final["per_task"], seed=args.seed)

    bfcl_result = None
    bfcl_notes: list[str] = []
    bfcl_dir = out / "bfcl"
    if (bfcl_dir / "baseline").is_dir() and (bfcl_dir / "evolved").is_dir():
        try:
            base_arm = bfcl_io.load_arm(
                bfcl_dir / "baseline", label="baseline",
                registry_key=args.bfcl_registry_key, category=args.bfcl_category,
            )
            evolved_arm = bfcl_io.load_arm(
                bfcl_dir / "evolved", label="evolved",
                registry_key=args.bfcl_registry_key, category=args.bfcl_category,
            )
            warning = bfcl_io.coverage_warning(base_arm, evolved_arm)
            bfcl_notes = [warning] if warning else []
            bfcl_result = bfcl_axis(
                bfcl_io.paired_entries(base_arm, evolved_arm), seed=args.seed, notes=bfcl_notes
            )
        except bfcl_io.BfclArtifactError as exc:
            # A benchmark that could not be read is *unmeasured*, not neutral.
            # Reporting it as neutral would let a broken harness pass the
            # transfer half of the claim by default.
            logger.warning("stage report: BFCL arms unreadable, transfer left unmeasured: %s", exc)
    else:
        logger.info("stage report: no BFCL arms under %s; transfer left unmeasured", bfcl_dir)

    evolved_playbook = load_playbook(iteration_files(out / "evolve")[-1])
    preconditions = check_preconditions(
        baseline_fingerprint=fingerprint(empty_playbook()),
        evolved_fingerprint=fingerprint(evolved_playbook),
        n_entries=len(evolved_playbook.entries),
        n_playbook_edits=count_playbook_edits(out / "evolve"),
        train_task_ids=train_task_ids(out / "evolve"),
        eval_task_ids=eval_task_ids(results),
        baseline_pass_rate=baseline.get("overall_pass_rate"),
    )

    report = build_report(preconditions, in_domain, bfcl_result)
    payload = report.as_dict()
    payload["arms"] = {
        "baseline": baseline_label,
        "final": final_label,
        "evolved_playbook": {
            "version": evolved_playbook.version,
            "n_entries": len(evolved_playbook.entries),
            "word_count": evolved_playbook.word_count,
            "tags": evolved_playbook.tags(),
        },
    }
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(format_report(report))
    logger.info("wrote %s", out / "report.json")
    # A void study is an infrastructure failure and must not be mistaken for a
    # completed experiment by whatever launched this.
    return 2 if report.verdict == "void" else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stages = parse_stages(args.stages)
    if args.skip_bfcl:
        stages.discard("bfcl")

    if "evolve" in stages:
        stage_evolve(args, out)
    if "indomain" in stages:
        stage_indomain(args, out)
    if "bfcl" in stages:
        stage_bfcl(args, out)
    if "report" in stages:
        return stage_report(args, out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="one directory for the whole study")
    p.add_argument("--stages", default="1,2,3,4", help="stage names or 1-based numbers, comma separated")
    p.add_argument("--skip-bfcl", action="store_true", help="in-domain only; transfer stays unmeasured")

    # stage 1
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--tasks-per-iteration", type=int, default=4)
    p.add_argument("--validation-tasks", type=int, default=8)
    p.add_argument("--validation-reps", type=int, default=2)
    p.add_argument("--bank-train", action="store_true", default=True)
    p.add_argument("--no-bank-train", dest="bank_train", action="store_false")

    # stage 2
    p.add_argument("--eval-tasks", type=int, default=24)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--reuse-eval-tasks", default=None,
                   help="frozen eval set from select_eval_set.py; strongly preferred over drawing fresh")

    # stage 3
    p.add_argument("--bfcl-category", default="simple_python")
    p.add_argument("--bfcl-registry-key", default=os.getenv("REGISTRY_KEY", "qwen3-8b-agentworld"))
    p.add_argument("--bfcl-smoke-n", type=int, default=0, help="0 = full category")

    # shared
    p.add_argument("--graph-complexity", type=int, default=None,
                   help="default: whatever tools/tool_graph_map.json records for --agent-model")
    # One judge for every stage of the study, for the same reason the arms
    # share one: a stage scored differently is not comparable to the others.
    p.add_argument("--judge-mode", default="checker", choices=["checker", "llm", "both"])
    p.add_argument("--judge-threshold", type=float, default=DEFAULT_JUDGE_THRESHOLD)
    p.add_argument("--agent-model", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--bank-dir", default="task_bank")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed; every CI is reproducible from it")
    return p


if __name__ == "__main__":
    sys.exit(main())
