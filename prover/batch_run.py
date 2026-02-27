"""Batch runner for the proof workflow on multiple problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from prover import config
from shared.agents import set_mock_mode
from prover.controller import ProofController
from prover.models import Problem


def load_problem(path: Path) -> Problem:
    return Problem(**json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    p = argparse.ArgumentParser(description="Batch-run the proof workflow on multiple problems")
    p.add_argument("problem_json", nargs="+", help="One or more problem JSON files")
    p.add_argument("--outdir", default="output/batch", help="Where to write artifacts")
    p.add_argument("--mock", action="store_true", help="Offline mock mode")
    p.add_argument("--mock-scenario", choices=["clean", "blocked"], default="clean")
    p.add_argument("--summary", default="output/batch_summary.json", help="Summary JSON path")

    args = p.parse_args()
    console = Console()

    if args.mock:
        set_mock_mode(True, scenario=args.mock_scenario)
    else:
        config.require_api_keys(allow_missing=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for f in args.problem_json:
        path = Path(f)
        problem = load_problem(path)

        controller = ProofController(problem, console=console)
        artifact = controller.run()

        base = outdir / problem.id
        (base.with_suffix(".json")).write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        (base.with_suffix(".md")).write_text(f"# {problem.id}\n\n{artifact.proof_text}\n", encoding="utf-8")

        meta = artifact.metadata or {}
        final_verdict = artifact.final_audit.verdict.value if artifact.final_audit else "N/A"
        rows.append({
            "problem_id": problem.id,
            "status": artifact.status,
            "final_verdict": final_verdict,
            "final_confidence": getattr(artifact.final_audit, "confidence", None),
            "lemmas_total": meta.get("lemmas_total"),
            "lemmas_proved": meta.get("lemmas_proved"),
            "lemmas_blocked": meta.get("lemmas_blocked"),
            "retries_total": meta.get("retries_total"),
            "audit_agree_rate": (meta.get("audit_agreement") or {}).get("agree_rate"),
            "runtime_s": meta.get("runtime_s"),
            "token_usage": meta.get("token_usage"),
            "run_dir": str(controller.run_dir),
        })

    table = Table(title="Batch summary")
    for col in ["problem_id", "status", "final_verdict", "lemmas_proved",
                 "lemmas_total", "lemmas_blocked", "retries_total",
                 "audit_agree_rate", "runtime_s"]:
        table.add_column(col)

    for r in rows:
        table.add_row(
            str(r["problem_id"]), str(r["status"]), str(r["final_verdict"]),
            str(r["lemmas_proved"]), str(r["lemmas_total"]), str(r["lemmas_blocked"]),
            str(r["retries_total"]),
            "" if r["audit_agree_rate"] is None else f"{r['audit_agree_rate']:.2f}",
            "" if r["runtime_s"] is None else f"{r['runtime_s']:.2f}",
        )

    console.print(table)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    console.print(f"\nWrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
