"""Entry point for the prover module.

Usage:
    python -m prover examples/prover/test_compactness.json
    python -m prover examples/prover/test_compactness.json output/proof_output
    python -m prover examples/prover/test_compactness.json --dry-run
    python -m prover examples/prover/test_compactness.json --mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prover import config
from shared.agents import set_mock_mode
from prover.controller import ProofController
from prover.models import Problem


def load_problem(problem_file: str | Path) -> Problem:
    path = Path(problem_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Problem(**data)


def save_artifact_md(artifact, out_base: Path) -> tuple[Path, Path]:
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

    final_verdict = artifact.final_audit.verdict.value if artifact.final_audit else "N/A"
    final_conf = artifact.final_audit.confidence if artifact.final_audit else None

    md = [f"# Proof Artifact: {artifact.problem_id}", "", f"**Status:** {artifact.status}", ""]
    if final_conf is not None:
        md.append(f"**Final Audit:** {final_verdict} (confidence: {final_conf:.2f})")
    else:
        md.append(f"**Final Audit:** {final_verdict}")

    md += ["", "---", "", artifact.proof_text, "", "---", "", "## Lemma Summary", ""]
    for l in artifact.lemmas:
        md.append(f"### {l.id} -- {l.status.value}")
        md.append("")
        md.append(f"**Statement:** {l.statement}")
        if l.interface:
            md.append("")
            md.append(l.interface)
        md.append("")

    md += ["## Metadata", "", "```json", json.dumps(artifact.metadata, indent=2), "```", ""]

    md_path.write_text("\n".join(md), encoding="utf-8")

    return md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mathematical Proof Workflow")
    parser.add_argument("problem_json", help="Path to a problem JSON file")
    parser.add_argument(
        "output_base", nargs="?", default="output/proof_output",
        help="Output base path (without extension). Default: output/proof_output",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose console output")
    parser.add_argument("--dry-run", action="store_true", help="Only draft the global plan; do not prove lemmas")
    parser.add_argument("--mock", action="store_true", help="Run without API calls (offline)")
    parser.add_argument("--mock-scenario", choices=["clean", "blocked"], default="clean",
                        help="Mock behavior preset")

    args = parser.parse_args()

    if args.mock:
        set_mock_mode(True, scenario=args.mock_scenario)
    else:
        config.require_api_keys(allow_missing=False)

    problem = load_problem(args.problem_json)

    print("=" * 60)
    print("Mathematical Proof Workflow")
    print("=" * 60)
    print(f"\nProblem ID: {problem.id}")
    print(f"Statement: {problem.statement}\n")

    controller = ProofController(problem, verbose=args.verbose)

    if args.dry_run:
        from prover.operators import draft_global_plan
        outline, _ = draft_global_plan(problem, run_dir=controller.run_dir)
        print("\n[DRY RUN] DraftGlobalPlan output:")
        print(outline.model_dump_json(indent=2))
        print(f"\nRun dir: {controller.run_dir}")
        return 0

    artifact = controller.run()

    out_base = Path(args.output_base)
    md_path, json_path = save_artifact_md(artifact, out_base)

    print("\nOutputs written:")
    print(f"  - {md_path}")
    print(f"  - {json_path}")
    print(f"\nRun dir (logs/checkpoints): {controller.run_dir}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Status: {artifact.status}")
    if artifact.final_audit:
        print(f"Final audit: {artifact.final_audit.verdict.value} (confidence: {artifact.final_audit.confidence:.2f})")
    print(
        f"Lemmas: {artifact.metadata.get('lemmas_proved')}/{artifact.metadata.get('lemmas_total')} proved; "
        f"{artifact.metadata.get('lemmas_blocked')} blocked"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
