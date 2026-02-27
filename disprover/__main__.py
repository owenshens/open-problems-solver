"""Entry point for the disprover module.

Usage:
    python -m disprover examples/disprover/erdos_1082.json
    python -m disprover examples/disprover/erdos_1082.json --resume runs/erdos_1082/20260218-120000/
    python -m disprover examples/disprover/erdos_1082.json --checkpoint-after-analysis
    python -m disprover examples/disprover/erdos_1082.json --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from disprover.controller import SearchDesignController
from disprover.models import SearchProblem


def load_problem(path: str) -> SearchProblem:
    """Load a SearchProblem from a JSON file.

    Supports two context_documents formats:
      1. Inline strings: "context_documents": ["full text here..."]
      2. File references: "context_documents": ["@path/to/file.md"]
         (paths starting with @ are loaded from disk, relative to the JSON file)
    """
    json_path = Path(path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Problem file not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Resolve file references in context_documents
    if "context_documents" in data:
        resolved = []
        for doc in data["context_documents"]:
            if isinstance(doc, str) and doc.startswith("@"):
                ref_path = json_path.parent / doc[1:]
                if ref_path.exists():
                    resolved.append(ref_path.read_text(encoding="utf-8"))
                else:
                    print(f"Warning: Context document not found: {ref_path}", file=sys.stderr)
                    resolved.append(f"[ERROR: File not found: {doc[1:]}]")
            else:
                resolved.append(doc)
        data["context_documents"] = resolved

    return SearchProblem(**data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterexample Search System — produce a search algorithm document"
    )
    parser.add_argument(
        "problem",
        help="Path to problem JSON file",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="Resume from a previous run's checkpoint directory",
    )
    parser.add_argument(
        "--run-dir",
        metavar="DIR",
        help="Explicit output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--checkpoint-after-analysis",
        action="store_true",
        help="Pause after Phase 2 for user inspection",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no API calls, for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Apply flags
    if args.checkpoint_after_analysis:
        from disprover import config as cfg
        cfg.CHECKPOINT_AFTER_ANALYSIS = True

    if args.mock:
        from shared.agents import set_mock_mode
        set_mock_mode(True)

    # Load problem
    problem = load_problem(args.problem)
    logging.info("Loaded problem: %s", problem.id)
    logging.info("  Statement: %s", problem.statement[:100] + "..." if len(problem.statement) > 100 else problem.statement)
    logging.info("  Context documents: %d", len(problem.context_documents))
    logging.info("  Strategy suggestions: %d", len(problem.strategy_suggestions))
    logging.info("  Prior attempts: %d", len(problem.prior_attempts))

    # Run pipeline
    controller = SearchDesignController(
        problem,
        run_dir=args.run_dir,
        resume_from=args.resume,
    )

    document = controller.run()

    # Report
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Run directory: {controller.run_dir}")
    print(f"Final document: {controller.run_dir / 'FINAL_DOCUMENT.md'}")

    code_path = controller.run_dir / "search_algorithm.py"
    if code_path.exists():
        print(f"Search code: {code_path}")

    print(f"Total API usage: {json.dumps(controller.total_usage, indent=2)}")


if __name__ == "__main__":
    main()
