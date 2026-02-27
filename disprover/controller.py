"""Controller for the counterexample search system.

Orchestrates the 6-phase deliberation pipeline:
  Phase 1: Literature Research
  Phase 2: Mathematical Structural Analysis (with review loop)
  Phase 3: Algorithm Design
  Phase 4: Multi-Round Review (correctness, optimality, adversarial)
  Phase 5: Complexity Estimation & Feasibility
  Phase 6: Final Document Assembly

Bug fixes relative to v3.0 proposal:
  1. Phase 2 review loop: analysis is set in phase_outputs BEFORE the
     review loop starts, and updated after each revision.
  2. Feasibility check result: handled properly --- if infeasible,
     the algorithm is revised with tighter constraints.
  3. revise_document: operator is defined and called correctly.
  4. Phase output serialization: all phase outputs are stored as strings
     in phase_outputs (not typed objects).
  5. Review version tracking: each review records which algorithm version
     it reviewed.
  6. MAX_REVISION_ROUNDS: used correctly --- each review round can trigger
     up to MAX_REVISION_ROUNDS revise-then-re-review cycles.
  7. Checkpointing: phase_outputs persisted to disk after each phase.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from disprover import config as cfg
from shared.utils import ensure_dir, merge_usage, now_timestamp
from disprover.models import ReviewResult, SearchProblem
from disprover.operators import (
    adversarial_review,
    analyze_structure,
    assemble_document,
    audit_literature,
    check_feasibility,
    correctness_review,
    design_algorithm,
    estimate_complexity,
    final_review,
    literature_research,
    optimality_review,
    process_literature,
    review_analysis,
    revise_algorithm,
    revise_analysis,
    revise_document,
    revise_literature,
)

logger = logging.getLogger(__name__)


class SearchDesignController:
    """Orchestrates the 6-phase counterexample search deliberation pipeline."""

    def __init__(
        self,
        problem: SearchProblem,
        *,
        run_dir: Optional[str | Path] = None,
        resume_from: Optional[str | Path] = None,
    ):
        self.problem = problem
        self.total_usage: dict[str, int] = {}

        # Set up run directory
        if run_dir:
            self.run_dir = ensure_dir(Path(run_dir))
        else:
            root = ensure_dir(cfg.RUNS_DIR)
            self.run_dir = ensure_dir(root / problem.id / now_timestamp())

        # Phase outputs: all stored as strings for prompt injection.
        # Keys: "literature", "analysis", "algorithm", "reviews",
        #        "complexity", "document"
        self.phase_outputs: dict[str, str] = {}

        # Track algorithm versions for review audit trail
        self._algorithm_version = 0

        # Resume from checkpoint if requested
        if resume_from:
            self._load_checkpoint(Path(resume_from))

    # ====================================================================
    # Main entry point
    # ====================================================================

    def run(self) -> str:
        """Run the full 6-phase pipeline. Returns the final document text."""
        start = time.time()
        logger.info("Starting counterexample search for %s", self.problem.id)
        logger.info("Run directory: %s", self.run_dir)

        # Save problem input
        (self.run_dir / "problem.json").write_text(
            self.problem.model_dump_json(indent=2), encoding="utf-8"
        )

        # ── Phase 1: Literature Research ──────────────────────────────
        if "literature" not in self.phase_outputs:
            self._run_phase1()

        # ── Phase 1.5: Literature Processing ─────────────────────────
        if "processed_literature" not in self.phase_outputs:
            self._run_phase1_5()

        # ── Phase 2: Structural Analysis ──────────────────────────────
        if "analysis" not in self.phase_outputs:
            self._run_phase2()

        # Optional checkpoint: pause for user inspection
        if cfg.CHECKPOINT_AFTER_ANALYSIS:
            logger.info(
                "CHECKPOINT: Structural analysis complete. "
                "Review %s/structural_analysis.md before continuing. "
                "Re-run with resume_from='%s' to continue.",
                self.run_dir,
                self.run_dir,
            )
            self._save_checkpoint()
            return self.phase_outputs.get("analysis", "")

        # ── Phase 3: Algorithm Design ─────────────────────────────────
        if "algorithm" not in self.phase_outputs:
            self._run_phase3()

        # ── Phase 4: Multi-Round Review ───────────────────────────────
        if "reviews" not in self.phase_outputs:
            self._run_phase4()

        # ── Phase 5: Complexity Estimation ────────────────────────────
        if "complexity" not in self.phase_outputs:
            self._run_phase5()

        # ── Phase 6: Final Document Assembly ──────────────────────────
        if "document" not in self.phase_outputs:
            self._run_phase6()

        elapsed = time.time() - start
        logger.info("Pipeline complete in %.1f seconds", elapsed)

        # Save metadata
        self._save_metadata(elapsed)

        return self.phase_outputs["document"]

    # ====================================================================
    # Phase implementations
    # ====================================================================

    def _run_phase1(self) -> None:
        """Phase 1: Literature Research."""
        logger.info("Phase 1: Literature Research")

        text, usage = literature_research(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        # Store as string (BUG FIX #4: always store strings, not objects)
        self.phase_outputs["literature"] = text

        # Persist
        (self.run_dir / "literature_survey.md").write_text(text, encoding="utf-8")
        self._save_checkpoint()
        logger.info("Phase 1 complete")

    def _run_phase1_5(self) -> None:
        """Phase 1.5: Literature Processing.

        Processes raw literature survey into structured, categorized,
        confidence-scored form. Includes an audit loop to verify
        completeness and accuracy.
        """
        logger.info("Phase 1.5: Literature Processing")

        # Initial processing
        processed_text, usage = process_literature(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)
        self.phase_outputs["processed_literature"] = processed_text

        # Audit loop (mirrors Phase 2's analysis review pattern)
        for audit_round in range(cfg.MAX_LITERATURE_REVIEWS):
            logger.info(
                "Phase 1.5: Literature audit round %d/%d",
                audit_round + 1,
                cfg.MAX_LITERATURE_REVIEWS,
            )

            review, usage = audit_literature(
                self.problem, self.phase_outputs, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)

            # Save audit review
            ensure_dir(self.run_dir / "reviews")
            review_path = self.run_dir / "reviews" / f"literature_audit_r{audit_round + 1}.md"
            review_path.write_text(review.format_for_prompt(), encoding="utf-8")

            if not review.has_blocking_issues():
                logger.info("Phase 1.5: Literature audit passed")
                break

            logger.info("Phase 1.5: Literature audit found issues, revising")
            processed_text, usage = revise_literature(
                self.problem, self.phase_outputs, review, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)
            self.phase_outputs["processed_literature"] = processed_text

        # Persist
        (self.run_dir / "processed_literature.json").write_text(
            processed_text, encoding="utf-8"
        )

        # Also write human-readable version (best-effort)
        try:
            from disprover.models import ProcessedLiterature
            proc = ProcessedLiterature.model_validate_json(processed_text)
            (self.run_dir / "processed_literature.md").write_text(
                proc.full_text(), encoding="utf-8"
            )
        except Exception:
            pass

        self._save_checkpoint()
        logger.info("Phase 1.5 complete")

    def _run_phase2(self) -> None:
        """Phase 2: Structural Analysis with review loop.

        BUG FIX #1: phase_outputs["analysis"] is set BEFORE the review loop
        and updated after each revision, so the reviewer can see the analysis.
        """
        logger.info("Phase 2: Structural Analysis")

        # Initial analysis
        analysis_text, usage = analyze_structure(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        # BUG FIX #1: Set analysis in phase_outputs BEFORE review loop
        self.phase_outputs["analysis"] = analysis_text

        # Review loop
        for review_round in range(cfg.MAX_ANALYSIS_REVIEWS):
            logger.info("Phase 2: Analysis review round %d/%d", review_round + 1, cfg.MAX_ANALYSIS_REVIEWS)

            review, usage = review_analysis(
                self.problem, self.phase_outputs, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)

            # Save review
            ensure_dir(self.run_dir / "reviews")
            (self.run_dir / "reviews" / f"analysis_review_r{review_round + 1}.md").write_text(
                review.format_for_prompt(), encoding="utf-8"
            )

            if not review.has_blocking_issues():
                logger.info("Phase 2: Analysis review passed")
                break

            logger.info("Phase 2: Analysis review found issues, revising")
            analysis_text, usage = revise_analysis(
                self.problem, self.phase_outputs, review, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)

            # BUG FIX #1: Update analysis in phase_outputs after revision
            self.phase_outputs["analysis"] = analysis_text

        # Persist final analysis
        (self.run_dir / "structural_analysis.md").write_text(
            self.phase_outputs["analysis"], encoding="utf-8"
        )
        self._save_checkpoint()
        logger.info("Phase 2 complete")

    def _run_phase3(self) -> None:
        """Phase 3: Algorithm Design."""
        logger.info("Phase 3: Algorithm Design")

        algorithm_text, usage = design_algorithm(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        self.phase_outputs["algorithm"] = algorithm_text
        self._algorithm_version = 1

        # Persist
        (self.run_dir / "algorithm_v1.md").write_text(algorithm_text, encoding="utf-8")
        self._save_checkpoint()
        logger.info("Phase 3 complete")

    def _run_phase4(self) -> None:
        """Phase 4: Multi-Round Review.

        Three review rounds: correctness, optimality, adversarial.
        Each round can trigger up to MAX_REVISION_ROUNDS
        revise-then-re-review cycles (BUG FIX #6).

        BUG FIX #5: Reviews track which algorithm version was reviewed.
        """
        logger.info("Phase 4: Multi-Round Review")
        ensure_dir(self.run_dir / "reviews")

        review_rounds = [
            ("correctness", correctness_review),
            ("optimality", optimality_review),
            ("adversarial", adversarial_review),
        ]

        all_reviews: list[str] = []

        for round_name, review_fn in review_rounds:
            logger.info("Phase 4: %s review", round_name)

            # BUG FIX #6: Each review round allows up to MAX_REVISION_ROUNDS cycles
            for revision_attempt in range(cfg.MAX_REVISION_ROUNDS + 1):
                review, usage = review_fn(
                    self.problem, self.phase_outputs, run_dir=self.run_dir
                )
                self.total_usage = merge_usage(self.total_usage, usage)

                # BUG FIX #5: Track which version was reviewed
                version_label = f"algorithm v{self._algorithm_version}"
                review_text = (
                    f"=== {round_name.upper()} REVIEW "
                    f"(reviewed {version_label}) ===\n"
                    f"{review.format_for_prompt()}"
                )

                # Save individual review
                (self.run_dir / "reviews" / f"round_{round_name}_attempt{revision_attempt + 1}.md").write_text(
                    review_text, encoding="utf-8"
                )

                if not review.has_blocking_issues():
                    logger.info("Phase 4: %s review passed", round_name)
                    all_reviews.append(review_text)
                    break

                if revision_attempt < cfg.MAX_REVISION_ROUNDS:
                    logger.info(
                        "Phase 4: %s review found issues, revising (attempt %d/%d)",
                        round_name,
                        revision_attempt + 1,
                        cfg.MAX_REVISION_ROUNDS,
                    )
                    algorithm_text, usage = revise_algorithm(
                        self.problem, self.phase_outputs, review, run_dir=self.run_dir
                    )
                    self.total_usage = merge_usage(self.total_usage, usage)

                    self.phase_outputs["algorithm"] = algorithm_text
                    self._algorithm_version += 1

                    # Persist revised algorithm
                    (self.run_dir / f"algorithm_v{self._algorithm_version}.md").write_text(
                        algorithm_text, encoding="utf-8"
                    )
                else:
                    # Max revisions reached, record the review and move on
                    logger.warning(
                        "Phase 4: %s review still has issues after %d revisions, moving on",
                        round_name,
                        cfg.MAX_REVISION_ROUNDS,
                    )
                    all_reviews.append(
                        review_text + "\n[NOTE: Max revision rounds reached. Unresolved issues remain.]"
                    )

        # Store combined reviews
        self.phase_outputs["reviews"] = "\n\n".join(all_reviews)

        # Persist final algorithm
        (self.run_dir / "algorithm_final.md").write_text(
            self.phase_outputs["algorithm"], encoding="utf-8"
        )
        self._save_checkpoint()
        logger.info("Phase 4 complete")

    def _run_phase5(self) -> None:
        """Phase 5: Complexity Estimation & Feasibility Check.

        BUG FIX #2: Feasibility check result is properly handled.
        If infeasible, the algorithm is revised with tighter constraints.
        """
        logger.info("Phase 5: Complexity Estimation")

        complexity_text, usage = estimate_complexity(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)
        self.phase_outputs["complexity"] = complexity_text

        (self.run_dir / "complexity_estimate.md").write_text(
            complexity_text, encoding="utf-8"
        )

        # BUG FIX #2: Handle feasibility check result
        logger.info("Phase 5: Feasibility check")
        feasibility, usage = check_feasibility(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        (self.run_dir / "feasibility_check.md").write_text(
            feasibility.format_for_prompt(), encoding="utf-8"
        )

        if feasibility.overall_verdict == "redesign":
            logger.warning(
                "Phase 5: Algorithm is computationally infeasible! "
                "Injecting feasibility warning into reviews."
            )
            # Add feasibility warning to reviews and algorithm
            warning = (
                "\n\n=== FEASIBILITY WARNING ===\n"
                "The complexity estimate indicates this algorithm is computationally "
                "infeasible within the stated constraints. The following issues were found:\n"
                + feasibility.format_for_prompt()
                + "\n\nThe algorithm may need to be scaled down or restructured."
            )
            self.phase_outputs["reviews"] = self.phase_outputs.get("reviews", "") + warning

            # Attempt to revise algorithm for feasibility
            logger.info("Phase 5: Revising algorithm for feasibility")
            algorithm_text, usage = revise_algorithm(
                self.problem, self.phase_outputs, feasibility, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)

            self.phase_outputs["algorithm"] = algorithm_text
            self._algorithm_version += 1
            (self.run_dir / f"algorithm_v{self._algorithm_version}_feasibility_fix.md").write_text(
                algorithm_text, encoding="utf-8"
            )

            # Re-estimate complexity after revision
            complexity_text, usage = estimate_complexity(
                self.problem, self.phase_outputs, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)
            self.phase_outputs["complexity"] = complexity_text

        elif feasibility.has_blocking_issues():
            logger.warning(
                "Phase 5: Feasibility check found issues (verdict: %s). "
                "Adding to review transcript.",
                feasibility.overall_verdict,
            )
            self.phase_outputs["reviews"] = (
                self.phase_outputs.get("reviews", "")
                + "\n\n=== FEASIBILITY CHECK ===\n"
                + feasibility.format_for_prompt()
            )

        self._save_checkpoint()
        logger.info("Phase 5 complete")

    def _run_phase6(self) -> None:
        """Phase 6: Final Document Assembly.

        BUG FIX #3: revise_document operator is properly defined and called.
        """
        logger.info("Phase 6: Final Document Assembly")

        document_text, usage = assemble_document(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        # Store the assembled document as the algorithm for the final review
        # (the final_review template uses {algorithm_document} to hold the doc)
        self.phase_outputs["document"] = document_text
        # Also put it in algorithm_document slot for the final review template
        saved_algo = self.phase_outputs.get("algorithm", "")
        self.phase_outputs["algorithm"] = document_text

        # Final review
        logger.info("Phase 6: Final review")
        review, usage = final_review(
            self.problem, self.phase_outputs, run_dir=self.run_dir
        )
        self.total_usage = merge_usage(self.total_usage, usage)

        # Restore algorithm
        self.phase_outputs["algorithm"] = saved_algo

        ensure_dir(self.run_dir / "reviews")
        (self.run_dir / "reviews" / "final_review.md").write_text(
            review.format_for_prompt(), encoding="utf-8"
        )

        if review.has_blocking_issues():
            # BUG FIX #3: revise_document is defined and called
            logger.info("Phase 6: Final review found issues, revising document")

            # For document revision, put the assembled document in the algorithm slot
            # AND inject the source algorithm (pre-assembly) so revision can rebuild
            # full sections from the authoritative 170KB+ algorithm rather than
            # producing truncated patches.
            self.phase_outputs["algorithm"] = document_text
            self.phase_outputs["_source_algorithm"] = saved_algo
            document_text, usage = revise_document(
                self.problem, self.phase_outputs, review, run_dir=self.run_dir
            )
            self.total_usage = merge_usage(self.total_usage, usage)
            self.phase_outputs.pop("_source_algorithm", None)
            self.phase_outputs["algorithm"] = saved_algo
            self.phase_outputs["document"] = document_text

        # Write final outputs
        (self.run_dir / "FINAL_DOCUMENT.md").write_text(document_text, encoding="utf-8")

        # Extract Python code from the document
        code = self._extract_python_code(document_text)
        if code:
            (self.run_dir / "search_algorithm.py").write_text(code, encoding="utf-8")
            logger.info("Extracted search_algorithm.py (%d lines)", code.count("\n") + 1)

        self._save_checkpoint()
        logger.info("Phase 6 complete")

    # ====================================================================
    # Checkpoint / Resume
    # ====================================================================

    def _save_checkpoint(self) -> None:
        """Persist phase_outputs to disk for crash recovery and inspection."""
        if not cfg.ENABLE_CHECKPOINTING:
            return
        checkpoint = {
            "phase_outputs": self.phase_outputs,
            "algorithm_version": self._algorithm_version,
            "total_usage": self.total_usage,
            "timestamp": now_timestamp(),
        }
        path = self.run_dir / "checkpoint.json"
        path.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
        logger.debug("Checkpoint saved to %s", path)

    def _load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load phase_outputs from a previous run for resuming."""
        path = checkpoint_dir / "checkpoint.json"
        if not path.exists():
            logger.warning("No checkpoint found at %s", path)
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        self.phase_outputs = data.get("phase_outputs", {})
        self._algorithm_version = data.get("algorithm_version", 0)
        self.total_usage = data.get("total_usage", {})

        # Use the checkpoint directory as run_dir for continuity
        self.run_dir = ensure_dir(checkpoint_dir)

        completed = [k for k in self.phase_outputs if k != "_current_review_feedback"]
        logger.info("Resumed from checkpoint. Completed phases: %s", completed)

    # ====================================================================
    # Helpers
    # ====================================================================

    def _extract_python_code(self, document: str) -> str:
        """Extract the largest Python code block from the document."""
        import re

        # Find all Python code blocks
        blocks = re.findall(r"```python\s*\n(.*?)```", document, re.DOTALL)
        if not blocks:
            # Try generic code blocks
            blocks = re.findall(r"```\s*\n(.*?)```", document, re.DOTALL)

        if not blocks:
            return ""

        # Return the longest block (likely the main search algorithm)
        return max(blocks, key=len).strip()

    def _save_metadata(self, elapsed_seconds: float) -> None:
        """Save run metadata (usage, timing, versions)."""
        metadata = {
            "problem_id": self.problem.id,
            "elapsed_seconds": elapsed_seconds,
            "algorithm_versions": self._algorithm_version,
            "total_usage": self.total_usage,
            "config": {
                "max_analysis_reviews": cfg.MAX_ANALYSIS_REVIEWS,
                "max_revision_rounds": cfg.MAX_REVISION_ROUNDS,
                "checkpoint_after_analysis": cfg.CHECKPOINT_AFTER_ANALYSIS,
            },
            "completed_at": now_timestamp(),
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
