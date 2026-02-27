from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time
import sys

from rich.console import Console
from rich.table import Table

from prover import config
from prover.models import Problem, Lemma, ProofArtifact, LemmaStatus, AuditVerdict
from prover.operators import (
    literature_research,
    process_literature,
    audit_literature,
    revise_literature,
    draft_global_plan,
    audit_plan,
    decompose_blocked_lemma,
    prove_lemma,
    prove_lemma_recursive,
    solve_problem,  # New recursive grammar
    self_audit,
    cross_audit,
    validate_output,
    extract_interface,
    integrate_proof,
    final_audit,
    get_retry_feedback,
)
from prover.models import ProcessedLiterature, LiteratureReviewResult
from shared.utils import ensure_dir, now_timestamp, merge_usage
from prover.utils import topo_sort_lemmas, Stopwatch, resolve_literature_for_phase


@dataclass
class RunMetrics:
    attempts_total: int = 0
    retries_total: int = 0
    lemmas_proved: int = 0
    lemmas_blocked: int = 0
    audit_agree: int = 0
    audit_disagree: int = 0


def _group_lemmas_by_level(lemmas: list[Lemma]) -> list[list[Lemma]]:
    """Group lemmas into levels by dependency depth.

    Returns:
        List of levels, where each level is a list of lemmas that can be proved
        in parallel (all dependencies are in previous levels).
    """
    # Build dependency map
    dep_map: dict[str, set[str]] = {lem.id: set(lem.dependencies) for lem in lemmas}

    # Compute depth for each lemma (max depth of any dependency + 1)
    depths: dict[str, int] = {}

    def compute_depth(lemma_id: str) -> int:
        if lemma_id in depths:
            return depths[lemma_id]

        deps = dep_map.get(lemma_id, set())
        if not deps:
            depths[lemma_id] = 0
            return 0

        max_dep_depth = max((compute_depth(dep_id) for dep_id in deps), default=-1)
        depths[lemma_id] = max_dep_depth + 1
        return depths[lemma_id]

    # Compute all depths
    for lem in lemmas:
        compute_depth(lem.id)

    # Group by level
    max_level = max(depths.values(), default=0)
    levels: list[list[Lemma]] = [[] for _ in range(max_level + 1)]

    for lem in lemmas:
        level = depths[lem.id]
        levels[level].append(lem)

    return levels


class ProofController:
    """Main orchestrator for the proof workflow."""

    def __init__(
        self,
        problem: Problem,
        *,
        console: Optional[Console] = None,
        run_root: Optional[str | Path] = None,
        verbose: bool = False,
    ):
        self.problem = problem
        self.console = console or Console()
        self.verbose = verbose

        root = Path(run_root or config.RUNS_DIR)
        self.run_dir = ensure_dir(root / problem.id / now_timestamp())

        self.outline = None
        self.iteration = 0
        self.max_iterations = config.MAX_ITERATIONS
        self.usage_totals: dict[str, int] = {}
        self.metrics = RunMetrics()
        self.timer = Stopwatch()
        self.all_lemmas: dict[str, Lemma] = {}  # Global tracking for recursive decomposition

        # Persist the problem for reproducibility
        (self.run_dir / "problem.json").write_text(problem.model_dump_json(indent=2), encoding="utf-8")

    def _run_literature_processing(self, raw_literature: str) -> Optional[ProcessedLiterature]:
        """Phase 0.5: Process raw literature into structured form with audit loop."""
        reviews_dir = ensure_dir(self.run_dir / "reviews")

        # Step 1: Process raw literature into structured JSON
        self.console.print("  Processing raw literature...")
        processed_json, proc_usage = process_literature(
            self.problem, raw_literature, run_dir=self.run_dir
        )
        merge_usage(self.usage_totals, proc_usage)

        # Save initial processing
        (self.run_dir / "processed_literature.json").write_text(processed_json, encoding="utf-8")

        # Step 2: Audit loop
        for audit_round in range(config.MAX_LITERATURE_REVIEWS):
            self.console.print(f"  Audit round {audit_round + 1}/{config.MAX_LITERATURE_REVIEWS}...")

            review, audit_usage = audit_literature(
                self.problem, raw_literature, processed_json, run_dir=self.run_dir
            )
            merge_usage(self.usage_totals, audit_usage)

            # Save review
            review_path = reviews_dir / f"literature_audit_r{audit_round + 1}.txt"
            review_path.write_text(review.format_for_prompt(), encoding="utf-8")

            if not review.has_blocking_issues():
                self.console.print(f"  ✓ Literature audit passed (round {audit_round + 1})")
                break

            self.console.print(f"  ✗ Literature audit: {review.overall_verdict} — revising...")

            # Revise
            processed_json, rev_usage = revise_literature(
                self.problem, raw_literature, processed_json,
                review.format_for_prompt(), run_dir=self.run_dir
            )
            merge_usage(self.usage_totals, rev_usage)

            # Save revised version
            (self.run_dir / "processed_literature.json").write_text(processed_json, encoding="utf-8")

        # Parse final result
        try:
            proc = ProcessedLiterature.model_validate_json(processed_json)
        except Exception:
            self.console.print("  [yellow]Warning: Could not parse processed literature, falling back to raw[/yellow]")
            self.problem.background += f"\n\n===== LITERATURE RESEARCH =====\n{raw_literature}"
            return None

        # Save human-readable version
        (self.run_dir / "processed_literature.md").write_text(proc.full_text(), encoding="utf-8")

        return proc

    def run(self) -> ProofArtifact:
        self.console.print(f"[bold]Proof workflow[/bold] problem_id={self.problem.id}")
        self.console.print(f"Run dir: {self.run_dir}")

        # Step 0: Literature research (if enabled)
        raw_literature = ""
        if config.ENABLE_LITERATURE_RESEARCH:
            self.console.print("\n[0/5] Researching literature...")
            literature_summary, lit_usage = literature_research(self.problem, run_dir=self.run_dir)
            merge_usage(self.usage_totals, lit_usage)
            raw_literature = literature_summary

            # Save raw literature research
            (self.run_dir / "literature_research.txt").write_text(literature_summary, encoding="utf-8")
            self.console.print(f"  ✓ Literature research complete")
        else:
            self.console.print("\n[0/5] Literature research disabled (skipping)")

        # Step 0.5: Process literature into structured form (if enabled)
        self.processed_literature: Optional[ProcessedLiterature] = None
        if raw_literature and config.ENABLE_LITERATURE_PROCESSING:
            self.console.print("\n[0.5/5] Processing literature into structured form...")
            self.processed_literature = self._run_literature_processing(raw_literature)
            self.console.print(f"  ✓ Literature processing complete")
        elif raw_literature:
            # No processing — fall back to raw dump in background (legacy behavior)
            self.problem.background += f"\n\n===== LITERATURE RESEARCH =====\n{raw_literature}"

        # Augment problem background with structured literature
        if self.processed_literature:
            self.problem.background += (
                f"\n\n===== PROCESSED LITERATURE (Executive Summary) =====\n"
                f"{self.processed_literature.executive_summary}"
            )

        # Step 1: draft plan with intelligent audit retry loop
        self.console.print("\n[1/5] Drafting and auditing global plan...")

        plan_audit_result = None
        plan_history = []  # Track all planning attempts

        for plan_round in range(config.MAX_PLAN_ROUNDS):
            # Generate plan with accumulated feedback
            if plan_round == 0:
                retry_feedback = ""
            else:
                retry_feedback = self._format_plan_retry_feedback(plan_history, plan_round)

            # Draft plan
            outline, usage = draft_global_plan(
                self.problem,
                run_dir=self.run_dir,
                retry_feedback=retry_feedback,
                literature_context=resolve_literature_for_phase(self.processed_literature, "draft_plan"),
            )
            self.outline = outline
            merge_usage(self.usage_totals, usage)

            # Sort lemmas once; preserves a clean dependency-respecting order.
            ordered = topo_sort_lemmas(self.outline.lemmas)
            self.outline.lemmas = ordered
            self.console.print(f"  → Round {plan_round + 1}: Generated {len(self.outline.lemmas)} lemmas")
            if self.outline.proof_strategy:
                strategy_preview = self.outline.proof_strategy[:100] + "..." if len(self.outline.proof_strategy) > 100 else self.outline.proof_strategy
                self.console.print(f"  Strategy: {strategy_preview}")

            # Audit the plan
            self.console.print("  Auditing decomposition plan...")
            plan_audit_result, audit_usage = audit_plan(
                self.problem, self.outline, run_dir=self.run_dir,
                literature_context=resolve_literature_for_phase(self.processed_literature, "audit_plan"),
            )
            merge_usage(self.usage_totals, audit_usage)

            # Record this attempt
            plan_history.append({
                'round': plan_round,
                'outline': outline,
                'audit': plan_audit_result,
            })

            self.console.print(
                f"    Plan audit: {plan_audit_result.verdict.value} ({plan_audit_result.confidence:.2f})"
            )

            # Decision logic based on audit
            if plan_audit_result.verdict == AuditVerdict.PASS and plan_audit_result.confidence >= config.MIN_PLAN_CONFIDENCE:
                self.console.print(f"  [green]✓ Plan approved (confidence: {plan_audit_result.confidence:.2f})[/green]")
                break

            elif plan_audit_result.confidence < config.ABORT_PLAN_CONFIDENCE:
                # Plan is fundamentally flawed - abort
                self.console.print(f"  [red]✗ Plan confidence too low ({plan_audit_result.confidence:.2f} < {config.ABORT_PLAN_CONFIDENCE})[/red]")
                self.console.print(f"  [red]✗ Critical issues cannot be resolved - aborting[/red]")
                self._print_critical_issues(plan_audit_result.issues)

                # Save failure report
                failure_report = self._format_planning_failure_report(plan_history, self.problem)
                (self.run_dir / "planning_failure.txt").write_text(failure_report, encoding="utf-8")

                # Save last outline and audit for analysis
                (self.run_dir / "outline.json").write_text(self.outline.model_dump_json(indent=2), encoding="utf-8")
                (self.run_dir / "plan_audit.json").write_text(plan_audit_result.model_dump_json(indent=2), encoding="utf-8")

                raise RuntimeError(
                    f"Planning failed after {len(plan_history)} rounds. "
                    f"Problem may be too difficult for current system capabilities. "
                    f"See {self.run_dir / 'planning_failure.txt'} for details."
                )

            elif plan_round < config.MAX_PLAN_ROUNDS - 1:
                # Uncertain - incorporate feedback and retry
                critical_count = len([i for i in plan_audit_result.issues if i.get('severity') == 'critical'])
                major_count = len([i for i in plan_audit_result.issues if i.get('severity') == 'major'])

                self.console.print(
                    f"  [yellow]⚠ Plan uncertain (confidence: {plan_audit_result.confidence:.2f}), "
                    f"round {plan_round + 1}/{config.MAX_PLAN_ROUNDS}[/yellow]"
                )
                self.console.print(f"  → {critical_count} critical issues, {major_count} major issues to address")

                # Show top issues
                if plan_audit_result.issues:
                    self.console.print("  → Top issues:")
                    for issue in plan_audit_result.issues[:3]:
                        location = issue.get('location', 'OVERALL')
                        summary = issue.get('summary', '')
                        self.console.print(f"      • {location}: {summary[:80]}...")

                self.console.print(f"  → Generating revised plan...")

            else:
                # Max rounds reached without approval
                self.console.print(f"  [red]✗ Max planning rounds ({config.MAX_PLAN_ROUNDS}) reached without viable plan[/red]")
                self.console.print(f"  [red]✗ Final confidence: {plan_audit_result.confidence:.2f}[/red]")
                self._print_critical_issues(plan_audit_result.issues)

                # Save failure report
                failure_report = self._format_planning_failure_report(plan_history, self.problem)
                (self.run_dir / "planning_failure.txt").write_text(failure_report, encoding="utf-8")

                # Save last outline and audit for analysis
                (self.run_dir / "outline.json").write_text(self.outline.model_dump_json(indent=2), encoding="utf-8")
                (self.run_dir / "plan_audit.json").write_text(plan_audit_result.model_dump_json(indent=2), encoding="utf-8")

                raise RuntimeError(
                    f"Planning failed after {config.MAX_PLAN_ROUNDS} rounds. "
                    f"Final confidence {plan_audit_result.confidence:.2f} below minimum {config.MIN_PLAN_CONFIDENCE}. "
                    f"See {self.run_dir / 'planning_failure.txt'} for details."
                )

        # Save successful plan
        (self.run_dir / "outline.json").write_text(self.outline.model_dump_json(indent=2), encoding="utf-8")
        if plan_audit_result:
            (self.run_dir / "plan_audit.json").write_text(plan_audit_result.model_dump_json(indent=2), encoding="utf-8")

        # Step 2: prove lemmas (with careful parallelization)
        self.console.print("\n[2/4] Proving lemmas...")

        # Initialize global lemma tracking for recursive decomposition
        self.all_lemmas = {l.id: l for l in self.outline.lemmas}

        # Group lemmas by dependency level for careful parallelization
        if config.ENABLE_PARALLEL:
            levels = _group_lemmas_by_level(self.outline.lemmas)
            self.console.print(f"  → Organized into {len(levels)} dependency levels")

            for level_idx, level_lemmas in enumerate(levels):
                self.console.print(f"\n  Level {level_idx}: {len(level_lemmas)} lemmas")

                # Parallelize first n lemmas in this level (default n=2)
                parallel_batch = level_lemmas[:config.MAX_PARALLEL_LEMMAS]
                sequential_batch = level_lemmas[config.MAX_PARALLEL_LEMMAS:]

                # Process parallel batch concurrently
                if len(parallel_batch) > 1:
                    self.console.print(f"    → Proving {len(parallel_batch)} lemmas in parallel")
                    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_WORKERS) as executor:
                        futures = {
                            executor.submit(self._prove_lemma_with_retries, lemma): lemma
                            for lemma in parallel_batch
                        }

                        for future in as_completed(futures):
                            lemma = futures[future]
                            try:
                                future.result()  # Wait for completion
                                self._save_state()
                            except Exception as e:
                                self.console.print(f"    [red]Exception in parallel proof of {lemma.id}: {e}[/red]")
                elif len(parallel_batch) == 1:
                    # Single lemma, no need for parallelization
                    self._prove_lemma_with_retries(parallel_batch[0])
                    self._save_state()

                # Process remaining lemmas sequentially
                for lemma in sequential_batch:
                    self._prove_lemma_with_retries(lemma)
                    self._save_state()

        else:
            # Sequential mode (original behavior)
            self.console.print("  → Sequential mode (parallelization disabled)")
            for lemma in self.outline.lemmas:
                # If a dependency is blocked, this lemma is unprovable
                if any(self._get_lemma(dep_id).status != LemmaStatus.PROVED for dep_id in lemma.dependencies):
                    lemma.status = LemmaStatus.BLOCKED
                    lemma.evidence.append(
                        {
                            "auditor": "controller",
                            "verdict": "FAIL",
                            "confidence": 1.0,
                            "issues": ["Dependency not proved"],
                            "feedback": f"Skipped because at least one dependency is not proved: {lemma.dependencies}",
                        }
                    )
                    self.metrics.lemmas_blocked += 1
                    self._save_state()
                    self.console.print(f"  [yellow]Skip/Block[/yellow] {lemma.id} (dependency not proved)")
                    continue

                self._prove_lemma_with_retries(lemma)
                self._save_state()

        # Step 3: integrate proof
        self.console.print("\n[3/4] Integrating proof...")
        proved_lemmas = [l for l in self.outline.lemmas if l.status == LemmaStatus.PROVED]
        blocked_lemmas = [l for l in self.outline.lemmas if l.status == LemmaStatus.BLOCKED]

        self.metrics.lemmas_proved = len(proved_lemmas)
        self.metrics.lemmas_blocked = len(blocked_lemmas)

        if not proved_lemmas:
            proof_text = "(No proved lemmas. Nothing to integrate.)"
            integration_usage = {}
        else:
            proof_text, integration_usage = integrate_proof(proved_lemmas, self.problem, run_dir=self.run_dir)
            merge_usage(self.usage_totals, integration_usage)

        # Step 4: final audit
        self.console.print("\n[4/4] Final audit...")
        final_audit_result, final_usage = final_audit(
            proof_text, self.problem, run_dir=self.run_dir,
            literature_context=resolve_literature_for_phase(self.processed_literature, "final_audit"),
        )
        merge_usage(self.usage_totals, final_usage)
        self.console.print(
            f"  → Verdict: {final_audit_result.verdict.value} (confidence: {final_audit_result.confidence:.2f})"
        )

        # Determine overall status
        all_proved = len(proved_lemmas) == len(self.outline.lemmas)
        final_pass = final_audit_result.verdict == AuditVerdict.PASS

        if all_proved and final_pass:
            status = "complete"
        elif proved_lemmas:
            status = "partial" if not final_pass or blocked_lemmas else "partial"
        else:
            status = "failed"

        artifact = ProofArtifact(
            problem_id=self.problem.id,
            status=status,
            proof_text=proof_text,
            lemmas=self.outline.lemmas,
            final_audit=final_audit_result,
            metadata=self._build_metadata(),
        )

        # Persist artifact
        (self.run_dir / "artifact.json").write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

        return artifact

    def _build_metadata(self) -> dict:
        agree = self.metrics.audit_agree
        disagree = self.metrics.audit_disagree
        denom = agree + disagree
        agree_rate = (agree / denom) if denom else None

        return {
            "iterations": self.iteration,
            "runtime_s": round(self.timer.elapsed_s(), 3),
            "lemmas_proved": self.metrics.lemmas_proved,
            "lemmas_total": len(self.outline.lemmas) if self.outline else 0,
            "lemmas_blocked": self.metrics.lemmas_blocked,
            "attempts_total": self.metrics.attempts_total,
            "retries_total": self.metrics.retries_total,
            "audit_agreement": {
                "agree": agree,
                "disagree": disagree,
                "agree_rate": agree_rate,
            },
            "token_usage": self.usage_totals,
        }

    def _save_state(self) -> None:
        """Save a lightweight checkpoint of current lemma states and context documents."""

        if not self.outline:
            return

        # Save outline with embedded contexts
        (self.run_dir / "state.json").write_text(self.outline.model_dump_json(indent=2), encoding="utf-8")

        # Also save separate context files for debugging
        context_dir = ensure_dir(self.run_dir / "contexts")

        for lemma in self.outline.lemmas:
            if lemma.context:
                context_path = context_dir / f"{lemma.id}_context.json"
                context_path.write_text(lemma.context.model_dump_json(indent=2), encoding="utf-8")

        # Also save contexts from all_lemmas (includes sub-lemmas)
        for lemma_id, lemma in self.all_lemmas.items():
            if lemma.context and lemma_id not in {l.id for l in self.outline.lemmas}:
                context_path = context_dir / f"{lemma_id}_context.json"
                context_path.write_text(lemma.context.model_dump_json(indent=2), encoding="utf-8")

    def _check_control_signals(self) -> None:
        """Check for external control signals from dashboard (pause/resume/stop).

        Reads control.json file in run directory and acts accordingly.
        Control signals:
            - pause: Pause execution and wait for resume signal
            - resume: Resume from paused state
            - stop: Gracefully stop execution
        """
        control_file = self.run_dir / "control.json"

        if not control_file.exists():
            return  # No control signals

        try:
            control_data = json.loads(control_file.read_text(encoding="utf-8"))
            action = control_data.get("action")

            if action == "pause":
                self.console.print("\n[yellow]⏸ PAUSED by dashboard[/yellow]")
                self.console.print("Waiting for resume signal...")

                # Wait loop - check every 5 seconds for resume/stop
                while True:
                    time.sleep(5)

                    if not control_file.exists():
                        # Control file deleted - resume
                        self.console.print("[green]▶️ Resumed (control file removed)[/green]")
                        break

                    control_data = json.loads(control_file.read_text(encoding="utf-8"))
                    new_action = control_data.get("action")

                    if new_action == "resume":
                        self.console.print("[green]▶️ Resumed by dashboard[/green]")
                        # Clear control file to prevent re-pausing
                        control_file.unlink()
                        break
                    elif new_action == "stop":
                        self.console.print("[red]⏹ Stopped by dashboard[/red]")
                        self._save_state()  # Final save before exit
                        sys.exit(0)

            elif action == "stop":
                self.console.print("[red]⏹ Stopped by dashboard[/red]")
                self._save_state()  # Final save before exit
                sys.exit(0)

            # Clear control file after processing (except pause, which needs to persist)
            if action != "pause" and control_file.exists():
                control_file.unlink()

        except Exception as e:
            # Don't crash on control file errors - just log and continue
            self.console.print(f"[yellow]Warning: Error reading control signals: {e}[/yellow]")

    def _get_lemma(self, lemma_id: str) -> Lemma:
        assert self.outline is not None
        for l in self.outline.lemmas:
            if l.id == lemma_id:
                return l
        raise KeyError(f"Lemma not found: {lemma_id}")

    def _get_dependency_interfaces(self, lemma: Lemma) -> dict[str, str]:
        interfaces: dict[str, str] = {}
        for dep_id in lemma.dependencies:
            dep = self._get_lemma(dep_id)
            if dep.interface:
                interfaces[dep_id] = dep.interface
        return interfaces

    def _prove_lemma_with_retries(self, lemma: Lemma) -> None:
        # Check for control signals (pause/resume/stop) from dashboard
        self._check_control_signals()

        self.console.print(f"\n  Proving [bold]{lemma.id}[/bold]: {lemma.statement[:90]}")

        # Check if any dependency is blocked (important for parallel execution)
        for dep_id in lemma.dependencies:
            dep = self._get_lemma(dep_id)
            if dep.status != LemmaStatus.PROVED:
                lemma.status = LemmaStatus.BLOCKED
                lemma.evidence.append(
                    {
                        "auditor": "controller",
                        "verdict": "FAIL",
                        "confidence": 1.0,
                        "issues": ["Dependency not proved"],
                        "feedback": f"Skipped because dependency {dep_id} is not proved (status: {dep.status.value})",
                    }
                )
                self.metrics.lemmas_blocked += 1
                self.console.print(f"  [yellow]Skip/Block[/yellow] {lemma.id} (dependency {dep_id} not proved)")
                return

        # Initialize lemma in global tracker
        self.all_lemmas[lemma.id] = lemma

        # Try recursive proving with new grammar (handles retries, decision, backtracking)
        dep_ifaces = self._get_dependency_interfaces(lemma)

        proved_lemma, usage = solve_problem(
            lemma,
            self.problem,
            dep_ifaces,
            self.all_lemmas,  # Pass global tracker
            run_dir=self.run_dir,
        )
        merge_usage(self.usage_totals, usage)

        # Update lemma reference
        lemma.proof = proved_lemma.proof
        lemma.status = proved_lemma.status
        lemma.sub_lemmas = proved_lemma.sub_lemmas
        lemma.context = proved_lemma.context  # Preserve context
        lemma.decomposition_strategies = proved_lemma.decomposition_strategies  # Preserve strategies

        # If blocked, no need to audit
        if lemma.status == LemmaStatus.BLOCKED:
            self.console.print(f"    [red]✗ Blocked[/red] (could not prove even with decomposition)")
            self.metrics.lemmas_blocked += 1
            return

        # Proof succeeded (either directly or via sub-lemmas) - now audit once
        self.iteration += 1
        self.metrics.attempts_total += 1

        # Self-audit
        self_audit_result, u1 = self_audit(
            lemma, self.problem.domain, run_dir=self.run_dir,
            literature_context=resolve_literature_for_phase(self.processed_literature, "self_audit"),
        )
        merge_usage(self.usage_totals, u1)
        lemma.evidence.append(self_audit_result.model_dump(mode="json"))
        self.console.print(
            f"    Self-audit: {self_audit_result.verdict.value} ({self_audit_result.confidence:.2f})"
        )

        # Cross-audit
        cross_audit_result, u2 = cross_audit(
            lemma, self.problem.domain, run_dir=self.run_dir,
            literature_context=resolve_literature_for_phase(self.processed_literature, "cross_audit"),
        )
        merge_usage(self.usage_totals, u2)
        lemma.evidence.append(cross_audit_result.model_dump(mode="json"))
        self.console.print(
            f"    Cross-audit: {cross_audit_result.verdict.value} ({cross_audit_result.confidence:.2f})"
        )

        # Agreement metric
        if self_audit_result.verdict.value == cross_audit_result.verdict.value:
            self.metrics.audit_agree += 1
        else:
            self.metrics.audit_disagree += 1

        decision = validate_output(lemma)

        if decision == "ACCEPT":
            interface_text, ui = extract_interface(lemma, run_dir=self.run_dir)
            merge_usage(self.usage_totals, ui)
            lemma.interface = interface_text
            lemma.status = LemmaStatus.PROVED
            self.console.print("    [green]✓ Accepted[/green]")
            self.metrics.lemmas_proved += 1
            return

        # Audit failed - mark as BLOCKED
        # (prove_lemma_recursive already tried all retries and decomposition)
        lemma.status = LemmaStatus.BLOCKED
        self.console.print(f"    [red]✗ Blocked[/red] (audit failed)")
        self.metrics.lemmas_blocked += 1

    def _format_plan_retry_feedback(self, plan_history: list, round_num: int) -> str:
        """Format feedback for next planning round based on audit issues."""
        if not plan_history:
            return ""

        prev_audit = plan_history[-1]['audit']

        feedback = f"=== PLAN REVISION (Round {round_num + 1}) ===\n\n"
        feedback += f"PREVIOUS ATTEMPT CONFIDENCE: {prev_audit.confidence:.2f}\n\n"

        # Critical issues
        critical_issues = [i for i in prev_audit.issues if i.get('severity') == 'critical']
        if critical_issues:
            feedback += "CRITICAL ISSUES TO ADDRESS:\n"
            for issue in critical_issues:
                feedback += f"\n- **{issue.get('summary', 'No summary')}**\n"
                feedback += f"  Location: {issue.get('location', 'OVERALL')}\n"
                detail = issue.get('detail', '')
                feedback += f"  Detail: {detail[:300]}{'...' if len(detail) > 300 else ''}\n"
                patch = issue.get('patch', '')
                feedback += f"  Suggested fix: {patch[:200]}{'...' if len(patch) > 200 else ''}\n"

        # Major issues
        major_issues = [i for i in prev_audit.issues if i.get('severity') == 'major']
        if major_issues:
            feedback += "\n\nMAJOR ISSUES TO ADDRESS:\n"
            for issue in major_issues:
                feedback += f"\n- {issue.get('summary', 'No summary')}\n"
                patch = issue.get('patch', '')
                feedback += f"  Suggested fix: {patch[:150]}{'...' if len(patch) > 150 else ''}\n"

        feedback += "\n\nYOUR TASK:\n"
        feedback += "Generate a NEW plan that addresses these issues. "
        feedback += "Do NOT repeat the previous plan. "
        feedback += "Incorporate the suggested fixes or find alternative approaches.\n"

        return feedback

    def _print_critical_issues(self, issues: list) -> None:
        """Print critical issues for user visibility."""
        critical = [i for i in issues if i.get('severity') == 'critical']
        if critical:
            self.console.print("\n  [red]Critical issues identified:[/red]")
            for issue in critical[:5]:  # Top 5
                location = issue.get('location', 'OVERALL')
                summary = issue.get('summary', 'No summary')
                self.console.print(f"    • {location}: {summary[:100]}...")

    def _format_planning_failure_report(self, plan_history: list, problem: Problem) -> str:
        """Generate detailed failure report for aborted planning."""
        report = "# Planning Failure Report\n\n"
        report += f"**Problem ID:** {problem.id}\n\n"
        report += f"**Statement:** {problem.statement}\n\n"
        report += f"**Domain:** {problem.domain}\n\n"
        report += f"**Planning Rounds Attempted:** {len(plan_history)}\n\n"
        report += "=" * 80 + "\n\n"

        for i, entry in enumerate(plan_history):
            report += f"## Round {entry['round'] + 1}\n\n"
            report += f"**Confidence:** {entry['audit'].confidence:.2f}\n"
            report += f"**Verdict:** {entry['audit'].verdict.value}\n"
            report += f"**Lemmas Generated:** {len(entry['outline'].lemmas)}\n"
            report += f"**Strategy:** {entry['outline'].proof_strategy[:200]}...\n\n"

            # Critical issues
            critical = [iss for iss in entry['audit'].issues if iss.get('severity') == 'critical']
            if critical:
                report += f"**Critical Issues:** {len(critical)}\n\n"
                for issue in critical:
                    report += f"- **{issue.get('location')}:** {issue.get('summary')}\n"
                    detail = issue.get('detail', '')
                    report += f"  - {detail[:300]}{'...' if len(detail) > 300 else ''}\n"
                report += "\n"

            # Major issues
            major = [iss for iss in entry['audit'].issues if iss.get('severity') == 'major']
            if major:
                report += f"**Major Issues:** {len(major)}\n\n"
                for issue in major[:3]:  # Top 3 major issues
                    report += f"- **{issue.get('location')}:** {issue.get('summary')}\n"
                report += "\n"

            report += "-" * 80 + "\n\n"

        report += "## Conclusion\n\n"
        report += "Unable to generate viable decomposition plan after "
        report += f"{len(plan_history)} rounds. "

        final_confidence = plan_history[-1]['audit'].confidence
        if final_confidence < config.ABORT_PLAN_CONFIDENCE:
            report += f"\n\nFinal confidence ({final_confidence:.2f}) fell below abort "
            report += f"threshold ({config.ABORT_PLAN_CONFIDENCE}). "
            report += "Critical issues could not be resolved.\n"
        else:
            report += f"\n\nFinal confidence ({final_confidence:.2f}) below minimum required "
            report += f"({config.MIN_PLAN_CONFIDENCE}) after maximum rounds.\n"

        report += "\n**Recommendation:** Problem may be too difficult for current system capabilities. "
        report += "Consider:\n"
        report += "1. Simplifying the problem statement\n"
        report += "2. Providing additional background/hints\n"
        report += "3. Breaking problem into smaller sub-problems manually\n"
        report += "4. Increasing MAX_PLAN_ROUNDS or adjusting confidence thresholds\n"

        return report
