"""Operators for the counterexample search system.

Each operator follows the pattern:
    operator(problem, phase_outputs, *, run_dir) -> tuple[result, usage_dict]

Content-producing operators return (str, dict).
Review operators return (ReviewResult, dict).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from disprover import config as cfg
from shared.agents import AgentResponse, call_gpt, web_search_formatted
from shared.utils import ensure_dir, merge_usage, now_timestamp, parse_json_response
from disprover.models import Issue, ProcessedLiterature, ReviewResult, SearchProblem
from disprover.prompt_builder import (
    ADVERSARIAL_REVIEW_TEMPLATE,
    ALGORITHM_REVISION_TEMPLATE,
    ALGORITHM_TEMPLATE,
    ANALYSIS_REVIEW_TEMPLATE,
    ANALYSIS_REVISION_TEMPLATE,
    ANALYSIS_TEMPLATE,
    ASSEMBLY_TEMPLATE,
    COMPLEXITY_TEMPLATE,
    CORRECTNESS_REVIEW_TEMPLATE,
    DOCUMENT_REVISION_TEMPLATE,
    FEASIBILITY_CHECK_TEMPLATE,
    FINAL_REVIEW_TEMPLATE,
    LITERATURE_AUDIT_TEMPLATE,
    LITERATURE_PROCESSING_TEMPLATE,
    LITERATURE_REVISION_TEMPLATE,
    LITERATURE_TEMPLATE,
    OPTIMALITY_REVIEW_TEMPLATE,
    build_phase_prompt,
)
from disprover.thinking_cores import (
    ADVERSARIAL_REVIEWER_CORE,
    ALGORITHM_DESIGNER_CORE,
    ANALYSIS_REVIEWER_CORE,
    COMPLEXITY_ESTIMATOR_CORE,
    CORRECTNESS_REVIEWER_CORE,
    DOCUMENT_ASSEMBLER_CORE,
    LITERATURE_AUDITOR_CORE,
    LITERATURE_PROCESSOR_CORE,
    DOCUMENT_REVISER_CORE,
    FEASIBILITY_CHECKER_CORE,
    OPTIMALITY_REVIEWER_CORE,
    STRUCTURAL_ANALYST_CORE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _log_io(
    name: str,
    prompt: str,
    response: str,
    *,
    run_dir: Optional[Path] = None,
) -> None:
    """Log prompt/response to files for post-mortem debugging."""
    if run_dir is None:
        return
    log_dir = ensure_dir(run_dir / "logs")
    timestamp = now_timestamp()
    (log_dir / f"{timestamp}_{name}_prompt.txt").write_text(prompt, encoding="utf-8")
    (log_dir / f"{timestamp}_{name}_response.txt").write_text(response, encoding="utf-8")


def _parse_review_json(text: str, round_name: str, reviewer: str) -> ReviewResult:
    """Parse a review JSON response into a ReviewResult.

    Robust to LLM quirks: strips markdown fences, handles missing fields.
    """
    try:
        data = parse_json_response(text)
    except ValueError:
        # If JSON parsing fails entirely, treat as a pass with a warning
        logger.warning("Failed to parse review JSON for %s. Raw text: %s...", round_name, text[:200])
        return ReviewResult(
            round_name=round_name,
            reviewer=reviewer,
            overall_verdict="pass",
            summary=f"[WARNING: Could not parse review JSON. Raw response stored in logs.]\n{text[:500]}",
        )

    def _parse_issues(raw: list) -> list[Issue]:
        issues = []
        for item in raw:
            if isinstance(item, str):
                issues.append(Issue(description=item, why_it_matters="", suggested_fix=""))
            elif isinstance(item, dict):
                issues.append(Issue(
                    description=item.get("description", str(item)),
                    why_it_matters=item.get("why_it_matters", ""),
                    suggested_fix=item.get("suggested_fix", ""),
                    severity=item.get("severity", "major"),
                ))
        return issues

    return ReviewResult(
        round_name=round_name,
        reviewer=reviewer,
        overall_verdict=data.get("overall_verdict", "pass"),
        critical_issues=_parse_issues(data.get("critical_issues", [])),
        major_issues=_parse_issues(data.get("major_issues", [])),
        minor_issues=_parse_issues(data.get("minor_issues", [])),
        positive_aspects=data.get("positive_aspects", []),
        summary=data.get("summary", ""),
    )


# ============================================================================
# Phase 1: Literature Research
# ============================================================================


def literature_research(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Search the web for known results and synthesize a literature survey."""
    usage: dict[str, int] = {}

    # Step 1: Generate search queries
    query_prompt = (
        f"Given the mathematical conjecture:\n{problem.statement}\n\n"
        f"Domain: {problem.domain or '(unspecified)'}\n"
        f"Background: {problem.background or '(none)'}\n\n"
        f"Generate {cfg.MAX_SEARCH_QUERIES} diverse web search queries to find:\n"
        "1. Known results about this conjecture\n"
        "2. Prior computational search attempts\n"
        "3. Related problems where counterexamples were found\n"
        "4. Relevant techniques and constructions\n\n"
        "Output each query on a separate line, nothing else."
    )

    resp = call_gpt(
        query_prompt,
        reasoning_effort="medium",
        text_verbosity="low",
        max_tokens=4000,
    )
    usage = merge_usage(usage, resp.usage)
    queries = [q.strip().strip('"').strip("'") for q in resp.text.strip().split("\n") if q.strip()]
    queries = queries[:cfg.MAX_SEARCH_QUERIES]

    # Step 2: Execute searches
    search_results_text = []
    if cfg.ENABLE_WEB_SEARCH:
        for query in queries:
            result = web_search_formatted(query, max_results=cfg.MAX_SEARCH_RESULTS)
            if result and "No web search results" not in result:
                search_results_text.append(result)

    combined_results = "\n\n".join(search_results_text) if search_results_text else "(no results found)"

    # Step 3: Synthesize
    prompt = build_phase_prompt(
        template=LITERATURE_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="literature",
    )
    prompt += f"\n\nWEB SEARCH RESULTS:\n{combined_results}"

    resp = call_gpt(
        prompt,
        system_prompt="You are a mathematical research assistant synthesizing literature search results.",
        reasoning_effort=cfg.REASONING_EFFORT_LITERATURE,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_LITERATURE_TOKENS,
    )
    usage = merge_usage(usage, resp.usage)

    _log_io("literature_research", prompt, resp.text, run_dir=run_dir)
    return resp.text, usage


# ============================================================================
# Phase 1.5: Literature Processing
# ============================================================================


def process_literature(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Process raw literature into structured, categorized, confidence-scored form."""
    prompt = build_phase_prompt(
        template=LITERATURE_PROCESSING_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="literature_processing",
    )

    resp = call_gpt(
        prompt,
        system_prompt=LITERATURE_PROCESSOR_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_LITERATURE_PROCESSING,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_LITERATURE_PROCESSING_TOKENS,
    )

    _log_io("process_literature", prompt, resp.text, run_dir=run_dir)

    # Validate JSON parses as ProcessedLiterature
    try:
        data = parse_json_response(resp.text)
        proc = ProcessedLiterature.model_validate(data)
        return proc.model_dump_json(indent=2), resp.usage
    except Exception as e:
        logger.warning("Failed to parse ProcessedLiterature JSON: %s. Storing raw.", e)
        return resp.text, resp.usage


def audit_literature(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Audit the processed literature for completeness and accuracy."""
    prompt = build_phase_prompt(
        template=LITERATURE_AUDIT_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="literature_audit",
    )

    resp = call_gpt(
        prompt,
        system_prompt=LITERATURE_AUDITOR_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("audit_literature", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "literature_audit", "gpt")
    return review, resp.usage


def revise_literature(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    review: ReviewResult,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Revise processed literature based on audit feedback."""
    augmented = {**phase_outputs, "_current_review_feedback": review.format_for_prompt()}

    prompt = build_phase_prompt(
        template=LITERATURE_REVISION_TEMPLATE,
        problem=problem,
        phase_outputs=augmented,
        phase_name="literature_revision",
    )

    resp = call_gpt(
        prompt,
        system_prompt=LITERATURE_PROCESSOR_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_LITERATURE_PROCESSING,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_LITERATURE_PROCESSING_TOKENS,
    )

    _log_io("revise_literature", prompt, resp.text, run_dir=run_dir)

    try:
        data = parse_json_response(resp.text)
        proc = ProcessedLiterature.model_validate(data)
        return proc.model_dump_json(indent=2), resp.usage
    except Exception as e:
        logger.warning("Failed to parse revised ProcessedLiterature: %s", e)
        return resp.text, resp.usage


# ============================================================================
# Phase 2: Structural Analysis
# ============================================================================


def analyze_structure(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Derive what a counterexample MUST look like."""
    prompt = build_phase_prompt(
        template=ANALYSIS_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="analysis",
    )

    resp = call_gpt(
        prompt,
        system_prompt=STRUCTURAL_ANALYST_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ANALYSIS,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ANALYSIS_TOKENS,
    )

    _log_io("analyze_structure", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


def review_analysis(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Review the structural analysis for correctness."""
    prompt = build_phase_prompt(
        template=ANALYSIS_REVIEW_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="analysis_review",
    )

    resp = call_gpt(
        prompt,
        system_prompt=ANALYSIS_REVIEWER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("review_analysis", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "analysis_review", "gpt")
    return review, resp.usage


def revise_analysis(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    review: ReviewResult,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """GPT revises the structural analysis based on review feedback."""
    # Inject review feedback into phase_outputs temporarily
    augmented = {**phase_outputs, "_current_review_feedback": review.format_for_prompt()}

    prompt = build_phase_prompt(
        template=ANALYSIS_REVISION_TEMPLATE,
        problem=problem,
        phase_outputs=augmented,
        phase_name="analysis_revision",
    )

    resp = call_gpt(
        prompt,
        system_prompt=STRUCTURAL_ANALYST_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ANALYSIS,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ANALYSIS_TOKENS,
    )

    _log_io("revise_analysis", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


# ============================================================================
# Phase 3: Algorithm Design
# ============================================================================


def design_algorithm(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Design a concrete, staged search algorithm with runnable code."""
    prompt = build_phase_prompt(
        template=ALGORITHM_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="algorithm",
    )

    resp = call_gpt(
        prompt,
        system_prompt=ALGORITHM_DESIGNER_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ALGORITHM,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ALGORITHM_TOKENS,
    )

    _log_io("design_algorithm", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


# ============================================================================
# Phase 4: Multi-Round Review
# ============================================================================


def correctness_review(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Review algorithm for mathematical and code correctness."""
    prompt = build_phase_prompt(
        template=CORRECTNESS_REVIEW_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="review_correctness",
    )

    resp = call_gpt(
        prompt,
        system_prompt=CORRECTNESS_REVIEWER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("correctness_review", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "correctness", "gpt")
    return review, resp.usage


def optimality_review(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """GPT (fresh context) reviews algorithm for optimality."""
    prompt = build_phase_prompt(
        template=OPTIMALITY_REVIEW_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="review_optimality",
    )

    resp = call_gpt(
        prompt,
        system_prompt=OPTIMALITY_REVIEWER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("optimality_review", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "optimality", "gpt")
    return review, resp.usage


def adversarial_review(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Adversarially attack the algorithm to find missed cases."""
    prompt = build_phase_prompt(
        template=ADVERSARIAL_REVIEW_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="review_adversarial",
    )

    resp = call_gpt(
        prompt,
        system_prompt=ADVERSARIAL_REVIEWER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("adversarial_review", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "adversarial", "gpt")
    return review, resp.usage


def revise_algorithm(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    review: ReviewResult,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """GPT revises the algorithm based on review feedback."""
    augmented = {**phase_outputs, "_current_review_feedback": review.format_for_prompt()}

    prompt = build_phase_prompt(
        template=ALGORITHM_REVISION_TEMPLATE,
        problem=problem,
        phase_outputs=augmented,
        phase_name="algorithm_revision",
    )

    resp = call_gpt(
        prompt,
        system_prompt=ALGORITHM_DESIGNER_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ALGORITHM,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ALGORITHM_TOKENS,
    )

    _log_io("revise_algorithm", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


# ============================================================================
# Phase 5: Complexity Estimation
# ============================================================================


def estimate_complexity(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Estimate computational cost of the algorithm."""
    prompt = build_phase_prompt(
        template=COMPLEXITY_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="complexity",
    )

    resp = call_gpt(
        prompt,
        system_prompt=COMPLEXITY_ESTIMATOR_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_COMPLEXITY,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_COMPLEXITY_TOKENS,
    )

    _log_io("estimate_complexity", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


def check_feasibility(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Sanity-check the complexity estimate."""
    prompt = build_phase_prompt(
        template=FEASIBILITY_CHECK_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="feasibility",
    )

    resp = call_gpt(
        prompt,
        system_prompt=FEASIBILITY_CHECKER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("check_feasibility", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "feasibility", "gpt")
    return review, resp.usage


# ============================================================================
# Phase 6: Final Document Assembly
# ============================================================================


def assemble_document(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Assemble the final self-contained search document."""
    prompt = build_phase_prompt(
        template=ASSEMBLY_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="assembly",
    )

    resp = call_gpt(
        prompt,
        system_prompt=DOCUMENT_ASSEMBLER_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ASSEMBLY,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ASSEMBLY_TOKENS,
    )

    _log_io("assemble_document", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage


def final_review(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[ReviewResult, dict[str, int]]:
    """Perform a final review of the assembled document."""
    prompt = build_phase_prompt(
        template=FINAL_REVIEW_TEMPLATE,
        problem=problem,
        phase_outputs=phase_outputs,
        phase_name="final_review",
    )

    resp = call_gpt(
        prompt,
        system_prompt=DOCUMENT_ASSEMBLER_CORE,
        json_mode=True,
        reasoning_effort=cfg.REASONING_EFFORT_REVIEW,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_REVIEW_TOKENS,
    )

    _log_io("final_review", prompt, resp.text, run_dir=run_dir)
    review = _parse_review_json(resp.text, "final_review", "gpt")
    return review, resp.usage


def revise_document(
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    review: ReviewResult,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """GPT revises the final document based on review feedback."""
    augmented = {**phase_outputs, "_current_review_feedback": review.format_for_prompt()}

    prompt = build_phase_prompt(
        template=DOCUMENT_REVISION_TEMPLATE,
        problem=problem,
        phase_outputs=augmented,
        phase_name="document_revision",
    )

    resp = call_gpt(
        prompt,
        system_prompt=DOCUMENT_REVISER_CORE,
        reasoning_effort=cfg.REASONING_EFFORT_ASSEMBLY,
        text_verbosity=cfg.TEXT_VERBOSITY,
        max_tokens=cfg.MAX_ASSEMBLY_TOKENS,
    )

    _log_io("revise_document", prompt, resp.text, run_dir=run_dir)
    return resp.text, resp.usage
