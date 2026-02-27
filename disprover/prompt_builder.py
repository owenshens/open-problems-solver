"""Prompt construction for the counterexample search system.

Composes phase prompts from:
  - Base template (with {placeholders})
  - Problem fields (selected per phase via PHASE_INCLUDES)
  - Context documents (injected in full where relevant)
  - Prior phase outputs (accumulated)

The PHASE_INCLUDES table controls which optional problem fields appear in
each phase's prompt.  Prior phase outputs flow through the template's
named placeholders regardless of PHASE_INCLUDES.
"""

from __future__ import annotations

import logging

from disprover.models import ProcessedLiterature, SearchProblem

logger = logging.getLogger(__name__)


# ============================================================================
# Which literature categories each downstream phase receives
# ============================================================================

PHASE_LITERATURE_CATEGORIES: dict[str, list[str]] = {
    "literature_processing": [],  # Gets raw survey directly, not filtered
    "literature_audit": [],  # Gets raw + processed directly
    "literature_revision": [],  # Gets raw + processed directly
    "analysis": [
        "known_bounds",
        "construction_techniques",
        "obstructions",
        "related_solved_problems",
        "prior_computational_attempts",
    ],
    "analysis_review": [
        "known_bounds",
        "obstructions",
    ],
    "analysis_revision": [
        "known_bounds",
        "construction_techniques",
        "obstructions",
        "related_solved_problems",
        "prior_computational_attempts",
    ],
    "algorithm": [
        "construction_techniques",
        "prior_computational_attempts",
        "known_bounds",
    ],
    "review_correctness": [
        "known_bounds",
        "obstructions",
    ],
    "review_optimality": [
        "prior_computational_attempts",
        "construction_techniques",
    ],
    "review_adversarial": [
        "known_bounds",
        "obstructions",
        "construction_techniques",
    ],
    "algorithm_revision": [
        "construction_techniques",
        "prior_computational_attempts",
        "known_bounds",
    ],
    "complexity": [],
    "feasibility": [],
    "assembly": [
        "known_bounds",
        "prior_computational_attempts",
        "related_solved_problems",
        "construction_techniques",
        "obstructions",
    ],
    "final_review": [
        "known_bounds",
    ],
    "document_revision": [],
}


# ============================================================================
# Which optional problem fields each phase receives
# ============================================================================

PHASE_INCLUDES: dict[str, dict[str, bool]] = {
    "literature": {
        "prior_attempts": True,
    },
    "literature_processing": {
        "prior_attempts": True,
    },
    "literature_audit": {},
    "literature_revision": {
        "prior_attempts": True,
    },
    "analysis": {
        "context_documents": True,
        "strategy_suggestions": True,
        "prior_attempts": True,
    },
    "analysis_review": {
        "context_documents": True,
    },
    "analysis_revision": {
        "context_documents": True,
        "strategy_suggestions": True,
        "prior_attempts": True,
    },
    "algorithm": {
        "context_documents": True,
        "strategy_suggestions": True,
        "prior_attempts": True,
        "search_constraints": True,
    },
    "review_correctness": {
        "context_documents": True,
    },
    "review_optimality": {
        "prior_attempts": True,
        "search_constraints": True,
    },
    "review_adversarial": {
        "context_documents": True,
        "prior_attempts": True,
    },
    "algorithm_revision": {
        "context_documents": True,
        "strategy_suggestions": True,
        "prior_attempts": True,
        "search_constraints": True,
    },
    "complexity": {
        "prior_attempts": True,
        "search_constraints": True,
    },
    "feasibility": {
        "search_constraints": True,
    },
    "assembly": {
        "context_documents": True,
    },
    "final_review": {
        "context_documents": True,
    },
    "document_revision": {
        "context_documents": True,
    },
}


# ============================================================================
# Formatting helpers
# ============================================================================


def _format_context_docs(problem: SearchProblem) -> str:
    if not problem.context_documents:
        return "(none provided)"
    docs = []
    for i, doc in enumerate(problem.context_documents, 1):
        docs.append(
            f"--- CONTEXT DOCUMENT {i} ---\n{doc}\n--- END CONTEXT DOCUMENT {i} ---"
        )
    return "\n\n".join(docs)


def _format_strategies(problem: SearchProblem) -> str:
    if not problem.strategy_suggestions:
        return "(none provided)"
    lines = []
    for s in problem.strategy_suggestions:
        lines.append(f"- [{s.priority.upper()}] {s.suggestion}")
        if s.rationale:
            lines.append(f"  Rationale: {s.rationale}")
    return "\n".join(lines)


def _format_prior_attempts(problem: SearchProblem) -> str:
    if not problem.prior_attempts:
        return "(none provided)"
    lines = []
    for a in problem.prior_attempts:
        lines.append(f"- {a.description}")
        lines.append(f"  Result: {a.result}")
        if a.parameter_range:
            lines.append(f"  Range: {a.parameter_range}")
        if a.what_was_ruled_out:
            lines.append(f"  Ruled out: {a.what_was_ruled_out}")
        if a.runtime:
            lines.append(f"  Runtime: {a.runtime}")
    return "\n".join(lines)


def _format_constraints(problem: SearchProblem) -> str:
    if not problem.search_constraints:
        return "(none provided)"
    lines = []
    for c in problem.search_constraints:
        lines.append(f"- {c.constraint}")
        if c.rationale:
            lines.append(f"  Rationale: {c.rationale}")
    return "\n".join(lines)


# ============================================================================
# Literature resolution
# ============================================================================


def _resolve_literature(phase_outputs: dict[str, str], phase_name: str) -> str:
    """Resolve literature content for a phase.

    If processed literature is available, return category-filtered content.
    Otherwise, fall back to the raw literature survey.
    """
    processed_json = phase_outputs.get("processed_literature")
    if not processed_json or processed_json == "(not yet available)":
        return phase_outputs.get("literature", "(not yet available)")

    categories = PHASE_LITERATURE_CATEGORIES.get(phase_name)
    if categories is None:
        # Phase not in the mapping: give full raw literature
        return phase_outputs.get("literature", "(not yet available)")

    if not categories:
        # Explicitly empty: phase does not need filtered literature
        return phase_outputs.get("literature", "(not yet available)")

    try:
        proc = ProcessedLiterature.model_validate_json(processed_json)
        header = f"[Filtered for: {', '.join(categories)}]\n\n"
        header += f"Executive Summary:\n{proc.executive_summary}\n\n"
        return header + proc.get_for_phase(categories)
    except Exception:
        logger.warning("Failed to deserialize ProcessedLiterature, falling back to raw")
        return phase_outputs.get("literature", "(not yet available)")


# ============================================================================
# Main builder
# ============================================================================


def build_phase_prompt(
    *,
    template: str,
    problem: SearchProblem,
    phase_outputs: dict[str, str],
    phase_name: str,
) -> str:
    """Compose a phase prompt from template + problem + context + prior outputs.

    The template contains {placeholders} that are filled from three sources:
      1. Problem fields (always: statement, win_condition, domain, etc.)
      2. Optional problem fields (controlled by PHASE_INCLUDES)
      3. Prior phase outputs (from phase_outputs dict)
    """
    includes = PHASE_INCLUDES.get(phase_name, {})

    # Format optional sections (controlled by phase)
    context_section = (
        _format_context_docs(problem) if includes.get("context_documents") else "(none)"
    )
    strategy_section = (
        _format_strategies(problem) if includes.get("strategy_suggestions") else "(none)"
    )
    prior_section = (
        _format_prior_attempts(problem) if includes.get("prior_attempts") else "(none)"
    )
    constraint_section = (
        _format_constraints(problem) if includes.get("search_constraints") else "(none)"
    )

    return template.format(
        # Always-available problem fields
        problem_statement=problem.statement,
        win_condition=problem.win_condition,
        domain=problem.domain or "(not specified)",
        background=problem.background or "(none)",
        known_bounds=problem.known_bounds or "(none)",
        # Optional problem fields (controlled by PHASE_INCLUDES)
        context_documents=context_section,
        strategy_suggestions=strategy_section,
        prior_attempts=prior_section,
        search_constraints=constraint_section,
        # Prior phase outputs (accumulated)
        literature_survey=_resolve_literature(phase_outputs, phase_name),
        processed_literature=phase_outputs.get("processed_literature", "(not yet available)"),
        structural_analysis=phase_outputs.get("analysis", "(not yet available)"),
        algorithm_document=phase_outputs.get("algorithm", "(not yet available)"),
        review_transcript=phase_outputs.get("reviews", "(not yet available)"),
        complexity_estimate=phase_outputs.get("complexity", "(not yet available)"),
        # For revision prompts: review feedback is injected separately
        review_feedback=phase_outputs.get("_current_review_feedback", "(none)"),
        # For document revision: the authoritative algorithm (pre-assembly)
        source_algorithm=phase_outputs.get("_source_algorithm", "(not available)"),
    )


# ============================================================================
# Phase templates
# ============================================================================

LITERATURE_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

======================================================================
TASK: Literature Research
======================================================================

Search for and synthesize what is known about this conjecture:

1. KNOWN RESULTS: What has been proved or disproved? What bounds are known?
2. PRIOR COMPUTATIONAL ATTEMPTS: Has anyone searched computationally?
   What ranges were covered? What methods were used?
3. RELATED PROBLEMS: Where have similar counterexamples been found?
   What techniques worked?
4. KNOWN OBSTRUCTIONS: Are there results that limit where counterexamples
   can exist?
5. PROMISING DIRECTIONS: Based on the literature, what approaches look
   most likely to succeed?

Avoid duplicating prior attempts listed above.

Structure your output as a survey with clear section headers.
"""

LITERATURE_PROCESSING_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

======================================================================
RAW LITERATURE SURVEY (from Phase 1):
======================================================================
{literature_survey}
======================================================================

======================================================================
TASK: Process and Structure Literature Findings
======================================================================

Transform the raw literature survey above into a structured, categorized,
confidence-scored synthesis. Follow the LITERATURE_PROCESSOR_CORE protocol.

Categorize every substantive finding into exactly one of:
- known_bounds
- prior_computational_attempts
- related_solved_problems
- construction_techniques
- obstructions

Output valid JSON matching this schema:
{{
  "findings": [
    {{
      "id": "finding_001",
      "claim": "...",
      "source_description": "...",
      "source_type": "journal_paper|arxiv_paper|preprint|textbook|survey|forum_post|blog|computational_report|unknown",
      "confidence": 0.0,
      "confidence_rationale": "...",
      "recency": "...",
      "category": "known_bounds|prior_computational_attempts|related_solved_problems|construction_techniques|obstructions",
      "relevance_score": 0.0,
      "corroborated_by": [],
      "contradicted_by": []
    }}
  ],
  "categories": {{
    "known_bounds": ["finding_001"],
    "prior_computational_attempts": [],
    "related_solved_problems": [],
    "construction_techniques": [],
    "obstructions": []
  }},
  "executive_summary": "...",
  "gaps_identified": ["..."],
  "contradictions": ["..."]
}}
"""

LITERATURE_AUDIT_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
RAW LITERATURE SURVEY (original, for comparison):
======================================================================
{literature_survey}
======================================================================

PROCESSED LITERATURE (to audit):
{processed_literature}

======================================================================
TASK: Audit Processed Literature
======================================================================

Compare the processed literature against the raw survey. Check for
completeness, accuracy, proper categorization, and identified contradictions.
Follow the LITERATURE_AUDITOR_CORE protocol.

Respond with JSON only.
"""

LITERATURE_REVISION_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

======================================================================
RAW LITERATURE SURVEY:
======================================================================
{literature_survey}
======================================================================

CURRENT PROCESSED LITERATURE:
{processed_literature}

AUDIT FEEDBACK:
{review_feedback}

======================================================================
TASK: Revise Processed Literature
======================================================================

Revise the processed literature to address ALL issues raised in the audit
feedback above. Produce the complete revised JSON (not just the changes).
Maintain the same schema.
"""

ANALYSIS_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

======================================================================
USER-PROVIDED ANALYSIS (read carefully --- this is expert context):
======================================================================
{context_documents}
======================================================================

STRATEGY SUGGESTIONS FROM USER:
{strategy_suggestions}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

LITERATURE SURVEY (from Phase 1):
{literature_survey}

======================================================================
TASK: Mathematical Structural Analysis
======================================================================

Perform a rigorous structural analysis of this conjecture. Determine what
a counterexample MUST look like, identify the most promising construction
families, derive the constraint equations, and state a search space
reduction theorem.

If user-provided analysis is present above, BUILD ON that work: verify it,
extend it, correct errors if any, and add what is missing. Do not re-derive
from scratch what the user has already correctly derived.

Follow the STRUCTURAL_ANALYST_CORE protocol for required output sections.
"""

ANALYSIS_REVIEW_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS (for consistency checking):
======================================================================
{context_documents}
======================================================================

STRUCTURAL ANALYSIS TO REVIEW:
{structural_analysis}

======================================================================
TASK: Review Structural Analysis
======================================================================

Review the structural analysis above for correctness and completeness.
Follow the ANALYSIS_REVIEWER_CORE protocol.

Respond with JSON only.
"""

ANALYSIS_REVISION_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

STRATEGY SUGGESTIONS FROM USER:
{strategy_suggestions}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

LITERATURE SURVEY:
{literature_survey}

CURRENT STRUCTURAL ANALYSIS:
{structural_analysis}

REVIEW FEEDBACK:
{review_feedback}

======================================================================
TASK: Revise Structural Analysis
======================================================================

Revise the structural analysis to address ALL issues raised in the review
feedback above. Produce the complete revised analysis (not just the changes).
"""

ALGORITHM_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

STRATEGY SUGGESTIONS FROM USER:
{strategy_suggestions}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

SEARCH CONSTRAINTS:
{search_constraints}

LITERATURE SURVEY:
{literature_survey}

STRUCTURAL ANALYSIS:
{structural_analysis}

======================================================================
TASK: Design Search Algorithm
======================================================================

Design a concrete, efficient, staged search algorithm based on the
structural analysis above. Follow the ALGORITHM_DESIGNER_CORE protocol.

The output must include complete, runnable Python code.
"""

CORRECTNESS_REVIEW_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS (for consistency checking):
======================================================================
{context_documents}
======================================================================

STRUCTURAL ANALYSIS:
{structural_analysis}

ALGORITHM AND CODE TO REVIEW:
{algorithm_document}

======================================================================
TASK: Correctness Review
======================================================================

Review the algorithm and code above for mathematical and computational
correctness. Follow the CORRECTNESS_REVIEWER_CORE protocol.

Respond with JSON only.
"""

OPTIMALITY_REVIEW_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

SEARCH CONSTRAINTS:
{search_constraints}

STRUCTURAL ANALYSIS:
{structural_analysis}

ALGORITHM AND CODE TO REVIEW:
{algorithm_document}

======================================================================
TASK: Optimality Review
======================================================================

You are seeing this algorithm for the first time. You did NOT design it.
Review it for optimality: is this the BEST approach, or is there a
better one? Follow the OPTIMALITY_REVIEWER_CORE protocol.

Respond with JSON only.
"""

ADVERSARIAL_REVIEW_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

STRUCTURAL ANALYSIS:
{structural_analysis}

ALGORITHM AND CODE TO ATTACK:
{algorithm_document}

======================================================================
TASK: Adversarial Review
======================================================================

Your goal is to find a scenario where this algorithm FAILS --- a
counterexample that exists but would be MISSED.
Follow the ADVERSARIAL_REVIEWER_CORE protocol.

Respond with JSON only.
"""

ALGORITHM_REVISION_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

STRATEGY SUGGESTIONS FROM USER:
{strategy_suggestions}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

SEARCH CONSTRAINTS:
{search_constraints}

STRUCTURAL ANALYSIS:
{structural_analysis}

CURRENT ALGORITHM AND CODE:
{algorithm_document}

REVIEW FEEDBACK:
{review_feedback}

======================================================================
TASK: Revise Algorithm
======================================================================

Revise the algorithm and code to address ALL issues raised in the review
feedback above. Produce the complete revised algorithm and code (not just
the changes). Maintain all sections: overview, stages, sufficiency,
optimality, exactification, and complete Python code.
"""

COMPLEXITY_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

PRIOR SEARCH ATTEMPTS AND RESULTS:
{prior_attempts}

SEARCH CONSTRAINTS:
{search_constraints}

ALGORITHM AND CODE:
{algorithm_document}

======================================================================
TASK: Complexity Estimation
======================================================================

Estimate the computational cost of the algorithm above.
Follow the COMPLEXITY_ESTIMATOR_CORE protocol.
"""

FEASIBILITY_CHECK_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

SEARCH CONSTRAINTS:
{search_constraints}

ALGORITHM AND CODE:
{algorithm_document}

COMPLEXITY ESTIMATE:
{complexity_estimate}

======================================================================
TASK: Feasibility Sanity Check
======================================================================

Perform a sanity check on the complexity estimate above.
Follow the FEASIBILITY_CHECKER_CORE protocol.

Respond with JSON only.
"""

ASSEMBLY_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

DOMAIN: {domain}
BACKGROUND: {background}
KNOWN BOUNDS: {known_bounds}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

LITERATURE SURVEY:
{literature_survey}

STRUCTURAL ANALYSIS:
{structural_analysis}

ALGORITHM AND CODE (final version):
{algorithm_document}

REVIEW TRANSCRIPT:
{review_transcript}

COMPLEXITY ESTIMATE:
{complexity_estimate}

======================================================================
TASK: Assemble Final Document
======================================================================

Assemble a self-contained counterexample search document from all the
material above. Follow the DOCUMENT_ASSEMBLER_CORE protocol.

The document must be copy-paste ready, with the complete runnable Python
code in a clearly marked section.
"""

FINAL_REVIEW_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS (for consistency checking):
======================================================================
{context_documents}
======================================================================

FINAL DOCUMENT TO REVIEW:
{algorithm_document}

======================================================================
TASK: Final Review
======================================================================

Review the assembled document for:
1) Internal consistency (do sections contradict each other?)
2) Completeness (are all required sections present?)
3) Code correctness (does the final code match the final algorithm?)
4) Readability (can a mathematician follow this without external docs?)

Respond with JSON only.

OUTPUT FORMAT:
{{
  "overall_verdict": "pass" | "revise",
  "critical_issues": [...],
  "major_issues": [...],
  "minor_issues": [...],
  "positive_aspects": ["..."],
  "summary": "..."
}}
"""

DOCUMENT_REVISION_TEMPLATE = """\
CONJECTURE:
{problem_statement}

WIN CONDITION:
{win_condition}

======================================================================
USER-PROVIDED ANALYSIS:
======================================================================
{context_documents}
======================================================================

CURRENT DOCUMENT (has issues identified below):
{algorithm_document}

REVIEW FEEDBACK (issues to fix):
{review_feedback}

======================================================================
SOURCE MATERIALS (use these to rebuild sections if needed):
======================================================================

FINAL ALGORITHM AND CODE (the authoritative version):
{source_algorithm}

STRUCTURAL ANALYSIS:
{structural_analysis}

REVIEW TRANSCRIPT:
{review_transcript}

COMPLEXITY ESTIMATE:
{complexity_estimate}

======================================================================
TASK: Revise Final Document
======================================================================

Revise the document to address ALL issues raised in the review feedback.

CRITICAL RULES:
1. Produce the COMPLETE revised document — NOT patches, diffs, or summaries.
2. The Code section (Section 4) must contain the FULL runnable Python code
   from the source algorithm above, with any fixes applied in-place.
3. If the current document is missing content that exists in the source
   materials, restore it.
4. The revised document must be self-contained and at least as long as the
   current document. If you find yourself producing a short summary, STOP
   and rebuild from the source materials instead.
5. Every section from the DOCUMENT_ASSEMBLER_CORE structure must be present
   and substantive.
"""
