"""Data models for the counterexample search system."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Input models
# ============================================================================


class StrategySuggestion(BaseModel):
    """User-provided hint about where or how to search."""

    suggestion: str  # "Try concentric N-gons with 3 layers"
    rationale: str = ""  # "Near-misses at N=127 suggest more layers help"
    priority: str = "medium"  # "high" / "medium" / "low"


class PriorAttempt(BaseModel):
    """Record of a previous search attempt."""

    description: str  # "Exhaustive 2-layer C_N scan for odd N"
    result: str  # "No counterexample found"
    parameter_range: str = ""  # "N in [3, 249]"
    runtime: str = ""  # "~8 hours on M2 MacBook"
    what_was_ruled_out: str = ""  # "All 2-layer C_N configs with odd N <= 249"


class SearchConstraint(BaseModel):
    """Bound on the search (parameter range, runtime budget, etc.)."""

    constraint: str  # "Focus on N <= 500"
    rationale: str = ""  # "Computational budget is 48 hours"


class SearchProblem(BaseModel):
    """Rich input for counterexample search.

    Required fields: id, statement, win_condition.
    All other fields are optional and provide additional context
    that flows through the pipeline to improve output quality.
    """

    # === REQUIRED ===
    id: str
    statement: str  # Precise conjecture statement
    win_condition: str  # What constitutes a valid counterexample

    # === OPTIONAL: METADATA ===
    domain: str = ""  # "Combinatorial Geometry", "Graph Theory", ...
    background: str = ""  # Known results, bounds
    tags: list[str] = Field(default_factory=list)
    known_bounds: str = ""
    falsifiable: bool = True

    # === OPTIONAL: RICH CONTEXT ===
    context_documents: list[str] = Field(default_factory=list)
    # Long-form mathematical analysis documents. Each entry can be thousands
    # of words. Injected IN FULL into prompts — not summarized.
    # The system builds ON this analysis rather than re-deriving.

    # === OPTIONAL: STRATEGY GUIDANCE ===
    strategy_suggestions: list[StrategySuggestion] = Field(default_factory=list)

    # === OPTIONAL: PRIOR WORK ===
    prior_attempts: list[PriorAttempt] = Field(default_factory=list)
    # The system will NOT re-search these ranges.

    # === OPTIONAL: COMPUTATIONAL BUDGET ===
    search_constraints: list[SearchConstraint] = Field(default_factory=list)


# ============================================================================
# Review models (used by controller for branching)
# ============================================================================


class Issue(BaseModel):
    """A specific issue found during review."""

    description: str
    why_it_matters: str
    suggested_fix: str
    severity: str = "major"  # "critical" / "major" / "minor"


class ReviewResult(BaseModel):
    """Output of a review round. Parsed from LLM JSON response."""

    round_name: str = ""  # "correctness", "optimality", "adversarial", ...
    reviewer: str = ""  # "claude" or "gpt"
    overall_verdict: str = "pass"  # "pass" / "revise" / "redesign"
    critical_issues: list[Issue] = Field(default_factory=list)
    major_issues: list[Issue] = Field(default_factory=list)
    minor_issues: list[Issue] = Field(default_factory=list)
    positive_aspects: list[str] = Field(default_factory=list)
    summary: str = ""

    def has_blocking_issues(self) -> bool:
        return self.overall_verdict != "pass"

    def format_for_prompt(self) -> str:
        """Format review result as readable text for injection into revision prompts."""
        parts = [f"REVIEW VERDICT: {self.overall_verdict.upper()}"]

        if self.critical_issues:
            parts.append("\nCRITICAL ISSUES (must fix):")
            for i, issue in enumerate(self.critical_issues, 1):
                parts.append(f"  {i}. {issue.description}")
                parts.append(f"     Why: {issue.why_it_matters}")
                parts.append(f"     Fix: {issue.suggested_fix}")

        if self.major_issues:
            parts.append("\nMAJOR ISSUES (should fix):")
            for i, issue in enumerate(self.major_issues, 1):
                parts.append(f"  {i}. {issue.description}")
                parts.append(f"     Why: {issue.why_it_matters}")
                parts.append(f"     Fix: {issue.suggested_fix}")

        if self.minor_issues:
            parts.append("\nMINOR ISSUES (nice to fix):")
            for i, issue in enumerate(self.minor_issues, 1):
                parts.append(f"  {i}. {issue.description}")

        if self.positive_aspects:
            parts.append("\nPOSITIVE ASPECTS:")
            for aspect in self.positive_aspects:
                parts.append(f"  - {aspect}")

        if self.summary:
            parts.append(f"\nSUMMARY: {self.summary}")

        return "\n".join(parts)


# ============================================================================
# Literature processing models (Phase 1.5)
# ============================================================================


LITERATURE_CATEGORIES = [
    "known_bounds",
    "prior_computational_attempts",
    "related_solved_problems",
    "construction_techniques",
    "obstructions",
]


class LiteratureFinding(BaseModel):
    """A single piece of information extracted from literature research."""

    id: str  # "finding_001"
    claim: str  # The factual claim
    source_description: str  # "Guth-Katz 2015, Annals of Mathematics"
    source_type: str  # arxiv_paper|journal_paper|textbook|survey|forum_post|blog|computational_report|unknown
    confidence: float  # 0.0-1.0
    confidence_rationale: str
    recency: str = ""
    category: str  # One of LITERATURE_CATEGORIES
    relevance_score: float = 1.0
    corroborated_by: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)


class ProcessedLiterature(BaseModel):
    """Structured, categorized, confidence-scored literature synthesis."""

    findings: list[LiteratureFinding] = Field(default_factory=list)
    categories: dict[str, list[str]] = Field(default_factory=dict)  # category -> finding IDs
    executive_summary: str = ""
    gaps_identified: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    def get_category(self, category: str) -> str:
        """Format findings for one category."""
        ids = self.categories.get(category, [])
        findings = [f for f in self.findings if f.id in ids]
        if not findings:
            return f"(no findings in category: {category})"
        lines = []
        for f in sorted(findings, key=lambda x: x.confidence, reverse=True):
            label = "HIGH" if f.confidence >= 0.8 else "MEDIUM" if f.confidence >= 0.5 else "LOW"
            lines.append(f"- [{label}] {f.claim}")
            lines.append(f"  Source: {f.source_description} ({f.source_type})")
            if f.contradicted_by:
                lines.append(f"  WARNING: Contradicted by {f.contradicted_by}")
        return "\n".join(lines)

    def get_for_phase(self, categories: list[str]) -> str:
        """Format multiple categories for prompt injection."""
        sections = []
        for cat in categories:
            text = self.get_category(cat)
            sections.append(f"### {cat.replace('_', ' ').title()}\n{text}")
        return "\n\n".join(sections)

    def full_text(self) -> str:
        """Full markdown for backward compat and human readability."""
        parts = [f"# Processed Literature\n\n{self.executive_summary}"]
        for cat in LITERATURE_CATEGORIES:
            section = self.get_category(cat)
            if "(no findings" not in section:
                parts.append(f"\n## {cat.replace('_', ' ').title()}\n{section}")
        if self.gaps_identified:
            parts.append("\n## Gaps\n" + "\n".join(f"- {g}" for g in self.gaps_identified))
        if self.contradictions:
            parts.append("\n## Contradictions\n" + "\n".join(f"- {c}" for c in self.contradictions))
        return "\n".join(parts)
