from __future__ import annotations

from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field


class LemmaStatus(str, Enum):
    OPEN = "open"
    CLAIMED = "claimed"
    CONDITIONAL = "conditional"
    PROVED = "proved"
    BLOCKED = "blocked"


class AuditVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class FailureCategory(str, Enum):
    """Classification of why a proof attempt failed."""
    DEPENDENCY_FAILURE = "dependency_failure"
    LOGICAL_GAP = "logical_gap"
    INTEGRATION_MISMATCH = "integration_mismatch"
    COMPLEXITY_BARRIER = "complexity_barrier"
    TECHNIQUE_INADEQUATE = "technique_inadequate"
    ASSUMPTION_VIOLATION = "assumption_violation"
    AUDIT_REJECTION = "audit_rejection"
    DECOMPOSITION_FAILURE = "decomposition_failure"
    MAX_RETRIES_EXHAUSTED = "max_retries_exhausted"


class StrategyType(str, Enum):
    """Types of proof and decomposition strategies."""
    # Direct proof techniques
    DIRECT_PROOF = "direct_proof"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CASE_ANALYSIS = "case_analysis"
    CONSTRUCTION = "construction"
    CONTRAPOSITIVE = "contrapositive"
    # Decomposition strategies
    HORIZONTAL_SPLIT = "horizontal_split"
    VERTICAL_CHAIN = "vertical_chain"
    HELPER_LEMMA = "helper_lemma"
    STRENGTHEN_CLAIM = "strengthen_claim"
    WEAKEN_ASSUMPTIONS = "weaken_assumptions"
    GRANULAR_DECOMPOSE = "granular_decompose"


class Problem(BaseModel):
    """Input problem specification."""

    id: str
    statement: str
    background: str
    domain: str
    tags: list[str] = Field(default_factory=list)


class Lemma(BaseModel):
    """Individual proof obligation with recursive decomposition support."""

    id: str
    statement: str
    assumptions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: LemmaStatus = LemmaStatus.OPEN
    proof: Optional[str] = None
    interface: Optional[str] = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0

    # Recursive decomposition fields
    depth: int = 0  # 0 = top-level, 1 = sub-lemma, 2 = sub-sub-lemma, etc.
    parent_lemma_id: Optional[str] = None  # ID of parent lemma (if sub-lemma)
    sub_lemmas: list[str] = Field(default_factory=list)  # IDs of child lemmas (if decomposed)

    # Context tracking for recursive grammar and backtracking
    context: Optional["ContextDocument"] = None  # Context accumulated during attempts
    decomposition_strategies: list["DecompositionStrategy"] = Field(default_factory=list)
    current_strategy_index: int = 0  # Which strategy is currently being attempted


class Outline(BaseModel):
    """Decomposition plan."""

    problem_id: str
    lemmas: list[Lemma]
    proof_strategy: str = ""


class DecompositionStrategy(BaseModel):
    """A specific decomposition strategy with rationale and tracking."""

    strategy_id: str  # e.g., "L1_strat1"
    rationale: str  # Why this strategy might work
    sub_lemmas: list[str] = Field(default_factory=list)  # IDs of generated sub-lemmas
    context_used: list[str] = Field(default_factory=list)  # What context informed this
    status: str = "untried"  # "untried" | "in_progress" | "failed" | "succeeded"


class ContextDocument(BaseModel):
    """Context accumulated during proof attempts for a single lemma."""

    lemma_id: str
    parent_lemma_id: Optional[str] = None
    depth: int = 0

    # Attempt tracking
    direct_attempts: list[dict[str, Any]] = Field(default_factory=list)
    decomposition_attempts: list[dict[str, Any]] = Field(default_factory=list)

    # Failure history
    failed_strategies: list[str] = Field(default_factory=list)
    audit_failures: list[dict[str, Any]] = Field(default_factory=list)
    integration_failures: list[dict[str, Any]] = Field(default_factory=list)

    # Learnings
    successful_approaches: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)

    # Metadata
    created_at: str = ""
    updated_at: str = ""


class AuditResult(BaseModel):
    """Audit verdict."""

    lemma_id: str
    auditor: str  # "gpt-self" | "claude-cross" | "claude-final"
    verdict: AuditVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str | dict] = Field(default_factory=list)  # Accept both strings and structured dicts
    feedback: str = ""
    # Optional extra fields used by final audit
    recommendation: Optional[str] = None


class ProofArtifact(BaseModel):
    """Final proof output."""

    problem_id: str
    status: str  # "complete", "partial", "failed"
    proof_text: str
    lemmas: list[Lemma]
    final_audit: Optional[AuditResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Literature processing models (Phase 0.5)
# ============================================================================


PROOF_LITERATURE_CATEGORIES = [
    "known_proof_strategies",
    "existing_partial_proofs",
    "analogous_proved_theorems",
    "proof_techniques_and_tools",
    "known_obstacles_and_pitfalls",
]


class LiteratureFinding(BaseModel):
    """A single piece of information extracted from literature research."""

    id: str  # "finding_001"
    claim: str  # The factual claim
    source_description: str  # "Rudin 1976, Principles of Mathematical Analysis"
    source_type: str  # journal_paper|arxiv_paper|textbook|survey|forum_post|blog|lecture_notes|unknown
    confidence: float  # 0.0-1.0
    confidence_rationale: str
    recency: str = ""
    category: str  # One of PROOF_LITERATURE_CATEGORIES
    relevance_score: float = 1.0
    actionability: str = "medium"  # "high" (concrete lemma) / "medium" (technique) / "low" (background)
    corroborated_by: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)


class ProcessedLiterature(BaseModel):
    """Structured, categorized, confidence-scored literature synthesis for proofs."""

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
            act = " [actionable]" if f.actionability == "high" else (" [background]" if f.actionability == "low" else "")
            lines.append(f"- [{label}]{act} {f.claim}")
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
        """Full markdown for human readability."""
        parts = [f"# Processed Literature\n\n{self.executive_summary}"]
        for cat in PROOF_LITERATURE_CATEGORIES:
            section = self.get_category(cat)
            if "(no findings" not in section:
                parts.append(f"\n## {cat.replace('_', ' ').title()}\n{section}")
        if self.gaps_identified:
            parts.append("\n## Gaps\n" + "\n".join(f"- {g}" for g in self.gaps_identified))
        if self.contradictions:
            parts.append("\n## Contradictions\n" + "\n".join(f"- {c}" for c in self.contradictions))
        return "\n".join(parts)


class LiteratureReviewResult(BaseModel):
    """Output of a literature audit round."""

    overall_verdict: str = "pass"  # "pass" / "revise"
    critical_issues: list[dict[str, str]] = Field(default_factory=list)
    major_issues: list[dict[str, str]] = Field(default_factory=list)
    minor_issues: list[dict[str, str]] = Field(default_factory=list)
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
                parts.append(f"  {i}. {issue.get('description', '')}")
                parts.append(f"     Why: {issue.get('why_it_matters', '')}")
                parts.append(f"     Fix: {issue.get('suggested_fix', '')}")

        if self.major_issues:
            parts.append("\nMAJOR ISSUES (should fix):")
            for i, issue in enumerate(self.major_issues, 1):
                parts.append(f"  {i}. {issue.get('description', '')}")
                parts.append(f"     Why: {issue.get('why_it_matters', '')}")
                parts.append(f"     Fix: {issue.get('suggested_fix', '')}")

        if self.minor_issues:
            parts.append("\nMINOR ISSUES (nice to fix):")
            for i, issue in enumerate(self.minor_issues, 1):
                parts.append(f"  {i}. {issue.get('description', '')}")

        if self.positive_aspects:
            parts.append("\nPOSITIVE ASPECTS:")
            for aspect in self.positive_aspects:
                parts.append(f"  - {aspect}")

        if self.summary:
            parts.append(f"\nSUMMARY: {self.summary}")

        return "\n".join(parts)
