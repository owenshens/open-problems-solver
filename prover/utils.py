"""Prover-specific utilities.

Domain-specific helpers for the proof workflow: prompt loading, topological sorting,
interface formatting, context management, and literature resolution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from prover import config
from prover.models import Lemma, ContextDocument
from shared.utils import now_timestamp


def load_prompt_template(name: str, prompts_dir: Optional[str | Path] = None) -> str:
    """Load a prompt template file by name."""
    prompts_path = Path(prompts_dir or config.PROMPTS_DIR)
    path = prompts_path / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def validate_outline_constraints(lemmas: list[Lemma]) -> None:
    """Validate constraints for the initial decomposition."""
    if len(lemmas) == 0:
        raise ValueError("Outline contains zero lemmas")
    if len(lemmas) > config.MAX_LEMMAS:
        raise ValueError(f"Too many lemmas: {len(lemmas)} > {config.MAX_LEMMAS}")

    lemma_ids = {l.id for l in lemmas}
    for l in lemmas:
        if len(l.dependencies) > config.MAX_FANOUT:
            raise ValueError(
                f"Lemma {l.id} has too many dependencies: {len(l.dependencies)} > {config.MAX_FANOUT}"
            )
        for dep in l.dependencies:
            if dep not in lemma_ids:
                raise ValueError(f"Invalid dependency {dep!r} in lemma {l.id}")
            if dep == l.id:
                raise ValueError(f"Lemma {l.id} depends on itself")

    _ = topo_sort_lemmas(lemmas)  # Raises on cycle


def topo_sort_lemmas(lemmas: list[Lemma]) -> list[Lemma]:
    """Topologically sort lemmas by their dependencies. Raises ValueError on cycles."""
    by_id = {l.id: l for l in lemmas}
    indeg: dict[str, int] = {l.id: 0 for l in lemmas}
    out: dict[str, list[str]] = {l.id: [] for l in lemmas}

    for l in lemmas:
        for dep in l.dependencies:
            indeg[l.id] += 1
            out[dep].append(l.id)

    queue = [lid for lid, d in indeg.items() if d == 0]
    order: list[str] = []

    while queue:
        lid = queue.pop(0)
        order.append(lid)
        for nxt in out[lid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(lemmas):
        remaining = [lid for lid, d in indeg.items() if d > 0]
        raise ValueError(f"Dependency cycle detected among lemmas: {remaining}")

    return [by_id[lid] for lid in order]


def format_dependency_interfaces(dep_interfaces: dict[str, str]) -> str:
    if not dep_interfaces:
        return "(none)"
    chunks = []
    for dep_id, iface in dep_interfaces.items():
        chunks.append(f"### {dep_id}\n{iface}")
    return "\n\n".join(chunks)


def format_all_interfaces(lemmas: Iterable[Lemma]) -> str:
    parts = []
    for l in lemmas:
        if l.interface:
            parts.append(l.interface)
        else:
            parts.append(f"**Lemma {l.id}:** (no interface extracted)")
    return "\n\n".join(parts)


@dataclass
class Stopwatch:
    """Lightweight wall-clock timer."""
    start: float = 0.0

    def __post_init__(self) -> None:
        if self.start == 0.0:
            self.start = time.time()

    def reset(self) -> None:
        self.start = time.time()

    def elapsed_s(self) -> float:
        return time.time() - self.start


# ============================================================================
# Context Management for Recursive Grammar
# ============================================================================


def create_context_document(lemma: Lemma) -> ContextDocument:
    """Create initial context document for a lemma."""
    timestamp = now_timestamp()
    return ContextDocument(
        lemma_id=lemma.id,
        parent_lemma_id=lemma.parent_lemma_id,
        depth=lemma.depth,
        created_at=timestamp,
        updated_at=timestamp,
    )


def propagate_context_from_parent(
    child_lemma: Lemma,
    parent_context: ContextDocument,
) -> None:
    """Propagate relevant context from parent to child lemma."""
    if child_lemma.context is None:
        child_lemma.context = create_context_document(child_lemma)

    child_lemma.context.failed_strategies.extend(
        [f"parent:{s}" for s in parent_context.failed_strategies]
    )
    child_lemma.context.successful_approaches.extend(
        parent_context.successful_approaches
    )
    child_lemma.context.insights.extend(
        [f"(from parent) {insight}" for insight in parent_context.insights]
    )
    child_lemma.context.updated_at = now_timestamp()


def record_attempt(
    lemma: Lemma,
    attempt_type: str,
    result: dict[str, Any],
) -> None:
    """Record a proof attempt in the lemma's context."""
    if lemma.context is None:
        lemma.context = create_context_document(lemma)

    timestamp = now_timestamp()
    record = {"timestamp": timestamp, **result}

    if attempt_type == "direct":
        lemma.context.direct_attempts.append(record)
    elif attempt_type == "decompose":
        lemma.context.decomposition_attempts.append(record)
    else:
        raise ValueError(f"Unknown attempt_type: {attempt_type}")

    lemma.context.updated_at = timestamp


def format_context_for_prompt(context: Optional[ContextDocument]) -> str:
    """Format context document into human-readable text for LLM prompts."""
    if not context:
        return "(no prior context)"

    parts = []

    if context.failed_strategies:
        parts.append("**Previously failed strategies:**")
        for s in context.failed_strategies[-3:]:
            parts.append(f"- {s}")

    if context.audit_failures:
        parts.append("\n**Previous audit failures:**")
        for af in context.audit_failures[-2:]:
            feedback = af.get('feedback', '')
            if feedback:
                truncated = feedback[:200] + ("..." if len(feedback) > 200 else "")
                parts.append(f"- {truncated}")

    if context.integration_failures:
        parts.append("\n**Previous integration failures:**")
        for ifl in context.integration_failures[-2:]:
            strategy = ifl.get('strategy', 'unknown')
            feedback = ifl.get('feedback', '')
            parts.append(f"- Strategy '{strategy}': {feedback}")

    if context.successful_approaches:
        parts.append("\n**Successful approaches from related lemmas:**")
        for sa in context.successful_approaches[:2]:
            parts.append(f"- {sa}")

    if context.insights:
        parts.append("\n**Accumulated insights:**")
        for insight in context.insights[-2:]:
            parts.append(f"- {insight}")

    return "\n".join(parts) if parts else "(no actionable context)"


# ============================================================================
# Selective Literature Retrieval
# ============================================================================

PHASE_LITERATURE_CATEGORIES: dict[str, list[str]] = {
    "draft_plan": [
        "known_proof_strategies",
        "existing_partial_proofs",
        "analogous_proved_theorems",
        "proof_techniques_and_tools",
        "known_obstacles_and_pitfalls",
    ],
    "audit_plan": [
        "known_proof_strategies",
        "known_obstacles_and_pitfalls",
    ],
    "prove_lemma": [
        "proof_techniques_and_tools",
        "analogous_proved_theorems",
    ],
    "self_audit": [
        "known_obstacles_and_pitfalls",
        "existing_partial_proofs",
    ],
    "cross_audit": [
        "known_proof_strategies",
        "known_obstacles_and_pitfalls",
    ],
    "integrate_proof": [
        "known_proof_strategies",
    ],
    "final_audit": [
        "known_proof_strategies",
        "known_obstacles_and_pitfalls",
        "existing_partial_proofs",
        "analogous_proved_theorems",
        "proof_techniques_and_tools",
    ],
}


def resolve_literature_for_phase(
    processed_literature: Optional[Any],
    phase_name: str,
) -> str:
    """Resolve processed literature content for a specific phase."""
    if processed_literature is None:
        return ""
    categories = PHASE_LITERATURE_CATEGORIES.get(phase_name)
    if not categories:
        return ""
    try:
        header = f"[Filtered literature for: {', '.join(categories)}]\n\n"
        header += f"Executive Summary:\n{processed_literature.executive_summary}\n\n"
        return header + processed_literature.get_for_phase(categories)
    except Exception:
        return ""
