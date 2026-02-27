"""Prover operators with thinking cores integration.

All operators now use:
- Thinking cores (strict protocols) as system prompts
- Domain-specific profiles for pitfalls and audit tests
- Enhanced prompts with step-labeled proofs
- Structured audit results
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from prover import config
from shared.agents import call_gpt, call_claude, AgentResponse, web_search_formatted
from shared.utils import parse_json_robust, extract_json_from_response, merge_usage
from prover.models import Problem, Outline, Lemma, LemmaStatus, AuditResult, AuditVerdict
from prover.thinking_cores import (
    GPT_PROVER_CORE,
    GPT_PLANNER_CORE,
    GPT_SELF_AUDIT_CORE,
    GPT_LITERATURE_RESEARCH_CORE,
    CLAUDE_CROSS_AUDIT_CORE,
    CLAUDE_PLAN_AUDIT_CORE,
    CLAUDE_FINAL_AUDIT_CORE,
    format_domain_pitfalls,
    format_domain_audit_tests,
)
from prover.utils import (
    load_prompt_template,
    validate_outline_constraints,
    format_dependency_interfaces,
    format_all_interfaces,
    topo_sort_lemmas,
)


def _log_io(run_dir: Optional[Path], name: str, *, prompt: str, response: str, thinking: Optional[str] = None) -> None:
    """Log prompt, response, and thinking (if available) to files."""
    if not run_dir:
        return
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    (log_dir / f"{safe_name}.prompt.txt").write_text(prompt, encoding="utf-8")
    (log_dir / f"{safe_name}.response.txt").write_text(response, encoding="utf-8")
    if thinking:
        (log_dir / f"{safe_name}.thinking.txt").write_text(thinking, encoding="utf-8")


def _coerce_verdict(value: object) -> AuditVerdict:
    v = str(value or "UNCERTAIN").strip().upper()
    if v in {"PASS", "FAIL", "UNCERTAIN"}:
        return AuditVerdict(v)
    return AuditVerdict.UNCERTAIN


def literature_research(
    problem: Problem,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Perform comprehensive literature research before planning."""

    if not config.ENABLE_LITERATURE_RESEARCH:
        return "(Literature research disabled)", {}

    template = load_prompt_template("literature_research", prompts_dir=config.PROMPTS_DIR)

    # Generate search queries
    search_queries = [
        f"{problem.statement} proof",
        f"{problem.statement} theorem mathematics",
        f"{problem.domain} {problem.statement}",
        f"standard proof techniques {problem.domain}",
    ]

    # Perform web searches for mathematical research
    search_results_text = ""
    if config.ENABLE_WEB_SEARCH:
        try:
            search_parts = []
            for query in search_queries[:4]:  # Limit to 4 queries to stay within reasonable time
                results = web_search_formatted(query, max_results=config.MAX_SEARCH_RESULTS)
                search_parts.append(results)
            search_results_text = "\n\n".join(search_parts)

            if not search_results_text.strip():
                search_results_text = "(Web search returned no results)"
        except Exception as e:
            search_results_text = f"(Web search failed: {e})"
    else:
        search_results_text = "(Web search disabled)"

    # Format prompt
    prompt = template.format(
        problem_statement=problem.statement,
        problem_background=problem.background,
        problem_domain=problem.domain,
        search_results=search_results_text,
    )

    # Call GPT with literature research core
    resp: AgentResponse = call_gpt(
        prompt,
        system_prompt=GPT_LITERATURE_RESEARCH_CORE,
        reasoning_effort=config.REASONING_EFFORT_LITERATURE,
        text_verbosity="high",
        max_tokens=config.MAX_LITERATURE_SEARCH_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, "literature_research", prompt=prompt, response=resp.text, thinking=resp.thinking)

    return resp.text, resp.usage


def draft_global_plan(problem: Problem, *, run_dir: Optional[Path] = None, retry_feedback: str = "", literature_context: str = "") -> tuple[Outline, dict[str, int]]:
    """Generate initial lemma decomposition plan with GPT_PLANNER_CORE.

    Args:
        problem: Problem to plan for
        run_dir: Directory to save logs
        retry_feedback: Feedback from previous planning attempts (if retrying)
        literature_context: Filtered literature for planning phase
    """

    template = load_prompt_template("draft_plan", prompts_dir=config.PROMPTS_DIR)

    fmt_kwargs = dict(
        problem_statement=problem.statement,
        problem_background=problem.background,
        problem_domain=problem.domain,
        literature_context=literature_context or "(none)",
    )

    # Incorporate retry feedback if provided
    if retry_feedback:
        prompt_with_feedback = retry_feedback + "\n\n" + template
        prompt = prompt_with_feedback.format(**fmt_kwargs)
    else:
        prompt = template.format(**fmt_kwargs)

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        max_tokens=config.MAX_PLAN_TOKENS,
        system_prompt=GPT_PLANNER_CORE,
        reasoning_effort=config.REASONING_EFFORT_PLAN,
        text_verbosity=config.TEXT_VERBOSITY_PLAN,
        use_background=False,  # Planning is usually fast
    )
    _log_io(run_dir, "draft_global_plan", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    lemmas: list[Lemma] = []
    for lem in data.get("lemmas", []):
        lemmas.append(
            Lemma(
                id=str(lem.get("id")),
                statement=str(lem.get("statement")),
                assumptions=list(lem.get("assumptions", []) or []),
                dependencies=list(lem.get("dependencies", []) or []),
                status=LemmaStatus.OPEN,
            )
        )

    validate_outline_constraints(lemmas)

    outline = Outline(
        problem_id=problem.id,
        lemmas=lemmas,
        proof_strategy=str(data.get("proof_strategy", "")),
    )

    return outline, resp.usage


def audit_plan(
    problem: Problem,
    outline: Outline,
    *,
    run_dir: Optional[Path] = None,
    literature_context: str = "",
) -> tuple[AuditResult, dict[str, int]]:
    """Audit the decomposition plan with CLAUDE_PLAN_AUDIT_CORE before proving begins."""

    template = load_prompt_template("audit_plan", prompts_dir=config.PROMPTS_DIR)

    # Format lemmas for the prompt
    proposed_lemmas = "\n\n".join([
        f"**Lemma {l.id}:**\n"
        f"Statement: {l.statement}\n"
        f"Assumptions: {l.assumptions}\n"
        f"Dependencies: {l.dependencies}"
        for l in outline.lemmas
    ])

    prompt = template.format(
        problem_statement=problem.statement,
        problem_background=problem.background,
        problem_domain=problem.domain,
        proposed_lemmas=proposed_lemmas,
        proof_strategy=outline.proof_strategy,
        literature_context=literature_context or "(none)",
    )

    resp: AgentResponse = call_claude(
        prompt,
        json_mode=True,
        max_tokens=config.MAX_AUDIT_TOKENS,
        system_prompt=CLAUDE_PLAN_AUDIT_CORE,
    )
    _log_io(run_dir, "audit_plan", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    result = AuditResult(
        lemma_id="PLAN",
        auditor="claude-plan",
        verdict=_coerce_verdict(data.get("verdict")),
        confidence=float(data.get("confidence", 0.0)),
        issues=[
            {
                "severity": str(issue.get("severity", "minor")),
                "location": str(issue.get("lemma_id", "OVERALL")),
                "summary": str(issue.get("summary", "")),
                "detail": str(issue.get("detail", "")),
                "patch": str(issue.get("patch", "")),
            }
            for issue in data.get("issues", [])
        ],
        feedback=str(data.get("recommendation", "")),
    )

    return result, resp.usage


def decompose_blocked_lemma(
    lemma: Lemma,
    problem: Problem,
    depth: int,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[Outline, dict[str, int]]:
    """Decompose a BLOCKED lemma into simpler sub-lemmas."""

    template = load_prompt_template("decompose_blocked_lemma", prompts_dir=config.PROMPTS_DIR)

    # Extract blocking reason from evidence
    blocking_reason = "Could not prove directly (max retries exceeded)"
    if lemma.evidence:
        for ev in reversed(lemma.evidence):
            if ev.get("feedback"):
                blocking_reason = ev["feedback"]
                break

    prompt = template.format(
        parent_lemma_id=lemma.id,
        parent_lemma_statement=lemma.statement,
        parent_lemma_assumptions="\n".join(f"- {a}" for a in lemma.assumptions) if lemma.assumptions else "(none)",
        parent_lemma_dependencies=", ".join(lemma.dependencies) if lemma.dependencies else "(none)",
        blocking_reason=blocking_reason,
        problem_background=problem.background,
        problem_domain=problem.domain,
        current_depth=depth,
        max_depth=config.MAX_DEPTH,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        system_prompt=GPT_PLANNER_CORE,  # Reuse planner core for decomposition
        json_mode=True,
        reasoning_effort="xhigh",  # Deep thinking for good decomposition
        text_verbosity=config.TEXT_VERBOSITY_PLAN,  # MAXIMIZED to high
        max_tokens=config.MAX_PLAN_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, f"decompose_{lemma.id}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    # Parse sub-lemmas with parent linkage
    sub_lemmas: list[Lemma] = []
    for lem in data.get("lemmas", []):
        sub_lemmas.append(
            Lemma(
                id=f"{lemma.id}.{lem['id']}",  # e.g., L1.1, L1.2
                statement=str(lem.get("statement")),
                assumptions=list(lem.get("assumptions", []) or []),
                dependencies=[f"{lemma.id}.{dep}" if dep in [l['id'] for l in data.get("lemmas", [])] else dep
                             for dep in lem.get("dependencies", [])],
                status=LemmaStatus.OPEN,
                depth=depth + 1,
                parent_lemma_id=lemma.id,
            )
        )

    # Validate sub-lemma decomposition
    if sub_lemmas:
        validate_outline_constraints(sub_lemmas)

    outline = Outline(
        problem_id=problem.id,
        lemmas=sub_lemmas,
        proof_strategy=str(data.get("proof_strategy", "")),
    )

    return outline, resp.usage


def decide_proof_strategy(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Decision operator: choose "direct" proof or "decompose" into sub-problems.

    Uses GPT with high reasoning to analyze:
    - Lemma complexity (length, quantifiers, nesting)
    - Available dependencies and interfaces
    - Past failures from context
    - Current depth vs max depth
    - Resource constraints

    Returns:
        ("direct", usage) or ("decompose", usage)
    """
    from prover.utils import format_context_for_prompt

    template = load_prompt_template("decide_strategy", prompts_dir=config.PROMPTS_DIR)

    # Format context for decision-making
    context_summary = format_context_for_prompt(lemma.context)

    prompt = template.format(
        lemma_id=lemma.id,
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        dependency_interfaces=format_dependency_interfaces(dependency_interfaces),
        problem_background=problem.background,
        problem_domain=problem.domain,
        context_summary=context_summary,
        current_depth=lemma.depth,
        max_depth=config.MAX_DEPTH,
        retry_count=lemma.retry_count,
        max_retries=config.MAX_RETRIES,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        system_prompt=GPT_PLANNER_CORE,  # Reuse planner core for strategic decisions
        reasoning_effort=config.REASONING_EFFORT_PLAN,  # Use xhigh for strategic thinking
        text_verbosity=config.TEXT_VERBOSITY_PLAN,
        max_tokens=config.MAX_DECISION_TOKENS,  # MAXIMIZED to 127998 for full strategic analysis
        use_background=False,  # Decision should be fast
    )

    _log_io(run_dir, f"decide_strategy_{lemma.id}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)
    decision = data.get("decision", "direct").lower()

    # Validate decision
    if decision not in {"direct", "decompose"}:
        # Fallback to safe default
        decision = "direct"

    return decision, resp.usage


def generate_decomposition_strategies(
    lemma: Lemma,
    problem: Problem,
    *,
    run_dir: Optional[Path] = None,
    num_strategies: int = 3,
) -> tuple[list, dict[str, int]]:
    """Generate multiple alternative decomposition strategies.

    Uses context to avoid previously failed approaches.
    Returns strategies ordered by estimated promise.

    Returns:
        (list[DecompositionStrategy], usage)
    """
    from prover.models import DecompositionStrategy
    from prover.utils import format_context_for_prompt

    template = load_prompt_template("generate_strategies", prompts_dir=config.PROMPTS_DIR)

    # Extract context and blocking reason
    context_summary = format_context_for_prompt(lemma.context)

    # Extract blocking reason from evidence
    blocking_reason = "Unknown - direct proof attempts exhausted"
    if lemma.evidence:
        for ev in reversed(lemma.evidence):
            if ev.get("feedback"):
                blocking_reason = ev["feedback"]
                break

    prompt = template.format(
        lemma_id=lemma.id,
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        blocking_reason=blocking_reason,
        context_summary=context_summary,
        problem_background=problem.background,
        problem_domain=problem.domain,
        current_depth=lemma.depth,
        max_depth=config.MAX_DEPTH,
        num_strategies=num_strategies,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        system_prompt=GPT_PLANNER_CORE,
        reasoning_effort=config.REASONING_EFFORT_PLAN,  # xhigh for strategic planning
        text_verbosity=config.TEXT_VERBOSITY_PLAN,
        max_tokens=config.MAX_PLAN_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )

    _log_io(run_dir, f"generate_strategies_{lemma.id}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    strategies = []
    for idx, strat in enumerate(data.get("strategies", [])):
        strategies.append(
            DecompositionStrategy(
                strategy_id=f"{lemma.id}_strat{idx+1}",
                rationale=str(strat.get("rationale", "")),
                sub_lemmas=[],  # Will be filled when strategy is attempted
                context_used=strat.get("context_used", []),
                status="untried",
            )
        )

    # Ensure we have at least one strategy (fallback)
    if not strategies:
        strategies.append(
            DecompositionStrategy(
                strategy_id=f"{lemma.id}_strat1",
                rationale="Standard decomposition: break into simpler sub-goals",
                sub_lemmas=[],
                context_used=[],
                status="untried",
            )
        )

    return strategies, resp.usage


def decompose_with_strategy(
    lemma: Lemma,
    strategy,  # DecompositionStrategy
    problem: Problem,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[Outline, dict[str, int]]:
    """Generate sub-lemmas following a specific decomposition strategy.

    Similar to decompose_blocked_lemma but guided by strategy rationale.

    Returns:
        (Outline, usage)
    """
    template = load_prompt_template("decompose_with_strategy", prompts_dir=config.PROMPTS_DIR)

    prompt = template.format(
        lemma_id=lemma.id,
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        strategy_rationale=strategy.rationale,
        problem_background=problem.background,
        problem_domain=problem.domain,
        current_depth=lemma.depth,
        max_depth=config.MAX_DEPTH,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        system_prompt=GPT_PLANNER_CORE,
        json_mode=True,
        reasoning_effort=config.REASONING_EFFORT_PLAN,
        text_verbosity=config.TEXT_VERBOSITY_PLAN,
        max_tokens=config.MAX_PLAN_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )

    _log_io(run_dir, f"decompose_strategy_{lemma.id}_{strategy.strategy_id}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    # Parse sub-lemmas
    sub_lemmas: list[Lemma] = []
    for lem in data.get("lemmas", []):
        sub_lemmas.append(
            Lemma(
                id=f"{lemma.id}.{lem['id']}",
                statement=str(lem.get("statement")),
                assumptions=list(lem.get("assumptions", []) or []),
                dependencies=[
                    f"{lemma.id}.{dep}" if dep in [l['id'] for l in data.get("lemmas", [])] else dep
                    for dep in lem.get("dependencies", [])
                ],
                status=LemmaStatus.OPEN,
                depth=lemma.depth + 1,
                parent_lemma_id=lemma.id,
            )
        )

    if sub_lemmas:
        validate_outline_constraints(sub_lemmas)

    outline = Outline(
        problem_id=problem.id,
        lemmas=sub_lemmas,
        proof_strategy=str(data.get("proof_strategy", "")),
    )

    return outline, resp.usage


def prove_lemma(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
    retry_feedback: str = "",
    literature_context: str = "",
) -> tuple[str, dict[str, int]]:
    """Attempt to prove a lemma with GPT_PROVER_CORE and domain-specific guidance."""

    template = load_prompt_template("prove_lemma", prompts_dir=config.PROMPTS_DIR)

    domain_pitfalls = format_domain_pitfalls(problem.domain)

    prompt = template.format(
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        dependency_interfaces=format_dependency_interfaces(dependency_interfaces),
        problem_background=problem.background,
        literature_context=literature_context or "(none)",
        domain_pitfalls=domain_pitfalls,
        retry_feedback=retry_feedback or "(none)",
        max_tokens=config.MAX_PROOF_TOKENS,
    )

    resp = call_gpt(
        prompt,
        json_mode=False,
        max_tokens=config.MAX_PROOF_TOKENS,
        system_prompt=GPT_PROVER_CORE,
        reasoning_effort=config.REASONING_EFFORT_PROVE,
        text_verbosity=config.TEXT_VERBOSITY_PROVE,
        use_background=config.USE_BACKGROUND_MODE,  # Proving can be slow
    )
    _log_io(run_dir, f"prove_{lemma.id}_try{lemma.retry_count}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    return resp.text.strip(), resp.usage


def prove_lemma_multiturn(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
    max_turns: int = 5,
    min_turns: int = 3,
    quality_threshold: float = 0.85,
) -> tuple[str, dict[str, int]]:
    """
    Multi-turn proving with iterative audit feedback.

    Flow:
    1. Generate proof (turn 1: no feedback, later: with feedback)
    2. Parallel dual audit (self + cross)
    3. Compute quality score
    4. Termination checks: early success, convergence, degradation
    5. Refine with structured feedback
    6. Return best proof
    """
    best_proof = None
    best_score = 0.0
    usage_total = {}
    audit_results = []

    for turn in range(1, max_turns + 1):
        # Generate proof
        if turn == 1:
            proof, u1 = prove_lemma(lemma, problem, dependency_interfaces,
                                    run_dir=run_dir, retry_feedback="")
        else:
            feedback = _format_multiturn_feedback(audit_results, turn)
            proof, u1 = prove_lemma(lemma, problem, dependency_interfaces,
                                    run_dir=run_dir, retry_feedback=feedback)

        merge_usage(usage_total, u1)
        lemma.proof = proof

        # Parallel dual audit
        self_result, cross_result, u2 = _dual_audit_parallel(lemma, problem.domain, run_dir)
        merge_usage(usage_total, u2)

        # Record
        audit_results.append({
            'turn': turn,
            'self': self_result,
            'cross': cross_result,
            'proof': proof,
        })

        # Score
        score = _compute_quality_score(self_result, cross_result)
        if score > best_score:
            best_score = score
            best_proof = proof

        # Termination checks
        if turn >= min_turns:
            if _both_pass_high_confidence(self_result, cross_result, quality_threshold):
                return proof, usage_total
            if _proof_converged(audit_results, turn):
                return best_proof, usage_total
            if _proof_degrading(audit_results, turn):
                return best_proof, usage_total

    return best_proof or proof, usage_total


def _dual_audit_parallel(lemma: Lemma, domain: str, run_dir: Optional[Path]) -> tuple[AuditResult, AuditResult, dict[str, int]]:
    """Run self_audit and cross_audit in parallel."""
    usage_total = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        self_future = executor.submit(self_audit, lemma, domain, run_dir=run_dir)
        cross_future = executor.submit(cross_audit, lemma, domain, run_dir=run_dir)
        self_result, u1 = self_future.result()
        cross_result, u2 = cross_future.result()
    merge_usage(usage_total, u1)
    merge_usage(usage_total, u2)
    return self_result, cross_result, usage_total


def _compute_quality_score(self_result: AuditResult, cross_result: AuditResult) -> float:
    """Compute quality score from audit results."""
    self_score = 1.0 if self_result.verdict == AuditVerdict.PASS else 0.0
    cross_score = 1.0 if cross_result.verdict == AuditVerdict.PASS else 0.0
    avg_conf = (self_result.confidence + cross_result.confidence) / 2
    return (self_score + cross_score) / 2 * avg_conf


def _both_pass_high_confidence(self_result: AuditResult, cross_result: AuditResult, threshold: float) -> bool:
    """Check if both audits pass with high confidence."""
    return (
        self_result.verdict == AuditVerdict.PASS and
        cross_result.verdict == AuditVerdict.PASS and
        self_result.confidence >= threshold and
        cross_result.confidence >= threshold
    )


def _proof_converged(audit_results: list[dict], current_turn: int, window: int = 2) -> bool:
    """Detect if proof quality has plateaued."""
    if current_turn < window + 1:
        return False
    recent_scores = [
        _compute_quality_score(ar['self'], ar['cross'])
        for ar in audit_results[-window:]
    ]
    return max(recent_scores) - min(recent_scores) < 0.05


def _proof_degrading(audit_results: list[dict], current_turn: int, window: int = 2) -> bool:
    """Detect if proof quality is decreasing."""
    if current_turn < window + 1:
        return False
    scores = [_compute_quality_score(ar['self'], ar['cross'])
              for ar in audit_results]
    recent_avg = sum(scores[-window:]) / window
    previous_avg = sum(scores[-2*window:-window]) / window
    return recent_avg < previous_avg - 0.1


def _format_multiturn_feedback(audit_results: list[dict], turn: int) -> str:
    """Format structured feedback for next turn."""
    current = audit_results[-1]
    prev = audit_results[-2] if len(audit_results) > 1 else None

    feedback = f"=== MULTI-TURN REFINEMENT (Turn {turn}/{config.MAX_MULTITURN_ROUNDS}) ===\n\n"
    feedback += f"CURRENT STATUS:\n"
    feedback += f"- Self-audit: {current['self'].verdict.value} (conf: {current['self'].confidence:.2f})\n"
    feedback += f"- Cross-audit: {current['cross'].verdict.value} (conf: {current['cross'].confidence:.2f})\n\n"

    # Critical issues
    issues = []
    for issue in current['self'].issues:
        if isinstance(issue, dict):
            issues.append(f"- {issue.get('summary', str(issue))}")
        else:
            issues.append(f"- {issue}")
    for issue in current['cross'].issues:
        if isinstance(issue, dict) and issue not in issues:
            issues.append(f"- {issue.get('summary', str(issue))}")
        elif isinstance(issue, str) and issue not in issues:
            issues.append(f"- {issue}")

    if issues:
        feedback += "CRITICAL ISSUES TO ADDRESS:\n"
        feedback += "\n".join(issues[:5])  # Top 5 issues
        feedback += "\n\n"

    # Evolution tracking
    if prev:
        feedback += f"FEEDBACK EVOLUTION (Turn {turn-1} → {turn}):\n"
        feedback += "Your task: Address the issues above while maintaining correct reasoning.\n"

    return feedback


def prove_lemma_recursive(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    all_lemmas: dict[str, Lemma],  # Global lemma tracker
    *,
    run_dir: Optional[Path] = None,
    max_retries: int = None,
) -> tuple[Lemma, dict[str, int]]:
    """
    Prove lemma with recursive decomposition if BLOCKED.

    If a lemma cannot be proved after max retries and depth < MAX_DEPTH,
    decompose it into sub-lemmas, prove them recursively, then retry parent.
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    usage_total = {}

    # Attempt to prove directly with retry loop
    for retry in range(max_retries + 1):
        # Set retry_feedback from previous attempts
        retry_feedback = ""
        if retry > 0 and lemma.evidence:
            feedback_parts = []
            for ev in lemma.evidence[-2:]:  # Last 2 audit results
                if ev.get("feedback"):
                    feedback_parts.append(ev["feedback"])
            retry_feedback = "\n\n".join(feedback_parts)

        # Attempt to prove
        proof_text, u1 = prove_lemma(
            lemma,
            problem,
            dependency_interfaces,
            run_dir=run_dir,
            retry_feedback=retry_feedback,
        )
        merge_usage(usage_total, u1)

        # Check if BLOCKED
        is_blocked = "BLOCKED" in proof_text.upper()

        if not is_blocked:
            # Proof succeeded! Set status and continue to audit
            lemma.proof = proof_text
            lemma.status = LemmaStatus.CLAIMED
            break
        elif retry < max_retries:
            # BLOCKED but have retries left - continue retry loop
            continue
        else:
            # BLOCKED and max retries exhausted - try recursive decomposition
            if lemma.depth < config.MAX_DEPTH:
                # Decompose into sub-lemmas
                sub_outline, u2 = decompose_blocked_lemma(
                    lemma, problem, lemma.depth, run_dir=run_dir
                )
                merge_usage(usage_total, u2)

                if not sub_outline.lemmas:
                    # Decomposition failed - mark as BLOCKED
                    lemma.status = LemmaStatus.BLOCKED
                    return lemma, usage_total

                # Track sub-lemmas
                lemma.sub_lemmas = [sl.id for sl in sub_outline.lemmas]
                for sub_lem in sub_outline.lemmas:
                    all_lemmas[sub_lem.id] = sub_lem

                # Recursively prove sub-lemmas in dependency order
                sub_interfaces = dependency_interfaces.copy()
                topo_sorted = topo_sort_lemmas(sub_outline.lemmas)

                for sub_lem in topo_sorted:
                    # Check if dependencies are met
                    if any(all_lemmas.get(dep, Lemma(id=dep, statement="", status=LemmaStatus.BLOCKED)).status != LemmaStatus.PROVED
                           for dep in sub_lem.dependencies):
                        # Dependency blocked - skip this sub-lemma
                        sub_lem.status = LemmaStatus.BLOCKED
                        continue

                    # Recursively prove sub-lemma
                    proved_sub, u3 = prove_lemma_recursive(
                        sub_lem,
                        problem,
                        sub_interfaces,
                        all_lemmas,
                        run_dir=run_dir,
                        max_retries=max_retries,
                    )
                    merge_usage(usage_total, u3)

                    # Update tracking
                    all_lemmas[proved_sub.id] = proved_sub

                    if proved_sub.status == LemmaStatus.PROVED:
                        # Add to interfaces for subsequent sub-lemmas
                        sub_interfaces[proved_sub.id] = proved_sub.interface or proved_sub.statement
                    else:
                        # Sub-lemma failed - parent remains BLOCKED
                        lemma.status = LemmaStatus.BLOCKED
                        return lemma, usage_total

                # All sub-lemmas proved! Retry parent with sub-interfaces
                proof_text, u4 = prove_lemma(
                    lemma,
                    problem,
                    sub_interfaces,  # Include sub-lemma interfaces
                    run_dir=run_dir,
                    retry_feedback=f"Sub-lemmas proved:\n" + "\n".join([f"- {sid}: {sub_interfaces[sid][:200]}" for sid in lemma.sub_lemmas]),
                )
                merge_usage(usage_total, u4)

                if "BLOCKED" not in proof_text.upper():
                    # Success with sub-lemmas!
                    lemma.proof = proof_text
                    lemma.status = LemmaStatus.CLAIMED
                    break
                else:
                    # Still blocked even with sub-lemmas
                    lemma.status = LemmaStatus.BLOCKED
                    return lemma, usage_total
            else:
                # Max depth reached - cannot decompose further
                lemma.status = LemmaStatus.BLOCKED
                return lemma, usage_total

    # Proof succeeded (either directly or with sub-lemmas) - now audit
    # (Audit logic will be handled by controller as before)
    return lemma, usage_total


# ============================================================================
# New Recursive Grammar with Backtracking (Phase 5)
# ============================================================================


def solve_problem(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    all_lemmas: dict[str, Lemma],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[Lemma, dict[str, int]]:
    """Recursive grammar for solving a problem (lemma) at any depth.

    Grammar:
        solve_problem(P) ::=
            decide = decide_strategy(P, context)
            if decide == "direct":
                return solve_directly(P)
            else:
                strategies = generate_strategies(P, context)
                for strategy in strategies:
                    result = decompose_and_solve(P, strategy)
                    if result.status == PROVED:
                        return result
                    else:
                        record_failure(P, strategy, result)
                        # Try next strategy (BACKTRACKING)
                return mark_blocked(P)
    """
    from prover.utils import create_context_document, record_attempt

    usage_total = {}

    # Initialize context if needed
    if lemma.context is None:
        lemma.context = create_context_document(lemma)

    # Termination check: depth limit
    if lemma.depth >= config.MAX_DEPTH:
        lemma.status = LemmaStatus.BLOCKED
        record_attempt(lemma, "direct", {
            "result": "blocked",
            "reason": "max_depth_reached"
        })
        return lemma, usage_total

    # DECISION POINT: Should we try direct proof or decompose?
    decision, u1 = decide_proof_strategy(
        lemma, problem, dependency_interfaces, run_dir=run_dir
    )
    merge_usage(usage_total, u1)

    if decision == "direct":
        # Direct proof branch
        proved_lemma, u2 = solve_directly(
            lemma, problem, dependency_interfaces, run_dir=run_dir
        )
        merge_usage(usage_total, u2)
        return proved_lemma, usage_total

    else:  # decompose
        # Generate multiple decomposition strategies
        strategies, u3 = generate_decomposition_strategies(
            lemma, problem, run_dir=run_dir, num_strategies=3
        )
        merge_usage(usage_total, u3)

        lemma.decomposition_strategies = strategies

        # Try strategies: parallel (race semantics) or sequential (backtracking)
        if config.ENABLE_PARALLEL_STRATEGIES:
            # Parallel strategy execution - first success wins
            with ThreadPoolExecutor(max_workers=min(len(strategies), 3)) as executor:
                # Submit all strategies concurrently
                futures = {
                    executor.submit(
                        attempt_decomposition_strategy,
                        lemma, strategy, problem, dependency_interfaces, all_lemmas, run_dir=run_dir
                    ): (strategy_idx, strategy)
                    for strategy_idx, strategy in enumerate(strategies)
                }

                # Wait for first success (race semantics)
                for future in as_completed(futures):
                    strategy_idx, strategy = futures[future]
                    lemma.current_strategy_index = strategy_idx

                    try:
                        result, u4 = future.result()
                        merge_usage(usage_total, u4)

                        if result.status == LemmaStatus.PROVED:
                            # First success wins - cancel remaining futures
                            strategy.status = "succeeded"
                            if lemma.context:
                                lemma.context.successful_approaches.append(strategy.rationale)

                            # Cancel other strategies
                            for f in futures:
                                if f != future and not f.done():
                                    f.cancel()

                            return result, usage_total
                        else:
                            strategy.status = "failed"
                            if lemma.context:
                                lemma.context.failed_strategies.append(
                                    f"{strategy.strategy_id}: {strategy.rationale}"
                                )
                    except Exception as e:
                        strategy.status = "failed"
                        if lemma.context:
                            lemma.context.failed_strategies.append(
                                f"{strategy.strategy_id}: Exception - {str(e)}"
                            )

            # All strategies failed
            lemma.status = LemmaStatus.BLOCKED
            return lemma, usage_total
        else:
            # Sequential fallback (BACKTRACKING LOOP)
            for strategy_idx, strategy in enumerate(strategies):
                lemma.current_strategy_index = strategy_idx
                strategy.status = "in_progress"

                # Attempt this decomposition strategy
                result, u4 = attempt_decomposition_strategy(
                    lemma,
                    strategy,
                    problem,
                    dependency_interfaces,
                    all_lemmas,
                    run_dir=run_dir,
                )
                merge_usage(usage_total, u4)

                if result.status == LemmaStatus.PROVED:
                    # Strategy succeeded!
                    strategy.status = "succeeded"
                    if lemma.context:
                        lemma.context.successful_approaches.append(strategy.rationale)
                    return result, usage_total
                else:
                    # Strategy failed - record and try next
                    strategy.status = "failed"
                    if lemma.context:
                        lemma.context.failed_strategies.append(
                            f"{strategy.strategy_id}: {strategy.rationale}"
                        )
                    # BACKTRACK to next strategy
                    continue

            # All strategies exhausted - mark as BLOCKED
            lemma.status = LemmaStatus.BLOCKED
            return lemma, usage_total


def solve_directly(
    lemma: Lemma,
    problem: Problem,
    dependency_interfaces: dict[str, str],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[Lemma, dict[str, int]]:
    """Attempt direct proof with retry loop.

    Similar to the retry loop in prove_lemma_recursive but records attempts in context.
    """
    from prover.utils import record_attempt

    usage_total = {}
    max_retries = config.MAX_RETRIES

    for retry in range(max_retries + 1):
        lemma.retry_count = retry

        # Get retry feedback from context
        retry_feedback = ""
        if lemma.context and lemma.context.audit_failures:
            feedback_parts = [
                af.get("feedback", "")
                for af in lemma.context.audit_failures[-2:]
            ]
            retry_feedback = "\n\n".join(feedback_parts)

        # Attempt proof (multi-turn or single-shot)
        if config.ENABLE_MULTITURN_PROVING:
            proof_text, u1 = prove_lemma_multiturn(
                lemma,
                problem,
                dependency_interfaces,
                run_dir=run_dir,
                max_turns=config.MAX_MULTITURN_ROUNDS,
                min_turns=config.MIN_MULTITURN_ROUNDS,
                quality_threshold=config.MULTITURN_QUALITY_THRESHOLD,
            )
        else:
            proof_text, u1 = prove_lemma(
                lemma,
                problem,
                dependency_interfaces,
                run_dir=run_dir,
                retry_feedback=retry_feedback,
            )
        merge_usage(usage_total, u1)

        # Check if blocked
        is_blocked = "BLOCKED" in proof_text.upper()

        if not is_blocked:
            # Proof succeeded
            lemma.proof = proof_text
            lemma.status = LemmaStatus.CLAIMED

            # Record successful attempt
            record_attempt(lemma, "direct", {
                "result": "claimed",
                "proof_length": len(proof_text),
                "retry": retry,
            })
            return lemma, usage_total

        elif retry < max_retries:
            # BLOCKED but have retries left
            record_attempt(lemma, "direct", {
                "result": "blocked",
                "reason": "proof_returned_blocked",
                "retry": retry,
            })
            continue
        else:
            # Max retries exhausted
            lemma.status = LemmaStatus.BLOCKED
            record_attempt(lemma, "direct", {
                "result": "blocked",
                "reason": "max_retries_exhausted",
                "retry": retry,
            })
            return lemma, usage_total

    # Shouldn't reach here
    lemma.status = LemmaStatus.BLOCKED
    return lemma, usage_total


def attempt_decomposition_strategy(
    lemma: Lemma,
    strategy,  # DecompositionStrategy
    problem: Problem,
    dependency_interfaces: dict[str, str],
    all_lemmas: dict[str, Lemma],
    *,
    run_dir: Optional[Path] = None,
) -> tuple[Lemma, dict[str, int]]:
    """Execute a specific decomposition strategy.

    1. Generate sub-lemmas based on strategy
    2. Recursively solve each sub-lemma
    3. If all succeed, attempt integration
    4. If integration fails, backtrack
    """
    from prover.utils import propagate_context_from_parent, record_attempt

    usage_total = {}

    # Generate sub-lemmas for this strategy
    sub_outline, u1 = decompose_with_strategy(
        lemma, strategy, problem, run_dir=run_dir
    )
    merge_usage(usage_total, u1)

    if not sub_outline.lemmas:
        # Decomposition failed
        lemma.status = LemmaStatus.BLOCKED
        record_attempt(lemma, "decompose", {
            "result": "blocked",
            "strategy": strategy.strategy_id,
            "reason": "decomposition_produced_no_sublemmas",
        })
        return lemma, usage_total

    # Track sub-lemmas
    lemma.sub_lemmas = [sl.id for sl in sub_outline.lemmas]
    strategy.sub_lemmas = lemma.sub_lemmas

    for sub_lem in sub_outline.lemmas:
        all_lemmas[sub_lem.id] = sub_lem
        # Propagate context from parent
        if lemma.context:
            propagate_context_from_parent(sub_lem, lemma.context)

    # Recursively solve sub-lemmas in dependency order
    sub_interfaces = dependency_interfaces.copy()
    topo_sorted = topo_sort_lemmas(sub_outline.lemmas)

    for sub_lem in topo_sorted:
        # Check dependencies
        if any(
            all_lemmas.get(dep, Lemma(id=dep, statement="", status=LemmaStatus.BLOCKED)).status != LemmaStatus.PROVED
            for dep in sub_lem.dependencies
        ):
            sub_lem.status = LemmaStatus.BLOCKED
            continue

        # RECURSIVE CALL - This is where the grammar recurses
        proved_sub, u2 = solve_problem(
            sub_lem,
            problem,
            sub_interfaces,
            all_lemmas,
            run_dir=run_dir,
        )
        merge_usage(usage_total, u2)

        # Update tracking
        all_lemmas[proved_sub.id] = proved_sub

        if proved_sub.status == LemmaStatus.PROVED:
            # Add interface for subsequent sub-lemmas
            sub_interfaces[proved_sub.id] = proved_sub.interface or proved_sub.statement
        else:
            # Sub-lemma failed - whole strategy fails
            lemma.status = LemmaStatus.BLOCKED
            record_attempt(lemma, "decompose", {
                "result": "blocked",
                "failed_sublemma": proved_sub.id,
                "strategy": strategy.strategy_id,
            })
            return lemma, usage_total

    # All sub-lemmas proved! Try to integrate with retry mechanism
    for integration_attempt in range(config.MAX_INTEGRATION_RETRIES):
        # Format retry feedback with increasing detail
        if integration_attempt == 0:
            retry_feedback = f"Sub-lemmas proved using strategy '{strategy.rationale}':\n" + \
                            "\n".join([f"- {sid}: {sub_interfaces.get(sid, '?')[:200]}"
                                      for sid in lemma.sub_lemmas])
        else:
            # Enhanced feedback from previous failures
            prev_failures = [
                f for f in (lemma.context.integration_failures if lemma.context else [])
                if f.get("strategy") == strategy.strategy_id
            ]
            failure_summary = "\n".join([
                f"Attempt {i+1}: {f.get('feedback', 'Unknown failure')}"
                for i, f in enumerate(prev_failures[-2:])  # Last 2 failures
            ])
            retry_feedback = f"Sub-lemmas proved using strategy '{strategy.rationale}':\n" + \
                            "\n".join([f"- {sid}: {sub_interfaces.get(sid, '?')[:200]}"
                                      for sid in lemma.sub_lemmas]) + \
                            f"\n\n### PREVIOUS INTEGRATION ATTEMPTS FAILED:\n{failure_summary}\n" + \
                            f"### YOUR TASK: Reformulate integration to avoid these issues."

        proof_text, u3 = prove_lemma(
            lemma, problem, sub_interfaces, run_dir=run_dir,
            retry_feedback=retry_feedback,
        )
        merge_usage(usage_total, u3)

        if "BLOCKED" not in proof_text.upper():
            # Integration successful!
            lemma.proof = proof_text
            lemma.status = LemmaStatus.CLAIMED
            record_attempt(lemma, "decompose", {
                "result": "claimed",
                "strategy": strategy.strategy_id,
                "num_sublemmas": len(lemma.sub_lemmas),
                "integration_attempt": integration_attempt + 1,
            })
            return lemma, usage_total
        else:
            # Integration failed - record and retry
            if lemma.context:
                lemma.context.integration_failures.append({
                    "strategy": strategy.strategy_id,
                    "attempt": integration_attempt + 1,
                    "feedback": "Could not integrate sub-lemma proofs",
                    "num_sublemmas": len(lemma.sub_lemmas),
                })

            if integration_attempt < config.MAX_INTEGRATION_RETRIES - 1:
                # Have retries left - continue loop
                continue
            else:
                # Max retries exhausted - fail strategy
                lemma.status = LemmaStatus.BLOCKED
                record_attempt(lemma, "decompose", {
                    "result": "blocked",
                    "strategy": strategy.strategy_id,
                    "reason": "integration_failed_after_retries",
                    "attempts": integration_attempt + 1,
                })
                return lemma, usage_total
        record_attempt(lemma, "decompose", {
            "result": "blocked",
            "strategy": strategy.strategy_id,
            "reason": "integration_failed",
        })
        return lemma, usage_total


def self_audit(lemma: Lemma, problem_domain: str, *, run_dir: Optional[Path] = None, literature_context: str = "") -> tuple[AuditResult, dict[str, int]]:
    """Self-audit with GPT_SELF_AUDIT_CORE and domain-specific tests."""

    template = load_prompt_template("self_audit", prompts_dir=config.PROMPTS_DIR)

    domain_audit_tests = format_domain_audit_tests(problem_domain)

    prompt = template.format(
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        lemma_proof=lemma.proof or "(missing)",
        domain_audit_tests=domain_audit_tests,
        literature_context=literature_context or "(none)",
    )

    resp = call_gpt(
        prompt,
        json_mode=True,
        max_tokens=config.MAX_AUDIT_TOKENS,
        system_prompt=GPT_SELF_AUDIT_CORE,
        reasoning_effort=config.REASONING_EFFORT_AUDIT,
        text_verbosity=config.TEXT_VERBOSITY_AUDIT,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, f"self_audit_{lemma.id}_try{lemma.retry_count}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    # Enhanced audit result parsing (supports structured issues)
    issues_raw = data.get("issues", []) or []
    issues_formatted = []
    for issue in issues_raw:
        if isinstance(issue, dict):
            # Structured issue
            issues_formatted.append(
                f"[{issue.get('severity', 'major')}] {issue.get('location', '?')}: {issue.get('summary', '')} | {issue.get('detail', '')}"
            )
        else:
            # Legacy string issue
            issues_formatted.append(str(issue))

    result = AuditResult(
        lemma_id=lemma.id,
        auditor="gpt-self",
        verdict=_coerce_verdict(data.get("verdict", "UNCERTAIN")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        issues=issues_formatted,
        feedback=str(data.get("feedback", "") or ""),
    )

    return result, resp.usage


def cross_audit(lemma: Lemma, problem_domain: str, *, run_dir: Optional[Path] = None, literature_context: str = "") -> tuple[AuditResult, dict[str, int]]:
    """Cross-audit with CLAUDE_CROSS_AUDIT_CORE and domain-specific tests."""

    template = load_prompt_template("cross_audit", prompts_dir=config.PROMPTS_DIR)

    domain_audit_tests = format_domain_audit_tests(problem_domain)

    prompt = template.format(
        lemma_statement=lemma.statement,
        lemma_assumptions="\n".join(lemma.assumptions) if lemma.assumptions else "(none)",
        lemma_proof=lemma.proof or "(missing)",
        domain_audit_tests=domain_audit_tests,
        literature_context=literature_context or "(none)",
    )

    resp = call_claude(
        prompt,
        json_mode=True,
        max_tokens=config.MAX_AUDIT_TOKENS,
        system_prompt=CLAUDE_CROSS_AUDIT_CORE,
        lemma_id_for_mock=lemma.id,
    )
    _log_io(run_dir, f"cross_audit_{lemma.id}_try{lemma.retry_count}", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    # Enhanced audit result parsing
    issues_raw = data.get("issues", []) or []
    issues_formatted = []
    for issue in issues_raw:
        if isinstance(issue, dict):
            issues_formatted.append(
                f"[{issue.get('severity', 'major')}] {issue.get('location', '?')}: {issue.get('summary', '')} | {issue.get('detail', '')}"
            )
        else:
            issues_formatted.append(str(issue))

    result = AuditResult(
        lemma_id=lemma.id,
        auditor="claude-cross",
        verdict=_coerce_verdict(data.get("verdict", "UNCERTAIN")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        issues=issues_formatted,
        feedback=str(data.get("feedback", "") or ""),
    )

    return result, resp.usage


def validate_output(lemma: Lemma) -> str:
    """Simple validation gate (unchanged from original)."""

    def last_for(auditor: str) -> Optional[dict]:
        for e in reversed(lemma.evidence):
            if e.get("auditor") == auditor:
                return e
        return None

    self_e = last_for("gpt-self")
    cross_e = last_for("claude-cross")

    if not self_e or not cross_e:
        return "RETRY" if lemma.retry_count < config.MAX_RETRIES else "REJECT"

    def _v(x: object) -> str:
        return getattr(x, "value", x) if x is not None else ""

    self_v = str(_v(self_e.get("verdict"))).upper()
    cross_v = str(_v(cross_e.get("verdict"))).upper()
    self_c = float(self_e.get("confidence", 0.0) or 0.0)
    cross_c = float(cross_e.get("confidence", 0.0) or 0.0)

    if self_v == "PASS" and cross_v == "PASS":
        if self_c >= config.MIN_CONFIDENCE and cross_c >= config.MIN_CONFIDENCE:
            return "ACCEPT"
        return "RETRY" if lemma.retry_count < config.MAX_RETRIES else "REJECT"

    if self_v == "FAIL" or cross_v == "FAIL":
        return "RETRY" if lemma.retry_count < config.MAX_RETRIES else "REJECT"

    # Mixed/uncertain
    return "RETRY" if lemma.retry_count < config.MAX_RETRIES else "REJECT"


def _combined_feedback(lemma: Lemma) -> str:
    """Combine last audit feedback strings for retry."""
    pieces = []
    for auditor in ("gpt-self", "claude-cross"):
        last = None
        for e in reversed(lemma.evidence):
            if e.get("auditor") == auditor:
                last = e
                break
        if last:
            pieces.append(
                f"[{auditor}] verdict={last.get('verdict')} conf={last.get('confidence')}\n"
                f"Issues: {last.get('issues')}\nFeedback: {last.get('feedback')}\n"
            )

    return "\n".join(pieces).strip() or "(none)"


def extract_interface(lemma: Lemma, *, run_dir: Optional[Path] = None) -> tuple[str, dict[str, int]]:
    """Extract interface (uses updated template but no thinking core needed)."""

    template = load_prompt_template("extract_interface", prompts_dir=config.PROMPTS_DIR)
    prompt = template.format(
        lemma_id=lemma.id,
        lemma_statement=lemma.statement,
        lemma_proof=lemma.proof or "(missing)",
    )

    resp = call_gpt(
        prompt,
        json_mode=False,
        max_tokens=config.MAX_INTERFACE_TOKENS,
        system_prompt="",  # No thinking core needed for summarization
        reasoning_effort=config.REASONING_EFFORT_INTERFACE,  # MAXIMIZED to xhigh
        text_verbosity=config.TEXT_VERBOSITY_INTERFACE,  # MAXIMIZED to high
        use_background=False,
    )
    _log_io(run_dir, f"extract_interface_{lemma.id}", prompt=prompt, response=resp.text, thinking=resp.thinking)
    return resp.text.strip(), resp.usage


def integrate_proof(lemmas: list[Lemma], problem: Problem, *, run_dir: Optional[Path] = None) -> tuple[str, dict[str, int]]:
    """Assemble a full proof from proved lemmas."""

    ordered = topo_sort_lemmas(lemmas) if lemmas else []

    template = load_prompt_template("integrate_proof", prompts_dir=config.PROMPTS_DIR)
    prompt = template.format(
        problem_statement=problem.statement,
        proved_lemma_interfaces=format_all_interfaces(ordered),
        proved_lemma_proofs="\n\n".join(
            [f"## {l.id}\n\n**Statement.** {l.statement}\n\n{l.proof or ''}" for l in ordered]
        ),
        max_total_tokens=config.MAX_INTEGRATION_TOKENS,
    )

    resp = call_gpt(
        prompt,
        json_mode=False,
        max_tokens=config.MAX_INTEGRATION_TOKENS,
        system_prompt="",  # Light thinking core (just assembly, not deep reasoning)
        reasoning_effort=config.REASONING_EFFORT_INTEGRATE,
        text_verbosity=config.TEXT_VERBOSITY_INTEGRATE,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, "integrate_proof", prompt=prompt, response=resp.text, thinking=resp.thinking)

    return resp.text.strip(), resp.usage


def final_audit(proof_text: str, problem: Problem, *, run_dir: Optional[Path] = None, literature_context: str = "") -> tuple[AuditResult, dict[str, int]]:
    """Final audit with CLAUDE_FINAL_AUDIT_CORE."""

    template = load_prompt_template("final_audit", prompts_dir=config.PROMPTS_DIR)
    prompt = template.format(
        problem_statement=problem.statement,
        full_proof=proof_text,
        literature_context=literature_context or "(none)",
    )

    resp = call_claude(
        prompt,
        json_mode=True,
        max_tokens=config.MAX_FINAL_AUDIT_TOKENS,
        system_prompt=CLAUDE_FINAL_AUDIT_CORE,
    )
    _log_io(run_dir, "final_audit", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    # Enhanced parsing for final audit
    issues_raw = data.get("issues", []) or []
    issues_formatted = []
    for issue in issues_raw:
        if isinstance(issue, dict):
            issues_formatted.append(
                f"[{issue.get('severity', 'major')}] {issue.get('location', '?')}: {issue.get('summary', '')}"
            )
        else:
            issues_formatted.append(str(issue))

    result = AuditResult(
        lemma_id="(full-proof)",
        auditor="claude-final",
        verdict=_coerce_verdict(data.get("verdict", "UNCERTAIN")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        issues=issues_formatted,
        feedback=str(data.get("feedback", "") or ""),
        recommendation=str(data.get("recommendation", "") or "") or None,
    )

    return result, resp.usage


# Exported helper
get_retry_feedback = _combined_feedback


# ============================================================================
# Phase 0.5: Literature Processing
# ============================================================================


def process_literature(
    problem: Problem,
    raw_literature: str,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Process raw literature into structured, categorized JSON.

    Returns:
        (processed_literature_json, usage_dict)
    """
    from prover.thinking_cores import LITERATURE_PROCESSOR_CORE
    from prover.models import ProcessedLiterature

    template = load_prompt_template("literature_processing", prompts_dir=config.PROMPTS_DIR)

    prompt = template.format(
        problem_statement=problem.statement,
        problem_background=problem.background,
        problem_domain=problem.domain,
        raw_literature=raw_literature,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        system_prompt=LITERATURE_PROCESSOR_CORE,
        reasoning_effort=config.REASONING_EFFORT_LITERATURE_PROCESSING,
        max_tokens=config.MAX_LITERATURE_PROCESSING_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, "literature_processing", prompt=prompt, response=resp.text, thinking=resp.thinking)

    # Validate the JSON parses correctly
    try:
        proc = ProcessedLiterature.model_validate_json(resp.text)
        result_json = proc.model_dump_json(indent=2)
    except Exception:
        # Try extracting JSON from the response
        data = parse_json_robust(resp.text)
        proc = ProcessedLiterature.model_validate(data)
        result_json = proc.model_dump_json(indent=2)

    return result_json, resp.usage


def audit_literature(
    problem: Problem,
    raw_literature: str,
    processed_literature_json: str,
    *,
    run_dir: Optional[Path] = None,
) -> tuple["LiteratureReviewResult", dict[str, int]]:
    """Audit processed literature for completeness and accuracy.

    Returns:
        (LiteratureReviewResult, usage_dict)
    """
    from prover.thinking_cores import LITERATURE_AUDITOR_CORE
    from prover.models import LiteratureReviewResult

    template = load_prompt_template("literature_audit", prompts_dir=config.PROMPTS_DIR)

    prompt = template.format(
        problem_statement=problem.statement,
        problem_domain=problem.domain,
        raw_literature=raw_literature,
        processed_literature=processed_literature_json,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        system_prompt=LITERATURE_AUDITOR_CORE,
        reasoning_effort=config.REASONING_EFFORT_LITERATURE_PROCESSING,
        max_tokens=config.MAX_LITERATURE_PROCESSING_TOKENS,
        use_background=False,
    )
    _log_io(run_dir, "literature_audit", prompt=prompt, response=resp.text, thinking=resp.thinking)

    data = parse_json_robust(resp.text)

    review = LiteratureReviewResult(
        overall_verdict=str(data.get("overall_verdict", "pass")).lower(),
        critical_issues=data.get("critical_issues", []),
        major_issues=data.get("major_issues", []),
        minor_issues=data.get("minor_issues", []),
        positive_aspects=data.get("positive_aspects", []),
        summary=str(data.get("summary", "")),
    )

    return review, resp.usage


def revise_literature(
    problem: Problem,
    raw_literature: str,
    processed_literature_json: str,
    review_feedback: str,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    """Revise processed literature based on audit feedback.

    Returns:
        (revised_processed_literature_json, usage_dict)
    """
    from prover.thinking_cores import LITERATURE_PROCESSOR_CORE
    from prover.models import ProcessedLiterature

    template = load_prompt_template("literature_revision", prompts_dir=config.PROMPTS_DIR)

    prompt = template.format(
        problem_statement=problem.statement,
        problem_background=problem.background,
        problem_domain=problem.domain,
        raw_literature=raw_literature,
        processed_literature=processed_literature_json,
        review_feedback=review_feedback,
    )

    resp: AgentResponse = call_gpt(
        prompt,
        json_mode=True,
        system_prompt=LITERATURE_PROCESSOR_CORE,
        reasoning_effort=config.REASONING_EFFORT_LITERATURE_PROCESSING,
        max_tokens=config.MAX_LITERATURE_PROCESSING_TOKENS,
        use_background=config.USE_BACKGROUND_MODE,
    )
    _log_io(run_dir, "literature_revision", prompt=prompt, response=resp.text, thinking=resp.thinking)

    # Validate
    try:
        proc = ProcessedLiterature.model_validate_json(resp.text)
        result_json = proc.model_dump_json(indent=2)
    except Exception:
        data = parse_json_robust(resp.text)
        proc = ProcessedLiterature.model_validate(data)
        result_json = proc.model_dump_json(indent=2)

    return result_json, resp.usage
