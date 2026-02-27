"""Prover-specific configuration.

Inherits shared settings and adds token budgets, reasoning effort,
control parameters, and paths specific to the proof workflow.
"""

from __future__ import annotations

import os
from pathlib import Path

from shared.config import *  # noqa: F401, F403 — re-export shared settings

# ============================================================================
# Token budgets (maximized for difficult problems)
# ============================================================================

MAX_PLAN_TOKENS = int(os.getenv("MAX_PLAN_TOKENS", "128000"))
MAX_PROOF_TOKENS = int(os.getenv("MAX_PROOF_TOKENS", "128000"))
MAX_AUDIT_TOKENS = int(os.getenv("MAX_AUDIT_TOKENS", "128000"))
MAX_INTERFACE_TOKENS = int(os.getenv("MAX_INTERFACE_TOKENS", "128000"))
MAX_INTEGRATION_TOKENS = int(os.getenv("MAX_INTEGRATION_TOKENS", "128000"))
MAX_FINAL_AUDIT_TOKENS = int(os.getenv("MAX_FINAL_AUDIT_TOKENS", "128000"))
MAX_LITERATURE_SEARCH_TOKENS = int(os.getenv("MAX_LITERATURE_SEARCH_TOKENS", "128000"))
MAX_LITERATURE_PROCESSING_TOKENS = int(os.getenv("MAX_LITERATURE_PROCESSING_TOKENS", "64000"))
MAX_DECISION_TOKENS = int(os.getenv("MAX_DECISION_TOKENS", "127998"))

# ============================================================================
# Reasoning effort (all xhigh for maximum rigor)
# ============================================================================

REASONING_EFFORT_PLAN = os.getenv("REASONING_EFFORT_PLAN", "xhigh")
REASONING_EFFORT_PROVE = os.getenv("REASONING_EFFORT_PROVE", "xhigh")
REASONING_EFFORT_AUDIT = os.getenv("REASONING_EFFORT_AUDIT", "xhigh")
REASONING_EFFORT_INTEGRATE = os.getenv("REASONING_EFFORT_INTEGRATE", "xhigh")
REASONING_EFFORT_LITERATURE = os.getenv("REASONING_EFFORT_LITERATURE", "xhigh")
REASONING_EFFORT_LITERATURE_PROCESSING = os.getenv("REASONING_EFFORT_LITERATURE_PROCESSING", "xhigh")
REASONING_EFFORT_INTERFACE = os.getenv("REASONING_EFFORT_INTERFACE", "xhigh")

# ============================================================================
# Text verbosity
# ============================================================================

TEXT_VERBOSITY_PLAN = os.getenv("TEXT_VERBOSITY_PLAN", "high")
TEXT_VERBOSITY_PROVE = os.getenv("TEXT_VERBOSITY_PROVE", "high")
TEXT_VERBOSITY_AUDIT = os.getenv("TEXT_VERBOSITY_AUDIT", "high")
TEXT_VERBOSITY_INTEGRATE = os.getenv("TEXT_VERBOSITY_INTEGRATE", "high")
TEXT_VERBOSITY_INTERFACE = os.getenv("TEXT_VERBOSITY_INTERFACE", "high")

# ============================================================================
# Control parameters
# ============================================================================

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MAX_INTEGRATION_RETRIES = int(os.getenv("MAX_INTEGRATION_RETRIES", "3"))
MAX_LEMMAS = int(os.getenv("MAX_LEMMAS", "300"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "25"))
MAX_FANOUT = int(os.getenv("MAX_FANOUT", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.75"))
REQUIRE_BOTH_PASS = os.getenv("REQUIRE_BOTH_PASS", "true").lower() in {"1", "true", "yes"}

# Planning quality control
MAX_PLAN_ROUNDS = int(os.getenv("MAX_PLAN_ROUNDS", "10"))
MIN_PLAN_CONFIDENCE = float(os.getenv("MIN_PLAN_CONFIDENCE", "0.6"))
ABORT_PLAN_CONFIDENCE = float(os.getenv("ABORT_PLAN_CONFIDENCE", "0.3"))

# Multi-turn proving
ENABLE_MULTITURN_PROVING = os.getenv("ENABLE_MULTITURN_PROVING", "true").lower() in {"1", "true", "yes"}
MAX_MULTITURN_ROUNDS = int(os.getenv("MAX_MULTITURN_ROUNDS", "2"))
MIN_MULTITURN_ROUNDS = int(os.getenv("MIN_MULTITURN_ROUNDS", "3"))
MULTITURN_QUALITY_THRESHOLD = float(os.getenv("MULTITURN_QUALITY_THRESHOLD", "0.85"))

# Parallelization
ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "true").lower() in {"1", "true", "yes"}
MAX_PARALLEL_LEMMAS = int(os.getenv("MAX_PARALLEL_LEMMAS", "10"))
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "16"))
ENABLE_PARALLEL_STRATEGIES = os.getenv("ENABLE_PARALLEL_STRATEGIES", "true").lower() in {"1", "true", "yes"}

# Literature
ENABLE_LITERATURE_RESEARCH = os.getenv("ENABLE_LITERATURE_RESEARCH", "true").lower() in {"1", "true", "yes"}
ENABLE_LITERATURE_PROCESSING = os.getenv("ENABLE_LITERATURE_PROCESSING", "true").lower() in {"1", "true", "yes"}
MAX_LITERATURE_REVIEWS = int(os.getenv("MAX_LITERATURE_REVIEWS", "2"))

# Runtime
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "500"))

# ============================================================================
# Paths
# ============================================================================

_PROVER_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = os.getenv("PROMPTS_DIR", str(_PROVER_ROOT / "prompts"))
RUNS_DIR = os.getenv("RUNS_DIR", str(_PROVER_ROOT / "runs"))
