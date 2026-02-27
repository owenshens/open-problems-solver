"""Disprover-specific configuration.

Inherits shared settings and adds token budgets, reasoning effort,
review loop limits, and paths specific to the disprover workflow.
"""

from __future__ import annotations

import os
from pathlib import Path

from shared.config import *  # noqa: F401, F403 — re-export shared settings

# ============================================================================
# Phase loop limits
# ============================================================================

MAX_LITERATURE_REVIEWS = int(os.getenv("MAX_LITERATURE_REVIEWS", "2"))
MAX_ANALYSIS_REVIEWS = int(os.getenv("MAX_ANALYSIS_REVIEWS", "2"))
MAX_REVISION_ROUNDS = int(os.getenv("MAX_REVISION_ROUNDS", "3"))

# ============================================================================
# Token budgets (output tokens per operator call)
# ============================================================================

MAX_LITERATURE_TOKENS = int(os.getenv("SEARCH_MAX_LITERATURE_TOKENS", "64000"))
MAX_LITERATURE_PROCESSING_TOKENS = int(os.getenv("SEARCH_MAX_LITERATURE_PROCESSING_TOKENS", "64000"))
MAX_ANALYSIS_TOKENS = int(os.getenv("SEARCH_MAX_ANALYSIS_TOKENS", "128000"))
MAX_ALGORITHM_TOKENS = int(os.getenv("SEARCH_MAX_ALGORITHM_TOKENS", "128000"))
MAX_REVIEW_TOKENS = int(os.getenv("SEARCH_MAX_REVIEW_TOKENS", "64000"))
MAX_COMPLEXITY_TOKENS = int(os.getenv("SEARCH_MAX_COMPLEXITY_TOKENS", "64000"))
MAX_ASSEMBLY_TOKENS = int(os.getenv("SEARCH_MAX_ASSEMBLY_TOKENS", "128000"))

# ============================================================================
# Reasoning effort (all xhigh for GPT-5.3 Codex via codex CLI)
# ============================================================================

REASONING_EFFORT_LITERATURE = os.getenv("SEARCH_REASONING_EFFORT_LITERATURE", "xhigh")
REASONING_EFFORT_LITERATURE_PROCESSING = os.getenv("SEARCH_REASONING_EFFORT_LITERATURE_PROCESSING", "xhigh")
REASONING_EFFORT_ANALYSIS = os.getenv("SEARCH_REASONING_EFFORT_ANALYSIS", "xhigh")
REASONING_EFFORT_ALGORITHM = os.getenv("SEARCH_REASONING_EFFORT_ALGORITHM", "xhigh")
REASONING_EFFORT_REVIEW = os.getenv("SEARCH_REASONING_EFFORT_REVIEW", "xhigh")
REASONING_EFFORT_COMPLEXITY = os.getenv("SEARCH_REASONING_EFFORT_COMPLEXITY", "xhigh")
REASONING_EFFORT_ASSEMBLY = os.getenv("SEARCH_REASONING_EFFORT_ASSEMBLY", "xhigh")

# ============================================================================
# Text verbosity
# ============================================================================

TEXT_VERBOSITY = os.getenv("SEARCH_TEXT_VERBOSITY", "high")

# ============================================================================
# Checkpointing
# ============================================================================

ENABLE_CHECKPOINTING = os.getenv("SEARCH_ENABLE_CHECKPOINTING", "true").lower() in {"1", "true", "yes"}
CHECKPOINT_AFTER_ANALYSIS = os.getenv("SEARCH_CHECKPOINT_AFTER_ANALYSIS", "false").lower() in {"1", "true", "yes"}

# ============================================================================
# Web search (disprover-specific overrides)
# ============================================================================

ENABLE_WEB_SEARCH = os.getenv("SEARCH_ENABLE_WEB_SEARCH", "true").lower() in {"1", "true", "yes"}
MAX_SEARCH_RESULTS = int(os.getenv("SEARCH_MAX_SEARCH_RESULTS", "10"))
MAX_SEARCH_QUERIES = int(os.getenv("SEARCH_MAX_SEARCH_QUERIES", "20"))

# ============================================================================
# Paths
# ============================================================================

_DISPROVER_ROOT = Path(__file__).resolve().parent
RUNS_DIR = Path(os.getenv("SEARCH_RUNS_DIR", str(_DISPROVER_ROOT / "runs")))

# ============================================================================
# Logging
# ============================================================================

VERBOSE = os.getenv("SEARCH_VERBOSE", "false").lower() in {"1", "true", "yes"}
