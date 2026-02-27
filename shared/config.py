"""Shared configuration for the open-problems-solver.

API keys, model settings, and infrastructure shared by prover and disprover.
All settings can be overridden via environment variables (.env recommended).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# API Keys
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================================
# API Mode: "api" (direct API calls) or "cli" (codex/claude CLIs)
# ============================================================================

API_MODE = os.getenv("API_MODE", "api")

# ============================================================================
# Models
# ============================================================================

if API_MODE == "cli":
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5.3-codex")
    CODEX_FALLBACK_MODELS = [
        "gpt-5.3-codex",
        "gpt-5-codex",
        "gpt-5.2-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
    ]
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "opus")
else:
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5.2-pro")
    CODEX_FALLBACK_MODELS = []
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

CLAUDE_MODEL_FALLBACK = os.getenv(
    "CLAUDE_MODEL_FALLBACK",
    "sonnet" if API_MODE == "cli" else "claude-sonnet-4-5-20250929",
)

# ============================================================================
# Claude thinking configuration
# ============================================================================

CLAUDE_THINKING_MODE = os.getenv("CLAUDE_THINKING_MODE", "extended")
CLAUDE_THINKING_BUDGET = int(os.getenv("CLAUDE_THINKING_BUDGET", "128000"))
CLAUDE_OUTPUT_EFFORT = os.getenv("CLAUDE_OUTPUT_EFFORT", "high")

# ============================================================================
# Default token budget (used by shared agents as fallback)
# ============================================================================

DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "128000"))
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS", "128000"))

# ============================================================================
# Temperature (not used when reasoning.effort != "none")
# ============================================================================

GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.7"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))

# ============================================================================
# Background mode for long-running calls
# ============================================================================

USE_BACKGROUND_MODE = os.getenv("USE_BACKGROUND_MODE", "true").lower() in {"1", "true", "yes"}

# ============================================================================
# Web search
# ============================================================================

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in {"1", "true", "yes"}
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "150"))

ENABLE_ARXIV_SEARCH = os.getenv("ENABLE_ARXIV_SEARCH", "false").lower() in {"1", "true", "yes"}
ENABLE_ZBMATH_SEARCH = os.getenv("ENABLE_ZBMATH_SEARCH", "false").lower() in {"1", "true", "yes"}
ENABLE_MATHOVERFLOW_SEARCH = os.getenv("ENABLE_MATHOVERFLOW_SEARCH", "false").lower() in {"1", "true", "yes"}
ENABLE_SEARCH_CACHE = os.getenv("ENABLE_SEARCH_CACHE", "true").lower() in {"1", "true", "yes"}
SEARCH_CACHE_TTL_HOURS = int(os.getenv("SEARCH_CACHE_TTL_HOURS", "24"))
USE_AGENT_QUERY_GENERATION = os.getenv("USE_AGENT_QUERY_GENERATION", "true").lower() in {"1", "true", "yes"}

# ============================================================================
# Logging
# ============================================================================

VERBOSE_DEFAULT = os.getenv("VERBOSE", "false").lower() in {"1", "true", "yes"}


def require_api_keys(*, allow_missing: bool = False) -> None:
    """Validate that API keys are present.

    In CLI mode, API keys are not required (uses codex/claude commands).
    """
    if allow_missing:
        return
    if API_MODE == "cli":
        return

    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Create a .env file or export them in your shell."
        )
