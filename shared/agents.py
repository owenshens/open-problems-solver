"""LLM agent layer for the open-problems-solver.

Supports:
- OpenAI Responses API with reasoning.effort (GPT-5.2 Pro)
- Claude with adaptive/extended thinking (Opus 4.6 / Sonnet 4.5)
- Optional CLI mode (codex / claude CLIs)
- Background mode for long-running calls
- Web search capability for all agents
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from shared import config

# Optional imports (for API mode)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


@dataclass
class AgentResponse:
    """Standard wrapper for LLM responses with minimal metadata."""

    text: str
    usage: dict[str, int]
    raw: Any | None = None
    thinking: str | None = None


# ============================================================================
# Mock mode (offline testing)
# ============================================================================

_MOCK_MODE: bool = False
_MOCK_SCENARIO: str = "clean"


def set_mock_mode(enabled: bool, *, scenario: str = "clean") -> None:
    global _MOCK_MODE, _MOCK_SCENARIO
    _MOCK_MODE = enabled
    _MOCK_SCENARIO = scenario


def is_mock_mode() -> bool:
    return _MOCK_MODE


def _mock_json(obj: dict) -> AgentResponse:
    return AgentResponse(text=json.dumps(obj, indent=2), usage={"mock_calls": 1}, raw=obj)


def _mock_text(text: str) -> AgentResponse:
    return AgentResponse(text=text, usage={"mock_calls": 1}, raw=None)


def _mock_should_fail(lemma_id: Optional[str] = None) -> bool:
    """Deterministic-ish failure to exercise retry/blocked paths."""
    if _MOCK_SCENARIO == "blocked" and lemma_id:
        return lemma_id.strip().upper() in {"L3", "L4", "L5", "L6", "L7", "L8"}
    return False


# ============================================================================
# API clients
# ============================================================================

_openai_client: OpenAI | None = None
_anthropic_client: Anthropic | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        if not config.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (set it in .env or env vars)")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if Anthropic is None:
            raise RuntimeError("anthropic package not installed")
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Missing ANTHROPIC_API_KEY (set it in .env or env vars)")
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


# ============================================================================
# CLI helpers
# ============================================================================

def _prepare_cli_env() -> dict[str, str]:
    """Prepare environment for CLI calls (handle nested sessions, etc.)."""
    env = os.environ.copy()
    if "CLAUDECODE" in env:
        del env["CLAUDECODE"]
    env["MAX_THINKING_TOKENS"] = "127998"
    return env


def _call_codex_cli(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system_prompt: str = "",
    json_mode: bool = False,
) -> AgentResponse:
    """Call codex CLI with a prompt and automatic model fallback."""
    env = _prepare_cli_env()

    parts = []
    if system_prompt:
        parts.append(system_prompt)
        parts.append("\n\n---\n")
    parts.append(prompt)
    if json_mode and "json" not in prompt.lower():
        parts.append("\n\nRespond in JSON format.")
    combined_prompt = "".join(parts)

    prompt_file = tempfile.mktemp(suffix=".txt", prefix="codex_prompt_")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(combined_prompt)

    fallback_models = getattr(config, 'CODEX_FALLBACK_MODELS', [model])
    if model not in fallback_models:
        fallback_models = [model] + fallback_models

    last_error = None

    for attempt_model in fallback_models:
        output_file = tempfile.mktemp(suffix=".txt", prefix="codex_out_")
        try:
            cmd = [
                "codex", "exec",
                "-m", attempt_model,
                "--skip-git-repo-check",
                "--ephemeral",
                "-s", "read-only",
                "--output-last-message", output_file,
                "-",
            ]
            result = subprocess.run(
                cmd, env=env, input=combined_prompt,
                capture_output=True, text=True, timeout=7200,
            )

            stderr_lower = result.stderr.lower()
            if "does not exist" in stderr_lower or "not supported" in stderr_lower or "no access" in stderr_lower:
                last_error = f"Model {attempt_model} not accessible: {result.stderr[:200]}"
                if os.path.exists(output_file):
                    try:
                        os.unlink(output_file)
                    except Exception:
                        pass
                continue

            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                os.unlink(output_file)
                if not text:
                    last_error = f"codex returned empty response with {attempt_model}. stderr: {result.stderr[:500]}"
                    continue
                return AgentResponse(text=text, usage={"codex_calls": 1, "model_used": attempt_model})

            elif result.returncode != 0:
                last_error = f"codex CLI failed with {attempt_model} (exit code {result.returncode}): {result.stderr[:500]}"
                continue
            else:
                text = result.stdout.strip() if result.stdout else ""
                if not text:
                    last_error = f"codex produced no output with {attempt_model}. stderr: {result.stderr[:500]}"
                    continue
                return AgentResponse(text=text, usage={"codex_calls": 1, "model_used": attempt_model})

        except Exception as e:
            last_error = f"Exception with {attempt_model}: {str(e)}"
            continue
        finally:
            if os.path.exists(output_file):
                try:
                    os.unlink(output_file)
                except Exception:
                    pass

    if os.path.exists(prompt_file):
        try:
            os.unlink(prompt_file)
        except Exception:
            pass

    raise RuntimeError(f"All codex models failed. Last error: {last_error}")


def _normalize_claude_model_for_cli(model: str) -> str:
    """Convert full model name to CLI alias if needed."""
    model_lower = model.lower()
    if "opus" in model_lower:
        return "opus"
    elif "sonnet" in model_lower:
        return "sonnet"
    elif "haiku" in model_lower:
        return "haiku"
    return model


def _call_claude_cli(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system_prompt: str = "",
) -> AgentResponse:
    """Call claude CLI with a prompt."""
    env = _prepare_cli_env()
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

    cli_model = _normalize_claude_model_for_cli(model)
    cmd = ["claude", "-p", full_prompt, "--model", cli_model, "--max-turns", "1"]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "(no error output)"
        raise RuntimeError(f"claude CLI failed (code {result.returncode}): {error_msg}")

    return AgentResponse(text=result.stdout.strip(), usage={"claude_calls": 1})


# ============================================================================
# Web Search
# ============================================================================

def web_search(query: str, *, max_results: int = 5) -> list[dict[str, str]]:
    """Perform web search using DuckDuckGo (no API key needed)."""
    try:
        import urllib.parse
        import urllib.request

        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')

        results = []
        result_blocks = re.findall(
            r'<a class="result__a".*?href="(.*?)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>',
            html, re.DOTALL,
        )

        for url, title, snippet in result_blocks[:max_results]:
            title = re.sub(r'<.*?>', '', title).strip()
            snippet = re.sub(r'<.*?>', '', snippet).strip()
            url = re.sub(r'<.*?>', '', url).strip()
            title = title.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            snippet = snippet.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            if url and title:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results
    except Exception as e:
        print(f"Warning: Web search failed: {e}")
        return []


def web_search_formatted(query: str, *, max_results: int = 5) -> str:
    """Perform web search and return formatted results as text."""
    results = web_search(query, max_results=max_results)
    if not results:
        return "No web search results found."

    output = [f"Web Search Results for: {query}\n", "=" * 60]
    for i, result in enumerate(results, 1):
        output.append(f"\n[{i}] {result['title']}")
        output.append(f"    {result['url']}")
        if result.get('snippet'):
            output.append(f"    {result['snippet'][:200]}...")
    return "\n".join(output)


# ============================================================================
# GPT (Responses API)
# ============================================================================

def _usage_openai_responses(resp: Any) -> dict[str, int]:
    """Extract usage from Responses API response."""
    usage: dict[str, int] = {"openai_calls": 1}
    u = getattr(resp, "usage", None)
    if u is None:
        return usage
    usage.update({
        "openai_input_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "openai_output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "openai_reasoning_tokens": int(getattr(u, "reasoning_tokens", 0) or 0),
        "openai_total_tokens": int(getattr(u, "total_tokens", 0) or 0),
    })
    return usage


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def call_gpt(
    prompt: str,
    *,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    system_prompt: str = "",
    reasoning_effort: str = "medium",
    text_verbosity: str = "medium",
    use_background: bool = False,
) -> AgentResponse:
    """Call GPT using Responses API with reasoning.effort support."""
    if _MOCK_MODE:
        lower = prompt.lower()
        if "decompose" in lower and "lemmas" in lower:
            return _mock_json({
                "lemmas": [
                    {"id": "L1", "statement": "(Mock) Key reduction.", "assumptions": [], "dependencies": [], "justification": "Setup"},
                    {"id": "L2", "statement": "(Mock) Auxiliary property.", "assumptions": [], "dependencies": ["L1"], "justification": "Technical step"},
                    {"id": "L3", "statement": "(Mock) Conclude.", "assumptions": [], "dependencies": ["L2"], "justification": "Finish"},
                ],
                "proof_strategy": "(Mock) Reduce -> prove -> conclude."
            })
        if "auditing" in lower and json_mode:
            return _mock_json({"verdict": "PASS", "confidence": 0.85, "issues": [], "feedback": "(Mock) Consistent."})
        if "interface" in lower:
            return _mock_text("**Lemma (Mock):** Result holds.\n**Assumptions:** As given.\n**Usage:** Apply directly.")
        if "assembling" in lower or "integrating" in lower:
            return _mock_text("# Proof (Mock)\n\n(Mock) Integrated proof.")
        return _mock_text("Proof: (Mock) Standard proof steps.")

    # CLI mode
    if config.API_MODE == "cli":
        return _call_codex_cli(
            prompt,
            model=config.GPT_MODEL,
            max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
            system_prompt=system_prompt,
            json_mode=json_mode,
        )

    # API mode with Responses API
    client = _get_openai_client()
    instructions = system_prompt if system_prompt else ""

    _model = config.GPT_MODEL
    if "codex" in _model and text_verbosity != "medium":
        text_verbosity = "medium"
    text_config: dict[str, Any] = {"verbosity": text_verbosity}
    if json_mode:
        text_config["format"] = {"type": "json_object"}
        if "json" not in prompt.lower():
            prompt += "\n\nRespond in JSON format."

    kwargs: dict[str, Any] = {
        "model": _model,
        "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": reasoning_effort},
        "text": text_config,
        "max_output_tokens": max_tokens or config.DEFAULT_MAX_TOKENS,
    }

    if use_background:
        kwargs["background"] = True

    resp = client.responses.create(**kwargs)

    if use_background:
        while resp.status != "completed":
            if resp.status == "failed":
                raise RuntimeError(f"GPT background job failed: {getattr(resp, 'error', 'unknown')}")
            time.sleep(2)
            resp = client.responses.retrieve(resp.id)

    text_parts = []
    thinking_parts = []
    for item in getattr(resp, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        for block in getattr(item, "summary", []) or []:
            if hasattr(block, "text"):
                thinking_parts.append(block.text)

    text = "".join(text_parts)
    thinking = "".join(thinking_parts) if thinking_parts else None

    return AgentResponse(
        text=text.strip(),
        usage=_usage_openai_responses(resp),
        raw=resp,
        thinking=thinking,
    )


# ============================================================================
# Claude (Messages API with thinking)
# ============================================================================

def _usage_anthropic(resp: Any) -> dict[str, int]:
    usage: dict[str, int] = {"anthropic_calls": 1}
    u = getattr(resp, "usage", None)
    if u is None:
        return usage
    usage.update({
        "anthropic_input_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "anthropic_output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "anthropic_thinking_tokens": int(getattr(u, "thinking_tokens", 0) or 0),
    })
    return usage


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def call_claude(
    prompt: str,
    *,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    system_prompt: str = "",
    lemma_id_for_mock: Optional[str] = None,
) -> AgentResponse:
    """Call Claude with adaptive/extended thinking support."""
    if _MOCK_MODE:
        lower = prompt.lower()
        if "output format" in lower and "verdict" in lower and json_mode:
            if _mock_should_fail(lemma_id_for_mock):
                return _mock_json({
                    "verdict": "FAIL", "confidence": 0.8,
                    "issues": [{"severity": "critical", "location": "[S3]", "summary": "(Mock) Simulated failure", "detail": "Exercise retry.", "patch": "Fix S3"}],
                    "feedback": "(Mock) Please revise."
                })
            return _mock_json({"verdict": "PASS", "confidence": 0.8, "issues": [], "feedback": "(Mock) Correct."})
        if "final verification" in lower and json_mode:
            return _mock_json({
                "verdict": "PASS", "confidence": 0.8, "issues": [], "feedback": "(Mock) Coherent.",
                "missing_links": [], "global_consistency": "None", "recommendation": "accept",
            })
        return _mock_text("(Mock) Claude output")

    # CLI mode
    if config.API_MODE == "cli":
        final_prompt = prompt
        if json_mode:
            final_prompt += "\n\nIMPORTANT: Respond with valid JSON only."
        return _call_claude_cli(
            final_prompt,
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
            system_prompt=system_prompt,
        )

    # API mode
    client = _get_anthropic_client()
    final_prompt = prompt
    if json_mode:
        final_prompt += "\n\nIMPORTANT: Respond with valid JSON only."

    full_system = system_prompt if system_prompt else ""

    thinking_config: dict[str, Any] = {}
    output_config: dict[str, Any] = {}

    if config.CLAUDE_MODEL.startswith("claude-opus-4"):
        thinking_config = {"type": "adaptive"}
        output_config = {"effort": config.CLAUDE_OUTPUT_EFFORT}
    elif config.CLAUDE_MODEL.startswith("claude-sonnet-4"):
        thinking_config = {"type": "enabled", "budget_tokens": config.CLAUDE_THINKING_BUDGET}

    kwargs: dict[str, Any] = {
        "model": config.CLAUDE_MODEL,
        "max_tokens": max_tokens or config.DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": final_prompt}],
    }
    if full_system:
        kwargs["system"] = full_system
    if thinking_config:
        kwargs["thinking"] = thinking_config
    if output_config:
        kwargs["output_config"] = output_config

    resp = client.messages.create(**kwargs)

    text_parts = []
    thinking_parts = []
    for block in getattr(resp, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            thinking_parts.append(getattr(block, "thinking", ""))

    text = "".join(text_parts)
    thinking = "".join(thinking_parts) if thinking_parts else None

    return AgentResponse(
        text=text.strip(),
        usage=_usage_anthropic(resp),
        raw=resp,
        thinking=thinking,
    )
