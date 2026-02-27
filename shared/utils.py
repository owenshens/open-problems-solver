"""Shared utilities for the open-problems-solver.

Generic helpers used by both prover and disprover modules.
No dependency on config or domain-specific models.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional


def ensure_dir(path: str | Path) -> Path:
    """Create directory and parents, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_timestamp() -> str:
    """ISO-like, filesystem-friendly timestamp."""
    return time.strftime("%Y%m%d-%H%M%S")


def merge_usage(total: dict[str, int], add: dict[str, int]) -> dict[str, int]:
    """Accumulate API usage dicts (handles string values like model_used)."""
    for k, v in add.items():
        if isinstance(v, str):
            total[k] = v
        else:
            total[k] = total.get(k, 0) + int(v)
    return total


# ============================================================================
# JSON parsing
# ============================================================================


def strip_code_fences(text: str) -> str:
    """If text is a fenced code block, return its contents."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _find_balanced_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
                continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from a model response.

    Handles raw JSON, markdown fenced code blocks, and surrounding commentary.
    """
    raw = text.strip()

    # 1) Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Look for explicit ```json fenced blocks
    fence_patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]
    for pat in fence_patterns:
        m = re.search(pat, raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # 3) Balanced object scan
    candidate = _find_balanced_json_object(raw)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from response")


def parse_json_robust(text: str, *, default: Optional[dict] = None) -> dict:
    """Parse JSON with multiple fallback strategies.

    Returns parsed dict, or default dict with error info if all strategies fail.
    """
    if not text or not text.strip():
        return default or {"error": "Empty text"}

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code blocks
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract first complete JSON object (regex)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Balanced brace scan
    try:
        brace_count = 0
        start_idx = text.find('{')
        if start_idx >= 0:
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start_idx:i + 1])
                        except json.JSONDecodeError:
                            break
    except Exception:
        pass

    # Strategy 5: Try array format
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {"items": parsed} if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            pass

    return default or {
        "error": "Could not parse JSON",
        "raw_text": text[:500],
        "text_length": len(text),
    }


def validate_json_structure(data: dict, required_keys: list[str]) -> tuple[bool, str]:
    """Validate that parsed JSON has required structure."""
    if not isinstance(data, dict):
        return False, f"Expected dict, got {type(data).__name__}"
    if "error" in data:
        return False, f"Parse error: {data.get('error')}"
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing required keys: {', '.join(missing_keys)}"
    return True, ""


def extract_json_from_response(
    response_text: str,
    *,
    required_keys: Optional[list[str]] = None,
    default: Optional[dict] = None,
) -> dict:
    """Extract and validate JSON from LLM response."""
    parsed = parse_json_robust(response_text, default=default)
    if required_keys:
        is_valid, error = validate_json_structure(parsed, required_keys)
        if not is_valid:
            return default or {"error": f"Validation failed: {error}", "partial_data": parsed}
    return parsed
