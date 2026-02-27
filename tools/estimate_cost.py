#!/usr/bin/env python3
"""Real-time cost estimator for proof runs."""

import json
from pathlib import Path
import sys

# Pricing (per 1M tokens)
GPT_INPUT = 50  # GPT-5.2 with xhigh reasoning (estimated)
GPT_OUTPUT = 100
CLAUDE_INPUT = 15  # Claude Opus 4.5
CLAUDE_OUTPUT = 75

def estimate_cost(usage_file):
    """Estimate cost from usage JSON."""
    if not Path(usage_file).exists():
        print(f"Usage file not found: {usage_file}")
        return

    with open(usage_file) as f:
        usage = json.load(f)

    total_cost = 0.0

    for model, counts in usage.items():
        input_tokens = counts.get("input_tokens", 0)
        output_tokens = counts.get("output_tokens", 0)

        if "gpt" in model.lower():
            cost = (input_tokens * GPT_INPUT + output_tokens * GPT_OUTPUT) / 1_000_000
        elif "claude" in model.lower():
            cost = (input_tokens * CLAUDE_INPUT + output_tokens * CLAUDE_OUTPUT) / 1_000_000
        else:
            cost = 0

        total_cost += cost
        print(f"{model}:")
        print(f"  Input:  {input_tokens:,} tokens")
        print(f"  Output: {output_tokens:,} tokens")
        print(f"  Cost:   ${cost:.2f}")

    print(f"\nTOTAL COST: ${total_cost:.2f}")
    return total_cost

if __name__ == "__main__":
    if len(sys.argv) > 1:
        estimate_cost(sys.argv[1])
    else:
        print("Usage: python estimate_cost.py <path_to_usage.json>")
        print("\nExample:")
        print("  python estimate_cost.py runs/erdos_993/20260216-165856/usage.json")
