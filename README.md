# open-problems-solver

AI-driven solver for open mathematical problems, covering both **proving** and **disproving** (counterexample search).

## Overview

This project contains two complementary pipelines that use large language models to attack open problems in mathematics:

- **Prover** — Decomposes a conjecture into lemmas, proves each via recursive decomposition and backtracking, and validates correctness through dual auditing (GPT self-audit + Claude cross-audit).
- **Disprover** — Designs a computational counterexample search through a 6-phase deliberation pipeline: literature research, structural analysis, algorithm design, multi-round adversarial review, complexity estimation, and document assembly.

Both pipelines support OpenAI (GPT / Codex) and Anthropic (Claude) models, with automatic retry and fallback chains.

## Setup

```bash
# Clone and install
git clone https://github.com/<your-username>/open-problems-solver.git
cd open-problems-solver
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI and Anthropic API keys
```

### Requirements

- Python >= 3.10
- An OpenAI API key (for GPT / Codex models)
- An Anthropic API key (for Claude models)

## Usage

### Prover

Prove a mathematical statement by decomposing it into lemmas and proving each:

```bash
# Run on a problem
python -m prover examples/prover/smoke_test_simple.json

# Specify output path
python -m prover examples/prover/erdos_993.json output/erdos_993

# Dry run (only generate the proof plan, don't prove lemmas)
python -m prover examples/prover/test_compactness.json --dry-run

# Mock mode (no API calls, for testing)
python -m prover examples/prover/smoke_test_simple.json --mock
```

Problem JSON format:
```json
{
  "id": "smoke_test_simple",
  "statement": "If f: R -> R is continuous and f(a) < 0 < f(b), then there exists c in (a,b) such that f(c) = 0",
  "background": "This is the Intermediate Value Theorem...",
  "domain": "real_analysis",
  "tags": []
}
```

### Disprover

Design a counterexample search algorithm for a conjecture:

```bash
# Run on a problem
python -m disprover examples/disprover/ramsey_r55.json

# Resume from a checkpoint
python -m disprover examples/disprover/erdos_1082.json --resume runs/erdos_1082/20260218-120000/

# Pause after structural analysis for inspection
python -m disprover examples/disprover/erdos_1082.json --checkpoint-after-analysis

# Mock mode
python -m disprover examples/disprover/ramsey_r55.json --mock
```

Problem JSON format:
```json
{
  "id": "ramsey_r55",
  "statement": "R(5,5) <= 48",
  "win_condition": "Exhibit a 2-coloring of E(K_49) with no monochromatic K_5",
  "domain": "Graph Theory / Ramsey Theory",
  "background": "R(5,5) >= 43 (Exoo 1989). R(5,5) <= 48 (Angeltveit-McKay 2017).",
  "known_bounds": "43 <= R(5,5) <= 48",
  "tags": ["ramsey-theory"],
  "strategy_suggestions": [
    {
      "suggestion": "Start from Paley-type constructions",
      "rationale": "Paley graphs are the primary source of Ramsey lower bounds",
      "priority": "high"
    }
  ]
}
```

The disprover also supports `context_documents` (inline text or `@file_reference` paths), `prior_attempts`, and `search_constraints` in the problem JSON.

### Batch Mode (Prover)

Run the prover on multiple problems:

```bash
python -m prover.batch_run examples/prover/*.json --outdir output/batch
```

### Cost Estimation

Estimate API costs from a run's usage log:

```bash
python tools/estimate_cost.py runs/<problem_id>/<timestamp>/usage.json
```

## Architecture

```
shared/          Common infrastructure (LLM agents, config, utilities)
prover/          Theorem proving pipeline
disprover/       Counterexample search pipeline
examples/        Example problem files
tools/           Utility scripts
```

### Prover Pipeline

1. **Literature Research** — Search arXiv, zbMATH, MathOverflow for relevant results
2. **Global Planning** — Decompose the problem into a DAG of lemmas
3. **Lemma Proving** — Prove each lemma (with recursive decomposition up to depth 25)
4. **Dual Auditing** — GPT self-audit + Claude cross-audit for each lemma
5. **Integration** — Assemble the full proof from proved lemmas
6. **Final Audit** — End-to-end verification of the complete proof

### Disprover Pipeline

1. **Literature Survey** — Research what is known about the conjecture
2. **Structural Analysis** — Determine what a counterexample must look like
3. **Algorithm Design** — Design a staged search algorithm with runnable Python code
4. **Multi-Round Review** — Correctness, optimality, and adversarial reviews with revision
5. **Complexity Estimation** — Estimate computational cost and feasibility
6. **Document Assembly** — Produce a self-contained search document

## Configuration

Both modules inherit shared settings from environment variables (or `.env`). Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `API_MODE` | `api` | `api` for API calls, `cli` for codex/claude CLI tools |
| `SEARCH_ENABLE_WEB_SEARCH` | `true` | Enable web search during literature phase |

See `shared/config.py`, `prover/config.py`, and `disprover/config.py` for the full list.

## CLI Mode

Both pipelines support running via the `codex` and `claude` CLI tools instead of the API. Set `API_MODE=cli` in your `.env` to use this mode. This is useful when you have CLI access but not direct API access.

## License

MIT
