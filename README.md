# home

A starter project for local-first generative agents using Ollama + Python.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

## Pull Ollama models

Before running, make sure Ollama is installed and pull the default models:

```bash
ollama pull qwen3-vl:latest
ollama pull nomic-embed-text
```

You can swap these using CLI flags.

## Run

Run with defaults:

```bash
generative-agents
```

Run with custom values:

```bash
generative-agents \
  --model-name qwen3-vl:latest \
  --embed-model nomic-embed-text \
  --agent-count 5 \
  --tick-interval 0.5 \
  --storage-path ./runtime_data \
  --sqlite-path ./runtime_data/state.sqlite3 \
  --chroma-path ./runtime_data/chroma \
  --ticks 20
```

## Runtime safeguards and utilities

- Startup now performs a model health check to verify the configured generation model is available in Ollama.
- `LLMClient` provides async text generation, vision generation, and 1-10 memory importance scoring.
- `EmbeddingClient` batches embedding requests with retry logic (default model `nomic-embed-text`).
- Prompt templates are standardized for perception, poignancy, reflection, planning, and dialogue flows.
- Context budgeting helpers can trim retrieved context to an approximate token budget (default 16k).
