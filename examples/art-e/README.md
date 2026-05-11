# ART-E Email Agent

This example trains an email research agent with ART and LangGraph. The agent receives a user question, searches a deterministic inbox with tools, reads matching messages, and returns a sourced final answer.

The checked-in dataset is intentionally small and local so the rollout, scoring, and tests run without Gmail, paid APIs, or private credentials. To train against a real model, set an inference endpoint and run the training script.

## Files

- `art_e/data.py` defines inbox fixtures and train/validation scenarios.
- `art_e/email_tools.py` contains the local search and read tools.
- `art_e/rollout.py` runs the LangGraph ReAct agent and scores the final answer.
- `art_e/train.py` registers a model and runs ART training.
- `tests/` covers retrieval and reward behavior without network access.

## Offline Checks

```bash
uv run --project examples/art-e pytest
```

## Training

```bash
export OPENAI_API_KEY=...
uv run --project examples/art-e python -m art_e.train
```

The default model uses `Qwen/Qwen2.5-7B-Instruct` through the local ART backend. Override the inference model with `ART_E_INFERENCE_MODEL`, `ART_E_INFERENCE_BASE_URL`, and `ART_E_INFERENCE_API_KEY` when using a hosted endpoint.

