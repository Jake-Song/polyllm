# polyllm

A single interface for calling OpenAI, Claude, and Gemini.

## Setup

```bash
uv sync
```

Set whichever API keys you need:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
```

## CLI usage

```bash
uv run main.py claude "What is the capital of France?"
uv run main.py openai "What is the capital of France?" --system "Be terse."
uv run main.py gemini "What is the capital of France?" --model gemini-2.0-flash
```

## Library usage

```python
from polyllm import PolyLLM

llm = PolyLLM(provider="claude")  # or "openai", "gemini"
print(llm.chat("What is the capital of France?"))
```

Each provider reads its default model from an env var (`ANTHROPIC_MODEL`,
`OPENAI_MODEL`, `GEMINI_MODEL`), or pass `model=` explicitly:

```python
llm = PolyLLM(provider="openai", model="gpt-4o-mini")
```
