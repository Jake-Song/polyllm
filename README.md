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
uv run main.py claude "Solve this puzzle..." --effort max
```

## Web UI

```bash
uv run -m polyllm.web
```

Then open <http://127.0.0.1:8000>. It's a multi-turn chat with a provider/model/system
sidebar and responses that stream in as they're generated. Use `--port` to move it and
`--host` if you really want it off localhost.

The model box is populated from the provider's own model list (fetched once per run and
cached), so you can pick from what your key can actually reach — and it stays free text,
so an id the list doesn't know still works. Leave it blank for the provider default.

Thinking effort is a dropdown whose levels come from the selected provider: Claude takes
`low`/`medium`/`high`/`xhigh`/`max`, OpenAI adds `none`, Gemini takes `low`/`medium`/`high`.
Leave it on "provider default" to send nothing.

The **Raw** tab logs the provider's own response object for every call — the same JSON the
CLI's `--raw` prints, one collapsible entry per turn, failures included. Gemini has no
single final object when streaming, so its entry is the array of chunks the SDK returned.

Conversations are saved as you go, so a reload or a restart doesn't lose them. Past chats
are listed at the top of the sidebar — click one to reopen it, `✎` to rename it, `✕` to
delete it, and **+ New chat** to start fresh. A chat is only created once you send the
first message, which also gives it its title.

Storage is a SQLite file at `~/.polyllm/conversations.db`; point `POLYLLM_DB` somewhere
else to move it, and delete the file to wipe everything. Only the transcript is stored:
the sidebar settings and the Raw log belong to the session, not the chat, so a reopened
conversation keeps whatever provider and model you have selected now and starts with an
empty Raw tab.

Only a turn you actually received is stored. A failed or stopped answer leaves the
question in the transcript with no reply, matching what stays on screen.

Providers whose API key isn't set are greyed out in the sidebar.

## Library usage

```python
from polyllm import PolyLLM

llm = PolyLLM(provider="claude")  # or "openai", "gemini"
print(llm.chat("What is the capital of France?"))
```

For a conversation, pass the history instead — or stream it:

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hi Alice!"},
    {"role": "user", "content": "What is my name?"},
]
print(llm.chat_messages(messages))

for kind, text in llm.stream(messages):
    if kind == "text":
        print(text, end="", flush=True)
```

`stream()` tags each chunk: `text` for output, `notice` for a warning about the call (such
as hitting the token cap), and — with `include_raw=True` — a final `raw` carrying the
provider's response object as JSON.

`chat`, `chat_messages` and `stream` all take `effort=` to set thinking depth. Ask a
provider what it supports:

```python
llm.efforts()      # ('low', 'medium', 'high', 'xhigh', 'max') for claude
llm.list_models()  # [{'id': 'claude-opus-5', 'label': 'Claude Opus 5'}, ...]
```

Each provider reads its default model from an env var (`ANTHROPIC_MODEL`,
`OPENAI_MODEL`, `GEMINI_MODEL`), or pass `model=` explicitly:

```python
llm = PolyLLM(provider="openai", model="gpt-4o-mini")
```
