import os

import anthropic

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")


def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    raw: bool = False,
) -> str:
    client = anthropic.Anthropic()

    kwargs = {}
    if system:
        kwargs["system"] = system

    response = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )

    if raw:
        return response.to_json(indent=2)

    return "".join(block.text for block in response.content if block.type == "text")
