import os
from collections.abc import Iterator

import anthropic

from .truncation import TRUNCATED


#: Thinking-effort levels this provider accepts, shallowest first.
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _effort_kwargs(effort: str | None) -> dict:
    return {"output_config": {"effort": effort}} if effort else {}


def default_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-opus-5")


def list_models() -> list[dict]:
    """Chat models the current key can use, newest first (the API's own order)."""
    return [
        {"id": model.id, "label": model.display_name}
        for model in anthropic.Anthropic().models.list()
    ]


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    raw: bool = False,
) -> str:
    client = anthropic.Anthropic()

    kwargs = _effort_kwargs(effort)
    if system:
        kwargs["system"] = system

    response = client.messages.create(
        model=model or default_model(),
        max_tokens=max_tokens,
        messages=messages,
        **kwargs,
    )

    if raw:
        return response.to_json(indent=2)

    return "".join(block.text for block in response.content if block.type == "text")


def stream(
    messages: list[dict],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    include_raw: bool = False,
) -> Iterator[tuple[str, str]]:
    client = anthropic.Anthropic()

    kwargs = _effort_kwargs(effort)
    if system:
        kwargs["system"] = system

    with client.messages.stream(
        model=model or default_model(),
        max_tokens=max_tokens,
        messages=messages,
        **kwargs,
    ) as response:
        for text in response.text_stream:
            yield "text", text

        final = response.get_final_message()
        if final.stop_reason == "max_tokens":
            yield "notice", TRUNCATED
        if include_raw:
            yield "raw", final.to_json(indent=2)
