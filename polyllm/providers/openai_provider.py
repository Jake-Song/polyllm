import os
from collections.abc import Iterator

from openai import OpenAI

from .truncation import TRUNCATED


#: Thinking-effort levels this provider accepts, shallowest first. The SDK's
#: type also allows "minimal", but the current models reject it with a 400, so
#: it isn't offered.
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _reasoning(effort: str | None) -> dict:
    reasoning = {"summary": "auto"}
    if effort:
        reasoning["effort"] = effort
    return reasoning


def default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.6-sol")


# models.list() returns every model on the account — speech, image, embedding
# and realtime ones included — and carries no capability field to filter on,
# so text models have to be picked out by name.
_NOT_TEXT = (
    "transcribe", "realtime", "tts", "audio", "whisper",
    "image", "sora", "dall-e",
    "embedding", "moderation", "search",
    "instruct", "davinci", "babbage",
)


def list_models() -> list[dict]:
    """Text models the current key can use, newest first."""
    models = sorted(OpenAI().models.list(), key=lambda m: m.created, reverse=True)
    return [
        {"id": model.id, "label": model.id}
        for model in models
        if not any(word in model.id for word in _NOT_TEXT)
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
    client = OpenAI()

    kwargs = {}
    if system:
        kwargs["instructions"] = system

    response = client.responses.create(
        model=model or default_model(),
        max_output_tokens=max_tokens,
        input=messages,
        reasoning=_reasoning(effort),
        **kwargs,
    )

    if raw:
        return response.to_json(indent=2)

    return response.output_text


def stream(
    messages: list[dict],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    include_raw: bool = False,
) -> Iterator[tuple[str, str]]:
    client = OpenAI()

    kwargs = {}
    if system:
        kwargs["instructions"] = system

    with client.responses.stream(
        model=model or default_model(),
        max_output_tokens=max_tokens,
        input=messages,
        reasoning=_reasoning(effort),
        **kwargs,
    ) as response:
        final = None
        for event in response:
            if event.type == "response.output_text.delta":
                yield "text", event.delta
            elif event.type in ("response.completed", "response.incomplete"):
                # get_final_response() raises on an incomplete response, so the
                # finished object has to be taken off the event itself.
                final = event.response

        if final is not None:
            details = final.incomplete_details
            if details and details.reason == "max_output_tokens":
                yield "notice", TRUNCATED
            if include_raw:
                yield "raw", final.to_json(indent=2)
