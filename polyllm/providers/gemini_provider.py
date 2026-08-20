import json
import os
from collections.abc import Iterator

from google import genai
from google.genai import types

from .truncation import TRUNCATED

# Gemini calls the assistant turn "model".
_ROLES = {"assistant": "model", "model": "model", "user": "user"}


#: Thinking-effort levels this provider accepts, shallowest first. The SDK enum
#: also has MINIMAL, but the current models reject it with a 400, so it isn't
#: offered.
EFFORTS = ("low", "medium", "high")


def _config(
    system: str | None, max_tokens: int, effort: str | None
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system,
        # No tools are declared, so automatic function calling only adds a warning.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # Gemini calls thinking effort a "level" and names it in caps.
        thinking_config=(
            types.ThinkingConfig(thinking_level=effort.upper()) if effort else None
        ),
    )


def default_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.7-flash")


def _to_contents(messages: list[dict]) -> list[types.Content]:
    return [
        types.Content(
            role=_ROLES.get(message["role"], "user"),
            parts=[types.Part.from_text(text=message["content"])],
        )
        for message in messages
    ]


# Speech and image models also answer to generateContent, so they need
# excluding by name on top of the capability check.
_NOT_TEXT = ("tts", "image", "embedding")


def list_models() -> list[dict]:
    """Text models the current key can use."""
    client = genai.Client()  # hold the reference: models.list() pages lazily
    models = []
    for model in client.models.list():
        if "generateContent" not in (model.supported_actions or ()):
            continue
        name = model.name.removeprefix("models/")
        if any(word in name for word in _NOT_TEXT):
            continue
        models.append({"id": name, "label": model.display_name or name})
    return models


def _hit_token_cap(chunk) -> bool:
    candidate = chunk.candidates[0] if chunk.candidates else None
    return bool(candidate) and candidate.finish_reason == types.FinishReason.MAX_TOKENS


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    raw: bool = False,
) -> str:
    client = genai.Client()

    response = client.models.generate_content(
        model=model or default_model(),
        contents=_to_contents(messages),
        config=_config(system, max_tokens, effort),
    )

    if raw:
        return response.model_dump_json(indent=2)

    return response.text


def stream(
    messages: list[dict],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    include_raw: bool = False,
) -> Iterator[tuple[str, str]]:
    client = genai.Client()

    chunks = []
    for chunk in client.models.generate_content_stream(
        model=model or default_model(),
        contents=_to_contents(messages),
        config=_config(system, max_tokens, effort),
    ):
        if include_raw:
            chunks.append(chunk.model_dump(mode="json", exclude_none=True))
        if chunk.text:
            yield "text", chunk.text
        if _hit_token_cap(chunk):
            yield "notice", TRUNCATED

    if include_raw:
        # A streamed Gemini call has no single final object, so the raw view is
        # the sequence of chunks the SDK handed back.
        yield "raw", json.dumps(chunks, indent=2)
