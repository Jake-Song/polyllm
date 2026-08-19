import os

from google import genai
from google.genai import types

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")


def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    raw: bool = False,
) -> str:
    client = genai.Client()

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )

    if raw:
        return response.model_dump_json(indent=2)

    return response.text
