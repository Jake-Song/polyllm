import os

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")


def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    raw: bool = False,
) -> str:
    client = OpenAI()

    kwargs = {}
    if system:
        kwargs["instructions"] = system

    response = client.responses.create(
        model=model or DEFAULT_MODEL,
        max_output_tokens=max_tokens,
        input=prompt,
        reasoning={"summary": "auto"},
        **kwargs,
    )

    if raw:
        return response.to_json(indent=2)

    return response.output_text
