from .providers import anthropic_provider, gemini_provider, openai_provider

_PROVIDERS = {
    "openai": openai_provider.complete,
    "claude": anthropic_provider.complete,
    "anthropic": anthropic_provider.complete,
    "gemini": gemini_provider.complete,
    "google": gemini_provider.complete,
}


class PolyLLM:
    """A single interface for calling OpenAI, Claude, or Gemini."""

    def __init__(self, provider: str, model: str | None = None):
        key = provider.lower()
        if key not in _PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose from: {sorted(set(_PROVIDERS))}"
            )
        self.provider = key
        self.model = model
        self._complete = _PROVIDERS[key]

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        raw: bool = False,
    ) -> str:
        return self._complete(
            prompt, model=self.model, system=system, max_tokens=max_tokens, raw=raw
        )
