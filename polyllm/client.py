from collections.abc import Iterator

from .providers import anthropic_provider, gemini_provider, openai_provider

_MODULES = {
    "openai": openai_provider,
    "claude": anthropic_provider,
    "anthropic": anthropic_provider,
    "gemini": gemini_provider,
    "google": gemini_provider,
}

_PROVIDERS = {name: module.complete for name, module in _MODULES.items()}
_STREAMERS = {name: module.stream for name, module in _MODULES.items()}

#: The canonical provider names, one per backend.
PROVIDERS = ["openai", "claude", "gemini"]


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
        self._module = _MODULES[key]
        self._complete = _PROVIDERS[key]
        self._stream = _STREAMERS[key]

    def efforts(self) -> tuple[str, ...]:
        """Thinking-effort levels this provider accepts, shallowest first."""
        return self._module.EFFORTS

    def list_models(self) -> list[dict]:
        """The provider's available text models as ``{"id", "label"}`` dicts."""
        return self._module.list_models()

    def default_model(self) -> str:
        """The model this instance will actually call."""
        return self.model or self._module.default_model()

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        raw: bool = False,
    ) -> str:
        return self.chat_messages(
            [{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            raw=raw,
        )

    def chat_messages(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        raw: bool = False,
    ) -> str:
        return self._complete(
            messages,
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            raw=raw,
        )

    def stream(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        include_raw: bool = False,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(kind, text)`` chunks as the response is generated.

        ``kind`` is ``"text"`` for output, ``"notice"`` for a warning about the
        call, and — when ``include_raw`` is set — a final ``"raw"`` carrying the
        provider's own response object as JSON.
        """
        return self._stream(
            messages,
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            include_raw=include_raw,
        )
