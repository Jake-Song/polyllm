import json
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..client import PROVIDERS, PolyLLM

STATIC_DIR = Path(__file__).parent / "static"

# Which env vars make a provider usable. Any one of them is enough.
_KEY_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    provider: str
    messages: list[Message]
    model: str | None = None
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    effort: str | None = None


#: Model lists change rarely and cost a network round-trip, so keep them for
#: the life of the process. Restart the server to pick up a new model.
_MODEL_CACHE: dict[str, list[dict]] = {}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="polyllm")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/providers")
    def providers() -> list[dict]:
        return [
            {
                "name": name,
                "default_model": PolyLLM(name).default_model(),
                "key_present": any(os.getenv(var) for var in _KEY_VARS[name]),
                "efforts": list(PolyLLM(name).efforts()),
            }
            for name in PROVIDERS
        ]

    @app.get("/api/models/{provider}")
    def models(provider: str) -> dict:
        try:
            llm = PolyLLM(provider=provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if provider not in _MODEL_CACHE:
            try:
                _MODEL_CACHE[provider] = llm.list_models()
            except Exception as e:
                # The model box still accepts a hand-typed id, so a failure here
                # degrades the UI rather than breaking it.
                return {
                    "models": [],
                    "default": llm.default_model(),
                    "error": f"{type(e).__name__}: {e}",
                }

        return {"models": _MODEL_CACHE[provider], "default": llm.default_model()}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        try:
            llm = PolyLLM(provider=request.provider, model=request.model or None)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        messages = [message.model_dump() for message in request.messages]

        # A plain (non-async) generator: Starlette runs it in a threadpool, so the
        # blocking SDK calls never stall the event loop.
        def event_stream() -> Iterator[str]:
            try:
                for kind, text in llm.stream(
                    messages,
                    system=request.system or None,
                    max_tokens=request.max_tokens,
                    effort=request.effort or None,
                    include_raw=True,
                ):
                    yield _sse({"type": kind, "text": text})
            except Exception as e:
                # The response has already started, so errors can only be reported
                # in-band rather than as an HTTP status.
                yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
