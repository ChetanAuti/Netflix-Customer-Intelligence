"""LLM access with a graceful offline fallback.

* If ANTHROPIC_API_KEY is set and the `anthropic` package is installed -> real Claude calls.
* Otherwise `available` is False and every capability falls back to deterministic, rule-based
  logic (so the whole project - app, tests, demos - runs without any API key).

Env: ANTHROPIC_API_KEY, LLM_MODEL (default claude-sonnet-5-5).
"""
from __future__ import annotations

import os


class LLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-5-5")
        self._client = None
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1200, temperature: float = 0.0) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured (set ANTHROPIC_API_KEY and install `anthropic`).")
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature, system=system,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
