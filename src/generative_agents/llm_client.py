"""Async LLM client wrappers for text, vision, and memory importance scoring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ollama import AsyncClient, ResponseError

from .config import MODEL_NAME


class LLMClient:
    """High-level async interface for text and vision generation."""

    def __init__(self, model_name: str = MODEL_NAME, host: str | None = None, timeout_s: float = 60.0) -> None:
        self.model_name = model_name
        self._client = AsyncClient(host=host)
        self.timeout_s = timeout_s

    async def generate_text(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt with an optional system instruction."""

        response = await asyncio.wait_for(
            self._client.generate(model=self.model_name, prompt=prompt, system=system),
            timeout=self.timeout_s,
        )
        return response["response"].strip()

    async def generate_with_vision(self, prompt: str, image_paths: list[str]) -> str:
        """Generate text from prompt + local images using a vision-capable model."""

        validated_paths = _validate_image_paths(image_paths)
        response = await asyncio.wait_for(
            self._client.generate(model=self.model_name, prompt=prompt, images=validated_paths),
            timeout=self.timeout_s,
        )
        return response["response"].strip()

    async def score_importance(self, memory_text: str, image_paths: list[str] | None = None) -> int:
        """Score memory importance on a 1-10 integer scale."""

        score_prompt = (
            "Rate the emotional/cognitive importance of the memory from 1 to 10. "
            "Return only JSON with key 'score'.\n"
            f"Memory: {memory_text}"
        )
        if image_paths:
            raw = await self.generate_with_vision(score_prompt, image_paths=image_paths)
        else:
            raw = await self.generate_text(score_prompt)
        parsed = _parse_score(raw)
        return min(max(parsed, 1), 10)


async def ensure_model_available(model_name: str = MODEL_NAME, host: str | None = None) -> None:
    """Raise when the configured generation model is unavailable in local Ollama."""

    client = AsyncClient(host=host)
    try:
        models_response = await client.list()
    except ResponseError as exc:  # pragma: no cover - network/service integration failure path
        raise RuntimeError(f"Unable to connect to Ollama while checking model '{model_name}': {exc}") from exc

    available_models = {
        model["model"]
        for model in models_response.get("models", [])
        if isinstance(model, dict) and "model" in model
    }
    if model_name not in available_models:
        raise RuntimeError(
            f"Required model '{model_name}' is not available. Pull it with: ollama pull {model_name}"
        )


def _validate_image_paths(image_paths: list[str]) -> list[str]:
    paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Image paths do not exist: {missing}")
    return paths


def _parse_score(raw_response: str) -> int:
    cleaned = raw_response.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "score" in payload:
            return int(payload["score"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback for models that ignore formatting constraints.
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Could not parse importance score from model output: {raw_response!r}")
