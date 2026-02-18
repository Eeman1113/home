"""Embedding client with batching and retry support for Ollama."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ollama import AsyncClient, ResponseError

from .config import EMBED_MODEL

DEFAULT_BATCH_SIZE = 32


class EmbeddingClient:
    """Async embedding client backed by Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        host: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        timeout_s: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self._client = AsyncClient(host=host)
        self.batch_size = max(batch_size, 1)
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_s = max(retry_backoff_s, 0.0)
        self.timeout_s = timeout_s

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts by batching requests and preserving input order."""

        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            batch_vectors = await self._embed_with_retry(batch)
            vectors.extend(batch_vectors)
        return vectors

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

        vectors = await self.embed_texts([text])
        return vectors[0]

    async def _embed_with_retry(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                response = await asyncio.wait_for(
                    self._client.embed(model=self.model_name, input=batch),
                    timeout=self.timeout_s,
                )
                embeddings = response.get("embeddings", [])
                if not embeddings:
                    raise RuntimeError("Ollama returned no embeddings for batch request")
                return embeddings
            except (ResponseError, TimeoutError, RuntimeError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding request failed after {attempt + 1} attempts for model '{self.model_name}'"
                    ) from exc
                sleep_s = self.retry_backoff_s * (2**attempt)
                await asyncio.sleep(sleep_s)
                attempt += 1


def resolve_embedding_model(preferred_model: str | None = None) -> str:
    """Resolve default embedding model.

    qwen3-vl embeddings are not used by default because support depends on model build/options.
    """

    if preferred_model:
        return preferred_model
    return EMBED_MODEL
