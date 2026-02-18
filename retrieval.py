"""Memory retrieval and scoring logic."""

from __future__ import annotations

from datetime import datetime, timezone

from memory import Memory, RetrievedMemory


def recency_score(last_accessed: datetime, now: datetime | None = None) -> float:
    """Compute recency using exponential decay by elapsed hours."""

    reference = now or datetime.now(timezone.utc)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    elapsed_hours = max((reference - last_accessed.astimezone(timezone.utc)).total_seconds() / 3600.0, 0.0)
    return 0.995 ** elapsed_hours


def final_score(recency: float, importance: float, relevance: float) -> float:
    """Combine component scores via direct additive rule."""

    return recency + importance + relevance


def score_memory(memory: Memory, relevance: float, now: datetime | None = None) -> RetrievedMemory:
    """Generate all score components for a single memory."""

    recency = recency_score(memory.last_accessed, now=now)
    importance = memory.importance_score
    total = final_score(recency=recency, importance=importance, relevance=relevance)
    return RetrievedMemory(
        memory=memory,
        recency=recency,
        importance=importance,
        relevance=relevance,
        final_score=total,
    )


def retrieve_top_memories(
    memories: list[Memory],
    relevance_by_embedding_ref: dict[str, float],
    top_k: int = 5,
    now: datetime | None = None,
) -> list[RetrievedMemory]:
    """Rank memories by final score and return top-k results."""

    scored = [
        score_memory(memory, relevance_by_embedding_ref.get(memory.embedding_vector_ref, 0.0), now=now)
        for memory in memories
    ]
    return sorted(scored, key=lambda item: item.final_score, reverse=True)[:top_k]
