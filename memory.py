"""Core memory models for the agent system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    """Supported categories of memory."""

    episodic = "episodic"
    semantic = "semantic"
    procedural = "procedural"
    reflective = "reflective"


class VisualContext(BaseModel):
    """Optional visual metadata tied to a memory."""

    scene_description: str | None = None
    image_ref: str | None = None
    extracted_entities: list[str] = Field(default_factory=list)


class Memory(BaseModel):
    """Canonical memory object used across retrieval and planning."""

    description: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance_score: float = Field(..., ge=0.0)
    memory_type: MemoryType
    embedding_vector_ref: str = Field(..., min_length=1)
    pointers_to_evidence: list[str] = Field(default_factory=list)
    visual_context: VisualContext | None = None

    @field_validator("created_at", "last_accessed")
    @classmethod
    def normalize_dt(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def mark_accessed(self, when: datetime | None = None) -> None:
        """Update last access time in-place."""

        self.last_accessed = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)


class RetrievedMemory(BaseModel):
    """Memory enriched with retrieval component scores."""

    memory: Memory
    recency: float
    importance: float
    relevance: float
    final_score: float


class ReflectionInsight(BaseModel):
    """High-level synthesis created from recent memories."""

    summary: str
    supporting_memories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
