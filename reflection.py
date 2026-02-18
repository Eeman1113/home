"""Reflection trigger and synthesis logic."""

from __future__ import annotations

from collections import Counter

from memory import Memory, ReflectionInsight

REFLECTION_TRIGGER_THRESHOLD = 150.0


def should_trigger_reflection(memories: list[Memory]) -> bool:
    """Trigger when cumulative importance is above threshold."""

    return sum(memory.importance_score for memory in memories) > REFLECTION_TRIGGER_THRESHOLD


def generate_high_level_insights(recent_memories: list[Memory], max_insights: int = 3) -> list[ReflectionInsight]:
    """Generate compact thematic insights from recent memories."""

    if not recent_memories:
        return []

    memory_types = Counter(memory.memory_type.value for memory in recent_memories)
    avg_importance = sum(memory.importance_score for memory in recent_memories) / len(recent_memories)

    insights: list[ReflectionInsight] = [
        ReflectionInsight(
            summary=(
                "Recent activity concentrates in "
                f"{', '.join(f'{kind} ({count})' for kind, count in memory_types.most_common())}."
            ),
            supporting_memories=[memory.description for memory in recent_memories[:5]],
            metadata={"average_importance": round(avg_importance, 3)},
        )
    ]

    if avg_importance >= 50:
        insights.append(
            ReflectionInsight(
                summary="Current memory stream indicates high-priority context that may require proactive planning.",
                supporting_memories=[memory.description for memory in sorted(recent_memories, key=lambda m: m.importance_score, reverse=True)[:3]],
                metadata={"signal": "high_importance"},
            )
        )

    if len(recent_memories) >= 4:
        latest = recent_memories[0]
        oldest = recent_memories[-1]
        insights.append(
            ReflectionInsight(
                summary=f"Context appears to evolve from '{oldest.description}' toward '{latest.description}'.",
                supporting_memories=[oldest.description, latest.description],
                metadata={"signal": "temporal_shift"},
            )
        )

    return insights[:max_insights]
