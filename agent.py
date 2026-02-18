"""Agent orchestration loop connecting memory, reflection, planning, and action selection."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from memory import Memory, ReflectionInsight, RetrievedMemory
from planning import DailyAgenda, generate_hierarchical_plan
from reflection import generate_high_level_insights, should_trigger_reflection
from retrieval import retrieve_top_memories


class Agent:
    """Simple async cognitive loop."""

    def __init__(self, memories: list[Memory] | None = None) -> None:
        self.memories: list[Memory] = memories or []
        self.insights: list[ReflectionInsight] = []
        self.current_plan: DailyAgenda | None = None

    def retrieve(self, relevance_by_embedding_ref: dict[str, float], top_k: int = 5) -> list[RetrievedMemory]:
        return retrieve_top_memories(self.memories, relevance_by_embedding_ref, top_k=top_k)

    def reflect(self, recent_memories: list[Memory]) -> list[ReflectionInsight]:
        if not should_trigger_reflection(recent_memories):
            return []
        self.insights = generate_high_level_insights(recent_memories)
        return self.insights

    def plan(self, retrieved_memories: list[RetrievedMemory]) -> DailyAgenda:
        self.current_plan = generate_hierarchical_plan(retrieved_memories, self.insights)
        return self.current_plan

    def select_action(self) -> str:
        """Select next 5-15 minute action from the current plan."""

        if not self.current_plan:
            return "No plan available."

        for hour in self.current_plan.hourly_plan:
            if hour.actions:
                return hour.actions[0].title

        return "No action available."

    async def tick(self, relevance_by_embedding_ref: dict[str, float], sleep_s: float = 0.0) -> dict[str, object]:
        """Single async loop iteration orchestrating retrieval -> reflection -> planning -> action."""

        now = datetime.now(timezone.utc)
        retrieved = self.retrieve(relevance_by_embedding_ref)
        recent_memories = [item.memory for item in retrieved]

        for memory in recent_memories:
            memory.mark_accessed(now)

        generated_insights = self.reflect(recent_memories)
        plan = self.plan(retrieved)
        action = self.select_action()

        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        return {
            "retrieved": retrieved,
            "insights": generated_insights,
            "plan": plan,
            "action": action,
            "timestamp": now,
        }
