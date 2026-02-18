"""Hierarchical planning utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field

from memory import ReflectionInsight, RetrievedMemory


class ActionStep(BaseModel):
    """Concrete short action (5-15 minutes)."""

    title: str
    duration_minutes: int = Field(..., ge=5, le=15)


class HourlyPlan(BaseModel):
    """One hour block containing several action steps."""

    hour_label: str
    objective: str
    actions: list[ActionStep] = Field(default_factory=list)


class DailyAgenda(BaseModel):
    """Top-level day plan composed of hourly blocks."""

    date_label: str
    goals: list[str] = Field(default_factory=list)
    hourly_plan: list[HourlyPlan] = Field(default_factory=list)


def generate_hierarchical_plan(
    retrieved_memories: list[RetrievedMemory],
    insights: list[ReflectionInsight],
    date_label: str = "today",
) -> DailyAgenda:
    """Build daily agenda -> hourly plan -> 5-15 minute actions."""

    goals = [insight.summary for insight in insights[:3]]
    if not goals:
        goals = ["Maintain progress on current priorities."]

    hourly_blocks: list[HourlyPlan] = []
    for index, memory in enumerate(retrieved_memories[:3], start=1):
        objective = f"Advance: {memory.memory.description}"
        actions = [
            ActionStep(title=f"Review context for {memory.memory.memory_type.value} memory", duration_minutes=10),
            ActionStep(title=f"Execute next step tied to '{memory.memory.description}'", duration_minutes=15),
            ActionStep(title="Capture concise outcome note", duration_minutes=5),
        ]
        hourly_blocks.append(HourlyPlan(hour_label=f"Hour {index}", objective=objective, actions=actions))

    if not hourly_blocks:
        hourly_blocks.append(
            HourlyPlan(
                hour_label="Hour 1",
                objective="Stabilize baseline operations",
                actions=[
                    ActionStep(title="Review pending tasks", duration_minutes=10),
                    ActionStep(title="Complete a highest-impact task", duration_minutes=15),
                    ActionStep(title="Log lessons learned", duration_minutes=5),
                ],
            )
        )

    return DailyAgenda(date_label=date_label, goals=goals, hourly_plan=hourly_blocks)
