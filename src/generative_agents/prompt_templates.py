"""Standardized prompt templates used by the cognitive loop."""

from __future__ import annotations

from typing import Any


PERCEPTION_SYSTEM_PROMPT = (
    "You are an observation module. Extract concrete, grounded details and avoid speculation."
)

POIGNANCY_SYSTEM_PROMPT = (
    "You score memory poignancy from 1-10 based on long-term impact, emotional weight, and urgency. "
    "Return strict JSON only."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a reflective reasoning module. Synthesize recurring themes, contradictions, and opportunities "
    "from recent memories."
)

PLANNING_SYSTEM_PROMPT = (
    "You are a planning module. Produce actionable, time-bounded plans aligned to current goals and constraints."
)

DIALOGUE_SYSTEM_PROMPT = (
    "You are a conversational agent. Stay in-character, context-aware, and concise while preserving facts."
)


def render_perception_prompt(observation_text: str, image_summary: str | None = None) -> str:
    visual = f"\nVisual summary: {image_summary}" if image_summary else ""
    return (
        "Perceive the following input and produce a structured summary with entities, events, and possible intents."
        f"\nInput: {observation_text}{visual}"
    )


def render_poignancy_prompt(memory_text: str) -> str:
    return (
        "Score the memory on importance/poignancy from 1 to 10. "
        "Respond as JSON: {\"score\": <int>, \"reason\": \"...\"}."
        f"\nMemory: {memory_text}"
    )


def render_reflection_prompt(memory_snippets: list[str]) -> str:
    joined = "\n".join(f"- {snippet}" for snippet in memory_snippets)
    return (
        "Reflect on these memories and produce up to 3 compact insights with supporting evidence."
        f"\nMemories:\n{joined}"
    )


def render_planning_prompt(goals: list[str], constraints: list[str], retrieved_context: list[str]) -> str:
    goals_text = "\n".join(f"- {goal}" for goal in goals)
    constraints_text = "\n".join(f"- {constraint}" for constraint in constraints)
    context_text = "\n".join(f"- {item}" for item in retrieved_context)
    return (
        "Create a hierarchical plan (day goals -> hourly blocks -> concrete steps) using the provided context."
        f"\nGoals:\n{goals_text}\nConstraints:\n{constraints_text}\nRetrieved context:\n{context_text}"
    )


def render_dialogue_prompt(agent_state: dict[str, Any], user_message: str, relevant_context: list[str]) -> str:
    context_text = "\n".join(f"- {item}" for item in relevant_context)
    return (
        "Respond to the user while integrating relevant context and preserving consistency with agent state."
        f"\nAgent state: {agent_state}\nContext:\n{context_text}\nUser: {user_message}"
    )
