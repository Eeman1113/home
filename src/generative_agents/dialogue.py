"""Agent-to-agent dialogue orchestration with shared visual context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from generative_agents.environment.models import AgentState, WorldState


class DialogueAgent(Protocol):
    agent_id: str
    name: str

    async def speak(self, prompt: str, context: dict[str, Any]) -> str:
        ...


@dataclass
class DialogueTurn:
    conversation_id: str
    speaker_id: str
    listener_id: str
    utterance: str
    shared_visual_context: dict[str, Any]


def co_located_agents(world_state: WorldState) -> list[tuple[AgentState, AgentState]]:
    """Return unique agent pairs occupying the same tile."""

    agents = list(world_state.agents.values())
    pairs: list[tuple[AgentState, AgentState]] = []
    for index, first in enumerate(agents):
        for second in agents[index + 1 :]:
            if first.position == second.position:
                pairs.append((first, second))
    return pairs


def shared_visual_context(
    world_state: WorldState,
    first: AgentState,
    second: AgentState,
    *,
    radius: int = 2,
) -> dict[str, Any]:
    min_x = min(first.position.x, second.position.x) - radius
    min_y = min(first.position.y, second.position.y) - radius
    max_x = max(first.position.x, second.position.x) + radius
    max_y = max(first.position.y, second.position.y) + radius

    visible_objects = world_state.objects_in_bounds(min_x, min_y, max_x, max_y)
    visible_agents = world_state.agents_in_bounds(min_x, min_y, max_x, max_y)
    location_names = [location.name for location in world_state.locations.values() if location.contains(first.position)]

    return {
        "tile": {"x": first.position.x, "y": first.position.y},
        "location_names": location_names,
        "visible_objects": [obj.name for obj in visible_objects],
        "visible_agents": [agent.name for agent in visible_agents],
        "summary": (
            f"{first.name} and {second.name} are co-located at ({first.position.x}, {first.position.y}) "
            f"with {len(visible_objects)} nearby objects and {len(visible_agents)} nearby agents."
        ),
    }


async def run_dialogue(
    first_agent: DialogueAgent,
    second_agent: DialogueAgent,
    first_state: AgentState,
    second_state: AgentState,
    world_state: WorldState,
    *,
    turns: int = 4,
    conversation_id: str | None = None,
    store: Any | None = None,
) -> list[DialogueTurn]:
    """Generate alternating dialogue turns and persist them when a store is provided."""

    if first_state.position != second_state.position:
        return []

    convo_id = conversation_id or str(uuid.uuid4())
    context = shared_visual_context(world_state, first_state, second_state)
    transcript: list[DialogueTurn] = []

    speaker: DialogueAgent = first_agent
    listener: DialogueAgent = second_agent

    for turn_index in range(turns):
        prompt = (
            f"Turn {turn_index + 1}: discuss immediate goals with {listener.name}. "
            f"Use shared context: {context['summary']}"
        )
        utterance = await speaker.speak(prompt=prompt, context=context)
        turn = DialogueTurn(
            conversation_id=convo_id,
            speaker_id=speaker.agent_id,
            listener_id=listener.agent_id,
            utterance=utterance,
            shared_visual_context=context,
        )
        transcript.append(turn)

        if store is not None and hasattr(store, "add_dialogue_turn"):
            await store.add_dialogue_turn(
                conversation_id=convo_id,
                speaker_id=turn.speaker_id,
                listener_id=turn.listener_id,
                utterance=turn.utterance,
                shared_visual_context=context,
            )

        speaker, listener = listener, speaker

    return transcript
