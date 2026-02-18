"""2D world environment models and rendering utilities."""

from .models import AgentScheduleEntry, AgentState, Location, Position, WorldObject, WorldState

__all__ = [
    "Position",
    "WorldObject",
    "Location",
    "AgentScheduleEntry",
    "AgentState",
    "WorldState",
    "PillowWorldRenderer",
]


def __getattr__(name: str):
    if name == "PillowWorldRenderer":
        from .renderer import PillowWorldRenderer

        return PillowWorldRenderer
    raise AttributeError(f"module 'generative_agents.environment' has no attribute {name!r}")
