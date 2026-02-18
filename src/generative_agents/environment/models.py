"""2D environment model for agents, locations, objects, and schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Position:
    """Tile position in the 2D world grid."""

    x: int
    y: int


@dataclass
class WorldObject:
    """A world object that can occupy a location in the grid."""

    object_id: str
    name: str
    position: Position
    symbol: str = "O"
    color: tuple[int, int, int] = (200, 200, 200)


@dataclass
class Location:
    """Named map area represented by one or more tiles."""

    location_id: str
    name: str
    tiles: set[Position] = field(default_factory=set)
    color: tuple[int, int, int] = (235, 235, 235)

    def contains(self, position: Position) -> bool:
        return position in self.tiles


@dataclass
class AgentScheduleEntry:
    """Scheduled location intent for a time window."""

    start_time: datetime
    end_time: datetime
    target_location_id: str
    activity: str


@dataclass
class AgentState:
    """Current state of an agent within the world."""

    agent_id: str
    name: str
    position: Position
    color: tuple[int, int, int] = (100, 149, 237)
    schedule: list[AgentScheduleEntry] = field(default_factory=list)


@dataclass
class WorldState:
    """Container for all simulation entities in the 2D world."""

    width: int
    height: int
    locations: dict[str, Location] = field(default_factory=dict)
    objects: dict[str, WorldObject] = field(default_factory=dict)
    agents: dict[str, AgentState] = field(default_factory=dict)

    def add_location(self, location: Location) -> None:
        self.locations[location.location_id] = location

    def add_object(self, world_object: WorldObject) -> None:
        self.objects[world_object.object_id] = world_object

    def add_agent(self, agent: AgentState) -> None:
        self.agents[agent.agent_id] = agent

    def move_agent(self, agent_id: str, position: Position) -> None:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        self.agents[agent_id].position = position

    def objects_in_bounds(self, min_x: int, min_y: int, max_x: int, max_y: int) -> list[WorldObject]:
        return [
            obj
            for obj in self.objects.values()
            if min_x <= obj.position.x <= max_x and min_y <= obj.position.y <= max_y
        ]

    def agents_in_bounds(self, min_x: int, min_y: int, max_x: int, max_y: int) -> list[AgentState]:
        return [
            agent
            for agent in self.agents.values()
            if min_x <= agent.position.x <= max_x and min_y <= agent.position.y <= max_y
        ]
