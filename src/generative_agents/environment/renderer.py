"""Pillow-based world renderer and viewport capture APIs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .models import AgentState, Location, WorldState


class PillowWorldRenderer:
    """Render full world maps or localized agent-centric viewports."""

    def __init__(self, tile_size: int = 32, background_color: tuple[int, int, int] = (245, 245, 245)) -> None:
        self.tile_size = tile_size
        self.background_color = background_color

    def render_world(self, world: WorldState) -> Image.Image:
        """Render the complete world map."""

        image = Image.new(
            "RGB",
            (world.width * self.tile_size, world.height * self.tile_size),
            color=self.background_color,
        )
        draw = ImageDraw.Draw(image)
        self._draw_locations(draw, world.locations.values())
        self._draw_grid(draw, world.width, world.height)
        self._draw_objects(draw, world)
        self._draw_agents(draw, world)
        return image

    def capture_viewport(
        self,
        world: WorldState,
        agent_id: str,
        radius: int,
        output_path: str | Path,
    ) -> Path:
        """Capture a square viewport around the specified agent and save it to disk."""

        if radius <= 0:
            raise ValueError("radius must be > 0")
        if agent_id not in world.agents:
            raise KeyError(f"Unknown agent_id: {agent_id}")

        agent = world.agents[agent_id]
        min_x = max(0, agent.position.x - radius)
        min_y = max(0, agent.position.y - radius)
        max_x = min(world.width - 1, agent.position.x + radius)
        max_y = min(world.height - 1, agent.position.y + radius)

        viewport_width = max_x - min_x + 1
        viewport_height = max_y - min_y + 1

        image = Image.new(
            "RGB",
            (viewport_width * self.tile_size, viewport_height * self.tile_size),
            color=self.background_color,
        )
        draw = ImageDraw.Draw(image)

        self._draw_locations(draw, self._viewport_locations(world.locations.values(), min_x, min_y, max_x, max_y), min_x, min_y)
        self._draw_grid(draw, viewport_width, viewport_height)
        self._draw_objects(draw, world, min_x, min_y, max_x, max_y)
        self._draw_agents(draw, world, min_x, min_y, max_x, max_y)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path

    def _draw_grid(self, draw: ImageDraw.ImageDraw, width_tiles: int, height_tiles: int) -> None:
        line_color = (215, 215, 215)
        for x in range(width_tiles + 1):
            draw.line(
                [(x * self.tile_size, 0), (x * self.tile_size, height_tiles * self.tile_size)],
                fill=line_color,
                width=1,
            )
        for y in range(height_tiles + 1):
            draw.line(
                [(0, y * self.tile_size), (width_tiles * self.tile_size, y * self.tile_size)],
                fill=line_color,
                width=1,
            )

    def _draw_locations(
        self,
        draw: ImageDraw.ImageDraw,
        locations: list[Location] | tuple[Location, ...] | object,
        min_x: int = 0,
        min_y: int = 0,
    ) -> None:
        for location in locations:
            for tile in location.tiles:
                x = (tile.x - min_x) * self.tile_size
                y = (tile.y - min_y) * self.tile_size
                draw.rectangle(
                    [(x, y), (x + self.tile_size, y + self.tile_size)],
                    fill=location.color,
                )

    def _draw_objects(self, draw: ImageDraw.ImageDraw, world: WorldState, min_x: int = 0, min_y: int = 0, max_x: int | None = None, max_y: int | None = None) -> None:
        objects = (
            world.objects_in_bounds(min_x, min_y, max_x, max_y)
            if max_x is not None and max_y is not None
            else world.objects.values()
        )
        for obj in objects:
            x = (obj.position.x - min_x) * self.tile_size
            y = (obj.position.y - min_y) * self.tile_size
            pad = max(self.tile_size // 5, 2)
            draw.ellipse(
                [(x + pad, y + pad), (x + self.tile_size - pad, y + self.tile_size - pad)],
                fill=obj.color,
            )

    def _draw_agents(
        self,
        draw: ImageDraw.ImageDraw,
        world: WorldState,
        min_x: int = 0,
        min_y: int = 0,
        max_x: int | None = None,
        max_y: int | None = None,
    ) -> None:
        agents = (
            world.agents_in_bounds(min_x, min_y, max_x, max_y)
            if max_x is not None and max_y is not None
            else world.agents.values()
        )
        for agent in agents:
            self._draw_agent(draw, agent, min_x, min_y)

    def _draw_agent(self, draw: ImageDraw.ImageDraw, agent: AgentState, min_x: int, min_y: int) -> None:
        x = (agent.position.x - min_x) * self.tile_size
        y = (agent.position.y - min_y) * self.tile_size
        pad = max(self.tile_size // 6, 2)
        draw.rectangle(
            [(x + pad, y + pad), (x + self.tile_size - pad, y + self.tile_size - pad)],
            fill=agent.color,
            outline=(30, 30, 30),
            width=2,
        )

    def _viewport_locations(
        self,
        locations: object,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> list[Location]:
        visible_locations: list[Location] = []
        for location in locations:
            visible_tiles = {
                tile for tile in location.tiles if min_x <= tile.x <= max_x and min_y <= tile.y <= max_y
            }
            if visible_tiles:
                visible_locations.append(
                    Location(
                        location_id=location.location_id,
                        name=location.name,
                        tiles=visible_tiles,
                        color=location.color,
                    )
                )
        return visible_locations
