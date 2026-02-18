from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import Memory, MemoryType
from src.generative_agents.environment import AgentScheduleEntry, AgentState, Location, Position, WorldState
from src.generative_agents.perception import VisualPerceptionService


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_with_vision(self, prompt: str, image_paths: list[str]) -> str:
        self.calls += 1
        return f"scene call={self.calls} paths={len(image_paths)}"


class VisualPerceptionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

        work_tiles = {Position(x=1, y=1), Position(x=2, y=1), Position(x=1, y=2), Position(x=2, y=2)}
        world = WorldState(width=8, height=8)
        world.add_location(Location(location_id="work", name="Workspace", tiles=work_tiles))
        world.add_agent(
            AgentState(
                agent_id="agent-1",
                name="Ada",
                position=Position(2, 2),
                schedule=[
                    AgentScheduleEntry(
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
                        target_location_id="work",
                        activity="coding",
                    )
                ],
            )
        )
        self.world = world

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_capture_and_describe_uses_cache_until_glance(self) -> None:
        llm = FakeLLMClient()
        service = VisualPerceptionService(
            llm_client=llm,
            image_base_dir=self.base / "images",
            glance_interval_ticks=3,
            change_threshold=0.05,
        )

        first = asyncio.run(service.capture_and_describe(self.world, "agent-1", radius=2))
        second = asyncio.run(service.capture_and_describe(self.world, "agent-1", radius=2))
        third = asyncio.run(service.capture_and_describe(self.world, "agent-1", radius=2))

        self.assertTrue(first[0].exists())
        self.assertTrue(second[0].exists())
        self.assertTrue(third[0].exists())
        self.assertEqual(llm.calls, 2)
        self.assertEqual(first[1], second[1])

    def test_update_memory_visual_context(self) -> None:
        llm = FakeLLMClient()
        service = VisualPerceptionService(llm_client=llm, image_base_dir=self.base / "images")
        memory = Memory(
            description="Observed scene",
            importance_score=1.0,
            memory_type=MemoryType.episodic,
            embedding_vector_ref="vec:1",
        )

        updated = asyncio.run(service.update_memory_visual_context(memory, self.world, "agent-1", radius=2))

        self.assertIsNotNone(updated.visual_context)
        assert updated.visual_context is not None
        self.assertTrue(Path(updated.visual_context.image_ref).exists())
        self.assertIn("scene call=", updated.visual_context.scene_description)


if __name__ == "__main__":
    unittest.main()
