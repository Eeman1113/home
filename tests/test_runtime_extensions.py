from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.generative_agents.dialogue import co_located_agents, run_dialogue
from src.generative_agents.environment.models import AgentState, Position, WorldState
from src.generative_agents.simulation import SimulationScheduler
from src.generative_agents.storage.sqlite_store import SQLiteStore
from src.generative_agents.storage.vector_store import ChromaVectorStore
from src.generative_agents.ui.dashboard import build_interview_questions


class RuntimeExtensionsTests(unittest.TestCase):
    def test_sqlite_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            async def scenario() -> None:
                store = SQLiteStore(Path(tmp) / "state.sqlite3")
                await store.connect()
                memory_id = await store.upsert_memory(
                    {
                        "agent_id": "a1",
                        "description": "Met Bob near the fountain",
                        "importance_score": 8.0,
                        "memory_type": "episodic",
                        "embedding_vector_ref": "vec:1",
                        "pointers_to_evidence": ["image://fountain.png"],
                        "visual_context": {"scene_description": "town square", "image_ref": "fountain.png"},
                    }
                )
                await store.add_reflection("a1", "Need to follow up with Bob", [memory_id])
                await store.upsert_plan("a1", "today", ["Follow up"], [{"hour": "09:00", "task": "message Bob"}])
                convo_id = "c1"
                await store.add_dialogue_turn(convo_id, "a1", "a2", "Hello!", {"summary": "at fountain"})

                memories = await store.get_agent_memories("a1")
                dialogue = await store.get_latest_dialogue_turns(convo_id)
                await store.close()

                self.assertEqual(len(memories), 1)
                self.assertEqual(memories[0]["memory_id"], memory_id)
                self.assertEqual(memories[0]["pointers_to_evidence"], ["image://fountain.png"])
                self.assertEqual(dialogue[0]["utterance"], "Hello!")

            asyncio.run(scenario())

    def test_vector_store_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(Path(tmp) / "chroma")
            store.upsert(
                memory_id="m1",
                embedding=[0.1, 0.2, 0.3],
                agent_id="a1",
                memory_type="episodic",
                document="Discussed lunch plans",
            )
            results = store.query([0.1, 0.2, 0.31], top_k=1, agent_id="a1")
            self.assertEqual(results[0].memory_id, "m1")

    def test_scheduler_runs_concurrently(self) -> None:
        class StubAgent:
            def __init__(self, agent_id: str) -> None:
                self.agent_id = agent_id

            async def tick(self, tick_index: int) -> dict[str, int]:
                return {"tick": tick_index}

        records: list[tuple[str, int]] = []

        async def persist(agent_id: str, tick_index: int, result: dict[str, int]) -> None:
            records.append((agent_id, result["tick"]))

        async def scenario() -> None:
            agents = [StubAgent(f"a{i}") for i in range(5)]
            scheduler = SimulationScheduler(agents, tick_interval_s=0, persist_callback=persist)
            history = await scheduler.run(2)
            self.assertEqual(len(history), 2)
            self.assertEqual(len(records), 10)

        asyncio.run(scenario())

    def test_dialogue_and_interview_prompt(self) -> None:
        class Talker:
            def __init__(self, agent_id: str, name: str) -> None:
                self.agent_id = agent_id
                self.name = name

            async def speak(self, prompt: str, context: dict[str, str]) -> str:
                return f"{self.name} acknowledges {context['summary']}"

        async def scenario() -> None:
            world = WorldState(width=5, height=5)
            a_state = AgentState(agent_id="a1", name="Ada", position=Position(2, 2))
            b_state = AgentState(agent_id="a2", name="Ben", position=Position(2, 2))
            world.add_agent(a_state)
            world.add_agent(b_state)

            pairs = co_located_agents(world)
            self.assertEqual(len(pairs), 1)

            transcript = await run_dialogue(Talker("a1", "Ada"), Talker("a2", "Ben"), a_state, b_state, world, turns=2)
            self.assertEqual(len(transcript), 2)

            prompts = build_interview_questions(
                "a1",
                [
                    {
                        "description": "Saw crowded plaza",
                        "visual_context": {"scene_description": "crowded plaza", "image_ref": "img.png"},
                    }
                ],
            )
            self.assertTrue(any("Visual-memory recall" in prompt for prompt in prompts))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
