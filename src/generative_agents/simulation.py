"""Async multi-agent simulation scheduler."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol


class TickAgent(Protocol):
    agent_id: str

    async def tick(self, tick_index: int) -> dict[str, Any]:
        ...


PersistCallback = Callable[[str, int, dict[str, Any]], Awaitable[None]]


@dataclass
class SimulationSnapshot:
    tick_index: int
    timestamp: datetime
    agent_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


class SimulationScheduler:
    """Tick-driven scheduler for 5-25 concurrent agents with safe persistence."""

    def __init__(
        self,
        agents: list[TickAgent],
        *,
        tick_interval_s: float = 0.5,
        max_concurrency: int = 10,
        persist_callback: PersistCallback | None = None,
    ) -> None:
        if not 5 <= len(agents) <= 25:
            raise ValueError("SimulationScheduler requires between 5 and 25 agents.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive.")

        self.agents = {agent.agent_id: agent for agent in agents}
        self.tick_interval_s = tick_interval_s
        self.max_concurrency = min(max_concurrency, len(agents))
        self.persist_callback = persist_callback

        self._running = False
        self._write_lock = asyncio.Lock()
        self._latest_snapshot: SimulationSnapshot | None = None

    @property
    def latest_snapshot(self) -> SimulationSnapshot | None:
        return self._latest_snapshot

    async def run(self, total_ticks: int) -> list[SimulationSnapshot]:
        if total_ticks <= 0:
            return []

        self._running = True
        history: list[SimulationSnapshot] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        for tick_index in range(total_ticks):
            if not self._running:
                break

            tasks = [
                asyncio.create_task(self._run_agent_tick(agent_id, agent, tick_index, semaphore))
                for agent_id, agent in self.agents.items()
            ]
            tick_results = await asyncio.gather(*tasks)
            snapshot = SimulationSnapshot(
                tick_index=tick_index,
                timestamp=datetime.now(timezone.utc),
                agent_outputs={agent_id: result for agent_id, result in tick_results},
            )
            history.append(snapshot)
            self._latest_snapshot = snapshot

            if tick_index < total_ticks - 1 and self.tick_interval_s > 0:
                await asyncio.sleep(self.tick_interval_s)

        self._running = False
        return history

    def stop(self) -> None:
        self._running = False

    async def _run_agent_tick(
        self,
        agent_id: str,
        agent: TickAgent,
        tick_index: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            result = await agent.tick(tick_index)
            if self.persist_callback:
                async with self._write_lock:
                    await self.persist_callback(agent_id, tick_index, result)
            return agent_id, result
