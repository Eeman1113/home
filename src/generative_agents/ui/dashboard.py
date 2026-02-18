"""Rich TUI helpers for simulation monitoring and interview mode."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def build_dashboard_renderable(snapshot: dict[str, Any], latest_memories: dict[str, list[dict[str, Any]]]) -> Group:
    clock = snapshot.get("clock") or datetime.utcnow().isoformat()
    tick = snapshot.get("tick", 0)
    states: dict[str, dict[str, Any]] = snapshot.get("agent_states", {})

    state_table = Table(title="Agent States")
    state_table.add_column("Agent")
    state_table.add_column("Status")
    state_table.add_column("Current Action")

    for agent_id, state in states.items():
        state_table.add_row(agent_id, str(state.get("status", "idle")), str(state.get("action", "-")))

    memory_table = Table(title="Latest Memories")
    memory_table.add_column("Agent")
    memory_table.add_column("Memory")
    memory_table.add_column("Importance")
    for agent_id, memories in latest_memories.items():
        if not memories:
            memory_table.add_row(agent_id, "-", "-")
            continue
        latest = memories[0]
        memory_table.add_row(
            agent_id,
            str(latest.get("description", ""))[:80],
            str(latest.get("importance_score", "?")),
        )

    top_panel = Panel.fit(f"Simulation clock: [bold]{clock}[/bold]\nTick: [bold]{tick}[/bold]", title="Runtime")
    return Group(top_panel, state_table, memory_table)


async def run_dashboard(
    snapshot_stream: Any,
    memory_provider: Any,
    *,
    refresh_per_second: float = 4.0,
) -> None:
    """Render a live dashboard from async snapshot + memory sources."""

    async with Live(refresh_per_second=refresh_per_second, screen=False) as live:
        async for snapshot in snapshot_stream:
            agent_ids = list(snapshot.get("agent_states", {}).keys())
            latest_memories = await memory_provider(agent_ids)
            renderable = build_dashboard_renderable(snapshot, latest_memories)
            live.update(renderable)


def build_interview_questions(agent_id: str, memories: list[dict[str, Any]]) -> list[str]:
    """Generate interview prompts from an agent's memory stream, including visual recall."""

    prompts: list[str] = [f"Agent {agent_id}: what is your current top priority?"]

    for memory in memories[:5]:
        prompts.append(f"How does memory '{memory.get('description', '')}' influence your next action?")
        visual = memory.get("visual_context") or {}
        if visual:
            scene_desc = visual.get("scene_description") or "the observed scene"
            image_ref = visual.get("image_ref") or "(no image reference)"
            prompts.append(
                "Visual-memory recall: what detail from "
                f"'{scene_desc}' (image: {image_ref}) is most decision-relevant right now?"
            )

    return prompts
