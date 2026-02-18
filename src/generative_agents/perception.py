"""Visual perception pipeline backed by viewport capture and qwen3-vl descriptions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from memory import Memory, VisualContext

from .environment.models import WorldState
from .environment.renderer import PillowWorldRenderer
from .llm_client import LLMClient


@dataclass
class VisionCacheState:
    """Per-agent cache record used for change detection and glance scheduling."""

    image_hash: str | None = None
    last_description: str | None = None
    ticks_since_inference: int = 0


class VisualPerceptionService:
    """Capture viewports and produce textual scene descriptions with caching."""

    def __init__(
        self,
        llm_client: LLMClient,
        renderer: PillowWorldRenderer | None = None,
        image_base_dir: str | Path = Path("data/images"),
        glance_interval_ticks: int = 5,
        change_threshold: float = 0.10,
    ) -> None:
        self.llm_client = llm_client
        self.renderer = renderer or PillowWorldRenderer()
        self.image_base_dir = Path(image_base_dir)
        self.glance_interval_ticks = glance_interval_ticks
        self.change_threshold = change_threshold
        self._cache: dict[str, VisionCacheState] = {}

    async def capture_and_describe(
        self,
        world: WorldState,
        agent_id: str,
        radius: int,
    ) -> tuple[Path, str]:
        """Capture viewport image and return path + scene description."""

        image_path = self.capture_viewport(world, agent_id, radius)
        cache_state = self._cache.setdefault(agent_id, VisionCacheState())

        current_hash = self._sha256(image_path)
        similarity = _hash_similarity(cache_state.image_hash, current_hash)
        cache_state.ticks_since_inference += 1

        should_run_inference = (
            cache_state.last_description is None
            or similarity < (1.0 - self.change_threshold)
            or cache_state.ticks_since_inference >= self.glance_interval_ticks
        )

        if should_run_inference:
            prompt = (
                "Describe this scene for an autonomous social simulation agent. "
                "Include salient entities, likely activities, and changes from routine context "
                "in concise plain text."
            )
            description = await self.llm_client.generate_with_vision(prompt=prompt, image_paths=[str(image_path)])
            cache_state.last_description = description
            cache_state.image_hash = current_hash
            cache_state.ticks_since_inference = 0
        else:
            description = cache_state.last_description or ""

        return image_path, description

    def capture_viewport(self, world: WorldState, agent_id: str, radius: int) -> Path:
        """Capture and store viewport under data/images/{agent_id}/{timestamp}.png."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_path = self.image_base_dir / agent_id / f"{timestamp}.png"
        return self.renderer.capture_viewport(world=world, agent_id=agent_id, radius=radius, output_path=out_path)

    async def update_memory_visual_context(
        self,
        memory: Memory,
        world: WorldState,
        agent_id: str,
        radius: int,
    ) -> Memory:
        """Populate memory.visual_context with the captured image path and description."""

        image_path, description = await self.capture_and_describe(world=world, agent_id=agent_id, radius=radius)
        memory.visual_context = VisualContext(
            scene_description=description,
            image_ref=str(image_path),
        )
        return memory

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as infile:
            digest.update(infile.read())
        return digest.hexdigest()


def _hash_similarity(previous_hash: str | None, current_hash: str) -> float:
    """Rough similarity score using hex-string hamming ratio."""

    if previous_hash is None:
        return 0.0
    if len(previous_hash) != len(current_hash):
        return 0.0

    equal = sum(1 for a, b in zip(previous_hash, current_hash) if a == b)
    return equal / len(current_hash)
