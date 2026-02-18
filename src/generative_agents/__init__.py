"""Generative Agents package."""

from .config import AgentConfig
from .context_budget import fit_context_to_budget
from .simulation import SimulationScheduler

__all__ = [
    "AgentConfig",
    "EmbeddingClient",
    "LLMClient",
    "VisualPerceptionService",
    "SimulationScheduler",
    "fit_context_to_budget",
]


def __getattr__(name: str):
    if name == "EmbeddingClient":
        from .embeddings import EmbeddingClient

        return EmbeddingClient
    if name == "LLMClient":
        from .llm_client import LLMClient

        return LLMClient
    if name == "VisualPerceptionService":
        from .perception import VisualPerceptionService

        return VisualPerceptionService
    raise AttributeError(f"module 'generative_agents' has no attribute {name!r}")
