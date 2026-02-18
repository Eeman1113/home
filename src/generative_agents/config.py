"""Runtime configuration defaults for the Generative Agents project."""

from dataclasses import dataclass

MODEL_NAME = "qwen3-vl:latest"
EMBED_MODEL = "nomic-embed-text"
RETRIEVAL_DECAY_FACTOR = 0.985
MAX_CONTEXT_TOKENS = 16000
MAX_GENERATION_TOKENS = 1024


@dataclass(frozen=True)
class AgentConfig:
    """Configuration used by the runtime loop."""

    model_name: str = MODEL_NAME
    embed_model: str = EMBED_MODEL
    retrieval_decay_factor: float = RETRIEVAL_DECAY_FACTOR
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    max_generation_tokens: int = MAX_GENERATION_TOKENS
