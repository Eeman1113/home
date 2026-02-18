"""Utilities to keep retrieval context within a target token budget."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_CONTEXT_TOKEN_BUDGET = 16_000
CHARS_PER_TOKEN_APPROX = 4


def approximate_token_count(text: str) -> int:
    """Approximate token count with a conservative chars/token heuristic."""

    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN_APPROX - 1) // CHARS_PER_TOKEN_APPROX)


def fit_context_to_budget(context_items: Sequence[str], token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET) -> list[str]:
    """Greedily keep ordered context items that fit under token budget."""

    if token_budget <= 0:
        return []

    kept: list[str] = []
    spent_tokens = 0
    for item in context_items:
        item_tokens = approximate_token_count(item)
        if spent_tokens + item_tokens > token_budget:
            break
        kept.append(item)
        spent_tokens += item_tokens
    return kept


def context_token_usage(context_items: Sequence[str]) -> int:
    """Compute approximate token usage for supplied context."""

    return sum(approximate_token_count(item) for item in context_items)
