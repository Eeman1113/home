"""CLI entrypoint for running a local Generative Agents simulation loop."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import AgentConfig, EMBED_MODEL, MODEL_NAME

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface parser."""
    parser = argparse.ArgumentParser(
        description="Run a local generative agents runtime backed by Ollama and vector storage.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Primary generation model served by Ollama.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="Embedding model for memory retrieval.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=3,
        help="Number of concurrently simulated agents.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=1.0,
        help="Seconds between simulation ticks.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path("./data"),
        help="Base directory for runtime storage.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("./data/state.sqlite3"),
        help="SQLite file used for structured state and logs.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("./data/chroma"),
        help="Chroma persistent collection directory for memories.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=10,
        help="Number of ticks to run before exiting.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.agent_count <= 0:
        raise ValueError("--agent-count must be greater than 0")
    if args.tick_interval <= 0:
        raise ValueError("--tick-interval must be greater than 0")
    if args.ticks <= 0:
        raise ValueError("--ticks must be greater than 0")


def ensure_paths(args: argparse.Namespace) -> None:
    args.storage_path.mkdir(parents=True, exist_ok=True)
    args.chroma_path.mkdir(parents=True, exist_ok=True)
    args.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def show_runtime_summary(args: argparse.Namespace, config: AgentConfig) -> None:
    table = Table(title="Generative Agents Runtime")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", args.model_name)
    table.add_row("Embedding model", args.embed_model)
    table.add_row("Agent count", str(args.agent_count))
    table.add_row("Tick interval", f"{args.tick_interval:.2f}s")
    table.add_row("Ticks", str(args.ticks))
    table.add_row("Storage path", str(args.storage_path.resolve()))
    table.add_row("SQLite path", str(args.sqlite_path.resolve()))
    table.add_row("Chroma path", str(args.chroma_path.resolve()))
    table.add_row("Retrieval decay", f"{config.retrieval_decay_factor:.4f}")
    table.add_row("Max context tokens", str(config.max_context_tokens))
    table.add_row("Max generation tokens", str(config.max_generation_tokens))
    console.print(table)


async def run_ticks(args: argparse.Namespace) -> None:
    for tick in range(1, args.ticks + 1):
        console.log(f"Tick {tick}/{args.ticks}: simulating {args.agent_count} agents")
        await asyncio.sleep(args.tick_interval)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    ensure_paths(args)

    config = AgentConfig(
        model_name=args.model_name,
        embed_model=args.embed_model,
    )
    show_runtime_summary(args, config)
    asyncio.run(run_ticks(args))


if __name__ == "__main__":
    main()
