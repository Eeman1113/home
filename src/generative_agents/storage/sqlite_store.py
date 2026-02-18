"""Async SQLite persistence for agent memories and runtime artifacts."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


class SQLiteStore:
    """Persistence facade for simulation artifacts."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.sqlite_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.commit()
        await self.init_schema()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def init_schema(self) -> None:
        """Create normalized schema for memory + dialogue persistence."""

        conn = self._require_conn()
        async with self._lock:
            await conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    memory_type TEXT NOT NULL,
                    embedding_vector_ref TEXT NOT NULL,
                    visual_context_json TEXT
                );

                CREATE TABLE IF NOT EXISTS evidence_pointers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    pointer TEXT NOT NULL,
                    source_type TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    supporting_memory_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    date_label TEXT NOT NULL,
                    goals_json TEXT NOT NULL,
                    hourly_plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dialogue_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    listener_id TEXT NOT NULL,
                    utterance TEXT NOT NULL,
                    shared_visual_context_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_agent_time ON memories(agent_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_dialogue_conversation_time ON dialogue_turns(conversation_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_reflections_agent_time ON reflections(agent_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_plans_agent_time ON plans(agent_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_evidence_memory ON evidence_pointers(memory_id);
                """
            )
            await conn.commit()

    async def upsert_memory(self, memory: dict[str, Any]) -> str:
        conn = self._require_conn()
        memory_id = memory.get("memory_id") or str(uuid.uuid4())
        now = _utc_now_iso()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO memories(
                    memory_id, agent_id, description, created_at, last_accessed,
                    importance_score, memory_type, embedding_vector_ref, visual_context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    description = excluded.description,
                    last_accessed = excluded.last_accessed,
                    importance_score = excluded.importance_score,
                    memory_type = excluded.memory_type,
                    embedding_vector_ref = excluded.embedding_vector_ref,
                    visual_context_json = excluded.visual_context_json
                """,
                (
                    memory_id,
                    memory["agent_id"],
                    memory["description"],
                    memory.get("created_at", now),
                    memory.get("last_accessed", now),
                    float(memory["importance_score"]),
                    memory["memory_type"],
                    memory["embedding_vector_ref"],
                    _json_dumps(memory.get("visual_context")),
                ),
            )
            for pointer in memory.get("pointers_to_evidence", []):
                await self.add_evidence_pointer(
                    memory_id=memory_id,
                    pointer=pointer,
                    source_type="memory",
                    metadata=None,
                    conn=conn,
                )
            await conn.commit()
        return memory_id

    async def add_evidence_pointer(
        self,
        memory_id: str,
        pointer: str,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        active_conn = conn or self._require_conn()
        await active_conn.execute(
            """
            INSERT INTO evidence_pointers(memory_id, pointer, source_type, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, pointer, source_type, _json_dumps(metadata), _utc_now_iso()),
        )

    async def add_reflection(
        self,
        agent_id: str,
        summary: str,
        supporting_memory_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        conn = self._require_conn()
        reflection_id = str(uuid.uuid4())
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO reflections(
                    reflection_id, agent_id, summary, supporting_memory_ids_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reflection_id,
                    agent_id,
                    summary,
                    _json_dumps(supporting_memory_ids),
                    _json_dumps(metadata or {}),
                    _utc_now_iso(),
                ),
            )
            await conn.commit()
        return reflection_id

    async def upsert_plan(
        self,
        agent_id: str,
        date_label: str,
        goals: list[str],
        hourly_plan: list[dict[str, Any]],
        plan_id: str | None = None,
    ) -> str:
        conn = self._require_conn()
        assigned_id = plan_id or str(uuid.uuid4())
        now = _utc_now_iso()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO plans(plan_id, agent_id, date_label, goals_json, hourly_plan_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    goals_json = excluded.goals_json,
                    hourly_plan_json = excluded.hourly_plan_json,
                    updated_at = excluded.updated_at
                """,
                (assigned_id, agent_id, date_label, _json_dumps(goals), _json_dumps(hourly_plan), now, now),
            )
            await conn.commit()
        return assigned_id

    async def add_dialogue_turn(
        self,
        conversation_id: str,
        speaker_id: str,
        listener_id: str,
        utterance: str,
        shared_visual_context: dict[str, Any] | None = None,
    ) -> str:
        conn = self._require_conn()
        turn_id = str(uuid.uuid4())
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO dialogue_turns(
                    turn_id, conversation_id, speaker_id, listener_id, utterance,
                    shared_visual_context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    conversation_id,
                    speaker_id,
                    listener_id,
                    utterance,
                    _json_dumps(shared_visual_context),
                    _utc_now_iso(),
                ),
            )
            await conn.commit()
        return turn_id

    async def get_agent_memories(self, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT memory_id, agent_id, description, created_at, last_accessed,
                   importance_score, memory_type, embedding_vector_ref, visual_context_json
            FROM memories
            WHERE agent_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_id, limit),
        )
        rows = await cursor.fetchall()
        memories: list[dict[str, Any]] = []
        for row in rows:
            pointers = await self.get_evidence_pointers(row["memory_id"])
            memories.append(
                {
                    "memory_id": row["memory_id"],
                    "agent_id": row["agent_id"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"],
                    "importance_score": row["importance_score"],
                    "memory_type": row["memory_type"],
                    "embedding_vector_ref": row["embedding_vector_ref"],
                    "visual_context": _json_loads(row["visual_context_json"]),
                    "pointers_to_evidence": pointers,
                }
            )
        return memories

    async def get_evidence_pointers(self, memory_id: str) -> list[str]:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT pointer FROM evidence_pointers WHERE memory_id = ? ORDER BY id ASC",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [row["pointer"] for row in rows]

    async def get_latest_dialogue_turns(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT turn_id, conversation_id, speaker_id, listener_id, utterance, shared_visual_context_json, created_at
            FROM dialogue_turns
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "turn_id": row["turn_id"],
                "conversation_id": row["conversation_id"],
                "speaker_id": row["speaker_id"],
                "listener_id": row["listener_id"],
                "utterance": row["utterance"],
                "shared_visual_context": _json_loads(row["shared_visual_context_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteStore is not connected. Call connect() first.")
        return self._conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)
