"""Vector index adapters for memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass
class VectorMatch:
    memory_id: str
    score: float
    metadata: dict[str, Any]
    document: str | None = None


class ChromaVectorStore:
    """Chroma-backed vector storage with memory-id linkage."""

    def __init__(self, persist_directory: str | Path, collection_name: str = "memories") -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        memory_id: str,
        embedding: list[float],
        *,
        agent_id: str,
        memory_type: str,
        document: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {"memory_id": memory_id, "agent_id": agent_id, "memory_type": memory_type}
        if metadata:
            payload.update(metadata)
        self.collection.upsert(ids=[memory_id], embeddings=[embedding], documents=[document], metadatas=[payload])

    def upsert_many(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        ids = [item["memory_id"] for item in items]
        embeddings = [item["embedding"] for item in items]
        docs = [item.get("document", "") for item in items]
        metadatas: list[dict[str, Any]] = []
        for item in items:
            metadata = {
                "memory_id": item["memory_id"],
                "agent_id": item["agent_id"],
                "memory_type": item["memory_type"],
            }
            metadata.update(item.get("metadata", {}))
            metadatas.append(metadata)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        agent_id: str | None = None,
    ) -> list[VectorMatch]:
        where = {"agent_id": agent_id} if agent_id else None
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]

        matches: list[VectorMatch] = []
        for memory_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
            score = 1.0 / (1.0 + float(distance))
            matches.append(
                VectorMatch(
                    memory_id=memory_id,
                    score=score,
                    metadata=metadata or {},
                    document=document,
                )
            )
        return matches

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self.collection.count()
