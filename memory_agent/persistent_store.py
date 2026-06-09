import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryItem:
    key: str
    value: dict[str, Any]
    score: float | None = None

class SQLiteVectorMemoryStore:
    """SQLite + local embedding semantic memory store."""

    def __init__(
        self,
        db_path: str = "./memory.db",
        embedding_model: str = "BAAI/bge-m3"
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.encoder = SentenceTransformer(embedding_model, device="cpu")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
             
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_namespace
                ON memories(namespace)
                """
            )
         
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category)
                """
            )

    @staticmethod
    def _namespace_to_str(namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)
    
    def _embed(self, text: str) -> list[float]:
        vector = self.encoder.encode(
            text,
            normalize_embeddings=True
        )

        return vector.astype(float).tolist()
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)

        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0

        return float(np.dot(va, vb) / denom)
    
    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        **_: Any,
    ) -> None:
        namespace_str = self._namespace_to_str(namespace)

        now = datetime.now().isoformat()
        content = str(value.get("content", ""))
        context = str(value.get("context", ""))
        category = str(value.get("category", "general"))

        embedding_text = f"{category}\n{content}\n{context}"
        embedding = self._embed(embedding_text)

        with self._connect() as conn:
            old = conn.execute(
                """
                SELECT created_at FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_str, key),
            ).fetchone()

            created_at = old["created_at"] if old else now

            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    namespace,
                    key,
                    value,
                    content,
                    context,
                    category,
                    embedding,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace_str,
                    key,
                    json.dumps(value, ensure_ascii=False),
                    content,
                    context,
                    category,
                    json.dumps(embedding),
                    created_at,
                    now,
                ),
            )

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        **_: Any,
    ) -> MemoryItem | None:
        namespace_str = self._namespace_to_str(namespace)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT key, value FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_str, key),
            ).fetchone()

        if row is None:
            return None

        return MemoryItem(
            key=row["key"],
            value=json.loads(row["value"]),
            score=None,
        )
    
    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        namespace_str = self._namespace_to_str(namespace)

        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_str, key),
            )

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[MemoryItem]:
        namespace_str = self._namespace_to_str(namespace)
        query = query or ""

        sql = """
        SELECT key, value, embedding
        FROM memories
        WHERE namespace = ?
        """
        params: list[Any] = [namespace_str]

        if filter and filter.get("category"):
            sql += " AND category = ?"
            params.append(filter["category"])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        query_embedding = self._embed(query)

        items: list[MemoryItem] = []

        for row in rows:
            memory_embedding = json.loads(row["embedding"])
            score = self._cosine_similarity(query_embedding, memory_embedding)

            items.append(
                MemoryItem(
                    key=row["key"],
                    value=json.loads(row["value"]),
                    score=score,
                )
            )

        items.sort(key=lambda item: item.score or 0.0, reverse=True)
        return items[:limit]