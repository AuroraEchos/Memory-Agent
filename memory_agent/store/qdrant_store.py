import asyncio
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from memory_agent.store.base import MemoryItem


class QdrantMemoryStore:
    """Qdrant-backed semantic memory store.

    Supports:
    - local persistent mode via QDRANT_PATH
    - remote Qdrant server via QDRANT_URL
    """

    def __init__(
        self,
        *,
        collection_name: str = "agent_memories",
        embedding_model: str = "./models/bge-m3",
        embedding_device: str = "auto",
        qdrant_path: str | None = "./qdrant_data",
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_device = self._resolve_embedding_device(embedding_device)
        self.encoder = SentenceTransformer(embedding_model, device=self.embedding_device)
        self._encoder_lock = threading.Lock()

        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=prefer_grpc,
            )
        else:
            path = qdrant_path or "./qdrant_data"
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)

        self.vector_size = self._detect_vector_size()
        self._ensure_collection_with_retry()

    @staticmethod
    def _resolve_embedding_device(device: str) -> str:
        normalized = device.strip().lower()

        if normalized != "auto":
            return normalized

        try:
            import torch
        except ImportError:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _namespace_to_str(namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    @staticmethod
    def _point_id(namespace: tuple[str, ...], key: str) -> str:
        """Qdrant point id must be int or UUID.

        We keep the original memory key in payload["memory_key"],
        and use deterministic UUIDv5 as the Qdrant point id.
        """
        raw = f"{'/'.join(namespace)}::{key}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def _detect_vector_size(self) -> int:
        vector = self._embed("dimension probe")
        return len(vector)

    def _embed(self, text: str) -> list[float]:
        with self._encoder_lock:
            vector = self.encoder.encode(
                text,
                normalize_embeddings=True,
            )

        return vector.astype(float).tolist()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _ensure_collection_with_retry(self, attempts: int = 10) -> None:
        for attempt in range(1, attempts + 1):
            try:
                self._ensure_collection()
                return
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(min(0.5 * attempt, 3.0))

    def _build_filter(
        self,
        namespace: tuple[str, ...],
        filter: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> Filter:
        conditions = [
            FieldCondition(
                key="namespace",
                match=MatchValue(value=self._namespace_to_str(namespace)),
            )
        ]

        if key is not None:
            conditions.append(
                FieldCondition(
                    key="memory_key",
                    match=MatchValue(value=key),
                )
            )

        if filter:
            category = filter.get("category")
            if category:
                conditions.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                )

        return Filter(must=conditions)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        **_: Any,
    ) -> None:
        await asyncio.to_thread(self._put_sync, namespace, key, value)

    def _put_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        namespace_str = self._namespace_to_str(namespace)
        point_id = self._point_id(namespace, key)

        now = datetime.now().isoformat()

        old = self._get_sync(namespace, key)
        old_value = old.value if old else {}

        content = str(value.get("content", ""))
        context = str(value.get("context", ""))
        category = str(value.get("category", "general"))

        created_at = str(old_value.get("created_at") or value.get("created_at") or now)

        normalized_value = {
            **value,
            "content": content,
            "context": context,
            "category": category,
            "created_at": created_at,
            "updated_at": str(value.get("updated_at") or now),
        }

        embedding_text = f"{category}\n{content}\n{context}"
        vector = self._embed(embedding_text)

        payload = {
            "namespace": namespace_str,
            "memory_key": key,
            "content": content,
            "context": context,
            "category": category,
            "confidence": float(normalized_value.get("confidence", 1.0)),
            "created_at": normalized_value["created_at"],
            "updated_at": normalized_value["updated_at"],
            "value": normalized_value,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        **_: Any,
    ) -> MemoryItem | None:
        return await asyncio.to_thread(self._get_sync, namespace, key)

    def _get_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        point_id = self._point_id(namespace, key)

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        payload = points[0].payload or {}

        if payload.get("namespace") != self._namespace_to_str(namespace):
            return None

        if payload.get("memory_key") != key:
            return None

        value = payload.get("value")
        if not isinstance(value, dict):
            value = {
                "content": payload.get("content", ""),
                "context": payload.get("context", ""),
                "category": payload.get("category", "general"),
                "confidence": payload.get("confidence", 1.0),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }

        return MemoryItem(
            key=str(payload.get("memory_key", key)),
            value=value,
            score=None,
        )

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        await asyncio.to_thread(self._delete_sync, namespace, key)

    def _delete_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        point_id = self._point_id(namespace, key)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
        )

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[MemoryItem]:
        return await asyncio.to_thread(
            self._search_sync,
            namespace,
            query,
            limit,
            filter,
        )

    def _search_sync(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        limit = max(1, int(limit))
        qdrant_filter = self._build_filter(namespace, filter)

        if not query or not query.strip():
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                self._point_to_memory_item(point, score=None)
                for point in points
                if point.payload
            ]

        query_vector = self._embed(query)

        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = getattr(result, "points", result)
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        return [
            self._point_to_memory_item(point, score=getattr(point, "score", None))
            for point in points
            if getattr(point, "payload", None)
        ]

    @staticmethod
    def _point_to_memory_item(point: Any, score: float | None) -> MemoryItem:
        payload = point.payload or {}

        value = payload.get("value")
        if not isinstance(value, dict):
            value = {
                "content": payload.get("content", ""),
                "context": payload.get("context", ""),
                "category": payload.get("category", "general"),
                "confidence": payload.get("confidence", 1.0),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }

        return MemoryItem(
            key=str(payload.get("memory_key", "")),
            value=value,
            score=score,
        )

    def close(self) -> None:
        self.client.close()
