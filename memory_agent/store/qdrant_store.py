"""Qdrant-backed implementation of the long-term memory store."""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
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

from memory_agent.embedding.base import EmbeddingProvider
from memory_agent.store.base import MemoryItem
from memory_agent.memory_taxonomy import (
    MEMORY_SCHEMA_VERSION,
    normalize_category,
    normalize_memory_type,
    normalize_str_list,
)


logger = logging.getLogger(__name__)


class QdrantMemoryStore:
    """Qdrant-backed semantic memory store.

    This class only handles vector database operations.
    Embedding is delegated to EmbeddingProvider.
    """

    _PAYLOAD_INDEX_FIELDS = (
        "namespace",
        "memory_type",
        "category",
        "subject",
        "memory_key",
    )

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_size: int,
        qdrant_url: str,
        collection_name: str = "agent_memories",
        qdrant_api_key: str | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        """Create a Qdrant client and ensure the target collection exists."""

        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.vector_size = int(vector_size)
        self._closed = False

        if self.vector_size <= 0:
            raise ValueError("vector_size must be a positive integer.")

        if not qdrant_url.strip():
            raise ValueError(
                "QDRANT_URL is required. Start Qdrant with Docker or provide "
                "a remote Qdrant server URL."
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=prefer_grpc,
        )

        self._ensure_collection_with_retry()

    @staticmethod
    def _namespace_to_str(namespace: tuple[str, ...]) -> str:
        """Convert a structured namespace into the stored payload value."""

        return "/".join(namespace)

    @staticmethod
    def _point_id(namespace: tuple[str, ...], key: str) -> str:
        """Derive a stable UUID point id from namespace and memory key."""

        raw = f"{'/'.join(namespace)}::{key}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    @staticmethod
    def _utc_now_iso() -> str:
        """Return the current UTC timestamp in ISO 8601 format."""

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float(value: Any, default: float = 1.0) -> float:
        """Coerce a value to float, returning a default when invalid."""

        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _ensure_open(self) -> None:
        """Raise when the store is used after it has been closed."""

        if self._closed:
            raise RuntimeError("QdrantMemoryStore is closed.")

    def _validate_vector(self, vector: list[float]) -> list[float]:
        """Ensure a vector matches the configured collection dimension."""

        if len(vector) != self.vector_size:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.vector_size}, got {len(vector)}."
            )

        return vector

    def _ensure_collection(self) -> None:
        """Create or validate the collection and its filter indexes."""

        self._ensure_open()

        if self.client.collection_exists(self.collection_name):
            self._validate_existing_collection()
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes()

    def _validate_existing_collection(self) -> None:
        """Validate the existing collection vector configuration."""

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors

        existing_size = getattr(vectors, "size", None)
        if existing_size is None and isinstance(vectors, dict):
            raise ValueError(
                "Existing Qdrant collection uses named vectors, but "
                "QdrantMemoryStore expects a single unnamed vector."
            )

        if existing_size is None:
            raise ValueError(
                "Unable to determine vector size for existing Qdrant "
                f"collection {self.collection_name!r}."
            )

        if int(existing_size) != self.vector_size:
            raise ValueError(
                "Existing Qdrant collection vector size mismatch: "
                f"collection {self.collection_name!r} has size "
                f"{existing_size}, but EMBEDDING_DIMENSION is "
                f"{self.vector_size}. Use a matching embedding model or "
                "create a new QDRANT_COLLECTION."
            )

    def _payload_index_exists(self, field_name: str) -> bool:
        """Return whether a payload field is already indexed."""

        try:
            info = self.client.get_collection(self.collection_name)
        except Exception:
            return False

        payload_schema = getattr(info, "payload_schema", None)
        if isinstance(payload_schema, dict):
            return field_name in payload_schema

        return False

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for fields used in filters.

        These indexes are a performance optimization. Failure to create them
        should not prevent the app from starting, because older Qdrant clients
        or restricted deployments may reject index creation even though normal
        reads/writes still work.
        """

        create_payload_index = getattr(self.client, "create_payload_index", None)
        if create_payload_index is None:
            logger.warning(
                "Qdrant client does not expose create_payload_index; "
                "payload filters will work without explicit indexes."
            )
            return

        for field_name in self._PAYLOAD_INDEX_FIELDS:
            if self._payload_index_exists(field_name):
                continue

            try:
                create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "already indexed" in message:
                    continue

                logger.warning(
                    "Failed to create Qdrant payload index for field %r; "
                    "continuing without it.",
                    field_name,
                    exc_info=True,
                )

    def _ensure_collection_with_retry(self, attempts: int = 10) -> None:
        """Retry collection initialization while Qdrant starts up."""

        for attempt in range(1, attempts + 1):
            try:
                self._ensure_collection()
                return
            except ValueError:
                raise
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(min(0.5 * attempt, 3.0))

    def _build_filter(
        self,
        namespace: tuple[str, ...],
        metadata_filter: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> Filter:
        """Build the Qdrant payload filter for namespace and metadata."""

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

        if metadata_filter:
            for field_name, value in metadata_filter.items():
                if value is None or value == "":
                    continue
                if isinstance(value, (list, tuple, set, dict)):
                    logger.warning(
                        "Skipping unsupported non-scalar Qdrant filter field %r",
                        field_name,
                    )
                    continue
                conditions.append(
                    FieldCondition(
                        key=str(field_name),
                        match=MatchValue(value=value),
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
        """Embed and upsert one memory item into Qdrant."""

        self._ensure_open()

        memory_type = normalize_memory_type(value.get("memory_type"))
        category = normalize_category(value.get("category"), memory_type=memory_type)
        content = str(value.get("content", ""))
        context = str(value.get("context", ""))
        subject = str(value.get("subject", ""))
        entities = ", ".join(normalize_str_list(value.get("entities")))
        topics = ", ".join(normalize_str_list(value.get("topics")))

        embedding_text = "\n".join(
            part
            for part in (
                f"type: {memory_type}",
                f"category: {category}",
                f"subject: {subject}",
                f"entities: {entities}",
                f"topics: {topics}",
                content,
                context,
            )
            if part.strip()
        )
        vector = self._validate_vector(
            await self.embedding_provider.aembed(embedding_text)
        )

        await asyncio.to_thread(
            self._put_sync,
            namespace,
            key,
            value,
            vector,
        )

    def _put_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        vector: list[float],
    ) -> None:
        """Synchronously normalize and upsert one Qdrant point."""

        self._ensure_open()

        namespace_str = self._namespace_to_str(namespace)
        point_id = self._point_id(namespace, key)

        now = self._utc_now_iso()

        old = self._get_sync(namespace, key)
        old_value = old.value if old else {}

        memory_type = normalize_memory_type(value.get("memory_type"))
        category = normalize_category(value.get("category"), memory_type=memory_type)
        content = str(value.get("content", ""))
        context = str(value.get("context", ""))
        subject = str(value.get("subject", ""))
        entities = normalize_str_list(value.get("entities"))
        topics = normalize_str_list(value.get("topics"))
        confidence = self._safe_float(value.get("confidence"), 1.0)
        source = value.get("source") if isinstance(value.get("source"), dict) else {}

        created_at = str(
            old_value.get("created_at")
            or value.get("created_at")
            or now
        )

        normalized_value = {
            **value,
            "schema_version": int(value.get("schema_version") or MEMORY_SCHEMA_VERSION),
            "memory_type": memory_type,
            "content": content,
            "context": context,
            "category": category,
            "subject": subject,
            "entities": entities,
            "topics": topics,
            "confidence": confidence,
            "source": source,
            "created_at": created_at,
            "updated_at": str(value.get("updated_at") or now),
        }

        payload = {
            "namespace": namespace_str,
            "memory_key": key,
            "schema_version": normalized_value["schema_version"],
            "memory_type": memory_type,
            "content": content,
            "context": context,
            "category": category,
            "subject": subject,
            "entities": entities,
            "topics": topics,
            "confidence": confidence,
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
        """Retrieve one memory item by key."""

        self._ensure_open()
        return await asyncio.to_thread(self._get_sync, namespace, key)

    def _get_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        """Synchronously retrieve and normalize one Qdrant point."""

        self._ensure_open()
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
            memory_type = normalize_memory_type(payload.get("memory_type"))
            value = {
                "schema_version": payload.get("schema_version", MEMORY_SCHEMA_VERSION),
                "memory_type": memory_type,
                "content": payload.get("content", ""),
                "context": payload.get("context", ""),
                "category": normalize_category(payload.get("category"), memory_type=memory_type),
                "subject": payload.get("subject", ""),
                "entities": normalize_str_list(payload.get("entities")),
                "topics": normalize_str_list(payload.get("topics")),
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
        """Delete one memory item by key."""

        self._ensure_open()
        await asyncio.to_thread(self._delete_sync, namespace, key)

    def _delete_sync(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Synchronously delete one Qdrant point."""

        self._ensure_open()
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
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemoryItem]:
        """Search memories semantically, or scroll when query is blank."""

        self._ensure_open()

        if metadata_filter is None:
            legacy_filter = kwargs.get("filter")
            if isinstance(legacy_filter, dict):
                metadata_filter = legacy_filter

        limit = max(1, int(limit))

        if not query or not query.strip():
            return await asyncio.to_thread(
                self._scroll_sync,
                namespace,
                limit,
                metadata_filter,
            )

        query_vector = self._validate_vector(
            await self.embedding_provider.aembed(query)
        )

        return await asyncio.to_thread(
            self._search_sync,
            namespace,
            query_vector,
            limit,
            metadata_filter,
        )

    def _scroll_sync(
        self,
        namespace: tuple[str, ...],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Synchronously list memory points matching the payload filter."""

        self._ensure_open()
        qdrant_filter = self._build_filter(namespace, metadata_filter)

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

    def _search_sync(
        self,
        namespace: tuple[str, ...],
        query_vector: list[float],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Synchronously run vector search against Qdrant."""

        self._ensure_open()
        qdrant_filter = self._build_filter(namespace, metadata_filter)

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
            self._point_to_memory_item(
                point,
                score=getattr(point, "score", None),
            )
            for point in points
            if getattr(point, "payload", None)
        ]

    @staticmethod
    def _point_to_memory_item(point: Any, score: float | None) -> MemoryItem:
        """Convert a Qdrant point-like object into a MemoryItem."""

        payload = point.payload or {}

        value = payload.get("value")
        if not isinstance(value, dict):
            memory_type = normalize_memory_type(payload.get("memory_type"))
            value = {
                "schema_version": payload.get("schema_version", MEMORY_SCHEMA_VERSION),
                "memory_type": memory_type,
                "content": payload.get("content", ""),
                "context": payload.get("context", ""),
                "category": normalize_category(payload.get("category"), memory_type=memory_type),
                "subject": payload.get("subject", ""),
                "entities": normalize_str_list(payload.get("entities")),
                "topics": normalize_str_list(payload.get("topics")),
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
        """Close the Qdrant client once."""

        if self._closed:
            return

        self.client.close()
        self._closed = True

    async def aclose(self) -> None:
        """Close both the Qdrant client and embedding provider."""

        try:
            self.close()
        finally:
            await self.embedding_provider.aclose()
