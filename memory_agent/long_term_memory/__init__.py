"""Public interfaces for the long-term memory subsystem."""

from memory_agent.long_term_memory.consolidator import (
    MemoryDecision,
    MemoryExtractionResult,
    consolidate_memories,
)
from memory_agent.long_term_memory.embedding import (
    EmbeddingProvider,
    RemoteEmbeddingProvider,
    create_embedding_provider,
)
from memory_agent.long_term_memory.retrieval import (
    build_memory_query,
    format_memories,
    retrieve_relevant_memories,
    serialize_memory_hit,
)
from memory_agent.long_term_memory.store import (
    MemoryItem,
    MemoryStore,
    QdrantMemoryStore,
    create_memory_store,
)
from memory_agent.long_term_memory.taxonomy import (
    MEMORY_SCHEMA_VERSION,
    MEMORY_TYPE_VALUES,
    MemoryRoute,
    MemoryType,
    iter_memory_routes,
    memory_namespace,
    normalize_memory_type,
)

__all__ = [
    "EmbeddingProvider",
    "MEMORY_SCHEMA_VERSION",
    "MEMORY_TYPE_VALUES",
    "MemoryDecision",
    "MemoryExtractionResult",
    "MemoryItem",
    "MemoryRoute",
    "MemoryStore",
    "MemoryType",
    "QdrantMemoryStore",
    "RemoteEmbeddingProvider",
    "build_memory_query",
    "consolidate_memories",
    "create_embedding_provider",
    "create_memory_store",
    "format_memories",
    "iter_memory_routes",
    "memory_namespace",
    "normalize_memory_type",
    "retrieve_relevant_memories",
    "serialize_memory_hit",
]
