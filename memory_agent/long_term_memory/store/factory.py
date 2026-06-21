"""Factory for the configured long-term memory store."""

from memory_agent.config import AppSettings
from memory_agent.long_term_memory.embedding import create_embedding_provider
from memory_agent.long_term_memory.store.base import MemoryStore
from memory_agent.long_term_memory.store.qdrant import QdrantMemoryStore


def create_memory_store(settings: AppSettings) -> MemoryStore:
    """Create the Qdrant-backed memory store and its embedding provider."""

    embedding_provider = create_embedding_provider(settings)

    return QdrantMemoryStore(
        collection_name=settings.qdrant_collection,
        embedding_provider=embedding_provider,
        vector_size=settings.embedding_dimension,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )
